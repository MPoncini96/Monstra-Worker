# db.py
import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from datetime import datetime, timezone
import yfinance as yf
import pandas as pd


DATABASE_URL = os.environ["DATABASE_URL"]

def get_conn():
    return psycopg2.connect(DATABASE_URL)


def get_bot_state(bot_id: str) -> dict:
    """
    Returns bot state in the format our bots expect:
      {"risk_off": bool, "equity": float, "peak": float}
    If no row exists, returns defaults.
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT bot_id, peak_equity, last_equity, risk_off
                FROM bot_state
                WHERE bot_id = %s
                """,
                (bot_id,),
            )
            row = cur.fetchone()

    if not row:
        return {"risk_off": False, "equity": 1.0, "peak": 1.0}

    return {
        "risk_off": bool(row["risk_off"]),
        "equity": float(row["last_equity"]),
        "peak": float(row["peak_equity"]),
    }

def set_bot_state(bot_id: str, state: dict) -> None:
    """
    Upserts bot state from our dict into the bot_state table.
    Expects keys: risk_off, equity, peak
    """
    risk_off = bool(state.get("risk_off", False))
    equity = float(state.get("equity", 1.0))
    peak = float(state.get("peak", equity))

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO bot_state (bot_id, peak_equity, last_equity, risk_off, updated_at)
                VALUES (%s, %s, %s, %s, NOW())
                ON CONFLICT (bot_id)
                DO UPDATE SET
                    peak_equity = EXCLUDED.peak_equity,
                    last_equity = EXCLUDED.last_equity,
                    risk_off = EXCLUDED.risk_off,
                    updated_at = NOW()
                """,
                (bot_id, peak, equity, risk_off),
            )
        conn.commit()
        
def write_signal(bot_id: str, ts, signal: str, note: str | None, payload: dict):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO signals (bot_id, ts, signal, note, payload)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (bot_id, ts, signal, note, Json(payload or {})),
            )
        conn.commit()

def get_latest_signal(bot_id: str) -> dict | None:
    """
    Returns the most recent signal row for a bot_id, or None if none exists.
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT bot_id, ts, signal, note, payload
                FROM signals
                WHERE bot_id = %s
                ORDER BY ts DESC
                LIMIT 1
                """,
                (bot_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

def upsert_bot_equity(bot_id: str, d, equity: float, ret: float | None, holdings: dict | None) -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO bot_equity (bot_id, d, equity, ret, holdings)
            VALUES (%s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (bot_id, d) DO UPDATE SET
              equity = EXCLUDED.equity,
              ret = EXCLUDED.ret,
              holdings = EXCLUDED.holdings,
              updated_at = now()
            """,
            (bot_id, d, float(equity), None if ret is None else float(ret), json.dumps(holdings or {})),
        )
        conn.commit()

def write_equity(bot_id: str, ts, equity: float) -> None:
    """Write equity snapshot to bot_equity table."""
    d = ts.date() if hasattr(ts, 'date') else ts
    upsert_bot_equity(bot_id=bot_id, d=d, equity=equity, ret=None, holdings=None)

def _today_utc_date():
    return datetime.now(timezone.utc).date()

def update_bot_equity(bot_id: str, signal_row: dict) -> None:
    """
    Maintains equity via bot_state:
      state['equity_state'] = {
        'equity': float,
        'weights': {ticker: w, ...},
        'last_date': 'YYYY-MM-DD'
      }
    Writes a snapshot row into bot_equity.
    """
    # load existing state (you already store bot_state)
    state = get_bot_state(bot_id) or {}
    eqs = state.get("equity_state") or {}

    equity = float(eqs.get("equity", 1.0))
    weights = dict(eqs.get("weights") or {})
    last_date_str = eqs.get("last_date")

    payload = signal_row.get("payload") or {}
    sig = signal_row.get("signal")
    ts = signal_row.get("ts", datetime.now(timezone.utc))

    # If REBALANCE, adopt new weights if provided
    if sig == "REBALANCE" and isinstance(payload.get("target_weights"), dict):
        weights = {k: float(v) for k, v in payload["target_weights"].items() if float(v) != 0.0}

    # If bot explicitly says "no position", go to cash (weights empty)
    note = (signal_row.get("note") or "").lower()
    if "no position" in note:
        weights = {}

    # If no weights, equity stays flat; still write snapshot
    if not weights:
        state["equity_state"] = {
            "equity": equity,
            "weights": {},
            "last_date": last_date_str or str(_today_utc_date()),
        }
        set_bot_state(bot_id, state)
        write_equity(bot_id=bot_id, ts=ts, equity=equity)
        return

    tickers = sorted(weights.keys())

    # Decide pricing window
    if last_date_str:
        start = last_date_str
    else:
        # first time pricing: start a few days back so we can compute at least one return
        start = str((_today_utc_date()))
    end = str(_today_utc_date())

    # Fetch prices (daily)
    px = (
        yf.download(
            tickers=tickers,
            start=start,
            end=None,              # yfinance end is exclusive; letting it default is fine
            auto_adjust=True,
            progress=False,
        )["Close"]
    )

    if px is None or len(px.index) < 2:
        # not enough data to compute a return; still store a snapshot
        state["equity_state"] = {
            "equity": equity,
            "weights": weights,
            "last_date": last_date_str or str(_today_utc_date()),
        }
        set_bot_state(bot_id, state)
        write_equity(bot_id=bot_id, ts=ts, equity=equity)
        return

    px = px.dropna(how="all")
    if isinstance(px, pd.Series):
        px = px.to_frame()

    # Compute daily returns and portfolio return
    rets = px.pct_change().dropna(how="all")
    w = pd.Series(weights, dtype="float64")
    w = w / w.sum()  # normalize in case of float dust

    # Align columns
    rets = rets[[c for c in rets.columns if c in w.index]]
    w = w[rets.columns]

    port_ret = rets.mul(w, axis=1).sum(axis=1)
    growth = (1.0 + port_ret).prod()
    equity = equity * float(growth)

    # Update last_date to the last priced date we used
    last_priced_date = str(rets.index[-1].date())

    state["equity_state"] = {
        "equity": equity,
        "weights": weights,
        "last_date": last_priced_date,
    }
    set_bot_state(bot_id, state)

    # Write snapshot
    write_equity(bot_id=bot_id, ts=ts, equity=equity)
