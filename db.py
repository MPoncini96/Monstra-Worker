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

def get_latest_bot_equity(bot_id: str) -> dict | None:
    """Return the most recent bot_equity row for a bot_id, or None if none exists."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT bot_id, d, equity, ret, holdings
                FROM bot_equity
                WHERE bot_id = %s
                ORDER BY d DESC
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
                        VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (bot_id, d) DO UPDATE SET
              equity = EXCLUDED.equity,
              ret = EXCLUDED.ret,
              holdings = EXCLUDED.holdings,
              updated_at = now()
            """,
                        (bot_id, d, float(equity), None if ret is None else float(ret), Json(holdings or {})),
        )
        conn.commit()

def write_equity(bot_id: str, ts, equity: float) -> None:
    """Write equity snapshot to bot_equity table."""
    d = ts.date() if hasattr(ts, 'date') else ts
    upsert_bot_equity(bot_id=bot_id, d=d, equity=equity, ret=None, holdings=None)

def _today_utc_date():
    return datetime.now(timezone.utc).date()

def update_bot_equity(bot_id: str, signal_row: dict) -> None:
    """Store holdings, daily ret, and compounded equity in bot_equity."""
    payload = signal_row.get("payload") or {}
    sig = signal_row.get("signal")
    ts = signal_row.get("ts", datetime.now(timezone.utc))
    d = ts.date() if hasattr(ts, "date") else ts

    prev = get_latest_bot_equity(bot_id) or {}
    prev_equity = float(prev.get("equity", 1.0))
    holdings = dict(prev.get("holdings") or {})

    # If REBALANCE with non-empty weights, adopt new weights; otherwise keep prior holdings.
    if sig == "REBALANCE" and isinstance(payload.get("target_weights"), dict):
        next_weights = {
            k: float(v)
            for k, v in payload["target_weights"].items()
            if float(v) != 0.0
        }
        if next_weights:
            holdings = next_weights

    # Only wipe weights for explicit no-position or drawdown kill-switch triggers.
    note = (signal_row.get("note") or "").lower()
    if "no position" in note:
        holdings = {}
    elif sig == "RISK_OFF" and ("kill-switch" in note or "kill switch" in note or "killswitch" in note):
        holdings = {}

    if not holdings:
        upsert_bot_equity(bot_id=bot_id, d=d, equity=prev_equity, ret=0.0, holdings={})
        return

    tickers = sorted(holdings.keys())

    # Fetch last two closes for close-to-close return
    px = (
        yf.download(
            tickers=tickers,
            period="7d",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )["Close"]
    )

    if px is None or len(px.index) < 2:
        upsert_bot_equity(bot_id=bot_id, d=d, equity=prev_equity, ret=0.0, holdings=holdings)
        return

    px = px.dropna(how="all")
    if isinstance(px, pd.Series):
        px = px.to_frame()

    last_two = px.tail(2)
    if len(last_two.index) < 2:
        upsert_bot_equity(bot_id=bot_id, d=d, equity=prev_equity, ret=0.0, holdings=holdings)
        return

    daily_ret = (last_two.iloc[-1] / last_two.iloc[-2]) - 1.0

    w = pd.Series(holdings, dtype="float64")
    w = w / w.sum()  # normalize in case of float dust

    # Align to available returns
    daily_ret = daily_ret[[c for c in daily_ret.index if c in w.index]]
    w = w[daily_ret.index]

    if len(daily_ret.index) == 0:
        upsert_bot_equity(bot_id=bot_id, d=d, equity=prev_equity, ret=0.0, holdings=holdings)
        return

    ret = float((daily_ret * w).sum())
    equity = prev_equity * (1.0 + ret)

    upsert_bot_equity(bot_id=bot_id, d=d, equity=equity, ret=ret, holdings=holdings)
