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
    Returns bot state by loading from bot_equity history.
    Computes peak from historical equity and loads last holdings as current_weights.
    If no row exists, returns defaults.
    """
    # First check bot_state table for risk_off flag
    risk_off = False
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT risk_off FROM bot_state WHERE bot_id = %s",
                (bot_id,),
            )
            row = cur.fetchone()
            if row:
                risk_off = bool(row["risk_off"])

    # Load equity history to compute peak and get last state
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get all equity values to compute peak
            cur.execute(
                "SELECT equity FROM bot_equity WHERE bot_id = %s ORDER BY d",
                (bot_id,),
            )
            equity_rows = cur.fetchall()
            
            # Get latest row for current state
            cur.execute(
                """
                SELECT d, equity, holdings
                FROM bot_equity
                WHERE bot_id = %s
                ORDER BY d DESC
                LIMIT 1
                """,
                (bot_id,),
            )
            latest = cur.fetchone()

    if not latest:
        return {"risk_off": False, "equity": 1.0, "peak": 1.0}

    equity = float(latest["equity"])
    peak = max([float(r["equity"]) for r in equity_rows]) if equity_rows else equity
    last_date = str(latest["d"])
    
    holdings = latest.get("holdings") or {}
    # Filter out _meta keys for current_weights
    current_weights = {
        k: float(v) for k, v in holdings.items() 
        if not str(k).startswith("_")
    }

    return {
        "risk_off": risk_off,
        "equity": equity,
        "peak": peak,
        "last_date": last_date,
        "current_weights": current_weights,
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
            # Ensure signals table exists
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS trading.signals (
                    bot_id TEXT NOT NULL,
                    ts TIMESTAMP NOT NULL,
                    signal TEXT NOT NULL,
                    note TEXT,
                    payload JSONB,
                    PRIMARY KEY (bot_id, ts)
                )
                """
            )
            
            cur.execute(
                """
                INSERT INTO trading.signals (bot_id, ts, signal, note, payload)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (bot_id, ts) DO UPDATE SET
                    signal = EXCLUDED.signal,
                    note = EXCLUDED.note,
                    payload = EXCLUDED.payload
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
                FROM trading.signals
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
                FROM trading.bot_equity
                WHERE bot_id = %s
                ORDER BY d DESC
                LIMIT 1
                """,
                (bot_id,),
            )
            row = cur.fetchone()
            if not row:
                return None

            data = dict(row)
            holdings = data.get("holdings")
            if isinstance(holdings, dict):
                data["holdings"] = {
                    k: v for k, v in holdings.items() if not str(k).startswith("_")
                }

            return data

def upsert_bot_equity(bot_id: str, d, equity: float, ret: float | None, holdings: dict | None) -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO trading.bot_equity (bot_id, d, equity, ret, holdings)
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
    """Store holdings, daily ret, and compounded equity in bot_equity.
    
    IMPORTANT: Only calculates returns for NEW trading dates. If market is still open
    (intraday), price data won't include today's close, so returns aren't calculated
    to avoid double-counting yesterday's movement.
    """
    payload = signal_row.get("payload") or {}
    sig = signal_row.get("signal")
    ts = signal_row.get("ts", datetime.now(timezone.utc))
    d = ts.date() if hasattr(ts, "date") else ts

    prev = get_latest_bot_equity(bot_id) or {}
    prev_equity = float(prev.get("equity", 1.0))
    prev_date = prev.get("d")  # Previous stored date
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

    # Get the dates of the last two prices
    last_price_date = last_two.index[-1].date() if hasattr(last_two.index[-1], 'date') else last_two.index[-1]
    second_last_price_date = last_two.index[-2].date() if hasattr(last_two.index[-2], 'date') else last_two.index[-2]
    
    # CRITICAL: Only apply return if the last price date is NEW (not already processed).
    # If market is open intraday, the last_price_date will be from yesterday, which was
    # already applied in a previous update. Skip it to avoid double-counting.
    if prev_date is not None and prev_date >= last_price_date:
        # The price data is for a date we've already processed. Store current equity with 0 return.
        upsert_bot_equity(bot_id=bot_id, d=d, equity=prev_equity, ret=0.0, holdings=holdings)
        return

    daily_ret = (last_two.iloc[-1] / last_two.iloc[-2]) - 1.0

    w = pd.Series(holdings, dtype="float64")
    w[w.abs() < 1e-6] = 0.0  # Clean up floating point dust
    w = w / w.sum()  # Renormalize to sum to 1

    # Align to available returns
    daily_ret = daily_ret[[c for c in daily_ret.index if c in w.index]]
    w = w[daily_ret.index]

    if len(daily_ret.index) == 0:
        upsert_bot_equity(bot_id=bot_id, d=d, equity=prev_equity, ret=0.0, holdings=holdings)
        return

    ret = float((daily_ret * w).sum())
    equity = prev_equity * (1.0 + ret)

    upsert_bot_equity(bot_id=bot_id, d=d, equity=equity, ret=ret, holdings=holdings)


def get_bot_config(bot_id: str) -> dict | None:
    """
    Fetch bot configuration from trading.alpha1 table.
    Returns dict with keys: universe, cash_equivalent, top_n, weights, 
    lookback_days, history_period, interval
    Returns None if bot_id not found or table doesn't exist.
    """
    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT bot_id, name, description, universe, cash_equivalent,
                           top_n, weights, lookback_days, history_period, 
                           interval, is_active
                    FROM trading.alpha1
                    WHERE bot_id = %s AND is_active = TRUE
                    """,
                    (bot_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                
                config = dict(row)
                # Parse JSONB fields
                config['universe'] = list(config.get('universe', []))
                config['weights'] = list(config.get('weights', [0.40, 0.30, 0.20, 0.10]))
                return config
    except Exception as e:
        # Table might not exist yet or other DB error
        print(f"Warning: Could not fetch bot config for {bot_id}: {e}")
        return None


def get_alpha2_config(bot_id: str = None) -> dict | None:
    """
    Fetch bot configuration from trading.alpha2 table.
    Works for cyclus (sector relay) and viator (country momentum).
    Returns None if no active row is found or table does not exist.
    """
    if bot_id is None:
        bot_id = "cyclus"  # Default to cyclus for backward compatibility
    
    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT bot_id, proxies, stocks,
                           rank_weights, lookback_days, history_period,
                           rebalance_freq, stock_bench_mode, is_active
                    FROM trading.alpha2
                    WHERE bot_id = %s AND is_active = TRUE
                    LIMIT 1
                    """,
                    (bot_id,)
                )
                row = cur.fetchone()
                if not row:
                    return None

                config = dict(row)
                config["proxies"] = dict(config.get("proxies") or {})
                config["stocks"] = dict(config.get("stocks") or {})
                config["rank_weights"] = list(config.get("rank_weights") or [0.40, 0.30, 0.20, 0.10])
                return config
    except Exception as e:
        print(f"Warning: Could not fetch alpha2 config for {bot_id}: {e}")
        return None
