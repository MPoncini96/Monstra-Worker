# db.py
import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor, Json


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
