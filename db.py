# db.py
import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor, Json

DATABASE_URL = os.environ["DATABASE_URL"]

def get_conn():
    return psycopg2.connect(DATABASE_URL)

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
