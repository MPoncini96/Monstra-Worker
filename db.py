import os
import json
import psycopg
from datetime import datetime

DATABASE_URL = os.environ["DATABASE_URL"]

def write_signal(bot_id: str, ts: datetime, signal: str, note: str | None, payload: dict):
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into bot_signals (bot_id, ts, signal, note, payload)
                values (%s, %s, %s, %s, %s::jsonb)
                """,
                (bot_id, ts, signal, note, json.dumps(payload or {})),
            )
        conn.commit()
