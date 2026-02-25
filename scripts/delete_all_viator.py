import os
import psycopg2

DATABASE_URL = os.environ.get("DATABASE_URL")
conn = psycopg2.connect(DATABASE_URL, sslmode="require")
cur = conn.cursor()
cur.execute("DELETE FROM bot_equity WHERE bot_id = 'viator'")
deleted = cur.rowcount
conn.commit()
cur.close()
conn.close()
print(f"Deleted {deleted} viator rows.")
