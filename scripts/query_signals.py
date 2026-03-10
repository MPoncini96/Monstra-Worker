import os
import psycopg2
from psycopg2.extras import RealDictCursor


def main() -> None:
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("SELECT now() as now_utc")
    print("now_utc:", cur.fetchone()["now_utc"])

    # Query signals from bot_equity (stored in holdings JSONB)
    cur.execute(
        "SELECT bot_id, d, holdings "
        "FROM trading.bot_equity "
        "WHERE d = (now() at time zone 'utc')::date AND holdings ? '_signal' "
        "ORDER BY d DESC LIMIT 50"
    )
    rows = cur.fetchall()
    print("today_rows:", len(rows))
    for row in rows:
        h = row['holdings']
        print(f"{row['bot_id']}: signal={h.get('_signal')}, note={h.get('_note')}")

    # Query last 10 signals
    cur.execute(
        "SELECT bot_id, d, holdings "
        "FROM trading.bot_equity "
        "WHERE holdings ? '_signal' "
        "ORDER BY d DESC LIMIT 10"
    )
    print("last_10:")
    for row in cur.fetchall():
        h = row['holdings']
        print(f"{row['bot_id']} {row['d']}: {h.get('_signal')}")

    conn.close()


if __name__ == "__main__":
    main()
