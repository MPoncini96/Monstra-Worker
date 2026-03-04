import os
import psycopg2
from psycopg2.extras import RealDictCursor


def main() -> None:
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("SELECT now() as now_utc")
    print("now_utc:", cur.fetchone()["now_utc"])

    cur.execute(
        "SELECT bot_id, ts, signal, note "
        "FROM trading.signals "
        "WHERE ts::date = (now() at time zone 'utc')::date "
        "ORDER BY ts DESC LIMIT 50"
    )
    rows = cur.fetchall()
    print("today_rows:", len(rows))
    for row in rows:
        print(row)

    cur.execute("SELECT bot_id, ts, signal FROM trading.signals ORDER BY ts DESC LIMIT 10")
    print("last_10:")
    for row in cur.fetchall():
        print(row)

    conn.close()


if __name__ == "__main__":
    main()
