import os
import psycopg2

CUTOFF_DATE = "2026-02-20"


def main() -> None:
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM bot_equity WHERE bot_id = 'imperium' AND d >= %s",
        (CUTOFF_DATE,),
    )
    deleted = cur.rowcount
    conn.commit()
    cur.close()
    conn.close()
    print(f"Deleted {deleted} imperium rows from {CUTOFF_DATE} onward.")


if __name__ == "__main__":
    main()
