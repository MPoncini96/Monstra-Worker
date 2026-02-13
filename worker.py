import time
import logging
from datetime import datetime, timezone

from db import write_signal

from bots.Vis import run_vis
from bots.Viator import run_viator

# (leave these imports commented until you’re ready to enable them)
# from bots.Imperium import run_imperium
# from bots.Vectura import run_vectura
# from bots.Medicus import run_medicus
# from bots.Cyclus import run_cyclus
# from bots.Bellator import run_bellator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

CHECK_INTERVAL_MINUTES = 5  # runs every 5 minutes


def is_market_open() -> bool:
    now = datetime.now(timezone.utc)

    # market hours 14:30–21:00 UTC (9:30–16:00 ET)
    if now.weekday() >= 5:
        return False

    hour = now.hour + now.minute / 60.0
    return 14.5 <= hour <= 21.0


def write_signal_safe(s: dict) -> None:
    """
    Small helper so one bad payload doesn't crash the worker.
    """
    write_signal(
        bot_id=s["bot_id"],
        ts=s["ts"],
        signal=s["signal"],
        note=s.get("note"),
        payload=s.get("payload", {}) or {},
    )
    logging.info(f"Wrote {s['bot_id']} signal={s['signal']} to DB")


def run_all_bots() -> None:
    logging.info("=== Running bots ===")

    runners = [
        run_vis,
        run_viator,
        # run_imperium,
        # run_vectura,
        # run_medicus,
        # run_cyclus,
        # run_bellator,
    ]

    for fn in runners:
        try:
            s = fn()
            write_signal_safe(s)
        except Exception as e:
            logging.error(f"Bot {getattr(fn, '__name__', str(fn))} failed: {e}")

    logging.info("=== Finished cycle ===")


def main_loop() -> None:
    logging.info("Monstra Worker started...")

    while True:
        try:
            if is_market_open():
                run_all_bots()
            else:
                logging.info("Market closed — sleeping")
        except Exception as e:
            logging.error(f"ERROR: {e}")

        time.sleep(CHECK_INTERVAL_MINUTES * 60)


if __name__ == "__main__":
    main_loop()
