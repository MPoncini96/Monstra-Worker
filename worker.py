import time
import logging
from datetime import datetime, timezone
from db import write_signal
from bots.Vis import run_vis
from bots.Imperium import run_imperium
from bots.Viator import run_viator
from bots.Vectura import run_vectura
from bots.Medicus import run_medicus
from bots.Cyclus import run_cyclus
from bots.Bellator import run_bellator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

CHECK_INTERVAL_MINUTES = 5  # runs every 5 minutes

def is_market_open():
    now = datetime.now(timezone.utc)

    # market hours 14:30–21:00 UTC (9:30–16:00 ET)
    if now.weekday() >= 5:
        return False

    hour = now.hour + now.minute/60
    return 14.5 <= hour <= 21

def run_all_bots():
    logging.info("=== Running bots ===")

    # TEMP TEST SIGNAL (we remove this later)
    write_signal(
        bot_id="vis",
        ts=datetime.now(timezone.utc),
        signal="TEST",
        note="First DB write from Render worker",
        payload={"status": "hello database"}
    )
    logging.info("Wrote test signal to database")


    logging.info("=== Finished cycle ===")

def main_loop():
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
