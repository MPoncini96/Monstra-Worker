import time
from datetime import datetime
from bots.Vis import run_vis
from bots.Imperium import run_imperium
from bots.Viator import run_viator
from bots.Vectura import run_vectura
from bots.Medicus import run_medicus
from bots.Cyclus import run_cyclus
from bots.Bellator import run_bellator

CHECK_INTERVAL_MINUTES = 5  # runs every 5 minutes

def is_market_open():
    now = datetime.utcnow()

    # market hours 14:30–21:00 UTC (9:30–16:00 ET)
    if now.weekday() >= 5:
        return False

    hour = now.hour + now.minute/60
    return 14.5 <= hour <= 21

def run_all_bots():
    print(f"\n=== Running bots {datetime.utcnow()} ===")

    run_vis()
    run_imperium()

    print("=== Finished cycle ===")

def main_loop():
    print("Monstra Worker started...")

    while True:
        try:
            if is_market_open():
                run_all_bots()
            else:
                print("Market closed — sleeping")

        except Exception as e:
            print("ERROR:", e)

        time.sleep(CHECK_INTERVAL_MINUTES * 60)

if __name__ == "__main__":
    main_loop()
