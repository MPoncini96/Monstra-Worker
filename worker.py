import time
import logging
import json
import hashlib
from datetime import datetime, timezone

from db import write_signal, get_latest_signal

from bots.Vis import run_vis
from bots.Viator import run_viator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

CHECK_INTERVAL_MINUTES = 60  # hourly


def is_market_open() -> bool:
    now = datetime.now(timezone.utc)

    # market hours 14:30–21:00 UTC (9:30–16:00 ET)
    if now.weekday() >= 5:
        return False

    hour = now.hour + now.minute / 60.0
    return 14.5 <= hour <= 21.0


def _stable_json(obj) -> str:
    """
    Deterministic JSON string for hashing.
    Ensures payload dict ordering doesn't cause false "changes".
    """
    return json.dumps(obj or {}, sort_keys=True, separators=(",", ":"), default=str)

def normalize_payload(payload: dict) -> dict:
    """
    Clean payload before hashing so tiny float noise
    doesn't create fake 'changes'.
    """
    if not payload:
        return {}

    p = dict(payload)

    # Round portfolio weights if present
    tw = p.get("target_weights")
    if isinstance(tw, dict):
        p["target_weights"] = {
            k: round(float(v), 6) for k, v in tw.items()
        }

    # Round drawdown if Vis includes it
    if "drawdown" in p:
        try:
            p["drawdown"] = round(float(p["drawdown"]), 6)
        except Exception:
            pass

    return p
def fingerprint_signal(s: dict) -> str:
    """
    Hash only the fields that represent meaningful change.
    Ignore timestamp and normalize floats.
    """
    core = {
        "bot_id": s.get("bot_id"),
        "signal": s.get("signal"),
        "note": s.get("note"),
        "payload": normalize_payload(s.get("payload", {})),
    }

    raw = _stable_json(core)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def should_write_signal(new_signal: dict) -> tuple[bool, str]:
    """
    Returns (should_write, reason)
    """
    bot_id = new_signal["bot_id"]
    last = get_latest_signal(bot_id)

    if last is None:
        return True, "no previous signal"

    # Build a dict shaped like our new signal (so fingerprint compares apples-to-apples)
    last_like_new = {
        "bot_id": last.get("bot_id"),
        "signal": last.get("signal"),
        "note": last.get("note"),
        "payload": last.get("payload") or {},
    }

    if fingerprint_signal(new_signal) == fingerprint_signal(last_like_new):
        return False, "no change vs latest"
    return True, "changed"


def write_signal_safe(s: dict) -> None:
    ok, reason = should_write_signal(s)
    if not ok:
        logging.info(f"Skip {s['bot_id']} ({reason})")
        return

    write_signal(
        bot_id=s["bot_id"],
        ts=s["ts"],
        signal=s["signal"],
        note=s.get("note"),
        payload=s.get("payload", {}) or {},
    )
    logging.info(f"Wrote {s['bot_id']} signal={s['signal']} to DB ({reason})")


def run_all_bots() -> None:
    logging.info("=== Running bots ===")

    runners = [
        run_vis,
        run_viator,
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
