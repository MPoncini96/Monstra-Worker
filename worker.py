import time
import logging
import json
import hashlib
from datetime import datetime, timezone

from db import write_signal, get_latest_signal, get_bot_state, set_bot_state

from bots.Vis import run_vis
from bots.Viator import run_viator
from bots.Vectura import run_vectura
from bots.Medicus import run_medicus
from bots.Imperium import run_imperium
from bots.Cyclus import run_cyclus
from bots.Bellator import run_bellator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

CHECK_INTERVAL_MINUTES = 60  # hourly

STATEFUL_BOTS = {"vectura", "medicus"}

def is_market_open() -> bool:
    now = datetime.now(timezone.utc)

    # US market hours 14:30–21:00 UTC (9:30–16:00 ET), Mon–Fri
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
    Clean payload before hashing so tiny float noise doesn't create fake 'changes'.
    Extend this as you add new bots/fields.
    """
    if not payload:
        return {}

    p = dict(payload)

    # Round portfolio weights if present
    tw = p.get("target_weights")
    if isinstance(tw, dict):
        p["target_weights"] = {k: round(float(v), 6) for k, v in tw.items()}

    # Round other common numeric fields (optional)
    for k in ("drawdown", "dd", "turnover", "fee_frac"):
        if k in p:
            try:
                p[k] = round(float(p[k]), 6)
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
    Returns (should_write, reason).
    Uses bot_id as the primary key to compare vs last saved row for that bot.
    """
    bot_id = new_signal.get("bot_id")
    if not bot_id:
        return True, "missing bot_id (cannot dedupe safely)"

    last = get_latest_signal(bot_id)
    if last is None:
        return True, "no previous signal"

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
    """
    Writes signal if it changed; otherwise skips.
    All bots write to the same DB table; bot_id distinguishes them.
    """
    ok, reason = should_write_signal(s)
    bot_id = s.get("bot_id", "UNKNOWN")

    if not ok:
        logging.info(f"Skip {bot_id} ({reason})")
        return

    write_signal(
        bot_id=bot_id,
        ts=s["ts"],
        signal=s["signal"],
        note=s.get("note"),
        payload=s.get("payload", {}) or {},
    )
    logging.info(f"Wrote {bot_id} signal={s['signal']} to DB ({reason})")
STATEFUL_BOTS = {"vectura", "medicus"}

def run_bot(name: str, fn):
    # Load state only for stateful bots
    state = get_bot_state(name) if name in STATEFUL_BOTS else None

    # Run bot (pass state if supported)
    if name in STATEFUL_BOTS:
        s = fn(state=state)
    else:
        s = fn()

    # Ensure identity fields
    s.setdefault("bot_id", name)
    s.setdefault("ts", datetime.now(timezone.utc))
    s.setdefault("payload", {})

    # Persist updated state if present
    if name in STATEFUL_BOTS and isinstance(s.get("state"), dict):
        set_bot_state(name, s["state"])

    # Don’t store state in the signals payload (optional but recommended)
    s.pop("state", None)

    # Write signal only if changed
    write_signal_safe(s)

def run_all_bots() -> None:
    logging.info("=== Running bots ===")

    runners = [
        ("vis", run_vis),
        ("viator", run_viator),
        ("vectura", run_vectura),
        ("medicus", run_medicus),
        ("imperium", run_imperium),
        ("cyclus", run_cyclus),
        ("Bellator", run_bellator),
    ]

    for name, fn in runners:
        try:
            state = get_bot_state(name) if name in STATEFUL_BOTS else None

            # Run bot (pass state only if stateful)
            s = fn(state=state) if name in STATEFUL_BOTS else fn()

            # Ensure required fields
            s.setdefault("bot_id", name)
            s.setdefault("ts", datetime.now(timezone.utc))
            s.setdefault("payload", {})

            # Persist state if returned
            if name in STATEFUL_BOTS and isinstance(s.get("state"), dict):
                set_bot_state(name, s["state"])

            # Do not store state in signals table
            s.pop("state", None)

            write_signal_safe(s)

        except Exception as e:
            logging.error(f"Bot {name} failed: {e}")

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
