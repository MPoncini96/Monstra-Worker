from datetime import datetime, timezone

def run_vis():
    # TODO: plug in your real logic
    return {
        "bot_id": "vis",
        "ts": datetime.now(timezone.utc),
        "signal": "HOLD",
        "note": "MVP: placeholder signal",
        "payload": {"universe": "energy_top", "bench": "VOO"}
    }
