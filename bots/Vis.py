import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timezone

ENERGY_TOP = [
    "XOM","CVX","COP","EOG","SLB","OXY","MPC","VLO",
    "KMI","WMB","OKE","TRGP",
    "CCJ","UEC","LEU","BWXT",
    "NEE","DUK","SO","EXC"
]
BENCH = "VOO"
RANK_WEIGHTS = np.array([0.40, 0.30, 0.20, 0.10])

def compute_trailing_returns(prices: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    return prices / prices.shift(lookback_days) - 1.0

def build_target_weights_vs_bench(
    trailing_ret_row: pd.Series,
    energy_top: list[str],
    bench_ticker: str,
) -> pd.Series:
    w = pd.Series(0.0, index=energy_top + [bench_ticker])

    bench_ret = trailing_ret_row.get(bench_ticker, np.nan)
    if pd.isna(bench_ret):
        w[bench_ticker] = 1.0
        return w

    r = trailing_ret_row[energy_top].dropna()
    eligible = r[r > bench_ret].sort_values(ascending=False)
    top = eligible.head(4).index.tolist()

    used = 0.0
    for i, t in enumerate(top):
        w[t] = float(RANK_WEIGHTS[i])
        used += float(RANK_WEIGHTS[i])

    w[bench_ticker] = 1.0 - used
    return w

def is_rebalance_day(asof: pd.Timestamp, freq: str = "W-FRI") -> bool:
    # simplest: only rebalance on Fridays (market day)
    # pandas Timestamp: Monday=0 ... Friday=4
    if freq == "W-FRI":
        return asof.weekday() == 4
    if freq == "D":
        return True
    # fallback: use pandas date_range to determine membership (more expensive)
    return asof in pd.date_range(asof.normalize(), asof.normalize(), freq=freq)

def run_vis_stateful(
    lookback_days: int = 10,
    history_period: str = "6mo",
    rebalance_freq: str = "W-FRI",
    max_drawdown: float = 0.12,
    exit_drawdown: float | None = None,
    state: dict | None = None,
):
    """
    Stateful version that mimics backtest behavior:
    - Weekly rebalance on W-FRI
    - Kill-switch with hysteresis
    - Returns updated `state` so you can persist it between runs
    """
    ts = datetime.now(timezone.utc)
    if exit_drawdown is None:
        exit_drawdown = max_drawdown / 2.0

    trigger_dd = -abs(max_drawdown)
    exit_dd = -abs(exit_drawdown)

    # --- load prices ---
    tickers = ENERGY_TOP + [BENCH]
    prices = yf.download(
        tickers=tickers,
        period=history_period,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )["Close"].sort_index()

    prices = prices.dropna(how="all")
    if prices.empty:
        return {
            "bot_id": "vis",
            "ts": ts,
            "signal": "HOLD",
            "note": "No price data returned",
            "payload": {"target_weights": {BENCH: 1.0}},
            "state": state or {},
        }

    trailing = compute_trailing_returns(prices, lookback_days).dropna(how="all")
    asof = prices.index[-1]

    # --- init state ---
    if state is None:
        state = {}
    current_w = pd.Series(state.get("current_weights", {BENCH: 1.0}), dtype=float)
    risk_off = bool(state.get("risk_off", False))
    equity = float(state.get("equity", 1.0))
    peak = float(state.get("peak", equity))
    last_date = state.get("last_date")  # ISO string
    last_rebalance_date = state.get("last_rebalance_date")  # ISO string

    # --- update equity based on daily return since last_date (approx incremental) ---
    if last_date is not None:
        last_date = pd.Timestamp(last_date)
        # use next day after last_date up to asof
        window = prices.loc[prices.index > last_date]
        if not window.empty:
            rets = window.pct_change().fillna(0.0)
            # align current_w to columns
            cols = ENERGY_TOP + [BENCH]
            current_w = current_w.reindex(cols).fillna(0.0)
            for d in rets.index:
                port_ret = float((current_w * rets.loc[d].reindex(cols).fillna(0.0)).sum())
                equity *= (1.0 + port_ret)

    peak = max(peak, equity)
    dd = (equity / peak) - 1.0

    # --- kill-switch state machine ---
    if risk_off:
        if dd >= exit_dd:
            risk_off = False
    else:
        if dd <= trigger_dd:
            risk_off = True

    # --- decide whether to rebalance today ---
    rebalance_due = is_rebalance_day(asof, rebalance_freq)

    # Don’t rebalance twice on same asof date
    if last_rebalance_date is not None and pd.Timestamp(last_rebalance_date) == asof:
        rebalance_due = False

    signal = "HOLD"
    note = f"As of {asof.date()}: hold; dd={dd:.2%} risk_off={risk_off}"

    target_w = current_w.copy()

    if rebalance_due:
        if trailing.empty or asof not in trailing.index:
            target_w = pd.Series({BENCH: 1.0}, dtype=float)
            signal = "HOLD"
            note = f"As of {asof.date()}: not enough trailing data; hold VOO"
        else:
            if risk_off:
                target_w = pd.Series(0.0, index=ENERGY_TOP + [BENCH])
                target_w[BENCH] = 1.0
                signal = "RISK_OFF"
                note = f"As of {asof.date()}: kill-switch active (dd={dd:.2%}); 100% VOO"
            else:
                tr = trailing.loc[asof]
                target_w = build_target_weights_vs_bench(tr, ENERGY_TOP, BENCH)
                target_w = target_w[target_w > 0].sort_values(ascending=False)

                if len(target_w) == 1 and BENCH in target_w.index and float(target_w.iloc[0]) == 1.0:
                    signal = "RISK_OFF"
                    note = f"As of {asof.date()}: no energy names beat VOO over {lookback_days}d"
                else:
                    signal = "REBALANCE"
                    note = f"As of {asof.date()}: rebalance to {', '.join(list(target_w.index)[:4])}"

        current_w = target_w
        last_rebalance_date = str(asof)

    # --- persist updated state ---
    cols = ENERGY_TOP + [BENCH]
    current_w = current_w.reindex(cols).fillna(0.0)

    new_state = {
        "risk_off": risk_off,
        "equity": equity,
        "peak": peak,
        "dd": dd,
        "last_date": str(asof),
        "last_rebalance_date": last_rebalance_date,
        "current_weights": {k: float(v) for k, v in current_w.items() if float(v) != 0.0},
    }

    return {
        "bot_id": "vis",
        "ts": ts,
        "signal": signal,
        "note": note,
        "payload": {
            "asof": str(asof),
            "lookback_days": lookback_days,
            "bench": BENCH,
            "target_weights": {k: float(v) for k, v in current_w.items() if float(v) != 0.0},
            "drawdown": float(dd),
            "risk_off": bool(risk_off),
        },
        "state": new_state,
    }
