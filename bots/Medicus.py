# bots/Medicus.py
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone


HEALTH_UNIVERSE = [
    "LLY","JNJ","UNH","ABBV","MRK","TMO","ABT","AMGN","DHR","PFE",
    "ISRG","VRTX","BMY","SYK","ELV","MDT","REGN","GILD","ZTS","BSX"
]
BENCH = "VOO"

RANK_WEIGHTS = np.array([0.40, 0.30, 0.20, 0.10], dtype=float)


def compute_trailing_returns(prices: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    return prices / prices.shift(lookback_days) - 1.0


def get_rebalance_dates(prices_index: pd.DatetimeIndex, freq: str = "W-FRI") -> pd.DatetimeIndex:
    """
    Matches your backtest snapping logic.
    """
    if freq == "D":
        return prices_index
    desired = pd.date_range(prices_index.min(), prices_index.max(), freq=freq)
    snapped = []
    for d in desired:
        prior = prices_index[prices_index <= d]
        if len(prior) > 0:
            snapped.append(prior[-1])
    return pd.DatetimeIndex(sorted(set(snapped)))


def build_target_weights_vs_bench(
    trailing_ret_row: pd.Series,
    universe: list[str],
    bench_ticker: str,
) -> pd.Series:
    """
    Eligible health stocks: trailing return > bench trailing return.
    Allocate 40/30/20/10 among top eligible names.
    Any leftover weight goes to bench.
    """
    w = pd.Series(0.0, index=universe + [bench_ticker])

    bench_ret = trailing_ret_row.get(bench_ticker, np.nan)
    if pd.isna(bench_ret):
        w[bench_ticker] = 1.0
        return w

    r = trailing_ret_row[universe].dropna()
    eligible = r[r > bench_ret].sort_values(ascending=False)
    top = eligible.head(4).index.tolist()

    used = 0.0
    for i, t in enumerate(top):
        w[t] = float(RANK_WEIGHTS[i])
        used += float(RANK_WEIGHTS[i])

    w[bench_ticker] = 1.0 - used
    return w


def _stable_cols_series(weights: pd.Series, cols: list[str]) -> pd.Series:
    return weights.reindex(cols).fillna(0.0)


def run_medicus(
    lookback_days: int = 10,
    history_period: str = "9mo",
    rebalance_freq: str = "W-FRI",
    max_drawdown: float = 0.12,
    exit_drawdown: float | None = None,
    state: dict | None = None,
) -> dict:
    """
    Stateful signal generator aligned with your killswitch backtest:
    - Weekly rebalance (snapped)
    - Kill-switch with hysteresis based on running equity/peak
    - Holds weights constant between rebalances
    - Returns updated state (caller should persist per-bot)
    """
    ts = datetime.now(timezone.utc)
    if exit_drawdown is None:
        exit_drawdown = max_drawdown / 2.0

    trigger_dd = -abs(max_drawdown)
    exit_dd = -abs(exit_drawdown)

    cols = HEALTH_UNIVERSE + [BENCH]

    # --- init state ---
    if state is None:
        state = {}

    current_w = pd.Series(state.get("current_weights", {BENCH: 1.0}), dtype=float)
    current_w = _stable_cols_series(current_w, cols)

    risk_off = bool(state.get("risk_off", False))
    equity = float(state.get("equity", 1.0))
    peak = float(state.get("peak", equity))
    last_date = state.get("last_date")  # ISO string of last processed trading day
    last_rebalance_date = state.get("last_rebalance_date")  # ISO string

    # --- download prices ---
    prices = yf.download(
        tickers=cols,
        period=history_period,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )["Close"].sort_index()

    prices = prices.dropna(how="all")
    if prices.empty or len(prices.index) < (lookback_days + 5):
        return {
            "bot_id": "medicus",
            "ts": ts,
            "signal": "HOLD",
            "note": "Not enough price history; holding bench",
            "payload": {"target_weights": {BENCH: 1.0}},
            "state": state,
        }

    asof = prices.index[-1]

    # --- update equity from last_date -> asof using current weights ---
    if last_date is not None:
        last_dt = pd.Timestamp(last_date)
        window = prices.loc[prices.index > last_dt]
        if not window.empty:
            rets = window.pct_change().fillna(0.0)
            for d in rets.index:
                port_ret = float((current_w * rets.loc[d].reindex(cols).fillna(0.0)).sum())
                equity *= (1.0 + port_ret)

    # update drawdown stats
    peak = max(peak, equity)
    dd = (equity / peak) - 1.0

    # kill-switch hysteresis update
    if risk_off:
        if dd >= exit_dd:
            risk_off = False
    else:
        if dd <= trigger_dd:
            risk_off = True

    trailing = compute_trailing_returns(prices[cols], lookback_days).dropna(how="all")
    rebalance_dates = get_rebalance_dates(prices.index, rebalance_freq)

    # mimic your backtest: if prev_date is a rebalance date, rebalance now
    prev_date = prices.index[-2] if len(prices.index) >= 2 else prices.index[-1]
    rebalance_due = prev_date in set(rebalance_dates)

    # prevent double rebalance on same prev_date
    if last_rebalance_date is not None and pd.Timestamp(last_rebalance_date) == prev_date:
        rebalance_due = False

    signal = "HOLD"
    note = f"As of {asof.date()}: hold; dd={dd:.2%} risk_off={risk_off}"

    if rebalance_due:
        if risk_off:
            target_w = pd.Series(0.0, index=cols)
            target_w[BENCH] = 1.0
            signal = "RISK_OFF"
            note = f"As of {asof.date()}: kill-switch active (dd={dd:.2%}); 100% {BENCH}"
        else:
            if trailing.empty or prev_date not in trailing.index:
                target_w = pd.Series(0.0, index=cols)
                target_w[BENCH] = 1.0
                signal = "HOLD"
                note = f"As of {asof.date()}: not enough trailing data; hold {BENCH}"
            else:
                tr = trailing.loc[prev_date]
                target_w = build_target_weights_vs_bench(tr, HEALTH_UNIVERSE, BENCH)
                target_w = target_w[target_w > 0].sort_values(ascending=False)

                if len(target_w) == 1 and BENCH in target_w.index and float(target_w.iloc[0]) == 1.0:
                    signal = "RISK_OFF"
                    note = f"As of {asof.date()}: no health names beat {BENCH} over {lookback_days}d"
                else:
                    signal = "REBALANCE"
                    note = f"As of {asof.date()}: top={', '.join(list(target_w.index)[:4])}"

        current_w = _stable_cols_series(target_w, cols)
        last_rebalance_date = str(prev_date)

    new_state = {
        "risk_off": bool(risk_off),
        "equity": float(equity),
        "peak": float(peak),
        "dd": float(dd),
        "last_date": str(asof),
        "last_rebalance_date": last_rebalance_date,
        "current_weights": {k: float(v) for k, v in current_w.items() if float(v) != 0.0},
    }

    return {
        "bot_id": "medicus",
        "ts": ts,
        "signal": signal,
        "note": note,
        "payload": {
            "asof": str(asof),
            "prev_date": str(prev_date),
            "lookback_days": lookback_days,
            "bench": BENCH,
            "target_weights": {k: float(v) for k, v in current_w.items() if float(v) != 0.0},
            "drawdown": float(dd),
            "risk_off": bool(risk_off),
            "rebalance_freq": rebalance_freq,
        },
        "state": new_state,
    }
