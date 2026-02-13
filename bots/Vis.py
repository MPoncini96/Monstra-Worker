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

def run_vis(
    lookback_days: int = 10,
    history_period: str = "6mo",
):
    ts = datetime.now(timezone.utc)

    tickers = ENERGY_TOP + [BENCH]
    prices = yf.download(
        tickers=tickers,
        period=history_period,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )["Close"].sort_index()

    prices = prices.dropna(how="all")
    trailing = compute_trailing_returns(prices, lookback_days).dropna(how="all")

    # If not enough data yet, fall back to bench
    if trailing.empty:
        return {
            "bot_id": "vis",
            "ts": ts,
            "signal": "HOLD",
            "note": "Not enough data for trailing returns; holding VOO",
            "payload": {"target_weights": {BENCH: 1.0}, "lookback_days": lookback_days},
        }

    asof = trailing.index[-1]
    tr = trailing.loc[asof]

    target_w = build_target_weights_vs_bench(tr, ENERGY_TOP, BENCH)
    target_w = target_w[target_w > 0].sort_values(ascending=False)

    # Decide signal label
    # If VOO weight == 1, it effectively means “no energy names beat VOO”
    if len(target_w) == 1 and BENCH in target_w.index and float(target_w.iloc[0]) == 1.0:
        signal = "RISK_OFF"
        note = f"As of {asof.date()}: no energy names beat VOO over {lookback_days}d"
    else:
        signal = "REBALANCE"
        note = f"As of {asof.date()}: top={', '.join(list(target_w.index)[:4])}"

    return {
        "bot_id": "vis",
        "ts": ts,
        "signal": signal,
        "note": note,
        "payload": {
            "asof": str(asof),
            "lookback_days": lookback_days,
            "bench": BENCH,
            "target_weights": {k: float(v) for k, v in target_w.items()},
        },
    }
