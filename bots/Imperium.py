# bots/Imperium.py
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone

UNIVERSE = [
    "VOO", "QQQ",
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL",
    "COST", "JPM", "UNH", "HD",
    "AMD", "AVGO",
]

DEFAULT_TOP_N = 4
DEFAULT_WEIGHTS = np.array([0.40, 0.30, 0.20, 0.10], dtype=float)


def _default_top_weights(n: int = 4, base=(0.40, 0.30, 0.20, 0.10)) -> np.ndarray:
    if n == 4:
        return np.array(base, dtype=float)
    w = np.linspace(n, 1, n, dtype=float)
    w /= w.sum()
    return w


def _compute_growth(prices: pd.DataFrame, lookback_days: int = 14, bars_per_day: int = 24) -> pd.DataFrame:
    lb = int(lookback_days * bars_per_day)
    return prices / prices.shift(lb) - 1.0


def _top_n_by_growth(growth_row: pd.Series, n: int = 4) -> list[str]:
    g = growth_row.dropna().sort_values(ascending=False)
    return list(g.index[:n])


def run_imperium(
    lookback_days: int = 14,
    top_n: int = DEFAULT_TOP_N,
    history_period: str = "60d",      # 1h data is typically limited; 60d is safe-ish
    interval: str = "1h",
) -> dict:
    """
    Imperium (Rank Rotation) — Signal generator:
    - Download 1h prices for UNIVERSE
    - Compute growth over lookback_days (assuming ~24 bars/day)
    - Select top_n tickers by growth
    - Output target_weights (40/30/20/10 default)
    """
    ts = datetime.now(timezone.utc)

    weights = _default_top_weights(top_n)
    weights = weights / weights.sum()

    # Download
    df = yf.download(
        tickers=UNIVERSE,
        period=history_period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=False,
    )

    # yfinance shape handling
    if isinstance(df.columns, pd.MultiIndex):
        prices = df["Close"].copy()
    else:
        # single ticker edge-case
        prices = df[["Close"]].rename(columns={"Close": UNIVERSE[0]})

    prices = prices.sort_index().dropna(how="all")

    # Not enough history -> HOLD cash-like (VOO as default)
    min_rows = int(lookback_days * 24) + 5
    if prices.empty or len(prices.index) < min_rows:
        return {
            "bot_id": "imperium",
            "ts": ts,
            "signal": "HOLD",
            "note": f"Not enough {interval} history for lookback={lookback_days}d; holding VOO",
            "payload": {
                "asof": str(prices.index[-1]) if not prices.empty else None,
                "interval": interval,
                "lookback_days": lookback_days,
                "top_n": top_n,
                "target_weights": {"VOO": 1.0},
            },
        }

    growth = _compute_growth(prices, lookback_days=lookback_days, bars_per_day=24)
    asof = growth.index[-1]
    g_row = growth.loc[asof]

    top = _top_n_by_growth(g_row, n=top_n)

    # Build target weights
    target_w = {sym: float(w) for sym, w in zip(top, weights)}
    # Any missing / odd cases: if top is empty, default to VOO
    if not target_w:
        target_w = {"VOO": 1.0}
        signal = "HOLD"
        note = f"As of {asof}: no valid growth ranks; holding VOO"
    else:
        signal = "REBALANCE"
        note = f"As of {asof}: top={', '.join(top)}"

    return {
        "bot_id": "imperium",
        "ts": ts,
        "signal": signal,
        "note": note,
        "payload": {
            "asof": str(asof),
            "interval": interval,
            "lookback_days": lookback_days,
            "top_n": top_n,
            "universe_size": len(UNIVERSE),
            "target_weights": target_w,
        },
    }
