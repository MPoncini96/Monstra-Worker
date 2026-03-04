# bots/alpha1.py
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone

# Default fallback configuration (used if database is unavailable)
DEFAULT_UNIVERSE = [
    "VOO", "QQQ",
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL",
    "COST", "JPM", "UNH", "HD",
    "AMD", "AVGO",
]

DEFAULT_CASH_EQUIVALENT = "VOO"
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


def run_alpha1(
    lookback_days: int | None = None,
    top_n: int | None = None,
    history_period: str | None = None,
    interval: str | None = None,
    use_db_config: bool = True,
) -> dict:
    """
    Alpha1 (Rank Rotation) — Signal generator:
    - Fetches configuration from database (trading.bots) if use_db_config=True
    - Falls back to default parameters if database unavailable or use_db_config=False
    - Download 1h prices for UNIVERSE
    - Compute growth over lookback_days (assuming ~24 bars/day)
    - Select top_n tickers by growth
    - Output target_weights (40/30/20/10 default)
    """
    ts = datetime.now(timezone.utc)
    
    # Fetch configuration from database
    universe = DEFAULT_UNIVERSE
    cash_equivalent = DEFAULT_CASH_EQUIVALENT
    top_n_config = DEFAULT_TOP_N
    weights_config = DEFAULT_WEIGHTS
    lookback_days_config = 14
    history_period_config = "60d"
    interval_config = "1h"
    
    if use_db_config:
        try:
            from db import get_bot_config
            config = get_bot_config("alpha1")
            if config:
                universe = config.get("universe", DEFAULT_UNIVERSE)
                cash_equivalent = config.get("cash_equivalent", DEFAULT_CASH_EQUIVALENT)
                top_n_config = config.get("top_n", DEFAULT_TOP_N)
                weights_config = np.array(config.get("weights", [0.40, 0.30, 0.20, 0.10]), dtype=float)
                lookback_days_config = config.get("lookback_days", 14)
                history_period_config = config.get("history_period", "60d")
                interval_config = config.get("interval", "1h")
        except Exception as e:
            print(f"Warning: Could not load config from database, using defaults: {e}")
    
    # Allow function parameters to override config
    lookback_days = lookback_days if lookback_days is not None else lookback_days_config
    top_n = top_n if top_n is not None else top_n_config
    history_period = history_period if history_period is not None else history_period_config
    interval = interval if interval is not None else interval_config

    weights = _default_top_weights(top_n)
    weights = weights / weights.sum()

    # Download
    df = yf.download(
        tickers=universe,
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
        prices = df[["Close"]].rename(columns={"Close": universe[0]})

    prices = prices.sort_index().dropna(how="all")

    # Not enough history -> HOLD cash-like (VOO as default)
    min_rows = int(lookback_days * 24) + 5
    if prices.empty or len(prices.index) < min_rows:
        return {
            "bot_id": "alpha1",
            "ts": ts,
            "signal": "HOLD",
            "note": f"Not enough {interval} history for lookback={lookback_days}d; holding {cash_equivalent}",
            "payload": {
                "asof": str(prices.index[-1]) if not prices.empty else None,
                "interval": interval,
                "lookback_days": lookback_days,
                "top_n": top_n,
                "target_weights": {cash_equivalent: 1.0},
            },
        }

    growth = _compute_growth(prices, lookback_days=lookback_days, bars_per_day=24)
    asof = growth.index[-1]
    g_row = growth.loc[asof]

    top = _top_n_by_growth(g_row, n=top_n)

    # Build target weights
    target_w = {sym: float(w) for sym, w in zip(top, weights)}
    # Any missing / odd cases: if top is empty, default to cash equivalent
    if not target_w:
        target_w = {cash_equivalent: 1.0}
        signal = "HOLD"
        note = f"As of {asof}: no valid growth ranks; holding {cash_equivalent}"
    else:
        signal = "REBALANCE"
        note = f"As of {asof}: top={', '.join(top)}"

    return {
        "bot_id": "alpha1",
        "ts": ts,
        "signal": signal,
        "note": note,
        "payload": {
            "asof": str(asof),
            "interval": interval,
            "lookback_days": lookback_days,
            "top_n": top_n,
            "universe_size": len(universe),
            "target_weights": target_w,
        },
    }
