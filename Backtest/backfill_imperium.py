import pandas as pd
import numpy as np
import yfinance as yf

from bots.Imperium import (
    UNIVERSE,
    DEFAULT_TOP_N,
    _default_top_weights,
    _compute_growth,
    _top_n_by_growth,
)

from db import upsert_bot_equity


START_DATE = "2025-01-01"
END_DATE = "2026-02-24"  # exclusive end (live begins this day)
BOT_ID = "imperium"

LOOKBACK_DAYS = 14
TOP_N = DEFAULT_TOP_N
INTERVAL = "1h"


def _end_of_day(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)


def backfill():
    cols = list(UNIVERSE)

    prices = yf.download(
        tickers=cols,
        start=START_DATE,
        end=END_DATE,
        interval=INTERVAL,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=False,
    )

    if isinstance(prices.columns, pd.MultiIndex):
        prices = prices["Close"].copy()
    else:
        prices = prices[["Close"]].rename(columns={"Close": cols[0]})

    prices = prices.dropna(how="all").sort_index()
    if prices.empty:
        print("No price data available.")
        return

    daily_close = prices.resample("1D").last().dropna(how="all")
    daily_rets = daily_close.pct_change().fillna(0.0)

    growth = _compute_growth(prices, lookback_days=LOOKBACK_DAYS, bars_per_day=24)
    weights = _default_top_weights(TOP_N)
    weights = weights / weights.sum()

    equity = 1.0
    current_w = pd.Series(0.0, index=cols)
    current_w["VOO"] = 1.0

    idx = daily_close.index

    for i in range(1, len(idx)):
        prev_day = idx[i - 1]
        today = idx[i]

        end_ts = _end_of_day(today)
        if not growth.empty and growth.index.min() <= end_ts:
            last_ts = growth.loc[:end_ts].index[-1]
            g_row = growth.loc[last_ts]

            top = _top_n_by_growth(g_row, n=TOP_N)
            if top:
                target_w = pd.Series(0.0, index=cols)
                for sym, w in zip(top, weights):
                    target_w[sym] = float(w)
            else:
                target_w = pd.Series(0.0, index=cols)
                target_w["VOO"] = 1.0
        else:
            target_w = pd.Series(0.0, index=cols)
            target_w["VOO"] = 1.0

        current_w = target_w.reindex(cols).fillna(0.0)

        day_ret_vec = daily_rets.loc[today].reindex(cols).fillna(0.0)
        port_ret = float((current_w * day_ret_vec).sum())
        equity *= (1 + port_ret)

        holdings = {
            k: float(v)
            for k, v in current_w.items()
            if abs(float(v)) > 1e-10
        }

        holdings["_meta"] = {"is_backtest": True}

        upsert_bot_equity(
            bot_id=BOT_ID,
            d=today.date(),
            equity=float(equity),
            ret=float(port_ret),
            holdings=holdings,
        )

    print("Backfill complete.")


if __name__ == "__main__":
    backfill()
