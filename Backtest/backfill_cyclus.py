from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf

from bots.Cyclus import (
    BENCH,
    SECTOR_PROXIES,
    SECTOR_STOCKS,
    compute_trailing_returns,
    get_rebalance_dates,
    pick_best_sector,
    build_stock_weights_with_fallback,
)

from db import upsert_bot_equity


START_DATE = "2025-01-01"
END_DATE = "2026-02-26"  # exclusive end (live begins this day)
BOT_ID = "cyclus"

LOOKBACK_DAYS = 20
REBALANCE_FREQ = "W-FRI"
STOCK_BENCH_MODE = "sector"  # "sector" or "voo"


def backfill():
    all_stocks = sorted({t for lst in SECTOR_STOCKS.values() for t in lst})
    all_proxies = sorted(set(SECTOR_PROXIES.values()))
    cols = sorted(set([BENCH] + all_proxies + all_stocks))

    prices = (
        yf.download(
            tickers=cols,
            start=START_DATE,
            end=END_DATE,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )["Close"]
    )

    prices = prices.dropna(how="all").sort_index()
    rets = prices.pct_change().fillna(0.0)
    trailing = compute_trailing_returns(prices, LOOKBACK_DAYS)

    rebalance_dates = set(get_rebalance_dates(prices.index, REBALANCE_FREQ))

    equity = 1.0
    current_w = pd.Series(0.0, index=cols)
    current_w[BENCH] = 1.0
    pending_w = None

    idx = prices.index

    for i in range(1, len(idx)):
        prev_day = idx[i - 1]
        today = idx[i]

        # Activate pending weights
        if pending_w is not None:
            current_w = pending_w
            pending_w = None

        # Apply return
        day_ret_vec = rets.loc[today].reindex(cols).fillna(0.0)
        port_ret = float((current_w * day_ret_vec).sum())
        equity *= (1 + port_ret)

        # Rebalance decision (based on prev_day)
        if prev_day in rebalance_dates:
            if prev_day in trailing.index:
                tr_row = trailing.loc[prev_day]
                best_sector = pick_best_sector(tr_row, SECTOR_PROXIES, BENCH)
                if best_sector is None:
                    target_w = pd.Series(0.0, index=cols)
                    target_w[BENCH] = 1.0
                else:
                    proxy = SECTOR_PROXIES[best_sector]
                    stock_universe = SECTOR_STOCKS[best_sector]
                    bench_ret = float(tr_row.get(BENCH, np.nan))
                    proxy_ret = float(tr_row.get(proxy, np.nan))

                    if np.isnan(bench_ret) or np.isnan(proxy_ret):
                        target_w = pd.Series(0.0, index=cols)
                        target_w[BENCH] = 1.0
                    else:
                        stock_bench_ret = proxy_ret if STOCK_BENCH_MODE == "sector" else bench_ret
                        w_local = build_stock_weights_with_fallback(
                            trailing_ret_row=tr_row,
                            stock_universe=stock_universe,
                            fallback_ticker=proxy,
                            bench_ret=stock_bench_ret,
                        )
                        target_w = w_local.reindex(cols).fillna(0.0)
            else:
                target_w = pd.Series(0.0, index=cols)
                target_w[BENCH] = 1.0

            target_w = target_w.reindex(cols).fillna(0.0)
            pending_w = target_w

        holdings = {
            k: float(v)
            for k, v in current_w.items()
            if abs(float(v)) > 1e-10
        }

        # Mark as backtest internally
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
