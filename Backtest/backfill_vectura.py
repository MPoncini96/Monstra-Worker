import pandas as pd
import numpy as np
import yfinance as yf

from bots.Vectura import (
    LOGISTICS_UNIVERSE,
    BENCH,
    compute_trailing_returns,
    get_rebalance_dates,
    build_target_weights_vs_bench,
)

from db import upsert_bot_equity


START_DATE = "2025-01-01"
END_DATE = "2026-02-24"  # exclusive end (live begins this day)
BOT_ID = "vectura"

LOOKBACK_DAYS = 10
REBALANCE_FREQ = "W-FRI"
MAX_DRAWDOWN = 0.12


def backfill():
    cols = LOGISTICS_UNIVERSE + [BENCH]

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
    peak = 1.0
    risk_off = False

    current_w = pd.Series(0.0, index=cols)
    current_w[BENCH] = 1.0
    pending_w = None

    trigger_dd = -abs(MAX_DRAWDOWN)
    exit_dd = -abs(MAX_DRAWDOWN / 2)

    idx = prices.index
    prev_w = current_w.copy()

    for i in range(1, len(idx)):
        prev_day = idx[i - 1]
        today = idx[i]

        # Compute return using yesterday's closing weights (prev_w) applied to today's return
        day_ret_vec = rets.loc[today].reindex(cols).fillna(0.0)
        port_ret = float((prev_w * day_ret_vec).sum())
        equity *= (1 + port_ret)

        peak = max(peak, equity)
        dd = (equity / peak) - 1.0

        # Kill-switch
        if risk_off:
            if dd >= exit_dd:
                risk_off = False
        else:
            if dd <= trigger_dd:
                risk_off = True

        # Rebalance decision (based on prev_day)
        if prev_day in rebalance_dates:
            if risk_off:
                target_w = pd.Series(0.0, index=cols)
                target_w[BENCH] = 1.0
            else:
                if prev_day in trailing.index:
                    tr = trailing.loc[prev_day]
                    target_w = build_target_weights_vs_bench(
                        tr,
                        LOGISTICS_UNIVERSE,
                        BENCH,
                    )
                else:
                    target_w = pd.Series(0.0, index=cols)
                    target_w[BENCH] = 1.0

            pending_w = target_w.reindex(cols).fillna(0.0)

        # Activate pending weights for end-of-day holdings
        if pending_w is not None:
            current_w = pending_w
            pending_w = None

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

        # Update prev_w for next iteration
        prev_w = current_w.copy()

    print("Backfill complete.")


if __name__ == "__main__":
    backfill()
