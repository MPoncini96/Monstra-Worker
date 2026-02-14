# bots/Cyclus.py
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone

BENCH = "VOO"

SECTOR_PROXIES = {
    "ai":        "SMH",
    "tech":      "XLK",
    "finance":   "XLF",
    "energy":    "XLE",
    "logistics": "IYT",
    "defense":   "ITA",
}

SECTOR_STOCKS = {
    "ai": [
        "NVDA","AMD","AVGO","ASML","TSM","AMAT","LRCX","KLAC","MU","QCOM","ARM","MRVL"
    ],
    "tech": [
        "AAPL","MSFT","GOOGL","AMZN","META","ORCL","CRM","ADBE","INTU","NOW","CSCO","IBM"
    ],
    "finance": [
        "JPM","BAC","WFC","C","GS","MS","BLK","SCHW","AXP","USB","PGR","AIG"
    ],
    "energy": [
        "XOM","CVX","COP","EOG","SLB","OXY","MPC","VLO","KMI","WMB","OKE","TRGP"
    ],
    "logistics": [
        "UPS","FDX","UNP","CSX","NSC","CP","CNI","JBHT","ODFL","XPO","SAIA","KNX"
    ],
    "defense": [
        "LMT","NOC","RTX","GD","BA","LHX","HII","TDG","HEI","TXT","BWXT","KTOS"
    ],
}

RANK_WEIGHTS = np.array([0.40, 0.30, 0.20, 0.10], dtype=float)


def compute_trailing_returns(prices: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    return prices / prices.shift(lookback_days) - 1.0


def get_rebalance_dates(prices_index: pd.DatetimeIndex, freq: str = "W-FRI") -> pd.DatetimeIndex:
    # your snapping logic
    if freq == "D":
        return prices_index
    desired = pd.date_range(prices_index.min(), prices_index.max(), freq=freq)
    snapped = []
    for d in desired:
        prior = prices_index[prices_index <= d]
        if len(prior) > 0:
            snapped.append(prior[-1])
    return pd.DatetimeIndex(sorted(set(snapped)))


def pick_best_sector(trailing_ret_row: pd.Series, sector_proxies: dict, bench: str) -> str | None:
    bench_ret = trailing_ret_row.get(bench, np.nan)
    if pd.isna(bench_ret):
        return None

    proxy_rets = {}
    for sector, proxy in sector_proxies.items():
        proxy_rets[sector] = trailing_ret_row.get(proxy, np.nan)

    proxy_rets = pd.Series(proxy_rets).dropna()
    eligible = proxy_rets[proxy_rets > bench_ret]
    if eligible.empty:
        return None
    return eligible.sort_values(ascending=False).index[0]


def build_stock_weights_with_fallback(
    trailing_ret_row: pd.Series,
    stock_universe: list[str],
    fallback_ticker: str,
    bench_ret: float,
) -> pd.Series:
    """
    Choose up to 4 stocks with trailing return > bench_ret and allocate 40/30/20/10.
    Remainder to fallback_ticker.
    """
    cols = list(dict.fromkeys(stock_universe + [fallback_ticker]))
    w = pd.Series(0.0, index=cols)

    r = trailing_ret_row.reindex(stock_universe).dropna()
    eligible = r[r > bench_ret].sort_values(ascending=False)
    top = eligible.head(4).index.tolist()

    used = 0.0
    for i, t in enumerate(top):
        w[t] = float(RANK_WEIGHTS[i])
        used += float(RANK_WEIGHTS[i])

    w[fallback_ticker] = 1.0 - used
    return w


def run_cyclus(
    lookback_days: int = 20,
    history_period: str = "12mo",
    rebalance_freq: str = "W-FRI",
    stock_bench_mode: str = "sector",  # "sector" or "voo"
) -> dict:
    """
    Cyclus (Sector Relay) — stateless signal generator.
    """
    ts = datetime.now(timezone.utc)

    # Build required ticker list: bench + all proxies + all stocks
    all_stocks = sorted({t for lst in SECTOR_STOCKS.values() for t in lst})
    all_proxies = sorted(set(SECTOR_PROXIES.values()))
    tickers = sorted(set([BENCH] + all_proxies + all_stocks))

    df = yf.download(
        tickers=tickers,
        period=history_period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=False,
    )

    if isinstance(df.columns, pd.MultiIndex):
        prices = df["Close"].copy()
    else:
        prices = df.copy()

    prices = prices.dropna(how="all").sort_index()
    if prices.empty or len(prices.index) < (lookback_days + 5):
        return {
            "bot_id": "cyclus",
            "ts": ts,
            "signal": "HOLD",
            "note": "Not enough price history; holding VOO",
            "payload": {"target_weights": {BENCH: 1.0}, "lookback_days": lookback_days},
        }

    trailing = compute_trailing_returns(prices, lookback_days).dropna(how="all")
    rebalance_dates = set(get_rebalance_dates(prices.index, rebalance_freq))

    asof = prices.index[-1]
    prev_date = prices.index[-2] if len(prices.index) >= 2 else prices.index[-1]

    # If it's not a rebalance checkpoint, we still emit HOLD with "suggested" weights
    # (your worker de-dupes anyway). Keeps behavior simple.
    tr_row = trailing.loc[prev_date] if (not trailing.empty and prev_date in trailing.index) else None

    if tr_row is None:
        return {
            "bot_id": "cyclus",
            "ts": ts,
            "signal": "HOLD",
            "note": f"As of {asof.date()}: no trailing data for prev_date; holding VOO",
            "payload": {
                "asof": str(asof),
                "prev_date": str(prev_date),
                "bench": BENCH,
                "lookback_days": lookback_days,
                "rebalance_freq": rebalance_freq,
                "target_weights": {BENCH: 1.0},
            },
        }

    best_sector = pick_best_sector(tr_row, SECTOR_PROXIES, BENCH)
    if best_sector is None:
        signal = "RISK_OFF"
        note = f"As of {asof.date()}: no sector proxy beats {BENCH} over {lookback_days}d"
        target_w = {BENCH: 1.0}
        payload = {
            "asof": str(asof),
            "prev_date": str(prev_date),
            "bench": BENCH,
            "lookback_days": lookback_days,
            "rebalance_freq": rebalance_freq,
            "best_sector": None,
            "proxy": None,
            "stock_bench_mode": stock_bench_mode,
            "target_weights": target_w,
        }
        return {"bot_id": "cyclus", "ts": ts, "signal": signal, "note": note, "payload": payload}

    proxy = SECTOR_PROXIES[best_sector]
    stock_universe = SECTOR_STOCKS[best_sector]

    bench_ret = float(tr_row.get(BENCH, np.nan))
    proxy_ret = float(tr_row.get(proxy, np.nan))

    if np.isnan(bench_ret) or np.isnan(proxy_ret):
        signal = "HOLD"
        note = f"As of {asof.date()}: missing bench/proxy trailing; holding {BENCH}"
        target_w = {BENCH: 1.0}
    else:
        stock_bench_ret = proxy_ret if stock_bench_mode == "sector" else bench_ret
        w_local = build_stock_weights_with_fallback(
            trailing_ret_row=tr_row,
            stock_universe=stock_universe,
            fallback_ticker=proxy,
            bench_ret=stock_bench_ret,
        )
        w_local = w_local[w_local > 0].sort_values(ascending=False)
        target_w = {k: float(v) for k, v in w_local.items()}

        # choose signal label
        if prev_date in rebalance_dates:
            signal = "REBALANCE"
            note = f"As of {asof.date()}: sector={best_sector} proxy={proxy} top={', '.join(list(w_local.index)[:4])}"
        else:
            signal = "HOLD"
            note = f"As of {asof.date()}: (not rebalance day) sector={best_sector} proxy={proxy}"

    payload = {
        "asof": str(asof),
        "prev_date": str(prev_date),
        "bench": BENCH,
        "lookback_days": lookback_days,
        "rebalance_freq": rebalance_freq,
        "best_sector": best_sector,
        "proxy": proxy,
        "stock_bench_mode": stock_bench_mode,
        "target_weights": target_w,
    }

    return {"bot_id": "cyclus", "ts": ts, "signal": signal, "note": note, "payload": payload}
