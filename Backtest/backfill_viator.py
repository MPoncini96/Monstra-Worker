import pandas as pd
import numpy as np
import yfinance as yf

from bots.Viator import (
    COUNTRY_TICKERS,
    ViatorConfig,
    apply_ticker_fixups,
    compute_momentum,
    get_rebalance_dates,
    select_country_and_holdings,
    clean_prices,
    _extract_adj_close,
    _quiet_yfinance,
)

from db import upsert_bot_equity


START_DATE = "2025-01-01"
END_DATE = "2026-02-24"  # exclusive end (live begins this day)
BOT_ID = "viator"


def prune_country_universe_from_px(country_map, px, cfg: ViatorConfig):
    pruned = {}
    for country, tickers in country_map.items():
        tickers = list(dict.fromkeys([t.strip() for t in tickers if t and isinstance(t, str)]))
        if not tickers:
            continue

        available = [t for t in tickers if t in px.columns]
        if not available:
            continue

        sub = px[available]
        non_na_counts = sub.notna().sum(axis=0)
        valid = non_na_counts[non_na_counts >= cfg.min_points].index.tolist()
        if len(valid) < cfg.keep_per_country:
            continue

        last_valid_date = sub[valid].apply(lambda s: s.dropna().index.max())
        rank_df = pd.DataFrame({"count": non_na_counts[valid], "last": last_valid_date})

        if cfg.prefer == "recent":
            rank_df = rank_df.sort_values(["last", "count"], ascending=[False, False])
        else:
            rank_df = rank_df.sort_values(["count", "last"], ascending=[False, False])

        kept = rank_df.head(cfg.keep_per_country).index.tolist()
        pruned[country] = kept

    return pruned


def backfill():
    cfg = ViatorConfig()

    fixed = apply_ticker_fixups(COUNTRY_TICKERS)
    all_tickers = sorted({t for lst in fixed.values() for t in lst})

    with _quiet_yfinance():
        data = yf.download(
            tickers=all_tickers,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=False,
            progress=False,
            threads=cfg.threads,
            group_by="column",
        )

    px_raw = _extract_adj_close(data, all_tickers).sort_index()
    px_raw = px_raw.dropna(how="all")

    if px_raw.empty:
        print("No price data available.")
        return

    pruned_map = prune_country_universe_from_px(fixed, px_raw, cfg)
    universe = sorted({t for lst in pruned_map.values() for t in lst if t in px_raw.columns})
    if not universe:
        print("No valid tickers after pruning.")
        return

    px = clean_prices(px_raw[universe], cfg.drop_na_threshold)
    if px.empty or len(px.index) < (cfg.momentum_lookback_days + 5):
        print("Not enough price history.")
        return

    rets = px.pct_change().fillna(0.0)
    mom = compute_momentum(px, cfg.momentum_lookback_days).dropna(how="all")
    rebalance_dates = set(get_rebalance_dates(px.index, cfg.rebalance_rule))

    equity = 1.0
    # Start with equal allocation across first available countries
    first_countries = list(pruned_map.keys())[:4] if pruned_map else []
    if not first_countries:
        print("No countries available after pruning.")
        return
    
    first_holdings = {}
    w_init = 1.0 / len(first_countries)
    for country in first_countries:
        for ticker in pruned_map.get(country, [])[:1]:  # One ticker per country initially
            first_holdings[ticker] = w_init
    
    current_w = pd.Series(0.0, index=universe)
    for t, w in first_holdings.items():
        if t in current_w.index:
            current_w[t] = w
    
    pending_w = None
    prev_w = current_w.copy()

    idx = px.index

    for i in range(1, len(idx)):
        prev_day = idx[i - 1]
        today = idx[i]

        # Activate pending weights
        if pending_w is not None:
            current_w = pending_w
            pending_w = None

        # Apply return using current_w
        day_ret_vec = rets.loc[today].reindex(universe).fillna(0.0)
        port_ret = float((current_w * day_ret_vec).sum())
        equity *= (1 + port_ret)

        # Rebalance decision (based on prev_day)
        if prev_day in rebalance_dates:
            if prev_day in mom.index and not mom.loc[prev_day].dropna().empty:
                try:
                    sel_country, holdings, scores = select_country_and_holdings(
                        mom,
                        prev_day,
                        pruned_map,
                        cfg,
                    )
                    target_w = pd.Series(0.0, index=universe)
                    for t, w in holdings:
                        if t in target_w.index:
                            target_w[t] = float(w)
                    pending_w = target_w.reindex(universe).fillna(0.0)
                except Exception:
                    pass

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
