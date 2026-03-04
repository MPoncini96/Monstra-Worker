#!/usr/bin/env python
"""
Backfill all Alpha2 bots (Cyclus, Viator) using trading.alpha2 configuration.
Uses actual historical price data for proxies and applies 40-30-20-10 rank weights.
"""

import os
import sys
import json
from datetime import timedelta, date
import pandas as pd
import numpy as np
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_loader import load_env
load_env()

from db import get_conn

START_DATE = "2025-01-01"
END_DATE = date.today().isoformat()

# Rank weights: top 1=40%, top 2=30%, top 3=20%, top 4=10%
RANK_WEIGHTS = np.array([0.40, 0.30, 0.20, 0.10])


def backfill_alpha2():
    """Backfill all active Alpha2 bots from trading.alpha2 configuration."""
    
    # Load all active bots from trading.alpha2
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT bot_id, proxies, stocks, lookback_days, rebalance_freq, stock_bench_mode, rank_weights
                FROM trading.alpha2
                WHERE is_active = true
                ORDER BY bot_id
                """
            )
            bots = cur.fetchall()
    
    if not bots:
        print("No active bots found in trading.alpha2")
        return
    
    print(f"Found {len(bots)} active bots in trading.alpha2")
    print(f"Backfilling from {START_DATE} to {END_DATE}\n")
    
    for bot_row in bots:
        bot_id, proxies_json, stocks_json, lookback_days, rebalance_freq, stock_bench_mode, rank_weights_json = bot_row
        
        # Parse JSONB fields
        try:
            if isinstance(proxies_json, str):
                proxies = json.loads(proxies_json)
            else:
                proxies = proxies_json or {}
        except:
            proxies = {}
        
        try:
            if isinstance(stocks_json, str):
                stocks = json.loads(stocks_json)
            else:
                stocks = stocks_json or {}
        except:
            stocks = {}
        
        try:
            if isinstance(rank_weights_json, str):
                rank_weights = np.array(json.loads(rank_weights_json))
            else:
                rank_weights = np.array(rank_weights_json or [0.40, 0.30, 0.20, 0.10])
        except:
            rank_weights = RANK_WEIGHTS
        
        print(f"Backfilling {bot_id.upper()}")
        print(f"  Mode: {stock_bench_mode}, Lookback: {lookback_days}d, Rebalance: {rebalance_freq}")
        print(f"  Proxies: {len(proxies)}")
        
        if not proxies:
            print(f"  Warning: Empty proxies, skipping\n")
            continue
        
        # Download historical price data for proxies
        proxy_list = list(proxies.values())
        print(f"  Downloading price data for {len(proxy_list)} proxies...")
        
        try:
            data = yf.download(
                proxy_list,
                start=START_DATE,
                end=END_DATE,
                interval="1d",
                auto_adjust=True,
                progress=False,
            )["Close"]
        except Exception as e:
            print(f"  Error downloading prices: {e}\n")
            continue
        
        if data.empty:
            print(f"  No price data available\n")
            continue
        
        data = data.dropna(how="all").sort_index()
        print(f"  {len(data)} trading days available")
        
        # Calculate daily returns
        rets = data.pct_change().fillna(0.0)
        
        # Backfill trading
        equity = 1.0
        processed = 0
        errors = 0
        
        # For each trading day, calculate which proxies are top N based on trailing returns
        for i, trading_date in enumerate(data.index):
            if i == 0:
                continue  # Skip first day (no returns yet)
            
            # Get lookback window
            lookback_start = max(0, i - lookback_days)
            lookback_data = data.iloc[lookback_start:i]
            
            if len(lookback_data) < 2:
                continue
            
            # Calculate trailing returns for lookback period
            trailing_rets = (lookback_data.iloc[-1] / lookback_data.iloc[0]) - 1
            trailing_rets = trailing_rets.sort_values(ascending=False)
            
            # Pick top 4 proxies (relay strategy)
            top_proxies = trailing_rets.head(4).index.tolist()
            
            # Assign 40-30-20-10 weights
            weights = rank_weights[:len(top_proxies)]
            weights = weights / weights.sum()  # Normalize
            
            # Calculate daily portfolio return
            day_ret_vec = rets.loc[trading_date]
            port_ret = 0.0
            
            for proxy, w in zip(top_proxies, weights):
                if proxy in day_ret_vec.index and not pd.isna(day_ret_vec[proxy]):
                    port_ret += w * day_ret_vec[proxy]
            
            equity *= (1 + port_ret)
            
            # Create holdings dictionary with proxies
            holdings = {proxy: float(w) for proxy, w in zip(top_proxies, weights)}
            holdings["_meta"] = {
                "is_backtest": True,
                "type": "alpha2",
                "mode": stock_bench_mode,
                "num_proxies": len(top_proxies),
                "lookback_days": lookback_days,
            }
            
            try:
                with get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO trading.bot_equity (bot_id, d, equity, ret, holdings)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (bot_id, d)
                            DO UPDATE SET equity = EXCLUDED.equity, ret = EXCLUDED.ret, holdings = EXCLUDED.holdings
                            """,
                            (bot_id, trading_date.date(), float(equity), float(port_ret), json.dumps(holdings))
                        )
                    conn.commit()
                processed += 1
            except Exception as e:
                errors += 1
                if errors <= 3:  # Log first 3 errors
                    print(f"    Error on {trading_date.date()}: {e}")
        
        final_ret_pct = (equity - 1) * 100
        print(f"  [OK] Processed {processed} days. Final equity: {equity:.4f} ({final_ret_pct:.2f}%)")
        if errors > 3:
            print(f"    ({errors - 3} additional errors not shown)")
        print()


if __name__ == "__main__":
    backfill_alpha2()
    print("[OK] Alpha2 backfill complete!")
