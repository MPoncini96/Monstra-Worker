#!/usr/bin/env python
"""
Backfill all Alpha1 bots (Bellator, Imperium, Medicus, Vectura, Vis) using trading.alpha1 configuration.
Uses actual historical price data and applies 40-30-20-10 rank weights.
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


def backfill_alpha1():
    """Backfill all active Alpha1 bots from trading.alpha1 configuration."""
    
    # Load all active bots from trading.alpha1
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT bot_id, name, universe, top_n, lookback_days, cash_equivalent
                FROM trading.alpha1
                WHERE is_active = true
                ORDER BY bot_id
                """
            )
            bots = cur.fetchall()
    
    if not bots:
        print("No active bots found in trading.alpha1")
        return
    
    print(f"Found {len(bots)} active bots in trading.alpha1")
    print(f"Backfilling from {START_DATE} to {END_DATE}\n")
    
    for bot_row in bots:
        bot_id, name, universe_json, top_n, lookback_days, cash_equiv = bot_row
        
        # Parse universe JSONB
        try:
            if isinstance(universe_json, str):
                universe = json.loads(universe_json)
            else:
                universe = universe_json or []
        except:
            universe = []
        
        print(f"Backfilling {bot_id.upper()} ({name})")
        print(f"  Universe: {len(universe)} tickers, Top N: {top_n}, Lookback: {lookback_days}d")
        
        if not universe:
            print(f"  Warning: Empty universe, skipping\n")
            continue
        
        # Download historical price data
        print(f"  Downloading price data...")
        try:
            data = yf.download(
                universe,
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
        
        # For each trading day, calculate which stocks are top N based on trailing returns
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
            
            # Pick top N stocks
            n_holdings = min(top_n or len(universe), len(trailing_rets))
            top_stocks = trailing_rets.head(n_holdings).index.tolist()
            
            # Assign 40-30-20-10 weights (or equal if less than 4)
            if len(top_stocks) <= len(RANK_WEIGHTS):
                weights = RANK_WEIGHTS[:len(top_stocks)]
            else:
                weights = RANK_WEIGHTS
            
            # Normalize in case we have fewer than 4 stocks
            weights = weights / weights.sum()
            
            # Calculate daily portfolio return
            day_ret_vec = rets.loc[trading_date]
            port_ret = 0.0
            
            for stock, w in zip(top_stocks, weights):
                if stock in day_ret_vec.index and not pd.isna(day_ret_vec[stock]):
                    port_ret += w * day_ret_vec[stock]
            
            equity *= (1 + port_ret)
            
            # Create holdings dictionary
            holdings = {stock: float(w) for stock, w in zip(top_stocks, weights)}
            holdings["_meta"] = {
                "is_backtest": True,
                "type": "alpha1",
                "num_holdings": len(top_stocks),
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
    backfill_alpha1()
    print("[OK] Alpha1 backfill complete!")
