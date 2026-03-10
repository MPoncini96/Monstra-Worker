#!/usr/bin/env python
"""Check equity data for alpha1 bots to diagnose issues."""

import sys
sys.path.insert(0, '.')

from env_loader import load_env
load_env()

from db import get_conn

# Check all alpha1 bots
bots = ['bellator', 'imperium', 'medicus', 'vectura', 'vis']

for bot_id in bots:
    print(f"\n{'='*60}")
    print(f"{bot_id.upper()} - Last 20 records")
    print(f"{'='*60}")
    
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT d, equity, ret 
                FROM trading.bot_equity 
                WHERE bot_id = %s 
                ORDER BY d DESC 
                LIMIT 20
                """,
                (bot_id,)
            )
            rows = cur.fetchall()
            
            if not rows:
                print(f"No data found for {bot_id}")
                continue
            
            print(f"{'Date':<12} {'Equity':<12} {'Return %':<12}")
            print("-" * 40)
            for date, equity, ret in rows:
                ret_pct = (ret * 100) if ret is not None else 0.0
                print(f"{str(date):<12} {equity:>10.4f}   {ret_pct:>8.2f}%")
