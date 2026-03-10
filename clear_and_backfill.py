#!/usr/bin/env python
"""
Clear bot_equity database and run backfill_alpha1 and backfill_alpha2
"""
import os
import sys
import psycopg2
from env_loader import load_env

load_env()

def clear_bot_equity():
    """Clear the bot_equity table"""
    DATABASE_URL = os.environ["DATABASE_URL"]
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        print("Clearing bot_equity table...")
        cursor.execute("TRUNCATE TABLE trading.bot_equity CASCADE")
        conn.commit()
        print("✓ bot_equity table cleared successfully")
        
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"✗ Error clearing bot_equity: {e}")
        raise

if __name__ == "__main__":
    # Clear the database
    clear_bot_equity()
    
    # Import and run backfill_alpha1
    print("\n" + "="*50)
    print("Running backfill_alpha1...")
    print("="*50 + "\n")
    from backfill_alpha1 import backfill_alpha1
    backfill_alpha1()
    
    # Import and run backfill_alpha2
    print("\n" + "="*50)
    print("Running backfill_alpha2...")
    print("="*50 + "\n")
    from backfill_alpha2 import backfill_alpha2
    backfill_alpha2()
    
    print("\n" + "="*50)
    print("All operations completed!")
    print("="*50)
