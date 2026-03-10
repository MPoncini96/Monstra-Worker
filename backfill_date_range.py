#!/usr/bin/env python
"""
Backfill Alpha1 and Alpha2 bots for a specific date range.
Usage: python backfill_date_range.py [start_date] [end_date]
Example: python backfill_date_range.py 2026-03-02 2026-03-05
"""

import os
import sys
from datetime import date

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_loader import load_env
load_env()

import backfill_alpha1
import backfill_alpha2


def backfill_date_range(start_date: str, end_date: str):
    """Backfill both Alpha1 and Alpha2 for a specific date range."""
    
    print(f"Starting backfill for date range: {start_date} to {end_date}\n")
    
    # Backfill Alpha1
    print("=" * 60)
    print("BACKFILLING ALPHA1 BOTS")
    print("=" * 60)
    backfill_alpha1.START_DATE = start_date
    backfill_alpha1.END_DATE = end_date
    backfill_alpha1.backfill_alpha1()
    print("[OK] Alpha1 backfill complete!\n")
    
    # Backfill Alpha2
    print("=" * 60)
    print("BACKFILLING ALPHA2 BOTS")
    print("=" * 60)
    backfill_alpha2.START_DATE = start_date
    backfill_alpha2.END_DATE = end_date
    backfill_alpha2.backfill_alpha2()
    print("[OK] Alpha2 backfill complete!\n")
    
    print("=" * 60)
    print("ALL BACKFILLS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        start_date = sys.argv[1]
        end_date = sys.argv[2]
    elif len(sys.argv) == 1:
        # Default to Mar 2 to Mar 5, 2026
        start_date = "2026-03-02"
        end_date = "2026-03-05"
        print(f"No date range provided. Using default: {start_date} to {end_date}")
    else:
        print("Usage: python backfill_date_range.py [start_date] [end_date]")
        print("Example: python backfill_date_range.py 2026-03-02 2026-03-05")
        sys.exit(1)
    
    backfill_date_range(start_date, end_date)
