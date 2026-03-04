#!/usr/bin/env python
"""
Run all 7 database-driven backfill scripts in sequence.
Requires: DATABASE_URL environment variable set with valid PostgreSQL connection.

Usage:
    export DATABASE_URL="postgresql://user:password@host:port/dbname"
    python run_all_backfills.py
"""

import os
import sys
import subprocess
from pathlib import Path
from env_loader import load_env

# Load environment variables from .env file
load_env()

# Verify DATABASE_URL is set
if not os.environ.get('DATABASE_URL'):
    print("ERROR: DATABASE_URL environment variable not set")
    print("Create .env file: cp .env.example .env")
    print("Then edit .env and add your DATABASE_URL")
    sys.exit(1)

BACKFILL_SCRIPTS = [
    "backfill_alpha1.py",
    "backfill_alpha2.py",
]

STRATEGIES = ["Alpha1 (Rank Rotation)", "Alpha2 (Stateless Relay)"]

def run_backfill(script_path):
    """Run a consolidated backfill script."""
    strategy_name = Path(script_path).stem.replace("backfill_", "").upper()
    print(f"\n{'='*60}")
    print(f"Running {strategy_name} backfill")
    print(f"Script: {script_path}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False
        )
        print(f"\n✓ {strategy_name} backfill completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {strategy_name} backfill FAILED with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ {strategy_name} backfill ERROR: {e}")
        return False

def main():
    """Run all consolidated backfills in sequence."""
    print("STARTING BACKFILL PROCESS")
    print(f"Strategies: Alpha1 (5 bots) + Alpha2 (2 bots) = 7 total")
    print(f"Database: {os.environ.get('DATABASE_URL', 'NOT SET')[:50]}...")
    
    results = {}
    for i, script in enumerate(BACKFILL_SCRIPTS, 1):
        if not Path(script).exists():
            print(f"WARNING: Script not found: {script}")
            continue
        
        strategy = STRATEGIES[i - 1]
        results[strategy] = run_backfill(script)
    
    # Summary
    print(f"\n{'='*60}")
    print("BACKFILL SUMMARY")
    print(f"{'='*60}")
    successful = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"Completed: {successful}/{total} strategies")
    
    for strategy, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {strategy:30} {status}")
    
    if successful == total:
        print(f"\n✓ All {total} strategy backfills completed successfully!")
        return 0
    else:
        print(f"\n✗ {total - successful} backfill(s) failed. Review output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
