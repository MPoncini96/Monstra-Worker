#!/usr/bin/env python
import psycopg2
import os

DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://monstra:IRNocUqjjnwdWHvqslU9TPEZ9qJt6sil@dpg-d66f04npm1nc73dhc5bg-a.oregon-postgres.render.com/monstra')

conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

# Count total rows
cur.execute('SELECT COUNT(*) FROM public.bot_equity')
total = cur.fetchone()[0]
print(f"✓ Total rows in bot_equity: {total:,}")

# List bots
cur.execute('SELECT DISTINCT bot_id FROM public.bot_equity ORDER BY bot_id')
bots = [row[0] for row in cur.fetchall()]
print(f"✓ Bots present: {', '.join(bots)}")

# Date ranges per bot
cur.execute('SELECT bot_id, MIN(d), MAX(d), COUNT(*) FROM public.bot_equity GROUP BY bot_id ORDER BY bot_id')
rows = cur.fetchall()
print("\nDate ranges per bot:")
for r in rows:
    print(f"  {r[0]:12} → {r[1]} to {r[2]} ({r[3]:,} rows)")

conn.close()
print("\n✓ Backfill verification complete")
