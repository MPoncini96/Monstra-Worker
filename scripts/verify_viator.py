import psycopg2
import json

conn = psycopg2.connect('postgresql://monstra:IRNocUqjjnwdWHvqslU9TPEZ9qJt6sil@dpg-d66f04npm1nc73dhc5bg-a.oregon-postgres.render.com/monstra')
cur = conn.cursor()

# Check viator equity for first, middle, and last rows
cur.execute('SELECT d, equity, ret, holdings FROM bot_equity WHERE bot_id = %s ORDER BY d LIMIT 1', ('viator',))
rows = cur.fetchall()
for r in rows:
    d, eq, ret, h = r
    if isinstance(h, str):
        h_dict = json.loads(h)
    else:
        h_dict = h
    print(f'First: {d}, equity={eq}, ret={ret}, holdings_keys={list(h_dict.keys())[:3]}')

cur.execute('SELECT COUNT(*) FROM bot_equity WHERE bot_id = %s', ('viator',))
count = cur.fetchone()[0]
print(f'Total rows: {count}')

cur.execute('SELECT d, equity, ret FROM bot_equity WHERE bot_id = %s ORDER BY d OFFSET %s LIMIT 1', ('viator', count // 2))
rows = cur.fetchall()
for r in rows:
    print(f'Mid:   {r[0]}, equity={r[1]}, ret={r[2]}')

cur.execute('SELECT d, equity, ret FROM bot_equity WHERE bot_id = %s ORDER BY d DESC LIMIT 1', ('viator',))
rows = cur.fetchall()
for r in rows:
    print(f'Last:  {r[0]}, equity={r[1]}, ret={r[2]}')

cur.close()
conn.close()
