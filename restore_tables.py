#!/usr/bin/env python
import psycopg2

DATABASE_URL = "postgresql://monstra:IRNocUqjjnwdWHvqslU9TPEZ9qJt6sil@dpg-d66f04npm1nc73dhc5bg-a.oregon-postgres.render.com/monstra"

create_tables_sql = """
CREATE TABLE IF NOT EXISTS bot_equity (
    bot_id VARCHAR(50) NOT NULL,
    d DATE NOT NULL,
    equity FLOAT NOT NULL,
    ret FLOAT,
    holdings JSONB DEFAULT '{}'::jsonb,
    updated_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (bot_id, d)
);

CREATE INDEX IF NOT EXISTS idx_bot_equity_bot_id_d ON bot_equity(bot_id, d);

CREATE TABLE IF NOT EXISTS bot_state (
    bot_id VARCHAR(50) PRIMARY KEY,
    peak_equity FLOAT DEFAULT 1.0,
    last_equity FLOAT DEFAULT 1.0,
    risk_off BOOLEAN DEFAULT FALSE,
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_bot_state_updated_at ON bot_state(updated_at);

CREATE TABLE IF NOT EXISTS signals (
    bot_id VARCHAR(50) NOT NULL,
    ts TIMESTAMP NOT NULL,
    signal VARCHAR(50) NOT NULL,
    note TEXT,
    payload JSONB DEFAULT '{}'::jsonb,
    PRIMARY KEY (bot_id, ts)
);

CREATE INDEX IF NOT EXISTS idx_signals_bot_id_ts ON signals(bot_id, ts);
"""

conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

try:
    cursor.execute(create_tables_sql)
    conn.commit()
    print("✓ Created bot_equity, bot_state, and signals tables")
except Exception as e:
    print(f"Error: {e}")
    conn.rollback()
finally:
    cursor.close()
    conn.close()
