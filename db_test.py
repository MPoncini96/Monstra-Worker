import os
import psycopg

print("Connecting to DB...")

conn = psycopg.connect(os.environ["DATABASE_URL"])
print("Connected!")

conn.execute("select 1")
print("Query worked!")
