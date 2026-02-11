import os
import logging
import psycopg

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

logging.info("Connecting to DB...")

conn = psycopg.connect(os.environ["DATABASE_URL"])
logging.info("Connected!")

conn.execute("select 1")
logging.info("Query worked!")
