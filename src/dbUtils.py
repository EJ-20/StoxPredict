import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path("data/stocks.db")

# Create a connection to the database
def create_connection():
    conn = sqlite3.connect(DB_PATH)
    return conn

# Create a table if it doesn't exist
def create_table_if_not_exists(ticker):
    query = f"""
    CREATE TABLE IF NOT EXISTS {ticker} (
        Date TEXT PRIMARY KEY,
        Open REAL,
        High REAL,
        Low REAL,
        Close REAL,
        Adj_Close REAL,
        Volume INTEGER
    );
    """
    with create_connection() as conn:
        conn.execute(query)
        conn.commit()

# Insert stock data into the database
def insert_stock_data(ticker, df):
    df = df.reset_index()
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
    create_table_if_not_exists(ticker)
    
    with create_connection() as conn:
        df.to_sql(ticker, conn, if_exists='replace', index=False)

# Fetch stock data from the database
def fetch_stock_data(ticker):
    with create_connection() as conn:
        return pd.read_sql(f"SELECT * FROM {ticker}", conn, parse_dates=['Date'])

# List all tables in the database
def list_tables():
    with create_connection() as conn:
        result = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        return [row[0] for row in result]
