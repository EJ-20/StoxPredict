# populate_db.py

import yfinance as yf
from dbUtils import insert_stock_data
from datetime import datetime

# === Configuration ===
TICKERS = ["NVDA", "AMD", "TSLA"]
START_DATE = "2015-01-01"
END_DATE = datetime.today().strftime('%Y-%m-%d')

# Download and store stock data for a given ticker
def download_and_store(ticker, start, end):
    print(f"Fetching {ticker} from {start} to {end}...")
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end)

        if df.empty:
            print(f"No data found for {ticker}")
            return

        # Reset index to move Date from index to column
        df = df.reset_index()

        # Add Ticker column
        df["Ticker"] = ticker

        # Optional: Drop dividend/splits if not needed
        df = df[["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]]

        insert_stock_data(ticker, df)
        print(f"Stored {ticker} ({len(df)} rows)")
    except Exception as e:
        print(f"Failed to fetch/store {ticker}: {e}")


# Populate the database with all tickers
def populate_all():
    for ticker in TICKERS:
        download_and_store(ticker, START_DATE, END_DATE)


if __name__ == "__main__":
    populate_all()
