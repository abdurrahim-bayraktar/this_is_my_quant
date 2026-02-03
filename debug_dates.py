
import pandas as pd
from pathlib import Path
from config import CACHE_DIR

def check_dates():
    # Pick a ticker that failed, e.g., LEN or AEO
    ticker = "LEN"
    
    sent_path = CACHE_DIR / "sentiment" / f"{ticker}_sentiment.parquet"
    if sent_path.exists():
        df = pd.read_parquet(sent_path)
        print(f"--- {ticker} Sentiment ---")
        if "date" in df.columns:
            print(f"Date Range: {df['date'].min()} to {df['date'].max()}")
        elif "timestamp" in df.columns:
            print(f"Timestamp Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(df.head())
    else:
        print(f"No sentiment file for {ticker}")

    # Check prices
    # We need to find where DatasetLoader saved the price file
    # It likely saved it in CACHE_DIR / f"prices_1stocks_..."
    # Let's list the price files
    price_files = list(CACHE_DIR.glob(f"prices_1stocks_*"))
    for p in price_files:
        df = pd.read_parquet(p)
        if ticker in df['ticker'].values:
            print(f"\n--- {ticker} Prices ({p.name}) ---")
            print(f"Date Range: {df['date'].min()} to {df['date'].max()}")
            print(df.head())

if __name__ == "__main__":
    check_dates()
