import yfinance as yf
import sys
import os
import time

# Add the parent directory to sys.path to import array_sorting
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from experiments.array_sorting import EXTENDED_TICKERS
except ImportError:
    # Fallback if running from within experiments folder directly
    try:
        from array_sorting import EXTENDED_TICKERS
    except ImportError:
        print("Error: Could not import EXTENDED_TICKERS from array_sorting.py")
        sys.exit(1)

def rank_tickers():
    print(f"Loaded {len(EXTENDED_TICKERS)} tickers.")
    
    # Remove duplicates if any
    tickers = list(set(EXTENDED_TICKERS))
    print(f"Unique tickers: {len(tickers)}")
    
    chunk_size = 1000
    ticker_data = []
    
    # We will sort by Price. 
    # High Price -> Low Price implies "High Quality" (Blue chips often > $1000 or $100) -> "Penny Stocks" (< $5)
    
    print("Fetching data (this may take a few minutes)...")
    
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        print(f"Processing chunk {i // chunk_size + 1} / {(len(tickers) + chunk_size - 1) // chunk_size}...")
        
        try:
            # download period='1d' to get latest price
            # threads=True is default
            data = yf.download(chunk, period="1d", progress=False)['Close']
            
            # If we only have one ticker in chunk (unlikely here) or unexpected format
            if data.empty:
                print("  Warning: Empty data for chunk.")
                continue

            # yfinance returns a DataFrame where columns are Tickers (if multiple)
            # If multiple tickers, data.columns are the tickers.
            # The last row contains the latest close.
            
            latest_prices = data.iloc[-1]
            
            for ticker in latest_prices.index:
                price = latest_prices[ticker]
                # Check for NaN (failed download or delisted)
                if price and price == price: # Check for not NaN
                    ticker_data.append((ticker, float(price)))
                
        except Exception as e:
            print(f"  Error processing chunk: {e}")
            # Fallback: try smaller chunks or just skip? 
            # ideally we shouldn't fail the whole chunk but yf.download is usually robust
            
        # Small sleep to be nice to API
        time.sleep(1)

    print(f"Successfully fetched prices for {len(ticker_data)} tickers.")
    
    # Sort by price descending
    # High Price first (High Quality?) -> Low Price last (Penny Stocks)
    ticker_data.sort(key=lambda x: x[1], reverse=True)
    
    # Extract just the tickers
    sorted_tickers = [t[0] for t in ticker_data]
    
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ranked_tickers.txt")
    
    with open(output_path, "w") as f:
        # custom formatting to match python array syntax but in a txt file
        f.write("[\n")
        # Write chunks of tickers per line to look nice
        line_buffer = []
        for i, t in enumerate(sorted_tickers):
            line_buffer.append(f'"{t}"')
            if len(line_buffer) >= 10:
                f.write(", ".join(line_buffer) + ",\n")
                line_buffer = []
        
        if line_buffer:
             f.write(", ".join(line_buffer) + "\n")
             
        f.write("]\n")
        
    print(f"Saved ranked tickers to {output_path}")

if __name__ == "__main__":
    rank_tickers()
