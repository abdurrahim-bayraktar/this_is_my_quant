"""
Check data availability for price-only baseline.

Analyzes:
1. How many trading days per stock (2015-2025)
2. Total samples available for multi-stock training
3. Sequence count after windowing
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yfinance as yf
import pandas as pd
from datetime import datetime

# Top stocks to analyze
STOCKS = [
    # High liquidity, long history
    "MSFT", "AAPL", "GOOGL", "AMZN", "META", "NVDA",  # Tech
    "JPM", "BAC", "WFC", "GS",  # Financials
    "JNJ", "PFE", "UNH",  # Healthcare
    "XOM", "CVX",  # Energy
    "WMT", "PG", "KO",  # Consumer
    "DIS", "NFLX",  # Entertainment
]

def check_data_availability():
    """Check how much price data is available per stock."""
    
    print("=" * 80)
    print("PRICE DATA AVAILABILITY CHECK (2015-01-01 to 2025-01-01)")
    print("=" * 80)
    
    results = []
    
    for ticker in STOCKS:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start="2015-01-01", end="2025-01-01")
            
            if len(df) > 0:
                # Calculate sequences (with lookback of 20)
                sequence_count = len(df) - 20
                
                results.append({
                    "ticker": ticker,
                    "trading_days": len(df),
                    "sequences": sequence_count,
                    "start_date": df.index[0].strftime("%Y-%m-%d"),
                    "end_date": df.index[-1].strftime("%Y-%m-%d"),
                    "years": (df.index[-1] - df.index[0]).days / 365.25
                })
                
                print(f"{ticker:6s}: {len(df):5d} days | {sequence_count:5d} sequences | "
                      f"{results[-1]['start_date']} to {results[-1]['end_date']}")
            else:
                print(f"{ticker:6s}: NO DATA")
                
        except Exception as e:
            print(f"{ticker:6s}: ERROR - {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total_days = sum(r["trading_days"] for r in results)
    total_sequences = sum(r["sequences"] for r in results)
    avg_days = total_days / len(results) if results else 0
    
    print(f"\nStocks analyzed: {len(results)}")
    print(f"Average trading days per stock: {avg_days:.0f}")
    print(f"Total trading days (all stocks): {total_days:,}")
    print(f"Total sequences (lookback=20): {total_sequences:,}")
    
    # Multi-stock training analysis
    print("\n" + "=" * 80)
    print("MULTI-STOCK TRAINING ANALYSIS")
    print("=" * 80)
    
    scenarios = [
        ("Single stock (MSFT)", 1, results[0]["sequences"] if results else 0),
        ("5 stocks", 5, sum(r["sequences"] for r in results[:5])),
        ("10 stocks", 10, sum(r["sequences"] for r in results[:10])),
        ("20 stocks", 20, total_sequences),
    ]
    
    model_params_small = 50_000  # 64 hidden, 1 layer
    model_params_medium = 150_000  # 128 hidden, 2 layers
    model_params_large = 265_000  # Current model
    
    print("\nSamples per Parameter Ratio (target: 10-50x)")
    print("-" * 60)
    
    for name, n_stocks, n_sequences in scenarios:
        ratio_small = n_sequences / model_params_small
        ratio_medium = n_sequences / model_params_medium
        ratio_large = n_sequences / model_params_large
        
        print(f"\n{name} ({n_sequences:,} samples):")
        print(f"  Small model (50K params):  {ratio_small:5.1f}x {'✓' if ratio_small >= 10 else '✗'}")
        print(f"  Medium model (150K params): {ratio_medium:5.1f}x {'✓' if ratio_medium >= 10 else '✗'}")
        print(f"  Large model (265K params): {ratio_large:5.1f}x {'✓' if ratio_large >= 10 else '✗'}")
    
    # Implications
    print("\n" + "=" * 80)
    print("IMPLICATIONS OF MULTI-STOCK TRAINING")
    print("=" * 80)
    
    print("""
PROS:
  + More data = less overfitting (can use larger/more expressive model)
  + Model learns general market patterns, not stock-specific quirks
  + Better generalization to unseen stocks
  + Can use full 40+ indicators without severe overfitting

CONS:
  - Different stocks have different dynamics (tech vs energy vs financials)
  - May need stock-specific features (sector encoding, market cap category)
  - Training time increases linearly with data
  - Need careful normalization (per-stock vs global)

RECOMMENDED APPROACH:
  1. First: Train on single stock (MSFT) with full 10 years data
  2. Second: Add 5 similar tech stocks (cross-validation)
  3. Third: Add diverse stocks with sector encoding
  
  With 10 years of single-stock data (~2500 samples), you can:
  - Use the current model size (265K params) with 9.4x ratio
  - Better: reduce to medium (150K) for 16.7x ratio (safe zone)
""")

    return results

if __name__ == "__main__":
    check_data_availability()
