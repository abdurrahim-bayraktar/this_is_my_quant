
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Add src to path
sys.path.insert(0, str(Path.cwd()))

from config import TOP_200_TICKERS, CACHE_DIR, REPORTS_DIR
from src.data import DataPreprocessor
from src.nlp.aggregator import SentimentAggregator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_cached_data(ticker):
    """Load cached parquet files."""
    base_dir = CACHE_DIR / "sentiment"
    
    news_file = base_dir / f"{ticker}_news.parquet"
    prices_file = base_dir / f"{ticker}_prices.parquet"
    sentiment_file = base_dir / f"{ticker}_sentiment.parquet"
    
    if not (news_file.exists() and prices_file.exists() and sentiment_file.exists()):
        logger.warning(f"Cache missing for {ticker}")
        return None, None, None
        
    prices_df = pd.read_parquet(prices_file)
    sentiment_df = pd.read_parquet(sentiment_file)
    
    return prices_df, sentiment_df

def plot_analysis(ticker, combined_df, save_dir):
    """Generate plots for debugging."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Price vs Sentiment Overlay
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price', color='tab:blue')
    ax1.plot(combined_df['date'], combined_df['close'], color='tab:blue', label='Close Price')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Sentiment (MA 5)', color='tab:orange')
    if 'sentiment_mean' in combined_df.columns:
        # Smooth sentiment for better visualization
        sent_ma = combined_df['sentiment_mean'].rolling(window=5).mean()
        ax2.plot(combined_df['date'], sent_ma, color='tab:orange', alpha=0.6, label='Sentiment (5d MA)')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
    
    plt.title(f'{ticker} Price vs Sentiment')
    fig.tight_layout()
    plt.savefig(save_dir / f"{ticker}_price_sentiment.png")
    plt.close()
    
    # 2. Return vs Sentiment Scatter
    if 'return' in combined_df.columns and 'sentiment_mean' in combined_df.columns:
        plt.figure(figsize=(8, 6))
        plt.scatter(combined_df['sentiment_mean'], combined_df['return'], alpha=0.3)
        plt.xlabel('Sentiment Score')
        plt.ylabel('Next Day Return')
        plt.title(f'{ticker} Return vs Sentiment')
        plt.axhline(0, color='gray', linestyle='--')
        plt.axvline(0, color='gray', linestyle='--')
        plt.savefig(save_dir / f"{ticker}_scatter.png")
        plt.close()

def check_distribution():
    # Only check stocks we know are cached
    cached_tickers = ["WMT", "MSFT", "DIS"]
    
    preprocessor = DataPreprocessor()
    aggregator = SentimentAggregator()
    
    reports_dir = REPORTS_DIR / "debug_plots"
    
    for ticker in cached_tickers:
        logger.info(f"Analyzing {ticker}...")
        
        prices_df, sentiment_df = load_cached_data(ticker)
        if prices_df is None:
            continue
            
        # Ensure ticker column exists
        if "ticker" not in prices_df.columns:
            prices_df["ticker"] = ticker
        if "ticker" not in sentiment_df.columns:
            sentiment_df["ticker"] = ticker
            
        # 1. Align and Aggregate
        # Check structure
        if "sentiment_value" not in sentiment_df.columns:
            logger.warning(f"Feature 'sentiment_value' missing for {ticker}, columns: {sentiment_df.columns}")
            # Try to reconstruct if label/score exist
            if "sentiment_score" in sentiment_df.columns and "sentiment_label" in sentiment_df.columns:
                 sentiment_df["sentiment_value"] = np.where(
                     sentiment_df["sentiment_label"] == "positive", sentiment_df["sentiment_score"],
                     np.where(sentiment_df["sentiment_label"] == "negative", -sentiment_df["sentiment_score"], 0)
                 )
            else:
                 continue

        # Calculate returns on prices
        prices_df = preprocessor.compute_returns(prices_df)
        prices_df = preprocessor.label_trends(prices_df)
        
        # Align news
        # Ensure date/timestamp
        if "date" not in sentiment_df.columns:
             if "timestamp" in sentiment_df.columns:
                 sentiment_df["date"] = sentiment_df["timestamp"]
             else:
                 logger.warning(f"No date column for {ticker}")
                 continue
                 
        aligned_news = preprocessor.align_news_to_trading_day(sentiment_df, prices_df)
        
        # Aggregate
        daily_sentiment = aggregator.aggregate_daily(aligned_news)
        
        # Merge - Explicit Type Conversion
        try:
            # Prices: ensure date is datetime and TZ-naive
            prices_df["date"] = pd.to_datetime(prices_df["date"]).dt.tz_localize(None).dt.normalize()
            # Sentiment: ensure trading_date is datetime and TZ-naive
            daily_sentiment["trading_date"] = pd.to_datetime(daily_sentiment["trading_date"]).dt.tz_localize(None).dt.normalize()
            
            # Print dtypes for debug
            print(f"Prices dtypes: {prices_df[['ticker', 'date']].dtypes}")
            print(f"Sentiment dtypes: {daily_sentiment[['ticker', 'trading_date']].dtypes}")
            
            combined = pd.merge(
                prices_df,
                daily_sentiment,
                left_on=["ticker", "date"],
                right_on=["ticker", "trading_date"],
                how="left"
            )
        except Exception as e:
            logger.error(f"Merge failed: {e}")
            continue
        
        # Fill missing sentiment
        combined["sentiment_mean"] = combined["sentiment_mean"].fillna(0)
        
        # Plot
        plot_analysis(ticker, combined, reports_dir)
        
        # Distribution stats
        print(f"\n=== {ticker} Distribution ===")
        print(combined["trend_label"].value_counts(normalize=True))
        
        # Sentiment stats
        print(f"Sentiment Mean: {combined['sentiment_mean'].mean():.4f}")
        print(f"Sentiment Std: {combined['sentiment_mean'].std():.4f}")
        
    print(f"\nPlots saved to {reports_dir}")

if __name__ == "__main__":
    check_distribution()
