
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path.cwd()))
from config import CACHE_DIR
from src.data import DataPreprocessor
from src.nlp.aggregator import SentimentAggregator

logging.basicConfig(level=logging.ERROR)

def load_cached_data(ticker):
    base_dir = CACHE_DIR / "sentiment"
    news_file = base_dir / f"{ticker}_news.parquet"
    prices_file = base_dir / f"{ticker}_prices.parquet"
    sentiment_file = base_dir / f"{ticker}_sentiment.parquet"
    
    if not (news_file.exists() and prices_file.exists() and sentiment_file.exists()):
        return None, None, None
        
    prices_df = pd.read_parquet(prices_file)
    sentiment_df = pd.read_parquet(sentiment_file)
    return prices_df, sentiment_df

def check_stats():
    tickers = ["WMT", "MSFT", "DIS"]
    preprocessor = DataPreprocessor()
    aggregator = SentimentAggregator()
    
    print("\n=== Data Statistics ===")
    
    for ticker in tickers:
        prices_df, sentiment_df = load_cached_data(ticker)
        if prices_df is None: continue
            
        prices_df["ticker"] = ticker
        sentiment_df["ticker"] = ticker
        
        # Reconstruct sentiment_value if needed
        if "sentiment_value" not in sentiment_df.columns and "sentiment_score" in sentiment_df.columns:
             sentiment_df["sentiment_value"] = np.where(
                 sentiment_df["sentiment_label"] == "positive", sentiment_df["sentiment_score"],
                 np.where(sentiment_df["sentiment_label"] == "negative", -sentiment_df["sentiment_score"], 0)
             )
             
        if "date" not in sentiment_df.columns and "timestamp" in sentiment_df.columns:
             sentiment_df["date"] = sentiment_df["timestamp"]

        # Align
        aligned_news = preprocessor.align_news_to_trading_day(sentiment_df, prices_df)
        
        # Aggregate
        daily_sentiment = aggregator.aggregate_daily(aligned_news)
        
        # Merge
        try:
            prices_df["date"] = pd.to_datetime(prices_df["date"]).dt.normalize()
            daily_sentiment["trading_date"] = pd.to_datetime(daily_sentiment["trading_date"]).dt.normalize()
            
            combined = pd.merge(prices_df, daily_sentiment, left_on=["ticker", "date"], right_on=["ticker", "trading_date"], how="left")
            combined["sentiment_mean"] = combined["sentiment_mean"].fillna(0)
        except Exception as e:
            print(f"Merge failed for {ticker}: {e}")
            continue
        
        # Stats
        total_days = len(combined)
        days_with_news = len(combined[combined['news_count'].notna()])
        coverage = days_with_news / total_days
        
        sent_mean = combined["sentiment_mean"].mean()
        sent_std = combined["sentiment_mean"].std()
        
        # Correlation with next day return
        # Returns are calculated on CLOSE. 'return' column is already next-day return?
        # preprocessor.compute_returns calculates pct_change(1) which is (Today - Yesterday)/Yesterday.
        # This is the return achieved AT the end of today.
        # We want to predict TOMORROW's return using TODAY's sentiment.
        # So we should shift returns backwards by 1 to align "Target Return" with "Today's Features"
        
        # In preprocessor.create_sequences:
        # y_all.append(targets[i + seq_len]) -> Target is the trend at step T (future).
        # X is 0..T-1.
        
        # For simple correlation check:
        # sentiment at T should correlate with Return at T+1.
        
        # Compute returns
        prices_df = prices_df.sort_values("date")
        prices_df["return"] = prices_df["close"].pct_change()
        prices_df["next_return"] = prices_df["return"].shift(-1)
        
        # Re-merge to get correct alignment for correlation
        combined_corr = pd.merge(prices_df, daily_sentiment, left_on=["ticker", "date"], right_on=["ticker", "trading_date"], how="left")
        combined_corr["sentiment_mean"] = combined_corr["sentiment_mean"].fillna(0)
        
        corr = combined_corr["sentiment_mean"].corr(combined_corr["next_return"])
        
        print(f"\nStock: {ticker}")
        print(f"  Coverage: {coverage:.1%} ({days_with_news}/{total_days} days)")
        print(f"  Sentiment Mean: {sent_mean:.4f}")
        print(f"  Sentiment Std:  {sent_std:.4f}")
        print(f"  Corr (Sent vs Next Ret): {corr:.4f}")

if __name__ == "__main__":
    check_stats()
