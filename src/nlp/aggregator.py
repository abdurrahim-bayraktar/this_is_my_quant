"""
Daily sentiment aggregation.

This module aggregates multiple sentiment scores (from tweets/news)
into daily features for each stock.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, NamedTuple
from datetime import datetime
import logging

from config import feature_config

logger = logging.getLogger(__name__)


class DailySentiment(NamedTuple):
    """Container for aggregated daily sentiment."""
    mean: float              # Average sentiment value
    dispersion: float        # Standard deviation (disagreement)
    count: int               # Number of documents
    dominant_ratio: float    # Ratio of most common label
    bullish_ratio: float     # Proportion of positive sentiments


class SentimentAggregator:
    """
    Aggregates multiple sentiment scores to daily features.
    
    Supports multiple aggregation strategies:
    - simple: Simple average
    - time_weighted: Recent news weighted more heavily
    - volume_weighted: High-engagement content weighted more
    """
    
    def __init__(self, strategy: str = None):
        """
        Initialize the aggregator.
        
        Args:
            strategy: Aggregation strategy. Default from config.
        """
        self.strategy = strategy or feature_config.sentiment_aggregation
    
    def aggregate_daily(
        self,
        df: pd.DataFrame,
        sentiment_col: str = "sentiment_value",
        date_col: str = "trading_date",
        ticker_col: str = "ticker",
        timestamp_col: str = None,
        engagement_col: str = None,
    ) -> pd.DataFrame:
        """
        Aggregate sentiment scores to daily level.
        
        Args:
            df: DataFrame with individual sentiment scores.
            sentiment_col: Column with sentiment values.
            date_col: Column with trading date.
            ticker_col: Column with stock ticker.
            timestamp_col: Column with timestamp (for time_weighted).
            engagement_col: Column with engagement metric (for volume_weighted).
            
        Returns:
            DataFrame with daily aggregated features.
        """
        if self.strategy == "simple":
            return self._aggregate_simple(df, sentiment_col, date_col, ticker_col)
        elif self.strategy == "time_weighted":
            return self._aggregate_time_weighted(
                df, sentiment_col, date_col, ticker_col, timestamp_col
            )
        elif self.strategy == "sticky":
            return self._aggregate_sticky(
                df, sentiment_col, date_col, ticker_col
            )
        else:  # volume_weighted or default
            # Fallback to simple
            if self.strategy != "simple":
                logger.warning(f"Unknown strategy {self.strategy}, using simple")
            return self._aggregate_simple(df, sentiment_col, date_col, ticker_col)

    def _aggregate_simple(
        self,
        df: pd.DataFrame,
        sentiment_col: str,
        date_col: str,
        ticker_col: str,
    ) -> pd.DataFrame:
        """
        Simple average aggregation.
        """
        df = df.copy()
        
        # Group by ticker and date
        grouped = df.groupby([ticker_col, date_col])
        
        # Aggregate
        daily_df = grouped[sentiment_col].mean().reset_index()
        daily_df = daily_df.rename(columns={sentiment_col: "sentiment_mean"})
        
        # Calculate dispersion (std dev)
        daily_df["sentiment_dispersion"] = grouped[sentiment_col].std().values
        daily_df["sentiment_dispersion"] = daily_df["sentiment_dispersion"].fillna(0)
        
        # Count
        daily_df["news_count"] = grouped[sentiment_col].count().values
        
        logger.info(f"Simple aggregation: {len(daily_df)} daily observations")
        return daily_df
            
    def _aggregate_sticky(
        self,
        df: pd.DataFrame,
        sentiment_col: str,
        date_col: str,
        ticker_col: str,
        threshold: float = 0.1,
        decay_alpha: float = 0.95,
    ) -> pd.DataFrame:
        """
        Sticky sentiment aggregation.
        
        Logic:
        - If today has significant news (|score| > threshold): update state
        - If today is neutral/silent: decay previous state slowly
        - Also splits into positive and negative sentiment streams
        """
        df = df.copy()
        
        # Sort chronologically
        df = df.sort_values([ticker_col, date_col])
        
        daily_records = []
        
        for ticker, group in df.groupby(ticker_col):
            # Process each ticker's timeline
            current_state = 0.0
            current_pos_state = 0.0
            current_neg_state = 0.0
            
            # Group by day first to get daily raw signals
            daily_groups = group.groupby(date_col)
            
            for date, daily_news in daily_groups:
                # Calculate daily raw stats
                daily_mean = daily_news[sentiment_col].mean()
                
                # Split features
                if "sentiment_positive" in daily_news.columns:
                    daily_pos = daily_news["sentiment_positive"].mean()
                    daily_neg = daily_news["sentiment_negative"].mean()
                else:
                    # Infer from value
                    daily_pos = daily_news[daily_news[sentiment_col] > 0][sentiment_col].mean() if (daily_news[sentiment_col] > 0).any() else 0
                    daily_neg = -daily_news[daily_news[sentiment_col] < 0][sentiment_col].mean() if (daily_news[sentiment_col] < 0).any() else 0
                
                # Update Sticky State
                # If significant news update state
                if abs(daily_mean) > threshold:
                    current_state = daily_mean
                else:
                    # Decay slowly
                    current_state = current_state * decay_alpha
                    
                # Update split states independently? 
                # Or just use daily values? 
                # Let's apply sticky logic to split states too
                if daily_pos > threshold:
                    current_pos_state = daily_pos
                else:
                    current_pos_state = current_pos_state * decay_alpha
                    
                if daily_neg > threshold:
                    current_neg_state = daily_neg
                else:
                    current_neg_state = current_neg_state * decay_alpha
                
                daily_records.append({
                    ticker_col: ticker,
                    date_col: date,
                    "sentiment_mean": current_state,  # Sticky mean
                    "sentiment_pos": current_pos_state, # Sticky positive
                    "sentiment_neg": current_neg_state, # Sticky negative
                    "sentiment_raw": daily_mean,      # Raw mean (for reference)
                    "news_count": len(daily_news),
                    "sentiment_dispersion": daily_news[sentiment_col].std() if len(daily_news) > 1 else 0
                })
                
        result_df = pd.DataFrame(daily_records)
        logger.info(f"Sticky aggregation: {len(result_df)} daily observations")
        return result_df
        
    def _aggregate_volume_weighted(
        self,
        df: pd.DataFrame,
        sentiment_col: str,
        date_col: str,
        ticker_col: str,
        engagement_col: str = None,
    ) -> pd.DataFrame:
        """
        Volume/engagement-weighted average.
        
        Higher engagement (retweets, likes) = higher weight.
        """
        df = df.copy()
        
        if engagement_col and engagement_col in df.columns:
            # Use engagement as weight (log scale to reduce outlier impact)
            df["engagement_weight"] = np.log1p(df[engagement_col])
        else:
            # No engagement data, use equal weights
            df["engagement_weight"] = 1.0
        
        # Normalize weights within each group
        def weighted_mean(group):
            weights = group["engagement_weight"]
            values = group[sentiment_col]
            return (values * weights).sum() / weights.sum()
        
        results = []
        for (ticker, date), group in df.groupby([ticker_col, date_col]):
            results.append({
                ticker_col: ticker,
                date_col: date,
                "sentiment_mean": weighted_mean(group),
                "sentiment_dispersion": group[sentiment_col].std() if len(group) > 1 else 0,
                "news_count": len(group),
                "total_engagement": group["engagement_weight"].sum(),
            })
        
        result_df = pd.DataFrame(results)
        logger.info(f"Volume-weighted aggregation: {len(result_df)} daily observations")
        return result_df
    
    def compute_sentiment_momentum(
        self,
        df: pd.DataFrame,
        sentiment_col: str = "sentiment_mean",
        windows: List[int] = [3, 5, 10],
    ) -> pd.DataFrame:
        """
        Compute sentiment momentum features.
        
        Captures change in sentiment over time.
        
        Args:
            df: DataFrame with daily sentiment.
            sentiment_col: Column with sentiment values.
            windows: Rolling window sizes.
            
        Returns:
            DataFrame with momentum features added.
        """
        df = df.copy()
        
        for ticker, group in df.groupby("ticker"):
            group = group.sort_values("trading_date")
            idx = group.index
            
            for window in windows:
                # Rolling average
                df.loc[idx, f"sentiment_ma{window}"] = (
                    group[sentiment_col].rolling(window, min_periods=1).mean()
                )
                
                # Sentiment change
                df.loc[idx, f"sentiment_change{window}"] = (
                    group[sentiment_col] - group[sentiment_col].shift(window)
                )
        
        return df


def main():
    """Test sentiment aggregation."""
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    
    sample_data = pd.DataFrame({
        "ticker": ["AAPL"] * 30,
        "trading_date": np.repeat(dates[:10], 3),
        "sentiment_value": np.random.randn(30) * 0.3,
        "sentiment_label": np.random.choice(["positive", "negative", "neutral"], 30),
    })
    
    aggregator = SentimentAggregator(strategy="simple")
    
    result = aggregator.aggregate_daily(sample_data)
    print("Aggregated sentiment:")
    print(result.head(10))
    
    # Add momentum
    result = aggregator.compute_sentiment_momentum(result)
    print("\nWith momentum features:")
    print(result.columns.tolist())


if __name__ == "__main__":
    main()
