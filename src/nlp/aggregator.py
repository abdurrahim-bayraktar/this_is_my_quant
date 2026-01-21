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
        elif self.strategy == "volume_weighted":
            return self._aggregate_volume_weighted(
                df, sentiment_col, date_col, ticker_col, engagement_col
            )
        else:
            logger.warning(f"Unknown strategy {self.strategy}, using simple")
            return self._aggregate_simple(df, sentiment_col, date_col, ticker_col)
    
    def _aggregate_simple(
        self,
        df: pd.DataFrame,
        sentiment_col: str,
        date_col: str,
        ticker_col: str,
    ) -> pd.DataFrame:
        """Simple mean aggregation."""
        
        agg_funcs = {
            sentiment_col: ["mean", "std", "count"],
        }
        
        # Add label distribution if available
        if "sentiment_label" in df.columns:
            agg_funcs["sentiment_label"] = lambda x: (x == "positive").mean()
        
        grouped = df.groupby([ticker_col, date_col]).agg(agg_funcs)
        grouped.columns = ["_".join(col).strip("_") for col in grouped.columns.values]
        grouped = grouped.reset_index()
        
        # Rename columns for clarity
        grouped = grouped.rename(columns={
            f"{sentiment_col}_mean": "sentiment_mean",
            f"{sentiment_col}_std": "sentiment_dispersion",
            f"{sentiment_col}_count": "news_count",
            "sentiment_label_<lambda>": "bullish_ratio",
        })
        
        # Fill NaN dispersion (when only 1 article)
        grouped["sentiment_dispersion"] = grouped["sentiment_dispersion"].fillna(0)
        
        logger.info(f"Aggregated {len(df)} records to {len(grouped)} daily observations")
        return grouped
    
    def _aggregate_time_weighted(
        self,
        df: pd.DataFrame,
        sentiment_col: str,
        date_col: str,
        ticker_col: str,
        timestamp_col: str = None,
    ) -> pd.DataFrame:
        """
        Time-weighted average: more recent news gets higher weight.
        
        Uses exponential decay with ~4 hour half-life.
        """
        df = df.copy()
        
        if timestamp_col and timestamp_col in df.columns:
            # Calculate hours until market close (16:00)
            df["timestamp"] = pd.to_datetime(df[timestamp_col])
            market_close = df["timestamp"].dt.normalize() + pd.Timedelta(hours=16)
            df["hours_to_close"] = (market_close - df["timestamp"]).dt.total_seconds() / 3600
            df["hours_to_close"] = df["hours_to_close"].clip(lower=0)
            
            # Exponential weight: more recent = higher weight
            half_life = 4  # hours
            df["time_weight"] = np.exp(-df["hours_to_close"] / half_life)
        else:
            # No timestamp, use equal weights
            df["time_weight"] = 1.0
        
        # Weighted aggregation
        def weighted_mean(group):
            weights = group["time_weight"]
            values = group[sentiment_col]
            return (values * weights).sum() / weights.sum()
        
        def weighted_std(group):
            weights = group["time_weight"]
            values = group[sentiment_col]
            mean = (values * weights).sum() / weights.sum()
            variance = (weights * (values - mean) ** 2).sum() / weights.sum()
            return np.sqrt(variance)
        
        results = []
        for (ticker, date), group in df.groupby([ticker_col, date_col]):
            results.append({
                ticker_col: ticker,
                date_col: date,
                "sentiment_mean": weighted_mean(group),
                "sentiment_dispersion": weighted_std(group) if len(group) > 1 else 0,
                "news_count": len(group),
                "bullish_ratio": (group["sentiment_label"] == "positive").mean() if "sentiment_label" in group.columns else 0.5,
            })
        
        result_df = pd.DataFrame(results)
        logger.info(f"Time-weighted aggregation: {len(result_df)} daily observations")
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
