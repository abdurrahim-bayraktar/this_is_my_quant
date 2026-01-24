"""
Feature engineering for financial time series.

This module computes:
- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Sentiment aggregation features
- Volume-based features
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    
from config import feature_config

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering for financial prediction.
    
    Computes technical indicators and aggregates sentiment features
    for use in LSTM models.
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        if not TA_AVAILABLE:
            logger.warning("'ta' library not installed. Some indicators will be unavailable.")
    
    def add_technical_indicators(
        self,
        df: pd.DataFrame,
        indicators: List[str] = None,
    ) -> pd.DataFrame:
        """
        Add technical indicators to price data.
        
        Args:
            df: DataFrame with OHLCV data.
            indicators: List of indicator names to compute.
            
        Returns:
            DataFrame with added indicator columns.
        """
        indicators = indicators or feature_config.technical_indicators
        df = df.copy()
        
        # Process each stock separately
        result_dfs = []
        
        for ticker, group in df.groupby("ticker"):
            group = group.sort_values("date").copy()
            group = self._compute_indicators_for_stock(group, indicators)
            result_dfs.append(group)
        
        result = pd.concat(result_dfs, ignore_index=True)
        
        # Fill NaN values from indicator warmup period
        indicator_cols = [c for c in result.columns if c not in df.columns]
        result[indicator_cols] = result[indicator_cols].fillna(method="bfill")
        
        logger.info(f"Added {len(indicator_cols)} technical indicators")
        return result
    
    def _compute_indicators_for_stock(
        self,
        df: pd.DataFrame,
        indicators: List[str],
    ) -> pd.DataFrame:
        """Compute indicators for a single stock."""
        
        if not TA_AVAILABLE:
            # Fallback: compute basic indicators manually
            return self._compute_basic_indicators(df)
        
        # Use 'ta' library for comprehensive indicators
        high = df["high"]
        low = df["low"]
        close = df["close"]
        volume = df["volume"]
        
        for ind in indicators:
            try:
                if ind == "rsi_14":
                    df["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
                    
                elif ind == "macd":
                    macd = ta.trend.MACD(close)
                    df["macd"] = macd.macd()
                    df["macd_signal"] = macd.macd_signal()
                    df["macd_hist"] = macd.macd_diff()
                    
                elif ind.startswith("bb_"):
                    bb = ta.volatility.BollingerBands(close)
                    df["bb_upper"] = bb.bollinger_hband()
                    df["bb_middle"] = bb.bollinger_mavg()
                    df["bb_lower"] = bb.bollinger_lband()
                    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
                    
                elif ind == "sma_5":
                    df["sma_5"] = ta.trend.SMAIndicator(close, window=5).sma_indicator()
                elif ind == "sma_20":
                    df["sma_20"] = ta.trend.SMAIndicator(close, window=20).sma_indicator()
                elif ind == "sma_50":
                    df["sma_50"] = ta.trend.SMAIndicator(close, window=50).sma_indicator()
                    
                elif ind == "ema_12":
                    df["ema_12"] = ta.trend.EMAIndicator(close, window=12).ema_indicator()
                elif ind == "ema_26":
                    df["ema_26"] = ta.trend.EMAIndicator(close, window=26).ema_indicator()
                    
                elif ind == "atr_14":
                    df["atr_14"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
                    
                elif ind == "mfi_14":
                    df["mfi_14"] = ta.volume.MFIIndicator(high, low, close, volume, window=14).money_flow_index()
                    
                elif ind == "obv":
                    df["obv"] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
                    
            except Exception as e:
                logger.warning(f"Failed to compute {ind}: {e}")
                
        return df
    
    def _compute_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute basic indicators without 'ta' library."""
        close = df["close"]
        
        # Simple Moving Averages
        df["sma_5"] = close.rolling(window=5).mean()
        df["sma_20"] = close.rolling(window=20).mean()
        
        # Exponential Moving Averages
        df["ema_12"] = close.ewm(span=12, adjust=False).mean()
        df["ema_26"] = close.ewm(span=26, adjust=False).mean()
        
        # MACD
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        
        # RSI (simplified)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        df["rsi_14"] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df["bb_middle"] = close.rolling(window=20).mean()
        std = close.rolling(window=20).std()
        df["bb_upper"] = df["bb_middle"] + 2 * std
        df["bb_lower"] = df["bb_middle"] - 2 * std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        
        return df
    
    def aggregate_daily_sentiment(
        self,
        sentiment_df: pd.DataFrame,
        strategy: str = None,
        date_col: str = "trading_date",
        ticker_col: str = "ticker",
        sentiment_col: str = "sentiment_score",
    ) -> pd.DataFrame:
        """
        Aggregate multiple sentiment scores to daily features.
        
        Args:
            sentiment_df: DataFrame with individual sentiment scores.
            strategy: Aggregation strategy ("simple", "volume_weighted", "time_weighted").
            date_col: Date column for grouping.
            ticker_col: Ticker column for grouping.
            sentiment_col: Column with sentiment scores.
            
        Returns:
            DataFrame with daily aggregated sentiment features.
        """
        strategy = strategy or feature_config.sentiment_aggregation
        df = sentiment_df.copy()
        
        # Group by ticker and date
        grouped = df.groupby([ticker_col, date_col])
        
        if strategy == "simple":
            agg_df = grouped.agg({
                sentiment_col: ["mean", "std", "count"],
            }).reset_index()
            
        elif strategy == "time_weighted":
            # More recent news gets higher weight
            # Assuming there's a timestamp column
            def time_weighted_mean(group):
                if "timestamp" in group.columns:
                    # Convert to seconds from market close
                    weights = np.exp(-group["hours_to_close"] / 4)  # 4-hour half-life
                    return (group[sentiment_col] * weights).sum() / weights.sum()
                return group[sentiment_col].mean()
            
            agg_df = grouped.apply(
                lambda g: pd.Series({
                    "sentiment_mean": g[sentiment_col].mean(),
                    "sentiment_std": g[sentiment_col].std(),
                    "sentiment_count": len(g),
                })
            ).reset_index()
            
        else:  # volume_weighted or default
            agg_df = grouped.agg({
                sentiment_col: ["mean", "std", "count"],
            }).reset_index()
        
        # Flatten column names
        agg_df.columns = [
            f"{a}_{b}" if b else a 
            for a, b in agg_df.columns
        ]
        
        # Rename for clarity
        agg_df = agg_df.rename(columns={
            f"{sentiment_col}_mean": "sentiment_mean",
            f"{sentiment_col}_std": "sentiment_dispersion",
            f"{sentiment_col}_count": "news_count",
        })
        
        # Fill NaN dispersion (when only 1 article)
        agg_df["sentiment_dispersion"] = agg_df["sentiment_dispersion"].fillna(0)
        
        logger.info(f"Aggregated sentiment for {len(agg_df)} ticker-day pairs")
        return agg_df
    
    def add_event_based_sentiment_features(
        self,
        sentiment_df: pd.DataFrame,
        date_col: str = "trading_date",
        ticker_col: str = "ticker",
        sentiment_col: str = "sentiment_score",
    ) -> pd.DataFrame:
        """
        Add event-based sentiment features that capture extremes.
        
        This is an Iteration 3 enhancement to improve signal quality by
        preserving extreme sentiment events instead of averaging them away.
        
        Features added:
        - sentiment_mean: Daily mean sentiment
        - sentiment_std: Daily sentiment standard deviation
        - sentiment_min: Daily minimum sentiment (most negative)
        - sentiment_max: Daily maximum sentiment (most positive)
        - sentiment_range: Max - Min (sentiment volatility)
        - has_extreme_news: Flag for |sentiment| > 0.7
        - news_count: Number of news items
        
        Args:
            sentiment_df: DataFrame with individual sentiment scores.
            date_col: Date column for grouping.
            ticker_col: Ticker column for grouping.
            sentiment_col: Column with sentiment scores.
            
        Returns:
            DataFrame with daily aggregated sentiment features.
        """
        df = sentiment_df.copy()
        
        # Ensure sentiment column exists
        if sentiment_col not in df.columns:
            # Try alternative column names
            for alt_col in ['sentiment_positive', 'sentiment_value', 'sentiment']:
                if alt_col in df.columns:
                    sentiment_col = alt_col
                    break
            else:
                logger.warning(f"Sentiment column not found, using zeros")
                df[sentiment_col] = 0.0
        
        # Group by ticker and date
        grouped = df.groupby([ticker_col, date_col])
        
        # Aggregate with multiple statistics
        agg_df = grouped.agg({
            sentiment_col: ['mean', 'std', 'min', 'max', 'count']
        }).reset_index()
        
        # Flatten column names
        agg_df.columns = [
            ticker_col, date_col,
            'sentiment_mean', 'sentiment_dispersion',
            'sentiment_min', 'sentiment_max', 'news_count'
        ]
        
        # Compute derived features
        agg_df['sentiment_range'] = agg_df['sentiment_max'] - agg_df['sentiment_min']
        
        # Fix: Use percentile-based threshold instead of fixed 0.7
        # This ensures ~10% of days are marked as "extreme" regardless of sentiment scale
        abs_max = agg_df['sentiment_max'].abs()
        abs_min = agg_df['sentiment_min'].abs()
        max_sentiment = np.maximum(abs_max, abs_min)
        
        # Per-ticker percentile threshold (90th percentile of max sentiment)
        thresholds = agg_df.groupby(ticker_col)[['sentiment_max']].transform(
            lambda x: np.percentile(x.abs(), 90)
        ).values.flatten()
        
        # Fallback to 0.3 if threshold is too low (sparse data)
        thresholds = np.maximum(thresholds, 0.3)
        
        agg_df['has_extreme_news'] = (max_sentiment > thresholds).astype(int)
        
        # Fill NaN (single article days have no std)
        agg_df['sentiment_dispersion'] = agg_df['sentiment_dispersion'].fillna(0)
        
        logger.info(f"Created event-based features for {len(agg_df)} ticker-day pairs")
        logger.info(f"  Extreme news days: {agg_df['has_extreme_news'].sum()} ({agg_df['has_extreme_news'].mean():.1%})")
        
        return agg_df
    
    def add_sentiment_momentum(
        self,
        df: pd.DataFrame,
        sentiment_col: str = "sentiment_mean",
        ticker_col: str = "ticker",
    ) -> pd.DataFrame:
        """
        Add sentiment momentum features.
        
        This is an Iteration 3 enhancement to capture sentiment trends
        over time, not just point-in-time values.
        
        Features added:
        - sentiment_momentum: 3-day rate of change in sentiment
        - sentiment_acceleration: Derivative of momentum (momentum of momentum)
        
        Args:
            df: DataFrame with daily sentiment (must have sentiment_mean column).
            sentiment_col: Column with sentiment values.
            ticker_col: Ticker column for grouping.
            
        Returns:
            DataFrame with added momentum features.
        """
        df = df.copy()
        
        if sentiment_col not in df.columns:
            logger.warning(f"Column {sentiment_col} not found, skipping momentum features")
            df['sentiment_momentum'] = 0.0
            df['sentiment_acceleration'] = 0.0
            return df
        
        # Compute momentum per ticker
        momentum_list = []
        accel_list = []
        
        for ticker, group in df.groupby(ticker_col):
            group = group.sort_values('date').copy()
            
            # 3-day momentum (rate of change)
            momentum = group[sentiment_col].diff(periods=3)
            
            # Acceleration (change in momentum)
            acceleration = momentum.diff(periods=1)
            
            momentum_list.append(momentum)
            accel_list.append(acceleration)
        
        # Combine results
        df['sentiment_momentum'] = pd.concat(momentum_list).reindex(df.index)
        df['sentiment_acceleration'] = pd.concat(accel_list).reindex(df.index)
        
        # Fill NaN with 0 (neutral)
        df['sentiment_momentum'] = df['sentiment_momentum'].fillna(0)
        df['sentiment_acceleration'] = df['sentiment_acceleration'].fillna(0)
        
        logger.info(f"Added sentiment momentum features")
        
        return df
    
    def merge_price_and_sentiment(
        self,
        prices_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        price_date_col: str = "date",
        sentiment_date_col: str = "trading_date",
        ticker_col: str = "ticker",
    ) -> pd.DataFrame:
        """
        Merge price data with aggregated sentiment features.
        
        Args:
            prices_df: DataFrame with OHLCV + technical indicators.
            sentiment_df: DataFrame with daily sentiment features.
            
        Returns:
            Merged DataFrame ready for model training.
        """
        # Standardize date columns - handle timezone issues
        prices = prices_df.copy()
        sentiment = sentiment_df.copy()
        
        # Convert to datetime and remove timezone info to avoid comparison issues
        # Using utc=True ensures uniform handling before removing timezone
        prices_dates = pd.to_datetime(prices[price_date_col], utc=True).dt.tz_localize(None)
        prices["merge_date"] = prices_dates.dt.date
        
        sentiment_dates = pd.to_datetime(sentiment[sentiment_date_col], utc=True).dt.tz_localize(None)
        sentiment["merge_date"] = sentiment_dates.dt.date
        
        # Merge on ticker and date
        merged = prices.merge(
            sentiment,
            on=[ticker_col, "merge_date"],
            how="left",
        )
        
        # Fill missing sentiment with neutral values
        # Iteration 3 updates: added max/min/range/momentum
        # Iteration 3b: Sticky Sentiment features
        sentiment_cols = [
            "sentiment_mean", "sentiment_dispersion", "news_count", "has_news",
            "sentiment_max", "sentiment_min", "sentiment_range", "has_extreme_news",
            "sentiment_momentum", "sentiment_acceleration",
            "sentiment_pos", "sentiment_neg", "sentiment_raw"
        ]
        
        for col in sentiment_cols:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0)
        
        merged = merged.drop(columns=["merge_date"])
        
        logger.info(f"Merged data: {len(merged)} rows, {len(merged.columns)} columns")
        return merged
    
    def get_feature_columns(self) -> List[str]:
        """
        Get list of all feature columns for model input.
        
        Returns:
            List of feature column names.
        """
        features = []
        
        # Price features (normalized)
        features.extend(["open", "high", "low", "close", "volume"])
        
        # Technical indicators
        features.extend(feature_config.technical_indicators)
        
        # Sentiment features (including has_news for temporal alignment)
        features.extend(["sentiment_mean", "sentiment_dispersion", "news_count", "has_news"])
        
        # Iteration 3: Event-based sentiment features
        features.extend([
            "sentiment_max", "sentiment_min", "sentiment_range", "has_extreme_news",
        ])
        
        # Iteration 3: Sentiment momentum features
        features.extend([
            "sentiment_momentum", "sentiment_acceleration",
        ])
        
        # Iteration 3b: Split Sentiment
        features.extend(["sentiment_pos", "sentiment_neg"])
        
        return features


def main():
    """Test feature engineering."""
    logging.basicConfig(level=logging.INFO)
    
    # Create sample OHLCV data
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        "date": dates,
        "ticker": "AAPL",
        "open": 100 + np.cumsum(np.random.randn(100) * 0.5),
        "high": 101 + np.cumsum(np.random.randn(100) * 0.5),
        "low": 99 + np.cumsum(np.random.randn(100) * 0.5),
        "close": 100 + np.cumsum(np.random.randn(100) * 0.5),
        "volume": np.random.randint(1000000, 10000000, 100),
    })
    
    engineer = FeatureEngineer()
    
    # Add technical indicators
    df = engineer.add_technical_indicators(sample_data)
    print("Technical indicators added:")
    print(df.columns.tolist())
    print(df[["date", "close", "rsi_14", "macd", "bb_width"]].head(10))


if __name__ == "__main__":
    main()
