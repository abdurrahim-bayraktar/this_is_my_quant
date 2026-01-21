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
        # Standardize date columns
        prices = prices_df.copy()
        sentiment = sentiment_df.copy()
        
        prices["merge_date"] = pd.to_datetime(prices[price_date_col]).dt.date
        sentiment["merge_date"] = pd.to_datetime(sentiment[sentiment_date_col]).dt.date
        
        # Merge on ticker and date
        merged = prices.merge(
            sentiment,
            on=[ticker_col, "merge_date"],
            how="left",
        )
        
        # Fill missing sentiment with neutral values
        sentiment_cols = ["sentiment_mean", "sentiment_dispersion", "news_count"]
        for col in sentiment_cols:
            if col in merged.columns:
                if col == "sentiment_mean":
                    merged[col] = merged[col].fillna(0)  # Neutral
                elif col == "sentiment_dispersion":
                    merged[col] = merged[col].fillna(0)  # No dispersion
                elif col == "news_count":
                    merged[col] = merged[col].fillna(0)  # No news
        
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
        
        # Sentiment features
        features.extend(["sentiment_mean", "sentiment_dispersion", "news_count"])
        
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
