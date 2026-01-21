"""
Data preprocessing and trend labeling.

This module handles:
- Time alignment of news to trading sessions
- Price normalization
- Trend classification (Up/Down/Neutral)
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timedelta
import logging

from config import (
    trend_config,
    feature_config,
    training_config,
)

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocessor for financial time series and text data.
    
    Handles:
    - Price data normalization (Z-score, MinMax, Robust)
    - Return calculation
    - Trend labeling based on thresholds
    - Time alignment of news to trading days
    """
    
    def __init__(
        self,
        up_threshold: float = None,
        down_threshold: float = None,
        normalization: str = None,
    ):
        """
        Initialize preprocessor.
        
        Args:
            up_threshold: Return threshold for "Up" classification.
            down_threshold: Return threshold for "Down" classification.
            normalization: Normalization method ("zscore", "minmax", "robust").
        """
        self.up_threshold = up_threshold or trend_config.up_threshold
        self.down_threshold = down_threshold or trend_config.down_threshold
        self.normalization = normalization or feature_config.normalization
        
        # Store normalization parameters for inference
        self._norm_params: Dict[str, Tuple[float, float]] = {}
        
    def compute_returns(
        self,
        prices_df: pd.DataFrame,
        price_col: str = "close",
        periods: int = 1,
    ) -> pd.DataFrame:
        """
        Compute percentage returns.
        
        Args:
            prices_df: DataFrame with price data.
            price_col: Column name for price.
            periods: Number of periods for return calculation.
            
        Returns:
            DataFrame with added 'return' column.
        """
        df = prices_df.copy()
        
        # Sort by ticker and date
        df = df.sort_values(["ticker", "date"])
        
        # Compute returns within each ticker group
        df["return"] = df.groupby("ticker")[price_col].pct_change(periods=periods)
        
        # Remove first row per ticker (NaN return)
        df = df.dropna(subset=["return"])
        
        logger.info(f"Computed returns for {df['ticker'].nunique()} stocks")
        return df
    
    def label_trends(
        self,
        df: pd.DataFrame,
        return_col: str = "return",
    ) -> pd.DataFrame:
        """
        Label price movements as Up/Down/Neutral based on thresholds.
        
        Uses ±0.5% threshold by default (configurable).
        
        Args:
            df: DataFrame with return column.
            return_col: Name of the return column.
            
        Returns:
            DataFrame with added 'trend' and 'trend_label' columns.
        """
        df = df.copy()
        
        # Classify trends
        conditions = [
            df[return_col] > self.up_threshold,    # Up
            df[return_col] < self.down_threshold,  # Down
        ]
        choices = [2, 0]  # Up=2, Down=0
        
        df["trend"] = np.select(conditions, choices, default=1)  # Neutral=1
        df["trend_label"] = df["trend"].map({0: "Down", 1: "Neutral", 2: "Up"})
        
        # Log class distribution
        dist = df["trend_label"].value_counts(normalize=True)
        logger.info(f"Trend distribution: {dist.to_dict()}")
        
        return df
    
    def normalize_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        Normalize numerical features.
        
        Args:
            df: DataFrame with features.
            columns: Columns to normalize.
            fit: Whether to fit normalization parameters (True for training).
            
        Returns:
            DataFrame with normalized features.
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping normalization")
                continue
                
            if self.normalization == "zscore":
                if fit:
                    mean = df[col].mean()
                    std = df[col].std()
                    self._norm_params[col] = (mean, std)
                else:
                    mean, std = self._norm_params.get(col, (0, 1))
                
                df[col] = (df[col] - mean) / (std + 1e-8)
                
            elif self.normalization == "minmax":
                if fit:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    self._norm_params[col] = (min_val, max_val)
                else:
                    min_val, max_val = self._norm_params.get(col, (0, 1))
                
                df[col] = (df[col] - min_val) / (max_val - min_val + 1e-8)
                
            elif self.normalization == "robust":
                if fit:
                    median = df[col].median()
                    iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
                    self._norm_params[col] = (median, iqr)
                else:
                    median, iqr = self._norm_params.get(col, (0, 1))
                
                df[col] = (df[col] - median) / (iqr + 1e-8)
        
        return df
    
    def align_news_to_trading_day(
        self,
        news_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        news_date_col: str = "date",
        news_time_col: str = "time",
    ) -> pd.DataFrame:
        """
        Align news timestamps to trading day influence windows.
        
        Influence windows:
        - Pre-market (previous close to open): Affects today's open
        - Intraday (open to close): Affects today's close
        - Post-market (close to next open): Affects tomorrow's open
        
        Args:
            news_df: DataFrame with news data.
            prices_df: DataFrame with price data.
            news_date_col: Column name for news date.
            news_time_col: Column name for news time (optional).
            
        Returns:
            News DataFrame with 'trading_date' column for alignment.
        """
        df = news_df.copy()
        
        # Convert date column
        df[news_date_col] = pd.to_datetime(df[news_date_col])
        
        # Get trading dates from prices
        trading_dates = set(pd.to_datetime(prices_df["date"]).dt.date)
        
        # Map news to trading dates
        # If news is on a trading day, use that day
        # If news is on weekend/holiday, use next trading day
        
        def get_trading_date(news_date):
            news_d = news_date.date()
            
            # If it's a trading day, use it
            if news_d in trading_dates:
                return news_date
            
            # Otherwise find next trading day
            for i in range(1, 5):  # Max 4 days forward
                next_d = news_d + timedelta(days=i)
                if next_d in trading_dates:
                    return pd.Timestamp(next_d)
            
            return None
        
        df["trading_date"] = df[news_date_col].apply(get_trading_date)
        df = df.dropna(subset=["trading_date"])
        
        logger.info(f"Aligned {len(df)} news records to trading dates")
        return df
    
    def create_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "trend",
        sequence_length: int = None,
        ticker_col: str = "ticker",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            df: DataFrame with features and target.
            feature_cols: List of feature column names.
            target_col: Target column name.
            sequence_length: Length of input sequences.
            ticker_col: Column for grouping by stock.
            
        Returns:
            Tuple of (X, y, tickers) arrays.
        """
        seq_len = sequence_length or training_config.sequence_length
        
        X_all, y_all, tickers_all = [], [], []
        
        for ticker, group in df.groupby(ticker_col):
            group = group.sort_values("date")
            
            features = group[feature_cols].values
            targets = group[target_col].values
            
            # Create sliding windows
            for i in range(len(group) - seq_len):
                X_all.append(features[i:i + seq_len])
                y_all.append(targets[i + seq_len])  # Predict next day
                tickers_all.append(ticker)
        
        X = np.array(X_all)
        y = np.array(y_all)
        tickers = np.array(tickers_all)
        
        logger.info(f"Created {len(X)} sequences of length {seq_len}")
        return X, y, tickers
    
    def train_val_test_split(
        self,
        df: pd.DataFrame,
        train_end: str = None,
        val_end: str = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically into train/val/test sets.
        
        Args:
            df: DataFrame with 'date' column.
            train_end: End date for training data.
            val_end: End date for validation data.
            
        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        from config import dataset_config
        
        train_end = train_end or dataset_config.train_end_date
        val_end = val_end or dataset_config.val_end_date
        
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        
        train_df = df[df["date"] <= train_end]
        val_df = df[(df["date"] > train_end) & (df["date"] <= val_end)]
        test_df = df[df["date"] > val_end]
        
        logger.info(
            f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
        )
        
        return train_df, val_df, test_df


def main():
    """Test the preprocessor."""
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    sample_data = pd.DataFrame({
        "date": dates,
        "ticker": "AAPL",
        "close": 100 + np.cumsum(np.random.randn(100) * 0.5),
        "volume": np.random.randint(1000000, 10000000, 100),
    })
    
    preprocessor = DataPreprocessor()
    
    # Compute returns
    df = preprocessor.compute_returns(sample_data)
    print("Returns computed:")
    print(df[["date", "close", "return"]].head(10))
    
    # Label trends
    df = preprocessor.label_trends(df)
    print("\nTrend labels:")
    print(df["trend_label"].value_counts())
    
    # Normalize
    df = preprocessor.normalize_features(df, columns=["close", "volume"])
    print("\nNormalized features:")
    print(df[["close", "volume"]].describe())


if __name__ == "__main__":
    main()
