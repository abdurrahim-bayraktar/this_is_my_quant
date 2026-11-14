"""
Data utility functions for stock experiments.

Provides reusable data transformations:
- Weekly resampling of OHLCV data
- Windowed sequence creation with temporal splits
- Batch stock loading from yfinance

Usage:
    from src.data.utils import resample_to_weekly, load_stocks, create_sequences
"""

import logging
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def resample_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily OHLCV data to weekly (Friday close).
    
    Args:
        df: Daily OHLCV DataFrame with DatetimeIndex.
        
    Returns:
        Weekly OHLCV DataFrame.
    """
    weekly = pd.DataFrame()
    weekly['Open'] = df['Open'].resample('W-FRI').first()
    weekly['High'] = df['High'].resample('W-FRI').max()
    weekly['Low'] = df['Low'].resample('W-FRI').min()
    weekly['Close'] = df['Close'].resample('W-FRI').last()
    weekly['Volume'] = df['Volume'].resample('W-FRI').sum()
    
    if 'Ticker' in df.columns:
        weekly['Ticker'] = df['Ticker'].iloc[0]
    
    return weekly.dropna()


def load_stocks(
    tickers: List[str],
    start_date: str,
    end_date: str,
    cache=None,
    min_rows: int = 100,
    weekly: bool = False,
    cache_suffix: str = "",
) -> Dict[str, pd.DataFrame]:
    """
    Load stock data from cache or yfinance.
    
    Args:
        tickers: List of ticker symbols.
        start_date: Start date string.
        end_date: End date string.
        cache: Optional PriceCache instance.
        min_rows: Minimum rows required per stock.
        weekly: Whether to resample to weekly data.
        cache_suffix: Suffix for cache file.
        
    Returns:
        Dict mapping ticker -> DataFrame.
    """
    import yfinance as yf
    
    stock_data = {}
    suffix = cache_suffix or ("_weekly" if weekly else "")
    
    # Try cache
    if cache:
        cached = cache.load(start_date, end_date, tickers, suffix=suffix)
        if cached:
            stock_data = cached
            if len(stock_data) >= len(tickers) * 0.9:
                return stock_data
    
    # Try daily cache + resample if weekly requested
    if weekly and cache:
        daily_cached = cache.load(start_date, end_date, tickers, suffix="")
        if daily_cached:
            for ticker, df in daily_cached.items():
                if ticker not in stock_data and ticker in tickers:
                    try:
                        weekly_df = resample_to_weekly(df)
                        if len(weekly_df) >= min_rows:
                            stock_data[ticker] = weekly_df
                    except Exception:
                        pass
    
    # Download missing
    missing = [t for t in tickers if t not in stock_data]
    if missing:
        logger.info(f"Downloading {len(missing)} stocks...")
        batch_size = 100
        for i in range(0, len(missing), batch_size):
            batch = missing[i:i + batch_size]
            try:
                data = yf.download(
                    batch, start=start_date, end=end_date,
                    group_by="ticker", threads=True, progress=True,
                )
                for ticker in batch:
                    try:
                        df = data[ticker].copy() if len(batch) > 1 else data.copy()
                        df = df.dropna()
                        if len(df) < min_rows:
                            continue
                        df.columns = [
                            c.capitalize() if isinstance(c, str) else c
                            for c in df.columns
                        ]
                        df['Ticker'] = ticker
                        
                        if weekly:
                            df = resample_to_weekly(df)
                            if len(df) < min_rows:
                                continue
                        
                        stock_data[ticker] = df
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"Batch download failed: {e}")
        
        # Save to cache
        if cache and stock_data:
            cache.save(stock_data, start_date, end_date, suffix=suffix)
    
    logger.info(f"Loaded {len(stock_data)} stocks")
    return stock_data


def create_sequences(
    feature_data: np.ndarray,
    labels: np.ndarray,
    dates: pd.DatetimeIndex,
    seq_length: int,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
) -> Tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
]:
    """
    Create windowed sequences with temporal train/val/test split.
    
    Args:
        feature_data: Array of shape (n_timesteps, n_features).
        labels: Array of shape (n_timesteps,).
        dates: DatetimeIndex aligned with feature_data.
        seq_length: Number of timesteps per sequence.
        train_end: Cutoff date for training data.
        val_end: Cutoff date for validation data.
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []
    
    for i in range(len(feature_data) - seq_length):
        target_date = dates[i + seq_length]
        seq = feature_data[i:i + seq_length]
        label = labels[i + seq_length - 1]
        
        if target_date < train_end:
            X_train.append(seq)
            y_train.append(label)
        elif target_date < val_end:
            X_val.append(seq)
            y_val.append(label)
        else:
            X_test.append(seq)
            y_test.append(label)
    
    def _to_array(lst, dtype):
        return np.array(lst, dtype=dtype) if lst else np.array([])
    
    return (
        _to_array(X_train, np.float32), _to_array(y_train, np.int64),
        _to_array(X_val, np.float32), _to_array(y_val, np.int64),
        _to_array(X_test, np.float32), _to_array(y_test, np.int64),
    )
