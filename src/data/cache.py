"""
Price data caching for yfinance downloads.

Provides persistent pickle-based caching to avoid re-downloading
stock data across experiment runs.

Usage:
    from src.data.cache import PriceCache
    
    cache = PriceCache()
    data = cache.load("2014-01-01", "2024-12-31", tickers)
    if data is None:
        data = download_stocks(...)
        cache.save(data, "2014-01-01", "2024-12-31")
"""

import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from config import DATA_DIR
    _DEFAULT_CACHE_DIR = DATA_DIR / "price_cache"
except ImportError:
    _DEFAULT_CACHE_DIR = Path("data/price_cache")


class PriceCache:
    """Persistent pickle cache for yfinance price data."""
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_path(self, start_date: str, end_date: str, suffix: str = "") -> Path:
        """Get cache file path for a date range."""
        return self.cache_dir / f"prices_{start_date}_{end_date}{suffix}.pkl"
    
    def load(
        self,
        start_date: str,
        end_date: str,
        requested_stocks: List[str] = None,
        suffix: str = "",
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Load cached price data.
        
        Args:
            start_date: Start date string.
            end_date: End date string.
            requested_stocks: Optional list to filter cached data.
            suffix: Optional suffix for cache file (e.g., "_weekly").
            
        Returns:
            Dict mapping ticker -> DataFrame, or None if cache miss.
        """
        cache_path = self.get_cache_path(start_date, end_date, suffix)
        if not cache_path.exists():
            return None
        
        logger.info(f"Loading cached prices from {cache_path}")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        
        if requested_stocks:
            data = {t: df for t, df in data.items() if t in requested_stocks}
        
        logger.info(f"Loaded {len(data)} stocks from cache")
        return data
    
    def save(
        self,
        data: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        suffix: str = "",
        overwrite: bool = False,
    ) -> None:
        """
        Save price data to cache.
        
        Args:
            data: Dict mapping ticker -> DataFrame.
            start_date: Start date string.
            end_date: End date string.
            suffix: Optional suffix for cache file.
            overwrite: Whether to overwrite existing cache.
        """
        cache_path = self.get_cache_path(start_date, end_date, suffix)
        if cache_path.exists() and not overwrite:
            logger.debug(f"Cache already exists: {cache_path}")
            return
        
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved {len(data)} stocks to {cache_path}")
