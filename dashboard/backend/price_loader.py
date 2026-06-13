"""
Price data loader for the dashboard.

Reads OHLCV data from the project's pickle-based price cache,
with yfinance fallback for missing tickers/date ranges.
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)

PRICE_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "price_cache"

# Ordered by preference (larger/newer caches first)
_CACHE_PREFERENCE = [
    "prices_2014-01-01_2025-12-31.pkl",
    "prices_2014-01-01_2026-01-01.pkl",
    "prices_2014-01-01_2025-04-25.pkl",
    "prices_2014-01-01_2024-12-31.pkl",
    "prices_594stocks_2014-01-01_2024-12-31.pkl",
    "prices_595stocks_2014-01-01_2024-12-31.pkl",
    "prices_2014-01-01_2023-01-01.pkl",
    "prices_2012-01-01_2019-12-31.pkl",
    "prices_2020-01-01_2024-12-31.pkl",
]

# In-memory cache
_loaded_caches: Dict[str, Dict[str, pd.DataFrame]] = {}


def _load_cache(cache_name: str) -> Optional[Dict[str, pd.DataFrame]]:
    """Load a single pickle cache file."""
    if cache_name in _loaded_caches:
        return _loaded_caches[cache_name]

    cache_path = PRICE_CACHE_DIR / cache_name
    if not cache_path.exists():
        return None

    try:
        logger.info(f"Loading price cache: {cache_name}")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        _loaded_caches[cache_name] = data
        logger.info(f"Loaded {len(data)} tickers from {cache_name}")
        return data
    except Exception as e:
        logger.warning(f"Failed to load {cache_name}: {e}")
        return None


def _find_ticker_in_caches(ticker: str) -> Optional[pd.DataFrame]:
    """Search through cache files for a specific ticker."""
    for cache_name in _CACHE_PREFERENCE:
        cache = _load_cache(cache_name)
        if cache and ticker in cache:
            return cache[ticker]

    # Try individual ticker files
    for suffix in ["_2014-01-01_2024-12-31"]:
        path = PRICE_CACHE_DIR / f"{ticker}{suffix}.pkl"
        if path.exists():
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass

    return None


def _download_ticker(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Download price data from yfinance as fallback."""
    try:
        import yfinance as yf
        logger.info(f"Downloading {ticker} from yfinance ({start} to {end})")
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            return None
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except ImportError:
        logger.warning("yfinance not installed — cannot download price data")
        return None
    except Exception as e:
        logger.warning(f"Failed to download {ticker}: {e}")
        return None


def get_ohlcv(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Get OHLCV data for a ticker, formatted for TradingView Lightweight Charts.

    Returns list of dicts with: time, open, high, low, close, volume
    """
    df = _find_ticker_in_caches(ticker)

    if df is None:
        # Fallback to yfinance
        s = start or "2014-01-01"
        e = end or datetime.now().strftime("%Y-%m-%d")
        df = _download_ticker(ticker, s, e)

    if df is None:
        return []

    # Filter by date range if specified
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]

    # Format for Lightweight Charts
    records = []
    for idx, row in df.iterrows():
        try:
            ts = idx
            if hasattr(ts, "tz") and ts.tz is not None:
                ts = ts.tz_localize(None)

            records.append({
                "time": ts.strftime("%Y-%m-%d"),
                "open": round(float(row.get("Open", row.get("open", 0))), 4),
                "high": round(float(row.get("High", row.get("high", 0))), 4),
                "low": round(float(row.get("Low", row.get("low", 0))), 4),
                "close": round(float(row.get("Close", row.get("close", 0))), 4),
                "volume": int(row.get("Volume", row.get("volume", 0))),
            })
        except (ValueError, TypeError):
            continue

    return records


def get_available_tickers() -> List[str]:
    """Get list of all tickers available in any cache."""
    tickers = set()
    for cache_name in _CACHE_PREFERENCE[:3]:  # Check top 3 caches only
        cache = _load_cache(cache_name)
        if cache:
            tickers.update(cache.keys())
            break  # First successful cache is enough

    return sorted(tickers)
