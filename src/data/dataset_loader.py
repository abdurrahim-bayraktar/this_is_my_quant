"""
Dataset loader for FNSPID, Twitter Financial Sentiment, and other datasets.

This module handles loading and initial filtering of datasets from HuggingFace
and local sources.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
import json

from datasets import load_dataset, Dataset
from tqdm import tqdm

from config import (
    TOP_200_TICKERS,
    DATE_RANGE,
    RAW_DATA_DIR,
    CACHE_DIR,
    dataset_config,
)

logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Unified loader for financial datasets.
    
    Supports:
    - FNSPID: Financial News and Stock Price Integration Dataset
    - Twitter Financial Sentiment: Labeled tweets for sentiment
    - StockNet: Historical tweets + prices (benchmarking)
    """
    
    def __init__(
        self,
        tickers: List[str] = None,
        date_range: Tuple[str, str] = None,
        use_cache: bool = True,
    ):
        """
        Initialize the dataset loader.
        
        Args:
            tickers: List of stock tickers to filter. Defaults to TOP_200_TICKERS.
            date_range: Tuple of (start_date, end_date) in YYYY-MM-DD format.
            use_cache: Whether to use cached data if available.
        """
        self.tickers = tickers or TOP_200_TICKERS
        self.date_range = date_range or DATE_RANGE
        self.use_cache = use_cache
        self._ticker_set = set(self.tickers)
        
    def load_fnspid(
        self,
        split: str = "train",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load FNSPID dataset from HuggingFace.
        
        The FNSPID dataset contains:
        - 15.7M financial news records
        - 29.7M stock price records
        - Data for 4,775 S&P 500 companies (1999-2023)
        
        We filter to TOP_200_TICKERS and DATE_RANGE.
        
        Args:
            split: Dataset split to load ("train" for full dataset).
            
        Returns:
            Tuple of (news_df, prices_df) DataFrames.
        """
        cache_path = CACHE_DIR / f"fnspid_{len(self.tickers)}stocks_{self.date_range[0]}_{self.date_range[1]}.parquet"
        
        if self.use_cache and cache_path.exists():
            logger.info(f"Loading cached FNSPID data from {cache_path}")
            news_df = pd.read_parquet(cache_path.with_suffix(".news.parquet"))
            prices_df = pd.read_parquet(cache_path.with_suffix(".prices.parquet"))
            return news_df, prices_df
        
        logger.info("Attempting to load FNSPID dataset...")
        
        try:
            # Try loading without trust_remote_code (deprecated)
            dataset = load_dataset(
                dataset_config.fnspid_repo,
                split=split,
            )
        except Exception as e:
            logger.warning(f"Could not load FNSPID from HuggingFace: {e}")
            logger.info("Falling back to Yahoo Finance for price data...")
            logger.info("(News data will be empty - sentiment features will be neutral)")
            return self._load_fnspid_from_github()
        
        # Convert to DataFrame and filter
        df = dataset.to_pandas()
        
        # Filter by ticker
        if "ticker" in df.columns:
            df = df[df["ticker"].isin(self._ticker_set)]
        elif "symbol" in df.columns:
            df = df[df["symbol"].isin(self._ticker_set)]
            
        # Filter by date
        date_col = "date" if "date" in df.columns else "Date"
        df[date_col] = pd.to_datetime(df[date_col])
        df = df[
            (df[date_col] >= self.date_range[0]) &
            (df[date_col] <= self.date_range[1])
        ]
        
        # Separate news and prices if combined, otherwise return as-is
        if "headline" in df.columns or "title" in df.columns:
            news_df = df
            prices_df = self._load_prices_yfinance()
        else:
            news_df = df[df["type"] == "news"] if "type" in df.columns else df
            prices_df = df[df["type"] == "price"] if "type" in df.columns else self._load_prices_yfinance()
        
        # Cache the filtered data
        if self.use_cache:
            news_df.to_parquet(cache_path.with_suffix(".news.parquet"))
            prices_df.to_parquet(cache_path.with_suffix(".prices.parquet"))
            logger.info(f"Cached FNSPID data to {cache_path}")
        
        logger.info(f"Loaded {len(news_df)} news records and {len(prices_df)} price records")
        return news_df, prices_df
    
    def _load_fnspid_from_github(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Alternative: Load Twitter Financial Sentiment and extract ticker symbols.
        
        Since FNSPID is large (5.7GB), this provides a lighter alternative using
        the Twitter Financial Sentiment dataset with ticker extraction.
        """
        logger.info("Loading Twitter Financial Sentiment as alternative news source...")
        
        try:
            twitter_data = self.load_twitter_sentiment()
            
            # Convert to DataFrame
            news_records = []
            ticker_pattern = r'\$([A-Z]{1,5})\b'
            import re
            
            for item in twitter_data:
                text = item['text']
                label = item['label']  # 0=Bearish, 1=Bullish, 2=Neutral
                
                # Extract ticker symbols from tweet (e.g., $AAPL, $TSLA)
                tickers_found = re.findall(ticker_pattern, text)
                
                # Map label to sentiment
                sentiment_map = {0: 'negative', 1: 'positive', 2: 'neutral'}
                sentiment = sentiment_map.get(label, 'neutral')
                
                for ticker in tickers_found:
                    if ticker in self._ticker_set:
                        news_records.append({
                            'date': pd.Timestamp.now(),  # No date in this dataset
                            'ticker': ticker,
                            'headline': text,
                            'sentiment_label': sentiment,
                            'sentiment_score': 1.0 if label != 2 else 0.5,
                        })
            
            news_df = pd.DataFrame(news_records)
            logger.info(f"Extracted {len(news_df)} ticker-specific sentiment records from Twitter data")
            
            if len(news_df) == 0:
                logger.warning("No matching tickers found in Twitter data. Using empty news DataFrame.")
                news_df = pd.DataFrame(columns=["date", "ticker", "headline", "sentiment_label"])
                
        except Exception as e:
            logger.warning(f"Could not load Twitter sentiment: {e}")
            news_df = pd.DataFrame(columns=["date", "ticker", "headline", "sentiment_label"])
        
        # Load prices from yfinance
        prices_df = self._load_prices_yfinance()
        
        return news_df, prices_df
    
    def _load_prices_yfinance(self) -> pd.DataFrame:
        """
        Load price data from Yahoo Finance as fallback.
        
        Returns:
            DataFrame with OHLCV data for all tickers.
        """
        import yfinance as yf
        
        cache_path = CACHE_DIR / f"prices_{len(self.tickers)}stocks_{self.date_range[0]}_{self.date_range[1]}.parquet"
        
        if self.use_cache and cache_path.exists():
            logger.info(f"Loading cached price data from {cache_path}")
            return pd.read_parquet(cache_path)
        
        logger.info(f"Downloading price data for {len(self.tickers)} stocks from Yahoo Finance...")
        
        all_prices = []
        
        for ticker in tqdm(self.tickers, desc="Downloading prices"):
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=self.date_range[0], end=self.date_range[1])
                
                if not hist.empty:
                    hist = hist.reset_index()
                    hist["ticker"] = ticker
                    hist.columns = [c.lower().replace(" ", "_") for c in hist.columns]
                    all_prices.append(hist)
            except Exception as e:
                logger.warning(f"Failed to download {ticker}: {e}")
                continue
        
        if not all_prices:
            raise ValueError("No price data could be downloaded")
        
        prices_df = pd.concat(all_prices, ignore_index=True)
        
        # Standardize column names
        column_mapping = {
            "date": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "dividends": "dividends",
            "stock_splits": "stock_splits",
        }
        prices_df = prices_df.rename(columns=column_mapping)
        
        if self.use_cache:
            prices_df.to_parquet(cache_path)
            logger.info(f"Cached price data to {cache_path}")
        
        logger.info(f"Loaded {len(prices_df)} price records for {len(self.tickers)} stocks")
        return prices_df
    
    def load_twitter_sentiment(self) -> Dataset:
        """
        Load Twitter Financial News Sentiment dataset.
        
        This dataset contains 11,932 labeled tweets:
        - LABEL_0: Bearish
        - LABEL_1: Bullish  
        - LABEL_2: Neutral
        
        Returns:
            HuggingFace Dataset object.
        """
        logger.info("Loading Twitter Financial Sentiment dataset...")
        
        dataset = load_dataset(
            dataset_config.twitter_sentiment_repo,
            split="train",
        )
        
        logger.info(f"Loaded {len(dataset)} labeled tweets")
        return dataset
    
    def load_stocknet(self) -> Dict[str, pd.DataFrame]:
        """
        Load StockNet dataset for benchmarking.
        
        StockNet contains:
        - 88 stocks from 9 sectors
        - 2 years of data (2014-2016)
        - Pre-aligned tweets and prices
        
        Returns:
            Dictionary with 'tweets' and 'prices' DataFrames.
        """
        logger.info("Loading StockNet dataset...")
        
        # StockNet is hosted on GitHub, not HuggingFace
        # This would require cloning or downloading the repo
        
        stocknet_path = RAW_DATA_DIR / "stocknet-dataset"
        
        if not stocknet_path.exists():
            logger.warning(
                "StockNet not found locally. Please clone from: "
                "https://github.com/yumoxu/stocknet-dataset"
            )
            return {"tweets": pd.DataFrame(), "prices": pd.DataFrame()}
        
        # Load tweets
        tweets_path = stocknet_path / "tweet" / "preprocessed"
        prices_path = stocknet_path / "price" / "preprocessed"
        
        # Implementation would parse the StockNet format
        # For now, return placeholder
        return {"tweets": pd.DataFrame(), "prices": pd.DataFrame()}
    
    def get_data_summary(self) -> Dict:
        """
        Get a summary of available data.
        
        Returns:
            Dictionary with data statistics.
        """
        summary = {
            "tickers": len(self.tickers),
            "date_range": self.date_range,
            "cache_dir": str(CACHE_DIR),
        }
        
        # Check cache status
        cache_files = list(CACHE_DIR.glob("*.parquet"))
        summary["cached_files"] = len(cache_files)
        summary["cache_size_mb"] = sum(f.stat().st_size for f in cache_files) / (1024 * 1024)
        
        return summary


def main():
    """Test the dataset loader."""
    logging.basicConfig(level=logging.INFO)
    
    loader = DatasetLoader(
        tickers=TOP_200_TICKERS[:10],  # Test with 10 stocks
        use_cache=True,
    )
    
    print("Data Summary:", loader.get_data_summary())
    
    # Load prices
    _, prices = loader.load_fnspid()
    print(f"\nPrices shape: {prices.shape}")
    print(prices.head())
    
    # Load Twitter sentiment
    twitter = loader.load_twitter_sentiment()
    print(f"\nTwitter dataset: {twitter}")
    print(twitter[0])


if __name__ == "__main__":
    main()
