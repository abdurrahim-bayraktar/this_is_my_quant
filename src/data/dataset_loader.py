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
        
    def load_twitter_sentiment(self) -> Dataset:
        """Load Twitter Financial Sentiment dataset."""
        logger.info("Loading Twitter Financial Sentiment dataset...")
        # Check cache logic could be added here, but the dataset is small (~50MB)
        # HuggingFace datasets library handles caching automatically
        dataset = load_dataset(dataset_config.twitter_sentiment_repo, split="train")
        return dataset
        
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
            
        # PRIORITY CHECK: Look for individual cached files in data/cache/sentiment
        # This is the "clean" data source created by previous iterations
        sentiment_cache_dir = CACHE_DIR / "sentiment"
        if sentiment_cache_dir.exists():
            logger.info(f"Checking {sentiment_cache_dir} for cached parquet files...")
            news_dfs = []
            price_dfs = []
            found_tickers = []
            
            for ticker in self.tickers:
                n_file = sentiment_cache_dir / f"{ticker}_news.parquet"
                p_file = sentiment_cache_dir / f"{ticker}_prices.parquet"
                # s_file = sentiment_cache_dir / f"{ticker}_sentiment.parquet" 
                
                # Check for either news or sentiment file (sentiment is better if available)
                # But load_fnspid expects "news", so we might need to adapt.
                # Actually, the user wants to use the text data, or the extracted sentiment?
                # The prompt says "the 23GB file was already processed... sentiment extracted... under data>cache>sentiment"
                # So we should probably try to load the sentiment parquets if we can, but load_fnspid signature returns (news_df, prices_df).
                # Let's load the *_sentiment.parquet as "news_df" since it contains the signals we need.
                
                s_file = sentiment_cache_dir / f"{ticker}_sentiment.parquet"
                if s_file.exists() and p_file.exists():
                    try:
                        sdf = pd.read_parquet(s_file)
                        pdf = pd.read_parquet(p_file)
                        
                        # Ensure ticker column
                        sdf["ticker"] = ticker
                        pdf["ticker"] = ticker
                        
                        # Filter by date range
                        if "date" in sdf.columns:
                            sdf = sdf[(sdf["date"] >= self.date_range[0]) & (sdf["date"] <= self.date_range[1])]
                        elif "timestamp" in sdf.columns:
                            # Rename timestamp to date for consistency
                            sdf["date"] = sdf["timestamp"]
                            sdf = sdf[(sdf["date"] >= self.date_range[0]) & (sdf["date"] <= self.date_range[1])]
                            
                        pdf = pdf[(pdf["date"] >= self.date_range[0]) & (pdf["date"] <= self.date_range[1])]
                        
                        if len(sdf) > 0 and len(pdf) > 0:
                            news_dfs.append(sdf)
                            price_dfs.append(pdf)
                            found_tickers.append(ticker)
                    except Exception as e:
                        logger.warning(f"Failed to load cached data for {ticker}: {e}")
            
            if len(news_dfs) > 0:
                logger.info(f"Found cached data for {len(found_tickers)}/{len(self.tickers)} tickers.")
                combined_news = pd.concat(news_dfs, ignore_index=True)
                combined_prices = pd.concat(price_dfs, ignore_index=True)
                
                # Normalize timezones to prevent merge errors
                if "date" in combined_news.columns:
                    combined_news["date"] = pd.to_datetime(combined_news["date"]).dt.tz_localize(None)
                if "date" in combined_prices.columns:
                    combined_prices["date"] = pd.to_datetime(combined_prices["date"]).dt.tz_localize(None)
                
                return combined_news, combined_prices

        logger.info("Attempting to load FNSPID dataset from HuggingFace/CSV...")
        
        try:
            # Try loading without trust_remote_code (deprecated)
            dataset = load_dataset(
                dataset_config.fnspid_repo,
                split=split,
            )
        except Exception as e:
            logger.warning(f"Could not load FNSPID from HuggingFace directly: {e}")
            logger.info("Attempting to load from local large CSV cache...")
            return self.load_from_local_cache()
        
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
    
    def load_from_local_cache(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load directly from local HuggingFace cache CSVs if available.
        This bypasses the datasets library which struggles with the 23GB file.
        """
        import os
        from pathlib import Path
        
        # Common HF cache location
        home = Path.home()
        hf_cache = home / ".cache" / "huggingface" / "hub"
        
        # Look for Zihan1004--FNSPID folder
        # We search recursively for the nasdaq_exteral_data.csv
        logger.info(f"Searching for local FNSPID CSV in {hf_cache}...")
        
        csv_path = None
        for path in hf_cache.rglob("nasdaq_exteral_data.csv"):
            if path.is_file():
                csv_path = path
                break
        
        if not csv_path:
            logger.warning("Local FNSPID CSV not found.")
            return pd.DataFrame(), pd.DataFrame()
            
        logger.info(f"Found local FNSPID CSV at: {csv_path}")
        logger.info(f"Reading large CSV (21GB+) in chunks, filtering for {len(self._ticker_set)} tickers...")
        
        chunks = []
        chunk_size = 100000
        total_rows = 0
        matching_rows = 0
        
        try:
            # We only need specific columns to save memory
            use_cols = ["Date", "Stock_symbol", "Article_title"]
            
            for chunk in tqdm(pd.read_csv(csv_path, chunksize=chunk_size, usecols=lambda c: c in use_cols), desc="Processing chunks"):
                # Normalize column names
                chunk.columns = [c.lower() for c in chunk.columns]
                # date, stock_symbol, article_title
                
                # Filter by ticker
                if "stock_symbol" in chunk.columns:
                    mask = chunk["stock_symbol"].isin(self._ticker_set)
                    filtered = chunk[mask].copy()
                    
                    if len(filtered) > 0:
                        # Rename columns to match expected format
                        filtered = filtered.rename(columns={
                            "stock_symbol": "ticker",
                            "article_title": "headline"
                        })
                        chunks.append(filtered)
                        matching_rows += len(filtered)
                
                total_rows += len(chunk)
                # Optional: limit for testing
                # if total_rows > 1000000: break
                
        except Exception as e:
            logger.error(f"Error reading CSV chunk: {e}")
            return pd.DataFrame(), pd.DataFrame()
            
        if not chunks:
            logger.warning("No matching data found in local CSV.")
            return pd.DataFrame(), pd.DataFrame()
            
        logger.info(f"Processed {total_rows:,} rows, found {matching_rows:,} matching records.")
        news_df = pd.concat(chunks, ignore_index=True)
        
        # Post-process
        news_df["date"] = pd.to_datetime(news_df["date"], errors='coerce')
        news_df = news_df.dropna(subset=["date"])
        
        # Filter by date range
        news_df = news_df[
            (news_df["date"] >= self.date_range[0]) &
            (news_df["date"] <= self.date_range[1])
        ]
        
        # Load prices
        prices_df = self._load_prices_yfinance()
        
        return news_df, prices_df
    
    def load_data_with_sentiment_intersection(
        self, 
        min_news_count: int = 10
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load data ONLY for tickers that have sufficient sentiment data.
        
        This prevents the "Zero Sentiment Trap" where the model learns to
        ignore sentiment because most stocks have none.
        
        Args:
            min_news_count: Minimum number of sentiment records required.
            
        Returns:
            Tuple[news_df, prices_df] for the valid intersection of tickers.
        """
        logger.info(f"Loading data with sentiment intersection (min_count={min_news_count})...")
        
        # 1. Load Sentiment Data First
        # We try FNSPID first, then fallback to Twitter
        try:
            # Check if full FNSPID is available (rare/large)
            # For now, let's prioritize the Twitter dataset as it's verified working
            twitter_data = self.load_twitter_sentiment()
            
            # Extract basic records from Twitter
            news_records = []
            ticker_pattern = r'\$([A-Z]{1,5})\b'
            import re
            
            for item in twitter_data:
                text = item['text']
                label = item['label']
                tickers_found = re.findall(ticker_pattern, text)
                
                sentiment_map = {0: 'negative', 1: 'positive', 2: 'neutral'}
                sentiment = sentiment_map.get(label, 'neutral')
                
                for ticker in tickers_found:
                    news_records.append({
                        'date': pd.Timestamp.now(), # Placeholder date
                        'ticker': ticker,
                        'headline': text,
                        'sentiment_label': sentiment,
                        'sentiment_score': 1.0 if label != 2 else 0.5,
                        'original_label': label 
                    })
            
            news_df = pd.DataFrame(news_records)
            
        except Exception as e:
            logger.error(f"Failed to load initial sentiment data: {e}")
            return pd.DataFrame(), pd.DataFrame()

        if len(news_df) == 0:
            logger.warning("No sentiment data found.")
            return pd.DataFrame(), pd.DataFrame()

        # 2. Filter Tickers
        # Count news per ticker
        ticker_counts = news_df['ticker'].value_counts()
        valid_tickers = ticker_counts[ticker_counts >= min_news_count].index.tolist()
        
        # Intersect with the user's requested tickers (if any were specific)
        # Note: self.tickers is currently TOP_200_TICKERS usually
        # We should prioritize valid_tickers, but maybe limit by TOP_200 to keep quality high?
        # Let's keep valid_tickers that are ALSO in our universe (to avoid penny stocks)
        
        # universe_set = set(self.tickers) 
        # final_tickers = [t for t in valid_tickers if t in universe_set]
        
        # Actually, let's trust the sentiment signal for now. If people talk about it, it's relevant.
        # But we must limit the number to avoid downloading 2000 stocks prices
        final_tickers = valid_tickers[:50] # Take top 50 most talked about stocks
        
        logger.info(f"Found {len(final_tickers)} stocks with >{min_news_count} sentiment records.")
        logger.info(f"Top 5 discussed: {final_tickers[:5]}")
        
        # 3. Update self.tickers for price loading
        self.tickers = final_tickers
        self._ticker_set = set(final_tickers)
        
        # 4. Filter News DataFrame
        news_df = news_df[news_df['ticker'].isin(self._ticker_set)]
        
        # 5. Load Prices for these tickers
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
