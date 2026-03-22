
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

# Add src to path
sys.path.insert(0, str(Path.cwd()))

from config import CACHE_DIR
from src.data import DatasetLoader, DataPreprocessor
from src.nlp.sentiment_extractor import SentimentExtractor
from src.nlp.aggregator import SentimentAggregator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_top_40_tickers():
    """Read top 40 tickers from coverage cache."""
    coverage_file = CACHE_DIR / "fnspid_ticker_coverage.csv"
    if not coverage_file.exists():
        logger.error(f"Coverage file not found at {coverage_file}")
        return []
    
    df = pd.read_csv(coverage_file)
    # Sort by news_count desc just in case
    df = df.sort_values("news_count", ascending=False)
    return df['ticker'].head(40).tolist()

def ensure_sentiment_data(ticker, extractor):
    """
    Ensure we have sentiment data for the ticker.
    If cached sentiment exists, load it.
    If only news exists or nothing exists, load news and extract sentiment.
    Returns: sentiment_df (with 'sentiment_value', 'sentiment_label')
    """
    base_dir = CACHE_DIR / "sentiment"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    sentiment_file = base_dir / f"{ticker}_sentiment.parquet"
    news_file = base_dir / f"{ticker}_news.parquet"
    
    # 1. Try to load existing sentiment
    if sentiment_file.exists():
        logger.info(f"[{ticker}] Loading cached sentiment...")
        return pd.read_parquet(sentiment_file)
        
    # 2. If no sentiment, we need news text
    news_df = None
    if news_file.exists():
        logger.info(f"[{ticker}] Loading cached news text...")
        news_df = pd.read_parquet(news_file)
    else:
        # If batch loading worked, this file should exist.
        # If it doesn't exist, it means either:
        # a) The batch loader failed (we already logged that)
        # b) The ticker has NO news in the FNSPID dataset
        # In either case, we should NOT fall back to scanning the 23GB CSV again for this single ticker.
        logger.warning(f"[{ticker}] No cached news found. Batch loader may have skipped it or no data exists.")
        return None
            
    if news_df is None or news_df.empty:
        logger.warning(f"[{ticker}] No news data found.")
        return None
        
    # 3. Extract Sentiment
    logger.info(f"[{ticker}] Extracting sentiment for {len(news_df)} items...")
    
    # Ensure headline/text column
    text_col = "headline"
    if "headline" not in news_df.columns and "title" in news_df.columns:
        text_col = "title"
    elif "article_title" in news_df.columns:
        text_col = "article_title"
        
    if text_col not in news_df.columns:
        logger.warning(f"[{ticker}] No text column found in news data. Cols: {news_df.columns}")
        return None
        
    # Run FinBERT
    sentiment_df = extractor.extract_dataframe(news_df, text_col=text_col)
    
    # Save for future
    sentiment_df.to_parquet(sentiment_file)
    logger.info(f"[{ticker}] Saved generated sentiment to {sentiment_file}")
    
    return sentiment_df

def run_experiment():
    # User requested specific tickers: MSFT, DIS, WMT
    tickers = ["MSFT", "DIS", "WMT"]
    logger.info(f"Target Tickers: {tickers}")
    
    if not tickers:
        return
        
    # Initialize tools
    preprocessor = DataPreprocessor()
    aggregator = SentimentAggregator(strategy="simple")
    
    # Initialize Extractor (once)
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    extractor = SentimentExtractor(device=device, batch_size=32)
    
    
    # User requested to skip fresh extraction and rely on cached sentiment files.
    # We skip the batch loading block entirely.
    
    results = []
    
    for ticker in tqdm(tickers, desc="Processing Tickers"):
        logger.info(f"--- Processing {ticker} ---")
        
        # 1. Get Sentiment Data (now should be cached or we fail)
        sentiment_df = ensure_sentiment_data(ticker, extractor)
        if sentiment_df is None or sentiment_df.empty:
            continue
            
        # 2. Filter Neutrals (User Request)
        # "filter those out so they don't skew the results"
        # We filter based on label 'neutral'
        if "sentiment_label" in sentiment_df.columns:
            original_len = len(sentiment_df)
            sentiment_df = sentiment_df[sentiment_df["sentiment_label"] != "neutral"]
            full_len = len(sentiment_df)
            logger.info(f"[{ticker}] Filtered neutrals: {original_len} -> {full_len}")
            
            if full_len < 10:
                logger.warning(f"[{ticker}] Too few records after filtering neutrals.")
                continue
        else:
            logger.warning(f"[{ticker}] sentiment_label column missing, cannot filter neutrals.")
            
        # 3. Get Price Data
        # We MUST disable cache here because DatasetLoader names files by count (prices_1stocks_...)
        # causing collisions (loading WMT prices for LEN).
        loader = DatasetLoader(tickers=[ticker], use_cache=False)
        try:
             # DIRECTLY load prices to avoid triggering the FNSPID news loading (which scans the 23GB CSV)
             prices_df = loader._load_prices_yfinance()
        except Exception as e:
            logger.warning(f"[{ticker}] Failed to load prices: {e}")
            continue
            
        if prices_df.empty:
            logger.warning(f"[{ticker}] No price data.")
            continue
            
        # 4. Process and Align
        # Calculate returns
        prices_df = preprocessor.compute_returns(prices_df)
        prices_df = preprocessor.label_trends(prices_df)
        
        # Align news
        aligned_news = preprocessor.align_news_to_trading_day(sentiment_df, prices_df)
        
        if aligned_news.empty:
            logger.warning(f"[{ticker}] No aligned news.")
            continue
            
        # 5. Aggregate Daily
        daily_sentiment = aggregator.aggregate_daily(aligned_news)
        
        # 6. Merge
        # Ensure dates are compatible
        prices_df["date"] = pd.to_datetime(prices_df["date"]).dt.tz_localize(None).dt.normalize()
        daily_sentiment["trading_date"] = pd.to_datetime(daily_sentiment["trading_date"]).dt.tz_localize(None).dt.normalize()
        
        combined = pd.merge(
            prices_df,
            daily_sentiment,
            left_on=["ticker", "date"],
            right_on=["ticker", "trading_date"],
            how="inner" # using inner to analyze only days WITH sentiment
        )
        
        if len(combined) < 10:
            logger.warning(f"[{ticker}] Not enough overlapping data points ({len(combined)}).")
            continue
            
        # 7. Calculate Correlation
        # Correlation between sentiment_mean and next day return? 
        # Or same day return? 
        # Usually sentiment today affects tomorrow, or today if during market.
        # But align_news_to_trading_day handles the shift (post-market goes to next day).
        # So 'date' in combined is the trading day the news affected.
        # So we correlate 'sentiment_mean' with 'return' (which is Close_t / Close_{t-1} - 1).
        # If news came before close, it affects 'return'.
        
        # Calculate Next Day Return (for prediction)
        # return is R_t = (P_t - P_{t-1})/P_{t-1}
        # next_return at t should be R_{t+1}
        combined["next_return"] = combined["return"].shift(-1)
        
        # Drop last row which will be NaN
        combined_clean = combined.dropna(subset=["next_return", "sentiment_mean"])
        
        if len(combined_clean) < 10:
             logger.warning(f"[{ticker}] Not enough data for next-day correlation.")
             continue

        # Correlations
        corr_curr = combined["sentiment_mean"].corr(combined["return"])
        corr_next = combined_clean["sentiment_mean"].corr(combined_clean["next_return"])
        
        logger.info(f"[{ticker}] Same-Day Corr: {corr_curr:.4f} | FDA (Next-Day) Corr: {corr_next:.4f}")
        
        # --- Generate Scatter Plot (Same Day) ---
        try:
            from config import REPORTS_DIR
            plot_dir = REPORTS_DIR / "top40_scatters"
            plot_dir.mkdir(parents=True, exist_ok=True)
            
            plt.figure(figsize=(10, 6))
            plt.scatter(combined['sentiment_mean'], combined['return'], alpha=0.5)
            
            # Add trend line
            z = np.polyfit(combined['sentiment_mean'], combined['return'], 1)
            p = np.poly1d(z)
            plt.plot(combined['sentiment_mean'], p(combined['sentiment_mean']), "r--", alpha=0.8)
            
            plt.xlabel('Daily Sentiment Score')
            plt.ylabel('Daily Return')
            plt.title(f'{ticker} Sentiment vs Return (Corr: {corr_curr:.3f})')
            plt.grid(True, alpha=0.3)
            plt.axhline(0, color='black', alpha=0.3)
            plt.axvline(0, color='black', alpha=0.3)
            
            save_path = plot_dir / f"{ticker}_scatter.png"
            plt.savefig(save_path)
            plt.close()
        except Exception as e:
            logger.warning(f"Failed to plot same-day {ticker}: {e}")

        # --- Generate Scatter Plot (Next Day) ---
        # User is likely interested in prediction, so let's plot Next Day Return
        try:
            from config import REPORTS_DIR
            plot_dir = REPORTS_DIR / "top40_scatters_next_day"
            plot_dir.mkdir(parents=True, exist_ok=True)
            
            plt.figure(figsize=(10, 6))
            plt.scatter(combined_clean['sentiment_mean'], combined_clean['next_return'], alpha=0.5)
            
            # Add trend line
            z = np.polyfit(combined_clean['sentiment_mean'], combined_clean['next_return'], 1)
            p = np.poly1d(z)
            plt.plot(combined_clean['sentiment_mean'], p(combined_clean['sentiment_mean']), "r--", alpha=0.8)
            
            plt.xlabel('Daily Sentiment Score (t)')
            plt.ylabel('Next Day Return (t+1)')
            plt.title(f'{ticker} Sentiment vs Next Day Return (Corr: {corr_next:.3f})')
            plt.grid(True, alpha=0.3)
            plt.axhline(0, color='black', alpha=0.3)
            plt.axvline(0, color='black', alpha=0.3)
            
            save_path = plot_dir / f"{ticker}_next_day_scatter.png"
            plt.savefig(save_path)
            plt.close()
        except Exception as e:
            logger.warning(f"Failed to plot {ticker}: {e}")
        
        results.append({
            "ticker": ticker,
            "corr_same_day": corr_curr,
            "corr_next_day": corr_next,
            "observations": len(combined_clean),
            "news_count": combined["news_count"].sum()
        })
        
    # Summary
    if not results:
        logger.info("No results generated.")
        return
        
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("corr_next_day", ascending=False)
    
    print("\n=== Experiment Results (Top 40 fnspid) ===")
    print(results_df[["ticker", "corr_same_day", "corr_next_day", "observations", "news_count"]])
    print("\nSummary Stats:")
    print(results_df[["corr_same_day", "corr_next_day"]].describe())
    
    # Save
    out_file = CACHE_DIR / "top40_correlation_results.csv"
    results_df.to_csv(out_file, index=False)
    print(f"\nSaved results to {out_file}")

if __name__ == "__main__":
    run_experiment()
