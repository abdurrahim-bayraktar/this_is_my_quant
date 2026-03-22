"""
Signal Quality Analysis: Correlation between sentiment and returns.

Tests if sentiment features have ANY predictive relationship with next-day returns.
If correlation is ~0, sentiment has no signal.

Usage:
    python experiments/signal_quality_test.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm
import logging
import json
from datetime import datetime

from config import REPORTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Configuration
TEST_STOCKS = ["TSLA", "AAPL", "NVDA", "AMD", "AMZN", "MSFT", "GOOGL", "META"]
FNSPID_PATH = Path(r"C:\Users\abdurrahim\.cache\huggingface\hub\datasets--Zihan1004--FNSPID\snapshots\bf9189c41527198897d1af3e17b1a0095279fc45\Stock_news\All_external.csv")


def load_fnspid_sample(csv_path: Path, tickers: list, max_rows_per_ticker: int = 5000) -> pd.DataFrame:
    """Load sample of FNSPID data for analysis."""
    logger.info(f"Loading FNSPID sample for {len(tickers)} stocks...")
    ticker_set = set(tickers)
    chunks = []
    
    for chunk in tqdm(pd.read_csv(csv_path, chunksize=50000), desc="Reading FNSPID"):
        if 'Stock_symbol' in chunk.columns:
            mask = chunk['Stock_symbol'].isin(ticker_set)
            filtered = chunk[mask].copy()
            if len(filtered) > 0:
                filtered = filtered.rename(columns={
                    'Stock_symbol': 'ticker',
                    'Article_title': 'headline',
                    'Date': 'date',
                })
                chunks.append(filtered[['date', 'ticker', 'headline']])
        
        # Stop early if we have enough
        if chunks and sum(len(c) for c in chunks) > max_rows_per_ticker * len(tickers):
            break
    
    news_df = pd.concat(chunks, ignore_index=True)
    news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce')
    news_df = news_df.dropna(subset=['date', 'headline'])
    news_df['date'] = news_df['date'].dt.date
    
    logger.info(f"Loaded {len(news_df):,} news records")
    return news_df


def extract_sentiment_simple(news_df: pd.DataFrame) -> pd.DataFrame:
    """Extract sentiment using FinBERT."""
    from src.nlp import SentimentExtractor
    
    logger.info(f"Extracting sentiment for {len(news_df):,} headlines...")
    extractor = SentimentExtractor()
    
    headlines = news_df['headline'].fillna('').tolist()
    results = extractor.batch_extract(headlines, show_progress=True)
    
    news_df = news_df.copy()
    news_df['sentiment_value'] = [r.positive - r.negative for r in results]
    return news_df


def load_prices(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Load price data from yfinance."""
    import yfinance as yf
    
    all_prices = []
    for ticker in tqdm(tickers, desc="Downloading prices"):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            if not hist.empty:
                hist = hist.reset_index()
                hist['ticker'] = ticker
                hist.columns = [c.lower().replace(' ', '_') for c in hist.columns]
                all_prices.append(hist)
        except Exception as e:
            logger.warning(f"Failed to download {ticker}: {e}")
    
    prices_df = pd.concat(all_prices, ignore_index=True)
    prices_df['date'] = pd.to_datetime(prices_df['date']).dt.date
    prices_df['return'] = prices_df.groupby('ticker')['close'].pct_change()
    prices_df['next_return'] = prices_df.groupby('ticker')['return'].shift(-1)
    
    return prices_df


def analyze_correlations(news_df: pd.DataFrame, prices_df: pd.DataFrame) -> dict:
    """Analyze correlation between sentiment and next-day returns."""
    logger.info("Analyzing sentiment-return correlations...")
    
    # Aggregate daily sentiment
    daily_sentiment = news_df.groupby(['ticker', 'date']).agg({
        'sentiment_value': ['mean', 'std', 'count']
    }).reset_index()
    daily_sentiment.columns = ['ticker', 'date', 'sentiment_mean', 'sentiment_std', 'news_count']
    
    # Merge with prices
    merged = prices_df.merge(daily_sentiment, on=['ticker', 'date'], how='left')
    merged = merged.dropna(subset=['sentiment_mean', 'next_return'])
    
    results = {
        'overall': {},
        'by_ticker': {},
        'by_sentiment_strength': {},
    }
    
    # Overall correlation
    if len(merged) > 10:
        corr, pvalue = stats.pearsonr(merged['sentiment_mean'], merged['next_return'])
        results['overall'] = {
            'correlation': float(corr),
            'p_value': float(pvalue),
            'n_samples': int(len(merged)),
            'significant': bool(pvalue < 0.05),
        }
        logger.info(f"Overall: corr={corr:.4f}, p={pvalue:.4f}, n={len(merged)}")
    
    # Per-ticker correlation
    for ticker in merged['ticker'].unique():
        ticker_data = merged[merged['ticker'] == ticker]
        if len(ticker_data) > 10:
            corr, pvalue = stats.pearsonr(ticker_data['sentiment_mean'], ticker_data['next_return'])
            results['by_ticker'][ticker] = {
                'correlation': float(corr),
                'p_value': float(pvalue),
                'n_samples': int(len(ticker_data)),
                'significant': bool(pvalue < 0.05),
            }
            logger.info(f"  {ticker}: corr={corr:.4f}, p={pvalue:.4f}")
    
    # High sentiment days vs low sentiment days
    high_sentiment = merged[merged['sentiment_mean'] > 0.2]
    low_sentiment = merged[merged['sentiment_mean'] < -0.2]
    neutral_sentiment = merged[(merged['sentiment_mean'] >= -0.2) & (merged['sentiment_mean'] <= 0.2)]
    
    results['by_sentiment_strength'] = {
        'high_sentiment_mean_return': float(high_sentiment['next_return'].mean()) if len(high_sentiment) > 0 else None,
        'low_sentiment_mean_return': float(low_sentiment['next_return'].mean()) if len(low_sentiment) > 0 else None,
        'neutral_mean_return': float(neutral_sentiment['next_return'].mean()) if len(neutral_sentiment) > 0 else None,
        'high_count': int(len(high_sentiment)),
        'low_count': int(len(low_sentiment)),
        'neutral_count': int(len(neutral_sentiment)),
    }
    
    return results


def run_signal_quality_test():
    """Main test function."""
    print("=" * 60)
    print("SIGNAL QUALITY TEST: Sentiment-Return Correlation")
    print("=" * 60)
    
    # Load data
    news_df = load_fnspid_sample(FNSPID_PATH, TEST_STOCKS)
    
    # Get date range
    start_date = str(news_df['date'].min())
    end_date = str(news_df['date'].max())
    
    # Extract sentiment
    news_df = extract_sentiment_simple(news_df)
    
    # Load prices
    prices_df = load_prices(TEST_STOCKS, start_date, end_date)
    
    # Analyze
    results = analyze_correlations(news_df, prices_df)
    
    # Save results
    output_dir = REPORTS_DIR / f"signal_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SIGNAL QUALITY RESULTS")
    print("=" * 60)
    
    overall = results['overall']
    if overall:
        print(f"\nOverall Correlation: {overall['correlation']:.4f}")
        print(f"P-value: {overall['p_value']:.4f}")
        print(f"Samples: {overall['n_samples']}")
        print(f"Statistically Significant: {'YES' if overall['significant'] else 'NO'}")
    
    print("\nPer-Ticker Correlations:")
    for ticker, data in results['by_ticker'].items():
        sig = "[SIG]" if data['significant'] else ""
        print(f"  {ticker}: {data['correlation']:+.4f} (p={data['p_value']:.3f}) {sig}")
    
    strength = results['by_sentiment_strength']
    print("\nReturns by Sentiment Strength:")
    if strength['high_sentiment_mean_return'] is not None:
        print(f"  High Sentiment (>0.2):  {strength['high_sentiment_mean_return']:+.4%} (n={strength['high_count']})")
    if strength['low_sentiment_mean_return'] is not None:
        print(f"  Low Sentiment (<-0.2):  {strength['low_sentiment_mean_return']:+.4%} (n={strength['low_count']})")
    if strength['neutral_mean_return'] is not None:
        print(f"  Neutral Sentiment:      {strength['neutral_mean_return']:+.4%} (n={strength['neutral_count']})")
    
    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    
    if overall and abs(overall['correlation']) < 0.02:
        print("[FAIL] Correlation is essentially ZERO.")
        print("   Sentiment has NO predictive power for next-day returns.")
        print("   Consider: longer prediction horizon, different features, or ")
        print("   accepting that daily prediction is near-random for this data.")
    elif overall and overall['correlation'] > 0.05 and overall['significant']:
        print("[PASS] Weak positive correlation detected.")
        print("   Some signal exists, but may be too weak for reliable prediction.")
    elif overall and overall['correlation'] < -0.05 and overall['significant']:
        print("[WARN] Negative correlation detected.")
        print("   This is unusual - positive sentiment leads to LOWER returns?")
        print("   Check for lookahead bias or data issues.")
    else:
        print("[WARN] Results inconclusive. More data may be needed.")
    
    print("=" * 60)
    print(f"\nDetailed results saved to: {output_dir}")
    
    return results


if __name__ == "__main__":
    run_signal_quality_test()
