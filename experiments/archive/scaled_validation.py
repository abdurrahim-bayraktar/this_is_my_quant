"""
Scaled-Up Validation: News-Rich Stocks

Selects stocks with the most news coverage from FNSPID (max 200),
then runs full training with proper temporal alignment.

Usage:
    python experiments/scaled_validation.py --max-stocks 200 --min-news 50 --epochs 30
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import logging
from datetime import datetime
from collections import Counter

from config import (
    CACHE_DIR,
    MODELS_DIR,
    training_config,
    lstm_config,
)
from src.data import DataPreprocessor, FeatureEngineer
from src.nlp import SentimentExtractor
from src.models import BaselineLSTM, CombinedLoss
from src.training import Trainer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Configuration
FNSPID_PATH = Path(r"C:\Users\abdurrahim\.cache\huggingface\hub\datasets--Zihan1004--FNSPID\snapshots\bf9189c41527198897d1af3e17b1a0095279fc45\Stock_news\All_external.csv")


def analyze_news_coverage(csv_path: Path, chunk_size: int = 100000) -> pd.DataFrame:
    """
    Analyze FNSPID to find stocks with most news coverage.
    
    Returns DataFrame with ticker, news_count, date_range.
    """
    logger.info("Analyzing news coverage in FNSPID...")
    
    ticker_counts = Counter()
    ticker_dates = {}
    
    for chunk in tqdm(pd.read_csv(csv_path, chunksize=chunk_size, usecols=['Stock_symbol', 'Date']), 
                      desc="Scanning FNSPID"):
        for ticker in chunk['Stock_symbol'].dropna():
            ticker_counts[ticker] += 1
        
        # Track date range per ticker (sample)
        for _, row in chunk.head(1000).iterrows():
            ticker = row.get('Stock_symbol')
            date = row.get('Date')
            if pd.notna(ticker) and pd.notna(date):
                if ticker not in ticker_dates:
                    ticker_dates[ticker] = {'min': date, 'max': date}
    
    # Create summary DataFrame
    summary = pd.DataFrame([
        {'ticker': ticker, 'news_count': count}
        for ticker, count in ticker_counts.most_common()
    ])
    
    logger.info(f"Found {len(summary)} unique tickers")
    logger.info(f"Top 10 by coverage:\n{summary.head(10)}")
    
    return summary


def select_news_rich_stocks(
    summary_df: pd.DataFrame,
    max_stocks: int = 200,
    min_news: int = 50,
) -> list:
    """Select stocks with most news coverage."""
    
    # Filter by minimum news count
    filtered = summary_df[summary_df['news_count'] >= min_news]
    
    # Take top N by news count
    selected = filtered.head(max_stocks)['ticker'].tolist()
    
    logger.info(f"Selected {len(selected)} stocks with >= {min_news} news items")
    if selected:
        logger.info(f"News count range: {filtered.head(max_stocks)['news_count'].min()} - {filtered.head(max_stocks)['news_count'].max()}")
    
    return selected


def load_fnspid_for_tickers(
    csv_path: Path,
    tickers: list,
    chunk_size: int = 100000,
) -> pd.DataFrame:
    """Load FNSPID news for selected tickers."""
    logger.info(f"Loading FNSPID for {len(tickers)} stocks...")
    ticker_set = set(tickers)
    
    chunks = []
    for chunk in tqdm(pd.read_csv(csv_path, chunksize=chunk_size), desc="Loading news"):
        if 'Stock_symbol' in chunk.columns:
            mask = chunk['Stock_symbol'].isin(ticker_set)
            filtered = chunk[mask].copy()
            
            if len(filtered) > 0:
                filtered = filtered.rename(columns={
                    'Stock_symbol': 'ticker',
                    'Article_title': 'headline',
                    'Date': 'date',
                })
                filtered['date'] = pd.to_datetime(filtered['date'], utc=True, errors='coerce')
                filtered['date'] = filtered['date'].dt.date
                chunks.append(filtered[['date', 'ticker', 'headline']])
    
    if not chunks:
        raise ValueError("No news data found for selected tickers")
    
    news_df = pd.concat(chunks, ignore_index=True)
    news_df = news_df.dropna(subset=['date', 'headline'])
    
    logger.info(f"Loaded {len(news_df):,} news records")
    logger.info(f"Date range: {news_df['date'].min()} to {news_df['date'].max()}")
    
    return news_df


def load_prices(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Load price data from yfinance."""
    import yfinance as yf
    
    logger.info(f"Downloading prices for {len(tickers)} stocks...")
    
    all_prices = []
    failed = []
    
    for ticker in tqdm(tickers, desc="Downloading prices"):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            if not hist.empty:
                hist = hist.reset_index()
                hist['ticker'] = ticker
                hist.columns = [c.lower().replace(' ', '_') for c in hist.columns]
                all_prices.append(hist)
            else:
                failed.append(ticker)
        except Exception as e:
            failed.append(ticker)
    
    if failed:
        logger.warning(f"Failed to download {len(failed)} tickers: {failed[:10]}...")
    
    if not all_prices:
        raise ValueError("No price data downloaded")
    
    prices_df = pd.concat(all_prices, ignore_index=True)
    prices_df['date'] = pd.to_datetime(prices_df['date']).dt.date
    
    logger.info(f"Loaded {len(prices_df):,} price records for {prices_df['ticker'].nunique()} stocks")
    return prices_df


def extract_sentiment(news_df: pd.DataFrame, cache_key: str = None) -> pd.DataFrame:
    """
    Extract sentiment using FinBERT on GPU.
    
    Caches results to parquet for reuse (important for real-time processing).
    """
    # Check for cached sentiment
    if cache_key:
        cache_path = CACHE_DIR / f"sentiment_cache_{cache_key}.parquet"
        if cache_path.exists():
            logger.info(f"Loading cached sentiment from {cache_path}")
            cached_df = pd.read_parquet(cache_path)
            # Merge with news_df on headline
            news_df = news_df.merge(
                cached_df[['headline', 'sentiment_value', 'sentiment_positive', 'sentiment_negative']],
                on='headline',
                how='left'
            )
            uncached_mask = news_df['sentiment_value'].isna()
            if uncached_mask.sum() == 0:
                logger.info("All headlines found in cache")
                return news_df
            logger.info(f"Found {(~uncached_mask).sum()} cached, {uncached_mask.sum()} need processing")
    
    logger.info(f"Extracting sentiment for {len(news_df):,} headlines...")
    
    extractor = SentimentExtractor()
    headlines = news_df['headline'].fillna('').tolist()
    results = extractor.batch_extract(headlines, show_progress=True)
    
    news_df = news_df.copy()
    news_df['sentiment_value'] = [r.positive - r.negative for r in results]
    news_df['sentiment_positive'] = [r.positive for r in results]
    news_df['sentiment_negative'] = [r.negative for r in results]
    news_df['sentiment_label'] = [r.label for r in results]
    
    # Save to cache for future runs
    if cache_key:
        cache_path = CACHE_DIR / f"sentiment_cache_{cache_key}.parquet"
        cache_cols = ['headline', 'ticker', 'date', 'sentiment_value', 
                      'sentiment_positive', 'sentiment_negative', 'sentiment_label']
        news_df[cache_cols].to_parquet(cache_path, index=False)
        logger.info(f"Saved sentiment cache to {cache_path} ({len(news_df):,} records)")
    
    return news_df


def aggregate_daily_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sentiment per ticker + date."""
    logger.info("Aggregating daily sentiment...")
    
    agg = news_df.groupby(['ticker', 'date']).agg({
        'sentiment_value': ['mean', 'std', 'count']
    }).reset_index()
    
    agg.columns = ['ticker', 'date', 'sentiment_mean', 'sentiment_std', 'news_count']
    agg['sentiment_std'] = agg['sentiment_std'].fillna(0)
    agg['has_news'] = 1
    
    logger.info(f"Created {len(agg):,} daily sentiment records")
    
    # Print variance stats
    for ticker in agg['ticker'].unique()[:5]:
        ticker_data = agg[agg['ticker'] == ticker]['sentiment_mean']
        logger.info(f"  {ticker}: variance={ticker_data.var():.4f}, records={len(ticker_data)}")
    
    return agg


def prepare_features(
    prices_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
) -> tuple:
    """Prepare features for training."""
    
    engineer = FeatureEngineer()
    preprocessor = DataPreprocessor()
    
    # Add technical indicators
    prices_df = engineer.add_technical_indicators(prices_df)
    prices_df = preprocessor.compute_returns(prices_df)
    prices_df = preprocessor.label_trends(prices_df)
    
    # Merge with sentiment
    prices_df = prices_df.merge(sentiment_df, on=['ticker', 'date'], how='left')
    
    # Fill missing sentiment
    prices_df['sentiment_mean'] = prices_df['sentiment_mean'].fillna(0)
    prices_df['sentiment_std'] = prices_df['sentiment_std'].fillna(0)
    prices_df['news_count'] = prices_df['news_count'].fillna(0)
    prices_df['has_news'] = prices_df['has_news'].fillna(0).astype(int)
    
    # Get features
    feature_cols = engineer.get_feature_columns()
    feature_cols = [c for c in feature_cols if c in prices_df.columns]
    
    logger.info(f"Using {len(feature_cols)} features")
    
    # Normalize
    prices_df = preprocessor.normalize_features(prices_df, feature_cols, fit=True)
    
    # Create sequences
    X, y, _ = preprocessor.create_sequences(prices_df, feature_cols=feature_cols, target_col='trend')
    
    return X.astype(np.float32), y, feature_cols


def train_and_evaluate(X, y, epochs):
    """Train with full model complexity."""
    
    n = len(X)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Full model (2 layers, as configured)
    model = BaselineLSTM(
        input_size=X.shape[-1],
        hidden_size=lstm_config.hidden_size,
        num_layers=lstm_config.num_layers,
        dropout=lstm_config.dropout,
    )
    
    logger.info(f"Model parameters: {model.get_num_parameters():,}")
    
    loss_fn = CombinedLoss(trend_weight=1.0, confidence_weight=0.5)
    trainer = Trainer(model, loss_fn)
    
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=training_config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
        batch_size=training_config.batch_size,
    )
    
    trainer.train(train_loader, val_loader, epochs=epochs)
    
    # Evaluate
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        predictions = model.predict(X_test_tensor)
    
    pred_classes = predictions['trend_class'].cpu().numpy()
    confidences = predictions['confidence'].cpu().numpy()
    
    accuracy = (pred_classes == y_test).mean()
    high_conf_mask = confidences > 0.5
    high_conf_acc = (pred_classes[high_conf_mask] == y_test[high_conf_mask]).mean() if high_conf_mask.sum() > 0 else 0
    
    return {
        'accuracy': accuracy,
        'high_confidence_accuracy': high_conf_acc,
        'high_confidence_ratio': high_conf_mask.mean(),
        'test_samples': len(X_test),
    }


def main():
    parser = argparse.ArgumentParser(description="Scaled validation with news-rich stocks")
    parser.add_argument("--max-stocks", type=int, default=200, help="Maximum stocks to use")
    parser.add_argument("--min-news", type=int, default=50, help="Minimum news items per stock")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip analysis, use cached tickers")
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("SCALED VALIDATION: News-Rich Stocks")
    logger.info("=" * 70)
    logger.info(f"Max stocks: {args.max_stocks}, Min news: {args.min_news}, Epochs: {args.epochs}")
    
    # Step 1: Analyze news coverage
    cache_path = CACHE_DIR / "fnspid_ticker_coverage.csv"
    
    if args.skip_analysis and cache_path.exists():
        logger.info(f"Loading cached coverage analysis from {cache_path}")
        summary_df = pd.read_csv(cache_path)
    else:
        summary_df = analyze_news_coverage(FNSPID_PATH)
        summary_df.to_csv(cache_path, index=False)
        logger.info(f"Saved coverage analysis to {cache_path}")
    
    # Step 2: Select news-rich stocks
    selected_tickers = select_news_rich_stocks(summary_df, args.max_stocks, args.min_news)
    
    if not selected_tickers:
        logger.error("No stocks meet the criteria!")
        return
    
    # Step 3: Load news for selected tickers
    news_df = load_fnspid_for_tickers(FNSPID_PATH, selected_tickers)
    
    # Step 4: Load prices
    start_date = str(news_df['date'].min())
    end_date = str(news_df['date'].max())
    prices_df = load_prices(selected_tickers, start_date, end_date)
    
    # Step 5: Extract sentiment (with caching for real-time processing reuse)
    cache_key = f"fnspid_{len(selected_tickers)}stocks"
    news_df = extract_sentiment(news_df, cache_key=cache_key)
    
    # Step 6: Aggregate daily
    sentiment_df = aggregate_daily_sentiment(news_df)
    
    # Step 7: Prepare features
    X, y, feature_cols = prepare_features(prices_df, sentiment_df)
    logger.info(f"Prepared data: X.shape={X.shape}")
    
    # Step 8: Train
    results = train_and_evaluate(X, y, args.epochs)
    
    # Print results
    print("\n" + "=" * 70)
    print("SCALED VALIDATION RESULTS")
    print("=" * 70)
    print(f"Stocks used:              {len(selected_tickers)}")
    print(f"Test samples:             {results['test_samples']}")
    print(f"Accuracy:                 {results['accuracy']:.2%}")
    print(f"High Confidence Accuracy: {results['high_confidence_accuracy']:.2%}")
    print(f"High Confidence Ratio:    {results['high_confidence_ratio']:.2%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
