"""
Validation Test: Temporal Alignment Fix

This script validates that sentiment features vary across time (not constant).
Uses FNSPID local data with 5 highly-discussed stocks.

Usage:
    python experiments/validate_temporal_alignment.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import logging
from datetime import datetime

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
TEST_STOCKS = ["TSLA", "AAPL", "NVDA", "AMD", "AMZN"]
FNSPID_PATH = Path(r"C:\Users\abdurrahim\.cache\huggingface\hub\datasets--Zihan1004--FNSPID\snapshots\bf9189c41527198897d1af3e17b1a0095279fc45\Stock_news\All_external.csv")
EPOCHS = 15
BATCH_SIZE = 32


def load_fnspid_local(
    csv_path: Path,
    tickers: list,
    chunk_size: int = 100000,
) -> pd.DataFrame:
    """
    Load FNSPID news data from local CSV for specific tickers.
    
    Args:
        csv_path: Path to All_external.csv
        tickers: List of stock tickers to filter
        chunk_size: Rows per chunk for memory efficiency
        
    Returns:
        DataFrame with columns: date, ticker, headline
    """
    logger.info(f"Loading FNSPID from {csv_path} for {len(tickers)} stocks...")
    ticker_set = set(tickers)
    
    chunks = []
    total_rows = 0
    matching_rows = 0
    
    for chunk in tqdm(pd.read_csv(csv_path, chunksize=chunk_size), desc="Reading FNSPID"):
        # Filter by ticker
        if 'Stock_symbol' in chunk.columns:
            mask = chunk['Stock_symbol'].isin(ticker_set)
            filtered = chunk[mask].copy()
            
            if len(filtered) > 0:
                # Standardize columns
                filtered = filtered.rename(columns={
                    'Stock_symbol': 'ticker',
                    'Article_title': 'headline',
                    'Date': 'date',
                })
                
                # Parse dates
                filtered['date'] = pd.to_datetime(filtered['date'], utc=True, errors='coerce')
                filtered['date'] = filtered['date'].dt.date  # Extract date only
                
                chunks.append(filtered[['date', 'ticker', 'headline']])
                matching_rows += len(filtered)
        
        total_rows += len(chunk)
    
    if not chunks:
        raise ValueError("No matching data found in FNSPID CSV")
    
    news_df = pd.concat(chunks, ignore_index=True)
    news_df = news_df.dropna(subset=['date', 'headline'])
    
    logger.info(f"Loaded {len(news_df):,} news records from {total_rows:,} total rows")
    logger.info(f"Date range: {news_df['date'].min()} to {news_df['date'].max()}")
    
    return news_df


def load_prices(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Load price data from yfinance."""
    import yfinance as yf
    
    logger.info(f"Downloading prices for {tickers} from {start_date} to {end_date}")
    
    all_prices = []
    for ticker in tickers:
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
    
    logger.info(f"Loaded {len(prices_df):,} price records")
    return prices_df


def extract_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """Extract sentiment using FinBERT."""
    logger.info(f"Extracting sentiment for {len(news_df):,} headlines on GPU...")
    
    extractor = SentimentExtractor()
    
    # Process in batches
    headlines = news_df['headline'].fillna('').tolist()
    results = extractor.batch_extract(headlines, show_progress=True)
    
    news_df = news_df.copy()
    news_df['sentiment_value'] = [r.positive - r.negative for r in results]
    news_df['sentiment_label'] = [r.label for r in results]
    
    return news_df


def aggregate_daily_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentiment per ticker + date (THE FIX).
    
    This is the critical change from the broken per-ticker aggregation.
    """
    logger.info("Aggregating sentiment per ticker + date...")
    
    agg = news_df.groupby(['ticker', 'date']).agg({
        'sentiment_value': ['mean', 'std', 'count']
    }).reset_index()
    
    agg.columns = ['ticker', 'date', 'sentiment_mean', 'sentiment_std', 'news_count']
    agg['sentiment_std'] = agg['sentiment_std'].fillna(0)
    agg['has_news'] = 1  # All rows here have news
    
    logger.info(f"Created {len(agg):,} daily sentiment records")
    
    # Check temporal variance
    for ticker in agg['ticker'].unique():
        ticker_data = agg[agg['ticker'] == ticker]['sentiment_mean']
        variance = ticker_data.var()
        logger.info(f"  {ticker}: sentiment variance = {variance:.4f}, records = {len(ticker_data)}")
    
    return agg


def merge_and_prepare_features(
    prices_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    engineer: FeatureEngineer,
    preprocessor: DataPreprocessor,
) -> tuple:
    """Merge prices with daily sentiment and prepare features."""
    
    # Add technical indicators
    prices_df = engineer.add_technical_indicators(prices_df)
    
    # Compute returns and labels
    prices_df = preprocessor.compute_returns(prices_df)
    prices_df = preprocessor.label_trends(prices_df)
    
    # Merge with sentiment
    prices_df = prices_df.merge(
        sentiment_df,
        on=['ticker', 'date'],
        how='left'
    )
    
    # Fill missing sentiment (no news that day)
    prices_df['sentiment_mean'] = prices_df['sentiment_mean'].fillna(0)
    prices_df['sentiment_std'] = prices_df['sentiment_std'].fillna(0)
    prices_df['news_count'] = prices_df['news_count'].fillna(0)
    prices_df['has_news'] = prices_df['has_news'].fillna(0).astype(int)
    
    # Get feature columns (ensure sentiment features are included)
    feature_cols = engineer.get_feature_columns()
    if 'has_news' not in feature_cols:
        feature_cols.append('has_news')
    feature_cols = [c for c in feature_cols if c in prices_df.columns]
    
    logger.info(f"Features: {feature_cols}")
    
    # Normalize
    prices_df = preprocessor.normalize_features(prices_df, feature_cols, fit=True)
    
    # Create sequences
    X, y, _ = preprocessor.create_sequences(
        prices_df,
        feature_cols=feature_cols,
        target_col='trend',
    )
    
    return X.astype(np.float32), y, feature_cols


def train_and_evaluate(X, y, feature_cols, epochs):
    """Train model and evaluate."""
    
    # Split (70/15/15)
    n = len(X)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Create model
    model = BaselineLSTM(
        input_size=X.shape[-1],
        hidden_size=lstm_config.hidden_size,
        num_layers=1,  # Simplified
        dropout=lstm_config.dropout,
    )
    
    loss_fn = CombinedLoss(trend_weight=1.0, confidence_weight=0.5)
    trainer = Trainer(model, loss_fn)
    
    # Data loaders
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
        batch_size=BATCH_SIZE,
    )
    
    # Train
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
    }


def main():
    logger.info("=" * 60)
    logger.info("VALIDATION TEST: Temporal Alignment Fix")
    logger.info("=" * 60)
    logger.info(f"Test Stocks: {TEST_STOCKS}")
    logger.info(f"Epochs: {EPOCHS}")
    
    # Load FNSPID news
    news_df = load_fnspid_local(FNSPID_PATH, TEST_STOCKS)
    
    # Get date range from news
    start_date = str(news_df['date'].min())
    end_date = str(news_df['date'].max())
    
    # Load prices
    prices_df = load_prices(TEST_STOCKS, start_date, end_date)
    
    # Extract sentiment
    news_df = extract_sentiment(news_df)
    
    # Aggregate daily (THE FIX)
    sentiment_df = aggregate_daily_sentiment(news_df)
    
    # Prepare features
    engineer = FeatureEngineer()
    preprocessor = DataPreprocessor()
    
    X, y, feature_cols = merge_and_prepare_features(
        prices_df, sentiment_df, engineer, preprocessor
    )
    
    logger.info(f"Prepared data: X.shape={X.shape}, y.shape={y.shape}")
    
    # Train and evaluate
    results = train_and_evaluate(X, y, feature_cols, EPOCHS)
    
    # Print results
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:                 {results['accuracy']:.2%}")
    print(f"High Confidence Accuracy: {results['high_confidence_accuracy']:.2%}")
    print(f"High Confidence Ratio:    {results['high_confidence_ratio']:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
