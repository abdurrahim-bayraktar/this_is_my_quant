"""
Sentiment Decay Experiment

Tests the hypothesis that sentiment should decay gradually rather than
instantly dropping to zero when there's no news.

Implementation: Exponential Moving Average (EMA) of sentiment
- On news days: sentiment = new_sentiment
- On no-news days: sentiment = previous_sentiment * decay_factor

Usage:
    python experiments/sentiment_decay_test.py
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
import json

from config import CACHE_DIR, REPORTS_DIR, lstm_config
from src.data import DataPreprocessor, FeatureEngineer
from src.nlp import SentimentExtractor
from src.models import BaselineLSTM, CombinedLoss
from src.training import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration
TEST_STOCKS = ["TSLA", "AAPL", "NVDA", "AMD", "AMZN", "MSFT", "GOOGL", "META"]
FNSPID_PATH = Path(r"C:\Users\abdurrahim\.cache\huggingface\hub\datasets--Zihan1004--FNSPID\snapshots\bf9189c41527198897d1af3e17b1a0095279fc45\Stock_news\All_external.csv")
EPOCHS = 15
BATCH_SIZE = 32
DECAY_FACTORS = [0.5, 0.7, 0.8, 0.9, 0.95]  # Test multiple decay rates


def load_fnspid_sample(csv_path: Path, tickers: list, max_chunks: int = 100) -> pd.DataFrame:
    """Load sample of FNSPID data."""
    logger.info(f"Loading FNSPID for {len(tickers)} stocks...")
    ticker_set = set(tickers)
    chunks = []
    
    for i, chunk in enumerate(tqdm(pd.read_csv(csv_path, chunksize=50000), desc="Reading FNSPID")):
        if i >= max_chunks:
            break
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
    
    news_df = pd.concat(chunks, ignore_index=True)
    news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce')
    news_df = news_df.dropna(subset=['date', 'headline'])
    news_df['date'] = news_df['date'].dt.date
    
    logger.info(f"Loaded {len(news_df):,} news records")
    return news_df


def extract_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """Extract sentiment using FinBERT."""
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
    return prices_df


def aggregate_daily_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sentiment per ticker + date."""
    agg = news_df.groupby(['ticker', 'date']).agg({
        'sentiment_value': ['mean', 'count']
    }).reset_index()
    agg.columns = ['ticker', 'date', 'sentiment_raw', 'news_count']
    return agg


def apply_sentiment_decay(prices_df: pd.DataFrame, sentiment_df: pd.DataFrame, decay_factor: float) -> pd.DataFrame:
    """
    Apply exponential sentiment decay.
    
    On news days: sentiment = new_sentiment
    On no-news days: sentiment = previous_sentiment * decay_factor
    
    This models the idea that sentiment "lingers" but fades over time.
    """
    logger.info(f"Applying sentiment decay with factor={decay_factor}")
    
    # Merge sentiment with prices
    merged = prices_df.merge(
        sentiment_df[['ticker', 'date', 'sentiment_raw', 'news_count']],
        on=['ticker', 'date'],
        how='left'
    )
    
    # Fill missing with NaN initially
    merged['sentiment_raw'] = merged['sentiment_raw'].fillna(np.nan)
    merged['news_count'] = merged['news_count'].fillna(0)
    merged['has_news'] = (merged['news_count'] > 0).astype(int)
    
    # Apply decay per ticker
    result_dfs = []
    for ticker in merged['ticker'].unique():
        ticker_df = merged[merged['ticker'] == ticker].copy()
        ticker_df = ticker_df.sort_values('date')
        
        decayed_sentiment = []
        current_sentiment = 0.0
        
        for _, row in ticker_df.iterrows():
            if not np.isnan(row['sentiment_raw']):
                # New news: use new sentiment (could also blend: alpha*new + (1-alpha)*old)
                current_sentiment = row['sentiment_raw']
            else:
                # No news: decay
                current_sentiment *= decay_factor
            decayed_sentiment.append(current_sentiment)
        
        ticker_df['sentiment_decayed'] = decayed_sentiment
        result_dfs.append(ticker_df)
    
    result = pd.concat(result_dfs, ignore_index=True)
    
    # Log statistics
    raw_variance = result['sentiment_raw'].dropna().var()
    decayed_variance = result['sentiment_decayed'].var()
    logger.info(f"  Raw sentiment variance: {raw_variance:.4f}")
    logger.info(f"  Decayed sentiment variance: {decayed_variance:.4f}")
    logger.info(f"  Non-zero days: {(result['sentiment_decayed'].abs() > 0.01).sum()}/{len(result)}")
    
    return result


def prepare_features(df: pd.DataFrame, use_decay: bool) -> tuple:
    """Prepare features for model training."""
    engineer = FeatureEngineer()
    preprocessor = DataPreprocessor()
    
    # Add technical indicators
    df = engineer.add_technical_indicators(df)
    
    # Compute returns and labels
    df = preprocessor.compute_returns(df)
    df = preprocessor.label_trends(df)
    
    # Choose sentiment column
    sentiment_col = 'sentiment_decayed' if use_decay else 'sentiment_raw'
    df['sentiment_mean'] = df[sentiment_col].fillna(0)
    
    # Get feature columns
    feature_cols = engineer.get_feature_columns()
    if 'has_news' not in feature_cols:
        feature_cols.append('has_news')
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    # Normalize
    df = preprocessor.normalize_features(df, feature_cols, fit=True)
    
    # Create sequences
    X, y, _ = preprocessor.create_sequences(df, feature_cols=feature_cols, target_col='trend')
    
    return X.astype(np.float32), y, feature_cols


def train_and_evaluate(X, y, label: str) -> dict:
    """Train model and evaluate."""
    # Split
    n = len(X)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    logger.info(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Create model
    model = BaselineLSTM(
        input_size=X.shape[-1],
        hidden_size=lstm_config.hidden_size,
        num_layers=1,
        dropout=lstm_config.dropout,
    )
    
    loss_fn = CombinedLoss(trend_weight=1.0, confidence_weight=0.5)
    trainer = Trainer(model, loss_fn)
    
    # Data loaders
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
        batch_size=BATCH_SIZE,
    )
    
    # Train
    trainer.train(train_loader, val_loader, epochs=EPOCHS)
    
    # Evaluate
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        predictions = model.predict(X_test_tensor)
    
    pred_classes = predictions['trend_class'].cpu().numpy()
    confidences = predictions['confidence'].cpu().numpy()
    
    accuracy = float((pred_classes == y_test).mean())
    high_conf_mask = confidences > 0.5
    high_conf_acc = float((pred_classes[high_conf_mask] == y_test[high_conf_mask]).mean()) if high_conf_mask.sum() > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'high_confidence_accuracy': high_conf_acc,
        'high_confidence_ratio': float(high_conf_mask.mean()),
    }


def main():
    print("=" * 70)
    print("SENTIMENT DECAY EXPERIMENT")
    print("=" * 70)
    print("Testing: Does sentiment carry-over improve predictions?")
    print("=" * 70)
    
    # Load data
    news_df = load_fnspid_sample(FNSPID_PATH, TEST_STOCKS)
    start_date = str(news_df['date'].min())
    end_date = str(news_df['date'].max())
    
    # Extract sentiment
    news_df = extract_sentiment(news_df)
    
    # Aggregate daily
    sentiment_df = aggregate_daily_sentiment(news_df)
    
    # Load prices
    prices_df = load_prices(TEST_STOCKS, start_date, end_date)
    
    # Results storage
    results = {}
    
    # Test 1: No decay (baseline - instant drop to zero)
    print("\n" + "=" * 60)
    print("Scenario: NO DECAY (Baseline)")
    print("=" * 60)
    
    df_baseline = apply_sentiment_decay(prices_df.copy(), sentiment_df, decay_factor=0.0)
    X, y, _ = prepare_features(df_baseline, use_decay=True)
    results['no_decay'] = train_and_evaluate(X, y, "No Decay")
    
    # Test with different decay factors
    for decay in DECAY_FACTORS:
        print("\n" + "=" * 60)
        print(f"Scenario: DECAY FACTOR = {decay}")
        print("=" * 60)
        
        df_decayed = apply_sentiment_decay(prices_df.copy(), sentiment_df, decay_factor=decay)
        X, y, _ = prepare_features(df_decayed, use_decay=True)
        results[f'decay_{decay}'] = train_and_evaluate(X, y, f"Decay {decay}")
    
    # Save results
    output_dir = REPORTS_DIR / f"sentiment_decay_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print comparison
    print("\n" + "=" * 70)
    print("SENTIMENT DECAY COMPARISON")
    print("=" * 70)
    print(f"{'Scenario':<20} {'Accuracy':>12} {'High Conf Acc':>15} {'Hi Conf Ratio':>15}")
    print("-" * 70)
    
    baseline_acc = results['no_decay']['accuracy']
    
    for scenario, data in results.items():
        acc = data['accuracy']
        hc_acc = data['high_confidence_accuracy']
        hc_ratio = data['high_confidence_ratio']
        
        diff = acc - baseline_acc
        diff_str = f"({diff:+.2%})" if scenario != 'no_decay' else ""
        
        print(f"{scenario:<20} {acc:>10.2%} {diff_str:>5} {hc_acc:>12.2%} {hc_ratio:>15.2%}")
    
    # Find best
    best_scenario = max(results.items(), key=lambda x: x[1]['accuracy'])
    print("-" * 70)
    print(f"Best: {best_scenario[0]} ({best_scenario[1]['accuracy']:.2%})")
    print(f"Improvement over baseline: {best_scenario[1]['accuracy'] - baseline_acc:+.2%}")
    print("=" * 70)
    print(f"\nDetailed results saved to: {output_dir}")
    
    return results


if __name__ == "__main__":
    main()
