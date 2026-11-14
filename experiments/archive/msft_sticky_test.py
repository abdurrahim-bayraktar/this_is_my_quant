
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path.cwd()))

from config import (
    TOP_200_TICKERS,
    DATE_RANGE,
    training_config,
    sentiment_config,
    lstm_config,
    MODELS_DIR,
    feature_config,
    trend_config
)
from src.data import DatasetLoader, DataPreprocessor, FeatureEngineer
from src.nlp import SentimentExtractor, SentimentAggregator
from src.models import BaselineLSTM, CombinedLoss
from src.training import Trainer

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("experiments/msft_test.log")
    ]
)
logger = logging.getLogger("MSFT_Test")

def main():
    logger.info("Starting MSFT Sticky Sentiment Test (2022-Q2 onwards)")
    
    # 1. Configuration
    # DATE_RANGE is global, we need to filter later or monkeypatch?
    # DatasetLoader takes date_range in init.
    
    test_start_date = "2022-04-01"
    test_end_date = "2024-12-31"
    
    feature_config.sentiment_aggregation = "sticky"  # Force sticky strategy
    
    # 2. Load Data
    logger.info("Loading data for MSFT...")
    loader = DatasetLoader(
        tickers=["MSFT"],
        date_range=("2016-01-01", "2024-12-31"), # Load all to ensure cache hit
        use_cache=True
    )
    
    news_df, prices_df = loader.load_fnspid()
    
    # Filter for our test period
    # Note: We need some lookback for indicators, so keep a buffer
    prices_df["date"] = pd.to_datetime(prices_df["date"]).dt.tz_localize(None)
    prices_df = prices_df[prices_df["date"] >= "2022-01-01"] 
    
    # 3. Feature Engineering
    logger.info("Engineering features...")
    engineer = FeatureEngineer()
    prices_df = engineer.add_technical_indicators(prices_df)
    
    # 4. Returns & Labels
    logger.info("Computing returns...")
    preprocessor = DataPreprocessor()
    prices_df = preprocessor.compute_returns(prices_df)
    prices_df = preprocessor.label_trends(prices_df)
    
    # 5. Sticky Sentiment
    logger.info("Aggregating sentiment with STICKY logic...")
    # Alignment
    news_df["date"] = pd.to_datetime(news_df["date"]).dt.tz_localize(None)
    # Ensure sentiment_pos/neg exists if not loaded from cache correctly
    if "sentiment_positive" in news_df.columns:
         pass # Good
    elif "sentiment_value" in news_df.columns:
         # Rough approx if raw pos/neg missing
         news_df["sentiment_positive"] = np.where(news_df["sentiment_value"] > 0, news_df["sentiment_value"], 0)
         news_df["sentiment_negative"] = np.where(news_df["sentiment_value"] < 0, -news_df["sentiment_value"], 0)
         
    aligned_news = preprocessor.align_news_to_trading_day(news_df, prices_df)
    
    # Aggregator
    aggregator = SentimentAggregator(strategy="sticky")
    sentiment_df = aggregator.aggregate_daily(aligned_news)
    
    # Merge
    prices_df = engineer.merge_price_and_sentiment(prices_df, sentiment_df)
    
    # Filter for strictly > 2022-04-01 for training
    prices_df = prices_df[prices_df["date"] >= test_start_date]
    
    logger.info(f"Training data shape: {prices_df.shape}")
    logger.info(f"Class distribution: {prices_df['trend_label'].value_counts(normalize=True).to_dict()}")
    
    # 6. Prepare for Model
    feature_cols = engineer.get_feature_columns()
    feature_cols = [c for c in feature_cols if c in prices_df.columns]
    
    logger.info(f"Using {len(feature_cols)} features: {feature_cols}")
    
    prices_df = preprocessor.normalize_features(prices_df, feature_cols, fit=True)
    
    X, y, _ = preprocessor.create_sequences(
        prices_df,
        feature_cols=feature_cols,
        sequence_length=20
    )
    
    # 7. Train
    # Use DualBranchLSTM to prevent price signal from drowning out sentiment
    prices_len = 20 # approx technicals + price
    sentiment_len = len(feature_cols) - prices_len
    
    # Calculate class weights
    counts = prices_df['trend_label'].value_counts()
    total = sum(counts)
    # Map labels to indices: Down=0, Neutral=1, Up=2 (need to verify mapping in Preprocessor)
    # Preprocessor defaults: Down, Neutral, Up
    weights = torch.tensor([
        total / counts.get('Down', 1),
        total / counts.get('Neutral', 1),
        total / counts.get('Up', 1)
    ], dtype=torch.float32)
    weights = weights / weights.sum() * 3  # Normalize
    logger.info(f"Using class weights: {weights}")
    
    from src.models import DualBranchLSTM
    
    class DualBranchWrapper(torch.nn.Module):
        def __init__(self, price_size, sentiment_size, hidden_size=64, num_layers=1, dropout=0.3):
            super().__init__()
            self.model = DualBranchLSTM(
                price_input_size=price_size,
                sentiment_input_size=sentiment_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout
            )
            self.price_size = price_size
            
        def forward(self, x, return_hidden=False): # Add return_hidden to match signature if needed
            # Split input
            # x: (batch, seq, features)
            # price features are first, then sentiment
            price_features = x[:, :, :self.price_size]
            sentiment_features = x[:, :, self.price_size:]
            
            return self.model(price_features, sentiment_features)
            
        def predict(self, x, threshold=0.5):
            self.eval()
            with torch.no_grad():
                trend_logits, confidence, _ = self.forward(x)
                trend_probs = torch.nn.functional.softmax(trend_logits, dim=-1)
                trend_class = trend_probs.argmax(dim=-1)
                return {
                    "trend_class": trend_class,
                    "trend_probs": trend_probs,
                    "confidence": confidence.squeeze(-1)
                }
    
    # Instantiate DualBranchWrapper
    # Prices: open, high, low, close, volume + technicals (16) = 21 approx
    # Actually count from feature list
    feature_cols_price = [c for c in feature_cols if "sentiment" not in c and "news" not in c and "has_" not in c]
    feature_cols_sentiment = [c for c in feature_cols if c not in feature_cols_price]
    
    logger.info(f"DualBranch Split: {len(feature_cols_price)} Price Features, {len(feature_cols_sentiment)} Sentiment Features")
    
    # Re-order X to match split? feature_cols list should be ordered price then sentiment
    # Engineer.get_feature_columns() does this order.
    # Let's verify and enforce order
    
    final_feature_cols = feature_cols_price + feature_cols_sentiment
    # We must re-create sequences with this order
    prices_df = preprocessor.normalize_features(prices_df, final_feature_cols, fit=True)
    X, y, _ = preprocessor.create_sequences(prices_df, feature_cols=final_feature_cols, sequence_length=20)
    
    # Define Model first
    model = DualBranchWrapper(
        price_size=len(feature_cols_price),
        sentiment_size=len(feature_cols_sentiment),
        hidden_size=128, # Increased capacity
        num_layers=2,
        dropout=0.3
    )
    
    loss_fn = CombinedLoss(class_weights=weights.to(training_config.device))
    trainer = Trainer(model, loss_fn)

    # Split Data
    train_size = int(0.8 * len(X))
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]
    
    train_loader = trainer._create_loader(X_train, y_train, shuffle=True)
    val_loader = trainer._create_loader(X_val, y_val)
    
    history = trainer.train(train_loader, val_loader, epochs=40)
    
    # 8. Report
    logger.info("=== Test Complete ===")
    logger.info(f"Final Val Accuracy: {history['history'][-1]['val_accuracy']:.2%}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Fatal error in MSFT test")
        raise
