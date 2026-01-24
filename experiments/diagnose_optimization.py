
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path.cwd()))

from config import training_config
from src.data import DatasetLoader, DataPreprocessor, FeatureEngineer
from src.models import CombinedLoss, BaselineLSTM, DualBranchLSTM
from src.training import Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OptimizationDiagnosis")

class DualBranchWrapper(torch.nn.Module):
    def __init__(self, price_size, sentiment_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.model = DualBranchLSTM(
            price_input_size=price_size,
            sentiment_input_size=sentiment_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        self.price_size = price_size
        
    def forward(self, x, return_hidden=False):
        price_features = x[:, :, :self.price_size]
        sentiment_features = x[:, :, self.price_size:]
        return self.model(price_features, sentiment_features)

def diagnose():
    logger.info("Starting Optimization Diagnosis...")
    
    # 1. Load Data (MSFT 2022+)
    loader = DatasetLoader(tickers=["MSFT"], date_range=("2016-01-01", "2024-12-31"), use_cache=True)
    news_df, prices_df = loader.load_fnspid()
    
    prices_df["date"] = pd.to_datetime(prices_df["date"]).dt.tz_localize(None)
    prices_df = prices_df[prices_df["date"] >= "2022-01-01"]
    
    engineer = FeatureEngineer()
    prices_df = engineer.add_technical_indicators(prices_df)
    
    preprocessor = DataPreprocessor()
    prices_df = preprocessor.compute_returns(prices_df)
    prices_df = preprocessor.label_trends(prices_df)
    
    # Fake sentiment for diagnosis if missing
    news_df["date"] = pd.to_datetime(news_df["date"]).dt.tz_localize(None)
    if "sentiment_positive" not in news_df.columns and "sentiment_value" in news_df.columns:
         news_df["sentiment_positive"] = np.where(news_df["sentiment_value"] > 0, news_df["sentiment_value"], 0)
         news_df["sentiment_negative"] = np.where(news_df["sentiment_value"] < 0, -news_df["sentiment_value"], 0)
         
    aligned_news = preprocessor.align_news_to_trading_day(news_df, prices_df)
    
    from src.nlp import SentimentAggregator
    aggregator = SentimentAggregator(strategy="sticky")
    sentiment_df = aggregator.aggregate_daily(aligned_news)
    
    prices_df = engineer.merge_price_and_sentiment(prices_df, sentiment_df)
    
    # 2. Check Data Stats
    feature_cols = engineer.get_feature_columns()
    final_cols = [c for c in feature_cols if c in prices_df.columns]
    
    prices_df = prices_df[prices_df["date"] >= "2022-04-01"]
    
    # Normalize!
    prices_df = preprocessor.normalize_features(prices_df, final_cols, fit=True)
    
    X, y, _ = preprocessor.create_sequences(prices_df, feature_cols=final_cols, sequence_length=20)
    
    # Inspect X
    X_np = X # It is already numpy
    logger.info(f"X shape: {X_np.shape}")
    logger.info(f"X mean: {X_np.mean():.4f}, std: {X_np.std():.4f}")
    logger.info(f"X min: {X_np.min():.4f}, max: {X_np.max():.4f}")
    
    # Check for dead features
    dead_features = []
    for i, col in enumerate(final_cols):
        col_mean = X_np[:, :, i].mean()
        col_std = X_np[:, :, i].std()
        if col_std < 1e-6:
            dead_features.append((col, col_mean))
    
    if dead_features:
        logger.warning(f"Dead features (std~0): {dead_features}")
    else:
        logger.info("No dead features found.")
        
    # 3. Overfit Single Batch Test
    logger.info("=== Running Overfit Single Batch Test ===")
    
    # Convert to Tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    # Select small batch
    batch_size = 32
    X_batch = X_tensor[:batch_size].to(training_config.device)
    y_batch = y_tensor[:batch_size].to(training_config.device)
    
    # Check for duplicates in X_batch
    # (Flatten to check uniqueness)
    X_flat = X_batch.reshape(batch_size, -1)
    unique_X = torch.unique(X_flat, dim=0)
    if len(unique_X) < len(X_batch):
        logger.warning(f"DUPLICATE INPUTS DETECTED! Unique: {len(unique_X)}/{len(X_batch)}")
    
    # Define feature groups
    price_cols = [c for c in final_cols if "sentiment" not in c and "news" not in c]
    sent_cols = [c for c in final_cols if c not in price_cols]
    
    # Setup Model - NO DROPOUT
    model = DualBranchWrapper(
        price_size=len(price_cols),
        sentiment_size=len(sent_cols),
        hidden_size=128,
        num_layers=2,
        dropout=0.0 # Disable dropout
    ).to(training_config.device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss() # Simple loss
    
    model.train()
    
    for epoch in range(100):
        optimizer.zero_grad()
        trend_logits, conf, _ = model(X_batch)
        
        # Simple Cross Entropy
        loss = loss_fn(trend_logits, y_batch)
        
        loss.backward()
        
        # Check gradients
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm().item()
        
        optimizer.step()
        
        if epoch % 10 == 0:
            # Calculate acc
            probs = torch.softmax(trend_logits, dim=-1)
            preds = probs.argmax(dim=-1)
            acc = (preds == y_batch).float().mean()
            logger.info(f"Epoch {epoch}: Loss={loss.item():.4f}, Acc={acc:.2%}, GradNorm={grad_norm:.4f}")
            
    logger.info("Using features: " + str(final_cols))

if __name__ == "__main__":
    try:
        diagnose()
    except Exception as e:
        logger.exception("Diagnosis failed")
