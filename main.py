"""
Main entry point for Sentiment-Driven Stock Trend Prediction.

Usage:
    python main.py --mode train
    python main.py --mode evaluate
    python main.py --mode demo
"""

import argparse
import logging
from pathlib import Path
import sys

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    TOP_200_TICKERS,
    DATE_RANGE,
    training_config,
    sentiment_config,
    lstm_config,
    MODELS_DIR,
)
from src.data import DatasetLoader, DataPreprocessor, FeatureEngineer
from src.nlp import SentimentExtractor, SentimentAggregator
from src.models import BaselineLSTM, CombinedLoss
from src.training import Trainer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_and_prepare_data(
    tickers: list = None,
    use_cache: bool = True,
) -> tuple:
    """
    Load and prepare data for training.
    
    Returns:
        Tuple of (X, y, feature_columns) ready for model training.
    """
    tickers = tickers or TOP_200_TICKERS[:50]  # Start with 50 stocks for testing
    
    logger.info(f"Loading data for {len(tickers)} stocks...")
    
    # 1. Load price data
    loader = DatasetLoader(tickers=tickers, use_cache=use_cache)
    news_df, prices_df = loader.load_fnspid()
    
    # 2. Feature engineering
    engineer = FeatureEngineer()
    prices_df = engineer.add_technical_indicators(prices_df)
    
    # 3. Compute returns and labels
    preprocessor = DataPreprocessor()
    prices_df = preprocessor.compute_returns(prices_df)
    prices_df = preprocessor.label_trends(prices_df)
    
    # 4. Extract sentiment (if news available)
    if len(news_df) > 0:
        logger.info("Extracting sentiment from news...")
        extractor = SentimentExtractor()
        news_df = extractor.extract_dataframe(
            news_df, 
            text_col="headline" if "headline" in news_df.columns else "title",
            cache_key=f"{len(tickers)}_stocks",
        )
        
        # Align and aggregate sentiment
        news_df = preprocessor.align_news_to_trading_day(news_df, prices_df)
        aggregator = SentimentAggregator()
        sentiment_df = aggregator.aggregate_daily(news_df)
        
        # Merge with prices
        prices_df = engineer.merge_price_and_sentiment(prices_df, sentiment_df)
    else:
        logger.warning("No news data available, using price features only")
        # Add dummy sentiment columns
        prices_df["sentiment_mean"] = 0.0
        prices_df["sentiment_dispersion"] = 0.0
        prices_df["news_count"] = 0
    
    # 5. Normalize features
    feature_cols = engineer.get_feature_columns()
    feature_cols = [c for c in feature_cols if c in prices_df.columns]
    prices_df = preprocessor.normalize_features(prices_df, feature_cols, fit=True)
    
    # 6. Create sequences
    X, y, tickers_arr = preprocessor.create_sequences(
        prices_df,
        feature_cols=feature_cols,
        target_col="trend",
    )
    
    logger.info(f"Prepared data: X.shape={X.shape}, y.shape={y.shape}")
    
    return X, y, feature_cols


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: list,
    epochs: int = None,
) -> BaselineLSTM:
    """
    Train the LSTM model.
    
    Args:
        X: Feature array (n_samples, seq_len, n_features).
        y: Target array (n_samples,).
        feature_cols: List of feature column names.
        epochs: Number of training epochs.
        
    Returns:
        Trained model.
    """
    epochs = epochs or training_config.epochs
    
    # Create model
    model = BaselineLSTM(
        input_size=X.shape[-1],
        hidden_size=lstm_config.hidden_size,
        num_layers=lstm_config.num_layers,
        dropout=lstm_config.dropout,
    )
    
    logger.info(f"Model parameters: {model.get_num_parameters():,}")
    
    # Create loss function
    loss_fn = CombinedLoss(
        trend_weight=1.0,
        confidence_weight=0.5,
    )
    
    # Create trainer
    trainer = Trainer(model, loss_fn)
    
    # Split data (time-series aware)
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train),
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val),
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=0,  # Windows compatibility
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        num_workers=0,
    )
    
    # Train
    trainer.train(train_loader, val_loader, epochs=epochs)
    
    # Load best model
    best_path = trainer.checkpoint_dir / "best_model.pt"
    trainer.load_checkpoint(str(best_path))
    
    return model


def evaluate_model(
    model: BaselineLSTM,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    model.eval()
    device = next(model.parameters()).device
    
    X_tensor = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        predictions = model.predict(X_tensor)
    
    pred_classes = predictions["trend_class"].cpu().numpy()
    confidences = predictions["confidence"].cpu().numpy()
    
    # Overall accuracy
    accuracy = (pred_classes == y_test).mean()
    
    # High confidence accuracy
    high_conf_mask = confidences > 0.5
    if high_conf_mask.sum() > 0:
        high_conf_acc = (pred_classes[high_conf_mask] == y_test[high_conf_mask]).mean()
    else:
        high_conf_acc = 0.0
    
    # Per-class metrics
    from sklearn.metrics import classification_report, confusion_matrix
    
    report = classification_report(
        y_test, pred_classes,
        target_names=["Down", "Neutral", "Up"],
        output_dict=True,
    )
    
    conf_matrix = confusion_matrix(y_test, pred_classes)
    
    results = {
        "accuracy": accuracy,
        "high_confidence_accuracy": high_conf_acc,
        "high_confidence_ratio": high_conf_mask.mean(),
        "classification_report": report,
        "confusion_matrix": conf_matrix.tolist(),
    }
    
    logger.info(f"Test Accuracy: {accuracy:.2%}")
    logger.info(f"High Confidence Accuracy: {high_conf_acc:.2%}")
    logger.info(f"High Confidence Ratio: {high_conf_mask.mean():.2%}")
    
    return results


def demo():
    """
    Demonstrate the sentiment extraction and model inference.
    """
    logger.info("Running demo...")
    
    # Demo sentiment extraction
    logger.info("\n=== Sentiment Extraction Demo ===")
    
    extractor = SentimentExtractor()
    
    test_headlines = [
        "Apple reports record quarterly revenue, beating analyst expectations",
        "Tesla faces investigation over autopilot safety concerns",
        "Federal Reserve holds interest rates steady amid economic uncertainty",
        "Microsoft acquires gaming company in billion-dollar deal",
        "Oil prices surge as OPEC cuts production targets",
    ]
    
    print("\nSentiment Analysis Results:")
    print("-" * 70)
    
    for headline in test_headlines:
        result = extractor.extract(headline)
        print(f"\n{headline}")
        print(f"  → {result.label.upper()} ({result.score:.1%})")
    
    # Demo model inference (if model exists)
    best_model_path = MODELS_DIR / "best_model.pt"
    
    if best_model_path.exists():
        logger.info("\n=== Model Inference Demo ===")
        
        model = BaselineLSTM(input_size=25)
        model.load_state_dict(torch.load(best_model_path, weights_only=False)["model_state_dict"])
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 20, 25)
        predictions = model.predict(dummy_input)
        
        trend_names = ["Down", "Neutral", "Up"]
        pred_class = predictions["trend_class"].item()
        confidence = predictions["confidence"].item()
        
        print(f"\nPrediction: {trend_names[pred_class]} (confidence: {confidence:.1%})")
    else:
        logger.info("No trained model found. Train a model first with --mode train")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Sentiment-Driven Stock Trend Prediction"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "demo", "full_pipeline"],
        default="demo",
        help="Execution mode",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--stocks",
        type=int,
        default=50,
        help="Number of stocks to use (for testing)",
    )
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        demo()
        
    elif args.mode == "train":
        logger.info("Starting training pipeline...")
        X, y, feature_cols = load_and_prepare_data(
            tickers=TOP_200_TICKERS[:args.stocks]
        )
        model = train_model(X, y, feature_cols, epochs=args.epochs)
        logger.info("Training complete!")
        
    elif args.mode == "evaluate":
        logger.info("Starting evaluation...")
        X, y, feature_cols = load_and_prepare_data(
            tickers=TOP_200_TICKERS[:args.stocks]
        )
        
        # Load model
        model = BaselineLSTM(input_size=len(feature_cols))
        best_path = list(MODELS_DIR.glob("*/best_model.pt"))
        
        if best_path:
            model.load_state_dict(
                torch.load(best_path[-1], weights_only=False)["model_state_dict"]
            )
            
            # Use last 15% as test
            test_start = int(0.85 * len(X))
            results = evaluate_model(model, X[test_start:], y[test_start:])
            
            print("\n=== Evaluation Results ===")
            print(f"Accuracy: {results['accuracy']:.2%}")
            print(f"High Conf Accuracy: {results['high_confidence_accuracy']:.2%}")
        else:
            logger.error("No trained model found. Train first with --mode train")
            
    elif args.mode == "full_pipeline":
        logger.info("Running full pipeline...")
        
        # 1. Load data
        X, y, feature_cols = load_and_prepare_data(
            tickers=TOP_200_TICKERS[:args.stocks]
        )
        
        # 2. Train
        model = train_model(X, y, feature_cols, epochs=args.epochs)
        
        # 3. Evaluate on test set
        test_start = int(0.85 * len(X))
        results = evaluate_model(model, X[test_start:], y[test_start:])
        
        logger.info("Full pipeline complete!")


if __name__ == "__main__":
    main()
