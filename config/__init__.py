"""Configuration module for Sentiment-Driven Stock Trend Prediction."""

from .settings import (
    # Paths
    PROJECT_ROOT,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CACHE_DIR,
    MODELS_DIR,
    LOGS_DIR,
    REPORTS_DIR,
    
    # Stock universe
    TOP_200_TICKERS,
    DATE_RANGE,
    
    # Config instances
    sentiment_config,
    lstm_config,
    training_config,
    feature_config,
    trend_config,
    dataset_config,
    
    # Config classes
    SentimentConfig,
    LSTMConfig,
    TrainingConfig,
    FeatureConfig,
    TrendConfig,
    DatasetConfig,
)

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "CACHE_DIR",
    "MODELS_DIR",
    "LOGS_DIR",
    "REPORTS_DIR",
    "TOP_200_TICKERS",
    "DATE_RANGE",
    "sentiment_config",
    "lstm_config",
    "training_config",
    "feature_config",
    "trend_config",
    "dataset_config",
    "SentimentConfig",
    "LSTMConfig",
    "TrainingConfig",
    "FeatureConfig",
    "TrendConfig",
    "DatasetConfig",
]
