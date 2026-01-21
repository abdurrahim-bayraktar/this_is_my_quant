"""
Configuration settings for Sentiment-Driven Stock Trend Prediction.

This module contains all hyperparameters, paths, and settings used across the project.
Modify values here rather than hardcoding throughout the codebase.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple
import torch


# =============================================================================
# Path Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR, MODELS_DIR, LOGS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Stock Universe Configuration
# =============================================================================

# Top 200 S&P 500 stocks by market cap (as of 2024)
# This list will be used to filter FNSPID dataset
TOP_200_TICKERS: List[str] = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "GOOG", "NVDA", "META", "AVGO", "ORCL", "CSCO", "ACN",
    "ADBE", "CRM", "AMD", "INTC", "IBM", "QCOM", "TXN", "INTU", "NOW", "AMAT",
    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "BKNG", "CMG",
    # Healthcare
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "GILD", "MDT", "ISRG", "CVS", "ELV", "SYK", "VRTX", "REGN", "ZTS",
    # Financials
    "BRK.B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "SPGI",
    "BLK", "C", "SCHW", "CB", "MMC", "PGR", "AON", "ICE", "CME", "USB",
    # Communication Services
    "NFLX", "DIS", "CMCSA", "VZ", "T", "CHTR", "TMUS", "EA", "ATVI", "WBD",
    # Industrials
    "CAT", "GE", "RTX", "HON", "UNP", "BA", "UPS", "LMT", "DE", "MMM",
    "ADP", "ETN", "ITW", "FDX", "NSC", "CSX", "WM", "GD", "NOC", "EMR",
    # Consumer Staples
    "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ", "CL", "EL",
    "KHC", "GIS", "HSY", "KMB", "SYY", "ADM", "STZ", "K", "CAG", "CPB",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "KMI",
    "WMB", "HAL", "DVN", "BKR", "FANG", "HES", "TRGP", "OKE", "CTRA", "MRO",
    # Utilities
    "NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "PEG", "ED",
    "WEC", "ES", "AWK", "DTE", "AEE", "CMS", "FE", "EVRG", "NI", "ATO",
    # Materials
    "LIN", "APD", "SHW", "ECL", "FCX", "NEM", "NUE", "DOW", "DD", "VMC",
    "MLM", "PPG", "CF", "ALB", "CTVA", "IFF", "LYB", "MOS", "FMC", "CE",
    # Real Estate
    "PLD", "AMT", "EQIX", "PSA", "CCI", "O", "SPG", "WELL", "DLR", "AVB",
    "EQR", "VTR", "ARE", "MAA", "UDR", "ESS", "SUI", "HST", "KIM", "REG",
]

# Date range for data filtering
DATE_RANGE: Tuple[str, str] = ("2016-01-01", "2024-12-31")


# =============================================================================
# Model Configuration
# =============================================================================

@dataclass
class SentimentConfig:
    """Configuration for sentiment extraction model."""
    model_name: str = "ProsusAI/finbert"
    max_length: int = 512
    batch_size: int = 16  # Optimized for 6GB VRAM
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cache_embeddings: bool = True  # Pre-compute and cache for training


@dataclass
class LSTMConfig:
    """Configuration for LSTM prediction model."""
    input_size: int = 0  # Set dynamically based on features
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = False
    
    # Output heads
    num_trend_classes: int = 3  # Up, Down, Neutral
    confidence_output: bool = True  # Meta-prediction


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Basic settings
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    # Sequence settings
    sequence_length: int = 20  # Look-back window in trading days
    
    # Optimizer
    optimizer: str = "adamw"
    scheduler: str = "cosine"  # cosine, step, none
    warmup_epochs: int = 5
    
    # Regularization
    gradient_clip: float = 1.0
    early_stopping_patience: int = 10
    
    # Validation
    val_split: float = 0.15
    walk_forward_folds: int = 5
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True  # FP16 for memory efficiency
    num_workers: int = 4


# =============================================================================
# Feature Configuration
# =============================================================================

@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    # Price features
    price_features: List[str] = field(default_factory=lambda: [
        "open", "high", "low", "close", "volume", "adj_close"
    ])
    
    # Technical indicators to compute
    technical_indicators: List[str] = field(default_factory=lambda: [
        "rsi_14", "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_middle", "bb_lower", "bb_width",
        "sma_5", "sma_20", "sma_50", "ema_12", "ema_26",
        "atr_14", "mfi_14", "obv"
    ])
    
    # Sentiment aggregation
    sentiment_aggregation: str = "time_weighted"  # simple, volume_weighted, time_weighted
    
    # Normalization
    normalization: str = "zscore"  # zscore, minmax, robust


# =============================================================================
# Trend Classification Configuration
# =============================================================================

@dataclass
class TrendConfig:
    """Configuration for trend classification."""
    # Threshold for trend classification (based on literature: ±0.5%)
    up_threshold: float = 0.005    # > 0.5% = Up
    down_threshold: float = -0.005  # < -0.5% = Down
    # Between thresholds = Neutral
    
    # Class labels
    labels: List[str] = field(default_factory=lambda: ["Down", "Neutral", "Up"])
    
    # Class weights for imbalanced data (optional, set to None for equal weights)
    class_weights: List[float] = None


# =============================================================================
# Dataset Configuration
# =============================================================================

@dataclass  
class DatasetConfig:
    """Configuration for dataset handling."""
    # FNSPID settings - CORRECT repo (not Zdong104 which doesn't work)
    fnspid_repo: str = "Zihan1004/FNSPID"
    
    # Twitter Financial Sentiment (for validation/fine-tuning)
    twitter_sentiment_repo: str = "zeroshot/twitter-financial-news-sentiment"
    
    # Data splits
    train_end_date: str = "2022-12-31"
    val_end_date: str = "2023-06-30"
    # Test: 2023-07-01 to 2024-12-31
    
    # Caching
    use_cache: bool = True
    cache_sentiment: bool = True  # Pre-extract sentiment to disk


# =============================================================================
# Global Config Instance
# =============================================================================

sentiment_config = SentimentConfig()
lstm_config = LSTMConfig()
training_config = TrainingConfig()
feature_config = FeatureConfig()
trend_config = TrendConfig()
dataset_config = DatasetConfig()
