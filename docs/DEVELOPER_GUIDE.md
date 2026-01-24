# Developer Guide

## Sentiment-Driven Stock Trend Prediction

A machine learning system that combines financial news sentiment with technical indicators to predict stock price movements.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Quick Start](#quick-start)
3. [Project Structure](#project-structure)
4. [Architecture](#architecture)
5. [Configuration](#configuration)
6. [Data Pipeline](#data-pipeline)
7. [Models](#models)
8. [Training](#training)
9. [Running Experiments](#running-experiments)
10. [Adding New Features](#adding-new-features)
11. [Troubleshooting](#troubleshooting)

---

## Project Overview

### Goal
Predict stock price trend direction (Up/Down/Neutral) using:
- **Technical indicators** (RSI, MACD, Bollinger Bands, etc.)
- **News sentiment** (extracted via FinBERT)
- **Meta-prediction** (model confidence in its own predictions)

### Key Technologies
- **PyTorch** - Deep learning framework
- **FinBERT** (via HuggingFace Transformers) - Financial sentiment extraction
- **LSTM/Attention** - Sequence modeling for time series

---

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data Cache

The project uses cached sentiment data to avoid repeated FinBERT extraction:

```bash
# Cache is located at: data/cache/sentiment/
# Files: {TICKER}_sentiment.parquet, {TICKER}_prices.parquet
```

### 3. Run Training

```bash
# Full training pipeline
python main.py --mode train

# Demo mode (quick test)
python main.py --mode demo

# Evaluation
python main.py --mode evaluate
```

### 4. Run Experiments

```bash
# Multi-stock experiment
python experiments/iteration3_multistock.py --epochs 40

# Speculation experiment (all fixes)
python experiments/speculation_experiment.py --epochs 40
```

---

## Project Structure

```
this_is_my_quant/
├── config/
│   ├── __init__.py          # Config exports
│   └── settings.py          # All hyperparameters and paths
│
├── src/
│   ├── data/
│   │   ├── dataset_loader.py    # FNSPID/yfinance data loading
│   │   ├── preprocessor.py      # Returns, trends, normalization
│   │   └── feature_engineering.py  # Technical indicators, sentiment merge
│   │
│   ├── nlp/
│   │   ├── sentiment_extractor.py  # FinBERT wrapper
│   │   └── aggregator.py           # Daily sentiment aggregation
│   │
│   ├── models/
│   │   ├── baseline_lstm.py     # LSTM models (Baseline, Attention, DualBranch)
│   │   └── losses.py            # CombinedLoss, FocalLoss
│   │
│   └── training/
│       └── trainer.py           # Training loop, early stopping, checkpoints
│
│   ├── features/
│   │   └── indicators.py        # ComprehensiveIndicators (70+ technical indicators)
│
├── experiments/                 # Experiment scripts
├── docs/                        # Documentation
├── data/                        # Data storage
│   ├── cache/sentiment/         # Cached sentiment parquets
│   └── raw/                     # Raw FNSPID data
├── models/                      # Saved model checkpoints
├── reports/                     # Experiment results
└── main.py                      # Main entry point
```

---

## Architecture

### Data Flow

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│   FNSPID    │───▶│ DatasetLoader│───▶│FeatureEngineer │
│ (News+Price)│    │              │    │                 │
└─────────────┘    └──────────────┘    └────────┬────────┘
                                                │
┌─────────────┐    ┌──────────────┐    ┌────────▼────────┐
│  FinBERT    │◀───│ SentimentEx- │◀───│ DataPreprocessor│
│             │    │   tractor    │    │                 │
└─────────────┘    └──────────────┘    └────────┬────────┘
                                                │
                   ┌──────────────┐    ┌────────▼────────┐
                   │   Trainer    │───▶│  LSTM Model     │
                   │              │    │                 │
                   └──────────────┘    └─────────────────┘
```

### Model Architecture

```
Input: [batch, seq_len=20, features=26+]
    │
    ▼
┌─────────────────────────────────────┐
│          LSTM (2 layers)            │
│  hidden_size=128, dropout=0.1       │
└─────────────────┬───────────────────┘
                  │
    ┌─────────────┴─────────────┐
    ▼                           ▼
┌─────────────┐          ┌─────────────┐
│ Trend Head  │          │Confidence   │
│ (3 classes) │          │Head (1 out) │
└─────────────┘          └─────────────┘
    │                           │
    ▼                           ▼
  Up/Down/Neutral          0.0 - 1.0
```

---

## Configuration

All settings are in `config/settings.py`:

### Key Config Classes

| Class | Purpose |
|-------|---------|
| `LSTMConfig` | Model architecture (hidden_size, dropout, layers) |
| `TrainingConfig` | Training params (epochs, batch_size, learning_rate) |
| `FeatureConfig` | Feature engineering (technical indicators list) |
| `TrendConfig` | Classification thresholds (±0.5% default) |
| `SentimentConfig` | FinBERT settings (model_name, batch_size) |

### Modifying Config

```python
from config import lstm_config, training_config

# Access
print(lstm_config.hidden_size)  # 128

# Modify at runtime (not recommended for production)
lstm_config.dropout = 0.2
```

---

## Data Pipeline

### 1. Loading Data

```python
from src.data import DatasetLoader

loader = DatasetLoader(tickers=["MSFT", "AAPL"], use_cache=True)
news_df, prices_df = loader.load_fnspid()
```

### 2. Feature Engineering

```python
from src.data import FeatureEngineer, DataPreprocessor

engineer = FeatureEngineer()
preprocessor = DataPreprocessor()

# Add technical indicators
prices_df = engineer.add_technical_indicators(prices_df)

# Compute returns and label trends
prices_df = preprocessor.compute_returns(prices_df)
prices_df = preprocessor.label_trends(prices_df)

# Create sequences for LSTM
X, y, tickers = preprocessor.create_sequences(
    prices_df, 
    feature_cols=engineer.get_feature_columns(),
    sequence_length=20
)
```

### 2.5. Comprehensive Technical Indicators

For experiments requiring the full suite of 70+ technical indicators, use the dedicated `ComprehensiveIndicators` class:

```python
from src.features import ComprehensiveIndicators

# Initialize
indicators = ComprehensiveIndicators()

# Compute all 70+ indicators (requires pandas_ta)
df = indicators.compute_all(df)  # df must have OHLCV columns

# Get list of indicator column names
feature_cols = indicators.get_indicator_columns(df)
print(f"Generated {len(feature_cols)} indicators")  # ~73 indicators
```

#### Indicator Categories

| Category | Indicators | Count |
|----------|------------|-------|
| **Moving Averages** | SMA (5,10,20,50), EMA (5,12,26), VWMA, KAMA, Ichimoku | ~14 |
| **Momentum** | RSI, Stochastic, StochRSI, MACD, Williams %R, CCI, ROC, PPO, TRIX, Aroon, Coppock, KST, AO | ~25 |
| **Volume** | MFI, BOP, PVO, OBV, Volume Ratio, RVGI | ~8 |
| **Volatility** | Bollinger Bands, ATR, True Range, Choppiness | ~10 |
| **Trend** | ADX (+DI/-DI), Bull/Bear Power | ~5 |
| **Price** | Returns (1d,5d,10d,20d), Log Return, Gap, High-Low %, Intraday Range | ~8 |

#### Usage in Experiments

```python
# In pooled experiments (recommended pattern)
from src.features import ComprehensiveIndicators

class MyExperiment:
    def __init__(self):
        self.indicator_computer = ComprehensiveIndicators()
    
    def prepare_stock_data(self, df):
        df = self.indicator_computer.compute_all(df)
        feature_cols = self.indicator_computer.get_indicator_columns(df)
        # ... rest of preprocessing
```

> **Note**: Requires `pandas_ta` package. Install with: `pip install pandas_ta`

### 3. Sentiment Extraction

```python
from src.nlp import SentimentExtractor, SentimentAggregator

# Extract sentiment from headlines
extractor = SentimentExtractor()
news_df = extractor.extract_dataframe(news_df, text_col="headline")

# Aggregate to daily
aggregator = SentimentAggregator(strategy="sticky")  # or "time_weighted"
sentiment_df = aggregator.aggregate_daily(news_df)
```

---

## Models

### Available Models

| Model | Description | Use Case |
|-------|-------------|----------|
| `BaselineLSTM` | Standard LSTM with trend + confidence heads | Default |
| `AttentionLSTM` | LSTM with self-attention over sequence | High-news stocks |
| `DualBranchLSTM` | Separate price/sentiment branches | Experimental |

### Using Models

```python
from src.models import BaselineLSTM, CombinedLoss

model = BaselineLSTM(
    input_size=26,           # Number of features
    hidden_size=128,
    num_layers=2,
    dropout=0.1,
)

# Forward pass
trend_logits, confidence, hidden = model(X)

# Inference
predictions = model.predict(X, threshold=0.5)
# Returns: trend_class, trend_probs, confidence, should_trade
```

---

## Training

### Using the Trainer

```python
from src.training import Trainer
from src.models import BaselineLSTM
import torch.nn as nn

model = BaselineLSTM(input_size=26)
loss_fn = nn.CrossEntropyLoss()  # or CombinedLoss()

trainer = Trainer(model, loss_fn)

# Create data loaders
train_loader = trainer._create_loader(X_train, y_train, shuffle=True)
val_loader = trainer._create_loader(X_val, y_val)

# Train
history = trainer.train(train_loader, val_loader, epochs=40)
```

### Loss Functions

| Loss | Description |
|------|-------------|
| `CrossEntropyLoss` | Simple classification (recommended for small datasets) |
| `CombinedLoss` | Trend + Confidence joint optimization |
| `FocalLoss` | Focus on hard examples (imbalanced classes) |

---

## Running Experiments

### Iteration 3 Multi-Stock

Tests multiple model variants across stocks:

```bash
python experiments/iteration3_multistock.py --epochs 40 --stocks WMT DIS MSFT
```

**Variants tested:**
- Baseline
- Event-based features
- Curriculum learning  
- Attention mechanism
- All combined

### Speculation Experiment

Implements all hypothesized fixes:

```bash
python experiments/speculation_experiment.py --epochs 40
```

**Fixes applied:**
- Dead zone filtering (per-stock cutoffs)
- Signal-to-noise filtering
- Sticky sentiment aggregation
- Dual-channel sentiment (pos/neg split)

### Diagnostic Scripts

```bash
# Optimization diagnosis (overfit test)
python experiments/diagnose_optimization.py

# Distribution analysis
python debug_distribution.py

# Signal quality test
python experiments/signal_quality_test.py
```

---

## Adding New Features

### Adding a Technical Indicator

1. Edit `src/data/feature_engineering.py`:

```python
def _compute_indicators_for_stock(self, df, indicators):
    # ... existing indicators ...
    
    if "my_indicator" in indicators:
        df["my_indicator"] = self._compute_my_indicator(df["close"])
```

2. Add to `config/settings.py`:

```python
technical_indicators: List[str] = field(default_factory=lambda: [
    # ... existing ...
    "my_indicator",
])
```

### Adding a New Model

1. Create class in `src/models/baseline_lstm.py`:

```python
class MyNewModel(nn.Module):
    def __init__(self, input_size, ...):
        super().__init__()
        # ... layers ...
    
    def forward(self, x, return_hidden=False):
        # Must return: (trend_logits, confidence, optional_hidden)
        return trend_logits, confidence, hidden
    
    def predict(self, x, threshold=0.5):
        # Must return dict with: trend_class, trend_probs, confidence, should_trade
        ...
```

2. Export in `src/models/__init__.py`

### Adding a New Aggregation Strategy

Edit `src/nlp/aggregator.py`:

```python
class SentimentAggregator:
    def __init__(self, strategy="time_weighted"):
        self.strategy = strategy
    
    def aggregate_daily(self, news_df):
        if self.strategy == "my_strategy":
            return self._my_aggregation(news_df)
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce `batch_size` in config or use `mixed_precision=True` |
| `No sentiment column found` | Load from `_sentiment.parquet` not `_news.parquet` |
| `Model not learning (flat loss)` | Reduce dropout, check for NaN in features |
| `All predictions same class` | Check class balance, use class weights |

### Debugging Training

```python
# Check for dead features
for i, col in enumerate(feature_cols):
    std = X[:, :, i].std()
    if std < 1e-6:
        print(f"Dead feature: {col}")

# Overfit test (should reach ~100%)
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    loss = loss_fn(model(X_batch)[0], y_batch)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: Loss={loss.item():.4f}")
```

### Performance Tips

1. **Use cached sentiment** - Avoid repeated FinBERT extraction
2. **Reduce dropout** - 0.1 works better than 0.3 for small datasets
3. **Use simple loss** - `CrossEntropyLoss` trains easier than `CombinedLoss`
4. **Disable early stopping** - Set `patience=999` to train full epochs

---

## Related Documentation

- [Data Pipeline](docs/DATA_PIPELINE.md)
- [Model Architecture](docs/MODEL_ARCHITECTURE.md)
- [Design Decisions](docs/DESIGN_DECISIONS.md)
- [Theoretical Foundation](theoretical_foundation.md)

---

## Version

**Current**: 0.1.0

**Last Updated**: January 2026
