# Sentiment-Driven Stock Trend Prediction

> **A deep learning system for predicting stock price movements using financial news sentiment with meta-prediction capability (confidence scoring).**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Design Decisions](#design-decisions)
- [Model Details](#model-details)
- [Data Pipeline](#data-pipeline)
- [Experiments](#experiments)
- [Results](#results)
- [References](#references)

---

## Overview

This project implements a **sentiment-driven stock trend prediction system** that combines:

1. **Financial News Sentiment**: Extracted using FinBERT from news headlines and tweets
2. **Technical Indicators**: RSI, MACD, Bollinger Bands, and more
3. **LSTM Neural Network**: Processes sequential features for trend prediction
4. **Meta-Prediction**: Outputs a confidence score indicating prediction reliability

### The Problem

Stock price prediction is inherently noisy. The Efficient Market Hypothesis (EMH) suggests that prices already incorporate all available information. However, **sentiment data introduces new information** about market psychology that may not yet be fully priced in.

### Our Solution

We use a dual-head LSTM that predicts:
- **Trend Direction**: Up (+0.5%), Down (-0.5%), or Neutral
- **Confidence Score**: Probability that the prediction is correct

This allows trading strategies to filter for **high-confidence predictions only**, improving expected returns.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **FinBERT Sentiment** | State-of-the-art financial sentiment extraction |
| **Meta-Prediction** | Model outputs confidence score for each prediction |
| **Mixed Precision Training** | FP16 for memory efficiency on consumer GPUs |
| **Walk-Forward Validation** | Proper time-series cross-validation |
| **Ablation Study** | Compare price-only vs sentiment-enhanced predictions |
| **Modular Design** | Easy to swap data sources, models, or features |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      DATA PIPELINE                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────┐   │
│  │   FNSPID    │   │   Twitter   │   │     yfinance        │   │
│  │   (News)    │   │  Sentiment  │   │     (Prices)        │   │
│  └──────┬──────┘   └──────┬──────┘   └──────────┬──────────┘   │
│         │                 │                      │              │
│         ▼                 ▼                      ▼              │
│  ┌─────────────────────────────┐    ┌───────────────────────┐  │
│  │        FinBERT              │    │  Technical Indicators │  │
│  │   Sentiment Extraction      │    │  RSI, MACD, BB, etc.  │  │
│  └─────────────┬───────────────┘    └───────────┬───────────┘  │
│                │                                 │              │
│                └────────────┬───────────────────┘               │
│                             ▼                                   │
│                  ┌─────────────────────┐                        │
│                  │  Feature Aggregator │                        │
│                  │  (Daily Sequences)  │                        │
│                  └──────────┬──────────┘                        │
└─────────────────────────────┼───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LSTM MODEL                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: [batch, seq_len=20, features=~25]                       │
│                     │                                           │
│                     ▼                                           │
│         ┌─────────────────────────┐                             │
│         │   LSTM (2 layers, 128)  │                             │
│         │      + Dropout 0.3      │                             │
│         └───────────┬─────────────┘                             │
│                     │                                           │
│                     ▼                                           │
│         ┌─────────────────────────┐                             │
│         │   Shared Dense (64)     │                             │
│         └───────────┬─────────────┘                             │
│                     │                                           │
│         ┌───────────┴───────────┐                               │
│         ▼                       ▼                               │
│  ┌─────────────────┐   ┌─────────────────┐                      │
│  │   Trend Head    │   │ Confidence Head │                      │
│  │   Dense(32→3)   │   │  Dense(32→1)    │                      │
│  │   [Softmax]     │   │   [Sigmoid]     │                      │
│  └────────┬────────┘   └────────┬────────┘                      │
│           │                     │                               │
│           ▼                     ▼                               │
│      Up/Down/Neutral       0.0 - 1.0                            │
│       Prediction          Confidence                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended: 6GB+ VRAM)
- 15GB disk space (for datasets)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/this_is_my_quant.git
cd this_is_my_quant

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA (if not already)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

---

## Quick Start

### Demo (Sentiment Extraction)

```bash
python main.py --mode demo
```

Output:
```
Apple reports record quarterly revenue, beating analyst expectations
  → POSITIVE (94.4%)

Tesla faces investigation over autopilot safety concerns
  → NEGATIVE (94.3%)
```

### Train Model

```bash
# Quick training (10 stocks, 10 epochs)
python main.py --mode train --stocks 10 --epochs 10

# Full training (50 stocks, 50 epochs)
python main.py --mode train --stocks 50 --epochs 50
```

### Run Ablation Study

```bash
# Compare: no sentiment vs Twitter vs News vs Combined
python experiments/ablation_sentiment.py --scenarios all --stocks 50 --epochs 30
```

---

## Project Structure

```
this_is_my_quant/
├── config/
│   ├── __init__.py
│   └── settings.py          # All hyperparameters and configurations
│
├── src/
│   ├── data/
│   │   ├── dataset_loader.py    # FNSPID, Twitter, yfinance loaders
│   │   ├── preprocessor.py      # Returns, trend labels, sequences
│   │   └── feature_engineering.py  # Technical indicators
│   │
│   ├── nlp/
│   │   ├── sentiment_extractor.py  # FinBERT wrapper
│   │   └── aggregator.py           # Daily sentiment aggregation
│   │
│   ├── models/
│   │   ├── baseline_lstm.py    # LSTM with dual heads
│   │   └── losses.py           # CombinedLoss, FocalLoss
│   │
│   └── training/
│       └── trainer.py          # Training loop, walk-forward validation
│
├── experiments/
│   └── ablation_sentiment.py   # Sentiment source comparison
│
├── main.py                     # CLI entry point
├── requirements.txt
└── README.md
```

---

## Design Decisions

### 1. Why LSTM over Transformer?

| Criterion | LSTM | Transformer |
|-----------|------|-------------|
| Sequence length | 20 days (short) | Best for long sequences |
| Training data | ~50k samples | Needs 100k+ |
| Memory usage | ~130k params | ~1M+ params |
| Interpretability | Hidden states | Attention matrices |

**Decision**: LSTM is more suitable for our short sequences and limited data.

### 2. Why ±0.5% Trend Threshold?

Based on literature review:
- **Academic standard**: Used in StockNet, DP-LSTM papers
- **Transaction costs**: ~0.1-0.3% per trade
- **Class balance**: ~30% Up, 40% Neutral, 30% Down

### 3. Why Meta-Prediction (Confidence Output)?

Standard classification gives equal weight to all predictions. With meta-prediction:
- Model learns to "know what it knows"
- High confidence predictions can be filtered
- Enables risk-adjusted trading strategies

**Training Target**: `confidence = 1` if prediction correct, `0` otherwise.

### 4. Why BCEWithLogitsLoss?

We use `BCEWithLogitsLoss` instead of `BCELoss` for the confidence head because:
- **Mixed precision safety**: BCE + sigmoid + autocast can produce NaN
- **Numerical stability**: LogSumExp trick is applied internally
- **No sigmoid in forward()**: Applied only at inference

### 5. Why Walk-Forward Validation?

Standard k-fold cross-validation violates time-series causality. Walk-forward:
- **Temporal order preserved**: Train on past, validate on future
- **Realistic backtesting**: Simulates actual trading conditions
- **No lookahead bias**: Model never sees future data during training

---

## Model Details

### Input Features (~25 dimensions)

| Category | Features | Count |
|----------|----------|-------|
| **Price** | Open, High, Low, Close, Volume | 5 |
| **Returns** | Daily return, Log return | 2 |
| **Momentum** | RSI(14), MFI(14) | 2 |
| **Trend** | MACD, MACD Signal, MACD Histogram | 3 |
| **Volatility** | BB Upper, BB Middle, BB Lower, BB Width, ATR(14) | 5 |
| **Moving Averages** | SMA(5), SMA(20), EMA(12), EMA(26) | 4 |
| **Sentiment** | Mean, Std, Count, Momentum | 4 |

### Model Architecture

```python
BaselineLSTM(
  (lstm): LSTM(25, 128, num_layers=2, batch_first=True, dropout=0.3)
  (fc_shared): Sequential(
    Linear(128, 64), ReLU(), Dropout(0.3)
  )
  (trend_head): Sequential(
    Linear(64, 32), ReLU(), Dropout(0.15), Linear(32, 3)
  )
  (confidence_head): Sequential(
    Linear(64, 32), ReLU(), Dropout(0.15), Linear(32, 1)
  )
)
# Total parameters: ~130,000
```

### Loss Function

```python
L_total = α * L_trend + β * L_confidence

where:
  L_trend = CrossEntropyLoss(predictions, targets)
  L_confidence = BCEWithLogitsLoss(confidence, is_correct)
  α = 1.0, β = 0.5
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-5 |
| Scheduler | Cosine Annealing with Warm Restarts |
| Batch Size | 32 |
| Epochs | 50 |
| Early Stopping | 10 epochs patience |
| Gradient Clipping | 1.0 |
| Mixed Precision | FP16 |

---

## Data Pipeline

### Data Sources

| Dataset | Description | Size |
|---------|-------------|------|
| **FNSPID** | 15.7M news + 29.7M prices (S&P 500) | 5.73 GB |
| **Twitter Financial Sentiment** | 9,543 labeled tweets | 50 MB |
| **yfinance** | Real-time price data fallback | Dynamic |

### Sentiment Extraction

```python
# Using FinBERT (ProsusAI/finbert)
extractor = SentimentExtractor()
result = extractor.extract("Apple reports record revenue")
# → SentimentResult(label='positive', score=0.944, ...)
```

### Daily Aggregation

```python
# Time-weighted average (recent news weighted higher)
aggregator = SentimentAggregator(strategy='time_weighted')
daily_sentiment = aggregator.aggregate_daily(news_df)
# → DataFrame with: date, ticker, sentiment_mean, sentiment_std, news_count
```

---

## Experiments

### Ablation Study: Sentiment Sources

```bash
python experiments/ablation_sentiment.py --scenarios all --stocks 50 --epochs 30
```

| Scenario | Description |
|----------|-------------|
| `no_sentiment` | Price + technical indicators only |
| `twitter_only` | Twitter Financial Sentiment |
| `news_only` | FNSPID news sentiment |
| `combined` | Twitter + FNSPID |

Results are saved to `reports/ablation_*/`.

---

## Results

### Expected Performance

| Scenario | Accuracy | High Conf Acc | Notes |
|----------|----------|---------------|-------|
| Price Only | ~33-35% | ~35-40% | Random baseline |
| + Sentiment | ~45-55% | ~55-65% | Academic benchmarks |

### Interpretation

- **33% accuracy** = random (3 classes)
- **50% accuracy** = meaningful signal detected
- **High confidence filtering** improves actionable accuracy

---

## References

### Datasets

1. **FNSPID**: [Zihan1004/FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID)
2. **Twitter Financial Sentiment**: [zeroshot/twitter-financial-news-sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)

### Models

1. **FinBERT**: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)

### Papers

1. Xu, Y., & Cohen, S. B. (2018). Stock Movement Prediction from Tweets and Historical Prices. ACL.
2. Liu, Z., et al. (2020). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.
3. Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
