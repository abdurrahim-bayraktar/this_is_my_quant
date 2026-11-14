# Researcher Transfer Guide
> **Complete knowledge transfer document for the Sentiment-Driven Stock Trend Prediction project**
> 
> Last Updated: 2026-02-07 | Author: Automated via AI Assistant

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Project Goal & Current State](#project-goal--current-state)
3. [Critical Lessons Learned](#critical-lessons-learned)
4. [Experiment Chronology](#experiment-chronology)
5. [Architecture Overview](#architecture-overview)
6. [Data Sources & Pipeline](#data-sources--pipeline)
7. [Feature Engineering Evolution](#feature-engineering-evolution)
8. [Key Files Reference](#key-files-reference)
9. [Reproduction Guide](#reproduction-guide)
10. [Recommended Next Steps](#recommended-next-steps)

---

## Executive Summary

This project investigates **whether financial news sentiment can improve stock trend prediction** using LSTM neural networks. After extensive experimentation:

| Metric | Result |
|--------|--------|
| **Best Honest Accuracy** | ~51% (V6, price-only, proper temporal split) |
| **Sentiment Contribution** | Minimal - near-zero predictive power for next-day returns |
| **Key Finding** | News moves markets **instantly** (same-day correlation ~0.35 for MSFT), but has no simple linear predictability for next-day |

### Key Takeaways
1. **V7's 81.45% accuracy was INVALID** - caused by data leakage (ICS_26, DPO indicators)
2. **Twitter sentiment was extracted but NEVER USED** - no timestamps in dataset
3. **Only FNSPID was used for sentiment** - FinBERT extraction on news headlines
4. **Sentiment has strong same-day correlation** but near-zero next-day predictive power

---

## Project Goal & Current State

### Original Goal
Build a sentiment-driven stock trend prediction system that:
1. Extracts sentiment from financial news using FinBERT
2. Combines sentiment with technical indicators
3. Predicts next-day price movement (Up/Down/Neutral)
4. Outputs confidence scores for each prediction

### Current State
- **Price-only baseline works** with ~51% accuracy (above 33% random chance)
- **Sentiment integration underperforms** - adds noise rather than signal
- **Pipeline is mature** - caching, temporal splitting, proper validation implemented
- **V10 is the current stable experiment** with data leakage fixes

---

## Critical Lessons Learned

### 🔴 Data Leakage Issues (CRITICAL)

#### 1. V7 Leaky Indicators
V7 achieved 81.45% accuracy - **this was fake due to data leakage**:

```python
# experiments/pooled_price_baseline_v10.py - EXCLUDED_FEATURES
EXCLUDED_FEATURES = [
    'ichimoku_ICS_26',  # Chikou Span - plots close 26 periods BACK (look-ahead!)
    'dpo',              # Detrended Price Oscillator - suspected leak
]
```

**ICS_26 (Ichimoku Chikou Span)**: This indicator plots the current close price 26 periods *back* in time. When calculating it for day T, it uses information from day T, but associates it with day T-26. This creates look-ahead bias when used in a sliding window.

#### 2. Off-by-One Error in Target Calculation
Early versions predicted T+2 instead of T+1:
```python
# WRONG (before fix)
label = labels[i + self.sequence_length]  # Predicting T+2

# CORRECT (after fix in V10)
label = labels[i + self.sequence_length - 1]  # Predicting T+1
```

#### 3. Temporal Aggregation Flaw
Sentiment was aggregated **per-ticker** instead of **per-ticker-per-day**:
```python
# WRONG - constant sentiment across all time
sentiment_agg = sentiment_df.groupby('ticker').agg(...)

# CORRECT - time-varying sentiment
sentiment_agg = sentiment_df.groupby(['ticker', 'date']).agg({
    'sentiment_value': ['mean', 'std', 'count']
})
```

### 🟡 Data Source Issues

#### Twitter Dataset: Unusable
- Extracted sentiment from `zeroshot/twitter-financial-news-sentiment` dataset
- **Problem**: No timestamps in the data
- **Result**: Cannot align to trading days, **never used in training**
- Only FNSPID `All_external.csv` was used for sentiment

#### FNSPID: Primary Sentiment Source
- Path: `C:\Users\abdurrahim\.cache\huggingface\hub\datasets--Zihan1004--FNSPID\...`
- File: `Stock_news/All_external.csv` (5.7 GB)
- Contains dates in format: `"2020-06-05 06:30:54 UTC"`

### 🟢 What Actually Worked

1. **Temporal Splitting Strategy (V10)**:
   - Train: 2014 - 2022-01-01
   - Validation: 2022-01-01 - 2023-06-01  
   - Test: 2023-06-01 - 2024-12-31
   - Each stock split by DATE, then pooled

2. **Domain Knowledge Features (when not leaky)**:
   - Golden Cross, Death Cross signals
   - RSI Overbought/Oversold  
   - MACD Bullish/Bearish signals
   - Bollinger Squeeze/Breakout

3. **Pooled Multi-Stock Training**:
   - Single-stock training fails (overfitting)
   - 500+ stocks provides enough diversity
   - Sample-to-parameter ratio should be >4x

---

## Experiment Chronology

### Phase 1: Single Stock Baseline (V1-V2)
| Version | Stocks | Accuracy | Problem |
|---------|--------|----------|---------|
| V1 | 1 (MSFT) | ~33% | Too few samples, memorization |
| V2 | 10 | ~35% | Still insufficient diversity |

### Phase 2: Multi-Stock Pooling (V3-V5)
| Version | Stocks | Features | Accuracy | Notes |
|---------|--------|----------|----------|-------|
| V3 | 200 | 73 | 48.1% | Overfitting (14% train-val gap) |
| V4 | 316 | 73 | 47.8% | Over-regularized |
| V5 | 289 | 73 | 49.1% | Sweet spot (0.4 dropout, 5e-4 LR) |

### Phase 3: Scale Up (V6)
| Version | Stocks | Samples | Accuracy | Notes |
|---------|--------|---------|----------|-------|
| V6 | 562 | 1.36M | **51.57%** | Best honest result |

### Phase 4: Domain Features (V7) - LEAKY!
| Version | Features | Accuracy | Status |
|---------|----------|----------|--------|
| V7 | 110 | 81.45% | ❌ **INVALID - Data leakage** |

### Phase 5: Data Leakage Investigation (V8-V10)
| Version | Focus | Accuracy | Status |
|---------|-------|----------|--------|
| V8 | Testing temporal split | ~49% | Better validation |
| V9 | Proper per-stock date split | ~50% | Correct methodology |
| V10 | ICS_26/DPO exclusion | ~50% | ✅ **Current stable** |

### Phase 6: Hyperparameter Tuning
Best configuration from `hyperparameter_tuning.py`:
- Dropout: 0.2
- Scheduler: ReduceLROnPlateau
- Label Smoothing: 0.1
- Confidence Weight: 0.0 (disabled)

### Phase 7: Sentiment Integration Attempts
| Experiment | Result | Finding |
|------------|--------|---------|
| Ablation Study | All ~40% | Sentiment was constant (bug) |
| Temporal Alignment Fix | ~40% | Signal too weak |
| Correlation Analysis | 0.35 same-day (MSFT) | News priced in instantly |
| Next-Day Correlation | ~0.00 | No linear predictability |

---

## Architecture Overview

### Model: Dual-Head LSTM
```
Input [batch, 20, ~108 features]
         │
         ▼
   LSTM (2 layers, 128 hidden)
         │
         ▼
    Shared Dense (64)
         │
    ┌────┴────┐
    ▼         ▼
Trend Head  Confidence Head
(3-class)   (probability)
```

### Why LSTM over Transformer?
- Sequence length is short (20 days)
- Dataset is relatively small (~1M samples)
- LSTM has 130k params vs 1M+ for Transformer
- Better sample efficiency

### Loss Function
```python
L_total = α * CrossEntropyLoss(trend) + β * BCEWithLogitsLoss(confidence)
# α = 1.0, β = 0.5 (confidence weight disabled in best config)
```

---

## Data Sources & Pipeline

### Data Sources
| Source | Usage | Notes |
|--------|-------|-------|
| **FNSPID** | Sentiment extraction | 5.7GB, 15.7M news records |
| **yfinance** | Price data | Primary, cached locally |
| **Twitter Financial** | ❌ Not used | No timestamps |

### Caching System
```
data/price_cache/
├── prices_2014-01-01_2024-12-31.pkl  # Raw OHLCV
└── (other date ranges)
```

### Feature Pipeline
1. Load prices from cache or yfinance
2. Compute 108 technical indicators (`indicators_v7.py`)
3. Exclude leaky features (ICS_26, DPO)
4. Normalize features
5. Create 20-day sequences
6. Split by DATE (not index!) per stock

---

## Feature Engineering Evolution

### Base Indicators (V3: 73 features)
- Moving Averages: SMA, EMA, VWMA, KAMA, Ichimoku
- Momentum: RSI, Stochastic, MACD, CCI, Williams %R
- Volume: MFI, OBV, Volume Ratio
- Volatility: Bollinger Bands, ATR
- Trend: ADX, Bull/Bear Power

### Domain Features (V7: +37 features)
Added signal features but introduced leakage:
- Golden/Death Cross
- RSI Overbought/Oversold (leaky if calculated wrong)
- MACD Signals
- Trend Strength, Volatility Regime

### Final Valid Feature Set (V10: ~106 features)
All V7 features minus:
- `ichimoku_ICS_26` (look-ahead bias)
- `dpo` (suspected leak)

---

## Key Files Reference

### Experiments (Priority)
| File | Purpose | Status |
|------|---------|--------|
| `pooled_price_baseline_v10.py` | **Current stable baseline** | ✅ Use this |
| `selected_stock_evaluation_v2.py` | Weekly data testing | Active |
| `hyperparameter_tuning.py` | HP search | Best config found |
| `fnspid_top40_correlation.py` | Sentiment correlation analysis | Insights |
| `validate_temporal_alignment.py` | Data leakage detection | Reference |

### Source Code
| Directory | Key Files |
|-----------|-----------|
| `src/features/` | `indicators_v7.py` (110 features) |
| `src/models/` | `baseline_lstm.py`, `simple_lstm.py` |
| `src/nlp/` | Sentiment extraction (FinBERT) |
| `src/training/` | Training loop |

### Documentation
| File | Content |
|------|---------|
| `docs/price_baseline_iterations.md` | V1-V7 evolution (note V7 is invalid) |
| `docs/NEXT_STEPS.md` | Known issues and fixes |
| `docs/DESIGN_DECISIONS.md` | Architectural rationale |
| `docs/sentiment_correlation_experiment.md` | Same-day vs next-day analysis |

---

## Reproduction Guide

### Environment Setup
```bash
# Python 3.10+, CUDA-capable GPU recommended
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### Run Current Best Experiment
```bash
# Price-only baseline with proper temporal split
python experiments/pooled_price_baseline_v10.py --epochs 100 --stocks 400
```

### Run Hyperparameter Tuning
```bash
python experiments/hyperparameter_tuning.py
```

### Generate Sentiment Correlation Report
```bash
python experiments/fnspid_top40_correlation.py
```

---

## Recommended Next Steps

### High Priority
1. **Intraday Analysis**: News has same-day effect (~0.35 correlation). Move to hourly/minute-level prediction.

2. **Event-Based Prediction**: Focus on high-news-volume days only, not every trading day.

3. **Sentiment Momentum**: Instead of single-day sentiment, use rolling 3-5 day trends.

### Medium Priority
4. **Sector-Level Sentiment**: Aggregate by sector (tech stocks influenced by AAPL/MSFT news).

5. **Alternative Sentiment Sources**: Reddit/WallStreetBets, earnings call transcripts, SEC filings.

6. **Ensemble Methods**: Combine multiple models with different timeframes.

### Low Priority
7. **Model Architecture**: Try Transformer with attention over news events (not time steps).

8. **Meta-Learning**: Train model to predict when sentiment will be informative.

---

## Appendix: Signal Quality Findings

### MSFT Correlation Analysis
```
Same-Day Correlation:  0.347 (strong)
Next-Day Correlation: -0.003 (none)
Observations: 406 days with news
News Count: 3,367 articles
```

**Interpretation**: News is priced in immediately. By the time daily aggregation is complete, the market has already reacted. Simple "buy if positive sentiment yesterday" strategies will fail.

### Class Distribution (Typical)
- Down (<-0.5%): ~28%
- Neutral (±0.5%): ~44%
- Up (>+0.5%): ~28%

This balanced distribution enables fair training without class weighting.

---

## Contact & Resources

- **Repository**: `c:\dev\this_is_my_quant`
- **Primary Data Cache**: `data/price_cache/`
- **Reports**: `reports/` (79 experiment runs archived)
- **FNSPID Local Path**: See `docs/NEXT_STEPS.md` for exact location

---

*This guide was generated to facilitate knowledge transfer. For questions not covered here, refer to the individual documentation files in `docs/` or examine the experiment scripts directly.*
