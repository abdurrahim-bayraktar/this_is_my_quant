# Critical Analysis & Strategic Next Steps
**Project**: Sentiment-Driven Stock Trend Prediction
**Date**: March 22, 2026
**Framework**: Analytical and Critical Review

---

## 1. Executive Diagnostics & Current State Assessment

The project has reached a critical inflection point. The transition from the wildly inflated results of V7 (81.45% accuracy) back to a mathematically sound baseline in V10 (~50-51%) underscores the paramount importance of rigorous validation in quantitative finance.

### 1.1 The Anatomy of Data Leakage (V6 & V7)
The project suffered from two distinct, catastrophic forms of data leakage during its experimentation phase:

1. **Temporal Bleed (V6 Leakage)**: 
   - **Mechanism**: In `pooled_price_baseline_v6.py`, the dataset of 800 stocks was compiled into a single massive array `X`, and then subjected to `np.random.permutation(len(X))` *before* being split into 80/10/10 train/val/test sets. 
   - **Impact**: This destroys the time-series integrity. The training set inadvertently contained market regimes and price action from 2023/2024, which the model then used to "predict" validation/test samples from 2015. 
   - **Correction**: Assumed "honest" results from V6 were contaminated. True cross-validation in time-series requires strict chronological splitting (`target_date < cutoff_date`), which was successfully implemented in V9/V10.

2. **Feature Look-Ahead (V7 Leakage)**:
   - **Mechanism**: Inclusion of indicators like `ichimoku_ICS_26` plotting current data 26 periods backwards, and `dpo` (Detrended Price Oscillator).
   - **Impact**: Allowed the model to peak into the future perfectly, yielding the impossible 81.45% accuracy.
   - **Correction**: Cleaned in V10 by selectively stripping look-ahead features from the dataset.

### 1.2 The Price-Only Baseline Reality (V10)
We are currently polishing the price-only model (V10). With data leakage effectively neutralized, the model demonstrates a ~50-51% categorical accuracy (Up/Down/Neutral). 
- **Contextualizing Performance**: While 51% seems marginally above the random baseline of 33%, in quantitative finance, a consistent 51% hit rate with favorable asymmetric risk/reward is highly tradeable. 
- **Architecture**: The dual-head AttentionLSTM is well-suited for sequence data of length 20. Hyperparameters have been somewhat optimized (Dropout: 0.2, LR Scheduler: Plateau).

### 1.3 The Sentiment Failure
The attempt to integrate sentiment (FNSPID) suffered from fundamental data misalignment (per-ticker vs. per-day aggregation) and poor temporal granularity. Furthermore, the correlation analysis (0.35 same-day vs -0.003 next-day) conclusively proves that daily-aggregated news sentiment has zero linear predictive power for T+1 closes. The market fully prices in news on the same day.

---

## 2. Strategic Next Steps: An Analytical Framework

To transition from a "working baseline" to an "alpha-generating system," the project must pivot. The following steps employ a critical framework prioritizing structural improvements over superficial parameter tweaking.

### Phase 1: Perfecting the Price-Only Model (Short-Term)
*Goal: Extract maximum signal from price action before introducing external noise.*

1. **Class Imbalance & Threshold Optimization**:
   - *Analysis*: The labels are currently rigidly binned `[-np.inf, -0.005, 0.005, np.inf]`. This assumes symmetric volatility across all stocks, which is empirically false (e.g., AAPL volatility != TSLA volatility).
   - *Action*: Implement dynamic, ATR-based (Average True Range) binning per stock. A "neutral" day for a low-beta stock might be $\pm 0.2\%$, while for a high-beta stock it might be $\pm 1.5\%$.
2. **Confidence Head Calibration**:
   - *Analysis*: The `confidence_weight` was disabled (0.0). We are throwing away the second head of our dual-head architecture.
   - *Action*: Re-enable BCEWithLogitsLoss for the confidence head, training it to identify samples where the trend prediction has a high likelihood of being correct. We can then execute trades *only* on the top decile of confidence scores.
3. **Walk-Forward Validation**:
   - *Analysis*: A static train/val/test split (2014-2022, 2022-2023, 2023-2024) is vulnerable to regime shifts (e.g., 2022 bear market, 2023 AI boom).
   - *Action*: Implement expanding window walk-forward validation. Train on Yr 1-5, Test on Yr 6; Train on Yr 2-6, Test on Yr 7, etc.

### Phase 2: Resolving the Sentiment Paradox (Medium-Term)
*Goal: Re-integrate sentiment data by aligning the model with the realities of market microstructure.*

1. **Shift to Intraday/High-Frequency Horizons**:
   - *Analysis*: The 0.35 same-day correlation proves the signal exists, but the daily timeframe is too slow.
   - *Action*: Shift the prediction horizon from Daily OHLCV to Hourly or 15-Minute intervals. If news breaks at 10:00 AM, the model must predict the 10:15 AM to 11:00 AM drift.
2. **Event-Driven Filtering (Sparsity Modeling)**:
   - *Analysis*: The baseline forces a prediction every day. Sentiment is mostly noise (0.0) on 80% of days.
   - *Action*: Restrict the neural network to only execute on "Event Days" (days with `news_count > threshold` or absolute `sentiment > 0.6`).
3. **Sentiment Momentum Vectors**:
   - *Analysis*: Instead of raw sentiment values, capture the *derivative* of sentiment. Is sentiment accelerating positively?
   - *Action*: Calculate 1-day, 3-day, and 5-day EMA of sentiment scores. Feature: `Sentiment_Divergence = Price_Trend - Sentiment_Trend`.

### Phase 3: Alternative Data and Meta-Learning (Long-Term)
*Goal: Build an institutional-grade, multi-modal forecasting engine.*

1. **Cross-Asset/Sector Interconnectedness**:
   - *Analysis*: Individual stock prediction ignores sector-wide liquidity flows.
   - *Action*: Introduce sector-aggregated sentiment and price momentum as features (e.g., XLK momentum for tech stocks). Implement Graph Neural Networks (GNNs) where nodes are tickers weighted by sentiment correlation.
2. **Regime-Switching Meta-Model**:
   - *Analysis*: The LSTM learns an 'average' behavior across 10 years, blending low-volatility regimes with crash regimes.
   - *Action*: Train an unguided Hidden Markov Model (HMM) to classify the current market regime (Bull/Bear/Crab). Feed the regime state as a one-hot feature into the LSTM, allowing it to dynamically adjust its weighting of technical vs. sentiment features.

---

## 3. Conclusion
The rollback from V7 to V10 was painful but necessary; the foundation is now mathematically sound. V6's temporal data leak highlights why random shuffling is toxic to financial ML. By calibrating the price-only model via ATR-binning and confidence thresholds, and strictly realigning sentiment integration to intraday or event-driven horizons, we have a clear, rigorous pathway to positive alpha.
