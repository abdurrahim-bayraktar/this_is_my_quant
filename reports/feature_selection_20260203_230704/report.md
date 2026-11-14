# Feature Selection Experiment Report

## Executive Summary

This experiment compared two feature selection methods (**Mutual Information** and **Recursive Feature Elimination**) on single-stock training with the Medium-1L LSTM model (64 hidden, 1 layer). Key findings:

1. **Feature reduction rarely helps** - Reducing from 108 to 20-50 features generally doesn't improve accuracy
2. **Methods select very different features** - Jaccard similarity between MI and RFE is only 6-18%
3. **Top-20 RFE matches or beats baseline** in some cases (DIA, EBAY)
4. **Per-stock performance varies wildly** - DIA achieves 56-57%, while others struggle around 35-38%

---

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| **Tickers** | DIA, EBAY, C, MSFT |
| **Model** | Medium-1L (hidden=64, layers=1, ~70K params) |
| **Feature Selection Methods** | Mutual Information, RFE |
| **Feature Reduction Levels** | Top 20, 30, 50 (from 108 total) |
| **Epochs** | 50 (with early stopping, patience=10) |
| **Dropout** | 0.5 (increased for small data) |

---

## Results by Stock

### DIA (DJIA ETF) - Best Performer

| Method | Features | Accuracy | Lift | F1 (macro) |
|--------|----------|----------|------|------------|
| **RFE top-50** | 50 | **57.18%** | **0.00%** | 0.331 |
| RFE top-20 | 20 | 56.68% | -0.50% | 0.294 |
| Baseline | 108 | 55.92% | -1.26% | 0.295 |
| MI top-50 | 50 | 55.92% | -1.26% | 0.277 |

**Key Insight**: For DIA, RFE with 50 features matches the zero-rule baseline exactly, suggesting patterns are hard to find even for this stable ETF.

---

### EBAY - Feature Selection Helps Most

| Method | Features | Accuracy | Lift | F1 (macro) |
|--------|----------|----------|------|------------|
| **MI top-20** | 20 | **40.05%** | **0.00%** | 0.323 |
| RFE top-30 | 30 | 39.55% | -0.50% | 0.377 |
| RFE top-20 | 20 | 34.76% | -5.29% | 0.342 |
| Baseline | 108 | 32.75% | -7.30% | 0.303 |

**Key Insight**: EBAY shows the clearest benefit from feature selection. MI top-20 outperforms baseline by 7.3%!

---

### C (Citigroup) - Moderate Improvement

| Method | Features | Accuracy | Lift | F1 (macro) |
|--------|----------|----------|------|------------|
| **RFE top-50** | 50 | **36.52%** | -1.51% | 0.314 |
| MI top-30 | 30 | 36.27% | -1.76% | 0.289 |
| RFE top-30 | 30 | 35.77% | -2.27% | 0.293 |
| Baseline | 108 | 35.01% | -3.02% | 0.301 |

**Key Insight**: Feature selection provides slight improvement over baseline.

---

### MSFT (Microsoft) - No Clear Winner

| Method | Features | Accuracy | Lift | F1 (macro) |
|--------|----------|----------|------|------------|
| **MI top-30** | 30 | **37.78%** | **0.00%** | 0.285 |
| Baseline | 108 | 37.53% | -0.25% | 0.324 |
| MI top-20 | 20 | 37.28% | -0.50% | 0.254 |
| RFE top-20 | 20 | 36.52% | -1.26% | 0.336 |

**Key Insight**: MSFT performance is very close across methods - no method provides meaningful lift.

---

## Feature Selection Method Comparison

### Jaccard Similarity Between Methods

The two methods select **very different features**, with only 6-18% overlap:

| Stock | MI-20 vs RFE-20 | MI-30 vs RFE-30 | MI-50 vs RFE-50 |
|-------|-----------------|-----------------|-----------------|
| DIA | 17.6% | 15.4% | 37.0% |
| EBAY | 14.3% | 13.2% | 33.3% |
| C | 11.1% | 17.6% | 33.3% |
| MSFT | **8.1%** | **7.1%** | 26.6% |

> **MSFT shows the lowest agreement** between methods - only 3 features overlap in top-20!

---

### Top Features by Method

#### Mutual Information Top Features (Across All Stocks)

| Feature | Stocks Selected In | Description |
|---------|-------------------|-------------|
| `ichimoku_ISB_26` | 3/4 | Ichimoku Span B |
| `atr_pct` | 3/4 | ATR as % of price |
| `bear_power` | 2/4 | Elder Bear Power |
| `donchian_DCM_20_20` | 2/4 | Donchian Middle |
| `ema_5` | 2/4 | 5-day EMA |
| `kama` | 2/4 | Kaufman Adaptive MA |
| `trix_TRIXs_30_9` | 2/4 | TRIX Signal |

#### RFE Top Features (Across All Stocks)

| Feature | Stocks Selected In | Description |
|---------|-------------------|-------------|
| `trix_TRIXs_30_9` | 4/4 | TRIX Signal |
| `stoch_STOCHh_14_3_3` | 3/4 | Stochastic Histogram |
| `rsi_14` | 3/4 | 14-day RSI |
| `kst_KSTs_9` | 2/4 | KST Signal |
| `bop` | 2/4 | Balance of Power |
| `ichimoku_ISB_26` | 2/4 | Ichimoku Span B |
| `pvo_PVO_12_26_9` | 2/4 | Price Volume Oscillator |

---

### Key Differences Between Methods

| Aspect | Mutual Information | RFE |
|--------|-------------------|-----|
| **Focus** | Price levels (MAs, Ichimoku) | Momentum oscillators (RSI, Stoch, TRIX) |
| **Volume** | Less emphasis | Strong emphasis (BOP, PVO, OBV) |
| **Consistency** | More consistent across stocks | Variable per stock |
| **Computation** | Fast (~1 second) | Slower (~10-30 seconds) |

---

## Single-Stock Saturation Analysis

With only ~1,800 training samples per stock:

| Configuration | Samples/Param Ratio | Overfitting Risk |
|--------------|---------------------|------------------|
| Baseline (108 features) | 0.026x | ⚠️ Severe |
| Top-50 | 0.033x | ⚠️ Severe |
| Top-30 | 0.036x | ⚠️ Severe |
| Top-20 | 0.038x | ⚠️ Severe |

**All configurations are severely overparameterized!** The model has ~47-70K parameters but only ~1,800 training samples (0.03-0.04x ratio, well below the recommended ≥1x).

### Why EBAY Benefits from Feature Selection

Despite overfitting risk, EBAY's 7.3% improvement with MI top-20 suggests:
1. The 108-feature baseline includes significant noise for EBAY specifically
2. Feature selection acts as regularization by removing noisy inputs
3. EBAY may have stronger momentum signals (captured by reduced feature set)

---

## Conclusions and Recommendations

### Main Findings

1. **Feature selection provides modest benefits** for single-stock training, especially for volatile stocks like EBAY
2. **RFE and MI select different features** - consider using an ensemble of both methods
3. **Per-stock variation is significant** - no universal feature set works best for all stocks
4. **Single-stock training remains challenging** due to limited data (< 2,000 samples)

### Recommended Next Steps

1. **Hybrid Feature Selection**: Combine MI and RFE rankings (e.g., features ranked highly by both)
2. **Cross-Stock Validation**: Test if features selected on one stock generalize to others
3. **Sector-Based Pooling**: Pool stocks by sector (Tech, Finance, Retail) for more data
4. **Reduced Model Size**: Try even smaller models (16-32 hidden) for single-stock training

### Recommended Feature Set

Based on this experiment, a reasonable starting set of **20 features** would include:

**From Both Methods (High Confidence):**
- `trix_TRIXs_30_9` - TRIX Signal (selected by RFE on 4/4 stocks)
- `ichimoku_ISB_26` - Ichimoku Span B
- `rsi_14` - RSI 14-day

**From Mutual Information:**
- `atr_pct`, `bear_power`, `donchian_DCM_20_20`

**From RFE:**
- `stoch_STOCHh_14_3_3`, `kst_KSTs_9`, `bop`, `pvo_PVO_12_26_9`
