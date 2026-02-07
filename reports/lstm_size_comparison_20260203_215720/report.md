# LSTM Size Comparison Experiment Report

## Executive Summary

This experiment compared LSTM models of different sizes for stock trend prediction (3-class: Down/Neutral/Up) using 400 stocks with proper temporal splits. **All configurations performed similarly**, with accuracy ranging from 41.3% to 41.7% and accuracy lift over baseline between 3.4% and 3.8%. This suggests the current bottleneck is **not model capacity** but rather the inherent difficulty of the prediction task and/or feature quality.

---

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| **Stocks** | 400 (from ranked_tickers.py) |
| **Train Period** | 2014-01-01 to 2022-01-01 |
| **Validation Period** | 2022-01-01 to 2023-06-01 |
| **Test Period** | 2023-06-01 to 2024-12-31 |
| **Features** | 108 technical indicators |
| **Sequence Length** | 20 trading days |
| **Dropout** | 0.4 |
| **Learning Rate** | 5e-4 |

### Data Leakage Prevention (from V10)
- Excluded `ichimoku_ICS_26` (Chikou Span - look-ahead bias)
- Excluded `dpo` (Detrended Price Oscillator)
- Scaler fit on training data only
- Correct T+1 prediction target alignment

---

## Results

### Model Comparison Table

| Config | Hidden | Layers | Parameters | Samples/Param | **Accuracy** | **Lift** | F1 (macro) | MCC |
|--------|--------|--------|------------|---------------|--------------|----------|------------|------|
| **Medium-2L** | 64 | 2 | 103K | 5.74x | **41.68%** | **+3.80%** | 0.332 | 0.119 |
| Small | 32 | 1 | 29K | 20.48x | 41.46% | +3.57% | 0.350 | 0.114 |
| Large (v10) | 128 | 2 | 333K | 1.78x | 41.42% | +3.54% | 0.368 | 0.107 |
| Medium-1L | 64 | 1 | 70K | 8.48x | 41.29% | +3.41% | 0.364 | 0.111 |

> **Zero-Rule Baseline**: 37.89% (always predicting the most common class)

### Per-Class Accuracy

| Config | Class 0 (Down) | Class 1 (Neutral) | Class 2 (Up) |
|--------|----------------|-------------------|--------------|
| Large | 12.0% | 36.3% | 71.2% |
| Medium-2L | 2.2% | 38.8% | **78.5%** |
| Medium-1L | 9.0% | **44.9%** | 67.0% |
| Small | 5.7% | 43.1% | 71.7% |

---

## Key Findings

### 1. Model Size Has Minimal Impact

All four configurations achieved nearly identical overall accuracy (~41.3-41.7%). This strongly suggests:

- **The model is not underfitting** due to capacity constraints
- **The bottleneck is elsewhere** (feature quality, signal-to-noise ratio, or task difficulty)
- The current architecture is already sufficient for capturing learnable patterns

### 2. Samples-per-Parameter Ratio Analysis

| Config | Samples/Param | Risk Level |
|--------|---------------|------------|
| Large | 1.78x | ⚠️ High overfitting risk |
| Medium-2L | 5.74x | ✅ Reasonable |
| Medium-1L | 8.48x | ✅ Good |
| Small | 20.48x | ✅ Well-regularized |

Despite having only 1.78x samples-per-parameter, the Large model did not significantly outperform smaller models, indicating:
- Dropout (0.4) and early stopping effectively prevent overfitting
- More parameters don't help when the underlying signal is weak

### 3. Class Imbalance Trade-offs

Different model sizes exhibit different class prediction biases:
- **Medium-2L** strongly favors "Up" (78.5%) at the expense of "Down" (2.2%)
- **Medium-1L** is most balanced across classes (9%, 45%, 67%)
- The neutral class remains hardest to predict accurately for all models

### 4. Depth vs. Width Trade-off

Comparing single-layer vs. two-layer models:
- **Medium-2L vs Medium-1L**: 2-layer marginally better (+0.39% accuracy)
- **Large-2L vs Small-1L**: Almost identical (+0.04% difference)

Additional layers provide minimal benefit for this task.

---

## Recommendations for Future Experiments

### A. Architecture Changes

1. **Transformer-based Models**
   - Replace LSTM attention with full self-attention
   - Temporal Convolutional Networks (TCN) for parallel processing
   - Informer or Autoformer for long-range dependencies

2. **Ensemble Methods**
   - Train multiple small models and aggregate predictions
   - Different random seeds, different feature subsets
   - Voting or stacking for final prediction

3. **Hybrid Architectures**
   - CNN for feature extraction + LSTM for temporal patterns
   - Wavelet decomposition + LSTM for multi-scale analysis

### B. Feature Engineering

1. **Feature Selection/Reduction**
   - Current 108 features may include noise
   - Use mutual information or SHAP values to identify top predictors
   - Reduce to 20-30 most informative features

2. **Alternative Features**
   - Order flow imbalance, bid-ask spread dynamics
   - Cross-sectional momentum (relative strength vs. market)
   - Volatility regime indicators (VIX-based)
   - Sector rotation signals

3. **Feature Interactions**
   - Explicit interaction terms (RSI × Volume, MACD × Volatility)
   - Polynomial features for non-linear relationships

### C. Single-Stock Saturation Experiment

**Question**: Can we saturate the smallest model (32 hidden, 1 layer, ~29K params) with single-stock data?

| Scenario | Training Samples | Samples/Param | Expected Outcome |
|----------|------------------|---------------|------------------|
| Single Stock (~2000 samples) | ~1,600 train | 0.055x | ⚠️ Severe overfitting |
| 5 Stocks | ~8,000 train | 0.28x | ⚠️ Likely overfitting |
| 20 Stocks | ~32,000 train | 1.1x | Borderline |
| 50 Stocks | ~80,000 train | 2.8x | ✅ Minimum viable |

**Recommended Experiment**:
```bash
# Test model saturation on single stock with extreme regularization
python experiments/single_stock_saturation.py \
    --ticker MSFT \
    --hidden-size 32 \
    --num-layers 1 \
    --dropout 0.6 \
    --epochs 200
```

**Hypothesis**: Single-stock training will show:
- Training accuracy >> Test accuracy (overfitting)
- Or near-random test performance (no learnable pattern for that stock)

### D. Training Strategy Improvements

1. **Class Weighting**
   - Apply weights inversely proportional to class frequency
   - Focus on improving Down/Neutral class prediction

2. **Focal Loss**
   - Down-weight easy examples (confident Up predictions)
   - Focus learning on hard examples

3. **Label Smoothing**
   - Reduce confidence in labels (±0.5% threshold may be too sharp)
   - Try ±1% or ±0.3% thresholds

4. **Walk-Forward Validation**
   - Instead of single temporal split, use rolling validation
   - Better estimate of true out-of-sample performance

---

## Conclusion

The LSTM size comparison reveals that **model capacity is not the limiting factor** for stock trend prediction performance. All configurations from 29K to 333K parameters achieve similar results (~41.5% accuracy, +3.5% lift over baseline).

**Key Insight**: The marginal improvement from larger models suggests we should focus on:
1. **Better features** rather than bigger models
2. **Alternative architectures** that may capture patterns LSTMs miss
3. **Single-stock experiments** to understand per-stock predictability
4. **Ensemble methods** to improve robustness

The **Medium-2L configuration (64 hidden, 2 layers, 103K params)** offers the best balance of performance and efficiency, and is recommended for production use.

---

## Appendix: Implementation Details

### Preserved V10 Fixes

1. **Temporal Alignment**: Split by date, not by index
   - Train: 2014-01-01 to 2022-01-01
   - Val: 2022-01-01 to 2023-06-01
   - Test: 2023-06-01 to 2024-12-31

2. **Data Leakage Prevention**: 
   - Excluded ICS_26 (plots current close 26 periods back)
   - Excluded DPO (contains future information in calculation)

3. **Correct T+1 Prediction**:
   ```python
   # Target: return_next = Close.pct_change().shift(-1)
   # Label for sequence ending at day T: labels[i + sequence_length - 1]
   ```

4. **Scaler Fit on Train Only**: StandardScaler fitted on training data, applied to val/test

### Files Generated

- `experiments/lstm_size_comparison.py` - Experiment script
- `reports/lstm_size_comparison_20260203_215720/results.csv` - Full results
- `reports/lstm_size_comparison_20260203_215720/features.txt` - 108 features used
