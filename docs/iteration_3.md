# Iteration 3: Improving Sentiment Signal

**Date**: 2026-01-22  
**Status**: Proposed

---

## Current State

- **Model**: Works correctly (100% overfit on synthetic data)
- **Temporal alignment**: Fixed (sentiment aggregated per-day, not per-ticker)
- **Accuracy**: ~40% (baseline random = 33%)
- **Conclusion**: Weak signal in sentiment→price relationship

---

## Root Causes of Weak Signal

### 1. Market Efficiency
News is priced in quickly—by the time we aggregate daily, the market has already reacted.

### 2. Aggregation Dilutes Signal
Daily mean sentiment loses extreme events (e.g., one very negative headline averaged with neutral ones).

### 3. Lag Between News and Price
Next-day prediction may be too late; intraday or same-day reaction is stronger.

### 4. Noise in Sentiment Data
FinBERT trained on generic financial text—may not capture stock-specific nuance.

---

## Proposed Fixes (Priority Order)

### ⚡ Priority 1: Event-Based Features (High Impact)

Instead of daily mean sentiment, use:
```python
# Sentiment extremes
df['sentiment_max'] = news.groupby(['ticker', 'date'])['sentiment_value'].max()
df['sentiment_min'] = news.groupby(['ticker', 'date'])['sentiment_value'].min()
df['sentiment_range'] = df['sentiment_max'] - df['sentiment_min']

# Spike detection
df['has_extreme_news'] = (abs(df['sentiment_value']) > 0.7).astype(int)
```

### ⚡ Priority 2: Curriculum Learning

Train on high-signal days first:
```python
# Phase 1: Train only on days with |sentiment| > 0.5
high_sentiment_mask = abs(X[:, -1, sentiment_col]) > 0.5
trainer.train(X[high_sentiment_mask], y[high_sentiment_mask], epochs=10)

# Phase 2: Fine-tune on all data
trainer.train(X, y, epochs=20)
```

### ⚡ Priority 3: Shorter Prediction Horizon

Switch from next-day to same-day prediction:
- Use morning sentiment → predict close direction
- Requires intraday price data

### Priority 4: Attention Mechanism

Add attention over the sentiment sequence:
```python
class SentimentAttention(nn.Module):
    def __init__(self, hidden_size):
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4)
    
    def forward(self, sentiment_features):
        # Attend to important days in the lookback window
        attended, weights = self.attention(sentiment_features, sentiment_features, sentiment_features)
        return attended
```

### Priority 5: Sentiment Momentum

Add rate-of-change features:
```python
df['sentiment_momentum'] = df['sentiment_mean'].diff(periods=3)
df['sentiment_acceleration'] = df['sentiment_momentum'].diff(periods=1)
```

### Priority 6: Sector-Specific Sentiment

Aggregate sentiment by sector, not just individual stock:
- Tech stocks influenced by AAPL/MSFT sentiment
- Bank stocks influenced by JPM/GS sentiment

### Priority 7: Alternative Sentiment Sources

- **Twitter API** (real-time social sentiment)
- **Reddit/WallStreetBets** (retail sentiment)
- **Earnings call transcripts** (management tone)
- **SEC filings** (10-K/10-Q sentiment)

---

## Experiments to Run

| Experiment | Description | Expected Improvement |
|------------|-------------|---------------------|
| A | Add sentiment_max/min/range | +2-5% accuracy |
| B | Curriculum learning | +3-5% accuracy |
| C | Same-day prediction | +5-10% accuracy |
| D | Attention mechanism | +2-4% accuracy |
| E | Sector aggregation | +1-3% accuracy |

---

## Success Criteria

- **Minimum**: Accuracy > 45% (significant above 40% baseline)
- **Target**: Accuracy > 50% (actionable for trading)
- **Stretch**: High-confidence accuracy > 60%

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/data/feature_engineering.py` | Add extreme sentiment features |
| `src/training/trainer.py` | Add curriculum learning |
| `src/models/baseline_lstm.py` | Add attention layer |
| `experiments/ablation_sentiment.py` | Add new scenarios |

---

## Next Steps

1. Implement Priority 1 (event-based features)
2. Re-run 5-stock validation
3. If improved, scale to full dataset
4. If not improved, try Priority 2 (curriculum learning)
