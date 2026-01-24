# Troubleshooting & Next Steps: Fixing Model Performance

> **Status**: Root cause identified—sentiment aggregated per-ticker not per-day. Implementation in progress.
> **Date**: 2026-01-22 (Updated)

---

## 📁 FNSPID Local Data Reference

The FNSPID dataset is available locally at:
```
C:\Users\abdurrahim\.cache\huggingface\hub\datasets--Zihan1004--FNSPID\snapshots\bf9189c41527198897d1af3e17b1a0095279fc45\
```

### Data Structure

| File | Size | Description |
|------|------|-------------|
| `Stock_news/All_external.csv` | 5.7 GB | Financial news with dates (RECOMMENDED) |
| `Stock_news/nasdaq_exteral_data.csv` | 23 GB | Extended NASDAQ news dataset |
| `Stock_price/full_history.zip` | 590 MB | Historical stock prices |

### CSV Columns (`All_external.csv`)
- `Date` - UTC timestamp (e.g., "2020-06-05 06:30:54 UTC")
- `Article_title` - News headline
- `Stock_symbol` - Ticker symbol (e.g., AAPL)
- `Summary`, `Lsa_summary`, `Luhn_summary`, `Textrank_summary`, `Lexrank_summary` - Various summaries

---

## ⛔ Identified Issues

### 1. ~~FNSPID Loading Failure~~ ✅ RESOLVED
**Resolution**: Local CSV files are available. Use `All_external.csv` (5.7GB) for faster loading.

### 2. The "Zero Sentiment" Problem (Ticker Mismatch)
**Observation**: We selected `TOP_200_TICKERS` first, then tried to attach sentiment.
- **Problem**: Many of the top 200 stocks may NOT be in the Twitter/StockNet datasets.
- **Result**: If 40 of 50 stocks have no news, their sentiment features are `0.0` (Neutral). The model learns that **sentiment feature = 0.0 (constant)** and ignores it entirely.
- **Fix**: We must **invert** the loading logic.
  - *Current*: `Select Tickers -> Get Data`
  - *Required*: `Get Sentiment Data -> Get Tickers that have enough data -> Get Prices for those Tickers`

### 3. **CRITICAL: Temporal Aggregation Flaw** 🔴
**Root Cause of ~40% Accuracy**: The ablation study aggregates sentiment **per-ticker only**, not per-day.
- **Current Code**: `sentiment_df.groupby('ticker').agg(...)` → One value per stock forever
- **Required**: `sentiment_df.groupby(['ticker', 'date']).agg(...)` → Daily varying sentiment
- **Result**: Sentiment is constant across time; model cannot learn sentiment→price relationships.

### 4. Training Instability
**Observation**: `Train_loss` flat, `val_accuracy` random.
- **Cause**: "Vanishing Signal". The useful signal (days with strong sentiment) is drowned out by noise (days with no news or 0 sentiment).
- **Likely Signal-to-Noise Ratio**: < 1%.

---

## 🛠️ Action Plan (Prioritized)

### ⚡ Priority 1: Fix Temporal Aggregation (CRITICAL)

**Status**: 🔄 In Progress

**Problem**: Sentiment is aggregated per-ticker, making it constant across time.

**Fix in `experiments/ablation_sentiment.py` and `feature_engineering.py`:**
```python
# BEFORE (broken):
sentiment_agg = sentiment_df.groupby('ticker').agg(...)

# AFTER (correct):
sentiment_agg = sentiment_df.groupby(['ticker', 'date']).agg({
    'sentiment_value': ['mean', 'count']
})
```

**Add `has_news` feature:**
```python
df['has_news'] = (df['news_count'] > 0).astype(int)
```

---

### ⚡ Priority 2: Create 5-Stock Validation Test

**Status**: 🔄 In Progress

**Test Stocks**: TSLA, AAPL, NVDA, AMD, AMZN (highly-discussed)

**Script**: `experiments/validate_temporal_alignment.py`
- Load FNSPID `All_external.csv` for 5 stocks
- Run FinBERT on GPU
- Aggregate sentiment per-day
- Train 15 epochs
- Compare price-only vs sentiment-enhanced

**Success Criteria**:
- Sentiment variance > 0.1 (not constant)
- Accuracy differs from baseline

---

### Priority 3: Dynamic Universe Selection (Partially Done ✅)

Already implemented in `load_data_with_sentiment_intersection()` in `dataset_loader.py`.

---

### Priority 4: Enhance Model Training (Deferred)

**Curriculum Learning**:
- Epoch 1-5: Train on high-sentiment days only (`abs(sentiment) > 0.5`)
- Epoch 6+: Fine-tune on all days

**Model Simplification**:
- Reduce to 1-layer LSTM or GRU
- Increase batch size (32 → 64)

---

## 📊 Ablation Results (Pre-Fix)

```
                    scenario  accuracy  high_conf_accuracy  high_conf_ratio
   No Sentiment (Price Only)    40.14%              50.33%            2.74%
      Twitter Sentiment Only    39.68%              38.71%            0.18%
News Sentiment Only (FNSPID)    39.89%              50.45%            3.93%
   Combined (Twitter + News)    40.24%              51.52%            1.95%
```

**Interpretation**: All ~40% because sentiment was constant across time.

---

## ✅ Changelog

| Date | Change |
|------|--------|
| 2026-01-22 | Added FNSPID local path and data structure documentation |
| 2026-01-22 | Identified root cause: temporal aggregation flaw |
| 2026-01-22 | Prioritized action plan with 5-stock validation |
| 2026-01-21 | Initial document created |

