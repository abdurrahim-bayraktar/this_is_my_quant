# Diagnostic Test Results

> Generated: 2026-01-22 14:02:02

---

## Summary

| Test | Status | Description |
|------|--------|-------------|
| Overfitting Sanity Check | ✅ PASS | Tests if model can memorize synthetic data (no bug in architecture) |
| Signal Quality Analysis | ✅ PASS | Measures correlation between sentiment and next-day returns |

---

## Detailed Results

### Overfitting Sanity Check

**Status**: PASS

#### Output

```
============================================================
OVERFITTING SANITY CHECK
============================================================
If model CAN'T reach ~100% train accuracy, there's a BUG.
============================================================

============================================================
OVERFITTING TEST RESULTS
============================================================
Train Accuracy: 100.00%
[PASS] Model CAN overfit - no obvious bug in architecture
   The issue is likely weak signal in real data
============================================================

```

---

### Signal Quality Analysis

**Status**: PASS

#### Output

```
============================================================
SIGNAL QUALITY TEST: Sentiment-Return Correlation
============================================================

============================================================
SIGNAL QUALITY RESULTS
============================================================

Overall Correlation: -0.0064
P-value: 0.7779
Samples: 1951
Statistically Significant: NO

Per-Ticker Correlations:
  TSLA: +0.0250 (p=0.707) 
  AAPL: -0.2297 (p=0.080) 
  NVDA: +0.0298 (p=0.318) 
  AMD: -0.2524 (p=0.015) [SIG]
  AMZN: -0.2720 (p=0.179) 
  GOOGL: -0.0380 (p=0.436) 

Returns by Sentiment Strength:
  High Sentiment (>0.2):  +0.2855% (n=610)
  Low Sentiment (<-0.2):  +0.2149% (n=445)
  Neutral Sentiment:      +0.1821% (n=896)

============================================================
INTERPRETATION
============================================================
[FAIL] Correlation is essentially ZERO.
   Sentiment has NO predictive power for next-day returns.
   Consider: longer prediction horizon, different features, or 
   accepting that daily prediction is near-random for this data.
============================================================

Detailed results saved to: C:\dev\this_is_my_quant\reports\signal_quality_20260122_140201

```

---
