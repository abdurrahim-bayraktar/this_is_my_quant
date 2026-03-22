Audit Results — All Clear ✅
Bug 1: V7 Leaky Indicators (ichimoku_ICS_26, dpo)
✅ Safe. The experiment calls compute_shap_top20() — NOT compute_all(). I verified:

compute_shap_top20() (lines 384-495 of indicators_v7.py) computes only these indicators: atr, bbands, cci, rsi, roc, adx, aroon, trix, ema, obv, ad, pvo + 5 price features
No ichimoku, no dpo — those are only in compute_all() which is never called
Zero grep hits for ichimoku or dpo in ultimate_model.py


Bug 2: Off-by-One Target (labels[i + seq_length] vs labels[i + seq_length - 1])
✅ Correct. create_sequences() in src/data/utils.py line 171:

python
label = labels[i + seq_length - 1]  # ← Correct: predicting T+1, not T+2
This matches the V10 fix. Both the shared utility and the experiment use create_sequences(), so there's only one code path.

Bug 3: Sentiment Temporal Aggregation
✅ Not applicable. Zero grep hits for sentiment in ultimate_model.py. No sentiment data is loaded, computed, or used anywhere in the new code. This is a price-only model.

Additional checks:
Temporal split integrity: Data split uses chronological cutoffs (train < 2022-01-01, val < 2023-06-01, test >= 2023-06-01) with target_date comparison — no random shuffling
Scaler leakage: StandardScaler.fit_transform() is called only on X_train, then .transform() on val/test ✅
Train/test contamination: Test stocks are explicitly excluded from the training pool (self.train_stocks = [t for t in self.train_stocks if t not in self.test_stocks]) ✅