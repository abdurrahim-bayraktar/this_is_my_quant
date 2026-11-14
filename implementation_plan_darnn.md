# DA-RNN Model & Experiment (Revised v3)

Implement a **Dual-Stage Attention-Based Recurrent Neural Network** (Qin et al., 2017) for volatility-adjusted forward return prediction, following the same experiment pipeline as [regression_v1.py](file:///c:/dev/this_is_my_quant/experiments/regression_v1.py).

## Background

The DA-RNN uses an **encoder-decoder** architecture with **two attention mechanisms**:

1. **Input Attention (Encoder):** At each time step, adaptively selects which driving features (technical indicators) are most relevant, using the encoder's previous hidden/cell state. This replaces the "treat all 20 features equally" approach of the current `RegressionAttentionLSTM`.

2. **Temporal Attention (Decoder):** The decoder attends over all encoder hidden states to select which past time steps matter most. At each step, the decoder also receives the **scaled historical target value** $y_{t-1}$ as an autoregressive input, following the original paper exactly.

### Why DA-RNN for this project

- The existing `RegressionAttentionLSTM` applies self-attention over LSTM hidden states (temporal only). DA-RNN adds **feature-level attention** — critical when not all 20 SHAP features are equally informative at every time step.
- The encoder-decoder structure with autoregressive target input lets the decoder condition predictions on recent realized returns, capturing return momentum/mean-reversion dynamics.
- Both attention weight matrices are interpretable: you can inspect which features and which time steps drove each prediction.

> [!IMPORTANT]
> **No look-ahead bias.** Because this is strictly a 1-step forward prediction ($t+1$), the realized return $y_{t-1}$ is fully known at prediction time $t$. Using it as decoder input follows the original paper exactly and does **not** leak future information. Teacher forcing during training feeds the actual historical $y_{t-1}$ at each decoder step.

## Proposed Changes

### Model — DA-RNN

#### [NEW] [darnn.py](file:///c:/dev/this_is_my_quant/src/models/darnn.py)

A new model file containing three classes:

**`InputAttentionEncoder`**
- LSTM cell that processes one time step at a time
- At each step $t$, computes attention weights $\alpha$ over all $n$ input features using $(h_{t-1}, s_{t-1})$ and the full input matrix $X$
- Produces weighted input $\tilde{x}_t = \alpha_t \odot x_t$, fed into the LSTM cell
- Returns all encoder hidden states $H = [h_1, \ldots, h_T]$

**`TemporalAttentionDecoder`**
- LSTM cell that iterates over decoder time steps up to $T$
- At each step $t$, computes attention weights $\beta$ over all encoder hidden states using the decoder's previous state $(d_{t-1}, s'_{t-1})$
- Produces context vector $c_t = \sum \beta_t^i \cdot h_i$
- Context vector $c_t$ is combined with the **scaled historical target value** $y_{t-1}$ and fed into the decoder LSTM cell
- The target value passes through a small linear embedding layer (`nn.Linear(1, 1)`) before concatenation with the context, matching the paper's formulation
- Final hidden state $d_T \rightarrow$ regression head $\rightarrow$ single scalar prediction $\hat{y}_T$

**`DARNN`** (wrapper)
- Combines encoder + decoder
- `forward(x_features, y_history)` takes two inputs:
  - `x_features`: `(batch, T, n_features)` — the driving series (technical indicators)
  - `y_history`: `(batch, T)` — the scaled historical target values (lagged by 1 step relative to features)
- Returns `(prediction, None, None)` for `RegressionTrainer` compatibility
- `predict(x_features, y_history)` returns `{"prediction": tensor}` matching the `RegressionAttentionLSTM` interface
- Optional `return_attention=True` to get both input and temporal attention weights

Key hyperparameters:
| Parameter | Default | Notes |
|---|---|---|
| `input_size` | (dynamic) | Number of driving features (n=20) |
| `encoder_hidden` | 64 | Encoder LSTM hidden size |
| `decoder_hidden` | 64 | Decoder LSTM hidden size |
| `dropout` | 0.2 | Applied to encoder/decoder outputs |

---

#### [MODIFY] [__init__.py](file:///c:/dev/this_is_my_quant/src/models/__init__.py)

Add `DARNN` to the model exports.

---

### Experiment

#### [NEW] [regression_darnn.py](file:///c:/dev/this_is_my_quant/experiments/regression_darnn.py)

A new experiment file that mirrors [regression_v1.py](file:///c:/dev/this_is_my_quant/experiments/regression_v1.py) in terms of:
- Data loading (`load_stocks` + `PriceCache`)
- Feature computation (`ComprehensiveIndicatorsV7.compute_shap_top20` with the same 20 SHAP features)
- Target creation (volatility-adjusted forward return with the same clipping)
- Temporal splits (same `train_end`, `val_end` defaults)
- Pooled training from `EXTENDED_TICKERS`
- Evaluation: pooled metrics, per-stock evaluation, cross-sectional IC
- Output: `config.json`, `summary.csv`, `per_stock_results.csv`, `model.pt`

**Key differences from regression_v1:**

##### 1. Sequence Creation — Correct $t+1$ Alignment

> [!CAUTION]
> **Critical alignment rule.** The decoder history `y_history` must be **lagged by exactly 1 step** relative to the feature window `X`. This ensures Today's forward return (the prediction target) is never leaked into the decoder input. The prediction target remains `targets[i + seq_length - 1]`, identical to regression_v1.

Given a 5-day window where Day 5 = "Today":

| Component | Slice | Days covered | Rationale |
|---|---|---|---|
| **Features X** | `feature_data[i : i + seq_length]` | Days 1–5 | Indicators up to Today |
| **Decoder y_history** | `targets[i-1 : i + seq_length - 1]` | Days 0–4 | Realized returns ending **Yesterday** |
| **Prediction target y** | `targets[i + seq_length - 1]` | Day 5 | Today's forward return (same as regression_v1) |

The loop starts at `i = 1` (not `i = 0`) to accommodate the 1-step lookback for `y_history`:

```python
def create_sequences_darnn(feature_data, targets, dates, seq_length, train_end, val_end):
    X_train, yh_train, y_train = [], [], []
    X_val, yh_val, y_val = [], [], []
    X_test, yh_test, y_test = [], [], []

    # Start at 1 to allow y_history to look back 1 step before the feature window
    for i in range(1, len(feature_data) - seq_length + 1):
        target_date = dates[i + seq_length - 1]

        # Features: [i, i + seq_length)
        seq_x = feature_data[i : i + seq_length]

        # Decoder history: lagged by 1 step → [i-1, i + seq_length - 1)
        seq_yh = targets[i - 1 : i + seq_length - 1]

        # Prediction target: anchored to last day of feature window
        target = targets[i + seq_length - 1]

        if target_date < train_end:
            X_train.append(seq_x); yh_train.append(seq_yh); y_train.append(target)
        elif target_date < val_end:
            X_val.append(seq_x); yh_val.append(seq_yh); y_val.append(target)
        else:
            X_test.append(seq_x); yh_test.append(seq_yh); y_test.append(target)

    # ... return arrays (float32)
```

##### 2. Target Scaling — Unified Scale for All Target Data

Both the decoder history inputs **and** the prediction targets are scaled using a single `target_scaler`. The model trains entirely in scaled space for clean gradient flow. During evaluation, predictions and targets are inverse-transformed back to original volatility-adjusted scale for interpretable metrics.

The experiment maintains **two scalers**:

- `self.feature_scaler`: `StandardScaler` fitted on training feature matrix (same as regression_v1)
- `self.target_scaler`: `StandardScaler` fitted on training target values, applied to decoder history **and** prediction targets

```python
# Fit on training targets (y_train covers the same distribution as yh_train)
self.target_scaler = StandardScaler()
self.target_scaler.fit(y_train.reshape(-1, 1))

# Scale decoder history for all splits
yh_train_s = self.target_scaler.transform(yh_train.reshape(-1, 1)).reshape(yh_train.shape)
yh_val_s   = self.target_scaler.transform(yh_val.reshape(-1, 1)).reshape(yh_val.shape)
yh_test_s  = self.target_scaler.transform(yh_test.reshape(-1, 1)).reshape(yh_test.shape)

# Scale prediction targets for all splits
y_train_s = self.target_scaler.transform(y_train.reshape(-1, 1)).ravel()
y_val_s   = self.target_scaler.transform(y_val.reshape(-1, 1)).ravel()
y_test_s  = self.target_scaler.transform(y_test.reshape(-1, 1)).ravel()
```

> [!IMPORTANT]
> **Train in scaled space, evaluate in original space.** The Huber loss is computed against scaled targets. During evaluation, both predictions and true targets are inverse-transformed via `self.target_scaler.inverse_transform()` before computing IC, directional accuracy, R², and cross-sectional metrics. This ensures gradient magnitudes are well-behaved during training while metrics remain interpretable in the original volatility-adjusted return space.

##### 3. Training Loop Adaptation

The `DARNNTrainer` extends `RegressionTrainer` to handle the two-input forward pass with teacher forcing. **All tensors in the DataLoader are scaled:**

```python
# DataLoader wraps TensorDataset of three tensors — all scaled
train_dataset = TensorDataset(
    torch.FloatTensor(X_train_s),    # scaled features
    torch.FloatTensor(yh_train_s),   # scaled target history
    torch.FloatTensor(y_train_s),    # scaled prediction target
)

# Training forward pass (teacher forcing with known y_{t-1})
for X_batch, yh_batch, y_batch in train_loader:
    prediction, _, _ = self.model(X_batch, yh_batch)
    loss = self.loss_fn(prediction, y_batch)  # both in scaled space
```

##### 4. Evaluation — Inverse Transform to Original Scale

During evaluation (pooled test, per-stock, cross-sectional IC), predictions and targets are unscaled before metric computation:

```python
# Model outputs scaled predictions
scaled_preds = model.predict(X_test_s, yh_test_s)["prediction"].cpu().numpy()

# Inverse transform to original vol-adj return space
preds_original = self.target_scaler.inverse_transform(scaled_preds.reshape(-1, 1)).ravel()
y_original     = self.target_scaler.inverse_transform(y_test_s.reshape(-1, 1)).ravel()

# Compute metrics in original space
metrics = compute_regression_metrics(y_original, preds_original)
```

##### 5. Model instantiation

```python
model = DARNN(
    input_size=n_features,
    encoder_hidden=64,
    decoder_hidden=64,
    dropout=self.dropout,
)
```

## Verification Plan

### Automated Tests
- **Shape test:** Verify output shape `(batch,)` for random inputs `(x_features, y_history)`
- **Gradient flow test:** Verify all parameters receive gradients (no dead params)
- **Smoke test:** `python experiments/regression_darnn.py --epochs 2 --stocks 5`

### Manual Verification
- Compare `summary.csv` metrics (IC, directional accuracy, cross-sectional IC) against regression_v1 runs to assess if dual-stage attention improves ranking quality
