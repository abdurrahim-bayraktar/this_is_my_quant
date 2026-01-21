# LSTM Model Architecture: Technical Documentation

> Detailed technical documentation for the BaselineLSTM model used in stock trend prediction.

---

## Table of Contents

1. [Model Overview](#model-overview)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Architecture Details](#architecture-details)
4. [Feature Engineering](#feature-engineering)
5. [Loss Function Design](#loss-function-design)
6. [Training Strategy](#training-strategy)
7. [Inference and Prediction](#inference-and-prediction)

---

## Model Overview

The `BaselineLSTM` is a dual-head recurrent neural network designed for:

1. **Primary Task**: 3-class trend classification (Up/Down/Neutral)
2. **Auxiliary Task**: Confidence estimation (meta-prediction)

### Why Dual Heads?

Traditional classifiers output predictions without uncertainty estimates. Our model learns to predict its own accuracy, enabling:

- **Selective trading**: Only act on high-confidence predictions
- **Risk management**: Scale position sizes by confidence
- **Performance improvement**: Filter out uncertain predictions

---

## Mathematical Foundation

### LSTM Cell Equations

For each timestep $t$, the LSTM computes:

```
Forget Gate:     f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
Input Gate:      i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
Candidate:       C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
Cell State:      C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
Output Gate:     o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
Hidden State:    h_t = o_t ⊙ tanh(C_t)
```

Where:
- $σ$ = sigmoid function
- $⊙$ = element-wise multiplication
- $[·,·]$ = concatenation

### Why LSTM over GRU?

| Property | LSTM | GRU |
|----------|------|-----|
| Gates | 3 (forget, input, output) | 2 (reset, update) |
| Cell state | Separate memory cell | Combined with hidden |
| Parameters | ~30% more | Fewer |
| Long-term memory | Better for long sequences | Better for short |

**Our choice**: LSTM provides better gradient flow for financial time series with varying temporal dependencies.

---

## Architecture Details

### Layer Specifications

```
┌───────────────────────────────────────────────────────────┐
│ Layer                    │ Shape          │ Parameters    │
├──────────────────────────┼────────────────┼───────────────┤
│ Input                    │ [B, 20, 25]    │ -             │
│ LSTM Layer 1             │ [B, 20, 128]   │ 79,360        │
│ LSTM Layer 2             │ [B, 20, 128]   │ 131,584       │
│ Take last hidden         │ [B, 128]       │ -             │
│ Shared Dense             │ [B, 64]        │ 8,256         │
│ Trend Head (32→3)        │ [B, 3]         │ 2,147         │
│ Confidence Head (32→1)   │ [B, 1]         │ 1,057         │
├──────────────────────────┼────────────────┼───────────────┤
│ TOTAL                    │                │ ~130,000      │
└───────────────────────────────────────────────────────────┘

B = batch size (32)
20 = sequence length (trading days)
25 = input features
128 = hidden size
```

### LSTM Layer Configuration

```python
nn.LSTM(
    input_size=25,        # Number of features per timestep
    hidden_size=128,      # Hidden state dimension
    num_layers=2,         # Stacked LSTM layers
    batch_first=True,     # Input shape: [batch, seq, features]
    dropout=0.3,          # Inter-layer dropout (only if num_layers > 1)
    bidirectional=False,  # Unidirectional for causal prediction
)
```

### Why Unidirectional?

Bidirectional LSTMs see future context, which would cause **lookahead bias** in financial prediction. We must ensure causality:

- Training: Only past data available
- Inference: Real-time prediction without future knowledge

### Shared Representation Layer

```python
fc_shared = nn.Sequential(
    nn.Linear(128, 64),   # Dimensionality reduction
    nn.ReLU(),            # Non-linearity
    nn.Dropout(0.3),      # Regularization
)
```

**Purpose**: Creates a shared representation that both heads can utilize, encouraging feature extraction useful for both tasks.

### Trend Classification Head

```python
trend_head = nn.Sequential(
    nn.Linear(64, 32),        # Further compression
    nn.ReLU(),
    nn.Dropout(0.15),         # Lighter dropout
    nn.Linear(32, 3),         # 3 classes: Down, Neutral, Up
    # Softmax applied in loss function (CrossEntropyLoss)
)
```

**Output**: Raw logits → Softmax for probabilities
- Class 0: Down (return < -0.5%)
- Class 1: Neutral (-0.5% ≤ return ≤ +0.5%)
- Class 2: Up (return > +0.5%)

### Confidence (Meta-Prediction) Head

```python
confidence_head = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Dropout(0.15),
    nn.Linear(32, 1),
    # NO Sigmoid - applied at inference only
)
```

**Critical Design Choice**: No sigmoid during training because:
1. `BCEWithLogitsLoss` applies log-sigmoid internally
2. Mixed precision (FP16) + sigmoid + BCE can cause NaN
3. More numerically stable gradient computation

---

## Feature Engineering

### Input Feature Categories

#### 1. Price Features (5)

| Feature  | Description    | Normalization |
|----------|----------------|---------------|
| `open`   | Opening price  | Z-score       |
| `high`   | Daily high     | Z-score       |
| `low`    | Daily low      | Z-score       |
| `close`  | Closing price  | Z-score       |
| `volume` | Trading volume | Log + Z-score |

#### 2. Return Features (2)

| Feature      | Formula                     | Purpose                 |
|--------------|-----------------------------|-------------------------|
| `return`     | $(P_t - P_{t-1}) / P_{t-1}$ | Daily percentage return |
| `log_return` | $\ln(P_t / P_{t-1})$        | Log return (additive)   |

#### 3. Momentum Indicators (2)

**RSI (Relative Strength Index)**:
```
RSI = 100 - 100 / (1 + RS)
RS = Average Gain / Average Loss (14-day)
```

**MFI (Money Flow Index)**:
```
MFI = 100 - 100 / (1 + Money Ratio)
Money Ratio = Positive Money Flow / Negative Money Flow
```

#### 4. Trend Indicators (3)

**MACD (Moving Average Convergence Divergence)**:
```
MACD Line = EMA(12) - EMA(26)
Signal Line = EMA(9) of MACD Line
Histogram = MACD Line - Signal Line
```

#### 5. Volatility Indicators (5)

**Bollinger Bands**:
```
Middle = SMA(20)
Upper = Middle + 2 × σ(20)
Lower = Middle - 2 × σ(20)
Width = (Upper - Lower) / Middle
```

**ATR (Average True Range)**:
```
TR = max(High - Low, |High - Prev Close|, |Low - Prev Close|)
ATR = SMA(TR, 14)
```

#### 6. Moving Averages (4)

| Feature  | Formula                              |
|----------|--------------------------------------|
| `sma_5`  | Simple Moving Average (5 days)       |
| `sma_20` | Simple Moving Average (20 days)      |
| `ema_12` | Exponential Moving Average (12 days) |
| `ema_26` | Exponential Moving Average (26 days) |

#### 7. Sentiment Features (4)

| Feature | Source | Aggregation |
|---------|--------|-------------|
| `sentiment_mean` | FinBERT | Time-weighted daily average |
| `sentiment_std` | FinBERT | Daily dispersion |
| `news_count` | News API | Number of articles |
| `sentiment_momentum` | Derived | 3-day sentiment change |

### Feature Normalization

**Z-Score Normalization** (rolling window):
```python
def normalize(x, window=60):
    mean = x.rolling(window).mean()
    std = x.rolling(window).std()
    return (x - mean) / (std + 1e-8)
```

**Why Z-Score?**
- Removes scale differences between features
- Preserves temporal patterns
- Handles non-stationarity via rolling window

---

## Loss Function Design

### Combined Multi-Task Loss

```python
L_total = α × L_trend + β × L_confidence

where:
    α = 1.0  (trend weight)
    β = 0.5  (confidence weight)
```

### Trend Loss: CrossEntropyLoss

```python
L_trend = -∑ y_c × log(p_c)

with label smoothing (ε = 0.1):
    y_smooth = (1 - ε) × y_onehot + ε / K
```

**Label Smoothing**: Prevents overconfident predictions, improves generalization.

### Confidence Loss: BCEWithLogitsLoss

```python
# Target: 1 if prediction correct, 0 otherwise
confidence_target = (predicted_class == true_class).float()

L_confidence = -[y × log(σ(z)) + (1-y) × log(1-σ(z))]
```

**Key Insight**: The confidence head learns to predict whether the trend head's output will be correct, creating a self-aware model.

### Alternative: Focal Loss

For highly imbalanced classes:
```python
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)

where γ = 2.0 (focusing parameter)
```

Focal Loss down-weights easy examples, focusing learning on hard cases.

---

## Training Strategy

### Optimizer: AdamW

```python
optimizer = AdamW(
    params=model.parameters(),
    lr=1e-3,
    weight_decay=1e-5,  # L2 regularization
)
```

**Why AdamW over Adam?**
- Decoupled weight decay (not coupled with gradients)
- Better generalization, especially with small datasets

### Learning Rate Schedule

**Cosine Annealing with Warm Restarts**:
```python
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,      # Initial cycle length
    T_mult=2,    # Cycle length multiplier
    eta_min=1e-6 # Minimum learning rate
)
```

Benefits:
- Escapes local minima via restarts
- Gradual learning rate decay within cycles
- Better exploration of loss landscape

### Regularization

| Technique | Value | Purpose |
|-----------|-------|---------|
| Dropout | 0.3 | Prevent overfitting |
| Weight Decay | 1e-5 | L2 regularization |
| Gradient Clipping | 1.0 | Prevent exploding gradients |
| Early Stopping | 10 epochs | Prevent overfitting |
| Label Smoothing | 0.1 | Calibration improvement |

### Walk-Forward Validation

```
Full Dataset: |-------- Train --------|-- Val --|-- Test --|

Fold 1: |===Train===|=Val=|
Fold 2:    |===Train===|=Val=|
Fold 3:       |===Train===|=Val=|
Fold 4:          |===Train===|=Val=|
Fold 5:             |===Train===|=Val=|
```

**Properties**:
- Preserves temporal order
- No lookahead bias
- Simulates actual trading conditions

---

## Inference and Prediction

### Forward Pass

```python
def forward(x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Args:
        x: [batch, seq_len=20, features=25]
    
    Returns:
        trend_logits: [batch, 3] - raw scores for each class
        confidence_logits: [batch, 1] - raw confidence score
        hidden: [batch, 128] - last hidden state (optional)
    """
    lstm_out, (h_n, c_n) = self.lstm(x)
    final_hidden = h_n[-1]  # Last layer, last timestep
    
    shared = self.fc_shared(final_hidden)
    trend_logits = self.trend_head(shared)
    confidence_logits = self.confidence_head(shared)
    
    return trend_logits, confidence_logits, final_hidden
```

### Prediction with Confidence Filtering

```python
def predict(x: Tensor, threshold: float = 0.5) -> Dict:
    """
    Make predictions with confidence filtering.
    
    Returns:
        - trend_class: Predicted class (0=Down, 1=Neutral, 2=Up)
        - trend_probs: Class probabilities
        - confidence: Model's confidence in prediction
        - should_trade: Boolean mask for high-confidence predictions
    """
    trend_logits, conf_logits, _ = self.forward(x)
    
    trend_probs = F.softmax(trend_logits, dim=-1)
    trend_class = trend_probs.argmax(dim=-1)
    confidence = torch.sigmoid(conf_logits).squeeze(-1)
    
    return {
        'trend_class': trend_class,
        'trend_probs': trend_probs,
        'confidence': confidence,
        'should_trade': confidence > threshold,
    }
```

### Trading Strategy Integration

```python
# Only trade when confidence > 0.6
predictions = model.predict(features, threshold=0.6)

for i, should_trade in enumerate(predictions['should_trade']):
    if should_trade:
        trend = predictions['trend_class'][i]
        confidence = predictions['confidence'][i]
        
        if trend == 2:  # Up
            execute_buy_order(ticker, size=confidence)
        elif trend == 0:  # Down
            execute_sell_order(ticker, size=confidence)
```

---

## Performance Considerations

### Memory Optimization

1. **Mixed Precision (FP16)**: Halves memory usage
2. **Gradient Checkpointing**: Trade compute for memory
3. **Batch Size Tuning**: Maximize GPU utilization

### Inference Speed

| Batch Size | Latency (RTX 3050) |
|------------|-------------------|
| 1 | ~2 ms |
| 32 | ~5 ms |
| 128 | ~15 ms |

### Model Size

```
Total Parameters: 130,000
Model Size: ~0.5 MB (FP32)
Model Size: ~0.25 MB (FP16)
```

---

## Common Issues and Solutions

### 1. NaN Loss During Training

**Cause**: BCE + sigmoid + FP16 instability
**Solution**: Use `BCEWithLogitsLoss`, no sigmoid in model

### 2. Class Imbalance

**Cause**: More neutral days than up/down
**Solution**: Class weights or Focal Loss

### 3. Overfitting

**Symptoms**: Training loss decreases, validation loss increases
**Solutions**:
- Increase dropout
- Add weight decay
- Early stopping
- More data augmentation

### 4. Vanishing Gradients

**Cause**: Deep LSTM without proper initialization
**Solutions**:
- Xavier/Orthogonal initialization
- Gradient clipping
- Layer normalization

---

## References

1. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.
2. Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. ICLR.
3. Lin, T. Y., et al. (2017). Focal Loss for Dense Object Detection. ICCV.
4. Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. ICLR.
