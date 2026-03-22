# Model Architecture Analysis

## Summary of Potential Issues

After reviewing the model architecture ([baseline_lstm.py](file:///c:/dev/this_is_my_quant/src/models/baseline_lstm.py)), loss functions ([losses.py](file:///c:/dev/this_is_my_quant/src/models/losses.py)), trainer ([trainer.py](file:///c:/dev/this_is_my_quant/src/training/trainer.py)), and configuration ([settings.py](file:///c:/dev/this_is_my_quant/config/settings.py)), I've identified several potential issues that may be hurting model learning.

---

## 🔴 Critical Issues

### 1. **Confidence Head is Unused but Trained (WASTE OF CAPACITY)**

**Location:** [AttentionLSTM](file:///c:/dev/this_is_my_quant/src/models/baseline_lstm.py#255-393) (lines 308-313), [Trainer](file:///c:/dev/this_is_my_quant/src/training/trainer.py#30-569) (lines 136-139)

**Problem:** The model has a `confidence_head` that outputs a meta-prediction ("will my trend prediction be correct?"). However:
- The experiments use `nn.CrossEntropyLoss()` directly, **not** [CombinedLoss](file:///c:/dev/this_is_my_quant/src/models/losses.py#19-118)
- The trainer detects this via `self._is_combined_loss = hasattr(loss_fn, 'trend_weight')` (line 99)
- When `CrossEntropyLoss` is used, the confidence output is **computed but never used in the loss**

```python
# In trainer.py line 158-160 (when NOT using CombinedLoss):
loss = self.loss_fn(trend_logits, y)  # confidence is IGNORED
loss_dict = {"accuracy": ...}
```

**Impact:** 
- ~2,113 extra parameters wasted (32×64 + 32 + 32×1 + 1)
- Confidence head learns nothing useful
- Gradients still flow through shared layers (`fc`) to confidence head even though unused
- May create competing optimization objectives in shared representation

**Recommendation:** Either:
- A) Remove confidence head entirely for pure classification
- B) Use [CombinedLoss](file:///c:/dev/this_is_my_quant/src/models/losses.py#19-118) properly and set `confidence_weight` appropriately

---

### 2. **Loss Function Mismatch in Experiments**

**Location:** [lstm_size_comparison.py](file:///c:/dev/this_is_my_quant/experiments/lstm_size_comparison.py) line 481

**Problem:** Experiments use `nn.CrossEntropyLoss()` but the model architecture was designed for [CombinedLoss](file:///c:/dev/this_is_my_quant/src/models/losses.py#19-118):

```python
# Current (lstm_size_comparison.py):
loss_fn = nn.CrossEntropyLoss()

# Model was designed for:
from src.models.losses import CombinedLoss
loss_fn = CombinedLoss(trend_weight=1.0, confidence_weight=0.5)
```

**Impact:** Model architecture has layers that never get useful gradients.

---

### 3. **Multi-Head Attention Adds Complexity for 20 Features**

**Location:** [SentimentAttention](file:///c:/dev/this_is_my_quant/src/models/baseline_lstm.py#218-253) class (lines 218-252), used in [AttentionLSTM](file:///c:/dev/this_is_my_quant/src/models/baseline_lstm.py#255-393)

**Problem:** Using 4-head self-attention on a 20-timestep sequence with 20 features:
- `embed_dim=hidden_size` (64-128) / `num_heads=4` = 16-32 dim/head
- Self-attention has O(n²) complexity for sequence length n
- For seq_len=20, this computes 400 attention weights per sample per head

**Impact:**
- Adds ~65K-260K parameters for attention layer
- May be overkill for such short sequences
- Original design was for sentiment weighting, but experiments use only price features

**Recommendation:** Consider simpler alternatives:
- Global average pooling over timesteps
- Single attention query (like LSTM with attention to last hidden state)
- Remove attention entirely for baseline comparison

---

## 🟡 Medium Priority Issues

### 4. **Label Smoothing May Hurt on Balanced 3-Class Problem**

**Location:** [CombinedLoss](file:///c:/dev/this_is_my_quant/src/models/losses.py#19-118) (line 59) - `label_smoothing=0.1`

**Problem:** Label smoothing is applied by default (0.1). For a 3-class problem with ±0.5% thresholds:
- Classes are already somewhat balanced
- Label smoothing "softens" targets which can hurt when classes are distinguishable
- May prevent model from being confident on clear signals

**Impact:** Reduces model's ability to learn sharp decision boundaries.

---

### 5. **CosineAnnealingWarmRestarts with T_0=10 May Restart Too Early**

**Location:** [trainer.py](file:///c:/dev/this_is_my_quant/src/training/trainer.py) lines 77-82

```python
self.scheduler = CosineAnnealingWarmRestarts(
    self.optimizer,
    T_0=10,      # First restart at epoch 10
    T_mult=2,    # Next at 10+20=30, then 30+40=70
    eta_min=1e-6,
)
```

**Problem:** 
- Learning rate resets to initial value at epoch 10
- With `early_stopping_patience=15`, we might stop before benefiting from restart
- LR oscillations can destabilize training on noisy financial data

**Recommendation:** Consider simpler schedulers like `ReduceLROnPlateau` or `StepLR`.

---

### 6. **Dropout Applied After Attention (Double Regularization)**

**Location:** [AttentionLSTM](file:///c:/dev/this_is_my_quant/src/models/baseline_lstm.py#255-393) lines 297-299

```python
self.fc = nn.Sequential(
    nn.Linear(hidden_size, 64),
    nn.ReLU(),
    nn.Dropout(dropout),  # dropout=0.4 in experiments
)
```

**Problem:**
- LSTM already has dropout between layers (if num_layers > 1)
- Attention has dropout (default 0.1)
- FC layer has more dropout (0.4)
- Trend/confidence heads have even more dropout (0.2)

**Total regularization chain:** LSTM dropout → Attention dropout → FC dropout → Head dropout

**Impact:** May be over-regularizing for 600K+ training samples.

---

## 🟢 Minor Issues

### 7. **Bidirectional Disabled but BaselineLSTM Supports It**

**Location:** [config/settings.py](file:///c:/dev/this_is_my_quant/config/settings.py) line 97: `bidirectional: bool = False`

**Note:** [AttentionLSTM](file:///c:/dev/this_is_my_quant/src/models/baseline_lstm.py#255-393) is NOT bidirectional, but [BaselineLSTM](file:///c:/dev/this_is_my_quant/src/models/baseline_lstm.py#20-216) has this option. Not necessarily an issue, but bidirectional might help with pattern recognition.

---

### 8. **Mixed Precision (FP16) Default ON**

**Location:** [config/settings.py](file:///c:/dev/this_is_my_quant/config/settings.py) line 131: `mixed_precision: bool = True`

**Potential Issue:** FP16 can cause gradient underflow for small learning rates (5e-4). The `GradScaler` should handle this, but worth noting.

---

### 9. **Input LayerNorm Missing**

**Problem:** No input normalization layer before LSTM. While data is StandardScaler'd, having a learnable LayerNorm could help with distribution shift across stocks.

---

## Summary Table

| Issue | Severity | Lines of Code | Fix Effort |
|-------|----------|---------------|------------|
| Confidence head unused | 🔴 Critical | baseline_lstm.py:308-313 | Small |
| Loss function mismatch | 🔴 Critical | lstm_size_comparison.py:481 | Trivial |
| Attention overkill | 🔴 Critical | baseline_lstm.py:291-292 | Medium |
| Label smoothing harmful | 🟡 Medium | losses.py:59 | Trivial |
| LR scheduler restarts | 🟡 Medium | trainer.py:77-82 | Small |
| Over-regularization | 🟡 Medium | Multiple files | Small |

---

## Recommended Experiments

1. **Baseline Without Confidence Head:** Create `SimpleLSTM` model with only trend classification
2. **Remove Attention:** Compare [AttentionLSTM](file:///c:/dev/this_is_my_quant/src/models/baseline_lstm.py#255-393) vs simple LSTM with last hidden state
3. **Remove Label Smoothing:** Set `label_smoothing=0.0` in loss
4. **Reduce Dropout:** Try `dropout=0.2` instead of `0.4`
5. **Use CombinedLoss Properly:** If keeping confidence head, actually use it

---

## Quick Fix: Create Simplified Model

Consider creating a streamlined model like:

```python
class SimpleTrendLSTM(nn.Module):
    """Simplified LSTM for trend prediction only."""
    
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3),  # 3 classes
        )
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.classifier(h_n[-1])
```

This removes:
- Confidence head (not used)
- Attention mechanism (overkill)
- Extra dropout layers
- Shared FC layer complexity
