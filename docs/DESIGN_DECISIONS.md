# Design Decisions & Rationale

> Explanation of key technical decisions made in the project with alternatives considered.

---

## Table of Contents

1. [Model Selection](#model-selection)
2. [Data Strategy](#data-strategy)
3. [Sentiment Analysis](#sentiment-analysis)
4. [Training Methodology](#training-methodology)
5. [Meta-Prediction Design](#meta-prediction-design)
6. [Hardware Optimization](#hardware-optimization)

---

## Model Selection

### Decision: LSTM over Transformer

| Factor | LSTM | Transformer |
|--------|------|-------------|
| Sequence Length | Short (20 days) ✓ | Long (100+ days) |
| Data Requirement | ~50k samples ✓ | 100k+ samples |
| Parameters | ~130k ✓ | 1M+ |
| Training Time | Hours ✓ | Days |
| Interpretability | Hidden states ✓ | Attention maps |

**Rationale**: Our sequence length (20 trading days) is short, and our dataset is relatively small. LSTM is more parameter-efficient and doesn't require the large datasets that Transformers need to generalize well.

**Considered Alternatives**:
- **Transformer**: More powerful but overkill for short sequences
- **GRU**: Simpler but LSTM has better gradient flow
- **CNN-LSTM**: Adds complexity without significant improvement for our use case

---

### Decision: 2 Layers, 128 Hidden Units

**Rationale**: 
- 1 layer: Insufficient capacity for complex patterns
- 3+ layers: Diminishing returns, harder to train, overfitting risk
- 64 hidden: Underfits on multi-stock data
- 256 hidden: Overfits without significant accuracy gain

**Evidence**: Preliminary experiments showed 2×128 LSTM achieved best validation accuracy.

---

### Decision: Unidirectional (Not Bidirectional)

**Critical**: Bidirectional LSTMs see future context, causing **lookahead bias**.

In financial prediction:
- We can only use past data to predict future prices
- Bidirectional would leak future information during training
- Model would fail catastrophically in real trading

---

## Data Strategy

### Decision: Top 200 Stocks (Not Full S&P 500)

| S&P 500 Subset | Pros | Cons |
|----------------|------|------|
| Full 500 | More data | Storage: 45GB, noise from small caps |
| Top 200 | Liquid stocks, less noise | Less data |
| Top 100 | Very liquid | May miss sector diversity |

**Rationale**: Top 200 by market cap covers major sectors, has sufficient liquidity (less noise), and fits our hardware constraints (~15GB).

---

### Decision: 2016-2024 Date Range

| Period | Pros | Cons |
|--------|------|------|
| 1999-2024 | More data | Outdated patterns, pre-smartphone |
| 2010-2024 | Modern market | May miss older patterns |
| 2016-2024 | Modern + COVID ✓ | Slightly less data |

**Rationale**: 
- Post-2015 reflects current market dynamics
- Includes COVID volatility (model learns extreme regimes)
- Mobile trading, algorithmic trading well-established

---

### Decision: ±0.5% Trend Threshold

| Threshold | Up | Neutral | Down | Notes |
|-----------|------|---------|------|-------|
| ±0.25% | 35% | 30% | 35% | Too sensitive, noisy |
| ±0.5% | 28% | 44% | 28% | Standard ✓ |
| ±1.0% | 20% | 60% | 20% | Too conservative |

**Rationale**:
- Academic standard used in StockNet, DP-LSTM papers
- Accounts for typical transaction costs (~0.1-0.3%)
- Balanced class distribution enables fair training

---

## Sentiment Analysis

### Decision: FinBERT over General BERT

| Model | Domain | F1 Score (Financial) |
|-------|--------|---------------------|
| BERT-base | General | 0.72 |
| FinBERT (ProsusAI) | Financial ✓ | 0.87 |
| DistilBERT | General | 0.70 |
| RoBERTa | General | 0.75 |

**Rationale**: FinBERT is pre-trained on financial text (Reuters, Financial PhraseBank) and significantly outperforms general models on financial sentiment.

---

### Decision: Daily Aggregation (Not Per-Article)

| Approach | Pros | Cons |
|----------|------|------|
| Per-article embedding | Fine-grained | Memory explosion, noisy |
| Daily aggregation | Stable, efficient ✓ | Loses article-level detail |
| Hierarchical attention | Best of both | Complex, overfitting risk |

**Rationale**: Daily aggregation provides stable features without memory issues. Multiple articles per day are summarized via time-weighted averaging.

---

### Decision: Time-Weighted Averaging

```
Recent news has higher impact on same-day prices
Weight = exp(-hours_to_close / 12)
```

**Evidence**: Pre-market news (7 AM) has more impact than yesterday's news. Our weighting scheme reflects temporal relevance.

---

## Training Methodology

### Decision: Walk-Forward Validation (Not K-Fold)

**Standard K-Fold Problem**:
```
Fold 1: Train [1,2,4,5]  Test [3]    ← WRONG: Training sees future
Fold 2: Train [1,2,3,5]  Test [4]    ← WRONG: Training sees future
```

**Walk-Forward (Correct)**:
```
Fold 1: Train [1,2,3]    Test [4]    ← Training only sees past
Fold 2: Train [1,2,3,4]  Test [5]    ← Training only sees past
```

**Rationale**: Financial data is temporal. Standard cross-validation violates causality and gives optimistically biased estimates.

---

### Decision: AdamW over Adam

| Aspect | Adam | AdamW |
|--------|------|-------|
| Weight Decay | Coupled with gradient | Decoupled ✓ |
| Generalization | Good | Better |
| Implementation | Original | Standard now |

**Rationale**: AdamW decouples weight decay from gradient updates, providing better regularization and generalization on small datasets.

---

### Decision: Cosine Annealing with Warm Restarts

```
Learning Rate: ▲ high → gradual decrease → restart → ...
```

**Benefits**:
- Escapes local minima via periodic restarts
- Natural warm-up at each restart
- Better exploration than step decay

---

### Decision: Early Stopping (Patience = 10)

**Rationale**: 
- Stop when validation loss stops improving for 10 epochs
- Prevents overfitting without manual epoch tuning
- Saves training time

---

## Meta-Prediction Design

### Decision: Dual-Head Architecture

```
Shared Features → Trend Head   → Up/Down/Neutral
                → Confidence Head → P(correct)
```

**Alternatives Considered**:
1. **Single softmax confidence**: Max probability as confidence
   - Problem: Poorly calibrated for NN
2. **Separate models**: One for trend, one for confidence
   - Problem: Misses shared representations
3. **Monte Carlo Dropout**: Multiple forward passes
   - Problem: Slow inference, inconsistent

**Rationale**: Dual-head shares representations, learns confidence jointly, and is efficient at inference.

---

### Decision: Binary Confidence Target

```python
confidence_target = (predicted_class == true_class).float()
# 1.0 if correct, 0.0 if wrong
```

**Alternatives**:
- Soft target based on class probability
- Distance-based confidence

**Rationale**: Binary target is simple, interpretable, and directly trains the model to predict correctness.

---

### Decision: BCEWithLogitsLoss (Not BCELoss)

**Problem with BCELoss + Sigmoid**:
```python
# During mixed precision training:
x = model(input)  # FP16
confidence = sigmoid(x)  # FP16 → underflow/overflow
loss = bce_loss(confidence, target)  # NaN!
```

**Solution**:
```python
# No sigmoid in model
x = model(input)  # FP16
loss = bce_with_logits_loss(x, target)  # Stable log-sigmoid internally
```

---

## Hardware Optimization

### Decision: Mixed Precision Training (FP16)

| Precision | VRAM | Speed | Accuracy |
|-----------|------|-------|----------|
| FP32 | 12 GB | 1x | Baseline |
| FP16 | 6 GB ✓ | 1.5-2x | Same |

**Rationale**: FP16 halves memory usage with no accuracy loss, enabling training on consumer GPUs (RTX 3050, 6GB).

---

### Decision: Batch Size 32

| Batch Size | VRAM | Gradient Noise | Convergence |
|------------|------|----------------|-------------|
| 8 | 2 GB | High | Slower |
| 32 | 4 GB ✓ | Moderate | Fast |
| 128 | 12 GB | Low | May overfit |

**Rationale**: 32 balances memory usage with stable gradients.

---

### Decision: Gradient Clipping at 1.0

**LSTM Problem**: Long sequences can cause exploding gradients.

**Solution**:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Rationale**: Prevents training instability without hurting convergence.

---

## Summary Table

| Decision | Choice | Primary Rationale |
|----------|--------|-------------------|
| Model | LSTM (2×128) | Short sequences, limited data |
| Direction | Unidirectional | Avoid lookahead bias |
| Stocks | Top 200 | Coverage + hardware limits |
| Period | 2016-2024 | Modern dynamics + COVID |
| Threshold | ±0.5% | Academic standard |
| Sentiment | FinBERT | Domain-specific accuracy |
| Aggregation | Time-weighted daily | Stable, efficient |
| Validation | Walk-forward | Temporal causality |
| Optimizer | AdamW | Better generalization |
| LR Schedule | Cosine warm restarts | Escape local minima |
| Confidence | Dual-head | Shared features, efficient |
| Loss | BCEWithLogitsLoss | FP16 stability |
| Precision | FP16 | 6GB VRAM constraint |
| Batch | 32 | Memory/gradient balance |
