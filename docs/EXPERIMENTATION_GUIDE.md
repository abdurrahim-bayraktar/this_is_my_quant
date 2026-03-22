# Experimentation Guide
**Project**: Sentiment-Driven Stock Trend Prediction  
**Updated**: March 2026

---

## Quick Reference

### Run an experiment
```bash
python experiments/ultimate_model.py --epochs 100 --stocks 400
```

### Compare results
```bash
python experiments/compare_experiments.py reports/ultimate_model_*
```

### Create a new experiment variant
```bash
cp experiments/ultimate_model.py experiments/ultimate_model_v2.py
# Edit the pipeline, then run it
```

---

## 1. Project Structure

```
this_is_my_quant/
├── experiments/               # Experiment scripts (one per experiment)
│   ├── ultimate_model.py      # Baseline experiment
│   ├── compare_experiments.py # Compare results across runs
│   ├── ranked_tickers.py      # Stock universe (EXTENDED_TICKERS)
│   └── archive/               # Completed/old experiments
├── src/                       # Shared utilities (stable code)
│   ├── data/
│   │   ├── cache.py           # PriceCache — yfinance data caching
│   │   └── utils.py           # resample_to_weekly, load_stocks, create_sequences
│   ├── evaluation/
│   │   └── metrics.py         # compute_metrics, evaluate_model_on_data
│   ├── features/
│   │   └── indicators_v7.py   # ComprehensiveIndicatorsV7 (SHAP Top 20)
│   └── models/
│       ├── baseline_lstm.py   # AttentionLSTM, BaselineLSTM
│       └── simple_lstm.py     # SimpleTrendLSTM, MediumTrendLSTM
├── reports/                   # Experiment outputs (timestamped)
│   ├── ultimate_model_20260322_235000/
│   │   ├── config.json        # Full experiment configuration
│   │   ├── summary.csv        # Pooled + per-stock aggregate metrics
│   │   ├── per_stock_results.csv  # Per-stock detailed metrics
│   │   └── model.pt           # Saved model weights
│   └── ...
├── config/                    # Global project config (settings.py)
└── docs/                      # Documentation
```

---

## 2. Experiment Anatomy

Every experiment is a **self-contained Python script**. It imports shared utilities but owns its entire pipeline. Here is what each part does and where to find it:

### Pipeline Stages

| Stage | What it does | Where to modify |
|-------|-------------|-----------------|
| **Data Loading** | Download/cache stock prices | `load_data()` method or `src/data/cache.py` |
| **Feature Computation** | Compute technical indicators | `prepare_stock()` method or `src/features/indicators_v7.py` |
| **Label Creation** | Classify next-period return as Up/Neutral/Down | `prepare_stock()` — change `threshold_low`, `threshold_high`, or use custom logic |
| **Sequence Creation** | Window features into (seq_length, n_features) arrays | `create_sequences()` from `src/data/utils.py` or inline |
| **Pooling** | Concatenate all stocks into train/val/test arrays | `prepare_pooled_data()` method |
| **Scaling** | StandardScaler fit on train, transform all | `scale_data()` method |
| **Model** | Neural network architecture | `create_model()` or import from `src/models/` |
| **Training** | Loss function, optimizer, scheduler, early stopping | `Trainer` class (inline in experiment) |
| **Evaluation** | Compute accuracy, lift, MCC on test data | `evaluate_model_on_data()` from `src/evaluation/metrics.py` |
| **Saving** | Write config.json, summary.csv, per_stock_results.csv, model.pt | `run()` method |

### Shared Utilities (import these, don't duplicate)

```python
from src.data.cache import PriceCache                    # Data caching
from src.data.utils import resample_to_weekly, load_stocks, create_sequences  # Data transforms
from src.evaluation.metrics import compute_metrics, evaluate_model_on_data    # Metrics
from src.features.indicators_v7 import ComprehensiveIndicatorsV7             # Features
from src.models.simple_lstm import SimpleTrendLSTM        # Models
from src.models import AttentionLSTM                      # Models
from experiments.ranked_tickers import EXTENDED_TICKERS    # Stock universe
```

---

## 3. How to Create a New Experiment

### Step 1: Copy the baseline
```bash
cp experiments/ultimate_model.py experiments/my_experiment.py
```

### Step 2: Modify what you want to test

Below are the most common modifications, with exact code locations:

#### Change the features
```python
# In my_experiment.py, modify SHAP_TOP20_FEATURES or replace with:
CUSTOM_FEATURES = ["rsi_14", "macd_MACD_12_26_9", "volume_ratio", ...]

# And in prepare_stock(), change:
self.feature_cols = CUSTOM_FEATURES
```

#### Change the target variable (e.g., volatility-adjusted return)
```python
# In prepare_stock(), replace the label creation block:
# BEFORE:
df['return_next'] = df['Close'].pct_change().shift(-1)
df['trend'] = pd.cut(df['return_next'], bins=[...], labels=[0, 1, 2])

# AFTER (volatility-adjusted):
df['return_next'] = df['Close'].pct_change().shift(-1)
df['atr_20'] = df['Close'].rolling(20).std()
df['adjusted_return'] = df['return_next'] / df['atr_20']
df['trend'] = pd.cut(df['adjusted_return'], bins=[-np.inf, -1.0, 1.0, np.inf], labels=[0, 1, 2])
```

#### Change the model architecture
```python
# In create_model(), add a new branch:
elif model_type == "transformer":
    return MyTransformerModel(input_size=input_size, ...)
```

#### Change the training loop (loss function, scheduler)
```python
# Modify the inline Trainer class directly:
class Trainer:
    def __init__(self, ...):
        self.loss_fn = FocalLoss(gamma=2.0)  # Changed
        self.scheduler = CosineAnnealingWarmRestarts(...)  # Changed
```

#### Change the training pool
```python
# In the config dict in main():
config = {
    "train_stocks": CURATED_100_STOCKS,    # Your curated list
    "test_stocks": SP500_STOCKS[:10],      # Evaluate on S&P components
    ...
}
```

#### Change data frequency
```bash
python experiments/my_experiment.py --frequency weekly --seq-length 12
```

### Step 3: Run the experiment
```bash
python experiments/my_experiment.py --name my_experiment --epochs 100 --stocks 400
```

### Step 4: Compare with baseline
```bash
python experiments/compare_experiments.py reports/ultimate_model_* reports/my_experiment_*
```

---

## 4. CLI Reference for `ultimate_model.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 100 | Training epochs |
| `--stocks` | 400 | Number of training stocks (from EXTENDED_TICKERS) |
| `--batch-size` | 1024 (daily) / 512 (weekly) | Batch size |
| `--frequency` | daily | `daily` or `weekly` |
| `--model` | simple | `simple` (SimpleTrendLSTM) or `attention` (AttentionLSTM) |
| `--seq-length` | 20 (daily) / 12 (weekly) | Lookback window length |
| `--threshold LOW HIGH` | ±0.005 (daily) / ±0.01 (weekly) | Trend classification thresholds |
| `--test-stocks` | MSFT AAPL JPM ... | Tickers for per-stock evaluation |
| `--dropout` | 0.2 | Dropout rate |
| `--label-smoothing` | 0.1 | Label smoothing factor |
| `--lr` | 5e-4 | Learning rate |
| `--patience` | 15 | Early stopping patience |
| `--name` | ultimate_model | Experiment name (used in output dir) |

---

## 5. Reading Results

### `config.json` — What was run
Contains all hyperparameters and pipeline choices for reproducibility.

### `summary.csv` — Key metrics
| Column | What it means |
|--------|---------------|
| `pooled_accuracy` | Accuracy on pooled test set (all train stocks' future data) |
| `pooled_lift` | Accuracy minus zero-rule baseline (positive = better than guessing) |
| `pooled_mcc` | Matthews Correlation Coefficient (−1 to +1, 0 = random) |
| `per_stock_avg_accuracy` | Average accuracy across the held-out test stocks |
| `per_stock_avg_lift` | Average lift across test stocks (negative = doesn't generalize) |
| `per_stock_avg_mcc` | Average MCC across test stocks |

### `per_stock_results.csv` — Per-stock breakdown
Individual metrics for each held-out stock. Look for:
- **Stocks with positive lift**: The model adds value for these
- **Stocks with negative lift**: The model is worse than guessing majority class

### What "good" looks like
From prior experiments, the current best is:
- **Pooled**: 44.5-44.7% accuracy, +4.8-9.2% lift, 0.15-0.18 MCC
- **Per-stock**: This is the weak point — typically negative lift

---

## 6. Key Insights for Future Experiments

### Things worth exploring
1. **Volatility-adjusted forward return** as target variable (instead of fixed thresholds)
2. **Training pool composition**: curated 100 idiosyncratic stocks vs. macro-representative set
3. **Small model on daily data**: SimpleTrendLSTM(h=32) hasn't been tested on daily pooled yet
4. **Walk-forward validation**: expanding window instead of single static split
5. **Different feature sets**: try subsets of SHAP top 20, or entirely different indicators
6. **Confidence head re-enablement**: filter predictions by confidence score

### Things already tested (see reports/ and archive/)
| Experiment | Finding |
|-----------|---------|
| `hyperparameter_tuning` | dropout=0.2, plateau scheduler, label_smoothing=0.1 is optimal |
| `model_size_comparison` | SimpleLSTM (9K params) ≥ AttentionLSTM (47K params) |
| `selected_stock_eval` | Daily per-stock: no generalization (39.6% acc, -1.6% lift) |
| `selected_stock_eval_weekly` | Weekly pooled: +4.4% lift, but -2.9% per-stock lift |
| `feature_selection_experiment` | SHAP top 20 features provide best signal |
| `class_balance_optimization` | ATR-based thresholds can balance classes |

### Rules of thumb
- **Lift > 0 is the minimum bar** — accuracy alone is meaningless without beating zero-rule
- **MCC > 0.1 indicates real signal** — MCC corrects for class imbalance
- **Per-stock lift matters more than pooled lift** for any real trading application
- **More parameters ≠ better** — this problem is low-SNR, small models generalize better

---

## 7. Git Workflow

After creating and running a new experiment:

```bash
# 1. Add the experiment script
git add experiments/my_experiment.py

# 2. Add results (optional — consider gitignoring large model.pt files)
git add reports/my_experiment_*/config.json reports/my_experiment_*/summary.csv
git add reports/my_experiment_*/per_stock_results.csv

# 3. Commit with a descriptive message
git commit -m "Experiment: test volatility-adjusted targets with SimpleLSTM"
```

When an experiment is complete and you've extracted all insights, move it:
```bash
mv experiments/my_experiment.py experiments/archive/
git add -A && git commit -m "Archive my_experiment"
```
