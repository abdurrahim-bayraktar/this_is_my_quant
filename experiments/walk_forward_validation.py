"""
Walk-Forward Validation Experiment.

Self-contained experiment using shared utilities.
Expanding-window walk-forward: trains and evaluates across multiple temporal
folds to test model stability over time.

Configurable:
    --n-folds N               Number of walk-forward folds (default 4)
    --fold-years Y            Length of each test window in years (default 1)
    --first-train-end DATE    When the first fold's training ends (default 2019-01-01)
    --frequency daily|weekly  Data frequency
    --model simple|attention  Model architecture
    --stocks N                Number of training stocks
    --threshold LOW HIGH      Trend classification thresholds
    --test-stocks TICKER...   Stocks to hold out for per-stock eval

Usage:
    python experiments/walk_forward_validation.py --epochs 100 --stocks 400
    python experiments/walk_forward_validation.py --epochs 2 --stocks 5 --n-folds 2  # smoke test
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import logging
import time
import gc
from datetime import datetime
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from collections import Counter

from experiments.ranked_tickers import EXTENDED_TICKERS
from src.data.cache import PriceCache
from src.data.utils import resample_to_weekly, load_stocks, create_sequences
from src.evaluation.metrics import compute_metrics, evaluate_model_on_data
from src.features.indicators_v7 import ComprehensiveIndicatorsV7

try:
    from config import REPORTS_DIR
except ImportError:
    REPORTS_DIR = Path("reports")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# === CUDA Diagnostics ===
logger.info(f"PyTorch {torch.__version__}")
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    _, total_vram = torch.cuda.mem_get_info(0)
    logger.info(f"CUDA: {gpu} ({total_vram / 1024**3:.1f} GB VRAM)")
else:
    logger.info("CUDA: Not available — running on CPU")

# ============================================================================
# FEATURES
# ============================================================================

SHAP_TOP20_FEATURES = [
    "atr_pct", "intraday_range", "bb_BBB_5_2.0_2.0", "gap", "cci",
    "high_low_pct", "volume_ratio", "adx_DMN_14", "return_1d", "aroon_AROOND_14",
    "adx_DMP_14", "obv", "ad", "bear_power", "roc_10",
    "pvo_PVOh_12_26_9", "trix_TRIXs_30_9", "rsi_14", "aroon_AROONU_14", "return_5d"
]

DEFAULT_TEST_STOCKS = [
    "MSFT", "AAPL", "JPM", "JNJ", "XOM",
    "WMT", "DIS", "BA", "KO", "NVDA",
]

# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_model(model_type: str, input_size: int, dropout: float = 0.2) -> nn.Module:
    """Create model by type name."""
    if model_type == "simple":
        from src.models.simple_lstm import SimpleTrendLSTM
        return SimpleTrendLSTM(
            input_size=input_size,
            hidden_size=32,
            num_layers=1,
            dropout=dropout,
        )
    elif model_type == "attention":
        from src.models import AttentionLSTM
        return AttentionLSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ============================================================================
# TRAINER (inline — can be modified per experiment)
# ============================================================================

class Trainer:
    """Training loop with ReduceLROnPlateau scheduler."""
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.1,
    ):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        self.scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None
        
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
    
    def train_epoch(self, train_loader: DataLoader) -> Dict:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        n_batches = 0
        epoch_start = time.time()
        
        for X, y in train_loader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            
            if self.scaler:
                with torch.amp.autocast(device_type='cuda'):
                    trend_logits, _, _ = self.model(X)
                    loss = self.loss_fn(trend_logits, y)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                trend_logits, _, _ = self.model(X)
                loss = self.loss_fn(trend_logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            total_loss += loss.item() * len(X)
            preds = trend_logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += len(y)
            n_batches += 1
        
        elapsed = time.time() - epoch_start
        return {
            "loss": total_loss / total,
            "accuracy": correct / total,
            "elapsed": elapsed,
            "it_per_sec": n_batches / elapsed if elapsed > 0 else 0,
            "samples_per_sec": total / elapsed if elapsed > 0 else 0,
        }
    
    def validate(self, val_loader: DataLoader) -> Dict:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(self.device), y.to(self.device)
                trend_logits, _, _ = self.model(X)
                loss = self.loss_fn(trend_logits, y)
                total_loss += loss.item() * len(X)
                preds = trend_logits.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += len(y)
        
        val_loss = total_loss / total
        self.scheduler.step(val_loss)
        return {"loss": val_loss, "accuracy": correct / total}
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        patience: int = 15,
    ) -> Dict:
        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                lr = self.optimizer.param_groups[0]['lr']
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train: {train_metrics['loss']:.4f} ({train_metrics['accuracy']:.2%}) | "
                    f"Val: {val_metrics['loss']:.4f} ({val_metrics['accuracy']:.2%}) | "
                    f"{train_metrics['it_per_sec']:.1f} it/s, "
                    f"{train_metrics['samples_per_sec']:.0f} samples/s | "
                    f"LR: {lr:.2e}"
                )
            
            if self.patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            self.model.to(self.device)
        
        return {"best_val_loss": self.best_val_loss}


# ============================================================================
# WALK-FORWARD FOLD GENERATION
# ============================================================================

def generate_folds(
    first_train_end: str,
    fold_years: int,
    n_folds: int,
) -> List[Dict[str, pd.Timestamp]]:
    """
    Generate expanding-window walk-forward fold definitions.
    
    Each fold has:
        train_end  — end of training window (expanding)
        val_end    — end of validation window (= train_end + fold_years)
        test_end   — end of test window (= val_end + fold_years)
    
    The training window always starts at the dataset start date; only the
    cutoff advances.
    """
    folds = []
    base = pd.Timestamp(first_train_end)
    
    for i in range(n_folds):
        offset_years = i * fold_years
        train_end = base + pd.DateOffset(years=offset_years)
        val_end = train_end + pd.DateOffset(years=fold_years)
        test_end = val_end + pd.DateOffset(years=fold_years)
        
        folds.append({
            "fold": i + 1,
            "train_end": train_end,
            "val_end": val_end,
            "test_end": test_end,
        })
    
    return folds


# ============================================================================
# EXPERIMENT
# ============================================================================

class WalkForwardExperiment:
    """
    Walk-forward validation experiment.
    
    Pipeline per fold:
    1. Load stocks (shared across folds)
    2. Compute SHAP Top 20 features
    3. Create labels with configurable thresholds
    4. Pool all stocks, temporal split for this fold
    5. Train a fresh SimpleLSTM or AttentionLSTM
    6. Evaluate: pooled test + per-stock test
    7. Save per-fold results
    
    After all folds: aggregate summary across folds.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        self.train_stocks = config.get("train_stocks", EXTENDED_TICKERS[:400])
        self.test_stocks = config.get("test_stocks", DEFAULT_TEST_STOCKS)
        self.train_stocks = [t for t in self.train_stocks if t not in self.test_stocks]
        
        self.start_date = config.get("start_date", "2014-01-01")
        self.end_date = config.get("end_date", "2024-12-31")
        
        self.frequency = config.get("frequency", "daily")
        self.sequence_length = config.get("sequence_length", 20 if self.frequency == "daily" else 12)
        self.model_type = config.get("model_type", "simple")
        self.epochs = config.get("epochs", 100)
        self.batch_size = config.get("batch_size", 1024 if self.frequency == "daily" else 512)
        self.dropout = config.get("dropout", 0.2)
        self.label_smoothing = config.get("label_smoothing", 0.1)
        self.learning_rate = config.get("learning_rate", 5e-4)
        self.patience = config.get("patience", 15)
        
        # Trend thresholds 
        if self.frequency == "weekly":
            self.threshold_low = config.get("threshold_low", -0.01)
            self.threshold_high = config.get("threshold_high", 0.01)
        else:
            self.threshold_low = config.get("threshold_low", -0.005)
            self.threshold_high = config.get("threshold_high", 0.005)
        
        self.feature_cols = config.get("features", SHAP_TOP20_FEATURES)
        
        # Walk-forward fold definitions
        self.folds = generate_folds(
            first_train_end=config.get("first_train_end", "2019-01-01"),
            fold_years=config.get("fold_years", 1),
            n_folds=config.get("n_folds", 4),
        )
        
        # Output directory
        exp_name = config.get("experiment_name", "walk_forward")
        self.output_dir = REPORTS_DIR / f"{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.indicator_computer = ComprehensiveIndicatorsV7()
        self.cache = PriceCache()
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all stock data."""
        all_tickers = list(set(self.train_stocks + self.test_stocks))
        weekly = self.frequency == "weekly"
        
        return load_stocks(
            all_tickers,
            self.start_date,
            self.end_date,
            cache=self.cache,
            min_rows=100 if not weekly else 52,
            weekly=weekly,
        )
    
    def prepare_stock(self, df: pd.DataFrame, train_end: pd.Timestamp, val_end: pd.Timestamp) -> Tuple:
        """Compute features and create sequences for a single stock with given split dates."""
        df = self.indicator_computer.compute_shap_top20(df)
        
        available = [c for c in self.feature_cols if c in df.columns]
        if len(available) < 5:
            return tuple(np.array([]) for _ in range(6))
        
        df['return_next'] = df['Close'].pct_change().shift(-1)
        df['trend'] = pd.cut(
            df['return_next'],
            bins=[-np.inf, self.threshold_low, self.threshold_high, np.inf],
            labels=[0, 1, 2]
        ).astype(float)
        
        df = df.dropna(subset=['trend'] + available[:5])
        if len(df) < self.sequence_length + 10:
            return tuple(np.array([]) for _ in range(6))
        
        feature_data = df[available].values
        feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
        labels = df['trend'].values
        
        return create_sequences(
            feature_data, labels, df.index,
            self.sequence_length, train_end, val_end,
        )
    
    def prepare_pooled_data(
        self, stock_data: Dict, tickers: List[str],
        train_end: pd.Timestamp, val_end: pd.Timestamp,
    ) -> Tuple:
        """Prepare pooled train/val/test arrays from multiple stocks for a given fold."""
        all_train_X, all_train_y = [], []
        all_val_X, all_val_y = [], []
        all_test_X, all_test_y = [], []
        n_features = None
        
        stocks = {t: stock_data[t] for t in tickers if t in stock_data}
        
        for ticker, df in tqdm(stocks.items(), desc="Processing stocks"):
            try:
                X_tr, y_tr, X_val, y_val, X_te, y_te = self.prepare_stock(df, train_end, val_end)
                
                if len(X_tr) == 0:
                    continue
                
                if n_features is None:
                    n_features = X_tr.shape[-1]
                elif X_tr.shape[-1] != n_features:
                    continue
                
                all_train_X.append(X_tr)
                all_train_y.append(y_tr)
                if len(X_val) > 0:
                    all_val_X.append(X_val)
                    all_val_y.append(y_val)
                if len(X_te) > 0:
                    all_test_X.append(X_te)
                    all_test_y.append(y_te)
                    
            except Exception as e:
                logger.warning(f"Failed {ticker}: {e}")
        
        X_train = np.concatenate(all_train_X, axis=0)
        y_train = np.concatenate(all_train_y, axis=0)
        X_val = np.concatenate(all_val_X) if all_val_X else np.array([])
        y_val = np.concatenate(all_val_y) if all_val_y else np.array([])
        X_test = np.concatenate(all_test_X) if all_test_X else np.array([])
        y_test = np.concatenate(all_test_y) if all_test_y else np.array([])
        
        logger.info(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}, Features: {n_features}")
        logger.info(f"Class distribution (train): {Counter(y_train)}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test, n_features
    
    def scale_data(self, X_train, X_val, X_test):
        """Fit scaler on train, transform all splits. Returns scaler for per-stock eval."""
        n_train, seq_len, n_features = X_train.shape
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train.reshape(-1, n_features))
        X_train_s = np.nan_to_num(X_train_s, nan=0.0, posinf=0.0, neginf=0.0)
        X_train_s = X_train_s.reshape(n_train, seq_len, n_features)
        
        def _transform(X):
            if len(X) == 0:
                return X
            s = X.shape
            X_t = scaler.transform(X.reshape(-1, n_features))
            return np.nan_to_num(X_t, nan=0.0, posinf=0.0, neginf=0.0).reshape(s)
        
        return X_train_s, _transform(X_val), _transform(X_test), scaler
    
    def evaluate_single_stock(
        self, model, stock_data, ticker,
        train_end: pd.Timestamp, val_end: pd.Timestamp,
        scaler: StandardScaler,
    ) -> Dict:
        """Evaluate model on a single held-out stock's test data for a given fold."""
        if ticker not in stock_data:
            return {"ticker": ticker, "error": "Not found"}
        
        _, _, _, _, X_test, y_test = self.prepare_stock(stock_data[ticker], train_end, val_end)
        
        if len(X_test) == 0:
            return {"ticker": ticker, "error": "No test data"}
        
        # Scale with fold's training scaler
        n, seq, feat = X_test.shape
        X_scaled = scaler.transform(X_test.reshape(-1, feat))
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0).reshape(n, seq, feat)
        
        metrics = evaluate_model_on_data(model, X_scaled, y_test)
        metrics["ticker"] = ticker
        metrics.pop("predictions", None)
        metrics.pop("class_distribution", None)
        return metrics
    
    def run_fold(
        self, fold_def: Dict, stock_data: Dict,
    ) -> Dict:
        """Execute a single walk-forward fold: train, evaluate, save results."""
        fold_num = fold_def["fold"]
        train_end = fold_def["train_end"]
        val_end = fold_def["val_end"]
        test_end = fold_def["test_end"]
        
        logger.info(f"\n{'=' * 80}")
        logger.info(f"FOLD {fold_num}: Train → {train_end.date()}, "
                     f"Val → {val_end.date()}, Test → {test_end.date()}")
        logger.info(f"{'=' * 80}")
        
        fold_dir = self.output_dir / f"fold_{fold_num}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare pooled training data
        X_train, y_train, X_val, y_val, X_test, y_test, n_features = \
            self.prepare_pooled_data(stock_data, self.train_stocks, train_end, val_end)
        
        if len(X_val) == 0:
            logger.warning(f"Fold {fold_num}: No validation data — skipping")
            return {"fold": fold_num, "error": "No validation data"}
        if len(X_test) == 0:
            logger.warning(f"Fold {fold_num}: No test data — skipping")
            return {"fold": fold_num, "error": "No test data"}
        
        # Scale
        X_train_s, X_val_s, X_test_s, scaler = self.scale_data(X_train, X_val, X_test)
        del X_train, X_val, X_test
        gc.collect()
        
        # Create a fresh model for this fold
        model = create_model(self.model_type, n_features, self.dropout)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {n_params:,}")
        
        # Train
        trainer = Trainer(
            model,
            learning_rate=self.learning_rate,
            label_smoothing=self.label_smoothing,
        )
        
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train_s), torch.LongTensor(y_train)),
            batch_size=self.batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val_s), torch.LongTensor(y_val)),
            batch_size=self.batch_size,
        )
        
        start_time = time.time()
        trainer.train(train_loader, val_loader, epochs=self.epochs, patience=self.patience)
        train_time = time.time() - start_time
        logger.info(f"Fold {fold_num} training time: {train_time:.1f}s")
        
        # Free training tensors
        del X_train_s, X_val_s, train_loader, val_loader
        gc.collect()
        
        # === POOLED TEST EVALUATION ===
        pooled_metrics = evaluate_model_on_data(model, X_test_s, y_test)
        pooled_metrics.pop("predictions", None)
        pooled_metrics.pop("class_distribution", None)
        
        if "error" not in pooled_metrics:
            logger.info(f"Fold {fold_num} Pooled: n={pooled_metrics['n_samples']:,}, "
                        f"Acc={pooled_metrics['accuracy']:.2%}, "
                        f"Lift={pooled_metrics['lift']:+.2%}, "
                        f"MCC={pooled_metrics['mcc']:.4f}")
        
        # === PER-STOCK EVALUATION ===
        per_stock_results = []
        for ticker in self.test_stocks:
            result = self.evaluate_single_stock(model, stock_data, ticker, train_end, val_end, scaler)
            per_stock_results.append(result)
            if "error" not in result:
                logger.info(f"  {ticker}: Acc={result['accuracy']:.2%}, "
                            f"Lift={result['lift']:+.2%}, "
                            f"MCC={result['mcc']:.4f}, n={result['n_samples']}")
            else:
                logger.warning(f"  {ticker}: {result['error']}")
        
        # === SAVE FOLD RESULTS ===
        valid = [r for r in per_stock_results if "error" not in r]
        avg_acc = np.mean([r["accuracy"] for r in valid]) if valid else 0
        avg_lift = np.mean([r["lift"] for r in valid]) if valid else 0
        avg_mcc = np.mean([r["mcc"] for r in valid]) if valid else 0
        
        fold_summary = {
            "fold": fold_num,
            "train_end": str(train_end.date()),
            "val_end": str(val_end.date()),
            "test_end": str(test_end.date()),
            "train_time": train_time,
            "n_params": n_params,
            # Pooled metrics
            "pooled_accuracy": pooled_metrics.get("accuracy", 0),
            "pooled_zero_rule": pooled_metrics.get("zero_rule", 0),
            "pooled_lift": pooled_metrics.get("lift", 0),
            "pooled_mcc": pooled_metrics.get("mcc", 0),
            "pooled_f1_macro": pooled_metrics.get("f1_macro", 0),
            "pooled_n_samples": pooled_metrics.get("n_samples", 0),
            # Per-stock averages
            "per_stock_avg_accuracy": avg_acc,
            "per_stock_avg_lift": avg_lift,
            "per_stock_avg_mcc": avg_mcc,
            "per_stock_count": len(valid),
        }
        
        pd.DataFrame([fold_summary]).to_csv(fold_dir / "summary.csv", index=False)
        pd.DataFrame(per_stock_results).to_csv(fold_dir / "per_stock_results.csv", index=False)
        torch.save(model.state_dict(), fold_dir / "model.pt")
        
        # Cleanup
        del model, trainer, X_test_s, scaler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return fold_summary
    
    def run(self):
        """Execute the full walk-forward pipeline across all folds."""
        logger.info("=" * 80)
        logger.info("WALK-FORWARD VALIDATION EXPERIMENT")
        logger.info("=" * 80)
        logger.info(f"Model: {self.model_type}, Frequency: {self.frequency}")
        logger.info(f"Train stocks: {len(self.train_stocks)}, Test stocks: {self.test_stocks}")
        logger.info(f"Thresholds: [{self.threshold_low}, {self.threshold_high}]")
        logger.info(f"Folds: {len(self.folds)}")
        for f in self.folds:
            logger.info(f"  Fold {f['fold']}: Train → {f['train_end'].date()}, "
                        f"Val → {f['val_end'].date()}, Test → {f['test_end'].date()}")
        
        # Save config
        config_to_save = {k: v for k, v in self.config.items()
                         if not isinstance(v, (list,)) or len(v) < 20}
        config_to_save["train_stock_count"] = len(self.train_stocks)
        config_to_save["test_stocks"] = self.test_stocks
        config_to_save["folds"] = [
            {k: str(v) for k, v in f.items()} for f in self.folds
        ]
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(config_to_save, f, indent=2, default=str)
        
        # Load data once (shared across folds)
        stock_data = self.load_data()
        
        # Run each fold
        fold_summaries = []
        for fold_def in self.folds:
            result = self.run_fold(fold_def, stock_data)
            fold_summaries.append(result)
        
        # === AGGREGATE RESULTS ===
        valid_folds = [s for s in fold_summaries if "error" not in s]
        
        if not valid_folds:
            logger.error("No valid folds completed!")
            return fold_summaries
        
        all_folds_df = pd.DataFrame(valid_folds)
        all_folds_df.to_csv(self.output_dir / "all_folds_summary.csv", index=False)
        
        # Compute aggregate stats
        metric_cols = [
            "pooled_accuracy", "pooled_lift", "pooled_mcc", "pooled_f1_macro",
            "per_stock_avg_accuracy", "per_stock_avg_lift", "per_stock_avg_mcc",
        ]
        
        agg_rows = []
        for col in metric_cols:
            values = [s[col] for s in valid_folds if col in s]
            if values:
                agg_rows.append({
                    "metric": col,
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "n_folds": len(values),
                })
        
        agg_df = pd.DataFrame(agg_rows)
        agg_df.to_csv(self.output_dir / "aggregate_summary.csv", index=False)
        
        # Final summary log
        logger.info(f"\n{'=' * 60}")
        logger.info("WALK-FORWARD AGGREGATE SUMMARY")
        logger.info(f"{'=' * 60}")
        logger.info(f"Completed folds: {len(valid_folds)} / {len(self.folds)}")
        
        for _, row in agg_df.iterrows():
            logger.info(f"  {row['metric']}: {row['mean']:.4f} ± {row['std']:.4f} "
                        f"[{row['min']:.4f}, {row['max']:.4f}]")
        
        logger.info(f"Results saved to: {self.output_dir}")
        
        return fold_summaries


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Validation Experiment")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--stocks", type=int, default=400, help="Number of training stocks")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--frequency", choices=["daily", "weekly"], default="daily")
    parser.add_argument("--model", choices=["simple", "attention"], default="simple")
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--threshold", type=float, nargs=2, default=None,
                        metavar=("LOW", "HIGH"), help="Trend thresholds e.g. -0.005 0.005")
    parser.add_argument("--test-stocks", nargs="+", default=None)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--name", type=str, default="walk_forward", help="Experiment name")
    # Walk-forward specific args
    parser.add_argument("--n-folds", type=int, default=4, help="Number of walk-forward folds")
    parser.add_argument("--fold-years", type=int, default=1, help="Length of each test window in years")
    parser.add_argument("--first-train-end", type=str, default="2019-01-01",
                        help="When the first fold's training period ends")
    args = parser.parse_args()
    
    config = {
        "experiment_name": args.name,
        "model_type": args.model,
        "frequency": args.frequency,
        "train_stocks": EXTENDED_TICKERS[:args.stocks],
        "test_stocks": args.test_stocks or DEFAULT_TEST_STOCKS,
        "epochs": args.epochs,
        "dropout": args.dropout,
        "label_smoothing": args.label_smoothing,
        "learning_rate": args.lr,
        "patience": args.patience,
        # Walk-forward config
        "n_folds": args.n_folds,
        "fold_years": args.fold_years,
        "first_train_end": args.first_train_end,
    }
    
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.seq_length:
        config["sequence_length"] = args.seq_length
    if args.threshold:
        config["threshold_low"] = args.threshold[0]
        config["threshold_high"] = args.threshold[1]
    
    experiment = WalkForwardExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
