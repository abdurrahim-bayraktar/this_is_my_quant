"""
Hyperparameter Tuning Experiment - Grid Search with CombinedLoss.

Tests combinations of:
- Dropout: 0.2, 0.3, 0.4
- Scheduler: CosineAnnealing, ReduceLROnPlateau  
- Label Smoothing: 0.0, 0.1
- Confidence Weight: 0.0 (disabled), 0.5

Total: 24 experiments

Usage:
    python experiments/hyperparameter_tuning.py --epochs 100 --stocks 400
    python experiments/hyperparameter_tuning.py --epochs 50 --stocks 200 --quick  # Quick test (5 configs)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import pickle
import itertools
from datetime import datetime
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, f1_score, matthews_corrcoef
from tqdm import tqdm
from collections import Counter
import time
import gc

from experiments.ranked_tickers import EXTENDED_TICKERS

# SHAP Top 20 features
SHAP_TOP20_FEATURES = [
    "atr_pct", "intraday_range", "bb_BBB_5_2.0_2.0", "gap", "cci",
    "high_low_pct", "volume_ratio", "adx_DMN_14", "return_1d", "aroon_AROOND_14",
    "adx_DMP_14", "obv", "ad", "bear_power", "roc_10",
    "pvo_PVOh_12_26_9", "trix_TRIXs_30_9", "rsi_14", "aroon_AROONU_14", "return_5d"
]

try:
    from config import REPORTS_DIR, DATA_DIR, training_config
    from src.models import AttentionLSTM
    from src.models.losses import CombinedLoss
    from src.features.indicators_v7 import ComprehensiveIndicatorsV7
    RUNNING_LOCAL = True
except ImportError:
    RUNNING_LOCAL = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logger.info(f"PyTorch Version: {torch.__version__}")
if torch.cuda.is_available():
    logger.info(f"CUDA: {torch.cuda.get_device_name(0)}")


# =============================================================================
# Hyperparameter Configurations
# =============================================================================

def get_hyperparameter_grid(quick: bool = False):
    """Return hyperparameter grid to search."""
    grid = {
        "dropout": [0.2, 0.3, 0.4],
        "scheduler": ["cosine", "plateau"],
        "label_smoothing": [0.0, 0.1],
        "confidence_weight": [0.0, 0.5],
    }
    
    # Generate all combinations
    keys = list(grid.keys())
    values = list(grid.values())
    configs = []
    
    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))
        configs.append(config)
    
    if quick:
        # Select representative subset
        configs = [
            {"dropout": 0.2, "scheduler": "plateau", "label_smoothing": 0.0, "confidence_weight": 0.5},
            {"dropout": 0.3, "scheduler": "cosine", "label_smoothing": 0.1, "confidence_weight": 0.0},
            {"dropout": 0.4, "scheduler": "plateau", "label_smoothing": 0.0, "confidence_weight": 0.0},
            {"dropout": 0.2, "scheduler": "cosine", "label_smoothing": 0.1, "confidence_weight": 0.5},
            {"dropout": 0.3, "scheduler": "plateau", "label_smoothing": 0.0, "confidence_weight": 0.5},
        ]
    
    return configs


class PriceCache:
    """Cache for yfinance price data."""
    
    def __init__(self, cache_dir: Path = None):
        if RUNNING_LOCAL:
            self.cache_dir = cache_dir or DATA_DIR / "price_cache"
        else:
            self.cache_dir = cache_dir or Path("/content/data/price_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_path(self, start_date: str, end_date: str) -> Path:
        return self.cache_dir / f"prices_{start_date}_{end_date}.pkl"
    
    def load(self, start_date: str, end_date: str, requested_stocks: List[str]) -> Dict[str, pd.DataFrame]:
        cache_path = self.get_cache_path(start_date, end_date)
        if cache_path.exists():
            logger.info(f"Loading cached prices from {cache_path}")
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            return cached
        return None
    
    def save(self, data: Dict[str, pd.DataFrame], start_date: str, end_date: str):
        cache_path = self.get_cache_path(start_date, end_date)
        if not cache_path.exists():
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)


class CustomTrainer:
    """Trainer with configurable scheduler and CombinedLoss support."""
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-4,
        scheduler_type: str = "cosine",
        mixed_precision: bool = True,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.scheduler_type = scheduler_type
        self.mixed_precision = mixed_precision and torch.cuda.is_available()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        
        # Setup scheduler based on type
        if scheduler_type == "cosine":
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )
            self.scheduler_step_per_epoch = True
        else:  # plateau
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
            )
            self.scheduler_step_per_epoch = False
        
        self.scaler = torch.amp.GradScaler() if self.mixed_precision else None
        
        # Check if using CombinedLoss
        self._is_combined_loss = hasattr(loss_fn, 'trend_weight')
        
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
    
    def train_epoch(self, train_loader: DataLoader) -> Dict:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for X, y in train_loader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                with torch.amp.autocast(device_type='cuda'):
                    trend_logits, confidence, _ = self.model(X)
                    if self._is_combined_loss:
                        loss, _ = self.loss_fn(trend_logits, confidence, y)
                    else:
                        loss = self.loss_fn(trend_logits, y)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                trend_logits, confidence, _ = self.model(X)
                if self._is_combined_loss:
                    loss, _ = self.loss_fn(trend_logits, confidence, y)
                else:
                    loss = self.loss_fn(trend_logits, y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            total_loss += loss.item() * len(X)
            preds = trend_logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += len(y)
        
        if self.scheduler_step_per_epoch:
            self.scheduler.step()
        
        return {
            "loss": total_loss / total,
            "accuracy": correct / total,
        }
    
    def validate(self, val_loader: DataLoader) -> Dict:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(self.device), y.to(self.device)
                trend_logits, confidence, _ = self.model(X)
                
                if self._is_combined_loss:
                    loss, _ = self.loss_fn(trend_logits, confidence, y)
                else:
                    loss = self.loss_fn(trend_logits, y)
                
                total_loss += loss.item() * len(X)
                preds = trend_logits.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += len(y)
        
        val_loss = total_loss / total
        
        # Update scheduler if using plateau
        if not self.scheduler_step_per_epoch:
            self.scheduler.step(val_loss)
        
        return {
            "loss": val_loss,
            "accuracy": correct / total,
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        patience: int = 15,
    ) -> Dict:
        history = []
        
        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            
            history.append({
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
            })
            
            # Early stopping
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train: {train_metrics['loss']:.4f} ({train_metrics['accuracy']:.2%}) | "
                    f"Val: {val_metrics['loss']:.4f} ({val_metrics['accuracy']:.2%})"
                )
            
            if self.patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            self.model.to(self.device)
        
        return {"history": history}


class HyperparameterTuningExperiment:
    """Grid search over hyperparameters with CombinedLoss."""
    
    def __init__(
        self,
        stocks: List[str] = None,
        start_date: str = "2014-01-01",
        end_date: str = "2024-12-31",
        train_end_date: str = "2022-01-01",
        val_end_date: str = "2023-06-01",
        sequence_length: int = 20,
        epochs: int = 100,
        batch_size: int = 1024,
        early_stopping_patience: int = 15,
        learning_rate: float = 5e-4,
        use_cache: bool = True,
        quick: bool = False,
    ):
        self.stocks = stocks or EXTENDED_TICKERS[:400]
        self.start_date = start_date
        self.end_date = end_date
        self.train_end_date = pd.Timestamp(train_end_date)
        self.val_end_date = pd.Timestamp(val_end_date)
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.learning_rate = learning_rate
        self.use_cache = use_cache
        self.quick = quick
        
        self.output_dir = REPORTS_DIR / f"hyperparameter_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.indicator_computer = ComprehensiveIndicatorsV7()
        self.price_cache = PriceCache()
        self.feature_cols = SHAP_TOP20_FEATURES
        
    def load_all_stocks(self) -> Dict[str, pd.DataFrame]:
        """Load stocks from cache or download."""
        stock_data = {}
        
        if self.use_cache:
            cached = self.price_cache.load(self.start_date, self.end_date, self.stocks)
            if cached:
                for ticker, df in cached.items():
                    if ticker in self.stocks:
                        stock_data[ticker] = df
                logger.info(f"Loaded {len(stock_data)} stocks from cache")
        
        missing = [t for t in self.stocks if t not in stock_data]
        if missing:
            batch_size = 100
            for i in range(0, len(missing), batch_size):
                batch = missing[i:i + batch_size]
                try:
                    data = yf.download(batch, start=self.start_date, end=self.end_date,
                                       group_by="ticker", threads=True, progress=True)
                    for ticker in batch:
                        try:
                            df = data[ticker].copy() if len(batch) > 1 else data.copy()
                            df = df.dropna()
                            if len(df) < 100:
                                continue
                            df.columns = [c.capitalize() if isinstance(c, str) else c for c in df.columns]
                            df['Ticker'] = ticker
                            stock_data[ticker] = df
                        except:
                            pass
                except Exception as e:
                    logger.warning(f"Batch download failed: {e}")
            
            if self.use_cache and stock_data:
                self.price_cache.save(stock_data, self.start_date, self.end_date)
        
        logger.info(f"Total: {len(stock_data)} stocks ready")
        return stock_data
    
    def prepare_stock_with_temporal_split(self, df: pd.DataFrame) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """Split stock by date, create sequences using SHAP top 20 features."""
        df = self.indicator_computer.compute_shap_top20(df)
        
        available_features = [col for col in self.feature_cols if col in df.columns]
        if len(available_features) < 5:
            return (np.array([]), np.array([]), np.array([]), 
                    np.array([]), np.array([]), np.array([]))
        
        df['return_next'] = df['Close'].pct_change().shift(-1)
        df['trend'] = pd.cut(
            df['return_next'],
            bins=[-np.inf, -0.005, 0.005, np.inf],
            labels=[0, 1, 2]
        ).astype(float)
        
        df = df.dropna(subset=['trend'] + available_features[:5])
        if len(df) < self.sequence_length + 10:
            return (np.array([]), np.array([]), np.array([]), 
                    np.array([]), np.array([]), np.array([]))
        
        feature_data = df[available_features].values
        feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
        labels = df['trend'].values
        dates = df.index
        
        X_train, y_train = [], []
        X_val, y_val = [], []
        X_test, y_test = [], []
        
        for i in range(len(feature_data) - self.sequence_length):
            target_date = dates[i + self.sequence_length]
            seq = feature_data[i:i + self.sequence_length]
            label = labels[i + self.sequence_length - 1]
            
            if target_date < self.train_end_date:
                X_train.append(seq)
                y_train.append(label)
            elif target_date < self.val_end_date:
                X_val.append(seq)
                y_val.append(label)
            else:
                X_test.append(seq)
                y_test.append(label)
        
        return (
            np.array(X_train, dtype=np.float32) if X_train else np.array([]),
            np.array(y_train, dtype=np.int64) if y_train else np.array([]),
            np.array(X_val, dtype=np.float32) if X_val else np.array([]),
            np.array(y_val, dtype=np.int64) if y_val else np.array([]),
            np.array(X_test, dtype=np.float32) if X_test else np.array([]),
            np.array(y_test, dtype=np.int64) if y_test else np.array([]),
        )
    
    def prepare_temporal_pooled_data(self, stock_data: Dict[str, pd.DataFrame]) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int
    ]:
        """Process all stocks with temporal split."""
        all_train_X, all_train_y = [], []
        all_val_X, all_val_y = [], []
        all_test_X, all_test_y = [], []
        n_features = None
        success_count = 0
        
        for ticker, df in tqdm(stock_data.items(), desc="Processing stocks"):
            try:
                X_tr, y_tr, X_val, y_val, X_te, y_te = self.prepare_stock_with_temporal_split(df)
                
                if len(X_tr) == 0 and len(X_val) == 0 and len(X_te) == 0:
                    continue
                
                if n_features is None:
                    if len(X_tr) > 0:
                        n_features = X_tr.shape[-1]
                    elif len(X_val) > 0:
                        n_features = X_val.shape[-1]
                    elif len(X_te) > 0:
                        n_features = X_te.shape[-1]
                
                current_n = X_tr.shape[-1] if len(X_tr) > 0 else (X_val.shape[-1] if len(X_val) > 0 else X_te.shape[-1])
                if current_n != n_features:
                    continue
                    
                if len(X_tr) > 0:
                    all_train_X.append(X_tr)
                    all_train_y.append(y_tr)
                if len(X_val) > 0:
                    all_val_X.append(X_val)
                    all_val_y.append(y_val)
                if len(X_te) > 0:
                    all_test_X.append(X_te)
                    all_test_y.append(y_te)
                    
                success_count += 1
                    
            except Exception as e:
                if success_count < 3:
                    logger.error(f"Failed {ticker}: {e}")
        
        logger.info(f"Processed: {success_count}/{len(stock_data)} stocks, {n_features} features")
        
        X_train = np.concatenate(all_train_X, axis=0)
        y_train = np.concatenate(all_train_y, axis=0)
        X_val = np.concatenate(all_val_X, axis=0) if all_val_X else np.array([])
        y_val = np.concatenate(all_val_y, axis=0) if all_val_y else np.array([])
        X_test = np.concatenate(all_test_X, axis=0) if all_test_X else np.array([])
        y_test = np.concatenate(all_test_y, axis=0) if all_test_y else np.array([])
        
        del all_train_X, all_train_y, all_val_X, all_val_y, all_test_X, all_test_y
        gc.collect()
        
        logger.info(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test, n_features
    
    def scale_data(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray):
        """Scale data - fit on train only."""
        n_train, seq_len, n_features = X_train.shape
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, n_features))
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_train_scaled = X_train_scaled.reshape(n_train, seq_len, n_features)
        
        if len(X_val) > 0:
            X_val_scaled = scaler.transform(X_val.reshape(-1, n_features))
            X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            X_val_scaled = X_val_scaled.reshape(X_val.shape)
        else:
            X_val_scaled = X_val
            
        if len(X_test) > 0:
            X_test_scaled = scaler.transform(X_test.reshape(-1, n_features))
            X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            X_test_scaled = X_test_scaled.reshape(X_test.shape)
        else:
            X_test_scaled = X_test
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def predict_batched(self, model: nn.Module, X: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        model.eval()
        device = next(model.parameters()).device
        all_preds = []
        
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size]
            with torch.no_grad():
                batch_tensor = torch.FloatTensor(batch).to(device)
                predictions = model.predict(batch_tensor)
                all_preds.append(predictions["trend_class"].cpu().numpy())
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return np.concatenate(all_preds)
    
    def compute_metrics(self, y_test: np.ndarray, pred_classes: np.ndarray) -> Dict:
        class_counts = Counter(y_test)
        most_common_count = class_counts.most_common(1)[0][1]
        zero_rule_accuracy = most_common_count / len(y_test)
        
        accuracy = (pred_classes == y_test).mean()
        f1_macro = f1_score(y_test, pred_classes, average='macro')
        f1_weighted = f1_score(y_test, pred_classes, average='weighted')
        mcc = matthews_corrcoef(y_test, pred_classes)
        
        return {
            "accuracy": accuracy,
            "zero_rule_baseline": zero_rule_accuracy,
            "accuracy_lift": accuracy - zero_rule_accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "mcc": mcc,
        }
    
    def train_single_config(
        self, 
        config: Dict,
        config_idx: int,
        total_configs: int,
        X_train_scaled: np.ndarray, 
        y_train: np.ndarray, 
        X_val_scaled: np.ndarray, 
        y_val: np.ndarray, 
        X_test_scaled: np.ndarray, 
        y_test: np.ndarray,
        n_features: int
    ) -> Dict:
        """Train a single hyperparameter configuration."""
        
        config_str = f"dropout={config['dropout']}, scheduler={config['scheduler']}, " \
                     f"label_smoothing={config['label_smoothing']}, conf_weight={config['confidence_weight']}"
        
        logger.info("=" * 60)
        logger.info(f"CONFIG {config_idx+1}/{total_configs}: {config_str}")
        logger.info("=" * 60)
        
        # Create model
        model = AttentionLSTM(
            input_size=n_features,
            hidden_size=64,
            num_layers=1,
            dropout=config["dropout"],
        )
        
        n_params = sum(p.numel() for p in model.parameters())
        
        # Create loss function
        if config["confidence_weight"] > 0:
            loss_fn = CombinedLoss(
                trend_weight=1.0,
                confidence_weight=config["confidence_weight"],
                label_smoothing=config["label_smoothing"],
            )
        else:
            # Standard CrossEntropyLoss with optional label smoothing
            loss_fn = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
        
        # Create trainer with specified scheduler
        trainer = CustomTrainer(
            model, loss_fn,
            learning_rate=self.learning_rate,
            weight_decay=1e-4,
            scheduler_type=config["scheduler"],
        )
        
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train)),
            batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val_scaled), torch.LongTensor(y_val)),
            batch_size=self.batch_size
        )
        
        # Train
        start_time = time.time()
        history = trainer.train(
            train_loader, val_loader, 
            epochs=self.epochs, 
            patience=self.early_stopping_patience
        )
        train_time = time.time() - start_time
        
        # Evaluate
        pred_classes = self.predict_batched(model, X_test_scaled)
        metrics = self.compute_metrics(y_test, pred_classes)
        
        logger.info(f"RESULT: Acc={metrics['accuracy']:.2%}, Lift={metrics['accuracy_lift']:+.2%}, "
                    f"MCC={metrics['mcc']:.4f}, Time={train_time:.1f}s")
        
        return {
            "config_idx": config_idx,
            **config,
            "n_params": n_params,
            "train_time_seconds": train_time,
            **metrics,
            "best_val_loss": trainer.best_val_loss,
            "epochs_trained": len(history.get("history", [])),
        }
    
    def run(self):
        logger.info("=" * 80)
        logger.info("HYPERPARAMETER TUNING EXPERIMENT")
        logger.info("=" * 80)
        
        # Get hyperparameter grid
        configs = get_hyperparameter_grid(quick=self.quick)
        logger.info(f"Testing {len(configs)} configurations")
        
        # Load and prepare data
        stock_data = self.load_all_stocks()
        X_train, y_train, X_val, y_val, X_test, y_test, n_features = \
            self.prepare_temporal_pooled_data(stock_data)
        
        del stock_data
        gc.collect()
        
        # Scale data
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_data(X_train, X_val, X_test)
        
        # Train each configuration
        all_results = []
        
        for idx, config in enumerate(configs):
            result = self.train_single_config(
                config, idx, len(configs),
                X_train_scaled, y_train,
                X_val_scaled, y_val,
                X_test_scaled, y_test,
                n_features
            )
            all_results.append(result)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Save intermediate results
            pd.DataFrame(all_results).to_csv(self.output_dir / "results_partial.csv", index=False)
        
        # Save final results
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(self.output_dir / "results.csv", index=False)
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("HYPERPARAMETER TUNING SUMMARY")
        logger.info("=" * 80)
        
        summary_cols = ["dropout", "scheduler", "label_smoothing", "confidence_weight", 
                        "accuracy", "accuracy_lift", "mcc", "train_time_seconds"]
        results_df = results_df.sort_values("accuracy", ascending=False)
        
        logger.info(f"\nTop 5 Configurations:")
        logger.info(results_df[summary_cols].head().to_string(index=False))
        
        best = results_df.iloc[0]
        logger.info(f"\n🏆 BEST CONFIG: dropout={best['dropout']}, scheduler={best['scheduler']}, "
                    f"label_smoothing={best['label_smoothing']}, confidence_weight={best['confidence_weight']}")
        logger.info(f"   Accuracy: {best['accuracy']:.2%}, Lift: {best['accuracy_lift']:+.2%}, MCC: {best['mcc']:.4f}")
        
        logger.info(f"\nResults saved to: {self.output_dir}")
        return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--stocks", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Test only 5 configs")
    args = parser.parse_args()
    
    stocks = EXTENDED_TICKERS[:args.stocks]
    
    experiment = HyperparameterTuningExperiment(
        stocks=stocks,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping_patience=args.patience,
        learning_rate=args.lr,
        use_cache=not args.no_cache,
        quick=args.quick,
    )
    
    experiment.run()


if __name__ == "__main__":
    main()
