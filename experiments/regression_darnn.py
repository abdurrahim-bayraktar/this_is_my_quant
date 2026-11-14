"""
Regression DA-RNN — Dual-Stage Attention for Volatility-Adjusted Forward Return.

Implements the DA-RNN experiment pipeline, mirroring regression_v1.py but with:
  - Dual-input sequences: features X + decoder target history y_history
  - Target scaling: unified StandardScaler for decoder history + prediction targets
  - DARNNTrainer: extends RegressionTrainer for the two-input forward pass
  - Evaluation in original scale via inverse_transform

Usage:
    python experiments/regression_darnn.py --epochs 100 --stocks 400
    python experiments/regression_darnn.py --epochs 2 --stocks 5  # smoke test
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
from typing import List, Dict, Tuple, Optional

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from scipy import stats
from tqdm import tqdm

from experiments.ranked_tickers import EXTENDED_TICKERS
from src.data.cache import PriceCache
from src.data.utils import load_stocks
from src.features.indicators_v7 import ComprehensiveIndicatorsV7
from src.models.darnn import DARNN

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

# Leaky / lookahead indicators excluded from the full V7 feature set
EXCLUDED_FEATURES = [
    'ichimoku_ICS_26',
    'dpo',
]

DEFAULT_TEST_STOCKS = [
    "MSFT", "AAPL", "JPM", "JNJ", "XOM",
    "WMT", "DIS", "BA", "KO", "NVDA", "SPY", "SPX",
]

# ============================================================================
# SEQUENCE CREATION FOR DA-RNN (features + decoder target history)
# ============================================================================

def create_sequences_darnn(
    feature_data: np.ndarray,
    targets: np.ndarray,
    dates: pd.DatetimeIndex,
    seq_length: int,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
]:
    """
    Create windowed sequences for DA-RNN with temporal split.

    Returns X (features), yh (decoder target history), y (prediction target)
    for each of train/val/test splits.

    Critical alignment:
      - Features X:       feature_data[i : i + seq_length]    → Days 1–T
      - Decoder y_history: targets[i-1 : i + seq_length - 1]  → Days 0–(T-1)
      - Prediction target: targets[i + seq_length - 1]        → Day T
    Loop starts at i=1 to accommodate the 1-step lookback.
    """
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
            X_train.append(seq_x)
            yh_train.append(seq_yh)
            y_train.append(target)
        elif target_date < val_end:
            X_val.append(seq_x)
            yh_val.append(seq_yh)
            y_val.append(target)
        else:
            X_test.append(seq_x)
            yh_test.append(seq_yh)
            y_test.append(target)

    def _to_array(lst, dtype):
        return np.array(lst, dtype=dtype) if lst else np.array([], dtype=dtype)

    return (
        _to_array(X_train, np.float32), _to_array(yh_train, np.float32), _to_array(y_train, np.float32),
        _to_array(X_val, np.float32), _to_array(yh_val, np.float32), _to_array(y_val, np.float32),
        _to_array(X_test, np.float32), _to_array(yh_test, np.float32), _to_array(y_test, np.float32),
    )


def create_sequences_darnn_with_dates(
    feature_data: np.ndarray,
    targets: np.ndarray,
    dates: pd.DatetimeIndex,
    seq_length: int,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
) -> Tuple:
    """Like create_sequences_darnn but also returns test dates for cross-sectional IC."""
    X_train, yh_train, y_train = [], [], []
    X_val, yh_val, y_val = [], [], []
    X_test, yh_test, y_test = [], [], []
    test_dates = []

    for i in range(1, len(feature_data) - seq_length + 1):
        target_date = dates[i + seq_length - 1]

        seq_x = feature_data[i : i + seq_length]
        seq_yh = targets[i - 1 : i + seq_length - 1]
        target = targets[i + seq_length - 1]

        if target_date < train_end:
            X_train.append(seq_x)
            yh_train.append(seq_yh)
            y_train.append(target)
        elif target_date < val_end:
            X_val.append(seq_x)
            yh_val.append(seq_yh)
            y_val.append(target)
        else:
            X_test.append(seq_x)
            yh_test.append(seq_yh)
            y_test.append(target)
            test_dates.append(target_date)

    def _to_array(lst, dtype):
        return np.array(lst, dtype=dtype) if lst else np.array([], dtype=dtype)

    return (
        _to_array(X_train, np.float32), _to_array(yh_train, np.float32), _to_array(y_train, np.float32),
        _to_array(X_val, np.float32), _to_array(yh_val, np.float32), _to_array(y_val, np.float32),
        _to_array(X_test, np.float32), _to_array(yh_test, np.float32), _to_array(y_test, np.float32),
        test_dates,
    )


# ============================================================================
# REGRESSION METRICS (same as regression_v1)
# ============================================================================

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute regression metrics for stock return prediction.

    Returns:
        mse, mae, r2, ic, ic_pvalue, directional_accuracy, n_samples
    """
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    ic_result = stats.spearmanr(y_pred, y_true)
    ic = float(ic_result.correlation) if not np.isnan(ic_result.correlation) else 0.0
    ic_pvalue = float(ic_result.pvalue) if not np.isnan(ic_result.pvalue) else 1.0

    # Directional accuracy: does the prediction get the sign right?
    mask = y_true != 0
    if mask.sum() > 0:
        dir_acc = float(np.mean(np.sign(y_pred[mask]) == np.sign(y_true[mask])))
    else:
        dir_acc = 0.0

    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "ic": ic,
        "ic_pvalue": ic_pvalue,
        "directional_accuracy": dir_acc,
        "n_samples": int(len(y_true)),
    }


# ============================================================================
# CROSS-SECTIONAL EVALUATION (same as regression_v1)
# ============================================================================

def compute_cross_sectional_ic(
    per_stock_predictions: Dict[str, Dict],
) -> Dict[str, float]:
    """
    Compute cross-sectional IC: at each date, rank stocks by prediction
    and correlate with actual returns.
    """
    date_data = {}
    for ticker, data in per_stock_predictions.items():
        for date, pred, actual in zip(data["dates"], data["predictions"], data["actuals"]):
            if date not in date_data:
                date_data[date] = {}
            date_data[date][ticker] = (pred, actual)

    ics = []
    quantile_spreads = []

    for date, stocks in sorted(date_data.items()):
        if len(stocks) < 5:
            continue

        preds = np.array([v[0] for v in stocks.values()])
        actuals = np.array([v[1] for v in stocks.values()])

        ic_result = stats.spearmanr(preds, actuals)
        if not np.isnan(ic_result.correlation):
            ics.append(ic_result.correlation)

        n = len(preds)
        quintile_size = max(1, n // 5)
        sorted_idx = np.argsort(preds)
        bottom_returns = actuals[sorted_idx[:quintile_size]].mean()
        top_returns = actuals[sorted_idx[-quintile_size:]].mean()
        quantile_spreads.append(top_returns - bottom_returns)

    if not ics:
        return {
            "cross_sectional_ic": 0.0,
            "ic_ir": 0.0,
            "ic_hit_rate": 0.0,
            "quantile_spread": 0.0,
            "n_dates": 0,
        }

    ics = np.array(ics)
    spreads = np.array(quantile_spreads)

    return {
        "cross_sectional_ic": float(np.mean(ics)),
        "ic_ir": float(np.mean(ics) / np.std(ics)) if np.std(ics) > 0 else 0.0,
        "ic_hit_rate": float(np.mean(ics > 0)),
        "quantile_spread": float(np.mean(spreads)),
        "n_dates": len(ics),
    }


# ============================================================================
# DA-RNN TRAINER (extends RegressionTrainer for dual-input forward pass)
# ============================================================================

class DARNNTrainer:
    """Training loop for DA-RNN with Huber loss and teacher forcing."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-4,
        huber_delta: float = 1.0,
    ):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.loss_fn = nn.HuberLoss(delta=huber_delta)
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
        total_mae = 0.0
        total = 0
        n_batches = 0
        epoch_start = time.time()

        for X, yh, y in train_loader:
            X, yh, y = X.to(self.device), yh.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            if self.scaler:
                with torch.amp.autocast(device_type='cuda'):
                    prediction, _, _ = self.model(X, yh)
                    loss = self.loss_fn(prediction, y)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                prediction, _, _ = self.model(X, yh)
                loss = self.loss_fn(prediction, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            total_loss += loss.item() * len(X)
            total_mae += F.l1_loss(prediction, y, reduction='sum').item()
            total += len(y)
            n_batches += 1

        elapsed = time.time() - epoch_start
        return {
            "loss": total_loss / total,
            "mae": total_mae / total,
            "elapsed": elapsed,
            "it_per_sec": n_batches / elapsed if elapsed > 0 else 0,
            "samples_per_sec": total / elapsed if elapsed > 0 else 0,
        }

    def validate(self, val_loader: DataLoader) -> Dict:
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        total = 0

        with torch.no_grad():
            for X, yh, y in val_loader:
                X, yh, y = X.to(self.device), yh.to(self.device), y.to(self.device)
                prediction, _, _ = self.model(X, yh)
                loss = self.loss_fn(prediction, y)
                total_loss += loss.item() * len(X)
                total_mae += F.l1_loss(prediction, y, reduction='sum').item()
                total += len(y)

        val_loss = total_loss / total
        self.scheduler.step(val_loss)
        return {"loss": val_loss, "mae": total_mae / total}

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
                    f"Train: {train_metrics['loss']:.4f} (MAE {train_metrics['mae']:.4f}) | "
                    f"Val: {val_metrics['loss']:.4f} (MAE {val_metrics['mae']:.4f}) | "
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
# EXPERIMENT
# ============================================================================

class DARNNExperiment:
    """
    DA-RNN regression experiment: predict volatility-adjusted forward returns.

    Pipeline:
    1. Load stocks (daily or weekly)
    2. Compute full V7 indicators (~100 features, excluding leaky ones)
    3. Create regression target: vol-adjusted forward return
    4. Pool all stocks, temporal split (features + decoder target history)
    5. Scale features (StandardScaler) and targets (separate StandardScaler)
    6. Train DARNN with Huber loss (teacher forcing)
    7. Evaluate: pooled test + per-stock test + cross-sectional IC (in original scale)
    8. Save config.json + summary.csv + per_stock_results.csv + model.pt
    """

    def __init__(self, config: Dict):
        self.config = config

        self.train_stocks = config.get("train_stocks", EXTENDED_TICKERS[:400])
        self.test_stocks = config.get("test_stocks", DEFAULT_TEST_STOCKS)
        self.train_stocks = [t for t in self.train_stocks if t not in self.test_stocks]

        self.start_date = config.get("start_date", "2014-01-01")
        self.end_date = config.get("end_date", "2024-12-31")
        self.train_end = pd.Timestamp(config.get("train_end", "2022-01-01"))
        self.val_end = pd.Timestamp(config.get("val_end", "2023-06-01"))

        self.frequency = config.get("frequency", "daily")
        self.sequence_length = config.get("sequence_length", 20 if self.frequency == "daily" else 12)
        self.epochs = config.get("epochs", 100)
        self.batch_size = config.get("batch_size", 1024 if self.frequency == "daily" else 512)
        self.dropout = config.get("dropout", 0.2)
        self.learning_rate = config.get("learning_rate", 5e-4)
        self.patience = config.get("patience", 15)
        self.huber_delta = config.get("huber_delta", 1.0)
        self.vol_lookback = config.get("vol_lookback", 20)
        self.target_clip = config.get("target_clip", 10.0)

        self.encoder_hidden = config.get("encoder_hidden", 64)
        self.decoder_hidden = config.get("decoder_hidden", 64)

        self.excluded_features = config.get("excluded_features", EXCLUDED_FEATURES)

        # Output directory
        exp_name = config.get("experiment_name", "regression_darnn")
        self.output_dir = REPORTS_DIR / f"{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.indicator_computer = ComprehensiveIndicatorsV7()
        self.cache = PriceCache()
        self.feature_scaler = None
        self.target_scaler = None

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

    def _compute_features_and_target(self, df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[pd.DatetimeIndex]]:
        """Compute full V7 features and vol-adjusted return target for a single stock."""
        df = self.indicator_computer.compute_all(df)

        # Dynamically discover all indicator columns and exclude leaky ones
        all_indicator_cols = self.indicator_computer.get_indicator_columns(df)
        available = [c for c in all_indicator_cols
                     if c not in self.excluded_features and c in df.columns]
        if len(available) < 5:
            return None, None, None

        # Volatility-adjusted forward return
        df['return_next'] = df['Close'].pct_change().shift(-1)
        df['rolling_vol'] = df['Close'].pct_change().rolling(self.vol_lookback).std()
        df['vol_adj_return'] = df['return_next'] / df['rolling_vol']
        df['vol_adj_return'] = df['vol_adj_return'].clip(-self.target_clip, self.target_clip)

        df = df.dropna(subset=['vol_adj_return'] + available[:5])
        if len(df) < self.sequence_length + 10:
            return None, None, None

        feature_data = df[available].values
        feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
        targets = df['vol_adj_return'].values.astype(np.float32)

        return feature_data, targets, df.index

    def prepare_stock(self, df: pd.DataFrame) -> Tuple:
        """Compute features, create target, and build DA-RNN sequences."""
        result = self._compute_features_and_target(df)
        if result[0] is None:
            return tuple(np.array([], dtype=np.float32) for _ in range(9))

        feature_data, targets, dates = result

        return create_sequences_darnn(
            feature_data, targets, dates,
            self.sequence_length, self.train_end, self.val_end,
        )

    def prepare_stock_with_dates(self, df: pd.DataFrame) -> Tuple:
        """Like prepare_stock but also returns test dates for cross-sectional IC."""
        result = self._compute_features_and_target(df)
        if result[0] is None:
            return tuple(np.array([], dtype=np.float32) for _ in range(10))

        feature_data, targets, dates = result

        return create_sequences_darnn_with_dates(
            feature_data, targets, dates,
            self.sequence_length, self.train_end, self.val_end,
        )

    def prepare_pooled_data(self, stock_data: Dict, tickers: List[str]) -> Tuple:
        """Prepare pooled train/val/test arrays from multiple stocks."""
        all_train_X, all_train_yh, all_train_y = [], [], []
        all_val_X, all_val_yh, all_val_y = [], [], []
        all_test_X, all_test_yh, all_test_y = [], [], []
        n_features = None

        stocks = {t: stock_data[t] for t in tickers if t in stock_data}

        for ticker, df in tqdm(stocks.items(), desc="Processing stocks"):
            try:
                X_tr, yh_tr, y_tr, X_val, yh_val, y_val, X_te, yh_te, y_te = self.prepare_stock(df)

                if len(X_tr) == 0:
                    continue

                if n_features is None:
                    n_features = X_tr.shape[-1]
                elif X_tr.shape[-1] != n_features:
                    continue

                all_train_X.append(X_tr)
                all_train_yh.append(yh_tr)
                all_train_y.append(y_tr)
                if len(X_val) > 0:
                    all_val_X.append(X_val)
                    all_val_yh.append(yh_val)
                    all_val_y.append(y_val)
                if len(X_te) > 0:
                    all_test_X.append(X_te)
                    all_test_yh.append(yh_te)
                    all_test_y.append(y_te)

            except Exception as e:
                logger.warning(f"Failed {ticker}: {e}")

        X_train = np.concatenate(all_train_X, axis=0)
        yh_train = np.concatenate(all_train_yh, axis=0)
        y_train = np.concatenate(all_train_y, axis=0)
        X_val = np.concatenate(all_val_X) if all_val_X else np.array([], dtype=np.float32)
        yh_val = np.concatenate(all_val_yh) if all_val_yh else np.array([], dtype=np.float32)
        y_val = np.concatenate(all_val_y) if all_val_y else np.array([], dtype=np.float32)
        X_test = np.concatenate(all_test_X) if all_test_X else np.array([], dtype=np.float32)
        yh_test = np.concatenate(all_test_yh) if all_test_yh else np.array([], dtype=np.float32)
        y_test = np.concatenate(all_test_y) if all_test_y else np.array([], dtype=np.float32)

        logger.info(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}, Features: {n_features}")
        logger.info(f"Target stats (train): mean={y_train.mean():.4f}, std={y_train.std():.4f}, "
                     f"min={y_train.min():.4f}, max={y_train.max():.4f}")

        return X_train, yh_train, y_train, X_val, yh_val, y_val, X_test, yh_test, y_test, n_features

    def scale_data(self, X_train, yh_train, y_train, X_val, yh_val, y_val, X_test, yh_test, y_test):
        """
        Fit feature scaler on train features, target scaler on train targets.
        Scale all splits for both features and targets.
        """
        n_train, seq_len, n_features = X_train.shape

        # === Feature scaler (same as regression_v1) ===
        self.feature_scaler = StandardScaler()
        X_train_s = self.feature_scaler.fit_transform(X_train.reshape(-1, n_features))
        X_train_s = np.nan_to_num(X_train_s, nan=0.0, posinf=0.0, neginf=0.0)
        X_train_s = X_train_s.reshape(n_train, seq_len, n_features)

        def _transform_features(X):
            if len(X) == 0:
                return X
            s = X.shape
            X_t = self.feature_scaler.transform(X.reshape(-1, n_features))
            return np.nan_to_num(X_t, nan=0.0, posinf=0.0, neginf=0.0).reshape(s)

        X_val_s = _transform_features(X_val)
        X_test_s = _transform_features(X_test)

        # === Target scaler ===
        self.target_scaler = StandardScaler()
        self.target_scaler.fit(y_train.reshape(-1, 1))

        # Scale decoder history for all splits
        def _scale_targets_2d(arr):
            """Scale a 2D array (batch, seq_len) of target values."""
            if len(arr) == 0:
                return arr
            shape = arr.shape
            return self.target_scaler.transform(arr.reshape(-1, 1)).reshape(shape).astype(np.float32)

        def _scale_targets_1d(arr):
            """Scale a 1D array of target values."""
            if len(arr) == 0:
                return arr
            return self.target_scaler.transform(arr.reshape(-1, 1)).ravel().astype(np.float32)

        yh_train_s = _scale_targets_2d(yh_train)
        yh_val_s = _scale_targets_2d(yh_val)
        yh_test_s = _scale_targets_2d(yh_test)

        y_train_s = _scale_targets_1d(y_train)
        y_val_s = _scale_targets_1d(y_val)
        y_test_s = _scale_targets_1d(y_test)

        return X_train_s, yh_train_s, y_train_s, X_val_s, yh_val_s, y_val_s, X_test_s, yh_test_s, y_test_s

    def _predict_darnn(self, model, X_scaled, yh_scaled, batch_size=2048):
        """Run DA-RNN predictions in batches, returns numpy array of scaled predictions."""
        device = next(model.parameters()).device
        model.eval()
        all_preds = []

        for i in range(0, len(X_scaled), batch_size):
            batch_X = X_scaled[i:i + batch_size]
            batch_yh = yh_scaled[i:i + batch_size]
            with torch.no_grad():
                X_tensor = torch.FloatTensor(batch_X).to(device)
                yh_tensor = torch.FloatTensor(batch_yh).to(device)
                result = model.predict(X_tensor, yh_tensor)
                all_preds.append(result["prediction"].cpu().numpy())

        return np.concatenate(all_preds)

    def evaluate_pooled(self, model, X_test_s, yh_test_s, y_test_s) -> Dict:
        """Evaluate on pooled test data, computing metrics in original scale."""
        if len(X_test_s) == 0:
            return {"error": "No data", "n_samples": 0}

        # Model outputs scaled predictions
        scaled_preds = self._predict_darnn(model, X_test_s, yh_test_s)

        # Inverse transform to original vol-adj return space
        preds_original = self.target_scaler.inverse_transform(scaled_preds.reshape(-1, 1)).ravel()
        y_original = self.target_scaler.inverse_transform(y_test_s.reshape(-1, 1)).ravel()

        metrics = compute_regression_metrics(y_original, preds_original)
        metrics["predictions"] = preds_original
        return metrics

    def evaluate_single_stock(self, model, stock_data, ticker) -> Dict:
        """Evaluate DA-RNN on a single held-out stock's test data."""
        if ticker not in stock_data:
            return {"ticker": ticker, "error": "Not found"}

        result = self.prepare_stock(stock_data[ticker])
        X_te, yh_te, y_te = result[6], result[7], result[8]

        if len(X_te) == 0:
            return {"ticker": ticker, "error": "No test data"}

        # Scale features
        n, seq, feat = X_te.shape
        X_scaled = self.feature_scaler.transform(X_te.reshape(-1, feat))
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0).reshape(n, seq, feat)

        # Scale decoder history and targets
        yh_scaled = self.target_scaler.transform(yh_te.reshape(-1, 1)).reshape(yh_te.shape).astype(np.float32)
        y_scaled = self.target_scaler.transform(y_te.reshape(-1, 1)).ravel().astype(np.float32)

        # Get scaled predictions
        scaled_preds = self._predict_darnn(model, X_scaled, yh_scaled)

        # Inverse transform to original scale for metrics
        preds_original = self.target_scaler.inverse_transform(scaled_preds.reshape(-1, 1)).ravel()
        y_original = self.target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()

        metrics = compute_regression_metrics(y_original, preds_original)
        metrics["ticker"] = ticker
        return metrics

    def evaluate_cross_sectional(self, model, stock_data, tickers: List[str]) -> Dict:
        """
        Run cross-sectional IC evaluation.

        For each test date, gather predictions across all available stocks,
        then compute rank correlation with actual vol-adj returns.
        """
        per_stock_predictions = {}

        for ticker in tqdm(tickers, desc="Cross-sectional eval"):
            if ticker not in stock_data:
                continue

            try:
                result = self.prepare_stock_with_dates(stock_data[ticker])
                if len(result) != 10:
                    continue
                _, _, _, _, _, _, X_te, yh_te, y_te, test_dates = result

                if len(X_te) == 0:
                    continue

                # Scale features
                n, seq, feat = X_te.shape
                X_scaled = self.feature_scaler.transform(X_te.reshape(-1, feat))
                X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0).reshape(n, seq, feat)

                # Scale decoder history and targets
                yh_scaled = self.target_scaler.transform(yh_te.reshape(-1, 1)).reshape(yh_te.shape).astype(np.float32)
                y_scaled = self.target_scaler.transform(y_te.reshape(-1, 1)).ravel().astype(np.float32)

                # Get scaled predictions
                scaled_preds = self._predict_darnn(model, X_scaled, yh_scaled)

                # Inverse transform to original scale
                preds_original = self.target_scaler.inverse_transform(scaled_preds.reshape(-1, 1)).ravel()
                y_original = self.target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()

                per_stock_predictions[ticker] = {
                    "dates": test_dates,
                    "predictions": preds_original,
                    "actuals": y_original,
                }

            except Exception as e:
                logger.warning(f"Cross-sectional eval failed for {ticker}: {e}")

        if not per_stock_predictions:
            return {"cross_sectional_ic": 0.0, "ic_ir": 0.0, "ic_hit_rate": 0.0,
                    "quantile_spread": 0.0, "n_dates": 0}

        return compute_cross_sectional_ic(per_stock_predictions)

    def run(self):
        """Execute the full DA-RNN regression experiment pipeline."""
        logger.info("=" * 80)
        logger.info("DA-RNN REGRESSION EXPERIMENT — Volatility-Adjusted Forward Return")
        logger.info("=" * 80)
        logger.info(f"Frequency: {self.frequency}")
        logger.info(f"Train stocks: {len(self.train_stocks)}, Test stocks: {self.test_stocks}")
        logger.info(f"Huber delta: {self.huber_delta}, Vol lookback: {self.vol_lookback}")
        logger.info(f"Encoder hidden: {self.encoder_hidden}, Decoder hidden: {self.decoder_hidden}")

        # Save config
        config_to_save = {k: v for k, v in self.config.items()
                         if not isinstance(v, (list,)) or len(v) < 20}
        config_to_save["train_stock_count"] = len(self.train_stocks)
        config_to_save["test_stocks"] = self.test_stocks
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(config_to_save, f, indent=2, default=str)

        # Load data
        stock_data = self.load_data()

        # Prepare pooled training data (from train stocks only)
        X_train, yh_train, y_train, X_val, yh_val, y_val, X_test, yh_test, y_test, n_features = \
            self.prepare_pooled_data(stock_data, self.train_stocks)

        # Scale features and targets
        X_train_s, yh_train_s, y_train_s, X_val_s, yh_val_s, y_val_s, X_test_s, yh_test_s, y_test_s = \
            self.scale_data(X_train, yh_train, y_train, X_val, yh_val, y_val, X_test, yh_test, y_test)
        del X_train, yh_train, y_train, X_val, yh_val, y_val, X_test, yh_test, y_test
        gc.collect()

        # Create model
        model = DARNN(
            input_size=n_features,
            seq_length=self.sequence_length,
            encoder_hidden=self.encoder_hidden,
            decoder_hidden=self.decoder_hidden,
            dropout=self.dropout,
        )
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {n_params:,}")

        # Train
        trainer = DARNNTrainer(
            model,
            learning_rate=self.learning_rate,
            huber_delta=self.huber_delta,
        )

        # DataLoader wraps TensorDataset of three tensors — all scaled
        train_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_train_s),
                torch.FloatTensor(yh_train_s),
                torch.FloatTensor(y_train_s),
            ),
            batch_size=self.batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(
                torch.FloatTensor(X_val_s),
                torch.FloatTensor(yh_val_s),
                torch.FloatTensor(y_val_s),
            ),
            batch_size=self.batch_size,
        )

        logger.info("\n" + "=" * 40)
        logger.info("TRAINING")
        logger.info("=" * 40)
        start_time = time.time()
        trainer.train(train_loader, val_loader, epochs=self.epochs, patience=self.patience)
        train_time = time.time() - start_time
        logger.info(f"Training time: {train_time:.1f}s")

        # === POOLED TEST EVALUATION (in original scale) ===
        logger.info("\n" + "=" * 40)
        logger.info("POOLED TEST EVALUATION")
        logger.info("=" * 40)

        pooled_metrics = self.evaluate_pooled(model, X_test_s, yh_test_s, y_test_s)
        pooled_preds = pooled_metrics.pop("predictions", None)

        if "error" not in pooled_metrics:
            logger.info(f"Pooled: n={pooled_metrics['n_samples']:,}, "
                       f"MSE={pooled_metrics['mse']:.4f}, "
                       f"MAE={pooled_metrics['mae']:.4f}, "
                       f"R²={pooled_metrics['r2']:.4f}, "
                       f"IC={pooled_metrics['ic']:.4f}, "
                       f"DirAcc={pooled_metrics['directional_accuracy']:.2%}")

        # === PER-STOCK EVALUATION ===
        logger.info("\n" + "=" * 40)
        logger.info("PER-STOCK EVALUATION")
        logger.info("=" * 40)

        per_stock_results = []
        for ticker in self.test_stocks:
            result = self.evaluate_single_stock(model, stock_data, ticker)
            per_stock_results.append(result)
            if "error" not in result:
                logger.info(f"{ticker}: MSE={result['mse']:.4f}, "
                           f"IC={result['ic']:.4f}, "
                           f"DirAcc={result['directional_accuracy']:.2%}, "
                           f"n={result['n_samples']}")
            else:
                logger.warning(f"{ticker}: {result['error']}")

        # === CROSS-SECTIONAL IC (ranking quality) ===
        logger.info("\n" + "=" * 40)
        logger.info("CROSS-SECTIONAL EVALUATION (Ranking Quality)")
        logger.info("=" * 40)

        # Use ALL stocks (train + test) for cross-sectional eval
        all_tickers = list(set(self.train_stocks + self.test_stocks))
        cs_metrics = self.evaluate_cross_sectional(model, stock_data, all_tickers)

        logger.info(f"Cross-Sectional IC: {cs_metrics['cross_sectional_ic']:.4f}")
        logger.info(f"IC IR (consistency): {cs_metrics['ic_ir']:.4f}")
        logger.info(f"IC Hit Rate: {cs_metrics['ic_hit_rate']:.2%}")
        logger.info(f"Quantile Spread (Q5-Q1): {cs_metrics['quantile_spread']:.4f}")
        logger.info(f"Evaluated on {cs_metrics['n_dates']} dates")

        # === SAVE RESULTS ===
        valid = [r for r in per_stock_results if "error" not in r]
        avg_mse = np.mean([r["mse"] for r in valid]) if valid else 0
        avg_mae = np.mean([r["mae"] for r in valid]) if valid else 0
        avg_ic = np.mean([r["ic"] for r in valid]) if valid else 0
        avg_r2 = np.mean([r["r2"] for r in valid]) if valid else 0
        avg_dir_acc = np.mean([r["directional_accuracy"] for r in valid]) if valid else 0

        summary = {
            "experiment_name": self.config.get("experiment_name", "regression_darnn"),
            "task": "regression",
            "model": "DARNN",
            "target": "vol_adj_forward_return",
            "loss_function": f"huber_delta_{self.huber_delta}",
            "frequency": self.frequency,
            "n_params": n_params,
            "train_stocks": len(self.train_stocks),
            "sequence_length": self.sequence_length,
            "encoder_hidden": self.encoder_hidden,
            "decoder_hidden": self.decoder_hidden,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "huber_delta": self.huber_delta,
            "vol_lookback": self.vol_lookback,
            "target_clip": self.target_clip,
            "epochs": self.epochs,
            "train_time": train_time,
            # Pooled metrics
            "pooled_mse": pooled_metrics.get("mse", 0),
            "pooled_mae": pooled_metrics.get("mae", 0),
            "pooled_r2": pooled_metrics.get("r2", 0),
            "pooled_ic": pooled_metrics.get("ic", 0),
            "pooled_directional_accuracy": pooled_metrics.get("directional_accuracy", 0),
            "pooled_n_samples": pooled_metrics.get("n_samples", 0),
            # Cross-sectional (ranking) metrics
            "cross_sectional_ic": cs_metrics.get("cross_sectional_ic", 0),
            "ic_ir": cs_metrics.get("ic_ir", 0),
            "ic_hit_rate": cs_metrics.get("ic_hit_rate", 0),
            "quantile_spread": cs_metrics.get("quantile_spread", 0),
            "cs_n_dates": cs_metrics.get("n_dates", 0),
            # Per-stock averages
            "per_stock_avg_mse": avg_mse,
            "per_stock_avg_mae": avg_mae,
            "per_stock_avg_ic": avg_ic,
            "per_stock_avg_r2": avg_r2,
            "per_stock_avg_directional_accuracy": avg_dir_acc,
            "per_stock_count": len(valid),
        }

        pd.DataFrame([summary]).to_csv(self.output_dir / "summary.csv", index=False)
        pd.DataFrame(per_stock_results).to_csv(self.output_dir / "per_stock_results.csv", index=False)
        torch.save(model.state_dict(), self.output_dir / "model.pt")

        logger.info(f"\n{'=' * 60}")
        logger.info("SUMMARY")
        logger.info(f"{'=' * 60}")
        logger.info(f"DARNN ({n_params:,} params)")
        logger.info(f"Pooled:    MSE={pooled_metrics.get('mse', 0):.4f}, "
                    f"IC={pooled_metrics.get('ic', 0):.4f}, "
                    f"DirAcc={pooled_metrics.get('directional_accuracy', 0):.2%}")
        logger.info(f"Per-stock: IC={avg_ic:.4f}, DirAcc={avg_dir_acc:.2%}")
        logger.info(f"Ranking:   CS-IC={cs_metrics.get('cross_sectional_ic', 0):.4f}, "
                    f"ICIR={cs_metrics.get('ic_ir', 0):.4f}, "
                    f"Q-Spread={cs_metrics.get('quantile_spread', 0):.4f}")
        logger.info(f"Results saved to: {self.output_dir}")

        return summary


def main():
    parser = argparse.ArgumentParser(description="DA-RNN Regression — Vol-Adjusted Forward Return")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--stocks", type=int, default=400, help="Number of training stocks")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--frequency", choices=["daily", "weekly"], default="daily")
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--test-stocks", nargs="+", default=None)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--huber-delta", type=float, default=1.0, help="Huber loss delta")
    parser.add_argument("--vol-lookback", type=int, default=20, help="Rolling vol window for target")
    parser.add_argument("--target-clip", type=float, default=10.0, help="Clip vol-adj return to ±N")
    parser.add_argument("--encoder-hidden", type=int, default=64, help="Encoder LSTM hidden size")
    parser.add_argument("--decoder-hidden", type=int, default=64, help="Decoder LSTM hidden size")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--name", type=str, default="regression_darnn", help="Experiment name")
    args = parser.parse_args()

    config = {
        "experiment_name": args.name,
        "frequency": args.frequency,
        "train_stocks": EXTENDED_TICKERS[:args.stocks],
        "test_stocks": args.test_stocks or DEFAULT_TEST_STOCKS,
        "epochs": args.epochs,
        "dropout": args.dropout,
        "learning_rate": args.lr,
        "patience": args.patience,
        "huber_delta": args.huber_delta,
        "vol_lookback": args.vol_lookback,
        "target_clip": args.target_clip,
        "encoder_hidden": args.encoder_hidden,
        "decoder_hidden": args.decoder_hidden,
    }

    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.seq_length:
        config["sequence_length"] = args.seq_length

    experiment = DARNNExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
