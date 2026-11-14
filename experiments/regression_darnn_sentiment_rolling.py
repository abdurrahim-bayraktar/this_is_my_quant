"""
Regression DA-RNN + Sentiment — Walk-Forward (Rolling-Origin) Validation.

Extends regression_darnn_rolling.py by:
  - Adding 2 RF-selected sentiment features from FNSPID (sent_count_log, sent_strength)
  - Constraining the stock universe to tickers with sentiment coverage
    (top N densest, excluding noisy tickers — same universe as
    regression_sentiment_v2_rolling.py)

Walk-forward validation simulates realistic deployment:
  - Each fold trains on ALL data up to the fold's start (expanding window)
  - Validates on a single quarter (e.g., Q1 2017)
  - A FRESH model + scalers are trained per fold — no warm-starting

Key design decisions:
  - DA-RNN dual-input sequences: features X + decoder target history y_history
  - Target scaling: unified StandardScaler for decoder history + prediction targets
  - DARNNTrainer: extends RegressionTrainer for the two-input forward pass
  - Evaluation in original scale via inverse_transform
  - Walk-forward folds: quarterly expanding window (default 12 folds over 2017–2019)
  - Training universe restricted to tickers with sentiment coverage
  - Date range: 2012–2019 (aligned with dense FNSPID sentiment era)
  - 2 sentiment features selected by RandomForest permutation importance:
    sent_count_log, sent_strength
  - No-news days filled via exponential decay (sentiment fades, doesn't vanish)

Usage:
    python experiments/regression_darnn_sentiment_rolling.py --epochs 100
    python experiments/regression_darnn_sentiment_rolling.py --epochs 2 --max-tickers 5  # smoke test
    python experiments/regression_darnn_sentiment_rolling.py --no-sentiment --name wf_darnn_baseline  # A/B
    python experiments/regression_darnn_sentiment_rolling.py --wf-step MS  # monthly (36 folds)
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
    from config import REPORTS_DIR, CACHE_DIR
except ImportError:
    REPORTS_DIR = Path("reports")
    CACHE_DIR = Path("data/cache")

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

# Full list of 8 sentiment feature candidates
SENTIMENT_FEATURES = [
    "sent_mean",           # Daily avg sentiment (positive − negative)
    "sent_std",            # Intraday sentiment dispersion
    "sent_count_log",      # log1p(daily_news_count)
    "sent_momentum_3d",    # 3-day diff of sent_mean
    "sent_ema_5d",         # 5-day EMA of sent_mean
    "sent_surprise",       # sent_mean − 20-day rolling mean
    "sent_strength",       # sent_mean × sent_count_log
    "sent_ema_20d",       
]

# ============================================================================
# EXCLUDED TICKERS (from scripts/visualize_sentiment_density.py)
# Noisy / low-quality sentiment tickers identified during density analysis.
# ============================================================================

EXCLUDE_TICKERS = {
    "FDX", "BHI", "AET", "CHN", "SLB", "VNQ", "ESRX", "XPP", "DGAZ", "UGAZ",
    "WFM", "ERO", "JD", "VGK", "FOXA", "YINN", "ABX", "FXP", "PGJ", "GILD",
    "GXC", "XLY", "XLU",
    "MYL", "GME", "MON", "EWU", "TSN", "JCP", "POT", "CI", "WBA", "BRRY",
    "SPLS", "FCX", "BTU", "GRUB", "REGN", "XLK", "AXP", "DISH", "TWC", "TJX",
    "PBR", "DZZ", "KR", "MDT", "PANW", "CSX", "AVGO", "KSS", "SHLD", "OXY",
    "EWJ", "ANTM", "COH", "TXN", "FCAU", "BABA", "WDC", "HAL", "EA",
    "GMCR", "DIS", "RSP", "WMT", "YHOO", "INTC", "JPM", "DOW", "SLV", "GRPN",
    "LEN", "LLY", "VMW", "GPRO", "CMG", "HD", "LUV", "AEO", "TWX", "JWN",
    "BRCM", "DD", "STZ", "NFLX", "BLK", "BBRY", "WDAY",
}

DEFAULT_MAX_TICKERS = 150  # Top N densest tickers to use

DEFAULT_TEST_STOCKS = [
    "MSFT", "DIS", "WMT",  # Test stocks with sentiment
]


# ============================================================================
# SENTIMENT DATA LOADER
# ============================================================================

def load_sentiment_data(sentiment_dir: Path = None) -> Dict[str, pd.DataFrame]:
    """
    Load all cached sentiment parquets and aggregate to daily features.

    Only computes the 2 RF-selected sentiment features:
      - sent_count_log: log1p(daily article count)
      - sent_strength:  sent_mean × sent_count_log

    Returns:
        {ticker: DataFrame} where DataFrame is date-indexed with
        columns: sent_count_log, sent_strength
    """
    if sentiment_dir is None:
        sentiment_dir = CACHE_DIR / "sentiment"

    if not sentiment_dir.exists():
        logger.warning(f"Sentiment directory not found: {sentiment_dir}")
        return {}

    sentiment_files = list(sentiment_dir.glob("*_sentiment.parquet"))
    if not sentiment_files:
        logger.warning("No sentiment parquets found")
        return {}

    logger.info(f"Loading sentiment from {len(sentiment_files)} ticker files "
                f"(excluding {len(EXCLUDE_TICKERS)} noisy tickers)...")
    result = {}

    for fpath in sentiment_files:
        ticker = fpath.stem.replace("_sentiment", "")

        # Skip excluded tickers
        if ticker in EXCLUDE_TICKERS:
            continue

        try:
            raw = pd.read_parquet(fpath)

            # Normalize date column
            raw['date'] = pd.to_datetime(raw['date'], utc=True).dt.tz_localize(None)
            raw['trading_date'] = raw['date'].dt.normalize()  # Strip time component

            # === Aggregate to daily level per ticker ===
            daily = raw.groupby('trading_date').agg(
                sent_mean=('sentiment_value', 'mean'),
                sent_count=('sentiment_value', 'count'),
            ).sort_index()

            # === Compute only the 2 selected features ===

            # Log-scaled news count — tames outliers, preserves ordinality
            daily['sent_count_log'] = np.log1p(daily['sent_count'])

            # Sentiment strength: sentiment × coverage
            # Strong sentiment + lots of articles = high-conviction signal
            daily['sent_strength'] = daily['sent_mean'] * daily['sent_count_log']

            # === Forward-fill with exponential decay for no-news days ===
            # Create a complete business day index to fill gaps
            if len(daily) > 0:
                full_idx = pd.bdate_range(daily.index.min(), daily.index.max())
                daily = daily.reindex(full_idx)

                # Track which days had news
                had_news = daily['sent_count'].notna()

                # Decay fill: for each gap, previous value × 0.95^days_since_news
                # Implementation: forward-fill then apply accumulated decay
                for col in SENTIMENT_FEATURES:
                    if col not in daily.columns:
                        continue

                    # Forward-fill first
                    filled = daily[col].ffill()

                    # Apply decay: multiply by 0.95 for each consecutive no-news day
                    decay_factor = 0.95
                    cumulative_decay = (~had_news).astype(float)

                    # Running count of consecutive no-news days
                    groups = had_news.cumsum()
                    consecutive_gaps = cumulative_decay.groupby(groups).cumsum()

                    # Apply decay: value × decay_factor^gap_length
                    decay_mask = consecutive_gaps > 0
                    filled[decay_mask] = filled[decay_mask] * (decay_factor ** consecutive_gaps[decay_mask])

                    daily[col] = filled

            # Fill remaining NaN with 0 (beginning of series)
            daily = daily.fillna(0)

            # Keep only the selected feature columns
            daily = daily[SENTIMENT_FEATURES]

            result[ticker] = daily

        except Exception as e:
            logger.warning(f"Failed to load sentiment for {ticker}: {e}")

    logger.info(f"Loaded sentiment for {len(result)} tickers (post-exclusion)")

    # Log date coverage
    if result:
        all_dates = pd.concat([df.index.to_series() for df in result.values()])
        logger.info(f"Sentiment date range: {all_dates.min().date()} to {all_dates.max().date()}")

    return result


def rank_tickers_by_density(
    sentiment_data: Dict[str, pd.DataFrame],
    start_date: str = "2012-01-01",
    end_date: str = "2019-12-31",
) -> List[str]:
    """
    Rank tickers by sentiment density (fraction of business days with news).

    Mirrors the density calculation in scripts/visualize_sentiment_density.py.

    Returns list of ticker symbols sorted by density (densest first).
    """
    full_idx = pd.bdate_range(start_date, end_date)
    total_trading_days = len(full_idx)

    rows = []
    for ticker, df in sentiment_data.items():
        mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
        period_df = df[mask]

        # Count days with non-zero feature values as proxy for having news
        has_signal = (period_df != 0).any(axis=1).sum()
        density = has_signal / total_trading_days if total_trading_days > 0 else 0

        rows.append({"ticker": ticker, "density": density})

    density_df = pd.DataFrame(rows).sort_values("density", ascending=False)
    ranked = density_df["ticker"].tolist()

    if rows:
        top = density_df.head(5)
        logger.info(f"Density ranking (top 5): "
                    f"{list(zip(top['ticker'], (top['density'] * 100).round(1)))}")

    return ranked


# ============================================================================
# PERIOD-BASED SEQUENCE CREATION FOR DA-RNN (walk-forward compatible)
# ============================================================================

def create_darnn_sequences_for_period(
    feature_data: np.ndarray,
    targets: np.ndarray,
    dates: pd.DatetimeIndex,
    seq_length: int,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create windowed DA-RNN sequences for a specific date period.

    Sequences whose TARGET date falls in [period_start, period_end) are included.
    The lookback window naturally extends before period_start.

    Critical alignment (same as regression_darnn.py):
      - Features X:       feature_data[i : i + seq_length]    → Days 1–T
      - Decoder y_history: targets[i-1 : i + seq_length - 1]  → Days 0–(T-1)
      - Prediction target: targets[i + seq_length - 1]        → Day T
    Loop starts at i=1 to accommodate the 1-step lookback.

    Returns (X, y_history, y_target) arrays, all float32.
    """
    X_list, yh_list, y_list = [], [], []

    for i in range(1, len(feature_data) - seq_length + 1):
        target_date = dates[i + seq_length - 1]

        if period_start <= target_date < period_end:
            # Features: [i, i + seq_length)
            seq_x = feature_data[i : i + seq_length]
            # Decoder history: lagged by 1 step → [i-1, i + seq_length - 1)
            seq_yh = targets[i - 1 : i + seq_length - 1]
            # Prediction target: anchored to last day of feature window
            target = targets[i + seq_length - 1]

            X_list.append(seq_x)
            yh_list.append(seq_yh)
            y_list.append(target)

    def _to_array(lst, dtype):
        return np.array(lst, dtype=dtype) if lst else np.array([], dtype=dtype)

    return _to_array(X_list, np.float32), _to_array(yh_list, np.float32), _to_array(y_list, np.float32)


def create_darnn_sequences_for_period_with_dates(
    feature_data: np.ndarray,
    targets: np.ndarray,
    dates: pd.DatetimeIndex,
    seq_length: int,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
    """
    Like create_darnn_sequences_for_period but also returns target dates.
    Used for cross-sectional IC computation.
    """
    X_list, yh_list, y_list, date_list = [], [], [], []

    for i in range(1, len(feature_data) - seq_length + 1):
        target_date = dates[i + seq_length - 1]

        if period_start <= target_date < period_end:
            seq_x = feature_data[i : i + seq_length]
            seq_yh = targets[i - 1 : i + seq_length - 1]
            target = targets[i + seq_length - 1]

            X_list.append(seq_x)
            yh_list.append(seq_yh)
            y_list.append(target)
            date_list.append(target_date)

    def _to_array(lst, dtype):
        return np.array(lst, dtype=dtype) if lst else np.array([], dtype=dtype)

    return (
        _to_array(X_list, np.float32),
        _to_array(yh_list, np.float32),
        _to_array(y_list, np.float32),
        date_list,
    )


# ============================================================================
# REGRESSION METRICS (same as regression_darnn)
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
# CROSS-SECTIONAL EVALUATION (same as regression_darnn)
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
# DA-RNN TRAINER (same as regression_darnn)
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
# EXPERIMENT — Walk-Forward DA-RNN + Sentiment
# ============================================================================

class DARNNSentimentRollingExperiment:
    """
    Walk-forward DA-RNN regression experiment with FNSPID sentiment features.

    Pipeline:
    1. Load sentiment data from cached parquets
    2. Rank tickers by sentiment density, select top N
    3. Load price data (only for tickers with sentiment)
    4. Compute full V7 indicators + 2 sentiment features
    5. Create regression target: vol-adjusted forward return
    6. Pre-compute features for all tickers (done once, reused across folds)
    7. Walk-forward: for each quarterly fold:
       a. Slice train = [start, fold_start), val = [fold_start, fold_end)
       b. Build DA-RNN sequences (X, y_history, y_target) for each period
       c. Fit fresh feature + target scalers on train
       d. Train fresh DARNN with Huber loss
       e. Evaluate on val (in original scale) → store predictions
    8. Aggregate all fold predictions → compute overall metrics
    9. Cross-sectional IC across all folds
    10. Per-stock evaluation using the final (most recent) fold's model
    11. Save config.json + summary.csv + per_fold_results.csv + per_stock_results.csv + model.pt
    """

    def __init__(self, config: Dict):
        self.config = config
        self.use_sentiment = config.get("use_sentiment", True)

        # Date range (aligned with the dense FNSPID sentiment era: 2012-2019)
        self.start_date = config.get("start_date", "2012-01-01")
        self.end_date = config.get("end_date", "2019-12-31")

        # Walk-forward parameters
        self.wf_val_start = pd.Timestamp(config.get("wf_val_start", "2017-01-01"))
        self.wf_val_end = pd.Timestamp(config.get("wf_val_end", "2019-12-31"))
        self.wf_step = config.get("wf_step", "QS")  # QS=quarterly, MS=monthly

        self.max_tickers = config.get("max_tickers", DEFAULT_MAX_TICKERS)

        # Always load sentiment data for ticker selection — this ensures
        # identical ticker universe in both sentiment and no-sentiment modes.
        _sentiment_data = load_sentiment_data()

        if not _sentiment_data:
            logger.warning("No sentiment data found, falling back to EXTENDED_TICKERS")
            available_sent_tickers = EXTENDED_TICKERS[:self.max_tickers]
        else:
            logger.info(f"Tickers with sentiment available (post-exclusion): {len(_sentiment_data)}")

            # Rank by density using ONLY the pre-validation period to avoid
            # look-ahead bias (val/test coverage must not influence selection)
            ranked = rank_tickers_by_density(
                _sentiment_data, self.start_date, str(self.wf_val_start.date())
            )
            available_sent_tickers = ranked[:self.max_tickers]
            logger.info(f"Using top {len(available_sent_tickers)} densest tickers "
                        f"(max={self.max_tickers}, ranked on pre-val period only)")

        # Keep or discard sentiment data based on mode
        if self.use_sentiment:
            self.sentiment_data = {
                t: _sentiment_data[t]
                for t in available_sent_tickers
                if t in _sentiment_data
            }
        else:
            self.sentiment_data = {}
            logger.info("Sentiment DISABLED — same ticker set, no sentiment features")

        # Test stocks: user-specified or default (only those with sentiment)
        self.test_stocks = config.get("test_stocks", DEFAULT_TEST_STOCKS)

        # Filter test stocks to those in the density-ranked universe
        available_set = set(available_sent_tickers)
        self.test_stocks = [t for t in self.test_stocks if t in available_set]
        if not self.test_stocks and available_sent_tickers:
            # Fallback: use last 3 sentiment tickers as test
            self.test_stocks = available_sent_tickers[-3:]
            logger.warning(f"No test stocks in density-ranked universe, using: {self.test_stocks}")

        # Train stocks: density-ranked tickers minus test stocks
        test_set = set(self.test_stocks)
        self.train_stocks = [t for t in available_sent_tickers if t not in test_set]

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
        exp_name = config.get("experiment_name", "wf_darnn_sentiment")
        self.output_dir = REPORTS_DIR / f"{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.indicator_computer = ComprehensiveIndicatorsV7()
        self.cache = PriceCache()

    # ------------------------------------------------------------------
    # Walk-forward fold generation
    # ------------------------------------------------------------------

    def _generate_walk_forward_folds(self) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Generate walk-forward fold boundaries.

        Returns list of (fold_start, fold_end) tuples. Each fold's training data
        is [self.start_date, fold_start) and validation data is [fold_start, fold_end).

        With QS (quarterly): ~12 folds over 2017–2019
        With MS (monthly):   ~36 folds over 2017–2019
        """
        fold_starts = pd.date_range(
            start=self.wf_val_start,
            end=self.wf_val_end,
            freq=self.wf_step,
        )

        folds = []
        for i in range(len(fold_starts)):
            fold_start = fold_starts[i]
            if i + 1 < len(fold_starts):
                fold_end = fold_starts[i + 1]
            else:
                # Last fold: extend to the overall end date
                fold_end = pd.Timestamp(self.end_date) + pd.Timedelta(days=1)
            folds.append((fold_start, fold_end))

        return folds

    # ------------------------------------------------------------------
    # Data loading and feature computation
    # ------------------------------------------------------------------

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

    def _merge_sentiment(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Merge pre-computed daily sentiment features onto a price DataFrame.

        The sentiment data is date-indexed. We align it to the price DataFrame's
        DatetimeIndex via a left join, then fill any gaps with 0 (no news = neutral).
        """
        if not self.use_sentiment or ticker not in self.sentiment_data:
            # No sentiment: all features are 0
            for col in SENTIMENT_FEATURES:
                df[col] = 0.0
            return df

        sent_df = self.sentiment_data[ticker]

        # Align sentiment to price dates
        # Price df is DatetimeIndex, sentiment is also DatetimeIndex (normalized dates)
        price_dates = df.index.normalize()

        # Create aligned sentiment series
        for col in SENTIMENT_FEATURES:
            if col in sent_df.columns:
                # Map sentiment values to price dates
                aligned = sent_df[col].reindex(price_dates)
                df[col] = aligned.values
            else:
                df[col] = 0.0

        # Fill any remaining NaN with 0 (dates outside sentiment coverage)
        for col in SENTIMENT_FEATURES:
            df[col] = df[col].fillna(0.0)

        return df

    def _compute_features_and_target(self, df: pd.DataFrame, ticker: str = "") -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[pd.DatetimeIndex]]:
        """Compute full V7 features + sentiment and vol-adjusted return target for a single stock."""
        df = self.indicator_computer.compute_all(df)

        # Merge sentiment features
        df = self._merge_sentiment(df, ticker)

        # Dynamically discover all indicator columns and exclude leaky ones
        all_indicator_cols = self.indicator_computer.get_indicator_columns(df)
        available = [c for c in all_indicator_cols
                     if c not in self.excluded_features and c in df.columns]

        # Add sentiment features to the available list
        for col in SENTIMENT_FEATURES:
            if col in df.columns and col not in available:
                available.append(col)

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

    def prepare_all_stock_features(
        self, stock_data: Dict, tickers: List[str]
    ) -> Dict[str, Tuple]:
        """
        Pre-compute features for all tickers (done once, reused across folds).

        Returns {ticker: (feature_data, targets, dates)} for tickers that
        have enough data.
        """
        prepared = {}
        n_features = None

        for ticker in tqdm(tickers, desc="Computing features"):
            if ticker not in stock_data:
                continue
            try:
                result = self._compute_features_and_target(stock_data[ticker], ticker)
                if result[0] is None:
                    continue

                feat, tgt, dates = result

                if n_features is None:
                    n_features = feat.shape[-1]
                elif feat.shape[-1] != n_features:
                    continue

                prepared[ticker] = (feat, tgt, dates)
            except Exception as e:
                logger.warning(f"Failed {ticker}: {e}")

        logger.info(f"Prepared features for {len(prepared)} tickers, "
                    f"{n_features} features each")
        return prepared

    # ------------------------------------------------------------------
    # Per-fold data building (DA-RNN 3-input variant)
    # ------------------------------------------------------------------

    def _build_fold_data(
        self,
        prepared: Dict[str, Tuple],
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        val_start: pd.Timestamp,
        val_end: pd.Timestamp,
    ) -> Tuple:
        """
        Build pooled train and val arrays for a single walk-forward fold.

        Returns:
            (X_train, yh_train, y_train, X_val, yh_val, y_val, n_features)
        """
        all_train_X, all_train_yh, all_train_y = [], [], []
        all_val_X, all_val_yh, all_val_y = [], [], []

        for ticker, (feat, tgt, dates) in prepared.items():
            # Training sequences
            X_tr, yh_tr, y_tr = create_darnn_sequences_for_period(
                feat, tgt, dates, self.sequence_length,
                train_start, train_end,
            )
            if len(X_tr) > 0:
                all_train_X.append(X_tr)
                all_train_yh.append(yh_tr)
                all_train_y.append(y_tr)

            # Validation sequences
            X_val, yh_val, y_val = create_darnn_sequences_for_period(
                feat, tgt, dates, self.sequence_length,
                val_start, val_end,
            )
            if len(X_val) > 0:
                all_val_X.append(X_val)
                all_val_yh.append(yh_val)
                all_val_y.append(y_val)

        if not all_train_X or not all_val_X:
            empty = np.array([], dtype=np.float32)
            return empty, empty, empty, empty, empty, empty, 0

        X_train = np.concatenate(all_train_X, axis=0)
        yh_train = np.concatenate(all_train_yh, axis=0)
        y_train = np.concatenate(all_train_y, axis=0)
        X_val = np.concatenate(all_val_X, axis=0)
        yh_val = np.concatenate(all_val_yh, axis=0)
        y_val = np.concatenate(all_val_y, axis=0)
        n_features = X_train.shape[-1]

        return X_train, yh_train, y_train, X_val, yh_val, y_val, n_features

    # ------------------------------------------------------------------
    # Scaling (feature scaler + target scaler, fresh per fold)
    # ------------------------------------------------------------------

    def _scale_data(self, X_train, yh_train, y_train, X_val, yh_val, y_val):
        """
        Fit fresh feature + target scalers on train, transform both splits.

        Uses chunked partial_fit + transform to avoid allocating the full
        flattened array in float64 (which can exceed available RAM).

        Returns:
            (X_train_s, yh_train_s, y_train_s,
             X_val_s, yh_val_s, y_val_s,
             feature_scaler, target_scaler)
        """
        n_train, seq_len, n_features = X_train.shape

        # === Feature scaler (chunked to avoid OOM) ===
        CHUNK_ROWS = 500_000  # rows of the flattened (n*seq, features) array

        feature_scaler = StandardScaler()
        X_flat = X_train.reshape(-1, n_features)  # view, no copy
        n_flat = X_flat.shape[0]
        del X_train  # free the original 3-D array

        # Step 1: partial_fit in chunks to learn mean/var without large allocs
        for start in range(0, n_flat, CHUNK_ROWS):
            chunk = X_flat[start : start + CHUNK_ROWS].astype(np.float64)
            feature_scaler.partial_fit(chunk)
            del chunk

        # Step 2: transform in chunks into a pre-allocated float32 output
        X_train_s = np.empty((n_flat, n_features), dtype=np.float32)
        for start in range(0, n_flat, CHUNK_ROWS):
            end = min(start + CHUNK_ROWS, n_flat)
            chunk = feature_scaler.transform(
                X_flat[start:end].astype(np.float64)
            )
            chunk = chunk.astype(np.float32)
            np.nan_to_num(chunk, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            X_train_s[start:end] = chunk
            del chunk

        del X_flat
        gc.collect()
        X_train_s = X_train_s.reshape(n_train, seq_len, n_features)

        def _transform_features(X):
            if len(X) == 0:
                return X
            s = X.shape
            X_t = feature_scaler.transform(X.reshape(-1, n_features)).astype(np.float32)
            np.nan_to_num(X_t, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            return X_t.reshape(s)

        X_val_s = _transform_features(X_val)

        # === Target scaler ===
        target_scaler = StandardScaler()
        target_scaler.fit(y_train.reshape(-1, 1))

        def _scale_targets_2d(arr):
            """Scale a 2D array (batch, seq_len) of target values."""
            if len(arr) == 0:
                return arr
            shape = arr.shape
            return target_scaler.transform(arr.reshape(-1, 1)).reshape(shape).astype(np.float32)

        def _scale_targets_1d(arr):
            """Scale a 1D array of target values."""
            if len(arr) == 0:
                return arr
            return target_scaler.transform(arr.reshape(-1, 1)).ravel().astype(np.float32)

        yh_train_s = _scale_targets_2d(yh_train)
        yh_val_s = _scale_targets_2d(yh_val)

        y_train_s = _scale_targets_1d(y_train)
        y_val_s = _scale_targets_1d(y_val)

        return (X_train_s, yh_train_s, y_train_s,
                X_val_s, yh_val_s, y_val_s,
                feature_scaler, target_scaler)

    # ------------------------------------------------------------------
    # Prediction helper
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def evaluate_fold_val(self, model, X_val_s, yh_val_s, y_val_s, target_scaler) -> Dict:
        """Evaluate on a fold's validation data, returning metrics in original scale."""
        if len(X_val_s) == 0:
            return {"error": "No data", "n_samples": 0}

        scaled_preds = self._predict_darnn(model, X_val_s, yh_val_s)

        # Inverse transform to original vol-adj return space
        preds_original = target_scaler.inverse_transform(scaled_preds.reshape(-1, 1)).ravel()
        y_original = target_scaler.inverse_transform(y_val_s.reshape(-1, 1)).ravel()

        metrics = compute_regression_metrics(y_original, preds_original)
        metrics["predictions"] = preds_original
        metrics["actuals"] = y_original
        return metrics

    def evaluate_single_stock(
        self, model, stock_data, ticker, feature_scaler, target_scaler,
        val_start, val_end,
    ) -> Dict:
        """Evaluate DA-RNN on a single stock using a fold's scalers."""
        if ticker not in stock_data:
            return {"ticker": ticker, "error": "Not found"}

        result = self._compute_features_and_target(stock_data[ticker], ticker)
        if result[0] is None:
            return {"ticker": ticker, "error": "No data"}

        feat, tgt, dates = result

        X_te, yh_te, y_te = create_darnn_sequences_for_period(
            feat, tgt, dates, self.sequence_length,
            val_start, val_end,
        )

        if len(X_te) == 0:
            return {"ticker": ticker, "error": "No test data in period"}

        # Scale features
        n, seq, f = X_te.shape
        X_scaled = feature_scaler.transform(X_te.reshape(-1, f))
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0).reshape(n, seq, f)

        # Scale decoder history and targets
        yh_scaled = target_scaler.transform(yh_te.reshape(-1, 1)).reshape(yh_te.shape).astype(np.float32)
        y_scaled = target_scaler.transform(y_te.reshape(-1, 1)).ravel().astype(np.float32)

        # Get scaled predictions
        scaled_preds = self._predict_darnn(model, X_scaled, yh_scaled)

        # Inverse transform to original scale for metrics
        preds_original = target_scaler.inverse_transform(scaled_preds.reshape(-1, 1)).ravel()
        y_original = target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()

        metrics = compute_regression_metrics(y_original, preds_original)
        metrics["ticker"] = ticker
        metrics["has_sentiment"] = ticker in self.sentiment_data
        return metrics

    def evaluate_cross_sectional_walkforward(
        self,
        prepared: Dict[str, Tuple],
        model: nn.Module,
        feature_scaler: StandardScaler,
        target_scaler: StandardScaler,
        val_start: pd.Timestamp,
        val_end: pd.Timestamp,
    ) -> Dict[str, Dict]:
        """
        Build per-stock predictions for a single fold's val period.
        Returns {ticker: {dates, predictions, actuals}} for cross-sectional IC.

        Predictions and actuals are returned in ORIGINAL scale.
        """
        per_stock_predictions = {}

        for ticker, (feat, tgt, dates) in prepared.items():
            X_val, yh_val, y_val, val_dates = create_darnn_sequences_for_period_with_dates(
                feat, tgt, dates, self.sequence_length,
                val_start, val_end,
            )

            if len(X_val) == 0:
                continue

            # Scale features
            n, seq, f = X_val.shape
            X_scaled = feature_scaler.transform(X_val.reshape(-1, f))
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0).reshape(n, seq, f)

            # Scale decoder history and targets
            yh_scaled = target_scaler.transform(yh_val.reshape(-1, 1)).reshape(yh_val.shape).astype(np.float32)
            y_scaled = target_scaler.transform(y_val.reshape(-1, 1)).ravel().astype(np.float32)

            # Get scaled predictions → inverse transform
            scaled_preds = self._predict_darnn(model, X_scaled, yh_scaled)
            preds_original = target_scaler.inverse_transform(scaled_preds.reshape(-1, 1)).ravel()
            y_original = target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()

            per_stock_predictions[ticker] = {
                "dates": val_dates,
                "predictions": preds_original,
                "actuals": y_original,
            }

        return per_stock_predictions

    # ------------------------------------------------------------------
    # Sentiment diagnostics
    # ------------------------------------------------------------------

    def _log_sentiment_diagnostics(self, stock_data: Dict):
        """Log diagnostics about sentiment feature coverage in the dataset."""
        if not self.use_sentiment:
            logger.info("Sentiment disabled — skipping diagnostics")
            return

        n_with = sum(1 for t in stock_data if t in self.sentiment_data)
        n_total = len(stock_data)
        logger.info(f"Sentiment coverage: {n_with}/{n_total} loaded stocks have sentiment data")

        # Check sentiment feature stats for a sample ticker
        sample_tickers = [t for t in stock_data if t in self.sentiment_data][:3]
        for ticker in sample_tickers:
            df = stock_data[ticker].copy()
            df = self.indicator_computer.compute_all(df)
            df = self._merge_sentiment(df, ticker)
            sent_cols = [c for c in SENTIMENT_FEATURES if c in df.columns]
            if sent_cols:
                nonzero_pct = (df[sent_cols] != 0).any(axis=1).mean()
                logger.info(f"  {ticker}: {nonzero_pct:.1%} of rows have non-zero sentiment")

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self):
        """Execute the walk-forward DA-RNN + sentiment regression experiment pipeline."""
        folds = self._generate_walk_forward_folds()

        logger.info("=" * 80)
        logger.info("WALK-FORWARD DA-RNN + SENTIMENT REGRESSION EXPERIMENT")
        logger.info("=" * 80)
        logger.info(f"Sentiment: {'ENABLED' if self.use_sentiment else 'DISABLED'}")
        logger.info(f"Frequency: {self.frequency}")
        logger.info(f"Train stocks: {len(self.train_stocks)}, Test stocks: {self.test_stocks}")
        logger.info(f"Date range: {self.start_date} to {self.end_date}")
        logger.info(f"Huber delta: {self.huber_delta}, Vol lookback: {self.vol_lookback}")
        logger.info(f"Encoder hidden: {self.encoder_hidden}, Decoder hidden: {self.decoder_hidden}")
        logger.info(f"Walk-forward: {len(folds)} folds, step={self.wf_step}, "
                    f"expanding window from {self.start_date}")
        for i, (fs, fe) in enumerate(folds):
            logger.info(f"  Fold {i+1}: val [{fs.date()} .. {fe.date()})")

        # Save config
        config_to_save = {k: v for k, v in self.config.items()
                         if not isinstance(v, (list,)) or len(v) < 20}
        config_to_save["train_stock_count"] = len(self.train_stocks)
        config_to_save["test_stocks"] = self.test_stocks
        config_to_save["sentiment_features"] = SENTIMENT_FEATURES if self.use_sentiment else []
        config_to_save["train_stocks_list"] = self.train_stocks
        config_to_save["n_folds"] = len(folds)
        config_to_save["folds"] = [
            {"fold_start": str(fs.date()), "fold_end": str(fe.date())}
            for fs, fe in folds
        ]
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(config_to_save, f, indent=2, default=str)

        # Load data
        stock_data = self.load_data()

        # Sentiment diagnostics
        self._log_sentiment_diagnostics(stock_data)

        # Pre-compute features for all stocks (done once, reused across folds)
        # Include test stocks so cross-sectional IC can use them
        all_tickers = list(set(self.train_stocks + self.test_stocks))
        logger.info("\n" + "=" * 40)
        logger.info("PRE-COMPUTING FEATURES (once for all folds)")
        logger.info("=" * 40)
        prepared = self.prepare_all_stock_features(stock_data, all_tickers)

        if not prepared:
            logger.error("No stocks prepared successfully — aborting")
            return {"error": "No data"}

        # Separate prepared data for training (exclude test stocks from training folds)
        test_set = set(self.test_stocks)
        prepared_train = {t: v for t, v in prepared.items() if t not in test_set}

        # === WALK-FORWARD LOOP ===
        train_start = pd.Timestamp(self.start_date)
        fold_results = []
        all_val_preds = []
        all_val_true = []
        all_cs_predictions = {}  # Accumulate cross-sectional predictions across folds
        total_train_time = 0.0
        last_model = None
        last_feature_scaler = None
        last_target_scaler = None
        n_features = None

        for fold_idx, (fold_start, fold_end) in enumerate(folds):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"FOLD {fold_idx+1}/{len(folds)}: "
                       f"Train [{train_start.date()} .. {fold_start.date()}) → "
                       f"Val [{fold_start.date()} .. {fold_end.date()})")
            logger.info(f"{'=' * 60}")

            # Build train/val arrays for this fold
            X_train, yh_train, y_train, X_val, yh_val, y_val, nf = self._build_fold_data(
                prepared_train, train_start, fold_start, fold_start, fold_end
            )

            if len(X_train) == 0 or len(X_val) == 0:
                logger.warning(f"Fold {fold_idx+1}: Insufficient data "
                             f"(train={len(X_train)}, val={len(X_val)}) — skipping")
                fold_results.append({
                    "fold": fold_idx + 1,
                    "val_start": str(fold_start.date()),
                    "val_end": str(fold_end.date()),
                    "error": "Insufficient data",
                })
                continue

            if n_features is None:
                n_features = nf

            logger.info(f"  Train: {len(X_train):,} samples, Val: {len(X_val):,} samples")
            logger.info(f"  Target stats (train): mean={y_train.mean():.4f}, "
                       f"std={y_train.std():.4f}")

            # Scale (fresh scalers per fold)
            (X_train_s, yh_train_s, y_train_s,
             X_val_s, yh_val_s, y_val_s,
             feature_scaler, target_scaler) = self._scale_data(
                X_train, yh_train, y_train, X_val, yh_val, y_val
            )
            del X_train, yh_train, y_train, X_val, yh_val, y_val
            gc.collect()

            # Create FRESH model
            model = DARNN(
                input_size=n_features,
                seq_length=self.sequence_length,
                encoder_hidden=self.encoder_hidden,
                decoder_hidden=self.decoder_hidden,
                dropout=self.dropout,
            )

            if fold_idx == 0:
                n_params = sum(p.numel() for p in model.parameters())
                logger.info(f"  Model parameters: {n_params:,}")
                logger.info(f"  Features: {n_features} (V7 indicators + "
                           f"{len(SENTIMENT_FEATURES) if self.use_sentiment else 0} sentiment)")

            # Train
            trainer = DARNNTrainer(
                model,
                learning_rate=self.learning_rate,
                huber_delta=self.huber_delta,
            )

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

            fold_start_time = time.time()
            trainer.train(train_loader, val_loader, epochs=self.epochs, patience=self.patience)
            fold_train_time = time.time() - fold_start_time
            total_train_time += fold_train_time

            # Evaluate on validation set (in original scale)
            val_metrics = self.evaluate_fold_val(model, X_val_s, yh_val_s, y_val_s, target_scaler)

            # Collect predictions for aggregate metrics
            if "predictions" in val_metrics:
                all_val_preds.append(val_metrics["predictions"])
                all_val_true.append(val_metrics["actuals"])

            # Cross-sectional IC for this fold (using ALL stocks including test)
            cs_preds = self.evaluate_cross_sectional_walkforward(
                prepared, model, feature_scaler, target_scaler, fold_start, fold_end
            )
            # Merge into accumulated predictions
            for ticker, data in cs_preds.items():
                if ticker not in all_cs_predictions:
                    all_cs_predictions[ticker] = {
                        "dates": [], "predictions": [], "actuals": []
                    }
                all_cs_predictions[ticker]["dates"].extend(data["dates"])
                all_cs_predictions[ticker]["predictions"] = np.concatenate([
                    all_cs_predictions[ticker]["predictions"]
                    if isinstance(all_cs_predictions[ticker]["predictions"], np.ndarray)
                    and len(all_cs_predictions[ticker]["predictions"]) > 0
                    else np.array([]),
                    data["predictions"]
                ])
                all_cs_predictions[ticker]["actuals"] = np.concatenate([
                    all_cs_predictions[ticker]["actuals"]
                    if isinstance(all_cs_predictions[ticker]["actuals"], np.ndarray)
                    and len(all_cs_predictions[ticker]["actuals"]) > 0
                    else np.array([]),
                    data["actuals"]
                ])

            fold_result = {
                "fold": fold_idx + 1,
                "val_start": str(fold_start.date()),
                "val_end": str(fold_end.date()),
                "train_samples": int(len(X_train_s)),
                "val_samples": int(len(X_val_s)),
                "train_time": fold_train_time,
                "mse": val_metrics.get("mse", 0),
                "mae": val_metrics.get("mae", 0),
                "r2": val_metrics.get("r2", 0),
                "ic": val_metrics.get("ic", 0),
                "directional_accuracy": val_metrics.get("directional_accuracy", 0),
            }
            fold_results.append(fold_result)

            logger.info(f"  → MSE={fold_result['mse']:.4f}, "
                       f"IC={fold_result['ic']:.4f}, "
                       f"DirAcc={fold_result['directional_accuracy']:.2%}, "
                       f"Time={fold_train_time:.1f}s")

            # Keep last fold's model and scalers for per-stock evaluation
            last_model = model
            last_feature_scaler = feature_scaler
            last_target_scaler = target_scaler

            # Cleanup
            del X_train_s, yh_train_s, y_train_s, X_val_s, yh_val_s, y_val_s
            del train_loader, val_loader, trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # === AGGREGATE METRICS ACROSS ALL FOLDS ===
        logger.info(f"\n{'=' * 60}")
        logger.info("AGGREGATED WALK-FORWARD RESULTS")
        logger.info(f"{'=' * 60}")

        valid_folds = [r for r in fold_results if "error" not in r]

        if all_val_preds:
            agg_preds = np.concatenate(all_val_preds)
            agg_true = np.concatenate(all_val_true)
            agg_metrics = compute_regression_metrics(agg_true, agg_preds)
        else:
            agg_metrics = {"mse": 0, "mae": 0, "r2": 0, "ic": 0,
                          "directional_accuracy": 0, "n_samples": 0}

        logger.info(f"Total val samples: {agg_metrics.get('n_samples', 0):,}")
        logger.info(f"Aggregate MSE={agg_metrics.get('mse', 0):.4f}, "
                    f"IC={agg_metrics.get('ic', 0):.4f}, "
                    f"DirAcc={agg_metrics.get('directional_accuracy', 0):.2%}, "
                    f"R²={agg_metrics.get('r2', 0):.4f}")

        # Per-fold breakdown
        ics, dir_accs, mses = [], [], []
        if valid_folds:
            ics = [r["ic"] for r in valid_folds]
            dir_accs = [r["directional_accuracy"] for r in valid_folds]
            mses = [r["mse"] for r in valid_folds]
            logger.info(f"Per-fold IC: mean={np.mean(ics):.4f}, "
                       f"std={np.std(ics):.4f}, "
                       f"min={np.min(ics):.4f}, max={np.max(ics):.4f}")
            logger.info(f"Per-fold DirAcc: mean={np.mean(dir_accs):.2%}, "
                       f"std={np.std(dir_accs):.2%}")

        # === CROSS-SECTIONAL IC (aggregated across all folds) ===
        logger.info(f"\n{'=' * 40}")
        logger.info("CROSS-SECTIONAL EVALUATION (Ranking Quality)")
        logger.info(f"{'=' * 40}")

        cs_metrics = compute_cross_sectional_ic(all_cs_predictions)
        logger.info(f"Cross-Sectional IC: {cs_metrics['cross_sectional_ic']:.4f}")
        logger.info(f"IC IR (consistency): {cs_metrics['ic_ir']:.4f}")
        logger.info(f"IC Hit Rate: {cs_metrics['ic_hit_rate']:.2%}")
        logger.info(f"Quantile Spread (Q5-Q1): {cs_metrics['quantile_spread']:.4f}")
        logger.info(f"Evaluated on {cs_metrics['n_dates']} dates")

        # === PER-STOCK EVALUATION (using last fold's model) ===
        per_stock_results = []
        if last_model is not None and last_feature_scaler is not None:
            logger.info(f"\n{'=' * 40}")
            logger.info("PER-STOCK EVALUATION (last fold model)")
            logger.info(f"{'=' * 40}")

            # Use the last fold's validation period for per-stock eval
            last_fold_start, last_fold_end = folds[-1]

            for ticker in self.test_stocks:
                result = self.evaluate_single_stock(
                    last_model, stock_data, ticker,
                    last_feature_scaler, last_target_scaler,
                    last_fold_start, last_fold_end,
                )
                per_stock_results.append(result)
                if "error" not in result:
                    sent_flag = "📰" if result.get("has_sentiment") else "  "
                    logger.info(f"{sent_flag} {ticker}: MSE={result['mse']:.4f}, "
                               f"IC={result['ic']:.4f}, "
                               f"DirAcc={result['directional_accuracy']:.2%}, "
                               f"n={result['n_samples']}")
                else:
                    logger.warning(f"{ticker}: {result['error']}")

        # === SAVE RESULTS ===
        valid_ps = [r for r in per_stock_results if "error" not in r]
        avg_mse = np.mean([r["mse"] for r in valid_ps]) if valid_ps else 0
        avg_mae = np.mean([r["mae"] for r in valid_ps]) if valid_ps else 0
        avg_ic = np.mean([r["ic"] for r in valid_ps]) if valid_ps else 0
        avg_r2 = np.mean([r["r2"] for r in valid_ps]) if valid_ps else 0
        avg_dir_acc = np.mean([r["directional_accuracy"] for r in valid_ps]) if valid_ps else 0

        n_params = sum(p.numel() for p in last_model.parameters()) if last_model else 0

        summary = {
            "experiment_name": self.config.get("experiment_name", "wf_darnn_sentiment"),
            "task": "regression",
            "model": "DARNN",
            "target": "vol_adj_forward_return",
            "loss_function": f"huber_delta_{self.huber_delta}",
            "frequency": self.frequency,
            "use_sentiment": self.use_sentiment,
            "n_features": n_features,
            "n_sentiment_features": len(SENTIMENT_FEATURES) if self.use_sentiment else 0,
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
            "total_train_time": total_train_time,
            # Walk-forward params
            "wf_step": self.wf_step,
            "wf_val_start": str(self.wf_val_start.date()),
            "wf_val_end": str(self.wf_val_end.date()),
            "n_folds": len(folds),
            "n_valid_folds": len(valid_folds),
            # Aggregated walk-forward metrics (all folds pooled)
            "wf_agg_mse": agg_metrics.get("mse", 0),
            "wf_agg_mae": agg_metrics.get("mae", 0),
            "wf_agg_r2": agg_metrics.get("r2", 0),
            "wf_agg_ic": agg_metrics.get("ic", 0),
            "wf_agg_directional_accuracy": agg_metrics.get("directional_accuracy", 0),
            "wf_agg_n_samples": agg_metrics.get("n_samples", 0),
            # Per-fold mean metrics
            "wf_mean_ic": float(np.mean(ics)) if valid_folds else 0,
            "wf_std_ic": float(np.std(ics)) if valid_folds else 0,
            "wf_mean_mse": float(np.mean(mses)) if valid_folds else 0,
            "wf_mean_directional_accuracy": float(np.mean(dir_accs)) if valid_folds else 0,
            # Cross-sectional (ranking) metrics
            "cross_sectional_ic": cs_metrics.get("cross_sectional_ic", 0),
            "ic_ir": cs_metrics.get("ic_ir", 0),
            "ic_hit_rate": cs_metrics.get("ic_hit_rate", 0),
            "quantile_spread": cs_metrics.get("quantile_spread", 0),
            "cs_n_dates": cs_metrics.get("n_dates", 0),
            # Per-stock averages (last fold model)
            "per_stock_avg_mse": avg_mse,
            "per_stock_avg_mae": avg_mae,
            "per_stock_avg_ic": avg_ic,
            "per_stock_avg_r2": avg_r2,
            "per_stock_avg_directional_accuracy": avg_dir_acc,
            "per_stock_count": len(valid_ps),
        }

        pd.DataFrame([summary]).to_csv(self.output_dir / "summary.csv", index=False)
        pd.DataFrame(fold_results).to_csv(self.output_dir / "per_fold_results.csv", index=False)
        pd.DataFrame(per_stock_results).to_csv(self.output_dir / "per_stock_results.csv", index=False)
        if last_model is not None:
            torch.save(last_model.state_dict(), self.output_dir / "model.pt")

        logger.info(f"\n{'=' * 60}")
        logger.info("SUMMARY")
        logger.info(f"{'=' * 60}")
        logger.info(f"DARNN ({n_params:,} params)")
        logger.info(f"Sentiment: {'ENABLED' if self.use_sentiment else 'DISABLED'}")
        logger.info(f"Features: {n_features} (V7 indicators + "
                    f"{len(SENTIMENT_FEATURES) if self.use_sentiment else 0} sentiment)")
        logger.info(f"Walk-forward: {len(valid_folds)}/{len(folds)} folds, "
                    f"total train time: {total_train_time:.1f}s")
        logger.info(f"WF Aggregate: MSE={agg_metrics.get('mse', 0):.4f}, "
                    f"IC={agg_metrics.get('ic', 0):.4f}, "
                    f"DirAcc={agg_metrics.get('directional_accuracy', 0):.2%}")
        if valid_folds:
            logger.info(f"WF Per-fold:  IC={np.mean(ics):.4f} ± {np.std(ics):.4f}")
        logger.info(f"Ranking:      CS-IC={cs_metrics.get('cross_sectional_ic', 0):.4f}, "
                    f"ICIR={cs_metrics.get('ic_ir', 0):.4f}, "
                    f"Q-Spread={cs_metrics.get('quantile_spread', 0):.4f}")
        logger.info(f"Per-stock:    IC={avg_ic:.4f}, DirAcc={avg_dir_acc:.2%}")
        logger.info(f"Results saved to: {self.output_dir}")

        return summary


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward DA-RNN + Sentiment Regression")
    parser.add_argument("--epochs", type=int, default=100)
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
    parser.add_argument("--no-sentiment", action="store_true", help="Disable sentiment features (baseline)")
    parser.add_argument("--max-tickers", type=int, default=DEFAULT_MAX_TICKERS,
                        help=f"Max tickers by density rank (default: {DEFAULT_MAX_TICKERS})")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--name", type=str, default="wf_darnn_sentiment", help="Experiment name")
    # Date range
    parser.add_argument("--start-date", type=str, default="2012-01-01")
    parser.add_argument("--end-date", type=str, default="2019-12-31")
    # Walk-forward params
    parser.add_argument("--wf-val-start", type=str, default="2017-01-01",
                        help="First fold validation start (default: 2017-01-01)")
    parser.add_argument("--wf-val-end", type=str, default="2019-12-31",
                        help="Last fold validation end (default: 2019-12-31)")
    parser.add_argument("--wf-step", type=str, default="QS",
                        choices=["MS", "QS"],
                        help="Fold frequency: MS=monthly (36 folds), QS=quarterly (12 folds)")
    args = parser.parse_args()

    config = {
        "experiment_name": args.name,
        "frequency": args.frequency,
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
        "use_sentiment": not args.no_sentiment,
        "max_tickers": args.max_tickers,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "wf_val_start": args.wf_val_start,
        "wf_val_end": args.wf_val_end,
        "wf_step": args.wf_step,
    }

    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.seq_length:
        config["sequence_length"] = args.seq_length

    experiment = DARNNSentimentRollingExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
