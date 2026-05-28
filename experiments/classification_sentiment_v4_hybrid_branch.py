"""
Classification with Sentiment — V4 Hybrid Branch Architecture.

Extends v3 (walk-forward rolling validation) by replacing the monolithic
LSTM with a DUAL-BRANCH architecture that processes technical indicators
and sentiment features through separate pathways:

  Branch 1 (Technical): Multi-layer LSTM + multi-head self-attention
      over the 20 SHAP-selected technical indicators.  Extracts
      structural market momentum.

  Branch 2 (Sentiment): Lightweight single-layer GRU over the 2
      RF-selected sentiment features (sent_surprise, sent_ema_20d).
      Respects the lower dimensionality and noisier nature of news data.

  Fusion: Concatenate branch outputs, then Dense layers → classification.

This prevents the fundamental issue of v3 where noisy sentiment features
were concatenated with clean technical data at the input layer, forcing
the LSTM to treat them identically.

Walk-forward validation simulates realistic deployment:
  - Each fold trains on ALL data up to the fold's start (expanding window)
  - Validates on a single quarter (e.g., Q1 2017)
  - A FRESH model is trained per fold — no warm-starting

Key design decisions:
  - Training universe restricted to tickers with sentiment coverage
  - Date range: 2012–2019 (aligned with dense FNSPID sentiment era)
  - Walk-forward folds: quarterly expanding window (12 folds over 2017–2019)
  - 2 sentiment features selected by RandomForest permutation importance
    (classification task, scoring=accuracy):
    sent_surprise, sent_ema_20d
    (out of 8 candidates; other 6 disabled to reduce noise)
  - No-news days filled via exponential decay (sentiment fades, doesn't vanish)
  - Classification target: 3-class trend (down/neutral/up) via return thresholds
  - Feature vector layout: [technical_cols] + [sentiment_cols] — the model
    splits internally by feature index

Usage:
    python experiments/classification_sentiment_v4_hybrid_branch.py --epochs 100
    python experiments/classification_sentiment_v4_hybrid_branch.py --epochs 2 --name smoke_hybrid  # smoke test
    python experiments/classification_sentiment_v4_hybrid_branch.py --no-sentiment --name baseline  # A/B
    python experiments/classification_sentiment_v4_hybrid_branch.py --model attention --name attn_flat  # A/B
    python experiments/classification_sentiment_v4_hybrid_branch.py --model hybrid --name hybrid_v4  # default
    python experiments/classification_sentiment_v4_hybrid_branch.py --wf-step QS  # quarterly (default)
    python experiments/classification_sentiment_v4_hybrid_branch.py --wf-step MS  # monthly (36 folds)
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
from collections import Counter

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

from experiments.ranked_tickers import EXTENDED_TICKERS
from src.data.cache import PriceCache
from src.data.utils import resample_to_weekly, load_stocks
from src.evaluation.metrics import compute_metrics, evaluate_model_on_data
from src.features.indicators_v7 import ComprehensiveIndicatorsV7

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
    logger.info("CUDA: Not available -- running on CPU")

# ============================================================================
# FEATURES
# ============================================================================

SHAP_TOP20_FEATURES = [
    "atr_pct", "intraday_range", "bb_BBB_5_2.0_2.0", "gap", "cci",
    "high_low_pct", "volume_ratio", "adx_DMN_14", "return_1d", "aroon_AROOND_14",
    "adx_DMP_14", "obv", "ad", "bear_power", "roc_10",
    "pvo_PVOh_12_26_9", "trix_TRIXs_30_9", "rsi_14", "aroon_AROONU_14", "return_5d"
]

# RF permutation importance selected these 2 from 8 candidates for classification
# (see experiments/sentiment_feature_selection.py for full ranking).
SENTIMENT_FEATURES = [
    "sent_surprise",       # sent_mean - 20-day rolling mean -- deviations from baseline  (perm #1)
    "sent_ema_20d",        # 20-day EMA of sent_mean -- long-horizon sentiment regime    (perm #2)
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
    "MSFT", "AAPL", "JPM", "JNJ", "XOM",
    "WMT", "DIS", "BA", "KO", "NVDA",
]

# ============================================================================
# SENTIMENT DATA LOADER
# ============================================================================

def load_sentiment_data(sentiment_dir: Path = None) -> Dict[str, pd.DataFrame]:
    """
    Load all cached sentiment parquets and aggregate to daily features.

    Only computes the 2 RF-selected sentiment features for classification:
      - sent_surprise: sent_mean - 20-day rolling mean (deviation from baseline)
      - sent_ema_20d:  20-day EMA of sent_mean (long-horizon regime)

    Normalization & time alignment:
      - Sentiment dates are normalized to midnight (trading_date)
      - Complete business-day index is created to expose gaps
      - Missing days are forward-filled with 0.95 exponential decay
      - Remaining leading NaNs filled with 0 (no news = neutral)

    Returns:
        {ticker: DataFrame} where DataFrame is date-indexed with
        columns: sent_surprise, sent_ema_20d
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

            # Normalize date column -- strip time, align to midnight
            raw['date'] = pd.to_datetime(raw['date'], utc=True).dt.tz_localize(None)
            raw['trading_date'] = raw['date'].dt.normalize()  # Strip time component

            # === Aggregate to daily level per ticker ===
            daily = raw.groupby('trading_date').agg(
                sent_mean=('sentiment_value', 'mean'),
                sent_count=('sentiment_value', 'count'),
            ).sort_index()

            # === Compute only the 2 selected features ===

            # Sentiment surprise: deviation from 20-day rolling mean
            # High surprise = abnormal sentiment -> stronger directional signal
            rolling_mean_20 = daily['sent_mean'].rolling(20, min_periods=5).mean()
            daily['sent_surprise'] = daily['sent_mean'] - rolling_mean_20

            # 20-day EMA: captures long-horizon sentiment regime shifts
            daily['sent_ema_20d'] = daily['sent_mean'].ewm(span=20, min_periods=1).mean()

            # === Forward-fill with exponential decay for no-news days ===
            # Create a complete business day index to fill gaps
            if len(daily) > 0:
                full_idx = pd.bdate_range(daily.index.min(), daily.index.max())
                daily = daily.reindex(full_idx)

                # Track which days had news
                had_news = daily['sent_count'].notna()

                # Decay fill: for each gap, previous value x 0.95^days_since_news
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

                    # Apply decay: value x decay_factor^gap_length
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


def get_tickers_with_sentiment(sentiment_dir: Path = None) -> List[str]:
    """Return list of tickers that have sentiment parquet files (excluding noisy ones)."""
    if sentiment_dir is None:
        sentiment_dir = CACHE_DIR / "sentiment"

    if not sentiment_dir.exists():
        return []

    return sorted(
        t for t in (
            f.stem.replace("_sentiment", "")
            for f in sentiment_dir.glob("*_sentiment.parquet")
        )
        if t not in EXCLUDE_TICKERS
    )


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
# MODEL FACTORY
# ============================================================================

def create_model(
    model_type: str,
    input_size: int,
    n_technical: int,
    n_sentiment: int,
    dropout: float = 0.2,
) -> nn.Module:
    """
    Create model by type name.

    Args:
        model_type: One of 'simple', 'attention', 'hybrid'.
        input_size: Total number of input features (technical + sentiment).
        n_technical: Number of technical features (first N columns).
        n_sentiment: Number of sentiment features (last M columns).
        dropout: Dropout probability.

    Returns:
        nn.Module with forward() -> (trend_logits, None/conf, None/attn).
    """
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
    elif model_type == "hybrid":
        from src.models.hybrid_attention_lstm import HybridAttentionLSTM
        return HybridAttentionLSTM(
            n_technical_features=n_technical,
            n_sentiment_features=n_sentiment,
            tech_hidden_size=128,
            tech_num_layers=2,
            tech_num_heads=4,
            sent_hidden_size=32,
            sent_num_layers=1,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ============================================================================
# TRAINER (classification with CrossEntropyLoss)
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
# WALK-FORWARD SEQUENCE UTILITIES
# ============================================================================

def create_sequences_for_period(
    feature_data: np.ndarray,
    labels: np.ndarray,
    dates: pd.DatetimeIndex,
    seq_length: int,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create windowed sequences for a specific date period.

    Sequences whose TARGET date falls in [period_start, period_end) are included.
    The lookback window naturally extends before period_start.

    Args:
        feature_data: Array of shape (n_timesteps, n_features).
        labels: Array of shape (n_timesteps,).
        dates: DatetimeIndex aligned with feature_data.
        seq_length: Number of timesteps per sequence.
        period_start: Inclusive start date for target dates.
        period_end: Exclusive end date for target dates.

    Returns:
        (X, y) tuple of arrays.
    """
    X_list, y_list = [], []

    for i in range(len(feature_data) - seq_length):
        target_date = dates[i + seq_length]
        if period_start <= target_date < period_end:
            X_list.append(feature_data[i:i + seq_length])
            y_list.append(labels[i + seq_length - 1])

    if X_list:
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)
    return np.array([], dtype=np.float32), np.array([], dtype=np.int64)


# ============================================================================
# EXPERIMENT
# ============================================================================

class HybridBranchExperiment:
    """
    Walk-forward classification experiment with dual-branch hybrid architecture.

    Pipeline:
    1. Load sentiment data from cached parquets
    2. Load price data (only for tickers with sentiment)
    3. Compute SHAP Top 20 features + 2 sentiment features (optional)
    4. Create classification target: 3-class trend (down/neutral/up)
    5. Walk-forward: for each quarterly fold:
       a. Slice train = [start, fold_start), val = [fold_start, fold_end)
       b. Fit fresh scaler on train
       c. Train fresh model with early stopping on val
       d. Evaluate on val -> store predictions
    6. Aggregate all fold predictions -> compute overall metrics
    7. Per-stock evaluation using the final (most recent) fold's model
    8. Save config.json + summary.csv + per_fold_results.csv + per_stock_results.csv + model.pt

    Unlike v3, the model splits features internally:
    - Technical features (first N columns) -> LSTM + Attention branch
    - Sentiment features (last M columns)  -> GRU branch
    - Fusion: concatenate branch outputs -> Dense -> classification
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

        # Always load sentiment data for ticker selection -- this ensures
        # identical ticker universe in both sentiment and no-sentiment modes.
        _sentiment_data = load_sentiment_data()

        if not _sentiment_data:
            logger.warning("No sentiment data found, falling back to EXTENDED_TICKERS")
            available_sent_tickers = EXTENDED_TICKERS[:config.get("stocks", 400)]
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
            logger.info("Sentiment DISABLED -- same ticker set, no sentiment features")

        # Test stocks: user-specified or default (only those in density-ranked universe)
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
        self.model_type = config.get("model_type", "hybrid")  # Default changed to hybrid
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

        # Feature columns — layout: [technical_cols] + [sentiment_cols]
        # The model splits internally by index, so ordering is critical.
        self.technical_cols = config.get("features", SHAP_TOP20_FEATURES)
        if self.use_sentiment:
            self.feature_cols = self.technical_cols + SENTIMENT_FEATURES
        else:
            self.feature_cols = list(self.technical_cols)

        # Track feature counts for the model factory
        self.n_technical = len(self.technical_cols)
        self.n_sentiment = len(SENTIMENT_FEATURES) if self.use_sentiment else 0

        # Output directory
        exp_name = config.get("experiment_name", "hybrid_branch_v4")
        self.output_dir = REPORTS_DIR / f"{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.indicator_computer = ComprehensiveIndicatorsV7()
        self.cache = PriceCache()
        self.scaler = None  # Will hold the LAST fold's scaler for per-stock eval

    def _generate_walk_forward_folds(self) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Generate walk-forward fold boundaries.

        Returns list of (fold_start, fold_end) tuples. Each fold's training data
        is [self.start_date, fold_start) and validation data is [fold_start, fold_end).

        With QS (quarterly): ~12 folds over 2017-2019
        With MS (monthly):   ~36 folds over 2017-2019
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

    def prepare_stock_features(self, df: pd.DataFrame, ticker: str = "") -> Tuple:
        """
        Compute features and labels for a stock's full date range.

        Returns raw (feature_data, labels, dates) WITHOUT splitting into
        train/val/test. Splitting is done per-fold by the walk-forward loop.

        Returns:
            (feature_data, labels, dates) or (empty, empty, empty) on failure.
        """
        df = self.indicator_computer.compute_shap_top20(df)

        # Merge sentiment
        df = self._merge_sentiment(df, ticker)

        available = [c for c in self.feature_cols if c in df.columns]
        if len(available) < 5:
            return np.array([]), np.array([]), pd.DatetimeIndex([])

        # Classification target: 3-class trend (down/neutral/up)
        df['return_next'] = df['Close'].pct_change().shift(-1)
        df['trend'] = pd.cut(
            df['return_next'],
            bins=[-np.inf, self.threshold_low, self.threshold_high, np.inf],
            labels=[0, 1, 2]
        ).astype(float)

        df = df.dropna(subset=['trend'] + available[:5])
        if len(df) < self.sequence_length + 10:
            return np.array([]), np.array([]), pd.DatetimeIndex([])

        feature_data = df[available].values
        feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
        labels = df['trend'].values

        return feature_data, labels, df.index

    def prepare_all_stock_features(
        self, stock_data: Dict, tickers: List[str]
    ) -> Dict[str, Tuple]:
        """
        Pre-compute features for all tickers (done once, reused across folds).

        Returns {ticker: (feature_data, labels, dates)} for tickers that
        have enough data.
        """
        prepared = {}
        n_features = None

        for ticker in tqdm(tickers, desc="Computing features"):
            if ticker not in stock_data:
                continue
            try:
                feat, lab, dates = self.prepare_stock_features(
                    stock_data[ticker], ticker
                )
                if len(feat) == 0:
                    continue

                if n_features is None:
                    n_features = feat.shape[-1]
                elif feat.shape[-1] != n_features:
                    continue

                prepared[ticker] = (feat, lab, dates)
            except Exception as e:
                logger.warning(f"Failed {ticker}: {e}")

        logger.info(f"Prepared features for {len(prepared)} tickers, "
                    f"{n_features} features each")
        return prepared

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

        Args:
            prepared: {ticker: (feature_data, labels, dates)} from prepare_all_stock_features
            train_start: Inclusive start of training window.
            train_end: Exclusive end of training window (= val_start).
            val_start: Inclusive start of validation window.
            val_end: Exclusive end of validation window.

        Returns:
            (X_train, y_train, X_val, y_val, n_features)
        """
        all_train_X, all_train_y = [], []
        all_val_X, all_val_y = [], []

        for ticker, (feat, lab, dates) in prepared.items():
            # Training sequences
            X_tr, y_tr = create_sequences_for_period(
                feat, lab, dates, self.sequence_length,
                train_start, train_end,
            )
            if len(X_tr) > 0:
                all_train_X.append(X_tr)
                all_train_y.append(y_tr)

            # Validation sequences
            X_val, y_val = create_sequences_for_period(
                feat, lab, dates, self.sequence_length,
                val_start, val_end,
            )
            if len(X_val) > 0:
                all_val_X.append(X_val)
                all_val_y.append(y_val)

        if not all_train_X or not all_val_X:
            return (np.array([]), np.array([]),
                    np.array([]), np.array([]), 0)

        X_train = np.concatenate(all_train_X, axis=0)
        y_train = np.concatenate(all_train_y, axis=0)
        X_val = np.concatenate(all_val_X, axis=0)
        y_val = np.concatenate(all_val_y, axis=0)
        n_features = X_train.shape[-1]

        return X_train, y_train, X_val, y_val, n_features

    def _scale_data(self, X_train, X_val):
        """Fit a fresh scaler on train, transform both splits. Returns (X_train_s, X_val_s, scaler)."""
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

        return X_train_s, _transform(X_val), scaler

    def evaluate_single_stock(self, model, stock_data, ticker, scaler) -> Dict:
        """Evaluate model on a single stock using the final fold's scaler."""
        if ticker not in stock_data:
            return {"ticker": ticker, "error": "Not found"}

        feat, lab, dates = self.prepare_stock_features(stock_data[ticker], ticker)

        if len(feat) == 0:
            return {"ticker": ticker, "error": "No data"}

        # Use the last fold's validation period for per-stock eval
        folds = self._generate_walk_forward_folds()
        last_fold_start, last_fold_end = folds[-1]

        X_test, y_test = create_sequences_for_period(
            feat, lab, dates, self.sequence_length,
            last_fold_start, last_fold_end,
        )

        if len(X_test) == 0:
            return {"ticker": ticker, "error": "No test data in last fold"}

        # Scale with the provided scaler
        n, seq, f = X_test.shape
        X_scaled = scaler.transform(X_test.reshape(-1, f))
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0).reshape(n, seq, f)

        metrics = evaluate_model_on_data(model, X_scaled, y_test)
        metrics["ticker"] = ticker
        metrics["has_sentiment"] = ticker in self.sentiment_data
        metrics.pop("predictions", None)
        metrics.pop("class_distribution", None)
        return metrics

    def _log_sentiment_diagnostics(self, stock_data: Dict):
        """Log diagnostics about sentiment feature coverage in the dataset."""
        if not self.use_sentiment:
            logger.info("Sentiment disabled -- skipping diagnostics")
            return

        n_with = sum(1 for t in stock_data if t in self.sentiment_data)
        n_total = len(stock_data)
        logger.info(f"Sentiment coverage: {n_with}/{n_total} loaded stocks have sentiment data")

        # Check sentiment feature stats for a sample ticker
        sample_tickers = [t for t in stock_data if t in self.sentiment_data][:3]
        for ticker in sample_tickers:
            df = stock_data[ticker].copy()
            df = self.indicator_computer.compute_shap_top20(df)
            df = self._merge_sentiment(df, ticker)
            sent_cols = [c for c in SENTIMENT_FEATURES if c in df.columns]
            if sent_cols:
                nonzero_pct = (df[sent_cols] != 0).any(axis=1).mean()
                logger.info(f"  {ticker}: {nonzero_pct:.1%} of rows have non-zero sentiment")

    def run(self):
        """Execute the walk-forward hybrid-branch experiment pipeline."""
        folds = self._generate_walk_forward_folds()

        logger.info("=" * 80)
        logger.info("WALK-FORWARD HYBRID BRANCH CLASSIFICATION + SENTIMENT EXPERIMENT (V4)")
        logger.info("=" * 80)
        logger.info(f"Sentiment: {'ENABLED' if self.use_sentiment else 'DISABLED'}")
        logger.info(f"Model: {self.model_type}, Frequency: {self.frequency}")
        logger.info(f"Train stocks: {len(self.train_stocks)}, Test stocks: {self.test_stocks}")
        logger.info(f"Thresholds: [{self.threshold_low}, {self.threshold_high}]")
        logger.info(f"Features: {len(self.feature_cols)} ({self.n_technical} technical + "
                    f"{self.n_sentiment} sentiment)")
        if self.model_type == "hybrid":
            logger.info(f"Architecture: Tech branch (LSTM 128x2 + Attn 4h) | "
                        f"Sent branch (GRU 32x1) | Fusion -> 64 -> 3")
        logger.info(f"Date range: {self.start_date} to {self.end_date}")
        logger.info(f"Walk-forward: {len(folds)} folds, step={self.wf_step}, "
                    f"expanding window from {self.start_date}")
        for i, (fs, fe) in enumerate(folds):
            logger.info(f"  Fold {i+1}: val [{fs.date()} .. {fe.date()})")

        # Save config
        config_to_save = {k: v for k, v in self.config.items()
                         if not isinstance(v, (list,)) or len(v) < 20}
        config_to_save["train_stock_count"] = len(self.train_stocks)
        config_to_save["test_stocks"] = self.test_stocks
        config_to_save["feature_count"] = len(self.feature_cols)
        config_to_save["n_technical_features"] = self.n_technical
        config_to_save["n_sentiment_features"] = self.n_sentiment
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

        # Pre-compute features for all training stocks (done once, reused across folds)
        logger.info("\n" + "=" * 40)
        logger.info("PRE-COMPUTING FEATURES (once for all folds)")
        logger.info("=" * 40)
        prepared = self.prepare_all_stock_features(stock_data, self.train_stocks)

        if not prepared:
            logger.error("No stocks prepared successfully -- aborting")
            return {"error": "No data"}

        # === WALK-FORWARD LOOP ===
        train_start = pd.Timestamp(self.start_date)
        fold_results = []
        all_val_preds = []
        all_val_true = []
        total_train_time = 0.0
        last_model = None
        last_scaler = None
        n_features = None

        for fold_idx, (fold_start, fold_end) in enumerate(folds):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"FOLD {fold_idx+1}/{len(folds)}: "
                       f"Train [{train_start.date()} .. {fold_start.date()}) -> "
                       f"Val [{fold_start.date()} .. {fold_end.date()})")
            logger.info(f"{'=' * 60}")

            # Build train/val arrays for this fold
            X_train, y_train, X_val, y_val, nf = self._build_fold_data(
                prepared, train_start, fold_start, fold_start, fold_end
            )

            if len(X_train) == 0 or len(X_val) == 0:
                logger.warning(f"Fold {fold_idx+1}: Insufficient data "
                             f"(train={len(X_train)}, val={len(X_val)}) -- skipping")
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
            logger.info(f"  Train class dist: {Counter(y_train)}")
            logger.info(f"  Val class dist:   {Counter(y_val)}")

            # Scale (fresh scaler per fold)
            X_train_s, X_val_s, scaler = self._scale_data(X_train, X_val)
            del X_train, X_val
            gc.collect()

            # Create FRESH model — pass feature split sizes for hybrid architecture
            model = create_model(
                self.model_type, n_features,
                n_technical=self.n_technical,
                n_sentiment=self.n_sentiment,
                dropout=self.dropout,
            )

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

            fold_start_time = time.time()
            trainer.train(train_loader, val_loader, epochs=self.epochs, patience=self.patience)
            fold_train_time = time.time() - fold_start_time
            total_train_time += fold_train_time

            # Evaluate on validation set
            val_metrics = evaluate_model_on_data(model, X_val_s, y_val)

            # Collect predictions for aggregate metrics
            if "predictions" in val_metrics:
                all_val_preds.append(val_metrics["predictions"])
                all_val_true.append(y_val)

            fold_result = {
                "fold": fold_idx + 1,
                "val_start": str(fold_start.date()),
                "val_end": str(fold_end.date()),
                "train_samples": int(len(X_train_s)),
                "val_samples": int(len(X_val_s)),
                "train_time": fold_train_time,
                "accuracy": val_metrics.get("accuracy", 0),
                "zero_rule": val_metrics.get("zero_rule", 0),
                "lift": val_metrics.get("lift", 0),
                "mcc": val_metrics.get("mcc", 0),
                "f1_macro": val_metrics.get("f1_macro", 0),
            }
            fold_results.append(fold_result)

            logger.info(f"  -> Acc={fold_result['accuracy']:.2%}, "
                       f"Lift={fold_result['lift']:+.2%}, "
                       f"MCC={fold_result['mcc']:.4f}, "
                       f"Time={fold_train_time:.1f}s")

            # Keep last fold's model and scaler for per-stock evaluation
            last_model = model
            last_scaler = scaler

            # Cleanup
            del X_train_s, X_val_s, train_loader, val_loader, trainer
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
            agg_metrics = compute_metrics(agg_true, agg_preds)
        else:
            agg_metrics = {"accuracy": 0, "zero_rule": 0, "lift": 0,
                          "mcc": 0, "f1_macro": 0, "n_samples": 0}

        logger.info(f"Total val samples: {agg_metrics.get('n_samples', 0):,}")
        logger.info(f"Aggregate Acc={agg_metrics.get('accuracy', 0):.2%}, "
                    f"Lift={agg_metrics.get('lift', 0):+.2%}, "
                    f"MCC={agg_metrics.get('mcc', 0):.4f}, "
                    f"F1_macro={agg_metrics.get('f1_macro', 0):.4f}")

        # Per-fold breakdown
        if valid_folds:
            accs = [r["accuracy"] for r in valid_folds]
            lifts = [r["lift"] for r in valid_folds]
            mccs = [r["mcc"] for r in valid_folds]
            logger.info(f"Per-fold Acc: mean={np.mean(accs):.2%}, "
                       f"std={np.std(accs):.2%}, "
                       f"min={np.min(accs):.2%}, max={np.max(accs):.2%}")
            logger.info(f"Per-fold Lift: mean={np.mean(lifts):+.2%}, "
                       f"std={np.std(lifts):.2%}")

        # === PER-STOCK EVALUATION (using last fold's model) ===
        per_stock_results = []
        if last_model is not None and last_scaler is not None:
            logger.info(f"\n{'=' * 40}")
            logger.info("PER-STOCK EVALUATION (last fold model)")
            logger.info(f"{'=' * 40}")

            for ticker in self.test_stocks:
                result = self.evaluate_single_stock(
                    last_model, stock_data, ticker, last_scaler
                )
                per_stock_results.append(result)
                if "error" not in result:
                    sent_flag = "[S]" if result.get("has_sentiment") else "   "
                    logger.info(f"{sent_flag} {ticker}: Acc={result['accuracy']:.2%}, "
                               f"Lift={result['lift']:+.2%}, "
                               f"MCC={result['mcc']:.4f}, n={result['n_samples']}")
                else:
                    logger.warning(f"{ticker}: {result['error']}")

        # === SAVE RESULTS ===
        valid_ps = [r for r in per_stock_results if "error" not in r]
        avg_acc = np.mean([r["accuracy"] for r in valid_ps]) if valid_ps else 0
        avg_lift = np.mean([r["lift"] for r in valid_ps]) if valid_ps else 0
        avg_mcc = np.mean([r["mcc"] for r in valid_ps]) if valid_ps else 0

        n_params = sum(p.numel() for p in last_model.parameters()) if last_model else 0

        summary = {
            "experiment_name": self.config.get("experiment_name", "hybrid_branch_v4"),
            "task": "classification",
            "target": "3class_trend",
            "model_type": self.model_type,
            "architecture": "dual_branch_hybrid" if self.model_type == "hybrid" else self.model_type,
            "frequency": self.frequency,
            "use_sentiment": self.use_sentiment,
            "n_features": n_features,
            "n_technical_features": self.n_technical,
            "n_sentiment_features": self.n_sentiment,
            "n_params": n_params,
            "train_stocks": len(self.train_stocks),
            "sequence_length": self.sequence_length,
            "threshold_low": self.threshold_low,
            "threshold_high": self.threshold_high,
            "dropout": self.dropout,
            "label_smoothing": self.label_smoothing,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "total_train_time": total_train_time,
            # Walk-forward params
            "wf_step": self.wf_step,
            "wf_val_start": str(self.wf_val_start.date()),
            "wf_val_end": str(self.wf_val_end.date()),
            "n_folds": len(folds),
            "n_valid_folds": len(valid_folds),
            # Aggregated walk-forward metrics (all folds pooled)
            "wf_agg_accuracy": agg_metrics.get("accuracy", 0),
            "wf_agg_zero_rule": agg_metrics.get("zero_rule", 0),
            "wf_agg_lift": agg_metrics.get("lift", 0),
            "wf_agg_mcc": agg_metrics.get("mcc", 0),
            "wf_agg_f1_macro": agg_metrics.get("f1_macro", 0),
            "wf_agg_n_samples": agg_metrics.get("n_samples", 0),
            # Per-fold mean metrics
            "wf_mean_accuracy": float(np.mean(accs)) if valid_folds else 0,
            "wf_std_accuracy": float(np.std(accs)) if valid_folds else 0,
            "wf_mean_lift": float(np.mean(lifts)) if valid_folds else 0,
            "wf_mean_mcc": float(np.mean(mccs)) if valid_folds else 0,
            # Per-stock averages (last fold model)
            "per_stock_avg_accuracy": avg_acc,
            "per_stock_avg_lift": avg_lift,
            "per_stock_avg_mcc": avg_mcc,
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
        logger.info(f"Model: {self.model_type} ({n_params:,} params)")
        logger.info(f"Features: {n_features} ({self.n_technical} technical + "
                    f"{self.n_sentiment} sentiment)")
        if self.model_type == "hybrid":
            logger.info(f"Architecture: Tech LSTM(128x2)+Attn(4h) | Sent GRU(32x1) | Fusion")
        logger.info(f"Walk-forward: {len(valid_folds)}/{len(folds)} folds, "
                    f"total train time: {total_train_time:.1f}s")
        logger.info(f"WF Aggregate: Acc={agg_metrics.get('accuracy', 0):.2%}, "
                    f"Lift={agg_metrics.get('lift', 0):+.2%}, "
                    f"MCC={agg_metrics.get('mcc', 0):.4f}")
        if valid_folds:
            logger.info(f"WF Per-fold:  Acc={np.mean(accs):.2%} +/- {np.std(accs):.2%}")
        logger.info(f"Per-stock:    Acc={avg_acc:.2%}, Lift={avg_lift:+.2%}, MCC={avg_mcc:.4f}")
        logger.info(f"Results saved to: {self.output_dir}")

        return summary


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Hybrid Branch Classification + Sentiment V4")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--stocks", type=int, default=400, help="Number of training stocks (fallback)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--frequency", choices=["daily", "weekly"], default="daily")
    parser.add_argument("--model", choices=["simple", "attention", "hybrid"], default="hybrid")
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--threshold", type=float, nargs=2, default=None,
                        metavar=("LOW", "HIGH"), help="Trend thresholds e.g. -0.005 0.005")
    parser.add_argument("--test-stocks", nargs="+", default=None)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--no-sentiment", action="store_true", help="Disable sentiment features (baseline)")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--name", type=str, default="hybrid_branch_v4", help="Experiment name")
    parser.add_argument("--max-tickers", type=int, default=DEFAULT_MAX_TICKERS,
                        help=f"Max tickers by density rank (default: {DEFAULT_MAX_TICKERS})")
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
        "model_type": args.model,
        "frequency": args.frequency,
        "test_stocks": args.test_stocks or DEFAULT_TEST_STOCKS,
        "epochs": args.epochs,
        "dropout": args.dropout,
        "label_smoothing": args.label_smoothing,
        "learning_rate": args.lr,
        "patience": args.patience,
        "use_sentiment": not args.no_sentiment,
        "max_tickers": args.max_tickers,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "wf_val_start": args.wf_val_start,
        "wf_val_end": args.wf_val_end,
        "wf_step": args.wf_step,
        "stocks": args.stocks,
    }

    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.seq_length:
        config["sequence_length"] = args.seq_length
    if args.threshold:
        config["threshold_low"] = args.threshold[0]
        config["threshold_high"] = args.threshold[1]

    experiment = HybridBranchExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
