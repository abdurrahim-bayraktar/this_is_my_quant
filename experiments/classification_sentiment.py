"""
Classification with Sentiment — Trend Classification + FNSPID News.

Extends ultimate_model_v2 by adding 7 domain-driven sentiment features from the
FNSPID dataset alongside the SHAP Top 20 technical indicators.

Key design decisions:
  - Training universe restricted to tickers with sentiment coverage
  - Date range: 2012-2019 (aligned with dense FNSPID sentiment era)
  - Temporal splits: train <2017-01, val <2018-01, test >=2018-01
  - 7 sentiment features engineered with domain knowledge:
    sent_mean, sent_std, sent_count_log, sent_momentum_3d, sent_ema_5d,
    sent_surprise, sent_strength
  - No-news days filled via exponential decay (sentiment fades, doesn't vanish)
  - Classification target: 3-class trend (down/neutral/up) via return thresholds

Usage:
    python experiments/classification_sentiment.py --epochs 100
    python experiments/classification_sentiment.py --epochs 2 --name smoke_test  # smoke test
    python experiments/classification_sentiment.py --no-sentiment --name baseline  # A/B comparison
    python experiments/classification_sentiment.py --model attention --name attn_sent
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
from src.data.utils import resample_to_weekly, load_stocks, create_sequences
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

SENTIMENT_FEATURES = [
    "sent_mean",           # Daily avg sentiment (positive - negative)
    "sent_std",            # Intraday sentiment dispersion (analyst disagreement)
    "sent_count_log",      # log1p(daily_news_count) — attention signal
    "sent_momentum_3d",    # 3-day change in sent_mean — drift/acceleration
    "sent_ema_5d",         # 5-day EMA of sent_mean — persistent sentiment trend
    "sent_surprise",       # sent_mean minus 20-day rolling mean — deviations from baseline
    "sent_strength",       # sent_mean × sent_count_log — high-conviction signal
]

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

    Returns:
        {ticker: DataFrame} where DataFrame is date-indexed with
        columns: sent_mean, sent_std, sent_count_log, sent_momentum_3d,
        sent_ema_5d, sent_surprise, sent_strength
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

    logger.info(f"Loading sentiment from {len(sentiment_files)} ticker files...")
    result = {}

    for fpath in sentiment_files:
        ticker = fpath.stem.replace("_sentiment", "")

        try:
            raw = pd.read_parquet(fpath)

            # Normalize date column
            raw['date'] = pd.to_datetime(raw['date'], utc=True).dt.tz_localize(None)
            raw['trading_date'] = raw['date'].dt.normalize()  # Strip time component

            # === Aggregate to daily level per ticker ===
            daily = raw.groupby('trading_date').agg(
                sent_mean=('sentiment_value', 'mean'),
                sent_std=('sentiment_value', 'std'),
                sent_count=('sentiment_value', 'count'),
            ).sort_index()

            # Fill single-article days (std = NaN) with 0
            daily['sent_std'] = daily['sent_std'].fillna(0)

            # === Derived features (domain knowledge) ===

            # Log-scaled news count — tames outliers, preserves ordinality
            daily['sent_count_log'] = np.log1p(daily['sent_count'])

            # Sentiment momentum: 3-day diff — captures sentiment acceleration
            daily['sent_momentum_3d'] = daily['sent_mean'].diff(3)

            # 5-day EMA: persistent sentiment regime (sticky)
            daily['sent_ema_5d'] = daily['sent_mean'].ewm(span=5, min_periods=1).mean()

            # Sentiment surprise: deviation from 20-day rolling mean
            # High surprise = abnormal sentiment → stronger signal
            rolling_mean_20 = daily['sent_mean'].rolling(20, min_periods=5).mean()
            daily['sent_surprise'] = daily['sent_mean'] - rolling_mean_20

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

            # Keep only the feature columns
            daily = daily[SENTIMENT_FEATURES]

            result[ticker] = daily

        except Exception as e:
            logger.warning(f"Failed to load sentiment for {ticker}: {e}")

    logger.info(f"Loaded sentiment for {len(result)} tickers")

    # Log date coverage
    if result:
        all_dates = pd.concat([df.index.to_series() for df in result.values()])
        logger.info(f"Sentiment date range: {all_dates.min().date()} to {all_dates.max().date()}")

    return result


def get_tickers_with_sentiment(sentiment_dir: Path = None) -> List[str]:
    """Return list of tickers that have sentiment parquet files."""
    if sentiment_dir is None:
        sentiment_dir = CACHE_DIR / "sentiment"

    if not sentiment_dir.exists():
        return []

    return sorted(
        f.stem.replace("_sentiment", "")
        for f in sentiment_dir.glob("*_sentiment.parquet")
    )


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
# EXPERIMENT
# ============================================================================

class ClassificationSentimentExperiment:
    """
    Classification experiment with FNSPID sentiment features.

    Pipeline:
    1. Load sentiment data from cached parquets
    2. Load price data (only for tickers with sentiment)
    3. Compute SHAP Top 20 features + 7 sentiment features (optional)
    4. Create classification target: 3-class trend (down/neutral/up)
    5. Pool all stocks, temporal split (within sentiment-covered period)
    6. Train SimpleLSTM or AttentionLSTM with CrossEntropyLoss
    7. Evaluate: pooled test + per-stock test
    8. Save config.json + summary.csv + per_stock_results.csv + model.pt
    """

    def __init__(self, config: Dict):
        self.config = config
        self.use_sentiment = config.get("use_sentiment", True)

        # Temporal splits (aligned with the dense FNSPID sentiment era: 2012-2019)
        self.start_date = config.get("start_date", "2012-01-01")
        self.end_date = config.get("end_date", "2019-12-31")
        self.train_end = pd.Timestamp(config.get("train_end", "2017-01-01"))
        self.val_end = pd.Timestamp(config.get("val_end", "2018-01-01"))

        # Always determine available tickers based on sentiment coverage
        # to ensure fair comparison between sentiment and no-sentiment runs.
        if self.use_sentiment:
            self.sentiment_data = load_sentiment_data()
            available_sent_tickers = list(self.sentiment_data.keys())
        else:
            self.sentiment_data = {}
            available_sent_tickers = get_tickers_with_sentiment()

        if not available_sent_tickers:
            logger.warning("No sentiment tickers found, falling back to EXTENDED_TICKERS")
            available_sent_tickers = EXTENDED_TICKERS[:config.get("stocks", 400)]

        logger.info(f"Tickers with sentiment available: {len(available_sent_tickers)}")

        # Test stocks: user-specified or default (only those with sentiment)
        self.test_stocks = config.get("test_stocks", DEFAULT_TEST_STOCKS)

        # Filter test stocks to those with sentiment
        self.test_stocks = [t for t in self.test_stocks if t in available_sent_tickers]
        if not self.test_stocks and available_sent_tickers:
            # Fallback: use last 3 sentiment tickers as test
            self.test_stocks = available_sent_tickers[-3:]
            logger.warning(f"No test stocks have sentiment, using: {self.test_stocks}")

        # Train stocks: sentiment tickers minus test stocks
        self.train_stocks = [t for t in available_sent_tickers if t not in self.test_stocks]

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

        # Feature columns
        self.technical_cols = config.get("features", SHAP_TOP20_FEATURES)
        if self.use_sentiment:
            self.feature_cols = self.technical_cols + SENTIMENT_FEATURES
        else:
            self.feature_cols = list(self.technical_cols)

        # Output directory
        exp_name = config.get("experiment_name", "classification_sentiment")
        self.output_dir = REPORTS_DIR / f"{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.indicator_computer = ComprehensiveIndicatorsV7()
        self.cache = PriceCache()
        self.scaler = None

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

    def prepare_stock(self, df: pd.DataFrame, ticker: str = "") -> Tuple:
        """Compute features, create trend label, and build sequences."""
        df = self.indicator_computer.compute_shap_top20(df)

        # Merge sentiment
        df = self._merge_sentiment(df, ticker)

        available = [c for c in self.feature_cols if c in df.columns]
        if len(available) < 5:
            return tuple(np.array([]) for _ in range(6))

        # Classification target: 3-class trend (down/neutral/up)
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
            self.sequence_length, self.train_end, self.val_end,
        )

    def prepare_pooled_data(self, stock_data: Dict, tickers: List[str]) -> Tuple:
        """Prepare pooled train/val/test arrays from multiple stocks."""
        all_train_X, all_train_y = [], []
        all_val_X, all_val_y = [], []
        all_test_X, all_test_y = [], []
        n_features = None

        stocks = {t: stock_data[t] for t in tickers if t in stock_data}

        for ticker, df in tqdm(stocks.items(), desc="Processing stocks"):
            try:
                X_tr, y_tr, X_val, y_val, X_te, y_te = self.prepare_stock(df, ticker)

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
        """Fit scaler on train, transform all splits."""
        n_train, seq_len, n_features = X_train.shape

        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train.reshape(-1, n_features))
        X_train_s = np.nan_to_num(X_train_s, nan=0.0, posinf=0.0, neginf=0.0)
        X_train_s = X_train_s.reshape(n_train, seq_len, n_features)

        def _transform(X):
            if len(X) == 0:
                return X
            s = X.shape
            X_t = self.scaler.transform(X.reshape(-1, n_features))
            return np.nan_to_num(X_t, nan=0.0, posinf=0.0, neginf=0.0).reshape(s)

        return X_train_s, _transform(X_val), _transform(X_test)

    def evaluate_single_stock(self, model, stock_data, ticker) -> Dict:
        """Evaluate model on a single held-out stock's test data."""
        if ticker not in stock_data:
            return {"ticker": ticker, "error": "Not found"}

        _, _, _, _, X_test, y_test = self.prepare_stock(stock_data[ticker], ticker)

        if len(X_test) == 0:
            return {"ticker": ticker, "error": "No test data"}

        # Scale with training scaler
        n, seq, feat = X_test.shape
        X_scaled = self.scaler.transform(X_test.reshape(-1, feat))
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0).reshape(n, seq, feat)

        metrics = evaluate_model_on_data(model, X_scaled, y_test)
        metrics["ticker"] = ticker
        metrics["has_sentiment"] = ticker in self.sentiment_data
        metrics.pop("predictions", None)
        metrics.pop("class_distribution", None)
        return metrics

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
            df = self.indicator_computer.compute_shap_top20(df)
            df = self._merge_sentiment(df, ticker)
            sent_cols = [c for c in SENTIMENT_FEATURES if c in df.columns]
            if sent_cols:
                nonzero_pct = (df[sent_cols] != 0).any(axis=1).mean()
                logger.info(f"  {ticker}: {nonzero_pct:.1%} of rows have non-zero sentiment")

    def run(self):
        """Execute the full classification+sentiment experiment pipeline."""
        logger.info("=" * 80)
        logger.info("CLASSIFICATION + SENTIMENT EXPERIMENT")
        logger.info("=" * 80)
        logger.info(f"Sentiment: {'ENABLED' if self.use_sentiment else 'DISABLED'}")
        logger.info(f"Model: {self.model_type}, Frequency: {self.frequency}")
        logger.info(f"Train stocks: {len(self.train_stocks)}, Test stocks: {self.test_stocks}")
        logger.info(f"Thresholds: [{self.threshold_low}, {self.threshold_high}]")
        logger.info(f"Features: {len(self.feature_cols)} ({len(self.technical_cols)} technical + "
                    f"{len(SENTIMENT_FEATURES) if self.use_sentiment else 0} sentiment)")
        logger.info(f"Date range: {self.start_date} to {self.end_date}")
        logger.info(f"Splits: train<{self.train_end.date()}, val<{self.val_end.date()}, test>=")

        # Save config
        config_to_save = {k: v for k, v in self.config.items()
                         if not isinstance(v, (list,)) or len(v) < 20}
        config_to_save["train_stock_count"] = len(self.train_stocks)
        config_to_save["test_stocks"] = self.test_stocks
        config_to_save["feature_count"] = len(self.feature_cols)
        config_to_save["sentiment_features"] = SENTIMENT_FEATURES if self.use_sentiment else []
        config_to_save["train_stocks_list"] = self.train_stocks
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(config_to_save, f, indent=2, default=str)

        # Load data
        stock_data = self.load_data()

        # Sentiment diagnostics
        self._log_sentiment_diagnostics(stock_data)

        # Prepare pooled training data (from train stocks only)
        X_train, y_train, X_val, y_val, X_test, y_test, n_features = \
            self.prepare_pooled_data(stock_data, self.train_stocks)

        # Scale
        X_train_s, X_val_s, X_test_s = self.scale_data(X_train, X_val, X_test)
        del X_train, X_val, X_test
        gc.collect()

        # Create model
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

        logger.info("\n" + "=" * 40)
        logger.info("TRAINING")
        logger.info("=" * 40)
        start_time = time.time()
        trainer.train(train_loader, val_loader, epochs=self.epochs, patience=self.patience)
        train_time = time.time() - start_time
        logger.info(f"Training time: {train_time:.1f}s")

        # === POOLED TEST EVALUATION ===
        logger.info("\n" + "=" * 40)
        logger.info("POOLED TEST EVALUATION")
        logger.info("=" * 40)

        pooled_metrics = evaluate_model_on_data(model, X_test_s, y_test)
        pooled_metrics.pop("predictions", None)
        pooled_metrics.pop("class_distribution", None)

        if "error" not in pooled_metrics:
            logger.info(f"Pooled: n={pooled_metrics['n_samples']:,}, "
                       f"Acc={pooled_metrics['accuracy']:.2%}, "
                       f"Lift={pooled_metrics['lift']:+.2%}, "
                       f"MCC={pooled_metrics['mcc']:.4f}")

        # === PER-STOCK EVALUATION ===
        logger.info("\n" + "=" * 40)
        logger.info("PER-STOCK EVALUATION")
        logger.info("=" * 40)

        per_stock_results = []
        for ticker in self.test_stocks:
            result = self.evaluate_single_stock(model, stock_data, ticker)
            per_stock_results.append(result)
            if "error" not in result:
                sent_flag = "📰" if result.get("has_sentiment") else "  "
                logger.info(f"{sent_flag} {ticker}: Acc={result['accuracy']:.2%}, "
                           f"Lift={result['lift']:+.2%}, "
                           f"MCC={result['mcc']:.4f}, n={result['n_samples']}")
            else:
                logger.warning(f"{ticker}: {result['error']}")

        # === SAVE RESULTS ===
        valid = [r for r in per_stock_results if "error" not in r]
        avg_acc = np.mean([r["accuracy"] for r in valid]) if valid else 0
        avg_lift = np.mean([r["lift"] for r in valid]) if valid else 0
        avg_mcc = np.mean([r["mcc"] for r in valid]) if valid else 0

        summary = {
            "experiment_name": self.config.get("experiment_name", "classification_sentiment"),
            "task": "classification",
            "target": "3class_trend",
            "model_type": self.model_type,
            "frequency": self.frequency,
            "use_sentiment": self.use_sentiment,
            "n_features": n_features,
            "n_technical_features": len(self.technical_cols),
            "n_sentiment_features": len(SENTIMENT_FEATURES) if self.use_sentiment else 0,
            "n_params": n_params,
            "train_stocks": len(self.train_stocks),
            "sequence_length": self.sequence_length,
            "threshold_low": self.threshold_low,
            "threshold_high": self.threshold_high,
            "dropout": self.dropout,
            "label_smoothing": self.label_smoothing,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "train_time": train_time,
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

        pd.DataFrame([summary]).to_csv(self.output_dir / "summary.csv", index=False)
        pd.DataFrame(per_stock_results).to_csv(self.output_dir / "per_stock_results.csv", index=False)
        torch.save(model.state_dict(), self.output_dir / "model.pt")

        logger.info(f"\n{'=' * 60}")
        logger.info("SUMMARY")
        logger.info(f"{'=' * 60}")
        logger.info(f"Model: {self.model_type} ({n_params:,} params)")
        logger.info(f"Features: {n_features} ({len(self.technical_cols)} technical + "
                    f"{n_features - len(self.technical_cols)} sentiment)")
        logger.info(f"Pooled:    Acc={pooled_metrics.get('accuracy', 0):.2%}, "
                    f"Lift={pooled_metrics.get('lift', 0):+.2%}, "
                    f"MCC={pooled_metrics.get('mcc', 0):.4f}")
        logger.info(f"Per-stock: Acc={avg_acc:.2%}, Lift={avg_lift:+.2%}, MCC={avg_mcc:.4f}")
        logger.info(f"Results saved to: {self.output_dir}")

        return summary


def main():
    parser = argparse.ArgumentParser(description="Classification + Sentiment Experiment")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--stocks", type=int, default=400, help="Number of training stocks (fallback)")
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
    parser.add_argument("--no-sentiment", action="store_true", help="Disable sentiment features (baseline)")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--name", type=str, default="classification_sentiment", help="Experiment name")
    # Temporal split overrides
    parser.add_argument("--start-date", type=str, default="2012-01-01")
    parser.add_argument("--end-date", type=str, default="2019-12-31")
    parser.add_argument("--train-end", type=str, default="2017-01-01")
    parser.add_argument("--val-end", type=str, default="2018-01-01")
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
        "start_date": args.start_date,
        "end_date": args.end_date,
        "train_end": args.train_end,
        "val_end": args.val_end,
        "stocks": args.stocks,
    }

    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.seq_length:
        config["sequence_length"] = args.seq_length
    if args.threshold:
        config["threshold_low"] = args.threshold[0]
        config["threshold_high"] = args.threshold[1]

    experiment = ClassificationSentimentExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
