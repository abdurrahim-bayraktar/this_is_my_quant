"""
Selected Stock Evaluation Experiment V2 - Weekly Data.

Uses WEEKLY resampled data instead of daily.
Trains on ALL pooled stock data, evaluates on selected individual stocks.

Best config from hyperparameter tuning:
- dropout: 0.2
- scheduler: ReduceLROnPlateau  
- label_smoothing: 0.1

Usage:
    python experiments/selected_stock_evaluation_v2.py --epochs 100 --stocks 400
    python experiments/selected_stock_evaluation_v2.py --test-stocks MSFT AAPL GOOGL
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import pickle
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, matthews_corrcoef
from tqdm import tqdm
from collections import Counter
import time
import gc

from experiments.ranked_tickers import EXTENDED_TICKERS

# SHAP Top 20 features (will be computed on weekly data)
SHAP_TOP20_FEATURES = [
    "atr_pct", "intraday_range", "bb_BBB_5_2.0_2.0", "gap", "cci",
    "high_low_pct", "volume_ratio", "adx_DMN_14", "return_1d", "aroon_AROOND_14",
    "adx_DMP_14", "obv", "ad", "bear_power", "roc_10",
    "pvo_PVOh_12_26_9", "trix_TRIXs_30_9", "rsi_14", "aroon_AROONU_14", "return_5d"
]

# Renamed for weekly context
WEEKLY_FEATURES = [
    "atr_pct", "intraday_range", "bb_BBB_5_2.0_2.0", "gap", "cci",
    "high_low_pct", "volume_ratio", "adx_DMN_14", "return_1w", "aroon_AROOND_14",
    "adx_DMP_14", "obv", "ad", "bear_power", "roc_10",
    "pvo_PVOh_12_26_9", "trix_TRIXs_30_9", "rsi_14", "aroon_AROONU_14", "return_4w"
]

DEFAULT_TEST_STOCKS = [
    "MSFT", "AAPL", "JPM", "JNJ", "XOM", 
    "WMT", "DIS", "BA", "KO", "NVDA",
]

try:
    from config import REPORTS_DIR, DATA_DIR
    from src.models import AttentionLSTM
    from src.features.indicators_v7 import ComprehensiveIndicatorsV7
    RUNNING_LOCAL = True
except ImportError:
    RUNNING_LOCAL = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PriceCache:
    """Cache for yfinance price data."""
    
    def __init__(self, cache_dir: Path = None):
        if RUNNING_LOCAL:
            self.cache_dir = cache_dir or DATA_DIR / "price_cache"
        else:
            self.cache_dir = cache_dir or Path("/content/data/price_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_path(self, start_date: str, end_date: str, weekly: bool = False) -> Path:
        suffix = "_weekly" if weekly else ""
        return self.cache_dir / f"prices_{start_date}_{end_date}{suffix}.pkl"
    
    def load(self, start_date: str, end_date: str, requested_stocks: List[str], weekly: bool = False) -> Dict[str, pd.DataFrame]:
        cache_path = self.get_cache_path(start_date, end_date, weekly)
        if cache_path.exists():
            logger.info(f"Loading cached prices from {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        return None
    
    def save(self, data: Dict[str, pd.DataFrame], start_date: str, end_date: str, weekly: bool = False):
        cache_path = self.get_cache_path(start_date, end_date, weekly)
        if not cache_path.exists():
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)


def resample_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLCV data to weekly."""
    weekly = pd.DataFrame()
    weekly['Open'] = df['Open'].resample('W-FRI').first()
    weekly['High'] = df['High'].resample('W-FRI').max()
    weekly['Low'] = df['Low'].resample('W-FRI').min()
    weekly['Close'] = df['Close'].resample('W-FRI').last()
    weekly['Volume'] = df['Volume'].resample('W-FRI').sum()
    
    if 'Ticker' in df.columns:
        weekly['Ticker'] = df['Ticker'].iloc[0]
    
    return weekly.dropna()


class Trainer:
    """Trainer with ReduceLROnPlateau scheduler."""
    
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
        
        return {"loss": total_loss / total, "accuracy": correct / total}
    
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
        
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            self.model.to(self.device)
        
        return {"best_val_loss": self.best_val_loss}


class SelectedStockEvaluationV2:
    """Train on all stocks (WEEKLY data), evaluate on selected test stocks."""
    
    def __init__(
        self,
        train_stocks: List[str] = None,
        test_stocks: List[str] = None,
        start_date: str = "2010-01-01",  # Earlier start for more weekly data
        end_date: str = "2024-12-31",
        train_end_date: str = "2022-01-01",
        val_end_date: str = "2023-06-01",
        sequence_length: int = 12,  # TODO:change to 12 20 weeks = ~5 months of weekly data
        epochs: int = 100,
        batch_size: int = 512,  # Smaller batch for less data
        dropout: float = 0.2,
        label_smoothing: float = 0.1,
        learning_rate: float = 5e-4,
        use_cache: bool = True,
    ):
        self.train_stocks = train_stocks or EXTENDED_TICKERS[:400]
        self.test_stocks = test_stocks or DEFAULT_TEST_STOCKS
        
        # Ensure test stocks are NOT in train stocks
        self.train_stocks = [t for t in self.train_stocks if t not in self.test_stocks]
        
        self.start_date = start_date
        self.end_date = end_date
        self.train_end_date = pd.Timestamp(train_end_date)
        self.val_end_date = pd.Timestamp(val_end_date)
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.label_smoothing = label_smoothing
        self.learning_rate = learning_rate
        self.use_cache = use_cache
        
        self.output_dir = REPORTS_DIR / f"selected_stock_eval_weekly_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.indicator_computer = ComprehensiveIndicatorsV7()
        self.price_cache = PriceCache()
        self.feature_cols = SHAP_TOP20_FEATURES  # Same features, computed on weekly
        self.scaler = None
    
    def load_all_stocks(self) -> Dict[str, pd.DataFrame]:
        """Load all stocks and resample to weekly."""
        all_tickers = list(set(self.train_stocks + self.test_stocks))
        stock_data = {}
        
        # Try weekly cache first
        if self.use_cache:
            cached = self.price_cache.load(self.start_date, self.end_date, all_tickers, weekly=True)
            if cached:
                for ticker, df in cached.items():
                    if ticker in all_tickers:
                        stock_data[ticker] = df
                logger.info(f"Loaded {len(stock_data)} weekly stocks from cache")
                if len(stock_data) == len(all_tickers):
                    return stock_data
        
        # Try daily cache and resample
        if self.use_cache:
            daily_cached = self.price_cache.load(self.start_date, self.end_date, all_tickers, weekly=False)
            if daily_cached:
                for ticker, df in daily_cached.items():
                    if ticker in all_tickers and ticker not in stock_data:
                        try:
                            weekly_df = resample_to_weekly(df)
                            if len(weekly_df) >= 52:  # At least 1 year of weekly data
                                stock_data[ticker] = weekly_df
                        except:
                            pass
                logger.info(f"Resampled {len(stock_data)} stocks to weekly from daily cache")
        
        missing = [t for t in all_tickers if t not in stock_data]
        if missing:
            logger.info(f"Downloading {len(missing)} stocks...")
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
                            df.columns = [c.capitalize() if isinstance(c, str) else c for c in df.columns]
                            df['Ticker'] = ticker
                            
                            # Resample to weekly
                            weekly_df = resample_to_weekly(df)
                            if len(weekly_df) >= 52:
                                stock_data[ticker] = weekly_df
                        except:
                            pass
                except Exception as e:
                    logger.warning(f"Batch download failed: {e}")
            
            # Save weekly cache
            if self.use_cache and stock_data:
                self.price_cache.save(stock_data, self.start_date, self.end_date, weekly=True)
        
        logger.info(f"Total: {len(stock_data)} stocks with weekly data")
        return stock_data
    
    def prepare_stock_data(self, df: pd.DataFrame) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """Prepare single stock with temporal split (weekly data)."""
        df = self.indicator_computer.compute_shap_top20(df)
        
        available = [c for c in self.feature_cols if c in df.columns]
        if len(available) < 5:
            return tuple(np.array([]) for _ in range(6))
        
        # Weekly return threshold: ~1% for significant move
        df['return_next'] = df['Close'].pct_change().shift(-1)
        df['trend'] = pd.cut(
            df['return_next'],
            bins=[-np.inf, -0.01, 0.01, np.inf],  # ±1% for weekly
            labels=[0, 1, 2]
        ).astype(float)
        
        df = df.dropna(subset=['trend'] + available[:5])
        if len(df) < self.sequence_length + 10:
            return tuple(np.array([]) for _ in range(6))
        
        feature_data = df[available].values
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
    
    def prepare_pooled_training_data(self, stock_data: Dict[str, pd.DataFrame]) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, int
    ]:
        """Prepare pooled training data from train stocks only."""
        all_train_X, all_train_y = [], []
        all_val_X, all_val_y = [], []
        n_features = None
        
        train_stocks_data = {t: stock_data[t] for t in self.train_stocks if t in stock_data}
        
        for ticker, df in tqdm(train_stocks_data.items(), desc="Processing train stocks"):
            try:
                X_tr, y_tr, X_val, y_val, _, _ = self.prepare_stock_data(df)
                
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
                    
            except Exception as e:
                logger.warning(f"Failed {ticker}: {e}")
        
        X_train = np.concatenate(all_train_X, axis=0)
        y_train = np.concatenate(all_train_y, axis=0)
        X_val = np.concatenate(all_val_X, axis=0) if all_val_X else np.array([])
        y_val = np.concatenate(all_val_y, axis=0) if all_val_y else np.array([])
        
        logger.info(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Features: {n_features}")
        logger.info(f"Class distribution (train): {Counter(y_train)}")
        
        return X_train, y_train, X_val, y_val, n_features
    
    def scale_data(self, X_train: np.ndarray, X_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Scale training and validation data, store scaler."""
        n_train, seq_len, n_features = X_train.shape
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train.reshape(-1, n_features))
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_train_scaled = X_train_scaled.reshape(n_train, seq_len, n_features)
        
        if len(X_val) > 0:
            X_val_scaled = self.scaler.transform(X_val.reshape(-1, n_features))
            X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            X_val_scaled = X_val_scaled.reshape(X_val.shape)
        else:
            X_val_scaled = X_val
        
        return X_train_scaled, X_val_scaled
    
    def evaluate_single_stock(self, model: nn.Module, stock_data: Dict[str, pd.DataFrame], ticker: str) -> Dict:
        """Evaluate model on a single test stock."""
        if ticker not in stock_data:
            return {"ticker": ticker, "error": "Not found"}
        
        df = stock_data[ticker]
        _, _, _, _, X_test, y_test = self.prepare_stock_data(df)
        
        if len(X_test) == 0:
            return {"ticker": ticker, "error": "No test data"}
        
        # Scale using training scaler
        n_test, seq_len, n_features = X_test.shape
        X_test_scaled = self.scaler.transform(X_test.reshape(-1, n_features))
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_scaled = X_test_scaled.reshape(n_test, seq_len, n_features)
        
        # Predict
        model.eval()
        device = next(model.parameters()).device
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test_scaled).to(device)
            predictions = model.predict(X_tensor)
            pred_classes = predictions["trend_class"].cpu().numpy()
        
        # Metrics
        accuracy = (pred_classes == y_test).mean()
        class_counts = Counter(y_test)
        most_common = class_counts.most_common(1)[0][1]
        zero_rule = most_common / len(y_test)
        
        f1_macro = f1_score(y_test, pred_classes, average='macro')
        mcc = matthews_corrcoef(y_test, pred_classes)
        
        return {
            "ticker": ticker,
            "n_samples": len(y_test),
            "accuracy": accuracy,
            "zero_rule": zero_rule,
            "lift": accuracy - zero_rule,
            "f1_macro": f1_macro,
            "mcc": mcc,
        }
    
    def prepare_pooled_test_data(self, stock_data: Dict[str, pd.DataFrame], n_features: int) -> Tuple[
        np.ndarray, np.ndarray
    ]:
        """Prepare pooled test data from ALL train stocks (not just selected)."""
        all_test_X, all_test_y = [], []
        
        train_stocks_data = {t: stock_data[t] for t in self.train_stocks if t in stock_data}
        
        for ticker, df in tqdm(train_stocks_data.items(), desc="Preparing pooled test"):
            try:
                _, _, _, _, X_te, y_te = self.prepare_stock_data(df)
                
                if len(X_te) == 0:
                    continue
                
                if X_te.shape[-1] != n_features:
                    continue
                
                all_test_X.append(X_te)
                all_test_y.append(y_te)
                    
            except:
                pass
        
        if not all_test_X:
            return np.array([]), np.array([])
            
        X_test = np.concatenate(all_test_X, axis=0)
        y_test = np.concatenate(all_test_y, axis=0)
        
        return X_test, y_test
    
    def evaluate_pooled_test(self, model: nn.Module, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate on pooled test set."""
        if len(X_test) == 0:
            return {"error": "No pooled test data"}
        
        # Scale using training scaler
        n_test, seq_len, n_features = X_test.shape
        X_test_scaled = self.scaler.transform(X_test.reshape(-1, n_features))
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_scaled = X_test_scaled.reshape(n_test, seq_len, n_features)
        
        # Predict in batches
        model.eval()
        device = next(model.parameters()).device
        all_preds = []
        
        for i in range(0, len(X_test_scaled), 2048):
            batch = X_test_scaled[i:i + 2048]
            with torch.no_grad():
                X_tensor = torch.FloatTensor(batch).to(device)
                predictions = model.predict(X_tensor)
                all_preds.append(predictions["trend_class"].cpu().numpy())
        
        pred_classes = np.concatenate(all_preds)
        
        accuracy = (pred_classes == y_test).mean()
        class_counts = Counter(y_test)
        most_common = class_counts.most_common(1)[0][1]
        zero_rule = most_common / len(y_test)
        
        f1_macro = f1_score(y_test, pred_classes, average='macro')
        mcc = matthews_corrcoef(y_test, pred_classes)
        
        return {
            "n_samples": len(y_test),
            "accuracy": accuracy,
            "zero_rule": zero_rule,
            "lift": accuracy - zero_rule,
            "f1_macro": f1_macro,
            "mcc": mcc,
        }
    
    def run(self):
        logger.info("=" * 80)
        logger.info("SELECTED STOCK EVALUATION V2 - WEEKLY DATA")
        logger.info("=" * 80)
        logger.info(f"Train stocks: {len(self.train_stocks)}")
        logger.info(f"Test stocks: {self.test_stocks}")
        logger.info(f"Sequence length: {self.sequence_length} weeks (~{self.sequence_length/4:.1f} months)")
        logger.info(f"Config: dropout={self.dropout}, label_smoothing={self.label_smoothing}")
        
        # Load data
        stock_data = self.load_all_stocks()
        
        # Prepare training data
        X_train, y_train, X_val, y_val, n_features = self.prepare_pooled_training_data(stock_data)
        
        # Scale
        X_train_scaled, X_val_scaled = self.scale_data(X_train, X_val)
        
        # Create model
        model = AttentionLSTM(
            input_size=n_features,
            hidden_size=64,
            num_layers=1,
            dropout=self.dropout,
        )
        
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {n_params:,}")
        
        # Create trainer
        trainer = Trainer(
            model,
            learning_rate=self.learning_rate,
            label_smoothing=self.label_smoothing,
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
        logger.info("\n" + "=" * 40)
        logger.info("TRAINING (WEEKLY DATA)")
        logger.info("=" * 40)
        start_time = time.time()
        trainer.train(train_loader, val_loader, epochs=self.epochs, patience=15)
        train_time = time.time() - start_time
        logger.info(f"Training time: {train_time:.1f}s")
        
        # === POOLED TEST SET EVALUATION ===
        logger.info("\n" + "=" * 40)
        logger.info("POOLED TEST SET EVALUATION")
        logger.info("=" * 40)
        
        X_pooled_test, y_pooled_test = self.prepare_pooled_test_data(stock_data, n_features)
        pooled_metrics = self.evaluate_pooled_test(model, X_pooled_test, y_pooled_test)
        
        if "error" not in pooled_metrics:
            logger.info(f"Pooled Test Set: n={pooled_metrics['n_samples']:,}")
            logger.info(f"  Accuracy: {pooled_metrics['accuracy']:.2%}")
            logger.info(f"  Zero-Rule: {pooled_metrics['zero_rule']:.2%}")
            logger.info(f"  Lift: {pooled_metrics['lift']:+.2%}")
            logger.info(f"  MCC: {pooled_metrics['mcc']:.4f}")
        
        # === PER-STOCK EVALUATION ===
        logger.info("\n" + "=" * 40)
        logger.info("SELECTED STOCKS EVALUATION (WEEKLY)")
        logger.info("=" * 40)
        
        results = []
        for ticker in self.test_stocks:
            result = self.evaluate_single_stock(model, stock_data, ticker)
            results.append(result)
            
            if "error" not in result:
                logger.info(
                    f"{ticker}: Acc={result['accuracy']:.2%}, "
                    f"Lift={result['lift']:+.2%}, "
                    f"MCC={result['mcc']:.4f}, "
                    f"n={result['n_samples']} weeks"
                )
            else:
                logger.warning(f"{ticker}: {result['error']}")
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_dir / "per_stock_results.csv", index=False)
        
        # === COMPARISON SUMMARY ===
        valid_results = [r for r in results if "error" not in r]
        if valid_results:
            avg_acc = np.mean([r["accuracy"] for r in valid_results])
            avg_lift = np.mean([r["lift"] for r in valid_results])
            avg_mcc = np.mean([r["mcc"] for r in valid_results])
            
            logger.info("\n" + "=" * 60)
            logger.info("COMPARISON: POOLED TEST vs SELECTED STOCKS")
            logger.info("=" * 60)
            logger.info(f"{'Metric':<20} {'Pooled Test':>15} {'Selected Stocks':>18} {'Difference':>12}")
            logger.info("-" * 65)
            
            if "error" not in pooled_metrics:
                acc_diff = avg_acc - pooled_metrics['accuracy']
                lift_diff = avg_lift - pooled_metrics['lift']
                mcc_diff = avg_mcc - pooled_metrics['mcc']
                
                logger.info(f"{'Accuracy':<20} {pooled_metrics['accuracy']:>14.2%} {avg_acc:>17.2%} {acc_diff:>+11.2%}")
                logger.info(f"{'Lift vs Zero-Rule':<20} {pooled_metrics['lift']:>+14.2%} {avg_lift:>+17.2%} {lift_diff:>+11.2%}")
                logger.info(f"{'MCC':<20} {pooled_metrics['mcc']:>14.4f} {avg_mcc:>17.4f} {mcc_diff:>+11.4f}")
                logger.info(f"{'Samples':<20} {pooled_metrics['n_samples']:>14,} {sum(r['n_samples'] for r in valid_results):>17,}")
            
            logger.info("\n" + "=" * 40)
            logger.info("SUMMARY")
            logger.info("=" * 40)
            logger.info(f"Selected stocks evaluated: {len(valid_results)}/{len(self.test_stocks)}")
            logger.info(f"Selected Avg Accuracy: {avg_acc:.2%}")
            logger.info(f"Selected Avg Lift: {avg_lift:+.2%}")
            logger.info(f"Selected Avg MCC: {avg_mcc:.4f}")
            
            summary = {
                "frequency": "weekly",
                "train_stocks": len(self.train_stocks),
                "test_stocks_count": len(valid_results),
                "sequence_length_weeks": self.sequence_length,
                # Pooled test metrics
                "pooled_test_n": pooled_metrics.get('n_samples', 0),
                "pooled_test_accuracy": pooled_metrics.get('accuracy', 0),
                "pooled_test_lift": pooled_metrics.get('lift', 0),
                "pooled_test_mcc": pooled_metrics.get('mcc', 0),
                # Selected stocks metrics
                "selected_avg_accuracy": avg_acc,
                "selected_avg_lift": avg_lift,
                "selected_avg_mcc": avg_mcc,
                # Difference
                "accuracy_diff": avg_acc - pooled_metrics.get('accuracy', 0),
                "train_time": train_time,
                "dropout": self.dropout,
                "label_smoothing": self.label_smoothing,
            }
            pd.DataFrame([summary]).to_csv(self.output_dir / "summary.csv", index=False)
        
        # Save model
        torch.save(model.state_dict(), self.output_dir / "model.pt")
        logger.info(f"\nResults saved to: {self.output_dir}")
        
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--stocks", type=int, default=400, help="Number of training stocks")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seq-length", type=int, default=12, help="Sequence length in weeks")
    parser.add_argument("--test-stocks", nargs="+", default=None, help="Test stock tickers")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    
    train_stocks = EXTENDED_TICKERS[:args.stocks]
    test_stocks = args.test_stocks or DEFAULT_TEST_STOCKS
    
    experiment = SelectedStockEvaluationV2(
        train_stocks=train_stocks,
        test_stocks=test_stocks,
        sequence_length=args.seq_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dropout=0.2,
        label_smoothing=0.1,
        use_cache=not args.no_cache,
    )
    
    experiment.run()


if __name__ == "__main__":
    main()
