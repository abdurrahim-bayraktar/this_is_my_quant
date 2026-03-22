"""
LSTM Size Comparison Experiment - Comparing LSTM models of different sizes.

Based on pooled_price_baseline_v10.py with all critical fixes preserved:
- Temporal split by date (no data leakage)
- Excluded ICS_26 and DPO features (data leakage prevention)
- Correct T+1 prediction target alignment

This experiment compares:
- Large (v10 default): hidden_size=128, num_layers=2
- Medium-2L: hidden_size=64, num_layers=2
- Medium-1L: hidden_size=64, num_layers=1
- Small: hidden_size=32, num_layers=1

Usage:
    python experiments/lstm_size_comparison.py --epochs 100 --stocks 400
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, f1_score, matthews_corrcoef
from tqdm import tqdm
from collections import Counter

# Import tickers from ranked_tickers.py
from experiments.ranked_tickers import EXTENDED_TICKERS

# Features to exclude (suspected data leaks from v10)
EXCLUDED_FEATURES = [
    'ichimoku_ICS_26',  # Chikou Span - plots close 26 periods back, possible look-ahead
    'dpo',              # Detrended Price Oscillator - suspected leak
]

try:
    from config import REPORTS_DIR, DATA_DIR, training_config
    from src.models import AttentionLSTM
    from src.training import Trainer
    from src.features.indicators_v7 import ComprehensiveIndicatorsV7
    RUNNING_LOCAL = True
except ImportError:
    RUNNING_LOCAL = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logger.info(f"Python Version: {sys.version}")
logger.info(f"PyTorch Version: {torch.__version__}")
if torch.cuda.is_available():
    logger.info(f"CUDA Available: True, Device: {torch.cuda.get_device_name(0)}")


# =============================================================================
# Model Configurations to Compare
# =============================================================================

MODEL_CONFIGS = {
    "large_2l": {
        "hidden_size": 128,
        "num_layers": 2,
        "description": "Large (v10 default): 128 hidden, 2 layers"
    },
    "medium_2l": {
        "hidden_size": 64,
        "num_layers": 2,
        "description": "Medium-2L: 64 hidden, 2 layers"
    },
    "medium_1l": {
        "hidden_size": 64,
        "num_layers": 1,
        "description": "Medium-1L: 64 hidden, 1 layer"
    },
    "small_1l": {
        "hidden_size": 32,
        "num_layers": 1,
        "description": "Small: 32 hidden, 1 layer"
    },
}


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
            logger.info(f"Cache contains {len(cached)} stocks")
            return cached
        return None
    
    def save(self, data: Dict[str, pd.DataFrame], start_date: str, end_date: str):
        cache_path = self.get_cache_path(start_date, end_date)
        if not cache_path.exists():
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)


class LSTMSizeComparisonExperiment:
    """
    Experiment to compare LSTM models of different sizes.
    
    Preserves all V10 fixes:
    - Temporal split by date
    - Exclude ICS_26 and DPO (data leak prevention)
    - Correct T+1 prediction target
    - Scaler fit on train only
    """
    
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
        dropout: float = 0.4,
        learning_rate: float = 5e-4,
        use_cache: bool = True,
        colab_mode: bool = False,
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
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.use_cache = use_cache
        self.colab_mode = colab_mode
        
        if colab_mode:
            self.output_dir = Path(f"/content/reports/lstm_size_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        else:
            self.output_dir = REPORTS_DIR / f"lstm_size_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.indicator_computer = ComprehensiveIndicatorsV7()
        self.price_cache = PriceCache()
        
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
        logger.info(f"Need to download {len(missing)} additional stocks")
        
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
            
            if self.use_cache and len(stock_data) > 0:
                self.price_cache.save(stock_data, self.start_date, self.end_date)
        
        logger.info(f"Total: {len(stock_data)} stocks ready")
        return stock_data
    
    def prepare_stock_with_temporal_split(self, df: pd.DataFrame, target_feature_cols: List[str] = None) -> Tuple[
        np.ndarray, np.ndarray,  # X_train, y_train
        np.ndarray, np.ndarray,  # X_val, y_val
        np.ndarray, np.ndarray,  # X_test, y_test
        List[str]                # feature_cols
    ]:
        """
        THE KEY FIX: Split THIS stock by DATE, then create sequences.
        Returns train/val/test already split for this stock.
        """
        df = self.indicator_computer.compute_all(df)
        
        if target_feature_cols:
            feature_cols = target_feature_cols
            # Ensure all required columns exist, fill with 0 if missing
            for col in feature_cols:
                if col not in df.columns:
                    df[col] = 0.0
        else:
            feature_cols = self.indicator_computer.get_indicator_columns(df)
            
            # V10: Remove suspected leak features
            feature_cols = [col for col in feature_cols if col not in EXCLUDED_FEATURES]
            logger.debug(f"Excluded features: {EXCLUDED_FEATURES}")
            
            # Filter valid columns
            valid_cols = [col for col in feature_cols 
                          if col in df.columns and df[col].notna().sum() > 10]
            feature_cols = valid_cols
        
        if len(feature_cols) < 5:
            return (np.array([]), np.array([]), np.array([]), 
                    np.array([]), np.array([]), np.array([]), [])
        
        # Create target - FIX: shift(-1) for T+1 prediction
        df['return_next'] = df['Close'].pct_change().shift(-1)
        df['trend'] = pd.cut(
            df['return_next'],
            bins=[-np.inf, -0.005, 0.005, np.inf],
            labels=[0, 1, 2]
        ).astype(float)
        
        df = df.dropna(subset=['trend'] + feature_cols[:10])
        if len(df) < self.sequence_length + 10:
            return (np.array([]), np.array([]), np.array([]), 
                    np.array([]), np.array([]), np.array([]), [])
        
        # Get feature data
        feature_data = df[feature_cols].values
        feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
        labels = df['trend'].values
        
        # Get the index (dates) for temporal splitting
        dates = df.index
        
        # Create sequences with date tracking
        X_train, y_train = [], []
        X_val, y_val = [], []
        X_test, y_test = [], []
        
        for i in range(len(feature_data) - self.sequence_length):
            seq_end_date = dates[i + self.sequence_length - 1]  # Last day of sequence
            target_date = dates[i + self.sequence_length]       # Target day
            
            seq = feature_data[i:i + self.sequence_length]
            # FIX: Predict T+1, not T+2 - use labels[i + self.sequence_length - 1]
            label = labels[i + self.sequence_length - 1]
            
            # Split by TARGET date (the day we're predicting)
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
            feature_cols
        )
    
    def prepare_temporal_pooled_data(self, stock_data: Dict[str, pd.DataFrame]) -> Tuple[
        np.ndarray, np.ndarray,  # X_train, y_train
        np.ndarray, np.ndarray,  # X_val, y_val
        np.ndarray, np.ndarray,  # X_test, y_test
        List[str]                # feature_cols
    ]:
        """
        Process all stocks, split each by date, then pool train/val/test separately.
        """
        import gc
        
        all_train_X, all_train_y = [], []
        all_val_X, all_val_y = [], []
        all_test_X, all_test_y = [], []
        feature_cols = None
        success_count = 0
        
        for ticker, df in tqdm(stock_data.items(), desc="Processing stocks with temporal split"):
            try:
                # Pass feature_cols if established, otherwise let the function determine them
                X_tr, y_tr, X_val, y_val, X_te, y_te, cols = self.prepare_stock_with_temporal_split(df, target_feature_cols=feature_cols)
                
                if len(X_tr) == 0 and len(X_val) == 0 and len(X_te) == 0:
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
                
                # Establish the feature columns from the first successful stock
                if feature_cols is None:
                    feature_cols = cols
                    logger.info(f"Established {len(feature_cols)} feature columns from {ticker}")
                    
            except Exception as e:
                if success_count < 3:
                    logger.error(f"Failed to process {ticker}: {e}")
        
        logger.info(f"Successfully processed: {success_count}/{len(stock_data)} stocks")
        
        if not all_train_X:
            raise ValueError("No training data available")
        
        X_train = np.concatenate(all_train_X, axis=0)
        y_train = np.concatenate(all_train_y, axis=0)
        X_val = np.concatenate(all_val_X, axis=0) if all_val_X else np.array([])
        y_val = np.concatenate(all_val_y, axis=0) if all_val_y else np.array([])
        X_test = np.concatenate(all_test_X, axis=0) if all_test_X else np.array([])
        y_test = np.concatenate(all_test_y, axis=0) if all_test_y else np.array([])
        
        del all_train_X, all_train_y, all_val_X, all_val_y, all_test_X, all_test_y
        gc.collect()
        
        logger.info(f"Temporal split results:")
        logger.info(f"  Train: {len(X_train):,} samples (before {self.train_end_date.date()})")
        logger.info(f"  Val:   {len(X_val):,} samples ({self.train_end_date.date()} to {self.val_end_date.date()})")
        logger.info(f"  Test:  {len(X_test):,} samples (after {self.val_end_date.date()})")
        
        return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols
    
    def scale_data(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray):
        """Scale data - fit on train only (prevent data leakage)."""
        logger.info("Scaling data (fit on train only)...")
        
        n_train, seq_len, n_features = X_train.shape
        
        X_train_flat = X_train.reshape(-1, n_features)
        X_val_flat = X_val.reshape(-1, n_features) if len(X_val) > 0 else None
        X_test_flat = X_test.reshape(-1, n_features) if len(X_test) > 0 else None
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_flat)
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_train_scaled = X_train_scaled.reshape(n_train, seq_len, n_features)
        
        if X_val_flat is not None:
            X_val_scaled = scaler.transform(X_val_flat)
            X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            X_val_scaled = X_val_scaled.reshape(X_val.shape)
        else:
            X_val_scaled = X_val
            
        if X_test_flat is not None:
            X_test_scaled = scaler.transform(X_test_flat)
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
        
        class_acc = {}
        for cls in [0, 1, 2]:
            mask = y_test == cls
            if mask.sum() > 0:
                class_acc[f"class_{cls}_acc"] = float((pred_classes[mask] == cls).mean())
        
        return {
            "accuracy": accuracy,
            "zero_rule_baseline": zero_rule_accuracy,
            "accuracy_lift": accuracy - zero_rule_accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "mcc": mcc,
            **class_acc,
        }
    
    def train_single_model(
        self, 
        config_name: str,
        config: Dict,
        X_train_scaled: np.ndarray, 
        y_train: np.ndarray, 
        X_val_scaled: np.ndarray, 
        y_val: np.ndarray, 
        X_test_scaled: np.ndarray, 
        y_test: np.ndarray,
        feature_cols: List[str]
    ) -> Dict:
        """Train a single model configuration and return metrics."""
        
        logger.info("=" * 60)
        logger.info(f"TRAINING: {config['description']}")
        logger.info("=" * 60)
        
        hidden_size = config["hidden_size"]
        num_layers = config["num_layers"]
        
        model = AttentionLSTM(
            input_size=X_train_scaled.shape[-1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout,
        )
        
        n_params = sum(p.numel() for p in model.parameters())
        ratio = len(X_train_scaled) / n_params
        logger.info(f"Model params: {n_params:,}, Samples/params: {ratio:.2f}x")
        
        loss_fn = nn.CrossEntropyLoss()
        trainer = Trainer(model, loss_fn, learning_rate=self.learning_rate, weight_decay=1e-4)
        
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train)),
            batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val_scaled), torch.LongTensor(y_val)),
            batch_size=self.batch_size
        )
        
        training_config.early_stopping_patience = self.early_stopping_patience
        history = trainer.train(train_loader, val_loader, epochs=self.epochs)
        
        logger.info("Evaluating on test set...")
        pred_classes = self.predict_batched(model, X_test_scaled)
        metrics = self.compute_metrics(y_test, pred_classes)
        cm = confusion_matrix(y_test, pred_classes)
        
        logger.info("\n" + "=" * 60)
        logger.info(f"RESULTS: {config['description']}")
        logger.info("=" * 60)
        logger.info(f"Test Accuracy:        {metrics['accuracy']:.2%}")
        logger.info(f"Zero-Rule Baseline:   {metrics['zero_rule_baseline']:.2%}")
        logger.info(f"Accuracy Lift:        {metrics['accuracy_lift']:+.2%}")
        logger.info(f"F1 (macro):           {metrics['f1_macro']:.4f}")
        logger.info(f"MCC:                  {metrics['mcc']:.4f}")
        logger.info("=" * 60)
        logger.info(f"\nConfusion Matrix:\n{cm}")
        logger.info(f"\n{classification_report(y_test, pred_classes, target_names=['Down', 'Neutral', 'Up'])}")
        
        return {
            "config_name": config_name,
            "description": config["description"],
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "n_features": X_train_scaled.shape[-1],
            "n_params": n_params,
            "n_stocks": len(self.stocks),
            "train_samples": len(X_train_scaled),
            "val_samples": len(X_val_scaled),
            "test_samples": len(X_test_scaled),
            "samples_per_param": ratio,
            **metrics,
            "confusion_matrix": cm.tolist(),
            "best_val_loss": trainer.best_val_loss,
            "epochs_trained": len(history.get("history", [])),
        }
    
    def run(self):
        logger.info("=" * 80)
        logger.info("LSTM SIZE COMPARISON EXPERIMENT")
        logger.info("=" * 80)
        logger.info(f"EXCLUDED FEATURES: {EXCLUDED_FEATURES}")
        logger.info(f"  Train period: 2014 to {self.train_end_date.date()}")
        logger.info(f"  Val period:   {self.train_end_date.date()} to {self.val_end_date.date()}")
        logger.info(f"  Test period:  {self.val_end_date.date()} to 2024")
        logger.info(f"  Configurations to compare: {list(MODEL_CONFIGS.keys())}")
        logger.info("=" * 80)
        
        # Load and prepare data once
        stock_data = self.load_all_stocks()
        X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = \
            self.prepare_temporal_pooled_data(stock_data)
        
        del stock_data
        import gc
        gc.collect()
        
        # Scale data once (fit on train only)
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_data(X_train, X_val, X_test)
        
        # Save feature list
        with open(self.output_dir / "features.txt", "w") as f:
            for col in feature_cols:
                f.write(f"{col}\n")
        
        # Train each configuration
        all_results = []
        for config_name, config in MODEL_CONFIGS.items():
            result = self.train_single_model(
                config_name, config,
                X_train_scaled, y_train,
                X_val_scaled, y_val,
                X_test_scaled, y_test,
                feature_cols
            )
            all_results.append(result)
            
            # Clear GPU memory between models
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # Save all results
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(self.output_dir / "results.csv", index=False)
        
        # Print comparison summary
        logger.info("\n" + "=" * 80)
        logger.info("COMPARISON SUMMARY")
        logger.info("=" * 80)
        summary_df = results_df[["config_name", "hidden_size", "num_layers", "n_params", "accuracy", "accuracy_lift", "f1_macro", "mcc"]].copy()
        summary_df = summary_df.sort_values("accuracy", ascending=False)
        logger.info(f"\n{summary_df.to_string(index=False)}")
        logger.info("=" * 80)
        
        logger.info(f"\nResults saved to: {self.output_dir}")
        return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--stocks", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--train-end", type=str, default="2022-01-01")
    parser.add_argument("--val-end", type=str, default="2023-06-01")
    parser.add_argument("--colab", action="store_true")
    args = parser.parse_args()
    
    stocks = EXTENDED_TICKERS[:args.stocks]
    
    experiment = LSTMSizeComparisonExperiment(
        stocks=stocks,
        train_end_date=args.train_end,
        val_end_date=args.val_end,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping_patience=args.patience,
        dropout=args.dropout,
        learning_rate=args.lr,
        use_cache=not args.no_cache,
        colab_mode=args.colab,
    )
    
    experiment.run()


if __name__ == "__main__":
    main()
