"""
Pooled Feature Selection Experiment - Multi-Stock Training with Feature Selection.

Based on pooled_price_baseline_v10.py temporal split approach.
Compares feature selection methods on pooled multi-stock data:
1. Mutual Information - measures statistical dependency
2. SHAP Values - Shapley values from LightGBM  
3. Recursive Feature Elimination (RFE) - iteratively removes least important features

Usage:
    python experiments/pooled_feature_selection.py --stocks 200 --epochs 50 --top-k 20 30 50
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import pickle
import json
import gc
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.metrics import confusion_matrix, classification_report, f1_score, matthews_corrcoef
from tqdm import tqdm
from collections import Counter

# Import LightGBM and SHAP
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("Warning: lightgbm not available.")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: shap not available.")

try:
    from config import REPORTS_DIR, DATA_DIR, training_config
    from src.models import AttentionLSTM
    from src.training import Trainer
    from src.features.indicators_v7 import ComprehensiveIndicatorsV7
    RUNNING_LOCAL = True
except ImportError:
    RUNNING_LOCAL = False

# Import tickers
from experiments.ranked_tickers import EXTENDED_TICKERS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logger.info(f"Python Version: {sys.version}")
logger.info(f"PyTorch Version: {torch.__version__}")
logger.info(f"LightGBM Available: {LGB_AVAILABLE}")
logger.info(f"SHAP Available: {SHAP_AVAILABLE}")
if torch.cuda.is_available():
    logger.info(f"CUDA Available: True, Device: {torch.cuda.get_device_name(0)}")

# Features to exclude (suspected data leaks from v10)
EXCLUDED_FEATURES = [
    'ichimoku_ICS_26',  # Chikou Span - plots close 26 periods back
    'dpo',              # Detrended Price Oscillator
]


class FeatureSelector:
    """Feature selection methods for stock prediction."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        
    def mutual_information(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        feature_names: List[str],
        top_k: int = 30
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Select top features using Mutual Information.
        """
        logger.info(f"Computing Mutual Information for {len(feature_names)} features...")
        
        # Handle NaN/Inf values
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        mi_scores = mutual_info_classif(
            X_clean, y, 
            discrete_features=False,
            random_state=self.random_state,
            n_neighbors=5
        )
        
        # Create feature -> score mapping
        score_dict = {feature_names[i]: float(mi_scores[i]) for i in range(len(feature_names))}
        
        # Get top-k indices
        top_indices = np.argsort(mi_scores)[-top_k:][::-1]
        selected_features = [feature_names[i] for i in top_indices]
        
        logger.info(f"MI: Top 5 features: {selected_features[:5]}")
        logger.info(f"MI: Top 5 scores: {[score_dict[f] for f in selected_features[:5]]}")
        return selected_features, score_dict
    
    def shap_importance(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        feature_names: List[str],
        top_k: int = 30,
        sample_size: int = 10000
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Select top features using SHAP values from LightGBM.
        """
        if not LGB_AVAILABLE:
            logger.warning("LightGBM not available, skipping SHAP")
            return [], {}
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available, skipping SHAP")
            return [], {}
            
        logger.info(f"Computing SHAP values using LightGBM...")
        
        # Handle NaN/Inf values
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Train a quick LightGBM model
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            num_leaves=31,
            random_state=self.random_state,
            verbosity=-1,
            force_col_wise=True,
            n_jobs=-1
        )
        
        logger.info("  Training LightGBM for SHAP...")
        lgb_model.fit(X_clean, y)
        
        # Get SHAP values (use subset for speed)
        sample_size = min(sample_size, len(X_clean))
        sample_idx = np.random.choice(len(X_clean), sample_size, replace=False)
        X_sample = X_clean[sample_idx]
        
        logger.info(f"  Computing SHAP values on {sample_size} samples...")
        explainer = shap.TreeExplainer(lgb_model)
        shap_values = explainer.shap_values(X_sample)
        
        # Debug: print shape info
        n_features = len(feature_names)
        logger.info(f"  Debug: n_features={n_features}, X_sample.shape={X_sample.shape}")
        
        # Handle different SHAP output formats:
        # 1. List of arrays [class0, class1, class2] - older format
        #    Each array has shape (n_samples, n_features)
        # 2. 3D array with shape (n_samples, n_features, n_classes) - newer format
        
        if isinstance(shap_values, list):
            logger.info(f"  Debug: shap_values is list of {len(shap_values)} arrays")
            logger.info(f"  Debug: shap_values[0].shape = {shap_values[0].shape}")
            # Average absolute SHAP across all classes
            abs_shap_per_class = [np.abs(sv).mean(axis=0) for sv in shap_values]
            mean_shap = np.mean(abs_shap_per_class, axis=0)
        elif len(shap_values.shape) == 3:
            # 3D array: (n_samples, n_features, n_classes)
            logger.info(f"  Debug: shap_values is 3D array with shape={shap_values.shape}")
            # Take mean over samples (axis=0), then mean over classes (axis=1 after first mean)
            # This gives us (n_features, n_classes), then mean over classes
            mean_per_sample = np.abs(shap_values).mean(axis=0)  # (n_features, n_classes)
            mean_shap = mean_per_sample.mean(axis=1)  # (n_features,)
            logger.info(f"  Debug: mean_per_sample.shape = {mean_per_sample.shape}, mean_shap.shape = {mean_shap.shape}")
        else:
            logger.info(f"  Debug: shap_values is 2D array with shape={shap_values.shape}")
            mean_shap = np.abs(shap_values).mean(axis=0)
        
        # Ensure mean_shap is 1D and has correct length
        mean_shap = np.asarray(mean_shap).flatten()
        logger.info(f"  Debug: mean_shap.shape after flatten = {mean_shap.shape}")
        
        # Verify dimensions match
        if len(mean_shap) != n_features:
            logger.error(f"  Shape mismatch: mean_shap has {len(mean_shap)} elements but expected {n_features}")
            return [], {}
        
        # Create feature -> score mapping
        score_dict = {feature_names[i]: float(mean_shap[i]) for i in range(n_features)}
        
        # Get top-k indices (ensure we don't exceed array bounds)
        top_k = min(top_k, n_features)
        top_indices = np.argsort(mean_shap)[-top_k:][::-1]
        selected_features = [feature_names[i] for i in top_indices]
        
        logger.info(f"SHAP: Top 5 features: {selected_features[:5]}")
        logger.info(f"SHAP: Top 5 scores: {[f'{score_dict[f]:.4f}' for f in selected_features[:5]]}")
        return selected_features, score_dict
    
    def recursive_feature_elimination(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        feature_names: List[str],
        top_k: int = 30
    ) -> Tuple[List[str], Dict[str, int]]:
        """
        Select top features using Recursive Feature Elimination.
        """
        if not LGB_AVAILABLE:
            logger.warning("LightGBM not available for RFE, skipping")
            return [], {}
            
        logger.info(f"Running RFE for top {top_k} features...")
        
        # Handle NaN/Inf values
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Use LightGBM as base estimator
        estimator = lgb.LGBMClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            random_state=self.random_state,
            verbosity=-1,
            force_col_wise=True,
            n_jobs=-1
        )
        
        # RFE with step=5 for faster execution
        step = max(5, (len(feature_names) - top_k) // 10)
        rfe = RFE(
            estimator=estimator, 
            n_features_to_select=top_k, 
            step=step,
            verbose=0
        )
        
        logger.info(f"  Running RFE with step={step}...")
        rfe.fit(X_clean, y)
        
        # Get selected features and rankings
        selected_mask = rfe.support_
        ranking = rfe.ranking_
        
        # Create feature -> ranking mapping (lower is better, 1 = selected)
        rank_dict = {feature_names[i]: int(ranking[i]) for i in range(len(feature_names))}
        
        selected_features = [feature_names[i] for i, selected in enumerate(selected_mask) if selected]
        
        logger.info(f"RFE: Selected {len(selected_features)} features")
        logger.info(f"RFE: Top 5 features: {selected_features[:5]}")
        return selected_features, rank_dict


class PriceCache:
    """Cache for yfinance price data (from v10)."""
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or DATA_DIR / "price_cache"
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


class PooledFeatureSelectionExperiment:
    """
    Experiment to compare feature selection methods on pooled multi-stock data.
    Uses v10's temporal split approach to avoid data leakage.
    """
    
    def __init__(
        self,
        stocks: List[str] = None,
        start_date: str = "2014-01-01",
        end_date: str = "2024-12-31",
        train_end_date: str = "2022-01-01",
        val_end_date: str = "2023-06-01",
        sequence_length: int = 20,
        epochs: int = 50,
        batch_size: int = 1024,
        early_stopping_patience: int = 10,
        dropout: float = 0.4,
        learning_rate: float = 5e-4,
        top_k_features: List[int] = None,
        use_cache: bool = True,
    ):
        self.stocks = stocks or EXTENDED_TICKERS[:200]
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
        self.top_k_features = top_k_features or [20, 30, 50]
        self.use_cache = use_cache
        
        self.output_dir = REPORTS_DIR / f"pooled_feature_selection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.indicator_computer = ComprehensiveIndicatorsV7()
        self.feature_selector = FeatureSelector()
        self.price_cache = PriceCache()
        
        # Medium-1L configuration
        self.hidden_size = 64
        self.num_layers = 1
        
    def load_all_stocks(self) -> Dict[str, pd.DataFrame]:
        """Load stocks from cache or download (from v10)."""
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
    
    def prepare_stock_with_temporal_split(
        self, df: pd.DataFrame, target_feature_cols: List[str] = None
    ) -> Tuple[
        np.ndarray, np.ndarray,  # X_train, y_train
        np.ndarray, np.ndarray,  # X_val, y_val
        np.ndarray, np.ndarray,  # X_test, y_test
        List[str]                # feature_cols
    ]:
        """
        V10 FIX: Split THIS stock by DATE, then create sequences.
        Returns train/val/test already split for this stock.
        """
        df = self.indicator_computer.compute_all(df)
        
        if target_feature_cols:
            feature_cols = target_feature_cols
            for col in feature_cols:
                if col not in df.columns:
                    df[col] = 0.0
        else:
            feature_cols = self.indicator_computer.get_indicator_columns(df)
            feature_cols = [col for col in feature_cols if col not in EXCLUDED_FEATURES]
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
        
        feature_data = df[feature_cols].values
        feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
        labels = df['trend'].values
        dates = df.index
        
        X_train, y_train = [], []
        X_val, y_val = [], []
        X_test, y_test = [], []
        
        for i in range(len(feature_data) - self.sequence_length):
            target_date = dates[i + self.sequence_length]
            seq = feature_data[i:i + self.sequence_length]
            # FIX: Predict T+1, not T+2
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
            feature_cols
        )
    
    def prepare_temporal_pooled_data(self, stock_data: Dict[str, pd.DataFrame]) -> Tuple[
        np.ndarray, np.ndarray,
        np.ndarray, np.ndarray,
        np.ndarray, np.ndarray,
        List[str]
    ]:
        """Process all stocks, split each by date, then pool train/val/test separately (from v10)."""
        all_train_X, all_train_y = [], []
        all_val_X, all_val_y = [], []
        all_test_X, all_test_y = [], []
        feature_cols = None
        success_count = 0
        
        for ticker, df in tqdm(stock_data.items(), desc="Processing stocks with temporal split"):
            try:
                X_tr, y_tr, X_val, y_val, X_te, y_te, cols = self.prepare_stock_with_temporal_split(
                    df, target_feature_cols=feature_cols
                )
                
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
        """Scale data - fit on train only."""
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
        
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler
    
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
    
    def train_model_with_features(
        self,
        X_train_scaled: np.ndarray,
        y_train: np.ndarray,
        X_val_scaled: np.ndarray,
        y_val: np.ndarray,
        X_test_scaled: np.ndarray,
        y_test: np.ndarray,
        feature_indices: List[int] = None,
        description: str = "all_features"
    ) -> Dict:
        """Train model with selected features."""
        
        # Filter features if indices provided
        if feature_indices is not None:
            X_train_sel = X_train_scaled[:, :, feature_indices]
            X_val_sel = X_val_scaled[:, :, feature_indices] if len(X_val_scaled) > 0 else X_val_scaled
            X_test_sel = X_test_scaled[:, :, feature_indices] if len(X_test_scaled) > 0 else X_test_scaled
        else:
            X_train_sel = X_train_scaled
            X_val_sel = X_val_scaled
            X_test_sel = X_test_scaled
        
        n_features = X_train_sel.shape[-1]
        
        model = AttentionLSTM(
            input_size=n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        
        n_params = sum(p.numel() for p in model.parameters())
        ratio = len(X_train_sel) / n_params
        
        logger.info(f"Training {description}: {n_features} features, {n_params:,} params, {ratio:.2f}x samples/params")
        
        loss_fn = nn.CrossEntropyLoss()
        trainer = Trainer(model, loss_fn, learning_rate=self.learning_rate, weight_decay=1e-4)
        
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train_sel), torch.LongTensor(y_train)),
            batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val_sel), torch.LongTensor(y_val)),
            batch_size=self.batch_size
        ) if len(X_val_sel) > 0 else None
        
        training_config.early_stopping_patience = self.early_stopping_patience
        history = trainer.train(train_loader, val_loader, epochs=self.epochs)
        
        # Evaluate
        pred_classes = self.predict_batched(model, X_test_sel)
        metrics = self.compute_metrics(y_test, pred_classes)
        
        logger.info(f"  {description}: Acc={metrics['accuracy']:.2%}, Lift={metrics['accuracy_lift']:+.2%}, MCC={metrics['mcc']:.4f}")
        
        # Clean up
        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return {
            "description": description,
            "n_features": n_features,
            "n_params": n_params,
            "train_samples": len(X_train_sel),
            "test_samples": len(X_test_sel),
            "samples_per_param": ratio,
            "epochs_trained": len(history.get("history", [])),
            **metrics,
        }
    
    def run(self):
        logger.info("=" * 80)
        logger.info("POOLED FEATURE SELECTION EXPERIMENT")
        logger.info("=" * 80)
        logger.info(f"Stocks: {len(self.stocks)}")
        logger.info(f"Feature reduction levels: {self.top_k_features}")
        logger.info(f"Model: Medium-1L (hidden={self.hidden_size}, layers={self.num_layers})")
        logger.info(f"Train period: 2014 to {self.train_end_date.date()}")
        logger.info(f"Val period: {self.train_end_date.date()} to {self.val_end_date.date()}")
        logger.info(f"Test period: {self.val_end_date.date()} to 2024")
        logger.info("=" * 80)
        
        # Load and prepare data
        stock_data = self.load_all_stocks()
        X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = \
            self.prepare_temporal_pooled_data(stock_data)
        
        del stock_data
        gc.collect()
        
        # Scale data
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = self.scale_data(X_train, X_val, X_test)
        
        # Free unscaled data
        del X_train, X_val, X_test
        gc.collect()
        
        # Save feature list
        with open(self.output_dir / "features.txt", "w") as f:
            for col in feature_cols:
                f.write(f"{col}\n")
        
        results = []
        all_feature_rankings = {}
        mi_scores = {}  # Initialize to avoid unbound error
        shap_scores = {}  # Initialize to avoid unbound error
        
        # Flatten training data for feature selection (use last timestep)
        X_train_flat = X_train_scaled[:, -1, :]
        
        # 1. Baseline with all features
        logger.info("\n" + "=" * 60)
        logger.info("BASELINE: All Features")
        logger.info("=" * 60)
        baseline_result = self.train_model_with_features(
            X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test,
            description="baseline_all"
        )
        baseline_result["method"] = "baseline"
        baseline_result["top_k"] = len(feature_cols)
        results.append(baseline_result)
        
        # 2. Mutual Information
        logger.info("\n" + "=" * 60)
        logger.info("MUTUAL INFORMATION")
        logger.info("=" * 60)
        
        for top_k in self.top_k_features:
            try:
                selected_features, mi_scores = self.feature_selector.mutual_information(
                    X_train_flat, y_train, feature_cols, top_k=top_k
                )
                
                feature_indices = [feature_cols.index(f) for f in selected_features]
                all_feature_rankings[f"mi_top{top_k}"] = selected_features
                
                result = self.train_model_with_features(
                    X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test,
                    feature_indices=feature_indices,
                    description=f"mi_top{top_k}"
                )
                result["method"] = "mutual_info"
                result["top_k"] = top_k
                result["selected_features"] = selected_features[:10]
                results.append(result)
                
            except Exception as e:
                logger.error(f"MI top-{top_k} failed: {e}")
        
        # Save MI scores
        if mi_scores:
            with open(self.output_dir / "mi_scores.json", "w") as f:
                json.dump(mi_scores, f, indent=2)
        
        # 3. SHAP
        logger.info("\n" + "=" * 60)
        logger.info("SHAP VALUES")
        logger.info("=" * 60)
        
        for top_k in self.top_k_features:
            try:
                selected_features, shap_scores = self.feature_selector.shap_importance(
                    X_train_flat, y_train, feature_cols, top_k=top_k
                )
                
                if len(selected_features) == 0:
                    logger.warning(f"SHAP top-{top_k} returned no features")
                    continue
                
                feature_indices = [feature_cols.index(f) for f in selected_features]
                all_feature_rankings[f"shap_top{top_k}"] = selected_features
                
                result = self.train_model_with_features(
                    X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test,
                    feature_indices=feature_indices,
                    description=f"shap_top{top_k}"
                )
                result["method"] = "shap"
                result["top_k"] = top_k
                result["selected_features"] = selected_features[:10]
                results.append(result)
                
            except Exception as e:
                logger.error(f"SHAP top-{top_k} failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Save SHAP scores
        if shap_scores:
            with open(self.output_dir / "shap_scores.json", "w") as f:
                json.dump(shap_scores, f, indent=2)
        
        # 4. RFE
        logger.info("\n" + "=" * 60)
        logger.info("RECURSIVE FEATURE ELIMINATION")
        logger.info("=" * 60)
        
        for top_k in self.top_k_features:
            try:
                selected_features, rfe_rankings = self.feature_selector.recursive_feature_elimination(
                    X_train_flat, y_train, feature_cols, top_k=top_k
                )
                
                if len(selected_features) == 0:
                    logger.warning(f"RFE top-{top_k} returned no features")
                    continue
                
                feature_indices = [feature_cols.index(f) for f in selected_features]
                all_feature_rankings[f"rfe_top{top_k}"] = selected_features
                
                result = self.train_model_with_features(
                    X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test,
                    feature_indices=feature_indices,
                    description=f"rfe_top{top_k}"
                )
                result["method"] = "rfe"
                result["top_k"] = top_k
                result["selected_features"] = selected_features[:10]
                results.append(result)
                
            except Exception as e:
                logger.error(f"RFE top-{top_k} failed: {e}")
        
        # Save all results
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_dir / "results.csv", index=False)
        
        # Save feature rankings
        with open(self.output_dir / "feature_rankings.json", "w") as f:
            json.dump(all_feature_rankings, f, indent=2)
        
        # Compute feature overlap
        overlap_records = []
        methods = list(all_feature_rankings.keys())
        for i, m1 in enumerate(methods):
            for m2 in methods[i+1:]:
                set1 = set(all_feature_rankings[m1])
                set2 = set(all_feature_rankings[m2])
                if len(set1) > 0 and len(set2) > 0:
                    jaccard = len(set1 & set2) / len(set1 | set2)
                    overlap_records.append({
                        "method_1": m1,
                        "method_2": m2,
                        "jaccard_similarity": jaccard,
                        "overlap_count": len(set1 & set2),
                        "overlap_features": list(set1 & set2)[:10]
                    })
        
        overlap_df = pd.DataFrame(overlap_records)
        overlap_df.to_csv(self.output_dir / "feature_overlap.csv", index=False)
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        
        summary_df = results_df[["method", "top_k", "n_features", "accuracy", "accuracy_lift", "f1_macro", "mcc"]].copy()
        summary_df = summary_df.sort_values("accuracy", ascending=False)
        logger.info(f"\n{summary_df.to_string(index=False)}")
        
        logger.info(f"\nResults saved to: {self.output_dir}")
        return results_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--top-k", nargs="+", type=int, default=[20, 30, 50])
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    
    stocks = EXTENDED_TICKERS[:args.stocks]
    
    experiment = PooledFeatureSelectionExperiment(
        stocks=stocks,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping_patience=args.patience,
        dropout=args.dropout,
        learning_rate=args.lr,
        top_k_features=args.top_k,
        use_cache=not args.no_cache,
    )
    
    experiment.run()


if __name__ == "__main__":
    main()
