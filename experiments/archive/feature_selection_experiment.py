"""
Feature Selection Experiment - Single Stock Training with Different Feature Selection Methods.

This experiment trains the Medium-1L LSTM model (64 hidden, 1 layer) on individual stocks
using different feature selection algorithms:
1. Mutual Information - measures statistical dependency
2. SHAP Values - Shapley values from LightGBM
3. Recursive Feature Elimination (RFE) - iteratively removes least important features

Usage:
    python experiments/feature_selection_experiment.py --tickers DIA EBAY C MSFT --top-k 30
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import pickle
import json
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

try:
    import lightgbm as lgb
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: lightgbm or shap not available. SHAP-based selection will be skipped.")

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

# Features to exclude (suspected data leaks from v10)
EXCLUDED_FEATURES = [
    'ichimoku_ICS_26',
    'dpo',
]

# Default tickers to test
DEFAULT_TICKERS = ["DIA", "EBAY", "C", "MSFT"]


class FeatureSelector:
    """Feature selection methods for stock prediction."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        
    def mutual_information(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        feature_names: List[str],
        top_k: int = 30
    ) -> Tuple[List[str], np.ndarray]:
        """
        Select top features using Mutual Information.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels
            feature_names: List of feature names
            top_k: Number of top features to select
            
        Returns:
            Tuple of (selected feature names, MI scores)
        """
        logger.info(f"Computing Mutual Information for {len(feature_names)} features...")
        
        mi_scores = mutual_info_classif(
            X, y, 
            discrete_features=False,
            random_state=self.random_state,
            n_neighbors=5
        )
        
        # Get top-k indices
        top_indices = np.argsort(mi_scores)[-top_k:][::-1]
        selected_features = [feature_names[i] for i in top_indices]
        selected_scores = mi_scores[top_indices]
        
        logger.info(f"MI: Top 5 features: {selected_features[:5]}")
        return selected_features, mi_scores
    
    def shap_importance(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        feature_names: List[str],
        top_k: int = 30,
        sample_size: int = 5000
    ) -> Tuple[List[str], np.ndarray]:
        """
        Select top features using SHAP values from LightGBM.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels
            feature_names: List of feature names
            top_k: Number of top features to select
            sample_size: Number of samples for SHAP computation
            
        Returns:
            Tuple of (selected feature names, SHAP importance scores)
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available, returning empty selection")
            return [], np.array([])
            
        logger.info(f"Computing SHAP values using LightGBM...")
        
        # Train a quick LightGBM model
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=self.random_state,
            verbosity=-1,
            force_col_wise=True
        )
        lgb_model.fit(X, y)
        
        # Get SHAP values (use subset for speed)
        sample_idx = np.random.choice(len(X), min(sample_size, len(X)), replace=False)
        X_sample = X[sample_idx]
        
        explainer = shap.TreeExplainer(lgb_model)
        shap_values = explainer.shap_values(X_sample)
        
        # For multi-class, shap_values is a list of arrays
        if isinstance(shap_values, list):
            # Average absolute SHAP across classes
            mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            mean_shap = np.abs(shap_values).mean(axis=0)
        
        # Get top-k indices
        top_indices = np.argsort(mean_shap)[-top_k:][::-1]
        selected_features = [feature_names[i] for i in top_indices]
        
        logger.info(f"SHAP: Top 5 features: {selected_features[:5]}")
        return selected_features, mean_shap
    
    def recursive_feature_elimination(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        feature_names: List[str],
        top_k: int = 30
    ) -> Tuple[List[str], np.ndarray]:
        """
        Select top features using Recursive Feature Elimination.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels
            feature_names: List of feature names
            top_k: Number of top features to select
            
        Returns:
            Tuple of (selected feature names, RFE ranking)
        """
        if not SHAP_AVAILABLE:
            logger.warning("LightGBM not available for RFE, returning empty selection")
            return [], np.array([])
            
        logger.info(f"Running Recursive Feature Elimination for top {top_k} features...")
        
        # Use LightGBM as base estimator (faster than sklearn trees)
        estimator = lgb.LGBMClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=self.random_state,
            verbosity=-1,
            force_col_wise=True
        )
        
        # RFE with step=10 for faster execution
        rfe = RFE(
            estimator=estimator, 
            n_features_to_select=top_k, 
            step=10,
            verbose=0
        )
        rfe.fit(X, y)
        
        # Get selected features
        selected_mask = rfe.support_
        selected_features = [feature_names[i] for i, selected in enumerate(selected_mask) if selected]
        ranking = rfe.ranking_
        
        logger.info(f"RFE: Selected {len(selected_features)} features")
        logger.info(f"RFE: Top 5 features: {selected_features[:5]}")
        return selected_features, ranking


class SingleStockFeatureExperiment:
    """
    Experiment to compare feature selection methods on single stocks.
    """
    
    def __init__(
        self,
        tickers: List[str] = None,
        start_date: str = "2014-01-01",
        end_date: str = "2024-12-31",
        train_end_date: str = "2022-01-01",
        val_end_date: str = "2023-06-01",
        sequence_length: int = 20,
        epochs: int = 100,
        batch_size: int = 64,
        early_stopping_patience: int = 15,
        dropout: float = 0.5,
        learning_rate: float = 5e-4,
        top_k_features: List[int] = None,
        use_cache: bool = True,
    ):
        self.tickers = tickers or DEFAULT_TICKERS
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
        
        self.output_dir = REPORTS_DIR / f"feature_selection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "per_stock").mkdir(exist_ok=True)
        
        self.indicator_computer = ComprehensiveIndicatorsV7()
        self.feature_selector = FeatureSelector()
        
        # Medium-1L configuration
        self.hidden_size = 64
        self.num_layers = 1
        
    def load_single_stock(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load data for a single stock."""
        cache_path = DATA_DIR / "price_cache" / f"{ticker}_{self.start_date}_{self.end_date}.pkl"
        
        if self.use_cache and cache_path.exists():
            with open(cache_path, "rb") as f:
                df = pickle.load(f)
            logger.info(f"Loaded {ticker} from cache: {len(df)} rows")
            return df
            
        try:
            df = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
            if len(df) < 100:
                logger.warning(f"Insufficient data for {ticker}: {len(df)} rows")
                return None
            df.columns = [c.capitalize() if isinstance(c, str) else c[0].capitalize() for c in df.columns]
            df['Ticker'] = ticker
            
            if self.use_cache:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump(df, f)
                    
            logger.info(f"Downloaded {ticker}: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Failed to load {ticker}: {e}")
            return None
    
    def prepare_stock_data(self, df: pd.DataFrame) -> Tuple[
        np.ndarray, np.ndarray,  # X_train, y_train
        np.ndarray, np.ndarray,  # X_val, y_val
        np.ndarray, np.ndarray,  # X_test, y_test
        List[str]                # feature_cols
    ]:
        """Prepare data for a single stock with temporal split."""
        df = self.indicator_computer.compute_all(df)
        
        feature_cols = self.indicator_computer.get_indicator_columns(df)
        feature_cols = [col for col in feature_cols if col not in EXCLUDED_FEATURES]
        valid_cols = [col for col in feature_cols 
                      if col in df.columns and df[col].notna().sum() > 10]
        feature_cols = valid_cols
        
        if len(feature_cols) < 5:
            return (np.array([]), np.array([]), np.array([]), 
                    np.array([]), np.array([]), np.array([]), [])
        
        # Create target - T+1 prediction
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
        
        # Create sequences with temporal split
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
            feature_cols
        )
    
    def train_model_with_features(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_indices: List[int] = None,
        description: str = "all_features"
    ) -> Dict:
        """Train model with selected features."""
        
        # Filter features if indices provided
        if feature_indices is not None:
            X_train = X_train[:, :, feature_indices]
            X_val = X_val[:, :, feature_indices]
            X_test = X_test[:, :, feature_indices]
        
        n_features = X_train.shape[-1]
        
        # Scale data
        n_train, seq_len, _ = X_train.shape
        scaler = StandardScaler()
        X_train_flat = X_train.reshape(-1, n_features)
        X_train_scaled = scaler.fit_transform(X_train_flat)
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_train_scaled = X_train_scaled.reshape(n_train, seq_len, n_features)
        
        if len(X_val) > 0:
            X_val_scaled = scaler.transform(X_val.reshape(-1, n_features))
            X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            X_val_scaled = X_val_scaled.reshape(X_val.shape[0], seq_len, n_features)
        else:
            X_val_scaled = X_val
            
        if len(X_test) > 0:
            X_test_scaled = scaler.transform(X_test.reshape(-1, n_features))
            X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            X_test_scaled = X_test_scaled.reshape(X_test.shape[0], seq_len, n_features)
        else:
            X_test_scaled = X_test
        
        # Create model
        model = AttentionLSTM(
            input_size=n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        
        n_params = sum(p.numel() for p in model.parameters())
        
        # Train
        loss_fn = nn.CrossEntropyLoss()
        trainer = Trainer(model, loss_fn, learning_rate=self.learning_rate, weight_decay=1e-4)
        
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train)),
            batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val_scaled), torch.LongTensor(y_val)),
            batch_size=self.batch_size
        ) if len(X_val_scaled) > 0 else None
        
        training_config.early_stopping_patience = self.early_stopping_patience
        history = trainer.train(train_loader, val_loader, epochs=self.epochs)
        
        # Evaluate on test set
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            test_tensor = torch.FloatTensor(X_test_scaled).to(device)
            predictions = model.predict(test_tensor)
            pred_classes = predictions["trend_class"].cpu().numpy()
        
        # Compute metrics
        accuracy = (pred_classes == y_test).mean()
        class_counts = Counter(y_test)
        zero_rule = class_counts.most_common(1)[0][1] / len(y_test)
        
        return {
            "description": description,
            "n_features": n_features,
            "n_params": n_params,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "accuracy": accuracy,
            "zero_rule_baseline": zero_rule,
            "accuracy_lift": accuracy - zero_rule,
            "f1_macro": f1_score(y_test, pred_classes, average='macro'),
            "mcc": matthews_corrcoef(y_test, pred_classes),
            "epochs_trained": len(history.get("history", [])),
        }
    
    def run_single_stock(self, ticker: str) -> Dict:
        """Run full feature selection experiment for a single stock."""
        logger.info("=" * 60)
        logger.info(f"PROCESSING: {ticker}")
        logger.info("=" * 60)
        
        # Load and prepare data
        df = self.load_single_stock(ticker)
        if df is None:
            return {"ticker": ticker, "error": "Failed to load data"}
        
        X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = self.prepare_stock_data(df)
        
        if len(X_train) < 100:
            return {"ticker": ticker, "error": f"Insufficient training data: {len(X_train)} samples"}
        
        logger.info(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        logger.info(f"  Features: {len(feature_cols)}")
        
        # Flatten for feature selection (use last timestep)
        X_train_flat = X_train[:, -1, :]
        
        results = []
        feature_rankings = {}
        
        # 1. Baseline with all features
        logger.info("Training with ALL features...")
        baseline_result = self.train_model_with_features(
            X_train, y_train, X_val, y_val, X_test, y_test,
            description="all_features"
        )
        baseline_result["ticker"] = ticker
        baseline_result["method"] = "baseline"
        baseline_result["top_k"] = len(feature_cols)
        results.append(baseline_result)
        
        # Feature selection methods
        methods = [
            ("mutual_info", self.feature_selector.mutual_information),
            ("shap", self.feature_selector.shap_importance),
            ("rfe", self.feature_selector.recursive_feature_elimination),
        ]
        
        for method_name, method_func in methods:
            logger.info(f"\nApplying {method_name.upper()} feature selection...")
            
            for top_k in self.top_k_features:
                if top_k >= len(feature_cols):
                    continue
                    
                try:
                    selected_features, scores = method_func(
                        X_train_flat, y_train, feature_cols, top_k=top_k
                    )
                    
                    if len(selected_features) == 0:
                        continue
                    
                    # Get feature indices
                    feature_indices = [feature_cols.index(f) for f in selected_features if f in feature_cols]
                    
                    # Store rankings
                    key = f"{method_name}_top{top_k}"
                    feature_rankings[key] = selected_features
                    
                    # Train model
                    result = self.train_model_with_features(
                        X_train, y_train, X_val, y_val, X_test, y_test,
                        feature_indices=feature_indices,
                        description=f"{method_name}_top{top_k}"
                    )
                    result["ticker"] = ticker
                    result["method"] = method_name
                    result["top_k"] = top_k
                    result["selected_features"] = selected_features[:10]  # Top 10 for reference
                    results.append(result)
                    
                    logger.info(f"  {method_name} top-{top_k}: Acc={result['accuracy']:.2%}, Lift={result['accuracy_lift']:+.2%}")
                    
                except Exception as e:
                    logger.error(f"  {method_name} top-{top_k} failed: {e}")
        
        # Save per-stock results
        stock_output = {
            "ticker": ticker,
            "feature_rankings": feature_rankings,
            "results": results,
        }
        with open(self.output_dir / "per_stock" / f"{ticker}_features.json", "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            stock_output_json = {
                "ticker": ticker,
                "feature_rankings": {k: list(v) if isinstance(v, np.ndarray) else v 
                                    for k, v in feature_rankings.items()},
                "results": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                            for k, v in r.items()} for r in results],
            }
            json.dump(stock_output_json, f, indent=2, default=str)
        
        return {"ticker": ticker, "results": results, "feature_rankings": feature_rankings}
    
    def compute_feature_overlap(self, all_rankings: Dict[str, Dict[str, List[str]]]) -> pd.DataFrame:
        """Compute Jaccard similarity between feature selection methods."""
        records = []
        
        for ticker, rankings in all_rankings.items():
            methods = list(rankings.keys())
            for i, m1 in enumerate(methods):
                for m2 in methods[i+1:]:
                    set1 = set(rankings[m1])
                    set2 = set(rankings[m2])
                    if len(set1) > 0 and len(set2) > 0:
                        jaccard = len(set1 & set2) / len(set1 | set2)
                        overlap_count = len(set1 & set2)
                        records.append({
                            "ticker": ticker,
                            "method_1": m1,
                            "method_2": m2,
                            "jaccard_similarity": jaccard,
                            "overlap_count": overlap_count,
                        })
        
        return pd.DataFrame(records)
    
    def run(self):
        logger.info("=" * 80)
        logger.info("FEATURE SELECTION EXPERIMENT")
        logger.info("=" * 80)
        logger.info(f"Tickers: {self.tickers}")
        logger.info(f"Feature reduction levels: {self.top_k_features}")
        logger.info(f"Model: Medium-1L (hidden={self.hidden_size}, layers={self.num_layers})")
        logger.info("=" * 80)
        
        all_results = []
        all_rankings = {}
        
        for ticker in self.tickers:
            stock_output = self.run_single_stock(ticker)
            
            if "error" not in stock_output:
                all_results.extend(stock_output["results"])
                all_rankings[ticker] = stock_output["feature_rankings"]
            else:
                logger.error(f"Skipping {ticker}: {stock_output.get('error')}")
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Save combined results
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(self.output_dir / "results.csv", index=False)
        
        # Compute and save feature overlap
        if all_rankings:
            overlap_df = self.compute_feature_overlap(all_rankings)
            overlap_df.to_csv(self.output_dir / "feature_overlap.csv", index=False)
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        
        summary = results_df.groupby(["ticker", "method"]).agg({
            "accuracy": "max",
            "accuracy_lift": "max",
            "n_features": "min"
        }).reset_index()
        logger.info(f"\n{summary.to_string(index=False)}")
        
        logger.info(f"\nResults saved to: {self.output_dir}")
        return results_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--top-k", nargs="+", type=int, default=[20, 30, 50])
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    
    experiment = SingleStockFeatureExperiment(
        tickers=args.tickers,
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
