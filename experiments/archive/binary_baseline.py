"""
Binary Classification Baseline - Drop Neutral Class

Rationale:
- 3-class classification with ambiguous "Neutral" class is very hard
- Binary (Up vs Down) is a cleaner signal
- Random chance is 50%, so >55% accuracy is meaningful

This experiment:
1. Uses binary classification (Up vs Down only)
2. Drops Neutral days from training
3. Uses class weights for imbalance
4. Tries both short-term (next day) and longer-term (3-day) predictions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from config import REPORTS_DIR, training_config, lstm_config
from src.models import BaselineLSTM, AttentionLSTM
from src.training import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SimplifiedIndicators:
    """
    Compute a focused set of indicators that are known to work.
    Less is more - avoid overfitting with 74 features.
    """
    
    @staticmethod
    def compute(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Ensure columns
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col not in df.columns and col.lower() in df.columns:
                df[col] = df[col.lower()]
        
        # Returns (most important feature!)
        df['return_1d'] = df['Close'].pct_change()
        df['return_3d'] = df['Close'].pct_change(3)
        df['return_5d'] = df['Close'].pct_change(5)
        df['return_10d'] = df['Close'].pct_change(10)
        df['return_20d'] = df['Close'].pct_change(20)
        
        # Moving averages - normalized
        df['sma_5'] = df['Close'].rolling(5).mean() / df['Close'] - 1
        df['sma_20'] = df['Close'].rolling(20).mean() / df['Close'] - 1
        df['sma_50'] = df['Close'].rolling(50).mean() / df['Close'] - 1
        
        # EMA
        df['ema_12'] = df['Close'].ewm(span=12).mean() / df['Close'] - 1
        df['ema_26'] = df['Close'].ewm(span=26).mean() / df['Close'] - 1
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        df['rsi_14_normalized'] = (df['rsi_14'] - 50) / 50  # Center around 0
        
        # MACD
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        df['macd'] = (ema12 - ema26) / df['Close']  # Normalized
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        sma20 = df['Close'].rolling(20).mean()
        std20 = df['Close'].rolling(20).std()
        df['bb_upper'] = (sma20 + 2 * std20)
        df['bb_lower'] = (sma20 - 2 * std20)
        df['bb_pctb'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['Close']
        
        # Volatility
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean() / df['Close']  # Normalized
        df['volatility_20'] = df['return_1d'].rolling(20).std()
        
        # Volume
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['volume_trend'] = df['Volume'].rolling(5).mean() / df['Volume'].rolling(20).mean()
        
        # Price position
        df['high_low_pct'] = (df['Close'] - df['Low']) / (df['High'] - df['Low']).replace(0, np.nan)
        
        # Momentum
        df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        df['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        
        # Drop raw price columns (we want normalized features only)
        df = df.drop(columns=['bb_upper', 'bb_lower'], errors='ignore')
        
        return df


class BinaryExperiment:
    """Binary classification experiment."""
    
    def __init__(
        self,
        stock: str = "MSFT",
        start_date: str = "2015-01-01",
        end_date: str = "2025-01-01",
        sequence_length: int = 20,
        epochs: int = 100,
        prediction_horizon: int = 1,  # 1 = next day, 3 = 3 days
    ):
        self.stock = stock
        self.start_date = start_date
        self.end_date = end_date
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.prediction_horizon = prediction_horizon
        
        self.output_dir = REPORTS_DIR / f"binary_{stock}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_and_prepare(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load and prepare data for binary classification."""
        
        # Load data
        logger.info(f"Loading {self.stock} data...")
        stock = yf.Ticker(self.stock)
        df = stock.history(start=self.start_date, end=self.end_date)
        logger.info(f"Loaded {len(df)} trading days")
        
        # Compute indicators
        logger.info("Computing indicators...")
        df = SimplifiedIndicators.compute(df)
        
        # Binary label: Did the stock go UP in next N days?
        df['future_return'] = df['Close'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
        df['target'] = (df['future_return'] > 0).astype(int)  # 1 = Up, 0 = Down
        
        # Drop NaN
        feature_cols = [c for c in df.columns if c not in 
                       ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 
                        'Stock Splits', 'future_return', 'target']]
        
        df = df.dropna()
        logger.info(f"After dropna: {len(df)} samples, {len(feature_cols)} features")
        
        # Check class balance
        class_dist = df['target'].value_counts(normalize=True)
        logger.info(f"Class distribution: Down={class_dist.get(0, 0):.1%}, Up={class_dist.get(1, 0):.1%}")
        
        # Normalize features (fit on train only!)
        train_end = int(0.8 * len(df))
        scaler = StandardScaler()
        
        # Fit on training data only
        df_train_features = df[feature_cols].iloc[:train_end]
        scaler.fit(df_train_features)
        
        # Transform all
        df[feature_cols] = scaler.transform(df[feature_cols])
        
        # Create sequences
        data = df[feature_cols].values
        labels = df['target'].values
        
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(labels[i + self.sequence_length])
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)
        
        logger.info(f"Final shape: X={X.shape}, y={y.shape}")
        
        return X, y, feature_cols
    
    def train_random_forest(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train Random Forest as a baseline."""
        logger.info("\n" + "="*50)
        logger.info("RANDOM FOREST BASELINE")
        logger.info("="*50)
        
        # Split
        n = len(X)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)
        
        # Use last timestep for RF
        X_train = X[:train_end, -1, :]
        y_train = y[:train_end]
        X_test = X[val_end:, -1, :]
        y_test = y[val_end:]
        
        # Train
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        
        # Predict
        y_pred = rf.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        
        logger.info(f"Random Forest Test Accuracy: {accuracy:.2%}")
        logger.info(f"Classification Report:\n{classification_report(y_test, y_pred, target_names=['Down', 'Up'])}")
        
        return {"model": "random_forest", "accuracy": float(accuracy)}
    
    def train_gradient_boosting(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train Gradient Boosting as a baseline."""
        logger.info("\n" + "="*50)
        logger.info("GRADIENT BOOSTING BASELINE")
        logger.info("="*50)
        
        # Split
        n = len(X)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)
        
        # Use last timestep
        X_train = X[:train_end, -1, :]
        y_train = y[:train_end]
        X_test = X[val_end:, -1, :]
        y_test = y[val_end:]
        
        # Train
        gb = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        gb.fit(X_train, y_train)
        
        # Predict
        y_pred = gb.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        
        logger.info(f"Gradient Boosting Test Accuracy: {accuracy:.2%}")
        logger.info(f"Classification Report:\n{classification_report(y_test, y_pred, target_names=['Down', 'Up'])}")
        
        return {"model": "gradient_boosting", "accuracy": float(accuracy)}
    
    def train_lstm(self, X: np.ndarray, y: np.ndarray, model_config: str = "small") -> Dict:
        """Train LSTM model."""
        logger.info("\n" + "="*50)
        logger.info(f"LSTM MODEL ({model_config})")
        logger.info("="*50)
        
        # Split
        n = len(X)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Class weights
        n_down = (y_train == 0).sum()
        n_up = (y_train == 1).sum()
        weight_down = n_up / (n_down + n_up) * 2
        weight_up = n_down / (n_down + n_up) * 2
        class_weights = torch.tensor([weight_down, weight_up], dtype=torch.float32)
        logger.info(f"Class weights: Down={weight_down:.2f}, Up={weight_up:.2f}")
        
        # Model config
        if model_config == "small":
            hidden_size, num_layers, dropout = 32, 1, 0.3
        elif model_config == "medium":
            hidden_size, num_layers, dropout = 64, 2, 0.3
        else:
            hidden_size, num_layers, dropout = 128, 2, 0.2
        
        # Create model (2 classes for binary)
        model = BaselineLSTM(
            input_size=X.shape[-1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            num_trend_classes=2,
        )
        
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model params: {n_params:,}")
        logger.info(f"Samples/params ratio: {len(X_train)/n_params:.1f}x")
        
        # Loss with class weights
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
        
        # Trainer with lower learning rate
        trainer = Trainer(model, loss_fn, learning_rate=1e-4, weight_decay=1e-3)
        
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
            batch_size=64,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
            batch_size=64,
        )
        
        # Train with more patience
        training_config.early_stopping_patience = 30
        history = trainer.train(train_loader, val_loader, epochs=self.epochs)
        
        # Evaluate
        model.eval()
        device = next(model.parameters()).device
        
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            outputs = model(X_test_tensor)
            logits = outputs[0]  # trend_logits
            y_pred = logits.argmax(dim=-1).cpu().numpy()
        
        accuracy = (y_pred == y_test).mean()
        
        logger.info(f"LSTM Test Accuracy: {accuracy:.2%}")
        logger.info(f"Classification Report:\n{classification_report(y_test, y_pred, target_names=['Down', 'Up'])}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")
        
        return {
            "model": f"lstm_{model_config}",
            "accuracy": float(accuracy),
            "n_params": n_params,
        }
    
    def run(self):
        """Run all experiments."""
        logger.info("="*80)
        logger.info(f"BINARY CLASSIFICATION EXPERIMENT: {self.stock}")
        logger.info(f"Prediction horizon: {self.prediction_horizon} day(s)")
        logger.info("="*80)
        
        X, y, feature_cols = self.load_and_prepare()
        
        results = []
        
        # 1. Random Forest baseline
        results.append(self.train_random_forest(X, y))
        
        # 2. Gradient Boosting baseline
        results.append(self.train_gradient_boosting(X, y))
        
        # 3. Small LSTM
        results.append(self.train_lstm(X, y, "small"))
        
        # 4. Medium LSTM
        results.append(self.train_lstm(X, y, "medium"))
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("SUMMARY")
        logger.info("="*80)
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("accuracy", ascending=False)
        print(results_df.to_string())
        
        results_df.to_csv(self.output_dir / "results.csv", index=False)
        
        best = results_df.iloc[0]
        logger.info(f"\nBest: {best['model']} with {best['accuracy']:.2%} accuracy")
        
        return results_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", default="MSFT", help="Stock ticker")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon (days)")
    args = parser.parse_args()
    
    exp = BinaryExperiment(
        stock=args.stock,
        epochs=args.epochs,
        prediction_horizon=args.horizon,
    )
    exp.run()


if __name__ == "__main__":
    main()
