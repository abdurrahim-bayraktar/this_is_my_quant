"""
Pooled Price-Only Baseline V3 - Attention LSTM with Feature Selection.

This experiment:
1. Uses ATTENTION LSTM (best performer from v1)
2. Full 74 indicator suite (from original pooled_price_baseline.py)
3. High early stopping patience (50)
4. Compares: Full features vs RF-selected top 12

Usage:
    python experiments/pooled_price_baseline_v3.py --epochs 100 --patience 50
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
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from config import REPORTS_DIR, TOP_200_TICKERS, training_config, lstm_config
from src.models import AttentionLSTM  # Use Attention LSTM only
from src.training import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Diagnostic Logging
logger.info(f"Python Version: {sys.version}")
logger.info(f"PyTorch Version: {torch.__version__}")
if torch.cuda.is_available():
    logger.info(f"CUDA Available: True")
    logger.info(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    logger.warning("CUDA NOT AVAILABLE - Training will be slow on CPU")


class ComprehensiveIndicators:
    """
    Compute all 40+ technical indicators.
    EXACT COPY from pooled_price_baseline.py for full indicator suite.
    """
    
    def __init__(self):
        try:
            import pandas_ta as ta
            self.ta = ta
            self.ta_available = True
        except ImportError:
            logger.warning("pandas_ta not installed. Run: pip install pandas_ta")
            self.ta_available = False
    
    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical indicators."""
        df = df.copy()
        
        # Ensure we have the required columns
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required:
            if col not in df.columns:
                if col.lower() in df.columns:
                    df[col] = df[col.lower()]
                elif col.capitalize() in df.columns:
                    df[col] = df[col.capitalize()]
        
        if not self.ta_available:
            return self._compute_basic_indicators(df)
        
        # ===== MOVING AVERAGES & TREND =====
        df['sma_5'] = self.ta.sma(df['Close'], length=5)
        df['sma_10'] = self.ta.sma(df['Close'], length=10)
        df['sma_20'] = self.ta.sma(df['Close'], length=20)
        df['sma_50'] = self.ta.sma(df['Close'], length=50)
        
        df['ema_5'] = self.ta.ema(df['Close'], length=5)
        df['ema_12'] = self.ta.ema(df['Close'], length=12)
        df['ema_26'] = self.ta.ema(df['Close'], length=26)
        
        df['vwma_20'] = self.ta.vwma(df['Close'], df['Volume'], length=20)
        df['kama'] = self.ta.kama(df['Close'], length=10)
        
        # Ichimoku Cloud
        ichimoku = self.ta.ichimoku(df['High'], df['Low'], df['Close'])
        if ichimoku is not None and len(ichimoku) > 0:
            for col in ichimoku[0].columns:
                df[f'ichimoku_{col}'] = ichimoku[0][col]
        
        # ===== MOMENTUM OSCILLATORS =====
        df['rsi_14'] = self.ta.rsi(df['Close'], length=14)
        df['rsi_7'] = self.ta.rsi(df['Close'], length=7)
        
        stoch = self.ta.stoch(df['High'], df['Low'], df['Close'])
        if stoch is not None:
            for col in stoch.columns:
                df[f'stoch_{col}'] = stoch[col]
        
        stochrsi = self.ta.stochrsi(df['Close'])
        if stochrsi is not None:
            for col in stochrsi.columns:
                df[f'stochrsi_{col}'] = stochrsi[col]
        
        macd = self.ta.macd(df['Close'])
        if macd is not None:
            for col in macd.columns:
                df[f'macd_{col}'] = macd[col]
        
        df['willr'] = self.ta.willr(df['High'], df['Low'], df['Close'])
        df['cci'] = self.ta.cci(df['High'], df['Low'], df['Close'])
        
        df['roc_10'] = self.ta.roc(df['Close'], length=10)
        df['roc_20'] = self.ta.roc(df['Close'], length=20)
        
        ppo = self.ta.ppo(df['Close'])
        if ppo is not None:
            if isinstance(ppo, pd.DataFrame):
                for col in ppo.columns:
                    df[f'ppo_{col}'] = ppo[col]
            else:
                df['ppo'] = ppo
        
        trix = self.ta.trix(df['Close'])
        if trix is not None:
            if isinstance(trix, pd.DataFrame):
                for col in trix.columns:
                    df[f'trix_{col}'] = trix[col]
            else:
                df['trix'] = trix
        
        df['ao'] = self.ta.ao(df['High'], df['Low'])
        
        aroon = self.ta.aroon(df['High'], df['Low'])
        if aroon is not None:
            for col in aroon.columns:
                df[f'aroon_{col}'] = aroon[col]
        
        df['coppock'] = self.ta.coppock(df['Close'])
        
        kst = self.ta.kst(df['Close'])
        if kst is not None:
            for col in kst.columns:
                df[f'kst_{col}'] = kst[col]
        
        df['psl'] = (df['Close'] > df['Close'].shift(1)).rolling(12).mean() * 100
        
        # ===== VOLUME INDICATORS =====
        df['mfi'] = self.ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'])
        df['bop'] = self.ta.bop(df['Open'], df['High'], df['Low'], df['Close'])
        
        pvo = self.ta.pvo(df['Volume'])
        if pvo is not None:
            if isinstance(pvo, pd.DataFrame):
                for col in pvo.columns:
                    df[f'pvo_{col}'] = pvo[col]
            else:
                df['pvo'] = pvo
        
        df['obv'] = self.ta.obv(df['Close'], df['Volume'])
        
        df['volume_sma_20'] = self.ta.sma(df['Volume'], length=20)
        df['volume_ratio'] = df['Volume'] / df['volume_sma_20'].replace(0, np.nan)
        
        df['rvgi'] = (df['Close'] - df['Open']) / (df['High'] - df['Low']).replace(0, np.nan)
        
        # ===== VOLATILITY INDICATORS =====
        bbands = self.ta.bbands(df['Close'])
        if bbands is not None:
            for col in bbands.columns:
                df[f'bb_{col}'] = bbands[col]
        
        df['atr'] = self.ta.atr(df['High'], df['Low'], df['Close'])
        df['atr_pct'] = df['atr'] / df['Close'] * 100
        
        df['true_range'] = self.ta.true_range(df['High'], df['Low'], df['Close'])
        df['mstd_20'] = self.ta.stdev(df['Close'], length=20)
        df['chop'] = self.ta.chop(df['High'], df['Low'], df['Close'])
        
        # ===== TREND STRENGTH =====
        adx = self.ta.adx(df['High'], df['Low'], df['Close'])
        if adx is not None:
            for col in adx.columns:
                df[f'adx_{col}'] = adx[col]
        
        df['bull_power'] = df['High'] - self.ta.ema(df['Close'], length=13)
        df['bear_power'] = df['Low'] - self.ta.ema(df['Close'], length=13)
        
        # ===== PRICE FEATURES =====
        df['return_1d'] = df['Close'].pct_change()
        df['return_5d'] = df['Close'].pct_change(5)
        df['return_10d'] = df['Close'].pct_change(10)
        df['return_20d'] = df['Close'].pct_change(20)
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['high_low_pct'] = (df['Close'] - df['Low']) / (df['High'] - df['Low']).replace(0, np.nan)
        df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['intraday_range'] = (df['High'] - df['Low']) / df['Open']
        
        # Drop intermediate columns
        if 'volume_sma_20' in df.columns:
            df = df.drop(columns=['volume_sma_20'])
        
        return df
    
    def _compute_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback when pandas_ta is not available."""
        logger.warning("Using basic indicators only")
        
        df['sma_5'] = df['Close'].rolling(5).mean()
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['sma_50'] = df['Close'].rolling(50).mean()
        
        df['ema_12'] = df['Close'].ewm(span=12).mean()
        df['ema_26'] = df['Close'].ewm(span=26).mean()
        
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        df['bb_mid'] = df['Close'].rolling(20).mean()
        df['bb_std'] = df['Close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        df['bb_pctb'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['return_1d'] = df['Close'].pct_change()
        df['return_5d'] = df['Close'].pct_change(5)
        
        return df
    
    def get_indicator_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of indicator columns (exclude OHLCV and Date)."""
        exclude = ['Open', 'High', 'Low', 'Close', 'Volume', 'Date', 'date', 
                   'Adj Close', 'Dividends', 'Stock Splits', 'ticker', 'Ticker']
        return [col for col in df.columns if col not in exclude]


class FeatureSelector:
    """Select top features using Random Forest importance."""
    
    def __init__(self, n_features: int = 12):
        self.n_features = n_features
        self.selected_features = None
        self.importance_df = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> List[str]:
        """Select top features using Random Forest."""
        logger.info(f"Running Random Forest feature selection on {len(feature_names)} features...")
        
        # Use last timestep for RF (samples, features)
        X_flat = X[:, -1, :]
        
        # Subsample for speed if too large
        if len(X_flat) > 50000:
            idx = np.random.choice(len(X_flat), 50000, replace=False)
            X_sample = X_flat[idx]
            y_sample = y[idx]
        else:
            X_sample = X_flat
            y_sample = y
        
        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_sample, y_sample)
        
        # Get importance
        importance = rf.feature_importances_
        
        # Create DataFrame
        self.importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Select top N
        self.selected_features = self.importance_df.head(self.n_features)['feature'].tolist()
        
        logger.info(f"Top {len(self.selected_features)} selected features:")
        for _, row in self.importance_df.head(self.n_features).iterrows():
            logger.info(f"  {row['feature']:30s} importance={row['importance']:.4f}")
        
        return self.selected_features
    
    def transform(self, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Filter X to selected features only."""
        if self.selected_features is None:
            raise ValueError("Must call fit() first")
        
        indices = [feature_names.index(f) for f in self.selected_features]
        X_selected = X[:, :, indices]
        return X_selected, self.selected_features


class PooledExperimentV3:
    """Pooled experiment with Attention LSTM and feature selection comparison."""
    
    def __init__(
        self,
        stocks: List[str] = None,
        start_date: str = "2014-01-01",
        end_date: str = "2024-12-31",
        sequence_length: int = 20,
        epochs: int = 100,
        batch_size: int = 1024,
        early_stopping_patience: int = 50,
    ):
        self.stocks = stocks or TOP_200_TICKERS
        self.start_date = start_date
        self.end_date = end_date
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        
        self.output_dir = REPORTS_DIR / f"pooled_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.indicator_computer = ComprehensiveIndicators()
        
    def load_all_stocks(self) -> Dict[str, pd.DataFrame]:
        """Load price data for all stocks."""
        logger.info(f"Loading {len(self.stocks)} stocks from yfinance...")
        
        data = yf.download(
            self.stocks,
            start=self.start_date,
            end=self.end_date,
            group_by="ticker",
            threads=True,
            progress=True,
        )
        
        stock_data = {}
        for ticker in tqdm(self.stocks, desc="Processing stocks"):
            try:
                if len(self.stocks) == 1:
                    df = data.copy()
                else:
                    df = data[ticker].copy()
                
                df = df.dropna()
                if len(df) < 100:
                    continue
                
                df.columns = [c.capitalize() for c in df.columns]
                df['Ticker'] = ticker
                stock_data[ticker] = df
                
            except Exception as e:
                logger.warning(f"{ticker} failed: {e}")
        
        logger.info(f"Successfully loaded {len(stock_data)} stocks")
        return stock_data
    
    def prepare_stock_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for a single stock."""
        df = self.indicator_computer.compute_all(df)
        feature_cols = self.indicator_computer.get_indicator_columns(df)
        
        # Labels
        df['return_next'] = df['Close'].pct_change().shift(-1)
        df['trend'] = pd.cut(
            df['return_next'],
            bins=[-np.inf, -0.005, 0.005, np.inf],
            labels=[0, 1, 2]
        ).astype(float)
        
        df = df.dropna()
        if len(df) < self.sequence_length + 10:
            return np.array([]), np.array([]), []
        
        # Normalize
        scaler = StandardScaler()
        feature_data = df[feature_cols].values
        feature_data = scaler.fit_transform(feature_data)
        feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Sequences
        labels = df['trend'].values
        X, y = [], []
        for i in range(len(feature_data) - self.sequence_length):
            X.append(feature_data[i:i + self.sequence_length])
            y.append(labels[i + self.sequence_length])
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), feature_cols
    
    def prepare_pooled_data(self, stock_data: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare and pool data from all stocks."""
        all_X, all_y = [], []
        feature_cols = None
        
        for ticker, df in tqdm(stock_data.items(), desc="Preparing features"):
            try:
                X, y, cols = self.prepare_stock_data(df)
                if len(X) == 0:
                    continue
                all_X.append(X)
                all_y.append(y)
                if feature_cols is None:
                    feature_cols = cols
            except Exception as e:
                logger.warning(f"Failed to prepare {ticker}: {e}")
        
        X_pooled = np.concatenate(all_X, axis=0)
        y_pooled = np.concatenate(all_y, axis=0)
        
        logger.info(f"Pooled data: {X_pooled.shape[0]} samples, {X_pooled.shape[2]} features")
        return X_pooled, y_pooled, feature_cols
    
    def train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        experiment_name: str,
    ) -> Dict:
        """Train and evaluate Attention LSTM model."""
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # Split
        n = len(X)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Use Attention LSTM (best from v1)
        model = AttentionLSTM(
            input_size=X.shape[-1],
            hidden_size=lstm_config.hidden_size,
            num_layers=lstm_config.num_layers,
            dropout=lstm_config.dropout,
        )
        
        n_params = sum(p.numel() for p in model.parameters())
        ratio = len(X_train) / n_params
        logger.info(f"Model params: {n_params:,}, Samples/params: {ratio:.2f}x")
        
        # Training
        loss_fn = nn.CrossEntropyLoss()
        trainer = Trainer(model, loss_fn, learning_rate=1e-3, weight_decay=1e-4)
        
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
            batch_size=self.batch_size,
        )
        
        # HIGH patience to let training continue longer
        training_config.early_stopping_patience = self.early_stopping_patience
        history = trainer.train(train_loader, val_loader, epochs=self.epochs)
        
        # Evaluate
        model.eval()
        device = next(model.parameters()).device
        
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            predictions = model.predict(X_test_tensor)
        
        pred_classes = predictions["trend_class"].cpu().numpy()
        accuracy = (pred_classes == y_test).mean()
        
        # Per-class
        class_acc = {}
        for cls in [0, 1, 2]:
            mask = y_test == cls
            if mask.sum() > 0:
                class_acc[f"class_{cls}_acc"] = float((pred_classes[mask] == cls).mean())
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(y_test, pred_classes)
        
        logger.info(f"\n{experiment_name} Test Accuracy: {accuracy:.2%}")
        logger.info(f"Per-class: {class_acc}")
        logger.info(f"\nClassification Report:\n{classification_report(y_test, pred_classes, target_names=['Down', 'Neutral', 'Up'])}")
        
        return {
            "experiment": experiment_name,
            "model": "attention_lstm",
            "n_features": X.shape[-1],
            "n_params": n_params,
            "samples_per_param": ratio,
            "total_samples": len(X),
            "test_accuracy": float(accuracy),
            **class_acc,
            "confusion_matrix": cm.tolist(),
            "best_val_loss": trainer.best_val_loss,
            "epochs_trained": len(history.get("history", [])),
        }
    
    def run(self):
        """Run both experiments: Full features vs Selected features."""
        logger.info("=" * 80)
        logger.info("POOLED BASELINE V3: Attention LSTM + Feature Selection")
        logger.info(f"Stocks: {len(self.stocks)}, Epochs: {self.epochs}, Patience: {self.early_stopping_patience}")
        logger.info("=" * 80)
        
        # Load data
        stock_data = self.load_all_stocks()
        X_full, y_full, feature_cols = self.prepare_pooled_data(stock_data)
        
        logger.info(f"\nFull dataset: {len(X_full)} samples, {len(feature_cols)} features")
        logger.info(f"Feature columns: {feature_cols[:10]}... (truncated)")
        
        # Save feature list
        with open(self.output_dir / "all_features.txt", "w") as f:
            for col in feature_cols:
                f.write(f"{col}\n")
        
        results = []
        
        # ========== Experiment 1: Full indicators ==========
        logger.info("\n" + "=" * 60)
        logger.info(f"EXPERIMENT 1: Full Indicators ({len(feature_cols)} features)")
        logger.info("=" * 60)
        result = self.train_model(X_full, y_full, "full_indicators")
        results.append(result)
        
        # ========== Feature Selection ==========
        logger.info("\n" + "=" * 60)
        logger.info("FEATURE SELECTION: Random Forest Top 12")
        logger.info("=" * 60)
        selector = FeatureSelector(n_features=12)
        selector.fit(X_full, y_full, feature_cols)
        X_selected, selected_cols = selector.transform(X_full, feature_cols)
        
        # Save feature importance
        selector.importance_df.to_csv(self.output_dir / "feature_importance.csv", index=False)
        
        # ========== Experiment 2: Selected indicators ==========
        logger.info("\n" + "=" * 60)
        logger.info("EXPERIMENT 2: RF-Selected 12 Features")
        logger.info("=" * 60)
        result = self.train_model(X_selected, y_full, "rf_selected_12")
        result["selected_features"] = selected_cols
        results.append(result)
        
        # ========== Summary ==========
        logger.info("\n" + "=" * 80)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("=" * 80)
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_dir / "results.csv", index=False)
        
        summary_cols = ["experiment", "n_features", "n_params", "test_accuracy", "epochs_trained"]
        print("\n")
        print(results_df[summary_cols].to_string())
        
        best = results_df.loc[results_df["test_accuracy"].idxmax()]
        logger.info(f"\nBest: {best['experiment']} with {best['test_accuracy']:.2%} accuracy")
        logger.info(f"Results saved to: {self.output_dir}")
        
        return results_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--stocks", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--start", type=str, default="2014-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    args = parser.parse_args()
    
    stocks = TOP_200_TICKERS[:args.stocks]
    
    experiment = PooledExperimentV3(
        stocks=stocks,
        start_date=args.start,
        end_date=args.end,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping_patience=args.patience,
    )
    
    experiment.run()


if __name__ == "__main__":
    main()
