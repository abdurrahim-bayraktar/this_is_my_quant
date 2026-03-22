"""
Price-Only Baseline Experiment with Comprehensive Indicators.

This experiment:
1. Loads 10 years of price data (no sentiment dependency)
2. Computes ALL 40+ technical indicators
3. Uses Random Forest for feature importance ranking
4. Trains LSTM with selected top features
5. Compares current model vs reduced model

Usage:
    python experiments/price_only_baseline.py --stock MSFT --epochs 50
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from config import REPORTS_DIR, training_config, lstm_config
from src.models import BaselineLSTM, AttentionLSTM
from src.training import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ComprehensiveIndicators:
    """
    Compute all 40+ technical indicators from the user's specification.
    
    Uses pandas-ta library for most indicators.
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
                # Try lowercase
                if col.lower() in df.columns:
                    df[col] = df[col.lower()]
                else:
                    raise ValueError(f"Missing required column: {col}")
        
        if not self.ta_available:
            return self._compute_basic_indicators(df)
        
        # ===== MOVING AVERAGES & TREND =====
        # SMA
        df['sma_5'] = self.ta.sma(df['Close'], length=5)
        df['sma_10'] = self.ta.sma(df['Close'], length=10)
        df['sma_20'] = self.ta.sma(df['Close'], length=20)
        df['sma_50'] = self.ta.sma(df['Close'], length=50)
        
        # EMA
        df['ema_5'] = self.ta.ema(df['Close'], length=5)
        df['ema_12'] = self.ta.ema(df['Close'], length=12)
        df['ema_26'] = self.ta.ema(df['Close'], length=26)
        
        # VWMA (Volume Weighted Moving Average)
        df['vwma_20'] = self.ta.vwma(df['Close'], df['Volume'], length=20)
        
        # KAMA (Kaufman's Adaptive Moving Average)
        df['kama'] = self.ta.kama(df['Close'], length=10)
        
        # Ichimoku Cloud
        ichimoku = self.ta.ichimoku(df['High'], df['Low'], df['Close'])
        if ichimoku is not None and len(ichimoku) > 0:
            for col in ichimoku[0].columns:
                df[f'ichimoku_{col}'] = ichimoku[0][col]
        
        # ===== MOMENTUM OSCILLATORS =====
        # RSI
        df['rsi_14'] = self.ta.rsi(df['Close'], length=14)
        df['rsi_7'] = self.ta.rsi(df['Close'], length=7)
        
        # Stochastic Oscillator (KDJ)
        stoch = self.ta.stoch(df['High'], df['Low'], df['Close'])
        if stoch is not None:
            for col in stoch.columns:
                df[f'stoch_{col}'] = stoch[col]
        
        # Stochastic RSI
        stochrsi = self.ta.stochrsi(df['Close'])
        if stochrsi is not None:
            for col in stochrsi.columns:
                df[f'stochrsi_{col}'] = stochrsi[col]
        
        # MACD
        macd = self.ta.macd(df['Close'])
        if macd is not None:
            for col in macd.columns:
                df[f'macd_{col}'] = macd[col]
        
        # Williams %R
        df['willr'] = self.ta.willr(df['High'], df['Low'], df['Close'])
        
        # CCI (Commodity Channel Index)
        df['cci'] = self.ta.cci(df['High'], df['Low'], df['Close'])
        
        # ROC (Rate of Change)
        df['roc_10'] = self.ta.roc(df['Close'], length=10)
        df['roc_20'] = self.ta.roc(df['Close'], length=20)
        
        # PPO (Percentage Price Oscillator) - returns multiple columns
        ppo = self.ta.ppo(df['Close'])
        if ppo is not None:
            if isinstance(ppo, pd.DataFrame):
                for col in ppo.columns:
                    df[f'ppo_{col}'] = ppo[col]
            else:
                df['ppo'] = ppo
        
        # TRIX - may return multiple columns
        trix = self.ta.trix(df['Close'])
        if trix is not None:
            if isinstance(trix, pd.DataFrame):
                for col in trix.columns:
                    df[f'trix_{col}'] = trix[col]
            else:
                df['trix'] = trix
        
        # Awesome Oscillator
        df['ao'] = self.ta.ao(df['High'], df['Low'])
        
        # Aroon
        aroon = self.ta.aroon(df['High'], df['Low'])
        if aroon is not None:
            for col in aroon.columns:
                df[f'aroon_{col}'] = aroon[col]
        
        # Coppock Curve
        df['coppock'] = self.ta.coppock(df['Close'])
        
        # KST (Know Sure Thing)
        kst = self.ta.kst(df['Close'])
        if kst is not None:
            for col in kst.columns:
                df[f'kst_{col}'] = kst[col]
        
        # PSL (Psychological Line) - custom implementation
        df['psl'] = (df['Close'] > df['Close'].shift(1)).rolling(12).mean() * 100
        
        # ===== VOLUME INDICATORS =====
        # MFI (Money Flow Index)
        df['mfi'] = self.ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # BOP (Balance of Power)
        df['bop'] = self.ta.bop(df['Open'], df['High'], df['Low'], df['Close'])
        
        # PVO (Percentage Volume Oscillator) - may return multiple columns
        pvo = self.ta.pvo(df['Volume'])
        if pvo is not None:
            if isinstance(pvo, pd.DataFrame):
                for col in pvo.columns:
                    df[f'pvo_{col}'] = pvo[col]
            else:
                df['pvo'] = pvo
        
        # OBV (On Balance Volume)
        df['obv'] = self.ta.obv(df['Close'], df['Volume'])
        
        # Volume Ratio (Volume / SMA Volume)
        df['volume_sma_20'] = self.ta.sma(df['Volume'], length=20)
        df['volume_ratio'] = df['Volume'] / df['volume_sma_20'].replace(0, np.nan)
        
        # RVGI (Relative Vigor Index) - approximation
        df['rvgi'] = (df['Close'] - df['Open']) / (df['High'] - df['Low']).replace(0, np.nan)
        
        # ===== VOLATILITY INDICATORS =====
        # Bollinger Bands
        bbands = self.ta.bbands(df['Close'])
        if bbands is not None:
            for col in bbands.columns:
                df[f'bb_{col}'] = bbands[col]
        
        # ATR (Average True Range)
        df['atr'] = self.ta.atr(df['High'], df['Low'], df['Close'])
        df['atr_pct'] = df['atr'] / df['Close'] * 100  # Normalized
        
        # True Range
        df['true_range'] = self.ta.true_range(df['High'], df['Low'], df['Close'])
        
        # Standard Deviation
        df['mstd_20'] = self.ta.stdev(df['Close'], length=20)
        
        # Choppiness Index
        df['chop'] = self.ta.chop(df['High'], df['Low'], df['Close'])
        
        # Kaufman's Efficiency Ratio
        df['ker'] = self.ta.kama(df['Close'], length=10)  # KER is internal to KAMA
        
        # ===== TREND STRENGTH =====
        # ADX and DMI
        adx = self.ta.adx(df['High'], df['Low'], df['Close'])
        if adx is not None:
            for col in adx.columns:
                df[f'adx_{col}'] = adx[col]
        
        # Elder-Ray
        df['bull_power'] = df['High'] - self.ta.ema(df['Close'], length=13)
        df['bear_power'] = df['Low'] - self.ta.ema(df['Close'], length=13)
        
        # ===== PRICE FEATURES =====
        # Returns
        df['return_1d'] = df['Close'].pct_change()
        df['return_5d'] = df['Close'].pct_change(5)
        df['return_10d'] = df['Close'].pct_change(10)
        df['return_20d'] = df['Close'].pct_change(20)
        
        # Log returns
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Price position in range
        df['high_low_pct'] = (df['Close'] - df['Low']) / (df['High'] - df['Low']).replace(0, np.nan)
        
        # Gap
        df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        
        # Intraday range
        df['intraday_range'] = (df['High'] - df['Low']) / df['Open']
        
        # Drop intermediate columns
        if 'volume_sma_20' in df.columns:
            df = df.drop(columns=['volume_sma_20'])
        
        return df
    
    def _compute_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback when pandas_ta is not available."""
        logger.warning("Using basic indicators only")
        
        # SMA
        df['sma_5'] = df['Close'].rolling(5).mean()
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['sma_50'] = df['Close'].rolling(50).mean()
        
        # EMA
        df['ema_12'] = df['Close'].ewm(span=12).mean()
        df['ema_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_mid'] = df['Close'].rolling(20).mean()
        df['bb_std'] = df['Close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        df['bb_pctb'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        # Volume ratio
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # Returns
        df['return_1d'] = df['Close'].pct_change()
        df['return_5d'] = df['Close'].pct_change(5)
        
        return df
    
    def get_indicator_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of indicator columns (exclude OHLCV and Date)."""
        exclude = ['Open', 'High', 'Low', 'Close', 'Volume', 'Date', 'date', 
                   'Adj Close', 'Dividends', 'Stock Splits', 'ticker']
        return [col for col in df.columns if col not in exclude]


class FeatureSelector:
    """
    Feature selection using Random Forest importance and mutual information.
    """
    
    def __init__(self, n_top_features: int = 15):
        self.n_top_features = n_top_features
        self.selected_features = None
        self.feature_importance = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> List[str]:
        """
        Select top features using Random Forest importance.
        
        Args:
            X: Feature matrix (samples, features) - use last timestep for RF
            y: Target labels
            feature_names: List of feature names
            
        Returns:
            List of selected feature names
        """
        logger.info(f"Running feature selection on {len(feature_names)} features...")
        
        # Random Forest feature importance
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X, y)
        rf_importance = rf.feature_importances_
        
        # Mutual Information
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        # Combine scores (normalized)
        rf_norm = rf_importance / rf_importance.max()
        mi_norm = mi_scores / (mi_scores.max() + 1e-10)
        combined_score = 0.6 * rf_norm + 0.4 * mi_norm
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'rf_importance': rf_importance,
            'mi_score': mi_scores,
            'combined_score': combined_score
        }).sort_values('combined_score', ascending=False)
        
        self.feature_importance = importance_df
        
        # Select top features
        self.selected_features = importance_df.head(self.n_top_features)['feature'].tolist()
        
        logger.info(f"Selected {len(self.selected_features)} features:")
        for i, row in importance_df.head(self.n_top_features).iterrows():
            logger.info(f"  {row['feature']:30s} RF={row['rf_importance']:.4f} MI={row['mi_score']:.4f}")
        
        return self.selected_features
    
    def plot_importance(self, save_path: Path = None):
        """Plot feature importance."""
        if self.feature_importance is None:
            return
        
        top_20 = self.feature_importance.head(20)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(len(top_20))
        
        ax.barh(y_pos, top_20['combined_score'], align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_20['feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Combined Importance Score')
        ax.set_title('Top 20 Feature Importance (RF + MI)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"Saved importance plot to {save_path}")
        else:
            plt.show()
        
        plt.close()


class PriceOnlyExperiment:
    """
    Price-only baseline experiment with full indicators.
    """
    
    def __init__(
        self,
        stock: str = "MSFT",
        start_date: str = "2015-01-01",
        end_date: str = "2025-01-01",
        sequence_length: int = 20,
        epochs: int = 50,
    ):
        self.stock = stock
        self.start_date = start_date
        self.end_date = end_date
        self.sequence_length = sequence_length
        self.epochs = epochs
        
        self.output_dir = REPORTS_DIR / f"price_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.indicator_computer = ComprehensiveIndicators()
        self.feature_selector = FeatureSelector(n_top_features=15)
        
    def load_data(self) -> pd.DataFrame:
        """Load price data from yfinance."""
        logger.info(f"Loading {self.stock} data from {self.start_date} to {self.end_date}")
        
        stock = yf.Ticker(self.stock)
        df = stock.history(start=self.start_date, end=self.end_date)
        
        logger.info(f"Loaded {len(df)} trading days")
        return df
    
    def prepare_data(self, df: pd.DataFrame, selected_features: List[str] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for training."""
        
        # Compute indicators
        logger.info("Computing technical indicators...")
        df = self.indicator_computer.compute_all(df)
        
        # Get feature columns
        feature_cols = self.indicator_computer.get_indicator_columns(df)
        logger.info(f"Computed {len(feature_cols)} indicator features")
        
        # Label trends (3-class: Down=0, Neutral=1, Up=2)
        df['return_next'] = df['Close'].pct_change().shift(-1)
        df['trend'] = pd.cut(
            df['return_next'],
            bins=[-np.inf, -0.005, 0.005, np.inf],
            labels=[0, 1, 2]
        ).astype(float)
        
        # Drop rows with NaN
        df = df.dropna()
        logger.info(f"After dropping NaN: {len(df)} rows")
        
        # If we have selected features, use those
        if selected_features:
            feature_cols = [f for f in selected_features if f in df.columns]
        
        # Normalize features (per-stock)
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        
        # Create sequences
        X, y = self._create_sequences(df, feature_cols)
        
        return X, y, feature_cols
    
    def _create_sequences(self, df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM."""
        data = df[feature_cols].values
        labels = df['trend'].values
        
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(labels[i + self.sequence_length])
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)
    
    def run_feature_selection(self, X: np.ndarray, y: np.ndarray, feature_cols: List[str]) -> List[str]:
        """Run feature selection on data."""
        # Use last timestep features for RF
        X_last = X[:, -1, :]
        
        selected = self.feature_selector.fit(X_last, y, feature_cols)
        self.feature_selector.plot_importance(self.output_dir / "feature_importance.png")
        
        # Save importance to CSV
        self.feature_selector.feature_importance.to_csv(
            self.output_dir / "feature_importance.csv", 
            index=False
        )
        
        return selected
    
    def train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_size: str = "current",
        use_attention: bool = False,
    ) -> Dict:
        """Train and evaluate model."""
        
        # Split data (80% train, 10% val, 10% test)
        n = len(X)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Model configuration
        if model_size == "current":
            hidden_size = 128
            num_layers = 2
            dropout = 0.1
        elif model_size == "reduced":
            hidden_size = 64
            num_layers = 1
            dropout = 0.3
        else:
            raise ValueError(f"Unknown model size: {model_size}")
        
        # Create model
        if use_attention:
            model = AttentionLSTM(
                input_size=X.shape[-1],
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
        else:
            model = BaselineLSTM(
                input_size=X.shape[-1],
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )
        
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model: {model_size}, params={n_params:,}, samples/params={len(X_train)/n_params:.1f}x")
        
        # Training
        loss_fn = nn.CrossEntropyLoss()
        trainer = Trainer(model, loss_fn, learning_rate=5e-4, weight_decay=1e-3)
        
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
            batch_size=64,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
            batch_size=64,
        )
        
        # Enable early stopping
        training_config.early_stopping_patience = 15
        
        history = trainer.train(train_loader, val_loader, epochs=self.epochs)
        
        # Evaluate on test set
        model.eval()
        device = next(model.parameters()).device
        
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            predictions = model.predict(X_test_tensor)
        
        pred_classes = predictions["trend_class"].cpu().numpy()
        
        # Metrics
        accuracy = (pred_classes == y_test).mean()
        
        # Per-class accuracy
        class_acc = {}
        for cls in [0, 1, 2]:
            mask = y_test == cls
            if mask.sum() > 0:
                class_acc[f"class_{cls}_acc"] = (pred_classes[mask] == cls).mean()
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, pred_classes)
        
        results = {
            "model_size": model_size,
            "use_attention": use_attention,
            "n_params": n_params,
            "train_samples": len(X_train),
            "test_accuracy": float(accuracy),
            **{k: float(v) for k, v in class_acc.items()},
            "confusion_matrix": cm.tolist(),
            "best_val_loss": trainer.best_val_loss,
        }
        
        logger.info(f"Test Accuracy: {accuracy:.2%}")
        logger.info(f"Per-class: {class_acc}")
        
        return results
    
    def run(self):
        """Run the full experiment."""
        logger.info("=" * 80)
        logger.info(f"PRICE-ONLY BASELINE EXPERIMENT: {self.stock}")
        logger.info("=" * 80)
        
        # Load and prepare data
        df = self.load_data()
        X_full, y_full, all_features = self.prepare_data(df)
        
        logger.info(f"Full dataset: {X_full.shape[0]} samples, {X_full.shape[2]} features")
        
        # Feature selection
        logger.info("\n" + "=" * 40)
        logger.info("PHASE 1: Feature Selection")
        logger.info("=" * 40)
        selected_features = self.run_feature_selection(X_full, y_full, all_features)
        
        # Prepare data with selected features
        df = self.load_data()
        X_selected, y_selected, _ = self.prepare_data(df, selected_features)
        
        logger.info(f"\nSelected features dataset: {X_selected.shape[0]} samples, {X_selected.shape[2]} features")
        
        results = []
        
        # Experiment 1: Current model with all features
        logger.info("\n" + "=" * 40)
        logger.info("EXPERIMENT 1: Current Model (128h, 2L) + All Features")
        logger.info("=" * 40)
        result = self.train_model(X_full, y_full, model_size="current")
        result["experiment"] = "current_all_features"
        result["n_features"] = X_full.shape[2]
        results.append(result)
        
        # Experiment 2: Current model with selected features
        logger.info("\n" + "=" * 40)
        logger.info("EXPERIMENT 2: Current Model (128h, 2L) + Selected Features")
        logger.info("=" * 40)
        result = self.train_model(X_selected, y_selected, model_size="current")
        result["experiment"] = "current_selected_features"
        result["n_features"] = X_selected.shape[2]
        results.append(result)
        
        # Experiment 3: Reduced model with selected features
        logger.info("\n" + "=" * 40)
        logger.info("EXPERIMENT 3: Reduced Model (64h, 1L) + Selected Features")
        logger.info("=" * 40)
        result = self.train_model(X_selected, y_selected, model_size="reduced")
        result["experiment"] = "reduced_selected_features"
        result["n_features"] = X_selected.shape[2]
        results.append(result)
        
        # Experiment 4: Current model with attention
        logger.info("\n" + "=" * 40)
        logger.info("EXPERIMENT 4: Attention Model + Selected Features")
        logger.info("=" * 40)
        result = self.train_model(X_selected, y_selected, model_size="current", use_attention=True)
        result["experiment"] = "attention_selected_features"
        result["n_features"] = X_selected.shape[2]
        results.append(result)
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("=" * 80)
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_dir / "results.csv", index=False)
        
        print("\n")
        print(results_df[["experiment", "n_features", "n_params", "test_accuracy"]].to_string())
        
        best = results_df.loc[results_df["test_accuracy"].idxmax()]
        logger.info(f"\nBest: {best['experiment']} with {best['test_accuracy']:.2%} accuracy")
        
        return results_df


def main():
    parser = argparse.ArgumentParser(description="Price-Only Baseline Experiment")
    parser.add_argument("--stock", type=str, default="MSFT", help="Stock ticker")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--start", type=str, default="2015-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2025-01-01", help="End date")
    args = parser.parse_args()
    
    experiment = PriceOnlyExperiment(
        stock=args.stock,
        start_date=args.start,
        end_date=args.end,
        epochs=args.epochs,
    )
    
    experiment.run()


if __name__ == "__main__":
    main()
