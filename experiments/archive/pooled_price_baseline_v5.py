"""
Pooled Price-Only Baseline V5 - Balanced + Batched Predictions.

Changes from V4:
1. Batched predictions (avoid OOM during test)
2. Balanced LR: 5e-4 (between 1e-3 and 1e-4)
3. Balanced dropout: 0.4 (between 0.3 and 0.5)
4. Keep caching and more stocks

Usage:
    python experiments/pooled_price_baseline_v5.py --epochs 100 --stocks 400
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
from tqdm import tqdm

from config import REPORTS_DIR, DATA_DIR, TOP_200_TICKERS, training_config, lstm_config
from src.models import AttentionLSTM
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

# Extended stock list
EXTENDED_TICKERS = TOP_200_TICKERS + [
    "A", "AAL", "AAP", "ABBV", "ABC", "ABMD", "ABT", "ACN", "ADBE", "ADI",
    "ADM", "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", "AIG", "AIV", "AIZ",
    "AJG", "AKAM", "ALB", "ALGN", "ALK", "ALL", "ALLE", "ALXN", "AMAT", "AMCR",
    "AMD", "AME", "AMGN", "AMP", "AMT", "AMZN", "ANET", "ANSS", "ANTM", "AON",
    "AOS", "APA", "APD", "APH", "APTV", "ARE", "ATO", "ATVI", "AVB", "AVGO",
    "AVY", "AWK", "AXP", "AZO", "BA", "BAC", "BAX", "BBY", "BDX", "BEN",
    "BF.B", "BIIB", "BIO", "BK", "BKNG", "BKR", "BLK", "BLL", "BMY", "BR",
    "BRK.B", "BSX", "BWA", "BXP", "C", "CAG", "CAH", "CARR", "CAT", "CB",
    "CBOE", "CBRE", "CCI", "CCL", "CDNS", "CDW", "CE", "CERN", "CF", "CFG",
    "CHD", "CHRW", "CHTR", "CI", "CINF", "CL", "CLX", "CMA", "CMCSA", "CME",
    "CMG", "CMI", "CMS", "CNC", "CNP", "COF", "COG", "COO", "COP", "COST",
    "CPB", "CPRT", "CRM", "CSCO", "CSX", "CTAS", "CTL", "CTSH", "CTVA", "CTXS",
    "CVS", "CVX", "CXO", "D", "DAL", "DD", "DE", "DFS", "DG", "DGX",
    "DHI", "DHR", "DIS", "DISCA", "DISCK", "DISH", "DLR", "DLTR", "DOV", "DOW",
    "DPZ", "DRE", "DRI", "DTE", "DUK", "DVA", "DVN", "DXC", "DXCM", "EA",
    "EBAY", "ECL", "ED", "EFX", "EIX", "EL", "EMN", "EMR", "ENPH", "EOG",
    "EQIX", "EQR", "ES", "ESS", "ETFC", "ETN", "ETR", "EVRG", "EW", "EXC",
    "EXPD", "EXPE", "EXR", "F", "FANG", "FAST", "FB", "FBHS", "FCX", "FDX",
    "FE", "FFIV", "FIS", "FISV", "FITB", "FLIR", "FLS", "FLT", "FMC", "FOX",
    "FOXA", "FRC", "FRT", "FTNT", "FTV", "GD", "GE", "GILD", "GIS", "GL",
]
EXTENDED_TICKERS = list(dict.fromkeys(EXTENDED_TICKERS))


class PriceCache:
    """Cache for yfinance price data."""
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or DATA_DIR / "price_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_path(self, start_date: str, end_date: str) -> Path:
        return self.cache_dir / f"prices_{start_date}_{end_date}.pkl"
    
    def load(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        cache_path = self.get_cache_path(start_date, end_date)
        if cache_path.exists():
            logger.info(f"Loading cached prices from {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        return None
    
    def save(self, data: Dict[str, pd.DataFrame], start_date: str, end_date: str):
        cache_path = self.get_cache_path(start_date, end_date)
        logger.info(f"Saving prices to cache: {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)


class ComprehensiveIndicators:
    """Compute all 40+ technical indicators."""
    
    def __init__(self):
        try:
            import pandas_ta as ta
            self.ta = ta
            self.ta_available = True
        except ImportError:
            self.ta_available = False
    
    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required:
            if col not in df.columns:
                if col.lower() in df.columns:
                    df[col] = df[col.lower()]
        
        if not self.ta_available:
            return self._compute_basic_indicators(df)
        
        # Moving Averages
        df['sma_5'] = self.ta.sma(df['Close'], length=5)
        df['sma_10'] = self.ta.sma(df['Close'], length=10)
        df['sma_20'] = self.ta.sma(df['Close'], length=20)
        df['sma_50'] = self.ta.sma(df['Close'], length=50)
        df['ema_5'] = self.ta.ema(df['Close'], length=5)
        df['ema_12'] = self.ta.ema(df['Close'], length=12)
        df['ema_26'] = self.ta.ema(df['Close'], length=26)
        df['vwma_20'] = self.ta.vwma(df['Close'], df['Volume'], length=20)
        df['kama'] = self.ta.kama(df['Close'], length=10)
        
        ichimoku = self.ta.ichimoku(df['High'], df['Low'], df['Close'])
        if ichimoku is not None and len(ichimoku) > 0:
            for col in ichimoku[0].columns:
                df[f'ichimoku_{col}'] = ichimoku[0][col]
        
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
        
        trix = self.ta.trix(df['Close'])
        if trix is not None:
            if isinstance(trix, pd.DataFrame):
                for col in trix.columns:
                    df[f'trix_{col}'] = trix[col]
        
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
        
        df['mfi'] = self.ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'])
        df['bop'] = self.ta.bop(df['Open'], df['High'], df['Low'], df['Close'])
        
        pvo = self.ta.pvo(df['Volume'])
        if pvo is not None:
            if isinstance(pvo, pd.DataFrame):
                for col in pvo.columns:
                    df[f'pvo_{col}'] = pvo[col]
        
        df['obv'] = self.ta.obv(df['Close'], df['Volume'])
        df['volume_sma_20'] = self.ta.sma(df['Volume'], length=20)
        df['volume_ratio'] = df['Volume'] / df['volume_sma_20'].replace(0, np.nan)
        df['rvgi'] = (df['Close'] - df['Open']) / (df['High'] - df['Low']).replace(0, np.nan)
        
        bbands = self.ta.bbands(df['Close'])
        if bbands is not None:
            for col in bbands.columns:
                df[f'bb_{col}'] = bbands[col]
        
        df['atr'] = self.ta.atr(df['High'], df['Low'], df['Close'])
        df['atr_pct'] = df['atr'] / df['Close'] * 100
        df['true_range'] = self.ta.true_range(df['High'], df['Low'], df['Close'])
        df['mstd_20'] = self.ta.stdev(df['Close'], length=20)
        df['chop'] = self.ta.chop(df['High'], df['Low'], df['Close'])
        
        adx = self.ta.adx(df['High'], df['Low'], df['Close'])
        if adx is not None:
            for col in adx.columns:
                df[f'adx_{col}'] = adx[col]
        
        df['bull_power'] = df['High'] - self.ta.ema(df['Close'], length=13)
        df['bear_power'] = df['Low'] - self.ta.ema(df['Close'], length=13)
        
        df['return_1d'] = df['Close'].pct_change()
        df['return_5d'] = df['Close'].pct_change(5)
        df['return_10d'] = df['Close'].pct_change(10)
        df['return_20d'] = df['Close'].pct_change(20)
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['high_low_pct'] = (df['Close'] - df['Low']) / (df['High'] - df['Low']).replace(0, np.nan)
        df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['intraday_range'] = (df['High'] - df['Low']) / df['Open']
        
        if 'volume_sma_20' in df.columns:
            df = df.drop(columns=['volume_sma_20'])
        
        return df
    
    def _compute_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['sma_5'] = df['Close'].rolling(5).mean()
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['ema_12'] = df['Close'].ewm(span=12).mean()
        df['ema_26'] = df['Close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi_14'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
        df['return_1d'] = df['Close'].pct_change()
        return df
    
    def get_indicator_columns(self, df: pd.DataFrame) -> List[str]:
        exclude = ['Open', 'High', 'Low', 'Close', 'Volume', 'Date', 'date', 
                   'Adj Close', 'Dividends', 'Stock Splits', 'ticker', 'Ticker']
        return [col for col in df.columns if col not in exclude]


class PooledExperimentV5:
    """V5: Balanced hyperparameters + batched predictions."""
    
    def __init__(
        self,
        stocks: List[str] = None,
        start_date: str = "2014-01-01",
        end_date: str = "2024-12-31",
        sequence_length: int = 20,
        epochs: int = 100,
        batch_size: int = 1024,
        early_stopping_patience: int = 15,  # V5: Sweet spot
        dropout: float = 0.4,               # V5: Balanced (between 0.3 and 0.5)
        learning_rate: float = 5e-4,        # V5: Balanced (between 1e-3 and 1e-4)
        use_cache: bool = True,
    ):
        self.stocks = stocks or EXTENDED_TICKERS[:400]
        self.start_date = start_date
        self.end_date = end_date
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.use_cache = use_cache
        
        self.output_dir = REPORTS_DIR / f"pooled_v5_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.indicator_computer = ComprehensiveIndicators()
        self.price_cache = PriceCache()
        
    def load_all_stocks(self) -> Dict[str, pd.DataFrame]:
        if self.use_cache:
            cached = self.price_cache.load(self.start_date, self.end_date)
            if cached:
                filtered = {k: v for k, v in cached.items() if k in self.stocks}
                if len(filtered) >= len(self.stocks) * 0.8:
                    logger.info(f"Using {len(filtered)} cached stocks")
                    return filtered
        
        logger.info(f"Downloading {len(self.stocks)} stocks from yfinance...")
        
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
            except:
                pass
        
        logger.info(f"Successfully loaded {len(stock_data)} stocks")
        
        if self.use_cache:
            self.price_cache.save(stock_data, self.start_date, self.end_date)
        
        return stock_data
    
    def prepare_stock_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        df = self.indicator_computer.compute_all(df)
        feature_cols = self.indicator_computer.get_indicator_columns(df)
        
        df['return_next'] = df['Close'].pct_change().shift(-1)
        df['trend'] = pd.cut(
            df['return_next'],
            bins=[-np.inf, -0.005, 0.005, np.inf],
            labels=[0, 1, 2]
        ).astype(float)
        
        df = df.dropna()
        if len(df) < self.sequence_length + 10:
            return np.array([]), np.array([]), []
        
        scaler = StandardScaler()
        feature_data = df[feature_cols].values
        feature_data = scaler.fit_transform(feature_data)
        feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        labels = df['trend'].values
        X, y = [], []
        for i in range(len(feature_data) - self.sequence_length):
            X.append(feature_data[i:i + self.sequence_length])
            y.append(labels[i + self.sequence_length])
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), feature_cols
    
    def prepare_pooled_data(self, stock_data: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
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
            except:
                pass
        
        X_pooled = np.concatenate(all_X, axis=0)
        y_pooled = np.concatenate(all_y, axis=0)
        
        logger.info(f"Pooled data: {X_pooled.shape[0]:,} samples, {X_pooled.shape[2]} features")
        return X_pooled, y_pooled, feature_cols
    
    def predict_batched(self, model: nn.Module, X: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        """Batched prediction to avoid OOM."""
        model.eval()
        device = next(model.parameters()).device
        
        all_preds = []
        
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size]
            with torch.no_grad():
                batch_tensor = torch.FloatTensor(batch).to(device)
                predictions = model.predict(batch_tensor)
                pred_classes = predictions["trend_class"].cpu().numpy()
                all_preds.append(pred_classes)
                
                # Clear CUDA cache after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return np.concatenate(all_preds)
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train with balanced settings and batched eval."""
        
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        n = len(X)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        logger.info(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
        
        # V5: Balanced dropout
        model = AttentionLSTM(
            input_size=X.shape[-1],
            hidden_size=lstm_config.hidden_size,
            num_layers=lstm_config.num_layers,
            dropout=self.dropout,
        )
        
        n_params = sum(p.numel() for p in model.parameters())
        ratio = len(X_train) / n_params
        logger.info(f"Model params: {n_params:,}, Samples/params: {ratio:.2f}x")
        
        loss_fn = nn.CrossEntropyLoss()
        
        # V5: Balanced learning rate
        trainer = Trainer(
            model, loss_fn,
            learning_rate=self.learning_rate,
            weight_decay=1e-4,
        )
        
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
            batch_size=self.batch_size,
        )
        
        training_config.early_stopping_patience = self.early_stopping_patience
        history = trainer.train(train_loader, val_loader, epochs=self.epochs)
        
        # V5: BATCHED PREDICTIONS (avoid OOM)
        logger.info("Evaluating on test set with batched predictions...")
        pred_classes = self.predict_batched(model, X_test, batch_size=2048)
        
        accuracy = (pred_classes == y_test).mean()
        
        class_acc = {}
        for cls in [0, 1, 2]:
            mask = y_test == cls
            if mask.sum() > 0:
                class_acc[f"class_{cls}_acc"] = float((pred_classes[mask] == cls).mean())
        
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(y_test, pred_classes)
        
        logger.info(f"\nTest Accuracy: {accuracy:.2%}")
        logger.info(f"Per-class: {class_acc}")
        logger.info(f"\nClassification Report:\n{classification_report(y_test, pred_classes, target_names=['Down', 'Neutral', 'Up'])}")
        
        return {
            "model": "attention_lstm",
            "n_features": X.shape[-1],
            "n_params": n_params,
            "n_stocks": len(self.stocks),
            "total_samples": len(X),
            "train_samples": len(X_train),
            "samples_per_param": ratio,
            "test_accuracy": float(accuracy),
            **class_acc,
            "confusion_matrix": cm.tolist(),
            "best_val_loss": trainer.best_val_loss,
            "epochs_trained": len(history.get("history", [])),
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "patience": self.early_stopping_patience,
        }
    
    def run(self):
        logger.info("=" * 80)
        logger.info("POOLED BASELINE V5: Balanced + Batched Predictions")
        logger.info("=" * 80)
        logger.info(f"  Stocks: {len(self.stocks)}")
        logger.info(f"  Dropout: {self.dropout} (v4: 0.5)")
        logger.info(f"  Learning Rate: {self.learning_rate} (v4: 1e-4)")
        logger.info(f"  Patience: {self.early_stopping_patience} (v4: 20)")
        logger.info("=" * 80)
        
        stock_data = self.load_all_stocks()
        X, y, feature_cols = self.prepare_pooled_data(stock_data)
        
        logger.info(f"\nDataset: {len(X):,} samples, {len(feature_cols)} features, {len(stock_data)} stocks")
        
        with open(self.output_dir / "features.txt", "w") as f:
            for col in feature_cols:
                f.write(f"{col}\n")
        
        result = self.train_model(X, y)
        
        results_df = pd.DataFrame([result])
        results_df.to_csv(self.output_dir / "results.csv", index=False)
        
        logger.info("\n" + "=" * 80)
        logger.info("V5 RESULTS")
        logger.info("=" * 80)
        logger.info(f"  Test Accuracy: {result['test_accuracy']:.2%}")
        logger.info(f"  Stocks: {result['n_stocks']}")
        logger.info(f"  Samples: {result['total_samples']:,}")
        logger.info(f"  Epochs Trained: {result['epochs_trained']}")
        logger.info(f"  Best Val Loss: {result['best_val_loss']:.4f}")
        logger.info(f"\nResults saved to: {self.output_dir}")
        
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--stocks", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=15)   # V5: 15
    parser.add_argument("--dropout", type=float, default=0.4)  # V5: 0.4
    parser.add_argument("--lr", type=float, default=5e-4)      # V5: 5e-4
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--start", type=str, default="2014-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    args = parser.parse_args()
    
    stocks = EXTENDED_TICKERS[:args.stocks]
    
    experiment = PooledExperimentV5(
        stocks=stocks,
        start_date=args.start,
        end_date=args.end,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping_patience=args.patience,
        dropout=args.dropout,
        learning_rate=args.lr,
        use_cache=not args.no_cache,
    )
    
    experiment.run()


if __name__ == "__main__":
    main()
