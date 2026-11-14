"""
Pooled Price-Only Baseline V6 - 800 Stocks.

Changes from V5:
1. Extended to 800 stocks (vs 316 in V5)
2. Uses centralized ComprehensiveIndicators module
3. Same DataLoader approach as V5 (simple, no pin_memory)
4. Uses existing price cache to avoid disk writes

Usage:
    python experiments/pooled_price_baseline_v6.py --epochs 100 --stocks 800
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
from src.features import ComprehensiveIndicators

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

# Massive stock list - 800+ unique tickers
EXTENDED_TICKERS = list(dict.fromkeys(TOP_200_TICKERS + [
    "A", "AAL", "AAP", "ABBV", "ABC", "ABMD", "ABT", "ACN", "ADBE", "ADI",
    "ADM", "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", "AIG", "AIV", "AIZ",
    "AJG", "AKAM", "ALB", "ALGN", "ALK", "ALL", "ALLE", "AMAT", "AMCR",
    "AMD", "AME", "AMGN", "AMP", "AMT", "AMZN", "ANET", "ANSS", "AON",
    "AOS", "APA", "APD", "APH", "APTV", "ARE", "ATO", "ATVI", "AVB", "AVGO",
    "AVY", "AWK", "AXP", "AZO", "BA", "BAC", "BAX", "BBY", "BDX", "BEN",
    "BIIB", "BIO", "BK", "BKNG", "BKR", "BLK", "BLL", "BMY", "BR",
    "BSX", "BWA", "BXP", "C", "CAG", "CAH", "CARR", "CAT", "CB",
    "CBOE", "CBRE", "CCI", "CCL", "CDNS", "CDW", "CE", "CERN", "CF", "CFG",
    "CHD", "CHRW", "CHTR", "CI", "CINF", "CL", "CLX", "CMA", "CMCSA", "CME",
    "CMG", "CMI", "CMS", "CNC", "CNP", "COF", "COO", "COP", "COST",
    "CPB", "CPRT", "CRM", "CSCO", "CSX", "CTAS", "CTSH", "CTVA",
    "CVS", "CVX", "D", "DAL", "DD", "DE", "DFS", "DG", "DGX",
    "DHI", "DHR", "DIS", "DLR", "DLTR", "DOV", "DOW",
    "DPZ", "DRE", "DRI", "DTE", "DUK", "DVA", "DVN", "DXCM", "EA",
    "EBAY", "ECL", "ED", "EFX", "EIX", "EL", "EMN", "EMR", "ENPH", "EOG",
    "EQIX", "EQR", "ES", "ESS", "ETN", "ETR", "EVRG", "EW", "EXC",
    "EXPD", "EXPE", "EXR", "F", "FANG", "FAST", "FBHS", "FCX", "FDX",
    "FE", "FFIV", "FIS", "FISV", "FITB", "FLT", "FMC", "FOX",
    "FOXA", "FRT", "FTNT", "FTV", "GD", "GE", "GILD", "GIS", "GL",
    "GLW", "GM", "GNRC", "GOOG", "GOOGL", "GPC", "GPN", "GPS", "GRMN", "GS",
    "GWW", "HAL", "HAS", "HBAN", "HBI", "HCA", "HD", "HES", "HIG", "HII",
    "HLT", "HOLX", "HON", "HPE", "HPQ", "HRB", "HRL", "HSIC", "HST", "HSY",
    "HUM", "HWM", "IBM", "ICE", "IDXX", "IEX", "IFF", "ILMN", "INCY", "INTC",
    "INTU", "IP", "IPG", "IPGP", "IQV", "IR", "IRM", "ISRG", "IT", "ITW",
    "IVZ", "J", "JBHT", "JCI", "JKHY", "JNJ", "JNPR", "JPM", "K", "KEY",
    "KEYS", "KHC", "KIM", "KLAC", "KMB", "KMI", "KMX", "KO", "KR", "KSU",
    "L", "LB", "LDOS", "LEG", "LEN", "LH", "LHX", "LIN", "LKQ", "LLY",
    "LMT", "LNC", "LNT", "LOW", "LRCX", "LUMN", "LUV", "LVS", "LW", "LYB",
    "LYV", "MA", "MAA", "MAR", "MAS", "MCD", "MCHP", "MCK", "MCO", "MDLZ",
    "MDT", "MET", "MGM", "MHK", "MKC", "MKTX", "MLM", "MMC", "MMM", "MNST",
    "MO", "MOS", "MPC", "MPWR", "MRK", "MRO", "MS", "MSCI", "MSFT", "MSI",
    "MTB", "MTD", "MU", "NCLH", "NDAQ", "NEE", "NEM", "NFLX", "NI", "NKE",
    "NLOK", "NLSN", "NOC", "NOV", "NOW", "NRG", "NSC", "NTAP", "NTRS", "NUE",
    "NVDA", "NVR", "NWL", "NWS", "NWSA", "O", "ODFL", "OGN", "OKE", "OMC",
    "ORCL", "ORLY", "OTIS", "OXY", "PAYC", "PAYX", "PBCT", "PCAR", "PEAK",
    "PEG", "PENN", "PEP", "PFE", "PFG", "PG", "PGR", "PH", "PHM", "PKG",
    "PKI", "PLD", "PM", "PNC", "PNR", "PNW", "POOL", "PPG", "PPL", "PRGO",
    "PRU", "PSA", "PSX", "PTC", "PVH", "PWR", "PXD", "PYPL", "QCOM", "QRVO",
    "RCL", "RE", "REG", "REGN", "RF", "RHI", "RJF", "RL", "RMD", "ROK",
    "ROL", "ROP", "ROST", "RSG", "RTX", "SBAC", "SBUX", "SCHW", "SEE", "SHW",
    "SIVB", "SJM", "SLB", "SLG", "SNA", "SNPS", "SO", "SPG", "SPGI", "SRE",
    "STE", "STT", "STX", "STZ", "SWK", "SWKS", "SYF", "SYK", "SYY", "T",
    "TAP", "TDG", "TDY", "TEL", "TER", "TFC", "TFX", "TGT", "TJX", "TMO",
    "TMUS", "TPR", "TRMB", "TROW", "TRV", "TSCO", "TSLA", "TSN", "TT", "TTWO",
    "TWTR", "TXN", "TXT", "TYL", "UA", "UAA", "UAL", "UDR", "UHS", "ULTA",
    "UNH", "UNP", "UPS", "URI", "USB", "V", "VFC", "VIAC", "VLO", "VMC",
    "VNO", "VRSK", "VRSN", "VRTX", "VTR", "VTRS", "VZ", "WAB", "WAT", "WBA",
    "WDC", "WEC", "WELL", "WFC", "WHR", "WLTW", "WM", "WMB", "WMT", "WRB",
    "WRK", "WST", "WU", "WY", "WYNN", "XEL", "XLNX", "XOM", "XRAY", "XYL",
    "YUM", "ZBH", "ZBRA", "ZION", "ZTS",
    "PLTR", "COIN", "HOOD", "RIVN", "LCID", "NIO", "XPEV", "LI", "BABA",
    "JD", "PDD", "BILI", "TME", "IQ", "VIPS", "WB", "BIDU", "NTES",
    "SNAP", "PINS", "SPOT", "ROKU", "ZM", "DOCU", "CRWD", "NET", "DDOG",
    "SNOW", "OKTA", "ZS", "PANW", "FTNT", "SPLK", "MDB", "TEAM", "WDAY",
    "NOW", "CRM", "ADBE", "INTU", "SHOP", "SQ", "MELI", "SE", "GRAB",
    "ABNB", "DASH", "UBER", "LYFT", "DKNG", "PENN", "MGM", "WYNN", "LVS",
    "RCL", "CCL", "NCLH", "MAR", "HLT", "H", "IHG", "WH", "CHH",
    "AAL", "DAL", "UAL", "LUV", "ALK", "JBLU", "SAVE", "HA",
    "XOM", "CVX", "COP", "EOG", "PXD", "DVN", "MRO", "APA", "HAL", "SLB",
    "OXY", "FANG", "CLR", "MPC", "VLO", "PSX", "HES", "BKR", "NOV",
    "JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", "TFC", "SCHW",
    "BLK", "BX", "KKR", "APO", "CG", "ARES", "OWL", "TPG",
    "UNH", "JNJ", "PFE", "MRK", "ABBV", "TMO", "ABT", "DHR", "BMY", "LLY",
    "AMGN", "GILD", "REGN", "VRTX", "BIIB", "MRNA", "BNTX", "NVAX",
    "ETSY", "W", "CHWY", "FVRR", "UPWK", "SKLZ", "DKNG", "PENN",
    "APPS", "U", "RBLX", "TTWO", "ATVI", "EA", "ZNGA",
    "MTCH", "BMBL", "IAC", "ANGI", "ZG", "RDFN", "OPEN", "CVNA", "CARG",
    "LMND", "ROOT", "HIPO", "OSCR", "CLVR", "SOFI", "UPST", "AFRM", "BILL",
    "TOST", "FIVN", "RNG", "TWLO", "ASAN", "MNDY", "DOCN", "PATH",
    "AI", "PLTR", "BBAI", "IONQ", "RGTI", "QUBT",
    "TSM", "ASML", "NVO", "TM", "SONY", "SAP", "SHOP", "RY", "TD", "BMO",
    "BNS", "CM", "SU", "CNQ", "ENB", "TRP", "CP", "CNI",
    "BP", "SHEL", "TTE", "EQNR", "E", "YPF", "PBR", "EC", "VALE", "RIO",
    "BHP", "FCX", "NEM", "GOLD", "AEM", "WPM", "PAAS", "AG",
]))


class PriceCache:
    """Cache for yfinance price data - uses existing cache to avoid disk writes."""
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or DATA_DIR / "price_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_path(self, start_date: str, end_date: str) -> Path:
        # Use generic path like V5 for compatibility
        return self.cache_dir / f"prices_{start_date}_{end_date}.pkl"
    
    def load(self, start_date: str, end_date: str, requested_stocks: List[str]) -> Dict[str, pd.DataFrame]:
        # First try exact match
        cache_path = self.get_cache_path(start_date, end_date)
        if cache_path.exists():
            logger.info(f"Loading cached prices from {cache_path}")
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            logger.info(f"Cache contains {len(cached)} stocks")
            return cached
        
        # Try to find any cache for this date range
        for f in self.cache_dir.glob(f"*_{start_date}_{end_date}.pkl"):
            logger.info(f"Found cache file: {f}")
            with open(f, "rb") as file:
                cached = pickle.load(file)
            logger.info(f"Cache contains {len(cached)} stocks")
            return cached
        
        return None
    
    def save(self, data: Dict[str, pd.DataFrame], start_date: str, end_date: str):
        cache_path = self.get_cache_path(start_date, end_date)
        # Only save if file doesn't exist
        if not cache_path.exists():
            logger.info(f"Saving prices to cache: {cache_path}")
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
        else:
            logger.info(f"Cache already exists, skipping write: {cache_path}")


class PooledExperimentV6:
    """V6: 800 stocks, same approach as V5."""
    
    def __init__(
        self,
        stocks: List[str] = None,
        start_date: str = "2014-01-01",
        end_date: str = "2024-12-31",
        sequence_length: int = 20,
        epochs: int = 100,
        batch_size: int = 1024,  # Same as V5
        early_stopping_patience: int = 15,
        dropout: float = 0.4,
        learning_rate: float = 5e-4,
        use_cache: bool = True,
    ):
        self.stocks = stocks or EXTENDED_TICKERS[:800]
        self.start_date = start_date
        self.end_date = end_date
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.use_cache = use_cache
        
        self.output_dir = REPORTS_DIR / f"pooled_v6_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.indicator_computer = ComprehensiveIndicators()
        self.price_cache = PriceCache()
        
    def load_all_stocks(self) -> Dict[str, pd.DataFrame]:
        """Load stocks: use cache + download only missing stocks."""
        stock_data = {}
        
        # Step 1: Load from cache first
        if self.use_cache:
            cached = self.price_cache.load(self.start_date, self.end_date, self.stocks)
            if cached:
                # Take all cached stocks that are in our target list
                for ticker, df in cached.items():
                    if ticker in self.stocks:
                        stock_data[ticker] = df
                logger.info(f"Loaded {len(stock_data)} stocks from cache")
        
        # Step 2: Find missing stocks
        missing = [t for t in self.stocks if t not in stock_data]
        logger.info(f"Need to download {len(missing)} additional stocks")
        
        # Step 3: Download only missing stocks
        if missing:
            batch_size = 100
            for i in range(0, len(missing), batch_size):
                batch = missing[i:i + batch_size]
                logger.info(f"Downloading batch {i//batch_size + 1}/{(len(missing)-1)//batch_size + 1} ({len(batch)} stocks)")
                
                try:
                    data = yf.download(
                        batch,
                        start=self.start_date,
                        end=self.end_date,
                        group_by="ticker",
                        threads=True,
                        progress=True,
                    )
                    
                    for ticker in batch:
                        try:
                            if len(batch) == 1:
                                df = data.copy()
                            else:
                                df = data[ticker].copy()
                            
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
            
            # Save merged data back to cache
            if self.use_cache and len(stock_data) > 0:
                self.price_cache.save(stock_data, self.start_date, self.end_date)
        
        logger.info(f"Total: {len(stock_data)} stocks ready")
        return stock_data
    
    def prepare_stock_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        df = self.indicator_computer.compute_all(df, exclude=['ichimoku_ICS_26'])
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
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return np.concatenate(all_preds)
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train - EXACTLY like V5."""
        
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
        
        trainer = Trainer(
            model, loss_fn,
            learning_rate=self.learning_rate,
            weight_decay=1e-4,
        )
        
        # Simple DataLoader - EXACTLY like V5
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
        logger.info("POOLED BASELINE V6: 800 Stocks")
        logger.info("=" * 80)
        logger.info(f"  Target Stocks: {len(self.stocks)}")
        logger.info(f"  Dropout: {self.dropout}")
        logger.info(f"  Learning Rate: {self.learning_rate}")
        logger.info(f"  Patience: {self.early_stopping_patience}")
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
        logger.info("V6 RESULTS")
        logger.info("=" * 80)
        logger.info(f"  Test Accuracy: {result['test_accuracy']:.2%}")
        logger.info(f"  Stocks Loaded: {len(stock_data)}")
        logger.info(f"  Total Samples: {result['total_samples']:,}")
        logger.info(f"  Sample/Param Ratio: {result['samples_per_param']:.1f}x")
        logger.info(f"  Epochs Trained: {result['epochs_trained']}")
        logger.info(f"  Best Val Loss: {result['best_val_loss']:.4f}")
        logger.info(f"\nResults saved to: {self.output_dir}")
        
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--stocks", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--start", type=str, default="2014-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    args = parser.parse_args()
    
    stocks = EXTENDED_TICKERS[:args.stocks]
    
    experiment = PooledExperimentV6(
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
