"""
Pooled Price-Only Baseline V9 - Proper Temporal Leakage Fix.

Critical fix from V8:
V8 BUG: We pooled sequences (Stock1_day1, Stock1_day2, ..., Stock2_day1, Stock2_day2)
        then split by index. This means train/test split by STOCK, not by TIME.
        Train has ALL days from stocks 1-280, test has ALL days from stocks 280-400.
        Model learns market regime patterns that apply to test stocks.

V9 FIX: Split EACH STOCK temporally FIRST, THEN pool.
        Train: Days 2014-2021 from ALL stocks
        Val:   Days 2021-2023 from ALL stocks  
        Test:  Days 2023-2024 from ALL stocks

Usage:
    python experiments/pooled_price_baseline_v9.py --epochs 100 --stocks 400
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

try:
    from config import REPORTS_DIR, DATA_DIR, training_config, lstm_config
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

# Stock list (same as V8)
EXTENDED_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "UNH", "JNJ",
    "V", "XOM", "JPM", "WMT", "PG", "MA", "HD", "CVX", "LLY", "MRK",
    "ABBV", "PEP", "KO", "COST", "AVGO", "TMO", "MCD", "CSCO", "ACN", "ABT",
    "DHR", "CRM", "ADBE", "NKE", "TXN", "CMCSA", "VZ", "NEE", "PM", "INTC",
    "RTX", "HON", "ORCL", "WFC", "BMY", "UNP", "T", "LOW", "QCOM", "UPS",
    "AMGN", "IBM", "ELV", "MS", "CAT", "GE", "SPGI", "BA", "SBUX", "LMT",
    "DE", "INTU", "AMD", "BLK", "GS", "AXP", "PLD", "MDT", "ISRG", "GILD",
    "ADI", "MDLZ", "SYK", "REGN", "TJX", "BKNG", "C", "CVS", "ADP", "VRTX",
    "A", "AAL", "AAP", "ABC", "ABMD", "ADM", "ADSK", "AEE", "AEP", "AES",
    "AFL", "AIG", "AIV", "AIZ", "AJG", "AKAM", "ALB", "ALGN", "ALK", "ALL",
    "ALLE", "AMAT", "AMCR", "AME", "AMP", "AMT", "ANET", "ANSS", "AON",
    "AOS", "APA", "APD", "APH", "APTV", "ARE", "ATO", "ATVI", "AVB",
    "AVY", "AWK", "AZO", "BAC", "BAX", "BBY", "BDX", "BEN",
    "BIIB", "BIO", "BK", "BKR", "BLL", "BR",
    "BSX", "BWA", "BXP", "CAG", "CAH", "CARR", "CB",
    "CBOE", "CBRE", "CCI", "CCL", "CDNS", "CDW", "CE", "CERN", "CF", "CFG",
    "CHD", "CHRW", "CHTR", "CI", "CINF", "CL", "CLX", "CMA", "CME",
    "CMG", "CMI", "CMS", "CNC", "CNP", "COF", "COO", "COP",
    "CPB", "CPRT", "CSX", "CTAS", "CTSH", "CTVA",
    "D", "DAL", "DD", "DFS", "DG", "DGX",
    "DHI", "DIS", "DLR", "DLTR", "DOV", "DOW",
    "DPZ", "DRE", "DRI", "DTE", "DUK", "DVA", "DVN", "DXCM", "EA",
    "EBAY", "ECL", "ED", "EFX", "EIX", "EL", "EMN", "EMR", "ENPH", "EOG",
    "EQIX", "EQR", "ES", "ESS", "ETN", "ETR", "EVRG", "EW", "EXC",
    "EXPD", "EXPE", "EXR", "F", "FANG", "FAST", "FBHS", "FCX", "FDX",
    "FE", "FFIV", "FIS", "FISV", "FITB", "FLT", "FMC", "FOX",
    "FOXA", "FRT", "FTNT", "FTV", "GD", "GIS", "GL",
    "GLW", "GM", "GNRC", "GOOG", "GPC", "GPN", "GPS", "GRMN",
    "GWW", "HAL", "HAS", "HBAN", "HBI", "HCA", "HES", "HIG", "HII",
    "HLT", "HOLX", "HPE", "HPQ", "HRB", "HRL", "HSIC", "HST", "HSY",
    "HUM", "HWM", "ICE", "IDXX", "IEX", "IFF", "ILMN", "INCY",
    "IP", "IPG", "IPGP", "IQV", "IR", "IRM", "IT", "ITW",
    "IVZ", "J", "JBHT", "JCI", "JKHY", "JNPR", "K", "KEY",
    "KEYS", "KHC", "KIM", "KLAC", "KMB", "KMI", "KMX", "KR",
    "L", "LDOS", "LEG", "LEN", "LH", "LHX", "LIN", "LKQ",
    "LNC", "LNT", "LRCX", "LUMN", "LUV", "LVS", "LW", "LYB",
    "LYV", "MAA", "MAR", "MAS", "MCHP", "MCK", "MCO",
    "MET", "MGM", "MHK", "MKC", "MKTX", "MLM", "MMC", "MMM", "MNST",
    "MO", "MOS", "MPC", "MPWR", "MRO", "MSCI", "MSI",
    "MTB", "MTD", "MU", "NCLH", "NDAQ", "NEM", "NFLX", "NI",
    "NLOK", "NLSN", "NOC", "NOV", "NOW", "NRG", "NSC", "NTAP", "NTRS", "NUE",
    "NVR", "NWL", "NWS", "NWSA", "O", "ODFL", "OGN", "OKE", "OMC",
    "ORLY", "OTIS", "OXY", "PAYC", "PAYX", "PBCT", "PCAR", "PEAK",
    "PEG", "PENN", "PFE", "PFG", "PGR", "PH", "PHM", "PKG",
    "PKI", "PNC", "PNR", "PNW", "POOL", "PPG", "PPL", "PRGO",
    "PRU", "PSA", "PSX", "PTC", "PVH", "PWR", "PXD", "PYPL", "QRVO",
    "RCL", "RE", "REG", "RF", "RHI", "RJF", "RL", "RMD", "ROK",
    "ROL", "ROP", "ROST", "RSG", "SBAC", "SCHW", "SEE", "SHW",
    "SJM", "SLB", "SLG", "SNA", "SNPS", "SO", "SPG", "SRE",
    "STE", "STT", "STX", "STZ", "SWK", "SWKS", "SYF", "SYY",
    "TAP", "TDG", "TDY", "TEL", "TER", "TFC", "TFX", "TGT",
    "TMUS", "TPR", "TRMB", "TROW", "TRV", "TSCO", "TSN", "TT", "TTWO",
    "TXT", "TYL", "UA", "UAA", "UAL", "UDR", "UHS", "ULTA",
    "UNP", "URI", "USB", "VFC", "VIAC", "VLO", "VMC",
    "VNO", "VRSK", "VRSN", "VTR", "VTRS", "WAB", "WAT", "WBA",
    "WDC", "WEC", "WELL", "WHR", "WLTW", "WM", "WMB", "WRB",
    "WRK", "WST", "WU", "WY", "WYNN", "XEL", "XLNX", "XRAY", "XYL",
    "YUM", "ZBH", "ZBRA", "ZION", "ZTS",
]
EXTENDED_TICKERS = list(dict.fromkeys(EXTENDED_TICKERS))


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


class PooledExperimentV9:
    """
    V9: Proper temporal split - split EACH STOCK by time, THEN pool.
    
    Key fix: Every stock is split into train/val/test by DATE, not index.
    Train: 2014-2021 from ALL stocks
    Val:   2021-2023 from ALL stocks  
    Test:  2023-2024 from ALL stocks
    """
    
    def __init__(
        self,
        stocks: List[str] = None,
        start_date: str = "2014-01-01",
        end_date: str = "2024-12-31",
        train_end_date: str = "2022-01-01",  # Train: everything before this
        val_end_date: str = "2023-06-01",     # Val: train_end to this
        # Test: everything after val_end
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
            self.output_dir = Path(f"/content/reports/pooled_v9_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        else:
            self.output_dir = REPORTS_DIR / f"pooled_v9_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    
    def prepare_stock_with_temporal_split(self, df: pd.DataFrame) -> Tuple[
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
        feature_cols = self.indicator_computer.get_indicator_columns(df)
        
        # Filter valid columns
        valid_cols = [col for col in feature_cols 
                      if col in df.columns and df[col].notna().sum() > 10]
        feature_cols = valid_cols
        
        if len(feature_cols) < 5:
            return (np.array([]), np.array([]), np.array([]), 
                    np.array([]), np.array([]), np.array([]), [])
        
        # Create target
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
            label = labels[i + self.sequence_length]
            
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
                X_tr, y_tr, X_val, y_val, X_te, y_te, cols = self.prepare_stock_with_temporal_split(df)
                
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
    
    def train_model(self, X_train, y_train, X_val, y_val, X_test, y_test, feature_cols) -> Dict:
        """Train with properly temporally-split data."""
        
        logger.info("=" * 60)
        logger.info("V9 FIX: TRUE TEMPORAL SPLIT")
        logger.info(f"  Train: {len(X_train):,} samples (2014 to {self.train_end_date.date()})")
        logger.info(f"  Val:   {len(X_val):,} samples ({self.train_end_date.date()} to {self.val_end_date.date()})")
        logger.info(f"  Test:  {len(X_test):,} samples ({self.val_end_date.date()} to 2024)")
        logger.info("=" * 60)
        
        # Scale after split
        X_train, X_val, X_test = self.scale_data(X_train, X_val, X_test)
        
        model = AttentionLSTM(
            input_size=X_train.shape[-1],
            hidden_size=lstm_config.hidden_size if RUNNING_LOCAL else 128,
            num_layers=lstm_config.num_layers if RUNNING_LOCAL else 2,
            dropout=self.dropout,
        )
        
        n_params = sum(p.numel() for p in model.parameters())
        ratio = len(X_train) / n_params
        logger.info(f"Model params: {n_params:,}, Samples/params: {ratio:.2f}x")
        
        loss_fn = nn.CrossEntropyLoss()
        trainer = Trainer(model, loss_fn, learning_rate=self.learning_rate, weight_decay=1e-4)
        
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
            batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
            batch_size=self.batch_size
        )
        
        training_config.early_stopping_patience = self.early_stopping_patience
        history = trainer.train(train_loader, val_loader, epochs=self.epochs)
        
        logger.info("Evaluating on test set...")
        pred_classes = self.predict_batched(model, X_test)
        metrics = self.compute_metrics(y_test, pred_classes)
        cm = confusion_matrix(y_test, pred_classes)
        
        logger.info("\n" + "=" * 60)
        logger.info("V9 RESULTS (TRUE TEMPORAL SPLIT)")
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
            "model": "attention_lstm",
            "version": "v9",
            "train_end_date": str(self.train_end_date.date()),
            "val_end_date": str(self.val_end_date.date()),
            "n_features": X_train.shape[-1],
            "n_params": n_params,
            "n_stocks": len(self.stocks),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "samples_per_param": ratio,
            **metrics,
            "confusion_matrix": cm.tolist(),
            "best_val_loss": trainer.best_val_loss,
            "epochs_trained": len(history.get("history", [])),
        }
    
    def run(self):
        logger.info("=" * 80)
        logger.info("POOLED BASELINE V9: TRUE TEMPORAL SPLIT")
        logger.info("=" * 80)
        logger.info("KEY FIX: Split each stock by DATE first, then pool")
        logger.info(f"  Train period: 2014 to {self.train_end_date.date()}")
        logger.info(f"  Val period:   {self.train_end_date.date()} to {self.val_end_date.date()}")
        logger.info(f"  Test period:  {self.val_end_date.date()} to 2024")
        logger.info("=" * 80)
        
        stock_data = self.load_all_stocks()
        X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = \
            self.prepare_temporal_pooled_data(stock_data)

        # Check for leaks
        check_for_leaks(X_train, y_train, feature_cols)
        
        del stock_data
        import gc
        gc.collect()
        
        with open(self.output_dir / "features_v9.txt", "w") as f:
            for col in feature_cols:
                f.write(f"{col}\n")
        
        result = self.train_model(X_train, y_train, X_val, y_val, X_test, y_test, feature_cols)
        
        results_df = pd.DataFrame([result])
        results_df.to_csv(self.output_dir / "results.csv", index=False)
        
        logger.info(f"\nResults saved to: {self.output_dir}")
        return result



def check_for_leaks(X_train, y_train, feature_cols):
    logger.info("🕵️ RUNNING LEAK DETECTION...")
    
    # 1. Convert back to DataFrame for easy analysis
    # Take a sample (first 50k rows) to save memory
    sample_size = min(50000, len(X_train))
    
    # We only need the LAST step of the sequence to check for direct correlation
    # Shape: (N, 20, Features) -> We take (N, 19, :) -> The most recent day in the seq
    last_day_features = X_train[:sample_size, -1, :] 
    
    df_check = pd.DataFrame(last_day_features, columns=feature_cols)
    df_check['TARGET_LABEL'] = y_train[:sample_size]
    
    # 2. correlation matrix
    correlations = df_check.corrwith(df_check['TARGET_LABEL'])
    
    # 3. Find the suspects
    suspects = correlations[abs(correlations) > 0.85]
    
    if len(suspects) > 0:
        logger.warning("🚨 LEAK DETECTED! The following features correlate too highly with the target:")
        logger.warning(suspects)
        logger.warning("Remove these features from indicators_v7.py")
    else:
        logger.info("✅ No direct correlation leaks detected.")


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
    
    experiment = PooledExperimentV9(
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
