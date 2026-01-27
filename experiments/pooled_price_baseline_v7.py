"""
Pooled Price-Only Baseline V7 - Domain Knowledge Features.

Changes from V6:
1. V7 indicators: domain knowledge binary features + additional indicators
2. ~100 features (vs 73 in V6)
3. Colab-ready: can run with --colab flag for Google Colab

Usage (Local):
    python experiments/pooled_price_baseline_v7.py --epochs 100 --stocks 800

Usage (Colab):
    python experiments/pooled_price_baseline_v7.py --epochs 150 --stocks 1000 --colab
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
import glob
import os
import shutil
# Import paths depend on environment
try:
    from config import REPORTS_DIR, DATA_DIR, TOP_200_TICKERS, training_config, lstm_config
    from src.models import AttentionLSTM
    from src.training import Trainer
    from src.features.indicators_v7 import ComprehensiveIndicatorsV7
    RUNNING_LOCAL = True
except ImportError:
    RUNNING_LOCAL = False
    # Colab fallback - will be handled later

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Diagnostic Logging
logger.info(f"Python Version: {sys.version}")
logger.info(f"PyTorch Version: {torch.__version__}")
if torch.cuda.is_available():
    logger.info(f"CUDA Available: True")
    logger.info(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    logger.warning("CUDA NOT AVAILABLE")

# Massive stock list - 1000+ unique tickers
EXTENDED_TICKERS = [
    # S&P 500 Core
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "UNH", "JNJ",
    "V", "XOM", "JPM", "WMT", "PG", "MA", "HD", "CVX", "LLY", "MRK",
    "ABBV", "PEP", "KO", "COST", "AVGO", "TMO", "MCD", "CSCO", "ACN", "ABT",
    "DHR", "CRM", "ADBE", "NKE", "TXN", "CMCSA", "VZ", "NEE", "PM", "INTC",
    "RTX", "HON", "ORCL", "WFC", "BMY", "UNP", "T", "LOW", "QCOM", "UPS",
    "AMGN", "IBM", "ELV", "MS", "CAT", "GE", "SPGI", "BA", "SBUX", "LMT",
    "DE", "INTU", "AMD", "BLK", "GS", "AXP", "PLD", "MDT", "ISRG", "GILD",
    "ADI", "MDLZ", "SYK", "REGN", "TJX", "BKNG", "C", "CVS", "ADP", "VRTX",
    # Extended S&P 500
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
    "KEYS", "KHC", "KIM", "KLAC", "KMB", "KMI", "KMX", "KO", "KR",
    "L", "LDOS", "LEG", "LEN", "LH", "LHX", "LIN", "LKQ", "LLY",
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
    "SJM", "SLB", "SLG", "SNA", "SNPS", "SO", "SPG", "SPGI", "SRE",
    "STE", "STT", "STX", "STZ", "SWK", "SWKS", "SYF", "SYK", "SYY", "T",
    "TAP", "TDG", "TDY", "TEL", "TER", "TFC", "TFX", "TGT", "TJX", "TMO",
    "TMUS", "TPR", "TRMB", "TROW", "TRV", "TSCO", "TSLA", "TSN", "TT", "TTWO",
    "TXN", "TXT", "TYL", "UA", "UAA", "UAL", "UDR", "UHS", "ULTA",
    "UNH", "UNP", "UPS", "URI", "USB", "V", "VFC", "VIAC", "VLO", "VMC",
    "VNO", "VRSK", "VRSN", "VRTX", "VTR", "VTRS", "VZ", "WAB", "WAT", "WBA",
    "WDC", "WEC", "WELL", "WFC", "WHR", "WLTW", "WM", "WMB", "WMT", "WRB",
    "WRK", "WST", "WU", "WY", "WYNN", "XEL", "XLNX", "XOM", "XRAY", "XYL",
    "YUM", "ZBH", "ZBRA", "ZION", "ZTS",
    # Tech & Growth
    "PLTR", "COIN", "HOOD", "RIVN", "LCID", "NIO", "XPEV", "LI", "BABA",
    "JD", "PDD", "BILI", "TME", "IQ", "VIPS", "WB", "BIDU", "NTES",
    "SNAP", "PINS", "SPOT", "ROKU", "ZM", "DOCU", "CRWD", "NET", "DDOG",
    "SNOW", "OKTA", "ZS", "PANW", "FTNT", "MDB", "TEAM", "WDAY",
    "ABNB", "DASH", "UBER", "LYFT", "DKNG", "PENN", "MGM", "WYNN", "LVS",
    # Financials & Banks
    "BLK", "BX", "KKR", "APO", "CG", "ARES", "OWL", "TPG",
    # Healthcare & Biotech
    "MRNA", "BNTX", "NVAX", "SGEN", "BIIB", "REGN", "VRTX",
    # Energy
    "OXY", "FANG", "MPC", "VLO", "PSX", "HES", "BKR", "NOV",
    # International ADRs
    "TSM", "ASML", "NVO", "TM", "SONY", "SAP", "SHOP", "RY", "TD", "BMO",
    "BP", "SHEL", "TTE", "EQNR", "VALE", "RIO", "BHP", "GOLD", "AEM", "WPM",
]
EXTENDED_TICKERS = list(dict.fromkeys(EXTENDED_TICKERS))


class ColabConfig:
    """Configuration for Colab environment."""
    def __init__(self):
        self.data_dir = Path("/content/data")
        self.reports_dir = Path("/content/reports")
        self.models_dir = Path("/content/models")
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


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
        
        for f in self.cache_dir.glob(f"*_{start_date}_{end_date}.pkl"):
            logger.info(f"Found cache file: {f}")
            with open(f, "rb") as file:
                cached = pickle.load(file)
            logger.info(f"Cache contains {len(cached)} stocks")
            return cached
        
        return None
    
    def save(self, data: Dict[str, pd.DataFrame], start_date: str, end_date: str):
        cache_path = self.get_cache_path(start_date, end_date)
        if not cache_path.exists():
            logger.info(f"Saving prices to cache: {cache_path}")
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
        else:
            logger.info(f"Cache exists, skipping write: {cache_path}")


class IncrementalFeatureCache:
    """
    Optimized cache that uses Memory Mapping (memmap) to handle
    datasets larger than available RAM.
    """
    
    def __init__(self, cache_dir: Path = None):
        if RUNNING_LOCAL:
            self.cache_dir = cache_dir or DATA_DIR / "feature_cache"
        else:
            self.cache_dir = cache_dir or Path("/content/data/feature_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = self.cache_dir / "temp_batches"
    
    def get_cache_path(self, start_date: str, end_date: str, version: str = "v7") -> Path:
        pattern = f"features_{version}_*_{start_date}_{end_date}.npy" # Changed to .npy
        files = list(self.cache_dir.glob(pattern))
        if files:
            return max(files, key=lambda p: p.stat().st_size)
        return self.cache_dir / f"features_{version}_FULL_{start_date}_{end_date}.npy"

    def get_metadata_path(self, npy_path: Path) -> Path:
        """Get path for the metadata (y and feature cols) sidecar file."""
        return npy_path.with_suffix('.metadata.pkl')

    def save_batch(self, batch_idx: int, X: np.ndarray, y: np.ndarray):
        """Saves a single batch as uncompressed .npy for fast mapping."""
        self.temp_dir.mkdir(exist_ok=True)
        # We save X and y separately for easier merging
        np.save(self.temp_dir / f"batch_{batch_idx:04d}_X.npy", X.astype(np.float32))
        np.save(self.temp_dir / f"batch_{batch_idx:04d}_y.npy", y)

    def compile_batches(self, feature_cols: List[str], n_stocks: int, start_date: str, end_date: str, version: str = "v7") -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Merges batches directly to disk using memmap. 
        Zero RAM spike.
        """
        logger.info("Merging batch files to disk (Memmap)...")
        batch_files_X = sorted(list(self.temp_dir.glob("batch_*_X.npy")))
        batch_files_y = sorted(list(self.temp_dir.glob("batch_*_y.npy")))
        
        if not batch_files_X:
            raise ValueError("No batch files found to merge.")

        # 1. Calculate total size without loading data
        total_samples = 0
        sample_shapes = []
        for f in batch_files_X:
            # Read header only to get shape
            shape = np.load(f, mmap_mode='r').shape
            total_samples += shape[0]
            if not sample_shapes:
                sample_shapes = shape[1:]
        
        logger.info(f"Total dataset size: {total_samples:,} samples. Creating memmap on disk...")

        # 2. Prepare Final Paths
        final_path_X = self.cache_dir / f"features_{version}_{n_stocks}stocks_{start_date}_{end_date}.npy"
        final_path_meta = self.get_metadata_path(final_path_X)

        # 3. Create Memmap on Disk
        fp_X = np.memmap(final_path_X, dtype='float32', mode='w+', shape=(total_samples, *sample_shapes))
        
        all_y = []

        # 4. Stream data into the memmap
        current_idx = 0
        for f_X, f_y in zip(tqdm(batch_files_X, desc="Streaming to Disk"), batch_files_y):
            batch_X = np.load(f_X)
            batch_y = np.load(f_y)
            
            n = len(batch_X)
            fp_X[current_idx : current_idx + n] = batch_X
            all_y.append(batch_y)
            current_idx += n
            
            del batch_X, batch_y
        
        fp_X.flush()
        y_pooled = np.concatenate(all_y)
        
        # Save metadata
        with open(final_path_meta, 'wb') as f:
            pickle.dump({
                'y': y_pooled,
                'feature_cols': feature_cols,
                'shape': (total_samples, *sample_shapes),
                'dtype': 'float32'
            }, f)

        shutil.rmtree(self.temp_dir)
        
        # FIXED: Returns 3 values now
        return fp_X, y_pooled, feature_cols

    def load(self, start_date: str, end_date: str, version: str = "v7") -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Loads the large dataset using memory mapping."""
        path_X = self.get_cache_path(start_date, end_date, version)
        if not path_X.exists():
            return None, None, None
            
        path_meta = self.get_metadata_path(path_X)
        if not path_meta.exists():
            logger.warning("Found data file but missing metadata. Recomputing.")
            return None, None, None

        logger.info(f"Loading memory-mapped features from {path_X}")
        
        # Load metadata
        with open(path_meta, 'rb') as f:
            meta = pickle.load(f)
            
        y = meta['y']
        feature_cols = meta['feature_cols']
        shape = meta['shape']
        dtype = meta.get('dtype', 'float32')

        # Load X as Read-Only Memmap
        X = np.memmap(path_X, dtype=dtype, mode='r', shape=shape)
        
        logger.info(f"Memmap linked: {X.shape[0]:,} samples (Disk Mode)")
        return X, y, feature_cols

class PooledExperimentV7:
    """V7: Domain knowledge features + more stocks."""
    
    def __init__(
        self,
        stocks: List[str] = None,
        start_date: str = "2014-01-01",
        end_date: str = "2024-12-31",
        sequence_length: int = 20,
        epochs: int = 100,
        batch_size: int = 1024,
        early_stopping_patience: int = 15,
        dropout: float = 0.4,
        learning_rate: float = 5e-4,
        use_cache: bool = True,
        colab_mode: bool = False,
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
        self.colab_mode = colab_mode
        
        if colab_mode:
            self.output_dir = Path(f"/content/reports/pooled_v7_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        else:
            self.output_dir = REPORTS_DIR / f"pooled_v7_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.indicator_computer = ComprehensiveIndicatorsV7()
        self.price_cache = PriceCache()
        self.feature_cache = IncrementalFeatureCache()
        
    def load_all_stocks(self) -> Dict[str, pd.DataFrame]:
        """Load stocks: use cache + download only missing stocks."""
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
                logger.info(f"Downloading batch {i//batch_size + 1}/{(len(missing)-1)//batch_size + 1}")
                
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
            
            if self.use_cache and len(stock_data) > 0:
                self.price_cache.save(stock_data, self.start_date, self.end_date)
        
        logger.info(f"Total: {len(stock_data)} stocks ready")
        return stock_data
    
    def prepare_pooled_data(self, stock_data: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Fixed: Processes stocks in batches, saves to disk immediately, 
        and uses 0 RAM for the final merge.
        """
        import gc
        import shutil
        
        # Reset temp dir to avoid mixing old runs
        if self.feature_cache.temp_dir.exists():
            shutil.rmtree(self.feature_cache.temp_dir)
        
        feature_cols = None
        stock_items = list(stock_data.items())
        batch_size = 50 
        
        for batch_idx in range(0, len(stock_items), batch_size):
            batch = stock_items[batch_idx:batch_idx + batch_size]
            batch_X, batch_y = [], []
            
            for ticker, df in tqdm(batch, desc=f"Processing Batch {batch_idx//batch_size + 1}"):
                try:
                    # calls your missing method
                    X, y, cols = self.prepare_stock_data(df) 
                    if len(X) > 0:
                        batch_X.append(X)
                        batch_y.append(y)
                        if feature_cols is None: 
                            feature_cols = cols
                except Exception as e:
                    # logger.warning(f"Error {ticker}: {e}")
                    pass
            
            # CRITICAL: Save batch to disk and delete from RAM immediately
            if batch_X:
                batch_X_arr = np.concatenate(batch_X, axis=0)
                batch_y_arr = np.concatenate(batch_y, axis=0)
                
                # Saves to .npy files in temp_dir
                self.feature_cache.save_batch(batch_idx, batch_X_arr, batch_y_arr)
                
                # Force cleanup of RAM
                del batch_X, batch_y, batch_X_arr, batch_y_arr
                gc.collect()
                
        # Merge all disk files into one memory-mapped array (uses Disk, not RAM)
        return self.feature_cache.compile_batches(feature_cols, len(stock_data), self.start_date, self.end_date)

    def prepare_stock_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        # Compute indicators
        df = self.indicator_computer.compute_all(df)
        
        # Get feature columns and filter out any invalid ones
        feature_cols = self.indicator_computer.get_indicator_columns(df)
        
        # Remove columns that are all NaN or don't exist
        valid_cols = []
        for col in feature_cols:
            if col in df.columns:
                # Check if column has at least some valid data
                if df[col].notna().sum() > 10:
                    valid_cols.append(col)
        feature_cols = valid_cols
        
        if len(feature_cols) < 5:
            # logger.debug(f"Too few valid features: {len(feature_cols)}")
            return np.array([]), np.array([]), []
        
        df['return_next'] = df['Close'].pct_change().shift(-1)
        df['trend'] = pd.cut(
            df['return_next'],
            bins=[-np.inf, -0.005, 0.005, np.inf],
            labels=[0, 1, 2]
        ).astype(float)
        
        df = df.dropna(subset=['trend'] + feature_cols[:10])  # Only require core features
        if len(df) < self.sequence_length + 10:
            return np.array([]), np.array([]), []
        
        # Scale features
        scaler = StandardScaler()
        feature_data = df[feature_cols].values
        
        # Replace infinite values before scaling
        feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
        feature_data = scaler.fit_transform(feature_data)
        feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        labels = df['trend'].values
        X, y = [], []
        for i in range(len(feature_data) - self.sequence_length):
            X.append(feature_data[i:i + self.sequence_length])
            y.append(labels[i + self.sequence_length])
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), feature_cols

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
        """
        Fixed: Uses TensorDataset with references to avoid copying data.
        """
        import gc
        
        # Shuffle indices instead of shuffling the huge array
        indices = np.random.permutation(len(X))
        n = len(X)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)
        
        # Helper to create dataset without exploding RAM
        # We use torch.from_numpy which creates a view, not a copy (if types match)
        def create_dataset(idx_subset):
            # X is likely float16 from cache, we cast to float32 on the fly or here
            x_tensor = torch.from_numpy(X[idx_subset]).float() 
            y_tensor = torch.from_numpy(y[idx_subset]).long()
            return TensorDataset(x_tensor, y_tensor)

        logger.info("Creating Tensor Datasets...")
        train_dataset = create_dataset(indices[:train_end])
        val_dataset = create_dataset(indices[train_end:val_end])
        
        # We can now aggressively clean up if we were copying, but since we are referencing
        # the main X array, we must keep X in memory. 
        # The key save here was removing the intermediate "X_train = ..." numpy copies.

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            # pin_memory=True if torch.cuda.is_available() else False # Optional speedup
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # --- Model Setup ---
        model = AttentionLSTM(
            input_size=X.shape[-1],
            hidden_size=lstm_config.hidden_size if RUNNING_LOCAL else 128,
            num_layers=lstm_config.num_layers if RUNNING_LOCAL else 2,
            dropout=self.dropout,
        )
        
        loss_fn = nn.CrossEntropyLoss()
        trainer = Trainer(
            model, loss_fn,
            learning_rate=self.learning_rate,
            weight_decay=1e-4,
        )
        
        training_config.early_stopping_patience = self.early_stopping_patience
        history = trainer.train(train_loader, val_loader, epochs=self.epochs)
        
        # --- Evaluation ---
        # Evaluate on Test set
        test_indices = indices[val_end:]
        X_test = X[test_indices]
        y_test = y[test_indices]
        
        logger.info("Evaluating on test set...")
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
            "version": "v7",
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
        logger.info("POOLED BASELINE V7: Incremental Cache Mode")
        logger.info("=" * 80)
        
        # 1. Fuzzy Load (matches any stock count)
        X, y, feature_cols = self.feature_cache.load(self.start_date, self.end_date, "v7")
        
        stock_data = {}
        if X is None:
            logger.info("Cache miss. Calculating...")
            stock_data = self.load_all_stocks()
            X, y, feature_cols = self.prepare_pooled_data(stock_data)
            
            # Free stock data memory immediately after feature extraction
            del stock_data
            import gc; gc.collect()
        else:
             logger.info("Cache hit!")

        with open(self.output_dir / "features_v7.txt", "w") as f:
            for col in feature_cols:
                f.write(f"{col}\n")
        
        result = self.train_model(X, y)
        
        results_df = pd.DataFrame([result])
        results_df.to_csv(self.output_dir / "results.csv", index=False)
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
    parser.add_argument("--colab", action="store_true", help="Enable Colab mode")
    args = parser.parse_args()
    
    stocks = EXTENDED_TICKERS[:args.stocks]
    
    experiment = PooledExperimentV7(
        stocks=stocks,
        start_date=args.start,
        end_date=args.end,
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
