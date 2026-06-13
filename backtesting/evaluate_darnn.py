"""
Backtesting evaluator for DA-RNN walk-forward experiments.

Loads a trained DARNN from a report directory, reconstructs the feature/target
scalers, generates predictions across test stocks, then runs both:
  1. Cross-sectional IC time series (ranking quality over time)
  2. Portfolio strategy simulations (regression + baseline strategies)

This evaluator handles DA-RNN specifics that evaluate_regression.py cannot:
  - Dual-input model: X (features) + y_history (decoder target history)
  - Full V7 indicator set (with leaky-indicator exclusions)
  - Optional sentiment features (sent_count_log, sent_strength)
  - Sentiment-density-ranked ticker universe with exclusion list
  - Separate feature scaler + target scaler (both reconstructed from training data)
  - Walk-forward fold structure (expanding-window training)

Usage:
    python backtesting/evaluate_darnn.py --report-dir reports/wf_darnn_baseline_20260612_011221
    python backtesting/evaluate_darnn.py --report-dir reports/wf_darnn_sentiment_20260612_... --top-n-test-stocks 50
"""

import sys
from pathlib import Path
import json
import logging
import argparse
import gc
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cache import PriceCache
from src.data.utils import load_stocks
from src.features.indicators_v7 import ComprehensiveIndicatorsV7
from src.models.darnn import DARNN
from backtesting.framework import Backtester

try:
    from config import CACHE_DIR
except ImportError:
    CACHE_DIR = Path("data/cache")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS (mirrored from regression_darnn_sentiment_rolling.py)
# ============================================================================

EXCLUDED_FEATURES = [
    'ichimoku_ICS_26',
    'dpo',
]

SENTIMENT_FEATURES = [
    "sent_count_log",
    "sent_strength",
]

EXCLUDE_TICKERS = {
    "FDX", "BHI", "AET", "CHN", "SLB", "VNQ", "ESRX", "XPP", "DGAZ", "UGAZ",
    "WFM", "ERO", "JD", "VGK", "FOXA", "YINN", "ABX", "FXP", "PGJ", "GILD",
    "GXC", "XLY", "XLU",
    "MYL", "GME", "MON", "EWU", "TSN", "JCP", "POT", "CI", "WBA", "BRRY",
    "SPLS", "FCX", "BTU", "GRUB", "REGN", "XLK", "AXP", "DISH", "TWC", "TJX",
    "PBR", "DZZ", "KR", "MDT", "PANW", "CSX", "AVGO", "KSS", "SHLD", "OXY",
    "EWJ", "ANTM", "COH", "TXN", "FCAU", "BABA", "WDC", "HAL", "EA",
    "GMCR", "DIS", "RSP", "WMT", "YHOO", "INTC", "JPM", "DOW", "SLV", "GRPN",
    "LEN", "LLY", "VMW", "GPRO", "CMG", "HD", "LUV", "AEO", "TWX", "JWN",
    "BRCM", "DD", "STZ", "NFLX", "BLK", "BBRY", "WDAY",
}


# ============================================================================
# SENTIMENT DATA LOADER (same as experiment)
# ============================================================================

def load_sentiment_data(sentiment_dir: Path = None):
    """Load sentiment parquets and aggregate to daily features."""
    if sentiment_dir is None:
        sentiment_dir = CACHE_DIR / "sentiment"

    if not sentiment_dir.exists():
        logger.warning(f"Sentiment directory not found: {sentiment_dir}")
        return {}

    sentiment_files = list(sentiment_dir.glob("*_sentiment.parquet"))
    if not sentiment_files:
        logger.warning("No sentiment parquets found")
        return {}

    logger.info(f"Loading sentiment from {len(sentiment_files)} ticker files "
                f"(excluding {len(EXCLUDE_TICKERS)} noisy tickers)...")
    result = {}

    for fpath in sentiment_files:
        ticker = fpath.stem.replace("_sentiment", "")
        if ticker in EXCLUDE_TICKERS:
            continue

        try:
            raw = pd.read_parquet(fpath)
            raw['date'] = pd.to_datetime(raw['date'], utc=True).dt.tz_localize(None)
            raw['trading_date'] = raw['date'].dt.normalize()

            daily = raw.groupby('trading_date').agg(
                sent_mean=('sentiment_value', 'mean'),
                sent_count=('sentiment_value', 'count'),
            ).sort_index()

            daily['sent_count_log'] = np.log1p(daily['sent_count'])
            daily['sent_strength'] = daily['sent_mean'] * daily['sent_count_log']

            if len(daily) > 0:
                full_idx = pd.bdate_range(daily.index.min(), daily.index.max())
                daily = daily.reindex(full_idx)
                had_news = daily['sent_count'].notna()

                for col in SENTIMENT_FEATURES:
                    if col not in daily.columns:
                        continue
                    filled = daily[col].ffill()
                    decay_factor = 0.95
                    cumulative_decay = (~had_news).astype(float)
                    groups = had_news.cumsum()
                    consecutive_gaps = cumulative_decay.groupby(groups).cumsum()
                    decay_mask = consecutive_gaps > 0
                    filled[decay_mask] = filled[decay_mask] * (decay_factor ** consecutive_gaps[decay_mask])
                    daily[col] = filled

            daily = daily.fillna(0)
            daily = daily[SENTIMENT_FEATURES]
            result[ticker] = daily

        except Exception as e:
            logger.warning(f"Failed to load sentiment for {ticker}: {e}")

    logger.info(f"Loaded sentiment for {len(result)} tickers")
    return result


# ============================================================================
# DA-RNN BACKTEST RUNNER
# ============================================================================

class DARNNBacktestRunner:
    """
    Loads a trained DARNN from a report directory and runs
    backtesting strategies + cross-sectional IC analysis.

    Handles the DA-RNN-specific dual-input architecture, full V7 features,
    optional sentiment features, and constrained ticker universe.
    """

    def __init__(
        self,
        report_dir: str,
        backtest_start: str = None,
        backtest_end: str = None,
        top_n_test_stocks: int = None,
    ):
        self.report_dir = Path(report_dir)
        with open(self.report_dir / "config.json", "r") as f:
            self.config = json.load(f)

        # --- Model parameters ---
        self.frequency = self.config.get("frequency", "daily")
        self.sequence_length = self.config.get("sequence_length", 20 if self.frequency == "daily" else 12)
        self.vol_lookback = self.config.get("vol_lookback", 20)
        self.target_clip = self.config.get("target_clip", 10.0)
        self.dropout = self.config.get("dropout", 0.2)
        self.encoder_hidden = self.config.get("encoder_hidden", 64)
        self.decoder_hidden = self.config.get("decoder_hidden", 64)
        self.excluded_features = self.config.get("excluded_features", EXCLUDED_FEATURES)

        # --- Sentiment ---
        self.use_sentiment = self.config.get("use_sentiment", False)
        sentiment_features_cfg = self.config.get("sentiment_features", [])
        # If sentiment is enabled but no features listed, use defaults
        if self.use_sentiment and not sentiment_features_cfg:
            self.sentiment_feature_names = SENTIMENT_FEATURES
        else:
            self.sentiment_feature_names = sentiment_features_cfg

        # --- Date range ---
        self.start_date = self.config.get("start_date", "2012-01-01")
        end_date_cfg = self.config.get("end_date", "2019-12-31")

        # Walk-forward folds define the eval period
        folds = self.config.get("folds", [])

        # For the backtest period, default to the full walk-forward val range
        if folds:
            default_backtest_start = folds[0]["fold_start"]
            default_backtest_end = folds[-1]["fold_end"]
        else:
            default_backtest_start = self.config.get("wf_val_start", "2017-01-01")
            default_backtest_end = self.config.get("wf_val_end", "2019-12-31")

        self.backtest_start = pd.Timestamp(backtest_start or default_backtest_start)
        self.backtest_end = pd.Timestamp(backtest_end or default_backtest_end)
        # Extend end_date if backtest_end is beyond config's end_date
        self.end_date = max(pd.Timestamp(end_date_cfg), self.backtest_end).strftime('%Y-%m-%d')

        # --- Training period for scaler reconstruction ---
        # The saved model is from the last fold, so train_end = last fold's start
        if folds:
            self.train_end = pd.Timestamp(folds[-1]["fold_start"])
        else:
            self.train_end = pd.Timestamp(self.config.get("wf_val_start", "2017-01-01"))

        # --- Ticker universe ---
        self.train_stocks = self.config.get("train_stocks_list", [])
        original_test_stocks = self.config.get("test_stocks", [])
        self.top_n_test_stocks = top_n_test_stocks

        # Load sentiment data for universe construction
        self.sentiment_data = {}
        if self.use_sentiment:
            self.sentiment_data = load_sentiment_data()

        if self.top_n_test_stocks is not None:
            # Override test stocks with top N from the density-ranked universe
            all_tickers_in_universe = list(set(self.train_stocks + original_test_stocks))
            self.test_stocks = all_tickers_in_universe[:self.top_n_test_stocks]
            logger.info(f"Overriding test_stocks with {len(self.test_stocks)} from universe")
        else:
            self.test_stocks = original_test_stocks

        # --- Infrastructure ---
        self.indicator_computer = ComprehensiveIndicatorsV7()
        self.cache = PriceCache()
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Sentiment merge (same logic as experiment)
    # ------------------------------------------------------------------

    def _merge_sentiment(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Merge sentiment features onto a price DataFrame."""
        if not self.use_sentiment or ticker not in self.sentiment_data:
            for col in self.sentiment_feature_names:
                df[col] = 0.0
            return df

        sent_df = self.sentiment_data[ticker]
        price_dates = df.index.normalize()

        for col in self.sentiment_feature_names:
            if col in sent_df.columns:
                aligned = sent_df[col].reindex(price_dates)
                df[col] = aligned.values
            else:
                df[col] = 0.0

        for col in self.sentiment_feature_names:
            df[col] = df[col].fillna(0.0)

        return df

    # ------------------------------------------------------------------
    # Feature + target computation (mirrors experiment exactly)
    # ------------------------------------------------------------------

    def _compute_features_and_target(self, df: pd.DataFrame, ticker: str = ""):
        """
        Compute full V7 features + sentiment and vol-adjusted return target.

        Returns (feature_data, targets, dates, available_cols) or (None, None, None, None).
        """
        df = self.indicator_computer.compute_all(df)
        df = self._merge_sentiment(df, ticker)

        # Dynamically discover all indicator columns and exclude leaky ones
        all_indicator_cols = self.indicator_computer.get_indicator_columns(df)
        available = [c for c in all_indicator_cols
                     if c not in self.excluded_features and c in df.columns]

        # Add sentiment features
        for col in self.sentiment_feature_names:
            if col in df.columns and col not in available:
                available.append(col)

        if len(available) < 5:
            return None, None, None, None

        # Vol-adjusted forward return (regression target)
        df['return_next_raw'] = df['Close'].pct_change().shift(-1)
        df['rolling_vol'] = df['Close'].pct_change().rolling(self.vol_lookback).std()
        df['vol_adj_return'] = df['return_next_raw'] / df['rolling_vol']
        df['vol_adj_return'] = df['vol_adj_return'].clip(-self.target_clip, self.target_clip)

        df = df.dropna(subset=['vol_adj_return'] + available[:5])
        if len(df) < self.sequence_length + 10:
            return None, None, None, None

        feature_data = df[available].values
        feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
        targets = df['vol_adj_return'].values.astype(np.float32)

        return feature_data, targets, df.index, available

    # ------------------------------------------------------------------
    # DA-RNN sequence creation with dates (for backtest)
    # ------------------------------------------------------------------

    def _create_sequences_with_dates(self, feature_data, targets, dates, period_start, period_end):
        """
        Create DA-RNN sequences for a date period, returning dates too.

        Returns: (X, y_history, y_target, raw_return_next, trade_dates)
        """
        X_list, yh_list, y_list, date_list = [], [], [], []

        for i in range(1, len(feature_data) - self.sequence_length + 1):
            target_idx = i + self.sequence_length - 1
            target_date = dates[target_idx]

            if period_start <= target_date < period_end:
                seq_x = feature_data[i : i + self.sequence_length]
                seq_yh = targets[i - 1 : i + self.sequence_length - 1]
                target = targets[target_idx]

                X_list.append(seq_x)
                yh_list.append(seq_yh)
                y_list.append(target)
                date_list.append(target_date)

        def _to_array(lst, dtype):
            return np.array(lst, dtype=dtype) if lst else np.array([], dtype=dtype)

        return (
            _to_array(X_list, np.float32),
            _to_array(yh_list, np.float32),
            _to_array(y_list, np.float32),
            date_list,
        )

    def _create_train_sequences(self, feature_data, targets, dates, train_end):
        """Create training sequences (for scaler fitting). No dates needed."""
        X_list, yh_list, y_list = [], [], []

        for i in range(1, len(feature_data) - self.sequence_length + 1):
            target_date = dates[i + self.sequence_length - 1]
            if target_date < train_end:
                X_list.append(feature_data[i : i + self.sequence_length])
                yh_list.append(targets[i - 1 : i + self.sequence_length - 1])
                y_list.append(targets[i + self.sequence_length - 1])

        def _to_array(lst, dtype):
            return np.array(lst, dtype=dtype) if lst else np.array([], dtype=dtype)

        return _to_array(X_list, np.float32), _to_array(yh_list, np.float32), _to_array(y_list, np.float32)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self):
        all_tickers = list(set(self.train_stocks + self.test_stocks))
        weekly = self.frequency == "weekly"
        return load_stocks(
            all_tickers, self.start_date, self.end_date,
            cache=self.cache, min_rows=100 if not weekly else 52, weekly=weekly,
        )

    # ------------------------------------------------------------------
    # Scaler reconstruction
    # ------------------------------------------------------------------

    def reconstruct_scalers(self, stock_data):
        """
        Fit feature scaler + target scaler on train_stocks up to train_end.

        The DA-RNN uses two scalers:
          - feature_scaler: StandardScaler on flattened feature sequences
          - target_scaler: StandardScaler on vol-adj return targets
        """
        logger.info("Reconstructing StandardScalers from training data...")
        all_train_X = []
        all_train_y = []
        n_features = None

        for ticker in tqdm(self.train_stocks, desc="Fitting Scalers"):
            if ticker not in stock_data:
                continue

            result = self._compute_features_and_target(stock_data[ticker], ticker)
            if result[0] is None:
                continue

            feat, tgt, dates, _ = result

            X_tr, yh_tr, y_tr = self._create_train_sequences(feat, tgt, dates, self.train_end)
            if len(X_tr) > 0:
                if n_features is None:
                    n_features = X_tr.shape[-1]
                if X_tr.shape[-1] == n_features:
                    all_train_X.append(X_tr)
                    all_train_y.append(y_tr)

        if all_train_X:
            X_train = np.concatenate(all_train_X, axis=0)
            y_train = np.concatenate(all_train_y, axis=0)

            n_train, seq_len, feat = X_train.shape

            # Feature scaler (chunked to avoid OOM)
            CHUNK_ROWS = 500_000
            X_flat = X_train.reshape(-1, feat)
            n_flat = X_flat.shape[0]

            for start in range(0, n_flat, CHUNK_ROWS):
                chunk = X_flat[start : start + CHUNK_ROWS].astype(np.float64)
                self.feature_scaler.partial_fit(chunk)
                del chunk

            del X_flat, X_train
            gc.collect()

            # Target scaler
            self.target_scaler.fit(y_train.reshape(-1, 1))

            logger.info(f"Scalers fitted on {n_train:,} sequences × {feat} features, "
                        f"{len(y_train):,} target values.")
            return feat
        else:
            logger.error("No training data found to fit scalers!")
            return 0

    # ------------------------------------------------------------------
    # Prediction generation
    # ------------------------------------------------------------------

    def prepare_test_predictions(self, stock_data, n_features):
        """
        Generate test-period predictions for all test stocks.

        Returns DataFrame with:
            Date, Ticker, Pred_Return, Return_Next, Actual_Vol_Adj
        """
        self.model.eval()
        all_predictions = []

        for ticker in tqdm(self.test_stocks, desc="Predicting Test Set"):
            if ticker not in stock_data:
                continue

            result = self._compute_features_and_target(stock_data[ticker], ticker)
            if result[0] is None:
                continue

            feat, tgt, dates, _ = result

            X_test, yh_test, y_test, test_dates = self._create_sequences_with_dates(
                feat, tgt, dates, self.backtest_start, self.backtest_end,
            )

            if len(X_test) == 0:
                continue

            n_samples, seq_len, n_feat = X_test.shape

            # Scale features
            X_scaled = self.feature_scaler.transform(X_test.reshape(-1, n_feat))
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            X_scaled = X_scaled.reshape(n_samples, seq_len, n_feat).astype(np.float32)

            # Scale decoder history and targets
            yh_scaled = self.target_scaler.transform(
                yh_test.reshape(-1, 1)
            ).reshape(yh_test.shape).astype(np.float32)

            # Predict in batches
            batch_size = 2048
            preds = []
            with torch.no_grad():
                for i in range(0, n_samples, batch_size):
                    batch_x = torch.FloatTensor(X_scaled[i:i+batch_size]).to(self.device)
                    batch_yh = torch.FloatTensor(yh_scaled[i:i+batch_size]).to(self.device)
                    result_pred = self.model.predict(batch_x, batch_yh)
                    preds.append(result_pred["prediction"].cpu().numpy())

            predictions_scaled = np.concatenate(preds)

            # Inverse transform predictions to original vol-adj scale
            predictions_original = self.target_scaler.inverse_transform(
                predictions_scaled.reshape(-1, 1)
            ).ravel()

            # Actual vol-adj returns (already in original scale)
            actuals_original = tgt  # raw targets before scaling

            # Build per-sample DataFrame
            # We also need the raw next-day return for PnL
            df_ticker = stock_data[ticker].copy()
            df_ticker = self.indicator_computer.compute_all(df_ticker)
            df_ticker['return_next_raw'] = df_ticker['Close'].pct_change().shift(-1)

            test_records = []
            for j, date in enumerate(test_dates):
                # Look up the raw return for this date
                try:
                    raw_ret = df_ticker.loc[date, 'return_next_raw']
                    if isinstance(raw_ret, pd.Series):
                        raw_ret = raw_ret.iloc[0]
                except (KeyError, IndexError):
                    raw_ret = 0.0

                if np.isnan(raw_ret):
                    raw_ret = 0.0

                test_records.append({
                    'Date': date,
                    'Ticker': ticker,
                    'Pred_Return': predictions_original[j],
                    'Return_Next': raw_ret,
                    'Actual_Vol_Adj': y_test[j],  # unscaled vol-adj target
                })

            all_predictions.append(pd.DataFrame(test_records))

        if not all_predictions:
            raise ValueError("No test predictions generated. Check test stocks vs available data.")

        return pd.concat(all_predictions, ignore_index=True)

    # ------------------------------------------------------------------
    # Cross-sectional IC time series
    # ------------------------------------------------------------------

    def compute_ic_time_series(self, preds_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute daily cross-sectional IC and quantile spread.

        For each date, correlate model predictions with actual vol-adj returns
        across all stocks present that day.
        """
        records = []

        for date, group in preds_df.groupby('Date'):
            if len(group) < 5:
                continue

            preds = group['Pred_Return'].values
            actuals = group['Actual_Vol_Adj'].values

            # Spearman rank correlation
            ic_result = stats.spearmanr(preds, actuals)
            ic = float(ic_result.correlation) if not np.isnan(ic_result.correlation) else 0.0

            # Quantile spread: top 20% vs bottom 20%
            n = len(preds)
            q_size = max(1, n // 5)
            sorted_idx = np.argsort(preds)
            bottom_ret = actuals[sorted_idx[:q_size]].mean()
            top_ret = actuals[sorted_idx[-q_size:]].mean()

            # Raw return spread
            raw_actuals = group['Return_Next'].values
            bottom_raw = raw_actuals[sorted_idx[:q_size]].mean()
            top_raw = raw_actuals[sorted_idx[-q_size:]].mean()

            records.append({
                'Date': date,
                'IC': ic,
                'N_Stocks': n,
                'Quantile_Spread': top_ret - bottom_ret,
                'Top_Q_Return': top_ret,
                'Bottom_Q_Return': bottom_ret,
                'Raw_Spread': top_raw - bottom_raw,
                'Raw_Top_Q': top_raw,
                'Raw_Bottom_Q': bottom_raw,
            })

        ic_df = pd.DataFrame(records)
        if not ic_df.empty:
            ic_df.set_index('Date', inplace=True)
            ic_df['IC_Rolling_20'] = ic_df['IC'].rolling(20, min_periods=5).mean()
            ic_df['Spread_Rolling_20'] = ic_df['Raw_Spread'].rolling(20, min_periods=5).mean()
            ic_df['Cumulative_Spread'] = (1 + ic_df['Raw_Spread']).cumprod() - 1

        return ic_df

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------

    def run(self):
        logger.info(f"Loading DA-RNN experiment from: {self.report_dir}")
        logger.info(f"Sentiment: {'ENABLED' if self.use_sentiment else 'DISABLED'}")
        logger.info(f"Train stocks: {len(self.train_stocks)}, Test stocks: {len(self.test_stocks)}")
        logger.info(f"Backtest period: {self.backtest_start.date()} → {self.backtest_end.date()}")
        logger.info(f"Train end (scaler fit boundary): {self.train_end.date()}")

        # 1. Load data
        stock_data = self.load_data()

        # 2. Reconstruct scalers
        n_features = self.reconstruct_scalers(stock_data)
        if n_features == 0:
            logger.error("Failed to reconstruct scalers — aborting")
            return

        # 3. Load model
        logger.info("Loading DARNN model weights...")
        self.model = DARNN(
            input_size=n_features,
            seq_length=self.sequence_length,
            encoder_hidden=self.encoder_hidden,
            decoder_hidden=self.decoder_hidden,
            dropout=0.0,  # inference mode
        )
        state_dict = torch.load(
            self.report_dir / "model.pt",
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"DARNN loaded: {sum(p.numel() for p in self.model.parameters()):,} params")

        # 4. Generate predictions
        preds_df = self.prepare_test_predictions(stock_data, n_features)
        logger.info(f"Generated {len(preds_df):,} predictions across "
                    f"{preds_df['Ticker'].nunique()} stocks")

        n_dates = preds_df['Date'].nunique()
        logger.info(f"Date range: {preds_df['Date'].min()} → {preds_df['Date'].max()} "
                    f"({n_dates} dates)")

        # 5. Cross-sectional IC time series
        logger.info("\n" + "=" * 60)
        logger.info("CROSS-SECTIONAL IC ANALYSIS")
        logger.info("=" * 60)

        ic_df = self.compute_ic_time_series(preds_df)

        if not ic_df.empty:
            mean_ic = ic_df['IC'].mean()
            std_ic = ic_df['IC'].std()
            ic_ir = mean_ic / std_ic if std_ic > 0 else 0.0
            hit_rate = (ic_df['IC'] > 0).mean()
            mean_spread = ic_df['Quantile_Spread'].mean()
            mean_raw_spread = ic_df['Raw_Spread'].mean()
            cum_spread = ic_df['Cumulative_Spread'].iloc[-1] if len(ic_df) > 0 else 0

            logger.info(f"Mean IC:           {mean_ic:.4f}")
            logger.info(f"IC Std:            {std_ic:.4f}")
            logger.info(f"IC IR:             {ic_ir:.4f}")
            logger.info(f"IC Hit Rate:       {hit_rate:.2%}")
            logger.info(f"Mean Q-Spread (vol-adj): {mean_spread:.4f}")
            logger.info(f"Mean Q-Spread (raw):     {mean_raw_spread:.4%}")
            logger.info(f"Cumulative Spread:       {cum_spread:.2%}")
            logger.info(f"Evaluated on {len(ic_df)} dates")

            ic_csv = self.report_dir / "cross_sectional_ic.csv"
            ic_df.to_csv(ic_csv)
            logger.info(f"Saved IC time series to: {ic_csv}")
        else:
            logger.warning("No cross-sectional IC data (not enough stocks per day)")

        # 6. Run strategies
        logger.info("\n" + "=" * 60)
        logger.info("STRATEGY SIMULATION")
        logger.info("=" * 60)

        backtester = Backtester(initial_capital=100000.0)

        strategies = [
            # Baselines
            ("Buy_Hold_Universe", {}),
            ("Daily_Rebalanced_Universe", {}),
            ("Random_Allocation", {'n': 3}),
            # Regression strategies
            ("Regression_Long_Top_Pct", {'long_pct': 0.10}),
            ("Regression_Long_Top_Pct", {'long_pct': 0.20}),
            ("Regression_Long_Top_Pct", {'long_pct': 0.30}),
            ("Regression_Long_Short", {'long_pct': 0.20, 'short_pct': 0.20}),
            ("Regression_Long_Short", {'long_pct': 0.10, 'short_pct': 0.10}),
            ("Regression_Quantile_Spread", {}),
            ("Regression_Threshold_Long", {'threshold': 0.0}),
            ("Regression_Threshold_Long", {'threshold': 0.005}),
        ]

        baseline_df = preds_df.copy()
        baseline_df['Prob_Up'] = 0.33

        # Ensure plots directory exists
        plots_dir = self.report_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        from src.evaluation.plots import plot_portfolio_composition, plot_trade_activity

        results = []
        for strat_name, kwargs in strategies:
            if strat_name.startswith("Regression_"):
                sim_df = preds_df
            else:
                sim_df = baseline_df

            label = f"{strat_name}"
            if kwargs:
                param_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
                label = f"{strat_name} ({param_str})"

            safe_label = label.replace("(", "").replace(")", "").replace(", ", "_").replace("=", "")

            logger.info(f"  Running: {label}")
            try:
                metrics, history, holdings_df, trades_df = backtester.run_strategy(
                    sim_df, strat_name, **kwargs
                )
                metrics['Strategy'] = label
                results.append(metrics)

                if not holdings_df.empty:
                    out_holdings = self.report_dir / f"{safe_label}_holdings.csv"
                    holdings_df.to_csv(out_holdings, index=False)
                    plot_portfolio_composition(
                        holdings_df, label, plots_dir / f"{safe_label}_composition.png"
                    )

                if not trades_df.empty:
                    out_trades = self.report_dir / f"{safe_label}_trades.csv"
                    trades_df.to_csv(out_trades, index=False)
                    plot_trade_activity(
                        trades_df, label, plots_dir / f"{safe_label}_activity.png"
                    )

            except Exception as e:
                logger.warning(f"  Strategy {label} failed: {e}")

        # 7. Output
        if results:
            results_df = pd.DataFrame(results)
            cols = ['Strategy'] + [c for c in results_df.columns if c != 'Strategy']
            results_df = results_df[cols]

            logger.info("\n" + "=" * 80)
            logger.info("BACKTEST RESULTS")
            logger.info("=" * 80)
            print(results_df.to_string(index=False))

            out_csv = self.report_dir / "backtest_results.csv"
            results_df.to_csv(out_csv, index=False)
            logger.info(f"\nSaved backtest results to: {out_csv}")

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Report dir: {self.report_dir}")
        logger.info(f"Model: DARNN (sentiment={'ON' if self.use_sentiment else 'OFF'})")
        if not ic_df.empty:
            logger.info(f"Cross-Sectional IC: {mean_ic:.4f} (IR={ic_ir:.4f}, Hit={hit_rate:.2%})")
            logger.info(f"Quantile Spread:    {mean_raw_spread:.4%} daily, {cum_spread:.2%} cumulative")
        if results:
            reg_results = [r for r in results if 'Regression' in r.get('Strategy', '')]
            if reg_results:
                best = max(reg_results, key=lambda r: r.get('Sharpe Ratio', 0))
                logger.info(f"Best Regression:    {best['Strategy']} "
                           f"(Sharpe={best['Sharpe Ratio']:.2f}, "
                           f"Return={best['Total Return']:.2%})")
            bh = [r for r in results if 'Buy_Hold' in r.get('Strategy', '')]
            if bh:
                logger.info(f"Buy & Hold:         Sharpe={bh[0]['Sharpe Ratio']:.2f}, "
                           f"Return={bh[0]['Total Return']:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DA-RNN Model Backtester")
    parser.add_argument("--report-dir", type=str, required=True,
                        help="Path to DA-RNN experiment report directory")
    parser.add_argument("--start", type=str, default=None,
                        help="Backtest start date (default: from walk-forward config)")
    parser.add_argument("--end", type=str, default=None,
                        help="Backtest end date (default: from walk-forward config)")
    parser.add_argument("--top-n-test-stocks", type=int, default=None,
                        help="Override test set with top N stocks from the training universe")
    args = parser.parse_args()

    runner = DARNNBacktestRunner(
        args.report_dir,
        backtest_start=args.start,
        backtest_end=args.end,
        top_n_test_stocks=args.top_n_test_stocks,
    )
    runner.run()
