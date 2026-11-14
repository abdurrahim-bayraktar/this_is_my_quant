"""
Backtesting evaluator for DA-RNN Classification walk-forward experiments.

Loads a trained DARNNClassifier from a report directory, reconstructs the
feature/target scalers, generates class probability predictions across test
stocks, then runs portfolio strategy simulations.

This evaluator handles the classification DA-RNN specifics:
  - Dual-input model: X (features) + y_history (decoder target history)
  - Hybrid target: continuous y_history (vol-adj returns) for decoder,
    discrete class labels for prediction head
  - Full V7 indicator set (with leaky-indicator exclusions)
  - Separate feature scaler + target scaler (target scaler for y_history only)
  - Walk-forward fold structure (expanding-window training)
  - Classification output: 3-way logits → softmax → Prob_Up / Prob_Down

Usage:
    python backtesting/evaluate_darnn_classification.py --report-dir reports/wf_classification_darnn_20260912_...
    python backtesting/evaluate_darnn_classification.py --report-dir reports/... --top-n-test-stocks 50
"""

import sys
from pathlib import Path
import json
import logging
import argparse
import gc
import shutil
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cache import PriceCache
from src.data.utils import load_stocks
from src.features.indicators_v7 import ComprehensiveIndicatorsV7
from src.models.darnn_classifier import DARNNClassifier
from backtesting.framework import Backtester

try:
    from config import CACHE_DIR
except ImportError:
    CACHE_DIR = Path("data/cache")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS (mirrored from classification_darnn_rolling.py)
# ============================================================================

EXCLUDED_FEATURES = [
    'ichimoku_ICS_26',
    'dpo',
]


# ============================================================================
# DA-RNN CLASSIFICATION BACKTEST RUNNER
# ============================================================================

class DARNNClassificationBacktestRunner:
    """
    Loads a trained DARNNClassifier from a report directory and runs
    backtesting strategies using classification probabilities.

    Handles the DA-RNN-specific dual-input architecture, full V7 features,
    and hybrid target pipeline (continuous y_history + discrete predictions).
    """

    def __init__(
        self,
        report_dir: str,
        backtest_start: str = None,
        backtest_end: str = None,
        top_n_test_stocks: int = None,
        use_train_stocks: bool = False,
        tickers: list = None,
    ):
        self.report_dir = Path(report_dir)
        with open(self.report_dir / "config.json", "r") as f:
            self.config = json.load(f)

        # --- Model parameters ---
        self.frequency = self.config.get("frequency", "daily")
        self.sequence_length = self.config.get("sequence_length", 20 if self.frequency == "daily" else 12)
        self.dropout = self.config.get("dropout", 0.2)
        self.encoder_hidden = self.config.get("encoder_hidden", 64)
        self.decoder_hidden = self.config.get("decoder_hidden", 64)
        self.num_classes = self.config.get("num_classes", 3)
        self.excluded_features = self.config.get("excluded_features", EXCLUDED_FEATURES)

        # --- Vol-adj return params (for continuous y_history) ---
        self.vol_lookback = self.config.get("vol_lookback", 20)
        self.target_clip = self.config.get("target_clip", 10.0)

        # --- Classification thresholds (for reference logging) ---
        self.threshold_low = self.config.get("threshold_low", -0.005)
        self.threshold_high = self.config.get("threshold_high", 0.005)

        # --- Date range ---
        self.start_date = self.config.get("start_date", "2014-01-01")
        end_date_cfg = self.config.get("end_date", "2024-12-31")

        # Walk-forward folds define the eval period
        folds = self.config.get("folds", [])

        if folds:
            default_backtest_start = folds[0]["fold_start"]
            default_backtest_end = folds[-1]["fold_end"]
        else:
            default_backtest_start = self.config.get("wf_val_start", "2023-01-01")
            default_backtest_end = self.config.get("wf_val_end", "2024-12-31")

        self.backtest_start = pd.Timestamp(backtest_start or default_backtest_start)
        self.backtest_end = pd.Timestamp(backtest_end or default_backtest_end)
        self.end_date = max(pd.Timestamp(end_date_cfg), self.backtest_end).strftime('%Y-%m-%d')

        # --- Training period for scaler reconstruction ---
        if folds:
            self.train_end = pd.Timestamp(folds[-1]["fold_start"])
        else:
            self.train_end = pd.Timestamp(self.config.get("wf_val_start", "2023-01-01"))

        # --- Ticker universe ---
        self.train_stocks = self.config.get("train_stocks_list", [])
        original_test_stocks = self.config.get("test_stocks", [])
        self.top_n_test_stocks = top_n_test_stocks

        # Reconstruct train_stocks if not stored in config
        if not self.train_stocks:
            try:
                from experiments.ranked_tickers import EXTENDED_TICKERS
                train_stock_count = self.config.get("train_stock_count", 400)
                all_train = EXTENDED_TICKERS[:train_stock_count]
                self.train_stocks = [t for t in all_train if t not in original_test_stocks]
            except ImportError:
                logger.warning("Could not import EXTENDED_TICKERS for scaler reconstruction")

        if use_train_stocks and self.train_stocks:
            self.test_stocks = list(self.train_stocks)
            logger.info(f"Using train_stocks as evaluation universe: {len(self.test_stocks)} stocks")
        elif tickers:
            self.test_stocks = tickers
            logger.info(f"Using custom ticker list: {tickers}")
        elif self.top_n_test_stocks is not None:
            # Deterministic, ranked selection — identical to evaluate_ultimate_model.py
            # (previously sliced an unordered set(), which is random per Python process)
            try:
                from experiments.ranked_tickers import EXTENDED_TICKERS
                self.test_stocks = list(EXTENDED_TICKERS[:self.top_n_test_stocks])
                logger.info(f"Overriding test_stocks with top {self.top_n_test_stocks} from ranked_tickers.py")
            except ImportError:
                logger.warning("Cannot use --top-n-test-stocks without ranked_tickers.py; using config test_stocks")
                self.test_stocks = original_test_stocks
        else:
            self.test_stocks = original_test_stocks

        # --- Infrastructure ---
        self.indicator_computer = ComprehensiveIndicatorsV7()
        self.cache = PriceCache()
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create backtest output directory
        self.output_dir = self._create_output_dir()

    def _create_output_dir(self) -> Path:
        """Create a backtest output directory by copying summary + config from the report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"bt_{self.report_dir.name}_{timestamp}"
        output_dir = self.report_dir.parent / output_name
        output_dir.mkdir(parents=True, exist_ok=True)

        for fname in ["summary.csv", "config.json"]:
            src = self.report_dir / fname
            if src.exists():
                shutil.copy2(src, output_dir / fname)

        logger.info(f"Backtest output directory: {output_dir}")
        return output_dir

    # ------------------------------------------------------------------
    # Feature + target computation (mirrors classification_darnn_rolling.py)
    # ------------------------------------------------------------------

    def _compute_features_and_target(self, df: pd.DataFrame):
        """
        Compute full V7 features and hybrid targets for a single stock.

        Returns (feature_data, continuous_targets, dates, available_cols) or (None, None, None, None).
        """
        df = self.indicator_computer.compute_all(df)

        all_indicator_cols = self.indicator_computer.get_indicator_columns(df)
        available = [c for c in all_indicator_cols
                     if c not in self.excluded_features and c in df.columns]

        if len(available) < 5:
            return None, None, None, None

        # Continuous target: vol-adjusted forward return (for decoder y_history)
        df = df.copy()
        df['return_next'] = df['Close'].pct_change().shift(-1)
        df['rolling_vol'] = df['Close'].pct_change().rolling(self.vol_lookback).std()
        df['vol_adj_return'] = df['return_next'] / df['rolling_vol']
        df['vol_adj_return'] = df['vol_adj_return'].clip(-self.target_clip, self.target_clip)

        df = df.dropna(subset=['vol_adj_return'] + available[:5])
        if len(df) < self.sequence_length + 10:
            return None, None, None, None

        feature_data = df[available].values
        feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
        continuous_targets = df['vol_adj_return'].values.astype(np.float32)

        return feature_data, continuous_targets, df.index, available

    # ------------------------------------------------------------------
    # DA-RNN sequence creation (for backtest)
    # ------------------------------------------------------------------

    def _create_sequences_with_dates(self, feature_data, continuous_targets, dates, period_start, period_end):
        """
        Create DA-RNN sequences for a date period, returning dates too.

        Returns: (X, y_history, trade_dates)
        y_history is continuous (for decoder input).
        """
        X_list, yh_list, date_list = [], [], []

        for i in range(1, len(feature_data) - self.sequence_length + 1):
            target_idx = i + self.sequence_length - 1
            target_date = dates[target_idx]

            if period_start <= target_date < period_end:
                seq_x = feature_data[i : i + self.sequence_length]
                seq_yh = continuous_targets[i - 1 : i + self.sequence_length - 1]

                X_list.append(seq_x)
                yh_list.append(seq_yh)
                date_list.append(target_date)

        def _to_array(lst, dtype):
            return np.array(lst, dtype=dtype) if lst else np.array([], dtype=dtype)

        return (
            _to_array(X_list, np.float32),
            _to_array(yh_list, np.float32),
            date_list,
        )

    def _create_train_sequences(self, feature_data, continuous_targets, dates, train_end):
        """Create training sequences (for scaler fitting). No dates needed."""
        X_list, yh_list, y_list = [], [], []

        for i in range(1, len(feature_data) - self.sequence_length + 1):
            target_date = dates[i + self.sequence_length - 1]
            if target_date < train_end:
                X_list.append(feature_data[i : i + self.sequence_length])
                yh_list.append(continuous_targets[i - 1 : i + self.sequence_length - 1])
                y_list.append(continuous_targets[i + self.sequence_length - 1])

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

        The DA-RNN classification uses two scalers:
          - feature_scaler: StandardScaler on flattened feature sequences
          - target_scaler: StandardScaler on continuous y_history values (NOT class labels)
        """
        logger.info("Reconstructing StandardScalers from training data...")
        all_train_X = []
        all_train_yh = []
        n_features = None

        for ticker in tqdm(self.train_stocks, desc="Fitting Scalers"):
            if ticker not in stock_data:
                continue

            result = self._compute_features_and_target(stock_data[ticker])
            if result[0] is None:
                continue

            feat, cont_tgt, dates, _ = result

            X_tr, yh_tr, _ = self._create_train_sequences(feat, cont_tgt, dates, self.train_end)
            if len(X_tr) > 0:
                if n_features is None:
                    n_features = X_tr.shape[-1]
                if X_tr.shape[-1] == n_features:
                    all_train_X.append(X_tr)
                    all_train_yh.append(yh_tr)

        if all_train_X:
            X_train = np.concatenate(all_train_X, axis=0)
            yh_train = np.concatenate(all_train_yh, axis=0)

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

            # Target scaler — fit on y_history (continuous decoder input)
            self.target_scaler.fit(yh_train.reshape(-1, 1))

            logger.info(f"Scalers fitted on {n_train:,} sequences × {feat} features, "
                        f"{len(yh_train):,} y_history sequences.")
            return feat
        else:
            logger.error("No training data found to fit scalers!")
            return 0

    # ------------------------------------------------------------------
    # Prediction generation
    # ------------------------------------------------------------------

    def prepare_test_predictions(self, stock_data, n_features):
        """
        Generate test-period class probability predictions for all test stocks.

        Returns DataFrame with:
            Date, Ticker, Prob_Up, Prob_Down, Return_Next, Label
        """
        self.model.eval()
        all_predictions = []

        for ticker in tqdm(self.test_stocks, desc="Predicting Test Set"):
            if ticker not in stock_data:
                continue

            result = self._compute_features_and_target(stock_data[ticker])
            if result[0] is None:
                continue

            feat, cont_tgt, dates, _ = result

            X_test, yh_test, test_dates = self._create_sequences_with_dates(
                feat, cont_tgt, dates, self.backtest_start, self.backtest_end,
            )

            if len(X_test) == 0:
                continue

            n_samples, seq_len, n_feat = X_test.shape

            # Scale features
            X_scaled = self.feature_scaler.transform(X_test.reshape(-1, n_feat))
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            X_scaled = X_scaled.reshape(n_samples, seq_len, n_feat).astype(np.float32)

            # Scale decoder history (continuous)
            yh_scaled = self.target_scaler.transform(
                yh_test.reshape(-1, 1)
            ).reshape(yh_test.shape).astype(np.float32)

            # Predict in batches → logits → softmax → Prob_Up / Prob_Down
            batch_size = 2048
            probs_up, probs_down, probs_neutral = [], [], []

            with torch.no_grad():
                for i in range(0, n_samples, batch_size):
                    batch_x = torch.FloatTensor(X_scaled[i:i+batch_size]).to(self.device)
                    batch_yh = torch.FloatTensor(yh_scaled[i:i+batch_size]).to(self.device)
                    result_pred = self.model.predict(batch_x, batch_yh)
                    probs = torch.softmax(result_pred["logits"], dim=-1).cpu().numpy()

                    # Classes: 0=Down, 1=Neutral, 2=Up
                    probs_down.extend(probs[:, 0])
                    probs_neutral.extend(probs[:, 1])
                    probs_up.extend(probs[:, 2])

            # Get the raw next-day return for PnL computation
            df_ticker = stock_data[ticker].copy()
            df_ticker['return_next'] = df_ticker['Close'].pct_change().shift(-1)

            # Also compute the classification label for reference
            df_ticker['trend'] = pd.cut(
                df_ticker['return_next'],
                bins=[-np.inf, self.threshold_low, self.threshold_high, np.inf],
                labels=[0, 1, 2]
            ).astype(float)

            test_records = []
            for j, date in enumerate(test_dates):
                try:
                    raw_ret = df_ticker.loc[date, 'return_next']
                    if isinstance(raw_ret, pd.Series):
                        raw_ret = raw_ret.iloc[0]
                except (KeyError, IndexError):
                    raw_ret = 0.0

                if raw_ret is None or (isinstance(raw_ret, float) and np.isnan(raw_ret)):
                    raw_ret = 0.0

                try:
                    label = df_ticker.loc[date, 'trend']
                    if isinstance(label, pd.Series):
                        label = label.iloc[0]
                    label = int(label) if not np.isnan(label) else 1
                except (KeyError, IndexError, ValueError):
                    label = 1  # Neutral default

                test_records.append({
                    'Date': date,
                    'Ticker': ticker,
                    'Prob_Up': float(probs_up[j]),
                    'Prob_Down': float(probs_down[j]),
                    'Prob_Neutral': float(probs_neutral[j]),
                    'Return_Next': float(raw_ret),
                    'Label': label,
                })

            all_predictions.append(pd.DataFrame(test_records))

        if not all_predictions:
            raise ValueError("No test predictions generated. Check test stocks vs available data.")

        return pd.concat(all_predictions, ignore_index=True)

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------

    def run(self):
        logger.info(f"Loading DA-RNN Classification experiment from: {self.report_dir}")
        logger.info(f"Train stocks: {len(self.train_stocks)}, Test stocks: {len(self.test_stocks)}")
        logger.info(f"Backtest period: {self.backtest_start.date()} → {self.backtest_end.date()}")
        logger.info(f"Train end (scaler fit boundary): {self.train_end.date()}")
        logger.info(f"Classification thresholds: [{self.threshold_low}, {self.threshold_high}]")

        # 1. Load data
        stock_data = self.load_data()

        # 2. Reconstruct scalers
        n_features = self.reconstruct_scalers(stock_data)
        if n_features == 0:
            logger.error("Failed to reconstruct scalers — aborting")
            return

        # 3. Load model
        logger.info("Loading DARNNClassifier model weights...")
        self.model = DARNNClassifier(
            input_size=n_features,
            seq_length=self.sequence_length,
            encoder_hidden=self.encoder_hidden,
            decoder_hidden=self.decoder_hidden,
            output_size=self.num_classes,
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
        logger.info(f"DARNNClassifier loaded: {sum(p.numel() for p in self.model.parameters()):,} params")

        # 4. Generate predictions
        preds_df = self.prepare_test_predictions(stock_data, n_features)
        preds_df.to_csv(self.output_dir / "predictions.csv", index=False)
        logger.info(f"Generated {len(preds_df):,} predictions across "
                    f"{preds_df['Ticker'].nunique()} stocks")

        n_dates = preds_df['Date'].nunique()
        logger.info(f"Date range: {preds_df['Date'].min()} → {preds_df['Date'].max()} "
                    f"({n_dates} dates)")

        # 5. Classification accuracy summary
        logger.info("\n" + "=" * 60)
        logger.info("CLASSIFICATION ACCURACY SUMMARY")
        logger.info("=" * 60)

        # Argmax over all 3 classes (0=Down, 1=Neutral, 2=Up).
        # Previously only Up vs Down was compared, so Neutral was never predicted.
        pred_classes = preds_df[['Prob_Down', 'Prob_Neutral', 'Prob_Up']].values.argmax(axis=1)
        actual_labels = preds_df['Label'].values
        accuracy = (pred_classes == actual_labels).mean()
        logger.info(f"Overall 3-class accuracy: {accuracy:.2%}")
        logger.info(f"Predicted class dist: "
                    f"{pd.Series(pred_classes).value_counts(normalize=True).sort_index().round(3).to_dict()}")
        logger.info(f"Label dist:           "
                    f"{pd.Series(actual_labels).value_counts(normalize=True).sort_index().round(3).to_dict()}")
        logger.info(f"Mean Prob_Up:   {preds_df['Prob_Up'].mean():.4f}")
        logger.info(f"Mean Prob_Down: {preds_df['Prob_Down'].mean():.4f}")

        # 6. Run strategies
        logger.info("\n" + "=" * 60)
        logger.info("STRATEGY SIMULATION")
        logger.info("=" * 60)

        backtester = Backtester(initial_capital=100000.0)

        strategies = [
            # Baselines
            ("Buy_Hold_Universe", {}),
            ("Daily_Rebalanced_Universe", {}),
            ("Random_Allocation", {'n': 5}),
            # Classification strategies
            ("Long_Top_N", {'n': 3}),
            ("Long_Top_N", {'n': 5}),
            ("Long_Short_Neutral", {'n': 3}),
            ("Long_Short_Neutral", {'n': 5}),
            ("Threshold_Long", {'threshold': 0.40}),
            ("Threshold_Long", {'threshold': 0.50}),
            ("Threshold_Long", {'threshold': 0.60}),
        ]

        # Ensure plots directory exists
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        from src.evaluation.plots import plot_portfolio_composition, plot_trade_activity

        results = []
        for strat_name, kwargs in strategies:
            label = f"{strat_name}"
            if kwargs:
                param_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
                label = f"{strat_name} ({param_str})"

            safe_label = label.replace("(", "").replace(")", "").replace(", ", "_").replace("=", "")

            logger.info(f"  Running: {label}")
            try:
                metrics, history, holdings_df, trades_df = backtester.run_strategy(
                    preds_df, strat_name, **kwargs
                )
                metrics['Strategy'] = label
                results.append(metrics)

                if not holdings_df.empty:
                    out_holdings = self.output_dir / f"{safe_label}_holdings.csv"
                    holdings_df.to_csv(out_holdings, index=False)
                    plot_portfolio_composition(
                        holdings_df, label, plots_dir / f"{safe_label}_composition.png"
                    )

                if not trades_df.empty:
                    out_trades = self.output_dir / f"{safe_label}_trades.csv"
                    trades_df.to_csv(out_trades, index=False)
                    plot_trade_activity(
                        trades_df, label, plots_dir / f"{safe_label}_activity.png"
                    )

            except Exception as e:
                logger.warning(f"  Strategy {label} failed: {e}")

        # 7. Output
        if results:
            results_df = pd.DataFrame(results)
            
            results_df.insert(1, 'Start Date', self.backtest_start.strftime('%Y-%m-%d'))
            results_df.insert(2, 'End Date', self.backtest_end.strftime('%Y-%m-%d'))
            
            cols = ['Strategy', 'Start Date', 'End Date'] + [c for c in results_df.columns if c not in ['Strategy', 'Start Date', 'End Date']]
            results_df = results_df[cols]

            logger.info("\n" + "=" * 80)
            logger.info("BACKTEST RESULTS")
            logger.info("=" * 80)
            print(results_df.to_string(index=False))

            out_csv = self.output_dir / "backtest_results.csv"
            results_df.to_csv(out_csv, index=False)
            logger.info(f"\nSaved backtest results to: {out_csv}")

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Report dir: {self.report_dir}")
        logger.info(f"Output dir: {self.output_dir}")
        logger.info(f"Model: DARNNClassifier (classes={self.num_classes})")
        logger.info(f"Classification Accuracy: {accuracy:.2%}")
        if results:
            cls_results = [r for r in results if r.get('Strategy', '').startswith(('Long_', 'Threshold_'))]
            if cls_results:
                best = max(cls_results, key=lambda r: r.get('Sharpe Ratio', 0))
                logger.info(f"Best Strategy:    {best['Strategy']} "
                           f"(Sharpe={best['Sharpe Ratio']:.2f}, "
                           f"Return={best['Total Return']:.2%})")
            bh = [r for r in results if 'Buy_Hold' in r.get('Strategy', '')]
            if bh:
                logger.info(f"Buy & Hold:       Sharpe={bh[0]['Sharpe Ratio']:.2f}, "
                           f"Return={bh[0]['Total Return']:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DA-RNN Classification Model Backtester")
    parser.add_argument("--report-dir", type=str, required=True,
                        help="Path to DA-RNN classification experiment report directory")
    parser.add_argument("--start", type=str, default=None,
                        help="Backtest start date (default: from walk-forward config)")
    parser.add_argument("--end", type=str, default=None,
                        help="Backtest end date (default: from walk-forward config)")
    parser.add_argument("--top-n-test-stocks", type=int, default=None,
                        help="Override test set with top N stocks from ranked_tickers.py (same as evaluate_ultimate_model.py)")
    parser.add_argument("--use-train-stocks", action="store_true",
                        help="Evaluate on train_stocks_list from config instead of test_stocks. "
                             "Useful for comparing with rule-based baselines on the same universe.")
    parser.add_argument("--tickers", type=str, default=None,
                        help="Comma-separated list of tickers to backtest on (e.g. AAPL,MSFT,GOOG)")
    args = parser.parse_args()

    custom_tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None

    runner = DARNNClassificationBacktestRunner(
        args.report_dir,
        backtest_start=args.start,
        backtest_end=args.end,
        top_n_test_stocks=args.top_n_test_stocks,
        use_train_stocks=args.use_train_stocks,
        tickers=custom_tickers,
    )
    runner.run()
