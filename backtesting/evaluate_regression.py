"""
Backtesting evaluator for the regression_v1 experiment.

Loads a trained RegressionAttentionLSTM from a report directory, reconstructs
the scaler, generates predictions across test stocks, then runs both:
  1. Cross-sectional IC time series (ranking quality over time)
  2. Portfolio strategy simulations (regression + baseline strategies)

Usage:
    python backtesting/evaluate_regression.py --report-dir reports/regression_v1_smoke_20260325_135809
    python backtesting/evaluate_regression.py --report-dir reports/regression_v1_full_scale_20260325_140551
"""

import sys
from pathlib import Path
import json
import logging
import argparse
import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cache import PriceCache
from src.data.utils import load_stocks
from src.features.indicators_v7 import ComprehensiveIndicatorsV7
from sklearn.preprocessing import StandardScaler
from backtesting.framework import Backtester

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Inline model definition (mirrors experiments/regression_v1.py)
# Kept inline to avoid import-time side effects from the experiment script.
# ============================================================================

import torch.nn as nn
from typing import Tuple, Optional, Dict


class SentimentAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        attended, weights = self.attention(x, x, x)
        attended = self.layer_norm(attended + x)
        return attended, weights


class RegressionAttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2, num_heads=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = SentimentAttention(hidden_size, num_heads, dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(dropout),
        )
        self.regression_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout / 2), nn.Linear(32, 1),
        )

    def forward(self, x, return_attention=False):
        lstm_out, _ = self.lstm(x)
        attended, attn_weights = self.attention(lstm_out)
        final = attended[:, -1, :]
        shared = self.fc(final)
        prediction = self.regression_head(shared).squeeze(-1)
        if return_attention:
            return prediction, None, attn_weights
        return prediction, None, None

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            prediction, _, _ = self.forward(x)
            return {"prediction": prediction}


# ============================================================================
# FEATURE LIST (same as regression_v1.py)
# ============================================================================

SHAP_TOP20_FEATURES = [
    "atr_pct", "intraday_range", "bb_BBB_5_2.0_2.0", "gap", "cci",
    "high_low_pct", "volume_ratio", "adx_DMN_14", "return_1d", "aroon_AROOND_14",
    "adx_DMP_14", "obv", "ad", "bear_power", "roc_10",
    "pvo_PVOh_12_26_9", "trix_TRIXs_30_9", "rsi_14", "aroon_AROONU_14", "return_5d"
]


# ============================================================================
# REGRESSION BACKTEST RUNNER
# ============================================================================

class RegressionBacktestRunner:
    """
    Loads a trained regression model from a report directory and runs
    backtesting strategies + cross-sectional IC analysis.
    """

    def __init__(self, report_dir: str, backtest_start: str = None, backtest_end: str = "2024-12-31", top_n_test_stocks: int = None):
        self.report_dir = Path(report_dir)
        with open(self.report_dir / "config.json", "r") as f:
            self.config = json.load(f)

        self.train_stock_count = self.config.get("train_stock_count", 400)
        self.top_n_test_stocks = top_n_test_stocks
        original_test_stocks = self.config.get("test_stocks", [])

        # Reconstruct train_stocks exactly as they were during training
        try:
            from experiments.ranked_tickers import EXTENDED_TICKERS
            all_train = EXTENDED_TICKERS[:self.train_stock_count]
            # Exclude original_test_stocks to perfectly match training scaler
            self.train_stocks = [t for t in all_train if t not in original_test_stocks]
            
            # Now set the actual test_stocks for evaluation
            if self.top_n_test_stocks is not None:
                self.test_stocks = EXTENDED_TICKERS[:self.top_n_test_stocks]
                logger.info(f"Overriding test_stocks with top {self.top_n_test_stocks} from ranked_tickers.py")
            else:
                self.test_stocks = original_test_stocks
        except ImportError:
            logger.warning("Could not import EXTENDED_TICKERS, using test_stocks from config only.")
            if self.top_n_test_stocks is not None:
                logger.warning("Cannot use --top-n-test-stocks without ranked_tickers.py")
            self.test_stocks = original_test_stocks
            self.train_stocks = []

        self.frequency = self.config.get("frequency", "daily")
        self.sequence_length = self.config.get("sequence_length", 20 if self.frequency == "daily" else 12)
        self.vol_lookback = self.config.get("vol_lookback", 20)
        self.target_clip = self.config.get("target_clip", 10.0)
        self.dropout = self.config.get("dropout", 0.2)

        # Temporal boundaries
        self.start_date = "2014-01-01"
        self.train_end = pd.Timestamp("2022-01-01")
        self.val_end = pd.Timestamp("2023-06-01")

        self.backtest_start = pd.Timestamp(backtest_start) if backtest_start else self.val_end
        self.backtest_end = pd.Timestamp(backtest_end)
        self.end_date = self.backtest_end.strftime('%Y-%m-%d')

        self.feature_cols = self.config.get("features", SHAP_TOP20_FEATURES)
        self.indicator_computer = ComprehensiveIndicatorsV7()
        self.cache = PriceCache()
        self.scaler = StandardScaler()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Data loading & feature extraction
    # ------------------------------------------------------------------

    def load_data(self):
        all_tickers = list(set(self.train_stocks + self.test_stocks))
        weekly = self.frequency == "weekly"
        return load_stocks(
            all_tickers, self.start_date, self.end_date,
            cache=self.cache, min_rows=100 if not weekly else 52, weekly=weekly,
        )

    def extract_sequences_with_dates(self, df: pd.DataFrame) -> tuple:
        """
        Extract windowed sequences, returning:
          - train_arr: training sequences (for scaler fitting)
          - test_df: DataFrame with Date, Sequence, Return_Next for test period
        """
        df = self.indicator_computer.compute_shap_top20(df)
        available = [c for c in self.feature_cols if c in df.columns]
        if len(available) < 5:
            return np.empty((0,)), pd.DataFrame()

        # Volatility-adjusted forward return (regression target)
        df['return_next_raw'] = df['Close'].pct_change().shift(-1)
        df['rolling_vol'] = df['Close'].pct_change().rolling(self.vol_lookback).std()
        df['vol_adj_return'] = df['return_next_raw'] / df['rolling_vol']
        df['vol_adj_return'] = df['vol_adj_return'].clip(-self.target_clip, self.target_clip)

        df = df.dropna(subset=['vol_adj_return'] + available[:5])
        if len(df) < self.sequence_length + 10:
            return np.empty((0,)), pd.DataFrame()

        dates = df.index
        feature_data = df[available].values
        feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)

        train_seqs = []
        test_data = []

        for i in range(len(feature_data) - self.sequence_length):
            trade_date = dates[i + self.sequence_length - 1]
            target_date = dates[i + self.sequence_length]

            seq = feature_data[i:i + self.sequence_length]
            ret_next = df['return_next_raw'].iloc[i + self.sequence_length - 1]
            vol_adj_ret = df['vol_adj_return'].iloc[i + self.sequence_length - 1]

            if target_date < self.train_end:
                train_seqs.append(seq)
            elif target_date >= self.backtest_start and target_date <= self.backtest_end:
                test_data.append({
                    'Date': trade_date,
                    'Sequence': seq,
                    'Return_Next': ret_next,        # raw next-day return for PnL
                    'Actual_Vol_Adj': vol_adj_ret,   # vol-adj return for IC calc
                })

        n_feat = len(available)
        train_arr = np.array(train_seqs) if train_seqs else np.empty((0, self.sequence_length, n_feat))
        test_df = pd.DataFrame(test_data)
        return train_arr, test_df

    # ------------------------------------------------------------------
    # Scaler reconstruction
    # ------------------------------------------------------------------

    def reconstruct_scaler(self, stock_data):
        """Fit scaler on train_stocks up to train_end (no leakage)."""
        logger.info("Reconstructing StandardScaler from training data...")
        all_train_X = []
        n_features = None

        for ticker in tqdm(self.train_stocks, desc="Fitting Scaler"):
            if ticker not in stock_data:
                continue
            train_arr, _ = self.extract_sequences_with_dates(stock_data[ticker])
            if len(train_arr) > 0:
                if n_features is None:
                    n_features = train_arr.shape[-1]
                if train_arr.shape[-1] == n_features:
                    all_train_X.append(train_arr)

        if all_train_X:
            X_train = np.concatenate(all_train_X, axis=0)
            n_train, seq_len, feat = X_train.shape
            self.scaler.fit(X_train.reshape(-1, feat))
            logger.info(f"Scaler fitted on {n_train:,} sequences × {feat} features.")
            return feat
        else:
            logger.error("No training data found to fit scaler!")
            return len(self.feature_cols)

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

            _, test_df = self.extract_sequences_with_dates(stock_data[ticker])
            if test_df.empty:
                continue

            seqs = np.stack(test_df['Sequence'].values)
            n_samples, seq_len, feat = seqs.shape

            # Scale using training scaler
            seqs_scaled = self.scaler.transform(seqs.reshape(-1, feat))
            seqs_scaled = np.nan_to_num(seqs_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            seqs_scaled = seqs_scaled.reshape(n_samples, seq_len, feat)

            # Predict in batches
            batch_size = 2048
            preds = []
            with torch.no_grad():
                for i in range(0, n_samples, batch_size):
                    batch_x = torch.FloatTensor(seqs_scaled[i:i+batch_size]).to(self.device)
                    result = self.model.predict(batch_x)
                    preds.append(result["prediction"].cpu().numpy())

            predictions = np.concatenate(preds)

            test_df = test_df.copy()
            test_df['Pred_Return'] = predictions
            test_df['Ticker'] = ticker

            all_predictions.append(
                test_df[['Date', 'Ticker', 'Pred_Return', 'Return_Next', 'Actual_Vol_Adj']]
            )

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

        Returns DataFrame indexed by Date with columns:
            IC, N_Stocks, Quantile_Spread, Top_Q_Return, Bottom_Q_Return
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

            # Quantile spread: top 20% actual return minus bottom 20%
            n = len(preds)
            q_size = max(1, n // 5)
            sorted_idx = np.argsort(preds)
            bottom_ret = actuals[sorted_idx[:q_size]].mean()
            top_ret = actuals[sorted_idx[-q_size:]].mean()

            # Also compute raw return spread for PnL interpretation
            raw_actuals = group['Return_Next'].values
            bottom_raw = raw_actuals[sorted_idx[:q_size]].mean()
            top_raw = raw_actuals[sorted_idx[-q_size:]].mean()

            records.append({
                'Date': date,
                'IC': ic,
                'N_Stocks': n,
                'Quantile_Spread': top_ret - bottom_ret,       # vol-adj
                'Top_Q_Return': top_ret,
                'Bottom_Q_Return': bottom_ret,
                'Raw_Spread': top_raw - bottom_raw,             # actual return spread
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
        logger.info(f"Loading regression experiment from: {self.report_dir}")
        logger.info(f"Test stocks: {self.test_stocks}")
        logger.info(f"Backtest period: {self.backtest_start.date()} → {self.backtest_end.date()}")

        # 1. Load data
        stock_data = self.load_data()

        # 2. Reconstruct scaler
        n_features = self.reconstruct_scaler(stock_data)

        # 3. Load model
        logger.info("Loading model weights...")
        self.model = RegressionAttentionLSTM(
            input_size=n_features,
            hidden_size=64,
            num_layers=1,
            dropout=0.0,  # inference mode, no dropout
        )
        state_dict = torch.load(
            self.report_dir / "model.pt",
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded: {sum(p.numel() for p in self.model.parameters()):,} params")

        # 4. Generate predictions
        preds_df = self.prepare_test_predictions(stock_data, n_features)
        logger.info(f"Generated {len(preds_df):,} predictions across {preds_df['Ticker'].nunique()} stocks")

        n_dates = preds_df['Date'].nunique()
        logger.info(f"Date range: {preds_df['Date'].min()} → {preds_df['Date'].max()} ({n_dates} dates)")

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

            # Save IC time series
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
            # Baselines (use Return_Next only, no model signal)
            ("Buy_Hold_Universe", {}),
            ("Daily_Rebalanced_Universe", {}),
            ("Random_Allocation", {'n': 3}),
            # Regression strategies (use Pred_Return)
            ("Regression_Long_Top_Pct", {'long_pct': 0.10}),
            ("Regression_Long_Top_Pct", {'long_pct': 0.20}),
            ("Regression_Long_Top_Pct", {'long_pct': 0.30}),
            ("Regression_Long_Short", {'long_pct': 0.20, 'short_pct': 0.20}),
            ("Regression_Long_Short", {'long_pct': 0.10, 'short_pct': 0.10}),
            ("Regression_Quantile_Spread", {}),
            ("Regression_Threshold_Long", {'threshold': 0.0}),
            ("Regression_Threshold_Long", {'threshold': 0.005}),
        ]

        # Baselines need Prob_Up/Prob_Down or at minimum just Return_Next.
        # For baseline strategies that don't use Prob_Up, we can pass dummy columns.
        baseline_df = preds_df.copy()
        baseline_df['Prob_Up'] = 0.33
        # Ensure plots directory exists
        plots_dir = self.report_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Import plotting functions inline to avoid circular imports if any
        from src.evaluation.plots import plot_portfolio_composition, plot_trade_activity

        results = []
        for strat_name, kwargs in strategies:
            # Baseline strategies need dummy Prob_Up/Prob_Down
            if strat_name.startswith("Regression_"):
                sim_df = preds_df
            else:
                sim_df = baseline_df

            label = f"{strat_name}"
            if kwargs:
                param_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
                label = f"{strat_name} ({param_str})"
                
            # Safely create filename from label, replacing problematic characters
            safe_label = label.replace("(", "").replace(")", "").replace(", ", "_").replace("=", "")

            logger.info(f"  Running: {label}")
            try:
                metrics, history, holdings_df, trades_df = backtester.run_strategy(sim_df, strat_name, **kwargs)
                metrics['Strategy'] = label
                results.append(metrics)
                
                # Save holding and trade logs
                if not holdings_df.empty:
                    out_holdings = self.report_dir / f"{safe_label}_holdings.csv"
                    holdings_df.to_csv(out_holdings, index=False)
                    
                    # Generate plot
                    plot_portfolio_composition(holdings_df, label, plots_dir / f"{safe_label}_composition.png")
                    
                if not trades_df.empty:
                    out_trades = self.report_dir / f"{safe_label}_trades.csv"
                    trades_df.to_csv(out_trades, index=False)
                    
                    # Generate plot
                    plot_trade_activity(trades_df, label, plots_dir / f"{safe_label}_activity.png")
                    
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

        # Print final summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Report dir: {self.report_dir}")
        if not ic_df.empty:
            logger.info(f"Cross-Sectional IC: {mean_ic:.4f} (IR={ic_ir:.4f}, Hit={hit_rate:.2%})")
            logger.info(f"Quantile Spread:    {mean_raw_spread:.4%} daily, {cum_spread:.2%} cumulative")
        if results:
            # Find best regression strategy by Sharpe
            reg_results = [r for r in results if 'Regression' in r.get('Strategy', '')]
            if reg_results:
                best = max(reg_results, key=lambda r: r.get('Sharpe Ratio', 0))
                logger.info(f"Best Regression:    {best['Strategy']} "
                           f"(Sharpe={best['Sharpe Ratio']:.2f}, "
                           f"Return={best['Total Return']:.2%})")
            # Find Buy & Hold for comparison
            bh = [r for r in results if 'Buy_Hold' in r.get('Strategy', '')]
            if bh:
                logger.info(f"Buy & Hold:         Sharpe={bh[0]['Sharpe Ratio']:.2f}, "
                           f"Return={bh[0]['Total Return']:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regression Model Backtester")
    parser.add_argument("--report-dir", type=str, required=True,
                        help="Path to regression experiment report directory")
    parser.add_argument("--start", type=str, default=None,
                        help="Backtest start date (default: 2023-06-01)")
    parser.add_argument("--end", type=str, default="2024-12-31",
                        help="Backtest end date")
    parser.add_argument("--top-n-test-stocks", type=int, default=None,
                        help="Override test set with top N stocks from ranked_tickers.py")
    args = parser.parse_args()

    runner = RegressionBacktestRunner(
        args.report_dir,
        backtest_start=args.start,
        backtest_end=args.end,
        top_n_test_stocks=args.top_n_test_stocks,
    )
    runner.run()
