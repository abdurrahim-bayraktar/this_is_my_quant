"""
Backtesting evaluator for the Karash rule-based algorithm.

Runs the Karash composite scoring system across a stock universe on a
rolling daily basis, then feeds the resulting signals into the standard
backtesting framework alongside identical baseline strategies.

Stock universe options:
  1. Full EXTENDED_TICKERS from ranked_tickers.py (default)
  2. Constrained to train_stocks_list from a model report config.json (--report-dir)
  3. Manually limited via --top-n-stocks

Usage:
    python backtesting/evaluate_karash.py
    python backtesting/evaluate_karash.py --top-n-stocks 50 --start 2023-06-01 --end 2024-12-31
    python backtesting/evaluate_karash.py --report-dir reports/wf_darnn_baseline_20260612_011221
"""

import sys
from pathlib import Path
import json
import logging
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cache import PriceCache
from src.data.utils import load_stocks
from backtesting.framework import Backtester
from backtesting.rule_based.karash import compute_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# KARASH BACKTEST RUNNER
# ============================================================================

class KarashBacktestRunner:
    """
    Evaluates the Karash rule-based scoring algorithm across a stock universe
    using the standard backtesting framework.
    """

    def __init__(
        self,
        report_dir: str = None,
        top_n_stocks: int = None,
        backtest_start: str = "2023-06-01",
        backtest_end: str = "2024-12-31",
        data_start: str = "2014-01-01",
        output_dir: str = None,
    ):
        self.backtest_start = pd.Timestamp(backtest_start)
        self.backtest_end = pd.Timestamp(backtest_end)
        self.data_start = data_start
        self.cache = PriceCache()

        # Determine GPU availability for kNN component
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device != 'cuda':
            logger.warning(
                "CUDA not available — kNN ML component will run on CPU. "
                "The ML sub-score may produce zeros for long sequences."
            )

        # Determine stock universe
        self.test_stocks = self._resolve_stock_universe(report_dir, top_n_stocks)

        # Output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        elif report_dir:
            self.output_dir = Path(report_dir) / "karash_eval"
        else:
            self.output_dir = Path("reports") / "karash_standalone"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_stock_universe(self, report_dir: str, top_n_stocks: int) -> list:
        """
        Resolve the stock universe in priority order:
        1. If --report-dir given, use train_stocks_list from its config.json
        2. If --top-n-stocks given, take that many from EXTENDED_TICKERS
        3. Otherwise, use full EXTENDED_TICKERS
        """
        if report_dir:
            config_path = Path(report_dir) / "config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)
                train_stocks = config.get("train_stocks_list", [])
                if train_stocks:
                    logger.info(
                        f"Using {len(train_stocks)} train stocks from "
                        f"{config_path}"
                    )
                    return train_stocks
                else:
                    logger.warning(
                        f"config.json found but no train_stocks_list; "
                        f"falling back to EXTENDED_TICKERS"
                    )
            else:
                logger.warning(f"No config.json found at {config_path}")

        # Fall back to ranked_tickers
        try:
            from experiments.ranked_tickers import EXTENDED_TICKERS
        except ImportError:
            logger.error("Cannot import EXTENDED_TICKERS from experiments.ranked_tickers")
            raise

        if top_n_stocks is not None:
            tickers = EXTENDED_TICKERS[:top_n_stocks]
            logger.info(f"Using top {top_n_stocks} from EXTENDED_TICKERS")
        else:
            tickers = EXTENDED_TICKERS
            logger.info(f"Using full EXTENDED_TICKERS ({len(tickers)} stocks)")

        return tickers

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self) -> dict:
        """Load OHLCV data for all stocks in the universe."""
        return load_stocks(
            self.test_stocks,
            self.data_start,
            self.backtest_end.strftime("%Y-%m-%d"),
            cache=self.cache,
            min_rows=250,  # Need 210+ bars for Karash minimum
        )

    # ------------------------------------------------------------------
    # Rolling signal generation
    # ------------------------------------------------------------------

    def generate_signals(self, stock_data: dict) -> pd.DataFrame:
        """
        Run Karash compute_score() on a rolling basis for each stock.

        For each evaluation date t, passes df[:t] to compute_score() so
        that only past data is used (no look-ahead bias).

        Returns DataFrame with columns:
            Date, Ticker, Karash_Score, Karash_Signal, Return_Next
        """
        all_records = []

        for ticker in tqdm(self.test_stocks, desc="Computing Karash Scores"):
            if ticker not in stock_data:
                continue

            df = stock_data[ticker].copy()

            # Ensure sorted by date
            df = df.sort_index()

            # Compute next-day return for PnL tracking
            df['Return_Next'] = df['Close'].pct_change().shift(-1)

            # Find evaluation dates within the backtest window
            eval_mask = (df.index >= self.backtest_start) & (df.index <= self.backtest_end)
            eval_dates = df.index[eval_mask]

            if len(eval_dates) == 0:
                continue

            for date in eval_dates:
                # Pass all data up to and including this date
                history = df.loc[:date]

                if len(history) < 210:
                    continue

                try:
                    result = compute_score(history, params={"device": self.device})
                except Exception as e:
                    # Gracefully handle any per-stock failures
                    logger.debug(f"  {ticker} @ {date}: compute_score failed: {e}")
                    continue

                ret_next = df.loc[date, 'Return_Next']
                if pd.isna(ret_next):
                    continue

                all_records.append({
                    'Date': date,
                    'Ticker': ticker,
                    'Karash_Score': result['score'],
                    'Karash_Signal': result['signal'],
                    'Return_Next': ret_next,
                })

        if not all_records:
            raise ValueError("No Karash signals generated. Check stock data availability.")

        signals_df = pd.DataFrame(all_records)
        logger.info(
            f"Generated {len(signals_df):,} signals across "
            f"{signals_df['Ticker'].nunique()} stocks, "
            f"{signals_df['Date'].nunique()} dates"
        )

        # Summary statistics
        score_stats = signals_df['Karash_Score'].describe()
        logger.info(f"Score distribution:\n{score_stats.to_string()}")

        signal_counts = signals_df['Karash_Signal'].value_counts()
        logger.info(f"Signal counts:\n{signal_counts.to_string()}")

        return signals_df

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------

    def run(self):
        logger.info("=" * 60)
        logger.info("KARASH RULE-BASED BACKTEST")
        logger.info("=" * 60)
        logger.info(f"Stock universe: {len(self.test_stocks)} stocks")
        logger.info(f"Backtest period: {self.backtest_start.date()} → {self.backtest_end.date()}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Output: {self.output_dir}")

        # 1. Load data
        stock_data = self.load_data()
        logger.info(f"Loaded {len(stock_data)} stocks with sufficient data")

        # 2. Generate rolling Karash signals
        signals_df = self.generate_signals(stock_data)

        # Save raw signals
        signals_csv = self.output_dir / "karash_signals.csv"
        signals_df.to_csv(signals_csv, index=False)
        logger.info(f"Saved signals to: {signals_csv}")

        # 3. Prepare DataFrame for backtester
        # Baselines need Prob_Up column (dummy)
        sim_df = signals_df.copy()
        sim_df['Prob_Up'] = 0.33
        sim_df['Prob_Down'] = 0.33

        # 4. Run strategies
        logger.info("\n" + "=" * 60)
        logger.info("STRATEGY SIMULATION")
        logger.info("=" * 60)

        backtester = Backtester(initial_capital=100000.0)

        strategies = [
            # Baselines (same as ML evaluators — no model signal used)
            ("Buy_Hold_Universe", {}),
            ("Daily_Rebalanced_Universe", {}),
            ("Random_Allocation", {'n': 5}),
            # Karash strategies
            ("Karash_Threshold_Long", {'threshold': 30}),
            ("Karash_Threshold_Long", {'threshold': 50}),
            ("Karash_Long_Top_Pct", {'long_pct': 0.10}),
            ("Karash_Long_Top_Pct", {'long_pct': 0.20}),
            ("Karash_Long_Top_Pct", {'long_pct': 0.30}),
            ("Karash_Long_Short", {'long_pct': 0.20, 'short_pct': 0.20}),
            ("Karash_Long_Short", {'long_pct': 0.10, 'short_pct': 0.10}),
        ]

        # Ensure plots directory exists
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        try:
            from src.evaluation.plots import plot_portfolio_composition, plot_trade_activity
            has_plots = True
        except ImportError:
            logger.warning("Could not import plotting functions; skipping plot generation")
            has_plots = False

        results = []
        for strat_name, kwargs in strategies:
            label = f"{strat_name}"
            if kwargs:
                param_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
                label = f"{strat_name} ({param_str})"

            safe_label = (
                label.replace("(", "").replace(")", "")
                .replace(", ", "_").replace("=", "")
            )

            logger.info(f"  Running: {label}")
            try:
                metrics, history, holdings_df, trades_df = backtester.run_strategy(
                    sim_df, strat_name, **kwargs
                )
                metrics['Strategy'] = label
                results.append(metrics)

                # Save holding and trade logs
                if not holdings_df.empty:
                    holdings_df.to_csv(
                        self.output_dir / f"{safe_label}_holdings.csv", index=False
                    )
                    if has_plots:
                        plot_portfolio_composition(
                            holdings_df, label,
                            plots_dir / f"{safe_label}_composition.png"
                        )

                if not trades_df.empty:
                    trades_df.to_csv(
                        self.output_dir / f"{safe_label}_trades.csv", index=False
                    )
                    if has_plots:
                        plot_trade_activity(
                            trades_df, label,
                            plots_dir / f"{safe_label}_activity.png"
                        )

            except Exception as e:
                logger.warning(f"  Strategy {label} failed: {e}")

        # 5. Output results
        if results:
            results_df = pd.DataFrame(results)
            cols = ['Strategy'] + [c for c in results_df.columns if c != 'Strategy']
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
        logger.info(f"Output dir: {self.output_dir}")
        logger.info(f"Evaluated {len(self.test_stocks)} stocks")

        if results:
            # Best Karash strategy by Sharpe
            karash_results = [r for r in results if 'Karash' in r.get('Strategy', '')]
            if karash_results:
                best = max(karash_results, key=lambda r: r.get('Sharpe Ratio', 0))
                logger.info(
                    f"Best Karash:    {best['Strategy']} "
                    f"(Sharpe={best['Sharpe Ratio']:.2f}, "
                    f"Return={best['Total Return']:.2%})"
                )
            # Buy & Hold baseline
            bh = [r for r in results if 'Buy_Hold' in r.get('Strategy', '')]
            if bh:
                logger.info(
                    f"Buy & Hold:     Sharpe={bh[0]['Sharpe Ratio']:.2f}, "
                    f"Return={bh[0]['Total Return']:.2%}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Karash Rule-Based Algorithm Backtester")
    parser.add_argument(
        "--report-dir", type=str, default=None,
        help="Path to a model report directory whose config.json contains "
             "train_stocks_list. Constrains the stock universe to those tickers."
    )
    parser.add_argument(
        "--top-n-stocks", type=int, default=None,
        help="Limit to top N stocks from EXTENDED_TICKERS "
             "(ignored if --report-dir provides train_stocks_list)"
    )
    parser.add_argument(
        "--start", type=str, default="2023-06-01",
        help="Backtest start date (default: 2023-06-01)"
    )
    parser.add_argument(
        "--end", type=str, default="2024-12-31",
        help="Backtest end date (default: 2024-12-31)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results (default: auto-generated)"
    )
    args = parser.parse_args()

    runner = KarashBacktestRunner(
        report_dir=args.report_dir,
        top_n_stocks=args.top_n_stocks,
        backtest_start=args.start,
        backtest_end=args.end,
        output_dir=args.output_dir,
    )
    runner.run()
