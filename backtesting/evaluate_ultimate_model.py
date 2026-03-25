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
from src.features.indicators_v7 import ComprehensiveIndicatorsV7
from experiments.ultimate_model import create_model, SHAP_TOP20_FEATURES
from sklearn.preprocessing import StandardScaler
from backtesting.framework import Backtester

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class UltimateBacktestRunner:
    def __init__(self, report_dir: str, backtest_start: str = None, backtest_end: str = "2024-12-31"):
        self.report_dir = Path(report_dir)
        with open(self.report_dir / "config.json", "r") as f:
            self.config = json.load(f)
            
        self.train_stock_count = self.config.get("train_stock_count", 400)
        self.test_stocks = self.config.get("test_stocks", [])
        
        # Load EXTENDED_TICKERS from ranked_tickers to reconstruct train_stocks
        try:
            from experiments.ranked_tickers import EXTENDED_TICKERS
            all_train = EXTENDED_TICKERS[:self.train_stock_count]
            self.train_stocks = [t for t in all_train if t not in self.test_stocks]
        except ImportError:
            logger.warning("Could not import EXTENDED_TICKERS, using test_stocks only for testing.")
            self.train_stocks = []
            
        self.frequency = self.config.get("frequency", "daily")
        self.sequence_length = self.config.get("sequence_length", 20 if self.frequency == "daily" else 12)
        self.model_type = self.config.get("model_type", "simple")
        self.threshold_low = self.config.get("threshold_low", -0.005)
        self.threshold_high = self.config.get("threshold_high", 0.005)
        
        # Dates (same as UltimateModelExperiment defaults)
        self.start_date = "2014-01-01"
        self.train_end = pd.Timestamp("2022-01-01")
        self.val_end = pd.Timestamp("2023-06-01")  # Default model test set bound
        
        self.backtest_start = pd.Timestamp(backtest_start) if backtest_start else self.val_end
        self.backtest_end = pd.Timestamp(backtest_end)
        
        self.end_date = self.backtest_end.strftime('%Y-%m-%d')
        
        self.feature_cols = self.config.get("features", SHAP_TOP20_FEATURES)
        self.indicator_computer = ComprehensiveIndicatorsV7()
        self.cache = PriceCache()
        self.scaler = StandardScaler()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_data(self):
        all_tickers = list(set(self.train_stocks + self.test_stocks))
        weekly = self.frequency == "weekly"
        return load_stocks(
            all_tickers, self.start_date, self.end_date,
            cache=self.cache, min_rows=100 if not weekly else 52, weekly=weekly,
        )

    def extract_sequences_with_dates(self, df: pd.DataFrame) -> tuple:
        """
        Extends create_sequences logic but explicitly maps sequences to dates
        and 'return_next' for backtesting.
        """
        df = self.indicator_computer.compute_shap_top20(df)
        available = [c for c in self.feature_cols if c in df.columns]
        if len(available) < 5:
            return pd.DataFrame(), pd.DataFrame()
            
        df['return_next'] = df['Close'].pct_change().shift(-1)
        df['trend'] = pd.cut(
            df['return_next'],
            bins=[-np.inf, self.threshold_low, self.threshold_high, np.inf],
            labels=[0, 1, 2]
        ).astype(float)
        
        # Drop rows where we can't compute features or label
        df = df.dropna(subset=['trend'] + available[:5])
        if len(df) < self.sequence_length + 10:
            return pd.DataFrame(), pd.DataFrame()
            
        dates = df.index
        feature_data = df[available].values
        feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # We need to collect:
        # 1. Sequences for training the scaler (train_end)
        # 2. Sequences for test set predicting (>= val_end) WITH Dates and Returns
        
        train_seqs = []
        test_data = [] # list of dicts {date, sequence, return_next}
        
        for i in range(len(feature_data) - self.sequence_length):
            trade_date = dates[i + self.sequence_length - 1] # The day we make the prediction (Close)
            target_date = dates[i + self.sequence_length]    # The day the return happens
            
            seq = feature_data[i:i + self.sequence_length]
            ret_next = df['return_next'].iloc[i + self.sequence_length - 1]
            label = df['trend'].iloc[i + self.sequence_length - 1]
            
            if target_date < self.train_end:
                train_seqs.append(seq)
            elif target_date >= self.backtest_start and target_date <= self.backtest_end:
                test_data.append({
                    'Date': trade_date, # Date of the prediction features
                    'Sequence': seq,
                    'Return_Next': ret_next,
                    'Label': label
                })
                
        # Convert train seqs to numpy for scaler fitting
        train_arr = np.array(train_seqs) if train_seqs else np.empty((0, self.sequence_length, len(available)))
        
        test_df = pd.DataFrame(test_data)
        return train_arr, test_df

    def reconstruct_scaler(self, stock_data):
        """Fit scaler on train_stocks up to train_end."""
        logger.info("Reconstructing StandardScaler from training data...")
        all_train_X = []
        n_features = None
        for ticker in tqdm(self.train_stocks, desc="Fitting Scaler"):
            if ticker not in stock_data: continue
            
            train_arr, _ = self.extract_sequences_with_dates(stock_data[ticker])
            if len(train_arr) > 0:
                if n_features is None: n_features = train_arr.shape[-1]
                if train_arr.shape[-1] == n_features:
                    all_train_X.append(train_arr)
                    
        if all_train_X:
            X_train = np.concatenate(all_train_X, axis=0)
            n_train, seq_len, feat = X_train.shape
            X_train_flat = X_train.reshape(-1, feat)
            self.scaler.fit(X_train_flat)
            logger.info("Scaler fitted successfully.")
            return feat
        else:
            logger.error("No training data found to fit scaler!")
            return len(self.feature_cols)

    def prepare_test_predictions(self, stock_data, n_features):
        """Generate test set predictions for test_stocks."""
        self.model.eval()
        all_predictions = []
        
        for ticker in tqdm(self.test_stocks, desc="Predicting Test Set"):
            if ticker not in stock_data: continue
            
            _, test_df = self.extract_sequences_with_dates(stock_data[ticker])
            if test_df.empty: continue
            
            seqs = np.stack(test_df['Sequence'].values)
            n_samples, seq_len, feat = seqs.shape
            
            # Scale
            seqs_flat = seqs.reshape(-1, feat)
            seqs_scaled = self.scaler.transform(seqs_flat)
            seqs_scaled = np.nan_to_num(seqs_scaled, nan=0.0).reshape(n_samples, seq_len, feat)
            
            # Predict in batches
            batch_size = 512
            probs_up, probs_down = [], []
            
            with torch.no_grad():
                for i in range(0, n_samples, batch_size):
                    batch_x = torch.FloatTensor(seqs_scaled[i:i+batch_size]).to(self.device)
                    logits, _, _ = self.model(batch_x)
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    
                    # Classes: 0: Down, 1: Neutral, 2: Up
                    probs_down.extend(probs[:, 0])
                    probs_up.extend(probs[:, 2])
            
            test_df['Prob_Up'] = probs_up
            test_df['Prob_Down'] = probs_down
            test_df['Ticker'] = ticker
            
            all_predictions.append(test_df[['Date', 'Ticker', 'Prob_Up', 'Prob_Down', 'Return_Next', 'Label']])
            
        if not all_predictions:
            raise ValueError("No test predictions generated.")
            
        return pd.concat(all_predictions, ignore_index=True)

    def run(self):
        logger.info(f"Loading experiment from: {self.report_dir}")
        stock_data = self.load_data()
        
        n_features = self.reconstruct_scaler(stock_data)
        
        logger.info("Loading model weights...")
        self.model = create_model(self.model_type, n_features, dropout=0.0)
        state_dict = torch.load(self.report_dir / "model.pt", map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        
        preds_df = self.prepare_test_predictions(stock_data, n_features)
        
        logger.info(f"Generated {len(preds_df)} predictions. Starting simulation...")
        
        backtester = Backtester(initial_capital=100000.0)
        
        strategies = [
            ("Buy_Hold_Universe", {}),
            ("Daily_Rebalanced_Universe", {}),
            ("Random_Allocation", {'n': 5}),
            ("Long_Top_N", {'n': 3}),
            ("Long_Top_N", {'n': 5}),
            ("Long_Short_Neutral", {'n': 3}),
            ("Threshold_Long", {'threshold': 0.60}),
            ("Threshold_Long", {'threshold': 0.70}),
        ]
        
        results = []
        for strat_name, kwargs in strategies:
            logger.info(f"Running strategy: {strat_name} {kwargs}")
            metrics, _ = backtester.run_strategy(preds_df, strat_name, **kwargs)
            metrics['Strategy'] = f"{strat_name}_{list(kwargs.values())}" if kwargs else strat_name
            results.append(metrics)
            
        results_df = pd.DataFrame(results)
        cols = ['Strategy'] + [c for c in results_df.columns if c != 'Strategy']
        results_df = results_df[cols]
        
        logger.info("\n" + "=" * 80)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 80)
        print(results_df.to_string(index=False))
        
        out_csv = self.report_dir / "backtest_results.csv"
        results_df.to_csv(out_csv, index=False)
        logger.info(f"Saved backtest metrics to: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=str, required=True, help="Path to ultimate_model report directory")
    parser.add_argument("--start", type=str, default=None, help="Backtest start date (e.g. 2023-06-01)")
    parser.add_argument("--end", type=str, default="2024-12-31", help="Backtest end date (e.g. 2024-12-31)")
    args = parser.parse_args()
    
    runner = UltimateBacktestRunner(args.report_dir, backtest_start=args.start, backtest_end=args.end)
    runner.run()
