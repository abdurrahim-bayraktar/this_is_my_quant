"""
Comprehensive Experiment: All Speculations.md Fixes

Implements all hypotheses from speculations.md:
1. Dead Zone Filtering - Only train on periods with valid sentiment data
2. Dual-Channel Sentiment - Separate positive/negative features
3. Sticky Sentiment - Hold sentiment when no significant news
4. Signal-to-Noise Filtering - Discard weak sentiment news

Usage:
    python experiments/speculation_experiment.py --epochs 40
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import json
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import REPORTS_DIR, training_config, lstm_config
from src.data import DatasetLoader, DataPreprocessor, FeatureEngineer
from src.nlp import SentimentAggregator
from src.models import BaselineLSTM, AttentionLSTM
from src.training import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Target stocks  
TEST_STOCKS = ["WMT", "DIS", "MSFT"]

# Dead zone cutoffs per stock (first date with consistent sentiment)
DEAD_ZONE_DATES = {
    "WMT": "2018-07-01",  # Q3 2018
    "DIS": "2019-01-01",  # 2019
    "MSFT": "2022-01-01", # 2022
}


class SpeculationExperiment:
    """Run experiment with all speculations.md fixes."""
    
    def __init__(self, epochs: int = 40):
        self.epochs = epochs
        self.output_dir = REPORTS_DIR / f"speculation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessor = DataPreprocessor()
        self.engineer = FeatureEngineer()
        self.cache_dir = Path(__file__).parent.parent / "data" / "cache" / "sentiment"
        
    def load_stock_data(self, ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load cached data for a stock."""
        # Sentiment cache has the extracted sentiment - use this instead of raw news
        sentiment_cache = self.cache_dir / f"{ticker}_sentiment.parquet"
        prices_cache = self.cache_dir / f"{ticker}_prices.parquet"
        
        if sentiment_cache.exists() and prices_cache.exists():
            news_df = pd.read_parquet(sentiment_cache)
            prices_df = pd.read_parquet(prices_cache)
            logger.info(f"Loaded {ticker}: {len(news_df)} sentiment records, {len(prices_df)} price records")
            return news_df, prices_df
        
        # Fallback to news cache if sentiment not available
        news_cache = self.cache_dir / f"{ticker}_news.parquet"
        if news_cache.exists() and prices_cache.exists():
            news_df = pd.read_parquet(news_cache)
            prices_df = pd.read_parquet(prices_cache)
            logger.warning(f"Using raw news cache for {ticker} - sentiment may not be available")
            return news_df, prices_df
        
        raise FileNotFoundError(f"Cache not found for {ticker}")
    
    def apply_dead_zone_filter(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Speculation #1: Dead Zone Filtering
        
        Only keep data after the dead zone cutoff date for each stock.
        This prevents training on periods where sentiment was effectively zero.
        """
        cutoff = DEAD_ZONE_DATES.get(ticker, "2016-01-01")
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        original_len = len(df)
        df = df[df["date"] >= cutoff]
        logger.info(f"Dead zone filter: {ticker} {original_len} -> {len(df)} rows (cutoff: {cutoff})")
        return df
    
    def apply_signal_noise_filter(self, news_df: pd.DataFrame, threshold: float = 0.2) -> pd.DataFrame:
        """
        Speculation #4: Signal-to-Noise Filtering
        
        Discard news items with |sentiment| < threshold before aggregation.
        This removes noise like "WMT opens at $X" which has neutral sentiment.
        """
        df = news_df.copy()
        
        # Find sentiment column
        sent_col = None
        for col in ['sentiment_score', 'sentiment_positive', 'sentiment_value']:
            if col in df.columns:
                sent_col = col
                break
        
        if sent_col is None:
            logger.warning("No sentiment column found for noise filtering")
            return df
        
        original_len = len(df)
        df = df[df[sent_col].abs() >= threshold]
        logger.info(f"Signal-noise filter: {original_len} -> {len(df)} news items (threshold: {threshold})")
        return df
    
    def create_dual_channel_sentiment(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Speculation #2: Dual-Channel Sentiment
        
        Instead of single sentiment_mean, create:
        - sentiment_positive: mean of positive news only
        - sentiment_negative: mean of negative news only (as positive value)
        
        This allows the model to learn different sensitivities to fear vs greed.
        """
        df = news_df.copy()
        
        # Find sentiment column
        sent_col = None
        for col in ['sentiment_score', 'sentiment_positive', 'sentiment_value']:
            if col in df.columns:
                sent_col = col
                break
        
        if sent_col is None:
            df['sentiment_pos_value'] = 0.0
            df['sentiment_neg_value'] = 0.0
            return df
        
        # Split into positive and negative
        df['sentiment_pos_value'] = df[sent_col].apply(lambda x: x if x > 0 else 0)
        df['sentiment_neg_value'] = df[sent_col].apply(lambda x: -x if x < 0 else 0)
        
        return df
    
    def aggregate_with_sticky_logic(self, news_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Speculation #3: Sticky Sentiment
        
        Use SentimentAggregator with "sticky" strategy - sentiment holds
        when there's no significant new information.
        """
        # Align news to trading days first
        news_df = self.preprocessor.align_news_to_trading_day(news_df, prices_df)
        
        if len(news_df) == 0:
            return pd.DataFrame()
        
        # Use sticky aggregation
        aggregator = SentimentAggregator(strategy="sticky")
        sentiment_df = aggregator.aggregate_daily(news_df)
        
        return sentiment_df
    
    def prepare_features(
        self,
        news_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        ticker: str,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features with all speculation fixes."""
        
        # 1. Dead Zone Filter on prices
        prices_df = self.apply_dead_zone_filter(prices_df, ticker)
        
        # Add technical indicators
        prices_df = self.engineer.add_technical_indicators(prices_df)
        prices_df = self.preprocessor.compute_returns(prices_df)
        prices_df = self.preprocessor.label_trends(prices_df)
        
        if len(news_df) == 0:
            logger.warning(f"No news data for {ticker}")
            return None, None, []
        
        # 2. Signal-to-Noise Filter on news (use lower threshold since sentiment_value is small scale)
        news_df = self.apply_signal_noise_filter(news_df, threshold=0.01)
        
        if len(news_df) == 0:
            logger.warning(f"No news left after noise filter for {ticker}")
            return None, None, []
        
        # 3. Create dual-channel sentiment features
        news_df = self.create_dual_channel_sentiment(news_df)
        
        # 4. Aggregate with sticky logic
        sentiment_df = self.aggregate_with_sticky_logic(news_df, prices_df)
        
        if len(sentiment_df) == 0:
            logger.warning(f"No sentiment aggregation for {ticker}")
            return None, None, []
        
        # Merge with prices
        prices_df = self.engineer.merge_price_and_sentiment(prices_df, sentiment_df)
        
        # Add momentum features
        if 'sentiment_mean' in prices_df.columns:
            prices_df = self.engineer.add_sentiment_momentum(prices_df)
        
        # Ensure default columns exist
        for col in ['sentiment_mean', 'sentiment_dispersion', 'news_count', 'has_news',
                    'sentiment_pos', 'sentiment_neg', 'sentiment_momentum', 'sentiment_acceleration']:
            if col not in prices_df.columns:
                prices_df[col] = 0.0
        
        # Get feature columns
        feature_cols = self.engineer.get_feature_columns()
        feature_cols = [c for c in feature_cols if c in prices_df.columns]
        
        # Normalize and create sequences
        prices_df = self.preprocessor.normalize_features(prices_df, feature_cols, fit=True)
        X, y, _ = self.preprocessor.create_sequences(prices_df, feature_cols, target_col="trend")
        
        if len(X) < 100:
            logger.warning(f"Not enough data for {ticker}: {len(X)} samples")
            return None, None, []
        
        return X.astype(np.float32), y, feature_cols
    
    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_cols: List[str],
        use_attention: bool = False,
    ) -> Dict:
        """Train model and return metrics."""
        
        # Split data (time-based)
        n = len(X)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        # Create model
        if use_attention:
            model = AttentionLSTM(
                input_size=X.shape[-1],
                hidden_size=lstm_config.hidden_size,
                num_layers=lstm_config.num_layers,
                dropout=lstm_config.dropout,
            )
        else:
            model = BaselineLSTM(
                input_size=X.shape[-1],
                hidden_size=lstm_config.hidden_size,
                num_layers=lstm_config.num_layers,
                dropout=lstm_config.dropout,
            )
        
        # Use simple CrossEntropyLoss
        loss_fn = torch.nn.CrossEntropyLoss()
        trainer = Trainer(model, loss_fn)
        
        # Create data loaders
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
            batch_size=training_config.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
            batch_size=training_config.batch_size,
        )
        
        # Train
        trainer.train(train_loader, val_loader, epochs=self.epochs)
        
        # Evaluate
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            predictions = model.predict(X_test_tensor)
        
        pred_classes = predictions["trend_class"].cpu().numpy()
        accuracy = (pred_classes == y_test).mean()
        
        return {
            "accuracy": float(accuracy),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "num_features": X.shape[-1],
        }
    
    def run_experiment(self, ticker: str) -> Dict:
        """Run experiment for a single stock."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Running speculation experiment for {ticker}")
        logger.info(f"{'='*60}")
        
        try:
            news_df, prices_df = self.load_stock_data(ticker)
        except FileNotFoundError as e:
            logger.error(f"Could not load data for {ticker}: {e}")
            return {"accuracy": 0.0, "error": str(e)}
        
        results = {}
        
        # Run with BaselineLSTM
        logger.info(f"\n--- {ticker}: Speculation + BaselineLSTM ---")
        try:
            X, y, feature_cols = self.prepare_features(news_df.copy(), prices_df.copy(), ticker)
            if X is not None:
                results["baseline"] = self.train_and_evaluate(X, y, feature_cols, use_attention=False)
            else:
                results["baseline"] = {"accuracy": 0.0, "error": "insufficient_data"}
        except Exception as e:
            logger.error(f"Baseline failed: {e}")
            results["baseline"] = {"accuracy": 0.0, "error": str(e)}
        
        # Run with AttentionLSTM
        logger.info(f"\n--- {ticker}: Speculation + AttentionLSTM ---")
        try:
            X, y, feature_cols = self.prepare_features(news_df.copy(), prices_df.copy(), ticker)
            if X is not None:
                results["attention"] = self.train_and_evaluate(X, y, feature_cols, use_attention=True)
            else:
                results["attention"] = {"accuracy": 0.0, "error": "insufficient_data"}
        except Exception as e:
            logger.error(f"Attention failed: {e}")
            results["attention"] = {"accuracy": 0.0, "error": str(e)}
        
        return results
    
    def run_all(self) -> pd.DataFrame:
        """Run experiments for all stocks."""
        all_results = []
        ticker_results = {}
        
        for ticker in TEST_STOCKS:
            results = self.run_experiment(ticker)
            ticker_results[ticker] = results
            
            for variant, metrics in results.items():
                all_results.append({
                    "ticker": ticker,
                    "variant": variant,
                    **metrics,
                })
        
        # Save results
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(self.output_dir / "results.csv", index=False)
        
        with open(self.output_dir / "results.json", "w") as f:
            json.dump(ticker_results, f, indent=2, default=str)
        
        return results_df
    
    def print_summary(self, results_df: pd.DataFrame):
        """Print summary of results."""
        print("\n" + "="*80)
        print("SPECULATION EXPERIMENT RESULTS")
        print("Fixes applied: Dead Zone Filter, Signal-Noise Filter, Sticky Sentiment, Dual-Channel")
        print("="*80)
        
        valid_df = results_df[results_df['accuracy'] > 0]
        
        if len(valid_df) > 0:
            pivot = valid_df.pivot(index="ticker", columns="variant", values="accuracy")
            print("\nAccuracy by Stock and Variant:")
            print(pivot.round(4).to_string())
            
            print("\n" + "-"*80)
            print("Average Accuracy by Variant:")
            avg = valid_df.groupby("variant")["accuracy"].mean().sort_values(ascending=False)
            for variant, acc in avg.items():
                print(f"  {variant}: {acc:.2%}")
            
            print("\n" + "-"*80)
            best = valid_df.loc[valid_df["accuracy"].idxmax()]
            print(f"Best Result: {best['ticker']} + {best['variant']} = {best['accuracy']:.2%}")
        else:
            print("No valid results to display")
        
        print(f"\nResults saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Speculation Experiment")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs")
    args = parser.parse_args()
    
    experiment = SpeculationExperiment(epochs=args.epochs)
    results_df = experiment.run_all()
    experiment.print_summary(results_df)


if __name__ == "__main__":
    main()
