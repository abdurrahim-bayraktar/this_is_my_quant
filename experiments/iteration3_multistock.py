"""
Iteration 3: Multi-Stock Validation Experiment

Tests the sentiment signal improvements on three different stocks:
- WMT (Walmart) - Consumer Staples
- DIS (Walt Disney) - Entertainment  
- MSFT (Microsoft) - Technology

Experiments:
- Baseline: Original implementation
- A: Event-based features only
- B: Curriculum learning only
- C: Attention mechanism only
- D: All combined

Usage:
    python experiments/iteration3_multistock.py --epochs 30
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
from src.nlp import SentimentExtractor, SentimentAggregator
from src.models import BaselineLSTM, AttentionLSTM, CombinedLoss
from src.training import Trainer


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Target stocks for testing
TEST_STOCKS = ["WMT", "DIS", "MSFT"]


class Iteration3Experiment:
    """Run iteration 3 experiments across multiple stocks."""
    
    def __init__(self, epochs: int = 30):
        self.epochs = epochs
        self.output_dir = REPORTS_DIR / f"iteration3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        self.preprocessor = DataPreprocessor()
        self.engineer = FeatureEngineer()
        
        # Sentiment cache to avoid re-extracting
        self.sentiment_cache = {}
        self.cache_dir = Path(__file__).parent.parent / "data" / "cache" / "sentiment"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def load_stock_data(self, ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and prepare data for a single stock with caching."""
        logger.info(f"Loading data for {ticker}...")
        
        # Check cache first
        news_cache_file = self.cache_dir / f"{ticker}_news.parquet"
        prices_cache_file = self.cache_dir / f"{ticker}_prices.parquet"
        
        if news_cache_file.exists() and prices_cache_file.exists():
            logger.info(f"Loading cached data for {ticker}")
            news_df = pd.read_parquet(news_cache_file)
            prices_df = pd.read_parquet(prices_cache_file)
            return news_df, prices_df
        
        # Load fresh data
        loader = DatasetLoader(tickers=[ticker], use_cache=True)
        news_df, prices_df = loader.load_from_local_cache()
        
        if len(news_df) == 0:
            logger.warning(f"No local news for {ticker}, trying intersection strategy...")
            news_df, prices_df = loader.load_data_with_sentiment_intersection(min_news_count=5)
        
        if len(prices_df) == 0:
            # Fallback to yfinance
            prices_df = loader._load_prices_yfinance()
        
        # Cache the data
        if len(news_df) > 0:
            news_df.to_parquet(news_cache_file, index=False)
        if len(prices_df) > 0:
            prices_df.to_parquet(prices_cache_file, index=False)
        logger.info(f"Cached data for {ticker}")
        
        return news_df, prices_df
    
    def prepare_features(
        self,
        news_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        ticker: str = None,
        use_event_features: bool = False,
        use_momentum: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features with optional iteration 3 enhancements."""
        prices_df = prices_df.copy()
        
        # Add technical indicators
        prices_df = self.engineer.add_technical_indicators(prices_df)
        prices_df = self.preprocessor.compute_returns(prices_df)
        prices_df = self.preprocessor.label_trends(prices_df)
        
        if len(news_df) > 0:
            # Check cache first
            cache_key = ticker or "unknown"
            cache_file = self.cache_dir / f"{cache_key}_sentiment.parquet"
            
            if cache_key in self.sentiment_cache:
                # Use in-memory cache
                logger.info(f"Using cached sentiment for {cache_key} (in-memory)")
                news_df = self.sentiment_cache[cache_key]
            elif cache_file.exists():
                # Use disk cache
                logger.info(f"Loading cached sentiment for {cache_key} from disk")
                news_df = pd.read_parquet(cache_file)
                self.sentiment_cache[cache_key] = news_df
            else:
                # Extract sentiment with FinBERT
                extractor = SentimentExtractor()
                text_col = "headline" if "headline" in news_df.columns else "title"
                if text_col not in news_df.columns:
                    text_col = news_df.columns[0]  # Fallback
                
                news_df = extractor.extract_dataframe(news_df, text_col)
                
                # Save to cache
                logger.info(f"Caching sentiment for {cache_key}")
                news_df.to_parquet(cache_file, index=False)
                self.sentiment_cache[cache_key] = news_df
            
            # Align to trading days
            news_df = self.preprocessor.align_news_to_trading_day(news_df, prices_df)
            
            # Aggregation with event-based features if enabled
            if use_event_features and len(news_df) > 0:
                sentiment_col = "sentiment_score" if "sentiment_score" in news_df.columns else "sentiment_positive"
                sentiment_df = self.engineer.add_event_based_sentiment_features(
                    news_df, 
                    sentiment_col=sentiment_col
                )
            else:
                aggregator = SentimentAggregator()
                sentiment_df = aggregator.aggregate_daily(news_df)
            
            # Merge with prices
            if len(sentiment_df) > 0:
                prices_df = self.engineer.merge_price_and_sentiment(prices_df, sentiment_df)
            
            # Add momentum if enabled
            if use_momentum and 'sentiment_mean' in prices_df.columns:
                prices_df = self.engineer.add_sentiment_momentum(prices_df)
        
        # Ensure all expected columns exist with defaults
        for col in ['sentiment_mean', 'sentiment_dispersion', 'news_count', 'has_news',
                    'sentiment_max', 'sentiment_min', 'sentiment_range', 'has_extreme_news',
                    'sentiment_momentum', 'sentiment_acceleration']:
            if col not in prices_df.columns:
                prices_df[col] = 0.0
        
        # Get feature columns
        feature_cols = self.engineer.get_feature_columns()
        feature_cols = [c for c in feature_cols if c in prices_df.columns]
        
        # Normalize and create sequences
        prices_df = self.preprocessor.normalize_features(prices_df, feature_cols, fit=True)
        X, y, _ = self.preprocessor.create_sequences(prices_df, feature_cols, target_col="trend")
        
        return X.astype(np.float32), y, feature_cols
    
    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_cols: List[str],
        use_attention: bool = False,
        use_curriculum: bool = False,
    ) -> Dict:
        """Train model and return metrics."""
        if len(X) < 100:
            logger.warning(f"Not enough data: {len(X)} samples")
            return {"accuracy": 0.0, "error": "insufficient_data"}
        
        # Split data
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
        
        # Use simple CrossEntropyLoss instead of CombinedLoss
        # CombinedLoss (trend + confidence) added too much optimization friction
        loss_fn = torch.nn.CrossEntropyLoss()
        trainer = Trainer(model, loss_fn)
        
        # Train
        if use_curriculum:
            # Find sentiment_mean column index
            sentiment_idx = feature_cols.index("sentiment_mean") if "sentiment_mean" in feature_cols else None
            if sentiment_idx is not None:
                trainer.curriculum_train(
                    X_train, y_train,
                    sentiment_col_idx=sentiment_idx,
                    phase1_epochs=self.epochs // 3,
                    phase2_epochs=self.epochs * 2 // 3,
                )
            else:
                logger.warning("sentiment_mean not found, using standard training")
                train_loader = trainer._create_loader(X_train, y_train, shuffle=True)
                val_loader = trainer._create_loader(X_val, y_val)
                trainer.train(train_loader, val_loader, epochs=self.epochs)
        else:
            train_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
                batch_size=training_config.batch_size,
                shuffle=True,
            )
            val_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
                batch_size=training_config.batch_size,
            )
            trainer.train(train_loader, val_loader, epochs=self.epochs)
        
        # Evaluate
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            predictions = model.predict(X_test_tensor)
        
        pred_classes = predictions["trend_class"].cpu().numpy()
        confidences = predictions["confidence"].cpu().numpy()
        
        accuracy = (pred_classes == y_test).mean()
        high_conf_mask = confidences > 0.5
        high_conf_acc = (pred_classes[high_conf_mask] == y_test[high_conf_mask]).mean() if high_conf_mask.sum() > 0 else 0.0
        
        return {
            "accuracy": float(accuracy),
            "high_conf_accuracy": float(high_conf_acc),
            "high_conf_ratio": float(high_conf_mask.mean()),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "num_features": X.shape[-1],
        }
    
    def run_experiment(self, ticker: str) -> Dict:
        """Run all experiment variants for a single stock."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Running experiments for {ticker}")
        logger.info(f"{'='*60}")
        
        news_df, prices_df = self.load_stock_data(ticker)
        results = {}
        
        # Baseline
        logger.info(f"\n--- {ticker}: Baseline ---")
        try:
            X, y, feature_cols = self.prepare_features(news_df.copy(), prices_df.copy(), ticker=ticker)
            results["baseline"] = self.train_and_evaluate(X, y, feature_cols)
        except Exception as e:
            logger.error(f"Baseline failed: {e}")
            results["baseline"] = {"accuracy": 0.0, "error": str(e)}
        
        # A: Event-based features
        logger.info(f"\n--- {ticker}: Event-Based Features ---")
        try:
            X, y, feature_cols = self.prepare_features(news_df.copy(), prices_df.copy(), ticker=ticker, use_event_features=True)
            results["event_features"] = self.train_and_evaluate(X, y, feature_cols)
        except Exception as e:
            logger.error(f"Event features failed: {e}")
            results["event_features"] = {"accuracy": 0.0, "error": str(e)}
        
        # B: Curriculum learning
        logger.info(f"\n--- {ticker}: Curriculum Learning ---")
        try:
            X, y, feature_cols = self.prepare_features(news_df.copy(), prices_df.copy(), ticker=ticker)
            results["curriculum"] = self.train_and_evaluate(X, y, feature_cols, use_curriculum=True)
        except Exception as e:
            logger.error(f"Curriculum failed: {e}")
            results["curriculum"] = {"accuracy": 0.0, "error": str(e)}
        
        # C: Attention mechanism
        logger.info(f"\n--- {ticker}: Attention Mechanism ---")
        try:
            X, y, feature_cols = self.prepare_features(news_df.copy(), prices_df.copy(), ticker=ticker)
            results["attention"] = self.train_and_evaluate(X, y, feature_cols, use_attention=True)
        except Exception as e:
            logger.error(f"Attention failed: {e}")
            results["attention"] = {"accuracy": 0.0, "error": str(e)}
        
        # D: All combined
        logger.info(f"\n--- {ticker}: All Combined ---")
        try:
            X, y, feature_cols = self.prepare_features(
                news_df.copy(), prices_df.copy(), ticker=ticker,
                use_event_features=True, use_momentum=True
            )
            results["combined"] = self.train_and_evaluate(
                X, y, feature_cols, 
                use_attention=True, use_curriculum=True
            )
        except Exception as e:
            logger.error(f"Combined failed: {e}")
            results["combined"] = {"accuracy": 0.0, "error": str(e)}
        
        return results
    
    def preload_all_data(self, tickers: List[str]):
        """Preload data for all tickers to minimize CSV scans."""
        # Check what's missing from cache
        missing_tickers = []
        for ticker in tickers:
            news_cache = self.cache_dir / f"{ticker}_news.parquet"
            prices_cache = self.cache_dir / f"{ticker}_prices.parquet"
            if not (news_cache.exists() and prices_cache.exists()):
                missing_tickers.append(ticker)
        
        if not missing_tickers:
            logger.info("All data already cached.")
            return

        logger.info(f"Preloading data for {len(missing_tickers)} tickers to avoid repeated scans: {missing_tickers}")
        
        # Load all missing tickers at once (single scan of 21GB file)
        loader = DatasetLoader(tickers=missing_tickers, use_cache=True)
        news_df, prices_df = loader.load_from_local_cache()
        
        # Warn if empty
        if len(news_df) == 0:
            logger.warning("No data found in local cache for requested tickers.")
            return

        # Split and cache per ticker
        for ticker in missing_tickers:
            t_news = news_df[news_df['ticker'] == ticker].copy()
            t_prices = prices_df[prices_df['ticker'] == ticker].copy()
            
            if len(t_news) > 0:
                t_news.to_parquet(self.cache_dir / f"{ticker}_news.parquet", index=False)
            
            if len(t_prices) > 0:
                t_prices.to_parquet(self.cache_dir / f"{ticker}_prices.parquet", index=False)
            elif len(t_prices) == 0:
                # Try yfinance fallback for prices now
                try:
                    loader.tickers = [ticker]
                    t_prices = loader._load_prices_yfinance()
                    t_prices.to_parquet(self.cache_dir / f"{ticker}_prices.parquet", index=False)
                except Exception as e:
                    logger.warning(f"Could not load prices for {ticker}: {e}")
            
            logger.info(f"Cached preloaded data for {ticker}: {len(t_news)} news, {len(t_prices)} prices")

    def run_all(self) -> pd.DataFrame:
        """Run experiments for all test stocks."""
        all_results = []
        
        # Use instance attribute if set, otherwise use global
        stocks = getattr(self, 'test_stocks', TEST_STOCKS)
        
        # Preload data for all stocks in one go
        self.preload_all_data(stocks)
        
        for ticker in stocks:
            try:
                ticker_results = self.run_experiment(ticker)
                self.results[ticker] = ticker_results
                
                for variant, metrics in ticker_results.items():
                    all_results.append({
                        "ticker": ticker,
                        "variant": variant,
                        **metrics,
                    })
            except Exception as e:
                logger.error(f"Failed for {ticker}: {e}")
                for variant in ["baseline", "event_features", "curriculum", "attention", "combined"]:
                    all_results.append({
                        "ticker": ticker,
                        "variant": variant,
                        "accuracy": 0.0,
                        "error": str(e),
                    })
        
        # Create results DataFrame
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(self.output_dir / "results.csv", index=False)
        
        with open(self.output_dir / "results.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        
        return results_df
    
    def print_summary(self, results_df: pd.DataFrame):
        """Print summary comparison."""
        print("\n" + "="*80)
        print("ITERATION 3: Multi-Stock Experiment Results")
        print("="*80)
        
        if 'accuracy' in results_df.columns:
            # Filter out errors
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
        else:
            print("Results DataFrame has no accuracy column")
        
        print(f"\nResults saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Iteration 3 Multi-Stock Experiment")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs per variant")
    parser.add_argument("--stocks", nargs="+", default=["WMT", "DIS", "MSFT"], help="Stocks to test")
    args = parser.parse_args()
    
    # Override TEST_STOCKS with command line argument
    stocks_to_test = args.stocks
    
    experiment = Iteration3Experiment(epochs=args.epochs)
    experiment.test_stocks = stocks_to_test  # Pass stocks to experiment
    results_df = experiment.run_all()
    experiment.print_summary(results_df)


if __name__ == "__main__":
    main()
