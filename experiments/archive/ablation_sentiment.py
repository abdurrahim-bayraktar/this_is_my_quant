"""
Ablation Study: Sentiment Data Source Comparison

This script compares model performance across different sentiment configurations:
1. NO_SENTIMENT - Price-only baseline (technical indicators only)
2. TWITTER_ONLY - Twitter Financial Sentiment + yfinance prices
3. NEWS_ONLY - FNSPID news sentiment + prices
4. COMBINED - Both Twitter and FNSPID news sentiment

Usage:
    python experiments/ablation_sentiment.py --scenarios all --stocks 50 --epochs 30
    python experiments/ablation_sentiment.py --scenarios no_sentiment twitter_only --stocks 20 --epochs 10
"""

import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    TOP_200_TICKERS,
    MODELS_DIR,
    REPORTS_DIR,
    training_config,
    lstm_config,
)
from src.data import DatasetLoader, DataPreprocessor, FeatureEngineer
from src.nlp import SentimentExtractor, SentimentAggregator
from src.models import BaselineLSTM, CombinedLoss
from src.training import Trainer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Scenario definitions
SCENARIOS = {
    "no_sentiment": {
        "name": "No Sentiment (Price Only)",
        "use_twitter": False,
        "use_fnspid": False,
        "description": "Baseline using only price and technical indicators",
    },
    "twitter_only": {
        "name": "Twitter Sentiment Only",
        "use_twitter": True,
        "use_fnspid": False,
        "description": "Twitter Financial Sentiment dataset with ticker extraction",
    },
    "news_only": {
        "name": "News Sentiment Only (FNSPID)",
        "use_twitter": False,
        "use_fnspid": True,
        "description": "FNSPID financial news sentiment",
    },
    "combined": {
        "name": "Combined (Twitter + News)",
        "use_twitter": True,
        "use_fnspid": True,
        "description": "Both Twitter and FNSPID news sentiment",
    },
}


class AblationExperiment:
    """
    Runs ablation study comparing different sentiment data sources.
    """
    
    def __init__(
        self,
        tickers: List[str],
        epochs: int = 30,
        output_dir: Path = None,
    ):
        self.tickers = tickers
        self.epochs = epochs
        self.output_dir = output_dir or REPORTS_DIR / f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        self.preprocessor = DataPreprocessor()
        self.engineer = FeatureEngineer()
        
    def load_price_data(self) -> pd.DataFrame:
        """Load and cache price data (shared across all scenarios)."""
        logger.info("Loading price data from Yahoo Finance...")
        loader = DatasetLoader(tickers=self.tickers, use_cache=True)
        prices_df = loader._load_prices_yfinance()
        
        # Add technical indicators
        prices_df = self.engineer.add_technical_indicators(prices_df)
        
        # Compute returns and labels
        prices_df = self.preprocessor.compute_returns(prices_df)
        prices_df = self.preprocessor.label_trends(prices_df)
        
        return prices_df
    
    def load_twitter_sentiment(self) -> pd.DataFrame:
        """Load Twitter Financial Sentiment with ticker extraction."""
        logger.info("Loading Twitter Financial Sentiment...")
        
        from datasets import load_dataset
        import re
        
        dataset = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")
        
        ticker_set = set(self.tickers)
        records = []
        
        for item in dataset:
            text = item['text']
            label = item['label']
            
            # Extract tickers ($AAPL, $TSLA, etc.)
            tickers_found = re.findall(r'\$([A-Z]{1,5})\b', text)
            
            sentiment_map = {0: -1.0, 1: 1.0, 2: 0.0}  # Bearish, Bullish, Neutral
            sentiment_value = sentiment_map.get(label, 0.0)
            
            for ticker in tickers_found:
                if ticker in ticker_set:
                    records.append({
                        'ticker': ticker,
                        'headline': text,
                        'sentiment_value': sentiment_value,
                        'sentiment_label': ['negative', 'positive', 'neutral'][label],
                    })
        
        df = pd.DataFrame(records)
        logger.info(f"Extracted {len(df)} Twitter sentiment records for {df['ticker'].nunique()} tickers")
        return df
    
    def load_fnspid_sentiment(self) -> pd.DataFrame:
        """Load FNSPID news sentiment."""
        logger.info("Loading FNSPID news sentiment (this may take a while)...")
        
        from datasets import load_dataset
        
        try:
            dataset = load_dataset("Zihan1004/FNSPID", split="train")
            df = dataset.to_pandas()
            
            # Filter to our tickers
            ticker_col = 'ticker' if 'ticker' in df.columns else 'symbol'
            if ticker_col in df.columns:
                df = df[df[ticker_col].isin(set(self.tickers))]
            
            logger.info(f"Loaded {len(df)} FNSPID news records")
            return df
        except Exception as e:
            logger.warning(f"Could not load FNSPID: {e}")
            return pd.DataFrame()
    
    def prepare_features(
        self,
        prices_df: pd.DataFrame,
        sentiment_df: pd.DataFrame = None,
        use_sentiment: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare feature matrix for training.
        
        Args:
            prices_df: Price data with technical indicators
            sentiment_df: Sentiment data (optional)
            use_sentiment: Whether to include sentiment features
            
        Returns:
            X, y, feature_columns
        """
        df = prices_df.copy()
        
        # Base features (price + technical)
        feature_cols = self.engineer.get_feature_columns()
        
        if use_sentiment and sentiment_df is not None and len(sentiment_df) > 0:
            # Aggregate sentiment by ticker (since Twitter data has no dates)
            sentiment_agg = sentiment_df.groupby('ticker').agg({
                'sentiment_value': ['mean', 'std', 'count']
            }).reset_index()
            sentiment_agg.columns = ['ticker', 'sentiment_mean', 'sentiment_std', 'sentiment_count']
            sentiment_agg['sentiment_std'] = sentiment_agg['sentiment_std'].fillna(0)
            
            # Merge with prices
            df = df.merge(sentiment_agg, on='ticker', how='left')
            df['sentiment_mean'] = df['sentiment_mean'].fillna(0)
            df['sentiment_std'] = df['sentiment_std'].fillna(0)
            df['sentiment_count'] = df['sentiment_count'].fillna(0)
        else:
            # No sentiment - add zeros
            df['sentiment_mean'] = 0.0
            df['sentiment_std'] = 0.0
            df['sentiment_count'] = 0
        
        # Filter to available columns
        feature_cols = [c for c in feature_cols if c in df.columns]
        
        # Normalize
        df = self.preprocessor.normalize_features(df, feature_cols, fit=True)
        
        # Create sequences
        X, y, _ = self.preprocessor.create_sequences(
            df,
            feature_cols=feature_cols,
            target_col="trend",
        )
        
        return X.astype(np.float32), y, feature_cols
    
    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        scenario_name: str,
    ) -> Dict:
        """
        Train model and return evaluation metrics.
        """
        logger.info(f"Training model for {scenario_name}...")
        
        # Split data (70/15/15)
        n = len(X)
        train_end = int(0.7 * n)
        val_end = int(0.85 * n)
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        # Create model
        model = BaselineLSTM(
            input_size=X.shape[-1],
            hidden_size=lstm_config.hidden_size,
            num_layers=lstm_config.num_layers,
            dropout=lstm_config.dropout,
        )
        
        loss_fn = CombinedLoss(trend_weight=1.0, confidence_weight=0.5)
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
        
        # Evaluate on test set
        model.eval()
        device = next(model.parameters()).device
        
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            predictions = model.predict(X_test_tensor)
        
        pred_classes = predictions['trend_class'].cpu().numpy()
        confidences = predictions['confidence'].cpu().numpy()
        
        # Calculate metrics
        accuracy = (pred_classes == y_test).mean()
        
        # Per-class accuracy
        class_metrics = {}
        for cls, name in enumerate(['Down', 'Neutral', 'Up']):
            mask = y_test == cls
            if mask.sum() > 0:
                class_metrics[f'{name}_accuracy'] = (pred_classes[mask] == cls).mean()
                class_metrics[f'{name}_count'] = mask.sum()
        
        # High confidence accuracy
        high_conf_mask = confidences > 0.5
        if high_conf_mask.sum() > 0:
            high_conf_acc = (pred_classes[high_conf_mask] == y_test[high_conf_mask]).mean()
        else:
            high_conf_acc = 0.0
        
        results = {
            'scenario': scenario_name,
            'accuracy': float(accuracy),
            'high_confidence_accuracy': float(high_conf_acc),
            'high_confidence_ratio': float(high_conf_mask.mean()),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'num_features': X.shape[-1],
            **{k: float(v) if isinstance(v, (np.floating, float)) else int(v) 
               for k, v in class_metrics.items()},
        }
        
        return results
    
    def run_scenario(self, scenario_key: str, prices_df: pd.DataFrame) -> Dict:
        """Run a single scenario."""
        scenario = SCENARIOS[scenario_key]
        logger.info(f"\n{'='*60}")
        logger.info(f"Running Scenario: {scenario['name']}")
        logger.info(f"Description: {scenario['description']}")
        logger.info(f"{'='*60}")
        
        # Load sentiment data based on scenario
        sentiment_df = None
        
        if scenario['use_twitter']:
            twitter_df = self.load_twitter_sentiment()
            sentiment_df = twitter_df
        
        if scenario['use_fnspid']:
            fnspid_df = self.load_fnspid_sentiment()
            if sentiment_df is not None and len(fnspid_df) > 0:
                # Combine Twitter and FNSPID
                # For FNSPID, we need to extract sentiment
                if 'sentiment_value' not in fnspid_df.columns:
                    # Run sentiment extraction on FNSPID headlines
                    logger.info("Extracting sentiment from FNSPID headlines...")
                    extractor = SentimentExtractor()
                    text_col = 'headline' if 'headline' in fnspid_df.columns else 'title'
                    if text_col in fnspid_df.columns:
                        fnspid_df = extractor.extract_dataframe(fnspid_df, text_col)
                        fnspid_df['sentiment_value'] = fnspid_df['sentiment_positive'] - fnspid_df['sentiment_negative']
                
                if 'sentiment_value' in fnspid_df.columns:
                    sentiment_df = pd.concat([sentiment_df, fnspid_df[['ticker', 'headline', 'sentiment_value']]], ignore_index=True)
            elif len(fnspid_df) > 0:
                sentiment_df = fnspid_df
        
        # Prepare features
        use_sentiment = scenario['use_twitter'] or scenario['use_fnspid']
        X, y, feature_cols = self.prepare_features(
            prices_df, 
            sentiment_df=sentiment_df,
            use_sentiment=use_sentiment,
        )
        
        logger.info(f"Prepared data: X.shape={X.shape}, features={len(feature_cols)}")
        
        # Train and evaluate
        results = self.train_and_evaluate(X, y, scenario['name'])
        results['feature_columns'] = feature_cols
        
        return results
    
    def run_all(self, scenario_keys: List[str] = None) -> pd.DataFrame:
        """Run all or selected scenarios."""
        if scenario_keys is None:
            scenario_keys = list(SCENARIOS.keys())
        
        logger.info(f"Running ablation study with {len(scenario_keys)} scenarios")
        logger.info(f"Tickers: {len(self.tickers)}, Epochs: {self.epochs}")
        
        # Load price data once (shared)
        prices_df = self.load_price_data()
        
        # Run each scenario
        all_results = []
        for key in scenario_keys:
            try:
                results = self.run_scenario(key, prices_df)
                self.results[key] = results
                all_results.append(results)
            except Exception as e:
                logger.error(f"Scenario {key} failed: {e}")
                all_results.append({
                    'scenario': SCENARIOS[key]['name'],
                    'accuracy': None,
                    'error': str(e),
                })
        
        # Create comparison DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Save results
        results_path = self.output_dir / "ablation_results.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"Results saved to {results_path}")
        
        # Save detailed JSON
        json_path = self.output_dir / "ablation_results.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        return results_df
    
    def print_comparison(self, results_df: pd.DataFrame):
        """Print a nice comparison table."""
        print("\n" + "="*80)
        print("ABLATION STUDY RESULTS: Sentiment Data Source Comparison")
        print("="*80)
        
        cols = ['scenario', 'accuracy', 'high_confidence_accuracy', 'high_confidence_ratio']
        display_df = results_df[cols].copy()
        display_df['accuracy'] = display_df['accuracy'].apply(lambda x: f"{x:.2%}" if x else "N/A")
        display_df['high_confidence_accuracy'] = display_df['high_confidence_accuracy'].apply(lambda x: f"{x:.2%}" if x else "N/A")
        display_df['high_confidence_ratio'] = display_df['high_confidence_ratio'].apply(lambda x: f"{x:.2%}" if x else "N/A")
        
        print(display_df.to_string(index=False))
        
        print("\n" + "-"*80)
        print("INTERPRETATION:")
        print("-"*80)
        
        if results_df['accuracy'].notna().any():
            best_idx = results_df['accuracy'].idxmax()
            worst_idx = results_df['accuracy'].idxmin()
            
            print(f"✓ Best performing: {results_df.loc[best_idx, 'scenario']} ({results_df.loc[best_idx, 'accuracy']:.2%})")
            print(f"✗ Worst performing: {results_df.loc[worst_idx, 'scenario']} ({results_df.loc[worst_idx, 'accuracy']:.2%})")
            
            no_sentiment_acc = results_df[results_df['scenario'].str.contains('Price Only', na=False)]['accuracy'].values
            if len(no_sentiment_acc) > 0:
                improvement = results_df.loc[best_idx, 'accuracy'] - no_sentiment_acc[0]
                print(f"↑ Improvement over baseline: {improvement:.2%}")


def main():
    parser = argparse.ArgumentParser(description="Ablation Study: Sentiment Data Sources")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["all"],
        choices=["all", "no_sentiment", "twitter_only", "news_only", "combined"],
        help="Scenarios to run",
    )
    parser.add_argument("--stocks", type=int, default=50, help="Number of stocks")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs per scenario")
    
    args = parser.parse_args()
    
    if "all" in args.scenarios:
        scenario_keys = list(SCENARIOS.keys())
    else:
        scenario_keys = args.scenarios
    
    experiment = AblationExperiment(
        tickers=TOP_200_TICKERS[:args.stocks],
        epochs=args.epochs,
    )
    
    results_df = experiment.run_all(scenario_keys)
    experiment.print_comparison(results_df)
    
    print(f"\nDetailed results saved to: {experiment.output_dir}")


if __name__ == "__main__":
    main()
