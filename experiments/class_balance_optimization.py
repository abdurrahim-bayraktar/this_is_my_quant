"""
Class Balance and Threshold Optimization Experiment

This script finds the optimal threshold for discretizing returns into three classes 
(Down, Neutral, Up) such that the resulting class distribution is ~33.3% per class.

Crucially, it optimizes for **Stock-to-Stock Coherence** (Per-Stock MSE). 
A global threshold might produce a 33/33/33 split overall by forcing volatile stocks 
to be 50/0/50 and stable stocks to be 0/100/0. Per-Stock MSE prevents this by 
grading the threshold on how well it balances classes for *each individual stock*.

Usage:
    python experiments/class_balance_optimization.py --stocks 400
"""

import sys
from pathlib import Path
import logging
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import yfinance as yf
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.pooled_price_baseline_v10 import PriceCache, EXTENDED_TICKERS
from config import REPORTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['High']
    low = df['Low']
    close_prev = df['Close'].shift(1)
    tr = pd.concat([high - low, (high - close_prev).abs(), (low - close_prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def load_data(stocks_count: int, start_date: str, end_date: str, use_cache: bool = True):
    stocks = EXTENDED_TICKERS[:stocks_count]
    price_cache = PriceCache()
    stock_data = {}
    if use_cache:
        cached = price_cache.load(start_date, end_date, stocks)
        if cached:
            for ticker, df in cached.items():
                if ticker in stocks: stock_data[ticker] = df
            logger.info(f"Loaded {len(stock_data)} stocks from cache")
    missing = [t for t in stocks if t not in stock_data]
    if missing:
        logger.info(f"Downloading {len(missing)} stocks")
        batch_size = 100
        for i in range(0, len(missing), batch_size):
            batch = missing[i:i + batch_size]
            try:
                data = yf.download(batch, start=start_date, end=end_date, group_by="ticker", threads=True, progress=False)
                for ticker in batch:
                    try:
                        df = data[ticker].copy() if len(batch) > 1 else data.copy()
                        df = df.dropna()
                        if len(df) < 100: continue
                        df.columns = [c.capitalize() if isinstance(c, str) else c for c in df.columns]
                        stock_data[ticker] = df
                    except: pass
            except: pass
        if use_cache and stock_data: price_cache.save(stock_data, start_date, end_date)
    return stock_data

def process_single_stock(ticker: str, df: pd.DataFrame):
    df = df.copy()
    try:
        df['return_next'] = df['Close'].pct_change().shift(-1)
        df['returns'] = df['Close'].pct_change()
        df['volatility_20d'] = df['returns'].rolling(20).std()
        df['atr_14'] = calculate_atr(df, 14)
        df['atr_pct'] = df['atr_14'] / df['Close']
        df = df.dropna(subset=['return_next', 'volatility_20d', 'atr_pct'])
        if len(df) < 100: return None
        return {
            'ticker': ticker,
            'returns': df['return_next'].values,
            'volatility_20d': df['volatility_20d'].values,
            'atr_pct': df['atr_pct'].values
        }
    except: return None

def evaluate_thresholds_per_stock(results, fixed_thresholds, multipliers):
    all_metrics = []
    ideal_dist = np.array([1/3, 1/3, 1/3])
    
    # Pre-calculate totals for global distribution info
    total_samples = sum(len(r['returns']) for r in results)
    
    # 1. FIXED THRESHOLDS
    for thresh in fixed_thresholds:
        mses = []
        global_dist = np.zeros(3)
        for r in results:
            ret = r['returns']
            down, up = np.sum(ret < -thresh), np.sum(ret > thresh)
            neutral = len(ret) - down - up
            dist = np.array([down, neutral, up]) / len(ret)
            mses.append(np.mean((dist - ideal_dist)**2))
            global_dist += np.array([down, neutral, up])
        
        global_dist /= total_samples
        all_metrics.append({
            'Strategy': 'Fixed', 'Parameter': thresh,
            'Per_Stock_MSE': np.mean(mses), 'Global_MSE': np.mean((global_dist - ideal_dist)**2),
            'Down_%': global_dist[0]*100, 'Neutral_%': global_dist[1]*100, 'Up_%': global_dist[2]*100
        })

    # 2. DYNAMIC ATR
    for mult in multipliers:
        mses = []
        global_dist = np.zeros(3)
        for r in results:
            ret, atr = r['returns'], r['atr_pct']
            thresh = atr * mult
            down, up = np.sum(ret < -thresh), np.sum(ret > thresh)
            neutral = len(ret) - down - up
            dist = np.array([down, neutral, up]) / len(ret)
            mses.append(np.mean((dist - ideal_dist)**2))
            global_dist += np.array([down, neutral, up])
            
        global_dist /= total_samples
        all_metrics.append({
            'Strategy': 'Dynamic_ATR', 'Parameter': mult,
            'Per_Stock_MSE': np.mean(mses), 'Global_MSE': np.mean((global_dist - ideal_dist)**2),
            'Down_%': global_dist[0]*100, 'Neutral_%': global_dist[1]*100, 'Up_%': global_dist[2]*100
        })

    # 3. DYNAMIC VOLATILITY
    for mult in multipliers:
        mses = []
        global_dist = np.zeros(3)
        for r in results:
            ret, vol = r['returns'], r['volatility_20d']
            thresh = vol * mult
            down, up = np.sum(ret < -thresh), np.sum(ret > thresh)
            neutral = len(ret) - down - up
            dist = np.array([down, neutral, up]) / len(ret)
            mses.append(np.mean((dist - ideal_dist)**2))
            global_dist += np.array([down, neutral, up])
            
        global_dist /= total_samples
        all_metrics.append({
            'Strategy': 'Dynamic_Vol20', 'Parameter': mult,
            'Per_Stock_MSE': np.mean(mses), 'Global_MSE': np.mean((global_dist - ideal_dist)**2),
            'Down_%': global_dist[0]*100, 'Neutral_%': global_dist[1]*100, 'Up_%': global_dist[2]*100
        })

    return pd.DataFrame(all_metrics)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", type=int, default=400)
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info(f"PER-STOCK CLASS COHERENCE OPTIMIZATION - {args.stocks} STOCKS")
    logger.info("=" * 60)
    
    stock_data = load_data(args.stocks, "2014-01-01", "2024-12-31")
    
    logger.info("Processing stocks...")
    results = Parallel(n_jobs=-1)(
        delayed(process_single_stock)(ticker, df) for ticker, df in tqdm(stock_data.items())
    )
    results = [r for r in results if r is not None]
    
    logger.info(f"Evaluated {len(results)} valid stocks.")
    
    fixed_thresholds = np.arange(0.001, 0.021, 0.001)
    multipliers = np.arange(0.1, 1.5, 0.05)
    
    logger.info("Evaluating Per-Stock Thresholds...")
    df_results = evaluate_thresholds_per_stock(results, fixed_thresholds, multipliers)
    
    # Sort by PER STOCK MSE now
    df_results = df_results.sort_values('Per_Stock_MSE')
    
    logger.info("\nTOP 5 FIXED THRESHOLDS (Sorted by Per-Stock MSE):")
    print(df_results[df_results['Strategy'] == 'Fixed'].head(5).to_string(index=False))
    
    logger.info("\nTOP 5 DYNAMIC ATR THRESHOLDS:")
    print(df_results[df_results['Strategy'] == 'Dynamic_ATR'].head(5).to_string(index=False))
    
    logger.info("\nTOP 5 DYNAMIC VOLATILITY (20d) THRESHOLDS:")
    print(df_results[df_results['Strategy'] == 'Dynamic_Vol20'].head(5).to_string(index=False))
    
    best = df_results.iloc[0]
    logger.info(f"\nOVERALL BEST STRATEGY: {best['Strategy']} with parameter {best['Parameter']:.4f}")
    logger.info(f"Per-Stock MSE: {best['Per_Stock_MSE']:.6f} | Global MSE: {best['Global_MSE']:.6f}")
    
    output_dir = REPORTS_DIR / "class_balance_optimization"
    output_dir.mkdir(exist_ok=True, parents=True)
    out_file = output_dir / "stock_coherence_optimization_results.csv"
    df_results.to_csv(out_file, index=False)
    logger.info(f"\nFull results saved to {out_file}")

if __name__ == "__main__":
    main()
