"""
Sentiment Feature Selection via RandomForest Permutation Importance.

Standalone analysis script — does NOT modify any existing experiment.
Performs feature selection for BOTH regression and classification tasks:

  - **Regression**: vol-adjusted forward return (for regression_sentiment.py)
  - **Classification**: 3-class trend labels (for ultimate_model_v2.py)

Ticker universe:
  Uses the same density-ranked, filtered tickers from
  scripts/visualize_sentiment_density.py — i.e. excludes the noisy/low-quality
  tickers and ranks the rest by sentiment coverage density (2012–2019).

Features ranked:
  Original 7 sentiment features + sent_ema_20d (20-day EMA of sent_mean).
  Ranking uses **permutation importance** (not Gini/MDI) because sentiment
  features are continuous with varying scales — Gini is biased toward
  high-cardinality / noisy features.

Outputs (per task):
  - Ranked table of all sentiment features by permutation importance
  - The top-2 selected features
  - Comparison bar chart (permutation vs Gini)
  - summary.json with full config + rankings

Usage:
    python experiments/sentiment_feature_selection.py
    python experiments/sentiment_feature_selection.py --top-k 3
    python experiments/sentiment_feature_selection.py --n-repeats 20
    python experiments/sentiment_feature_selection.py --max-tickers 100
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Tuple

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.data.cache import PriceCache
from src.data.utils import load_stocks
from src.features.indicators_v7 import ComprehensiveIndicatorsV7

try:
    from config import REPORTS_DIR, CACHE_DIR
except ImportError:
    REPORTS_DIR = Path("reports")
    CACHE_DIR = Path("data/cache")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# EXCLUDED TICKERS (from scripts/visualize_sentiment_density.py)
# ============================================================================

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
# SENTIMENT FEATURE DEFINITIONS
# ============================================================================

# Original 7 features from regression_sentiment.py
BASE_SENTIMENT_FEATURES = [
    "sent_mean",           # Daily avg sentiment (positive − negative)
    "sent_std",            # Intraday sentiment dispersion
    "sent_count_log",      # log1p(daily_news_count)
    "sent_momentum_3d",    # 3-day diff of sent_mean
    "sent_ema_5d",         # 5-day EMA of sent_mean
    "sent_surprise",       # sent_mean − 20-day rolling mean
    "sent_strength",       # sent_mean × sent_count_log
]

# Extended: add 20-day EMA
EXTENDED_SENTIMENT_FEATURES = BASE_SENTIMENT_FEATURES + [
    "sent_ema_20d",        # 20-day EMA of sent_mean — longer-horizon sentiment regime
]

# Classification thresholds (same as ultimate_model_v2.py daily defaults)
CLASSIFICATION_THRESHOLD_LOW = -0.005
CLASSIFICATION_THRESHOLD_HIGH = 0.005


# ============================================================================
# SENTIMENT DATA LOADER (extended with 20-day EMA + density filtering)
# ============================================================================

def load_sentiment_data_extended(
    sentiment_dir: Path = None,
    exclude_tickers: set = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load sentiment parquets, aggregate to daily features, and add sent_ema_20d.

    Parameters
    ----------
    sentiment_dir : directory containing *_sentiment.parquet files
    exclude_tickers : set of ticker symbols to skip

    Returns
    -------
    {ticker: DataFrame} with EXTENDED_SENTIMENT_FEATURES columns
    """
    if sentiment_dir is None:
        sentiment_dir = CACHE_DIR / "sentiment"
    if exclude_tickers is None:
        exclude_tickers = EXCLUDE_TICKERS

    if not sentiment_dir.exists():
        logger.warning(f"Sentiment directory not found: {sentiment_dir}")
        return {}

    sentiment_files = list(sentiment_dir.glob("*_sentiment.parquet"))
    if not sentiment_files:
        logger.warning("No sentiment parquets found")
        return {}

    logger.info(f"Loading sentiment from {len(sentiment_files)} ticker files "
                f"(excluding {len(exclude_tickers)} noisy tickers)...")
    result = {}

    for fpath in sentiment_files:
        ticker = fpath.stem.replace("_sentiment", "")

        # Apply exclusion filter
        if ticker in exclude_tickers:
            continue

        try:
            raw = pd.read_parquet(fpath)

            # Normalise date column
            raw['date'] = pd.to_datetime(raw['date'], utc=True).dt.tz_localize(None)
            raw['trading_date'] = raw['date'].dt.normalize()

            # === Aggregate to daily level ===
            daily = raw.groupby('trading_date').agg(
                sent_mean=('sentiment_value', 'mean'),
                sent_std=('sentiment_value', 'std'),
                sent_count=('sentiment_value', 'count'),
            ).sort_index()

            daily['sent_std'] = daily['sent_std'].fillna(0)

            # === Derived features ===
            daily['sent_count_log'] = np.log1p(daily['sent_count'])
            daily['sent_momentum_3d'] = daily['sent_mean'].diff(3)
            daily['sent_ema_5d'] = daily['sent_mean'].ewm(span=5, min_periods=1).mean()

            rolling_mean_20 = daily['sent_mean'].rolling(20, min_periods=5).mean()
            daily['sent_surprise'] = daily['sent_mean'] - rolling_mean_20

            daily['sent_strength'] = daily['sent_mean'] * daily['sent_count_log']

            # NEW: 20-day EMA — captures longer-horizon sentiment regime shifts
            daily['sent_ema_20d'] = daily['sent_mean'].ewm(span=20, min_periods=1).mean()

            # === Forward-fill with exponential decay for no-news days ===
            if len(daily) > 0:
                full_idx = pd.bdate_range(daily.index.min(), daily.index.max())
                daily = daily.reindex(full_idx)

                had_news = daily['sent_count'].notna()

                decay_factor = 0.95
                for col in EXTENDED_SENTIMENT_FEATURES:
                    if col not in daily.columns:
                        continue

                    filled = daily[col].ffill()

                    cumulative_decay = (~had_news).astype(float)
                    groups = had_news.cumsum()
                    consecutive_gaps = cumulative_decay.groupby(groups).cumsum()

                    decay_mask = consecutive_gaps > 0
                    filled[decay_mask] = filled[decay_mask] * (
                        decay_factor ** consecutive_gaps[decay_mask]
                    )

                    daily[col] = filled

            daily = daily.fillna(0)
            daily = daily[EXTENDED_SENTIMENT_FEATURES]
            result[ticker] = daily

        except Exception as e:
            logger.warning(f"Failed to load sentiment for {ticker}: {e}")

    logger.info(f"Loaded sentiment for {len(result)} tickers (post-exclusion)")

    if result:
        all_dates = pd.concat([df.index.to_series() for df in result.values()])
        logger.info(
            f"Sentiment date range: {all_dates.min().date()} to {all_dates.max().date()}"
        )

    return result


def rank_tickers_by_density(
    sentiment_data: Dict[str, pd.DataFrame],
    start_date: str = "2012-01-01",
    end_date: str = "2019-12-31",
) -> pd.DataFrame:
    """
    Rank tickers by sentiment density (fraction of business days with news).

    Mirrors the density calculation in scripts/visualize_sentiment_density.py.

    Returns DataFrame with columns: ticker, days_with_news, density_pct
    """
    full_idx = pd.bdate_range(start_date, end_date)
    total_trading_days = len(full_idx)

    rows = []
    for ticker, df in sentiment_data.items():
        # Count days with non-zero sentiment in the target period
        mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
        period_df = df[mask]

        # A day "has news" if sent_mean != 0 (after decay fill, all days have
        # values, but originally-newsless days have decayed-toward-zero values).
        # Use sent_count_log > 0 as proxy: log1p(count) > 0 means count >= 1.
        days_with_news = (period_df['sent_count_log'] > 0).sum()
        density = days_with_news / total_trading_days if total_trading_days > 0 else 0

        rows.append({
            "ticker": ticker,
            "days_with_news": int(days_with_news),
            "density_pct": density * 100,
        })

    density_df = pd.DataFrame(rows).sort_values("density_pct", ascending=False).reset_index(drop=True)
    return density_df


# ============================================================================
# FLAT FEATURE MATRIX BUILDERS
# ============================================================================

def _merge_sentiment_to_df(
    df: pd.DataFrame,
    sentiment_data: Dict[str, pd.DataFrame],
    ticker: str,
) -> pd.DataFrame:
    """Merge sentiment features onto a price DataFrame."""
    if ticker in sentiment_data:
        sent_df = sentiment_data[ticker]
        price_dates = df.index.normalize()
        for col in EXTENDED_SENTIMENT_FEATURES:
            if col in sent_df.columns:
                aligned = sent_df[col].reindex(price_dates)
                df[col] = aligned.values
            else:
                df[col] = 0.0
        for col in EXTENDED_SENTIMENT_FEATURES:
            df[col] = df[col].fillna(0.0)
    else:
        for col in EXTENDED_SENTIMENT_FEATURES:
            df[col] = 0.0
    return df


def build_flat_regression_dataset(
    stock_data: Dict[str, pd.DataFrame],
    sentiment_data: Dict[str, pd.DataFrame],
    tickers: List[str],
    indicator_computer: ComprehensiveIndicatorsV7,
    vol_lookback: int = 20,
    target_clip: float = 10.0,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Build flat dataset with regression target (vol-adjusted forward return).
    Only sentiment features as inputs — RF is a feature selector, not a model.
    """
    rows = []

    for ticker in tqdm(tickers, desc="Building regression dataset"):
        if ticker not in stock_data:
            continue

        df = stock_data[ticker].copy()
        df = indicator_computer.compute_shap_top20(df)
        df = _merge_sentiment_to_df(df, sentiment_data, ticker)

        # Target: vol-adjusted forward return
        df['return_next'] = df['Close'].pct_change().shift(-1)
        df['rolling_vol'] = df['Close'].pct_change().rolling(vol_lookback).std()
        df['vol_adj_return'] = df['return_next'] / df['rolling_vol']
        df['vol_adj_return'] = df['vol_adj_return'].clip(-target_clip, target_clip)

        df = df.dropna(subset=['vol_adj_return'])
        if len(df) < 50:
            continue

        chunk = df[EXTENDED_SENTIMENT_FEATURES + ['vol_adj_return']].copy()
        chunk['ticker'] = ticker
        rows.append(chunk)

    if not rows:
        raise RuntimeError("No usable data — check sentiment parquets and price cache")

    combined = pd.concat(rows, axis=0)
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna()

    X = combined[EXTENDED_SENTIMENT_FEATURES].values.astype(np.float32)
    y = combined['vol_adj_return'].values.astype(np.float32)

    logger.info(f"Regression dataset: {X.shape[0]:,} samples × {X.shape[1]} features")
    logger.info(f"Target stats: mean={y.mean():.4f}, std={y.std():.4f}")

    return combined, y


def build_flat_classification_dataset(
    stock_data: Dict[str, pd.DataFrame],
    sentiment_data: Dict[str, pd.DataFrame],
    tickers: List[str],
    indicator_computer: ComprehensiveIndicatorsV7,
    threshold_low: float = CLASSIFICATION_THRESHOLD_LOW,
    threshold_high: float = CLASSIFICATION_THRESHOLD_HIGH,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Build flat dataset with 3-class trend classification target.

    Labels match ultimate_model_v2.py:
      0 = down  (return_next < threshold_low)
      1 = flat  (threshold_low <= return_next <= threshold_high)
      2 = up    (return_next > threshold_high)

    Only sentiment features as inputs — RF is a feature selector.
    """
    rows = []

    for ticker in tqdm(tickers, desc="Building classification dataset"):
        if ticker not in stock_data:
            continue

        df = stock_data[ticker].copy()
        df = indicator_computer.compute_shap_top20(df)
        df = _merge_sentiment_to_df(df, sentiment_data, ticker)

        # Target: 3-class trend
        df['return_next'] = df['Close'].pct_change().shift(-1)
        df['trend'] = pd.cut(
            df['return_next'],
            bins=[-np.inf, threshold_low, threshold_high, np.inf],
            labels=[0, 1, 2],
        ).astype(float)

        df = df.dropna(subset=['trend'])
        if len(df) < 50:
            continue

        chunk = df[EXTENDED_SENTIMENT_FEATURES + ['trend']].copy()
        chunk['ticker'] = ticker
        rows.append(chunk)

    if not rows:
        raise RuntimeError("No usable data — check sentiment parquets and price cache")

    combined = pd.concat(rows, axis=0)
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna()

    X = combined[EXTENDED_SENTIMENT_FEATURES].values.astype(np.float32)
    y = combined['trend'].values.astype(int)

    from collections import Counter
    class_dist = Counter(y)
    logger.info(f"Classification dataset: {X.shape[0]:,} samples × {X.shape[1]} features")
    logger.info(f"Class distribution: {dict(sorted(class_dist.items()))}")

    return combined, y


# ============================================================================
# RANDOMFOREST + PERMUTATION IMPORTANCE (supports both tasks)
# ============================================================================

def run_feature_selection_regression(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    top_k: int = 2,
    n_repeats: int = 10,
    n_estimators: int = 200,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Fit RandomForestRegressor and rank features by permutation importance.
    Scoring: R² (drop in R² when feature is shuffled).
    """
    logger.info(f"[Regression] Fitting RF ({n_estimators} trees) on {X.shape[0]:,} samples ...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_leaf=20,
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state,
    )

    t0 = time.time()
    rf.fit(X_scaled, y)
    fit_time = time.time() - t0
    logger.info(f"[Regression] RF fit in {fit_time:.1f}s  |  train R²={rf.score(X_scaled, y):.4f}")

    logger.info(f"[Regression] Computing permutation importance ({n_repeats} repeats) ...")
    t0 = time.time()
    perm_result = permutation_importance(
        rf, X_scaled, y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
        scoring="r2",
    )
    perm_time = time.time() - t0
    logger.info(f"[Regression] Permutation importance computed in {perm_time:.1f}s")

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": perm_result.importances_mean,
        "importance_std": perm_result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    importance_df["rank"] = range(1, len(importance_df) + 1)
    importance_df["selected"] = importance_df["rank"] <= top_k

    gini_map = dict(zip(feature_names, rf.feature_importances_))
    importance_df["gini_importance"] = importance_df["feature"].map(gini_map)
    importance_df["task"] = "regression"

    return importance_df


def run_feature_selection_classification(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    top_k: int = 2,
    n_repeats: int = 10,
    n_estimators: int = 200,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Fit RandomForestClassifier and rank features by permutation importance.
    Scoring: accuracy (drop in accuracy when feature is shuffled).
    """
    logger.info(f"[Classification] Fitting RF ({n_estimators} trees) on {X.shape[0]:,} samples ...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_leaf=20,
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state,
    )

    t0 = time.time()
    rf.fit(X_scaled, y)
    fit_time = time.time() - t0
    logger.info(f"[Classification] RF fit in {fit_time:.1f}s  |  train acc={rf.score(X_scaled, y):.4f}")

    logger.info(f"[Classification] Computing permutation importance ({n_repeats} repeats) ...")
    t0 = time.time()
    perm_result = permutation_importance(
        rf, X_scaled, y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
        scoring="accuracy",  # classification metric
    )
    perm_time = time.time() - t0
    logger.info(f"[Classification] Permutation importance computed in {perm_time:.1f}s")

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": perm_result.importances_mean,
        "importance_std": perm_result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    importance_df["rank"] = range(1, len(importance_df) + 1)
    importance_df["selected"] = importance_df["rank"] <= top_k

    gini_map = dict(zip(feature_names, rf.feature_importances_))
    importance_df["gini_importance"] = importance_df["feature"].map(gini_map)
    importance_df["task"] = "classification"

    return importance_df


# ============================================================================
# VISUALISATION
# ============================================================================

def save_importance_chart(
    importance_df: pd.DataFrame,
    output_path: Path,
    top_k: int = 2,
    task_label: str = "Regression",
):
    """Save a horizontal bar chart of permutation importance."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping chart")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left: Permutation importance ---
    ax = axes[0]
    df_sorted = importance_df.sort_values("importance_mean", ascending=True)
    colors = [
        "#2ecc71" if sel else "#95a5a6"
        for sel in df_sorted["selected"]
    ]
    ax.barh(
        df_sorted["feature"], df_sorted["importance_mean"],
        xerr=df_sorted["importance_std"],
        color=colors, edgecolor="white", linewidth=0.5,
    )
    scoring_label = "drop in R²" if task_label == "Regression" else "drop in Accuracy"
    ax.set_xlabel(f"Permutation Importance ({scoring_label})")
    ax.set_title(f"{task_label} — Permutation Importance\n(top {top_k} selected in green)")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    # --- Right: Gini importance (for comparison) ---
    ax2 = axes[1]
    df_sorted2 = importance_df.sort_values("gini_importance", ascending=True)
    ax2.barh(
        df_sorted2["feature"], df_sorted2["gini_importance"],
        color="#3498db", edgecolor="white", linewidth=0.5,
    )
    ax2.set_xlabel("Gini (MDI) Importance")
    ax2.set_title("Gini Importance (reference only — biased)")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Chart saved to {output_path}")


def save_combined_chart(
    reg_df: pd.DataFrame,
    cls_df: pd.DataFrame,
    output_path: Path,
    top_k: int = 2,
):
    """Save a side-by-side comparison chart: regression vs classification."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping combined chart")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, df, task in [(axes[0], reg_df, "Regression"), (axes[1], cls_df, "Classification")]:
        df_sorted = df.sort_values("importance_mean", ascending=True)
        colors = ["#2ecc71" if sel else "#95a5a6" for sel in df_sorted["selected"]]
        ax.barh(
            df_sorted["feature"], df_sorted["importance_mean"],
            xerr=df_sorted["importance_std"],
            color=colors, edgecolor="white", linewidth=0.5,
        )
        scoring = "drop in R²" if task == "Regression" else "drop in Accuracy"
        ax.set_xlabel(f"Permutation Importance ({scoring})")
        ax.set_title(f"{task}\n(top {top_k} in green)")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Combined chart saved to {output_path}")


# ============================================================================
# LOGGING HELPERS
# ============================================================================

def _log_rankings(importance_df: pd.DataFrame, task: str):
    """Pretty-print feature rankings."""
    logger.info(f"\n{'=' * 70}")
    logger.info(f"FEATURE RANKING — {task.upper()} (Permutation Importance)")
    logger.info(f"{'=' * 70}")
    for _, row in importance_df.iterrows():
        marker = "★" if row["selected"] else " "
        logger.info(
            f"  {marker} #{int(row['rank']):d}  {row['feature']:<22s}  "
            f"perm={row['importance_mean']:+.6f} ± {row['importance_std']:.6f}  "
            f"gini={row['gini_importance']:.6f}"
        )
    selected = importance_df.loc[importance_df["selected"], "feature"].tolist()
    logger.info(f"→ Selected: {selected}")
    return selected


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sentiment Feature Selection — Regression + Classification "
                    "(RandomForest Permutation Importance)"
    )
    parser.add_argument("--top-k", type=int, default=2,
                        help="Number of top features to select (default: 2)")
    parser.add_argument("--n-repeats", type=int, default=10,
                        help="Permutation importance repeats (default: 10)")
    parser.add_argument("--n-estimators", type=int, default=200,
                        help="Number of RF trees (default: 200)")
    parser.add_argument("--max-tickers", type=int, default=150,
                        help="Max tickers by density rank (default: 150, matching "
                             "visualize_sentiment_density.py top-150)")
    parser.add_argument("--start-date", type=str, default="2012-01-01")
    parser.add_argument("--end-date", type=str, default="2019-12-31")
    parser.add_argument("--threshold-low", type=float, default=CLASSIFICATION_THRESHOLD_LOW,
                        help="Classification threshold (down), default: -0.005")
    parser.add_argument("--threshold-high", type=float, default=CLASSIFICATION_THRESHOLD_HIGH,
                        help="Classification threshold (up), default: 0.005")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("SENTIMENT FEATURE SELECTION")
    logger.info("  Tasks: Regression + Classification")
    logger.info("  Method: RandomForest + Permutation Importance")
    logger.info("=" * 70)
    logger.info(f"Sentiment features: {EXTENDED_SENTIMENT_FEATURES}")
    logger.info(f"New feature: sent_ema_20d (20-day EMA of sent_mean)")
    logger.info(f"Excluded tickers: {len(EXCLUDE_TICKERS)} (from visualize_sentiment_density.py)")
    logger.info(f"Max tickers (by density): {args.max_tickers}")
    logger.info(f"Selecting top-{args.top_k} features per task")

    # --- Load sentiment (with exclusion filter) ---
    sentiment_data = load_sentiment_data_extended()
    if not sentiment_data:
        logger.error("No sentiment data found — aborting")
        sys.exit(1)

    # --- Rank by density and take top N ---
    density_df = rank_tickers_by_density(
        sentiment_data, args.start_date, args.end_date
    )
    logger.info(f"\nDensity ranking (top 10):\n{density_df.head(10).to_string(index=False)}")

    top_tickers = density_df.head(args.max_tickers)['ticker'].tolist()
    logger.info(f"\nUsing {len(top_tickers)} densest tickers for feature selection")

    # Filter sentiment data to only the selected tickers
    sentiment_data = {t: sentiment_data[t] for t in top_tickers if t in sentiment_data}

    # --- Load price data ---
    cache = PriceCache()
    indicator_computer = ComprehensiveIndicatorsV7()

    stock_data = load_stocks(
        top_tickers,
        args.start_date, args.end_date,
        cache=cache,
        min_rows=100,
        weekly=False,
    )
    logger.info(f"Loaded price data for {len(stock_data)} tickers")

    # ==================================================================
    # TASK 1: REGRESSION (vol-adjusted forward return)
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TASK 1: REGRESSION FEATURE SELECTION")
    logger.info("=" * 70)

    reg_combined, reg_y = build_flat_regression_dataset(
        stock_data, sentiment_data, top_tickers,
        indicator_computer,
    )
    reg_X = reg_combined[EXTENDED_SENTIMENT_FEATURES].values.astype(np.float32)

    reg_importance = run_feature_selection_regression(
        reg_X, reg_y,
        feature_names=EXTENDED_SENTIMENT_FEATURES,
        top_k=args.top_k,
        n_repeats=args.n_repeats,
        n_estimators=args.n_estimators,
        random_state=args.seed,
    )
    reg_selected = _log_rankings(reg_importance, "Regression")

    # ==================================================================
    # TASK 2: CLASSIFICATION (3-class trend for ultimate_model_v2.py)
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("TASK 2: CLASSIFICATION FEATURE SELECTION")
    logger.info(f"  (thresholds: [{args.threshold_low}, {args.threshold_high}])")
    logger.info("=" * 70)

    cls_combined, cls_y = build_flat_classification_dataset(
        stock_data, sentiment_data, top_tickers,
        indicator_computer,
        threshold_low=args.threshold_low,
        threshold_high=args.threshold_high,
    )
    cls_X = cls_combined[EXTENDED_SENTIMENT_FEATURES].values.astype(np.float32)

    cls_importance = run_feature_selection_classification(
        cls_X, cls_y,
        feature_names=EXTENDED_SENTIMENT_FEATURES,
        top_k=args.top_k,
        n_repeats=args.n_repeats,
        n_estimators=args.n_estimators,
        random_state=args.seed,
    )
    cls_selected = _log_rankings(cls_importance, "Classification")

    # ==================================================================
    # COMPARE TASKS
    # ==================================================================
    logger.info("\n" + "=" * 70)
    logger.info("CROSS-TASK COMPARISON")
    logger.info("=" * 70)
    logger.info(f"Regression  top-{args.top_k}: {reg_selected}")
    logger.info(f"Classification top-{args.top_k}: {cls_selected}")
    overlap = set(reg_selected) & set(cls_selected)
    if overlap:
        logger.info(f"Overlap (selected by both tasks): {sorted(overlap)}")
    else:
        logger.info("No overlap between regression and classification selections")

    # ==================================================================
    # SAVE OUTPUTS
    # ==================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = REPORTS_DIR / f"sentiment_feature_selection_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Rankings CSVs
    reg_importance.to_csv(output_dir / "regression_feature_rankings.csv", index=False)
    cls_importance.to_csv(output_dir / "classification_feature_rankings.csv", index=False)

    # Combined rankings
    combined_rankings = pd.concat([reg_importance, cls_importance], ignore_index=True)
    combined_rankings.to_csv(output_dir / "all_feature_rankings.csv", index=False)

    # Density ranking
    density_df.to_csv(output_dir / "ticker_density_ranking.csv", index=False)

    # Summary JSON
    summary = {
        "experiment": "sentiment_feature_selection",
        "timestamp": timestamp,
        "ticker_filter": {
            "excluded_count": len(EXCLUDE_TICKERS),
            "max_tickers_by_density": args.max_tickers,
            "actual_tickers_used": len(top_tickers),
            "top_10_by_density": density_df.head(10).to_dict(orient="records"),
        },
        "sentiment_features": EXTENDED_SENTIMENT_FEATURES,
        "new_feature": "sent_ema_20d (20-day EMA of sent_mean)",
        "top_k": args.top_k,
        "n_repeats": args.n_repeats,
        "n_estimators": args.n_estimators,
        "random_state": args.seed,
        "regression": {
            "n_samples": int(len(reg_y)),
            "selected_features": reg_selected,
            "scoring": "r2",
            "rankings": reg_importance.to_dict(orient="records"),
        },
        "classification": {
            "n_samples": int(len(cls_y)),
            "threshold_low": args.threshold_low,
            "threshold_high": args.threshold_high,
            "selected_features": cls_selected,
            "scoring": "accuracy",
            "rankings": cls_importance.to_dict(orient="records"),
            "note": "Labels match ultimate_model_v2.py 3-class trend (down/flat/up)",
        },
        "cross_task_overlap": sorted(overlap),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Charts
    save_importance_chart(
        reg_importance, output_dir / "regression_feature_importance.png",
        top_k=args.top_k, task_label="Regression",
    )
    save_importance_chart(
        cls_importance, output_dir / "classification_feature_importance.png",
        top_k=args.top_k, task_label="Classification",
    )
    save_combined_chart(
        reg_importance, cls_importance,
        output_dir / "combined_feature_importance.png",
        top_k=args.top_k,
    )

    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("Done.")

    return {
        "regression": (reg_importance, reg_selected),
        "classification": (cls_importance, cls_selected),
    }


if __name__ == "__main__":
    main()
