"""
Regression Feature Selection via SHAP (TreeExplainer + LightGBM).

Standalone analysis script — does NOT modify any existing experiment.
Selects the top-N features from the FULL ComprehensiveIndicatorsV7 feature set
for the regression task (vol-adjusted forward return).

SHAP values are computed ONCE and cached to disk so subsequent runs
skip the expensive SHAP computation and just load the cached results.

Pipeline:
  1. Load stock data → compute ALL V7 indicators
  2. Build flat regression dataset (vol-adjusted forward return target)
  3. Fit LightGBM surrogate model (fast, tree-based)
  4. Compute SHAP values via TreeExplainer
  5. Rank features by mean |SHAP|, select top-N
  6. Cache SHAP results + rankings to reports/

Excluded (leaky) features:
  - ichimoku_ICS_26  (forward-shifted Ichimoku span)
  - dpo              (Detrended Price Oscillator — lookahead)

Usage:
    python experiments/regression_feature_selection.py
    python experiments/regression_feature_selection.py --top-k 20 --stocks 200
    python experiments/regression_feature_selection.py --force  # re-run SHAP
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
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from experiments.ranked_tickers import EXTENDED_TICKERS
from src.data.cache import PriceCache
from src.data.utils import load_stocks
from src.features.indicators_v7 import ComprehensiveIndicatorsV7

try:
    from config import REPORTS_DIR
except ImportError:
    REPORTS_DIR = Path("reports")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# EXCLUDED (LEAKY) FEATURES
# ============================================================================

EXCLUDED_FEATURES = [
    'ichimoku_ICS_26',
    'dpo',
]


# ============================================================================
# CACHE PATHS
# ============================================================================

CACHE_DIR = REPORTS_DIR / "regression_feature_selection_cache"


def _get_cache_paths(cache_dir: Path) -> Dict[str, Path]:
    """Return paths for all cached artifacts."""
    return {
        "shap_values": cache_dir / "shap_values.npy",
        "feature_names": cache_dir / "feature_names.json",
        "rankings": cache_dir / "feature_rankings.csv",
        "summary": cache_dir / "summary.json",
        "shap_summary_plot": cache_dir / "shap_summary_plot.png",
        "top_features_chart": cache_dir / "top_features_chart.png",
    }


def _cache_exists(cache_dir: Path) -> bool:
    """Check if a valid SHAP cache exists."""
    paths = _get_cache_paths(cache_dir)
    required = ["shap_values", "feature_names", "rankings"]
    return all(paths[k].exists() for k in required)


# ============================================================================
# DATASET BUILDER — full V7 features for regression
# ============================================================================

def build_flat_regression_dataset(
    stock_data: Dict[str, pd.DataFrame],
    tickers: List[str],
    indicator_computer: ComprehensiveIndicatorsV7,
    excluded_features: List[str],
    vol_lookback: int = 20,
    target_clip: float = 10.0,
    train_end: str = "2022-01-01",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build a flat (non-sequential) dataset for SHAP feature selection.

    Uses ALL V7 indicators (minus excluded/leaky ones) as features,
    and vol-adjusted forward return as the regression target.

    Only uses data before train_end to avoid data leakage into test set.

    Returns:
        X: (n_samples, n_features) array
        y: (n_samples,) array
        feature_names: list of feature column names
    """
    train_end_ts = pd.Timestamp(train_end)
    exclude_cols = set([
        'Open', 'High', 'Low', 'Close', 'Volume', 'Date', 'date',
        'Adj Close', 'Dividends', 'Stock Splits', 'ticker', 'Ticker',
        'trend', 'return_next', 'actual_return', 'actual_trend',
        'rolling_vol', 'vol_adj_return',
    ] + excluded_features)

    all_chunks = []
    feature_cols = None

    for ticker in tqdm(tickers, desc="Building regression dataset"):
        if ticker not in stock_data:
            continue

        df = stock_data[ticker].copy()

        try:
            df = indicator_computer.compute_all(df)
        except Exception as e:
            logger.warning(f"Failed to compute indicators for {ticker}: {e}")
            continue

        # Vol-adjusted forward return target
        df['return_next'] = df['Close'].pct_change().shift(-1)
        df['rolling_vol'] = df['Close'].pct_change().rolling(vol_lookback).std()
        df['vol_adj_return'] = df['return_next'] / df['rolling_vol']
        df['vol_adj_return'] = df['vol_adj_return'].clip(-target_clip, target_clip)

        # Only use training data
        df = df[df.index < train_end_ts]

        # Determine feature columns from first valid stock
        if feature_cols is None:
            feature_cols = [
                c for c in df.columns
                if c not in exclude_cols
                and not c.startswith('psar_')  # SAR has direction-encoded NaNs
            ]
            # Filter out columns that are mostly NaN
            valid_cols = []
            for c in feature_cols:
                if df[c].notna().mean() > 0.5:
                    valid_cols.append(c)
            feature_cols = valid_cols
            logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols[:10]}... ")

        df = df.dropna(subset=['vol_adj_return'])
        if len(df) < 50:
            continue

        chunk = df[feature_cols + ['vol_adj_return']].copy()
        all_chunks.append(chunk)

    if not all_chunks:
        raise RuntimeError("No usable data — check price cache")

    combined = pd.concat(all_chunks, axis=0)
    combined = combined.replace([np.inf, -np.inf], np.nan).fillna(0)

    X = combined[feature_cols].values.astype(np.float32)
    y = combined['vol_adj_return'].values.astype(np.float32)

    logger.info(f"Dataset: {X.shape[0]:,} samples × {X.shape[1]} features")
    logger.info(f"Target stats: mean={y.mean():.4f}, std={y.std():.4f}, "
                f"min={y.min():.4f}, max={y.max():.4f}")

    return X, y, feature_cols


# ============================================================================
# SHAP FEATURE SELECTION
# ============================================================================

def run_shap_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    top_k: int = 20,
    n_estimators: int = 500,
    max_shap_samples: int = 50_000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Fit LightGBM and compute SHAP values for feature ranking.

    Uses TreeExplainer (exact, fast for tree models).
    If the dataset is large, subsamples for SHAP computation.

    Returns:
        importance_df: DataFrame with feature rankings
        shap_values: raw SHAP values array (n_samples, n_features)
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError(
            "LightGBM required for SHAP feature selection. "
            "Install with: pip install lightgbm"
        )

    try:
        import shap
    except ImportError:
        raise ImportError(
            "SHAP required for feature selection. "
            "Install with: pip install shap"
        )

    logger.info(f"Fitting LightGBM ({n_estimators} trees) on {X.shape[0]:,} samples, "
                f"{X.shape[1]} features...")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Fit LightGBM
    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=8,
        learning_rate=0.05,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=random_state,
        verbose=-1,
    )

    t0 = time.time()
    model.fit(X_scaled, y)
    fit_time = time.time() - t0
    logger.info(f"LightGBM fit in {fit_time:.1f}s")

    # Subsample if needed for SHAP
    if X_scaled.shape[0] > max_shap_samples:
        logger.info(f"Subsampling {max_shap_samples:,} from {X_scaled.shape[0]:,} for SHAP...")
        rng = np.random.RandomState(random_state)
        idx = rng.choice(X_scaled.shape[0], max_shap_samples, replace=False)
        X_shap = X_scaled[idx]
    else:
        X_shap = X_scaled

    # SHAP TreeExplainer
    logger.info(f"Computing SHAP values ({X_shap.shape[0]:,} samples)...")
    t0 = time.time()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
    shap_time = time.time() - t0
    logger.info(f"SHAP computed in {shap_time:.1f}s")

    # Rank by mean |SHAP|
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
        "std_abs_shap": np.abs(shap_values).std(axis=0),
        "lgbm_importance": model.feature_importances_,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    importance_df["rank"] = range(1, len(importance_df) + 1)
    importance_df["selected"] = importance_df["rank"] <= top_k

    # Normalized importance (fraction of total)
    total_shap = mean_abs_shap.sum()
    importance_df["shap_pct"] = (importance_df["mean_abs_shap"] / total_shap * 100).round(2)
    importance_df["cumulative_shap_pct"] = importance_df["shap_pct"].cumsum().round(2)

    return importance_df, shap_values


# ============================================================================
# VISUALIZATION
# ============================================================================

def save_shap_charts(
    importance_df: pd.DataFrame,
    shap_values: np.ndarray,
    feature_names: List[str],
    output_dir: Path,
    top_k: int = 20,
):
    """Save feature importance visualizations."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping charts")
        return

    # --- 1. Horizontal bar chart of top features ---
    fig, ax = plt.subplots(figsize=(12, max(8, top_k * 0.4)))

    top_df = importance_df.head(top_k).iloc[::-1]
    colors = ["#2ecc71" if sel else "#95a5a6" for sel in top_df["selected"]]

    ax.barh(
        top_df["feature"], top_df["mean_abs_shap"],
        xerr=top_df["std_abs_shap"],
        color=colors, edgecolor="white", linewidth=0.5,
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"SHAP Feature Importance — Top {top_k} for Regression\n"
                 f"(vol-adjusted forward return, LightGBM surrogate)")

    # Add percentage labels
    for i, (_, row) in enumerate(top_df.iterrows()):
        ax.text(row["mean_abs_shap"] + row["std_abs_shap"] * 0.1, i,
                f" {row['shap_pct']:.1f}%", va='center', fontsize=8)

    plt.tight_layout()
    chart_path = output_dir / "top_features_chart.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Chart saved to {chart_path}")

    # --- 2. SHAP summary beeswarm plot ---
    try:
        import shap

        fig, ax = plt.subplots(figsize=(12, max(8, top_k * 0.4)))

        # Create a subset of SHAP values for top features only
        top_indices = [feature_names.index(f) for f in importance_df.head(top_k)["feature"]]
        shap_subset = shap_values[:, top_indices]
        feature_subset = [feature_names[i] for i in top_indices]

        shap.summary_plot(
            shap_subset,
            feature_names=feature_subset,
            show=False,
            max_display=top_k,
        )
        plt.tight_layout()
        summary_path = output_dir / "shap_summary_plot.png"
        plt.savefig(summary_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP summary plot saved to {summary_path}")

    except Exception as e:
        logger.warning(f"Failed to create SHAP summary plot: {e}")

    # --- 3. Cumulative importance chart ---
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        range(1, len(importance_df) + 1),
        importance_df["cumulative_shap_pct"],
        'b-o', markersize=3, linewidth=1.5,
    )
    ax.axhline(y=80, color='r', linestyle='--', alpha=0.5, label='80% threshold')
    ax.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='90% threshold')
    ax.axvline(x=top_k, color='green', linestyle='--', alpha=0.5,
               label=f'Top {top_k} cutoff')
    ax.set_xlabel("Number of Features (ranked by SHAP)")
    ax.set_ylabel("Cumulative SHAP Importance (%)")
    ax.set_title("Cumulative Feature Importance — Diminishing Returns Analysis")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    cumulative_path = output_dir / "cumulative_importance.png"
    fig.savefig(cumulative_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Cumulative chart saved to {cumulative_path}")


# ============================================================================
# LOGGING HELPERS
# ============================================================================

def _log_rankings(importance_df: pd.DataFrame, top_k: int) -> List[str]:
    """Pretty-print feature rankings."""
    logger.info(f"\n{'=' * 80}")
    logger.info(f"SHAP FEATURE RANKING — REGRESSION (Top {top_k})")
    logger.info(f"{'=' * 80}")

    for _, row in importance_df.iterrows():
        marker = "★" if row["selected"] else " "
        logger.info(
            f"  {marker} #{int(row['rank']):3d}  {row['feature']:<30s}  "
            f"SHAP={row['mean_abs_shap']:.6f} ± {row['std_abs_shap']:.6f}  "
            f"({row['shap_pct']:.1f}%, cum={row['cumulative_shap_pct']:.1f}%)  "
            f"lgbm={row['lgbm_importance']}"
        )

    selected = importance_df.loc[importance_df["selected"], "feature"].tolist()
    logger.info(f"\n→ Selected top-{top_k}: {selected}")

    return selected


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Regression Feature Selection — SHAP (LightGBM TreeExplainer)"
    )
    parser.add_argument("--top-k", type=int, default=20,
                        help="Number of top features to select (default: 20)")
    parser.add_argument("--stocks", type=int, default=200,
                        help="Number of training stocks (default: 200)")
    parser.add_argument("--n-estimators", type=int, default=500,
                        help="Number of LightGBM trees (default: 500)")
    parser.add_argument("--max-shap-samples", type=int, default=50_000,
                        help="Max samples for SHAP computation (default: 50000)")
    parser.add_argument("--start-date", type=str, default="2014-01-01")
    parser.add_argument("--end-date", type=str, default="2024-12-31")
    parser.add_argument("--train-end", type=str, default="2022-01-01",
                        help="Only use data before this date for feature selection")
    parser.add_argument("--vol-lookback", type=int, default=20)
    parser.add_argument("--target-clip", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true",
                        help="Force re-run SHAP even if cache exists")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Custom cache directory (default: reports/regression_feature_selection_cache)")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir else CACHE_DIR
    cache_paths = _get_cache_paths(cache_dir)

    logger.info("=" * 80)
    logger.info("REGRESSION FEATURE SELECTION — SHAP (LightGBM TreeExplainer)")
    logger.info("=" * 80)
    logger.info(f"Selecting top-{args.top_k} features from full V7 indicator set")
    logger.info(f"Excluded (leaky) features: {EXCLUDED_FEATURES}")
    logger.info(f"Training stocks: {args.stocks}")
    logger.info(f"Cache dir: {cache_dir}")

    # ======================================================================
    # CHECK CACHE — skip SHAP if results exist
    # ======================================================================

    if not args.force and _cache_exists(cache_dir):
        logger.info("\n" + "=" * 60)
        logger.info("LOADING CACHED SHAP RESULTS (use --force to re-run)")
        logger.info("=" * 60)

        with open(cache_paths["feature_names"], "r") as f:
            feature_names = json.load(f)
        shap_values = np.load(cache_paths["shap_values"])
        importance_df = pd.read_csv(cache_paths["rankings"])

        logger.info(f"Loaded {len(feature_names)} features, "
                    f"SHAP values shape: {shap_values.shape}")

        # Re-apply top-k selection (may differ from cached run)
        importance_df = importance_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        importance_df["rank"] = range(1, len(importance_df) + 1)
        importance_df["selected"] = importance_df["rank"] <= args.top_k

        total_shap = importance_df["mean_abs_shap"].sum()
        importance_df["shap_pct"] = (importance_df["mean_abs_shap"] / total_shap * 100).round(2)
        importance_df["cumulative_shap_pct"] = importance_df["shap_pct"].cumsum().round(2)

        selected = _log_rankings(importance_df, args.top_k)

        # Generate code snippet
        _print_code_snippet(selected)

        # Re-save charts with potentially new top-k
        save_shap_charts(importance_df, shap_values, feature_names, cache_dir, args.top_k)

        logger.info(f"\nResults in: {cache_dir}")
        return

    # ======================================================================
    # FULL SHAP PIPELINE
    # ======================================================================

    logger.info("\n" + "=" * 60)
    logger.info("RUNNING FULL SHAP PIPELINE")
    logger.info("=" * 60)

    tickers = EXTENDED_TICKERS[:args.stocks]

    # Load price data
    cache = PriceCache()
    stock_data = load_stocks(
        tickers,
        args.start_date, args.end_date,
        cache=cache,
        min_rows=100,
        weekly=False,
    )
    logger.info(f"Loaded price data for {len(stock_data)} tickers")

    # Build flat dataset
    indicator_computer = ComprehensiveIndicatorsV7()
    X, y, feature_names = build_flat_regression_dataset(
        stock_data, tickers, indicator_computer,
        excluded_features=EXCLUDED_FEATURES,
        vol_lookback=args.vol_lookback,
        target_clip=args.target_clip,
        train_end=args.train_end,
    )

    # Run SHAP
    importance_df, shap_values = run_shap_feature_selection(
        X, y, feature_names,
        top_k=args.top_k,
        n_estimators=args.n_estimators,
        max_shap_samples=args.max_shap_samples,
        random_state=args.seed,
    )

    # Log rankings
    selected = _log_rankings(importance_df, args.top_k)

    # ======================================================================
    # SAVE CACHE + OUTPUTS
    # ======================================================================

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Save SHAP values (large array)
    np.save(cache_paths["shap_values"], shap_values)
    logger.info(f"SHAP values saved: {cache_paths['shap_values']} "
                f"({shap_values.nbytes / 1024**2:.1f} MB)")

    # Save feature names
    with open(cache_paths["feature_names"], "w") as f:
        json.dump(feature_names, f, indent=2)

    # Save rankings CSV
    importance_df.to_csv(cache_paths["rankings"], index=False)

    # Save summary JSON
    summary = {
        "experiment": "regression_feature_selection",
        "method": "SHAP_TreeExplainer_LightGBM",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "top_k": args.top_k,
            "n_stocks": args.stocks,
            "n_estimators": args.n_estimators,
            "max_shap_samples": args.max_shap_samples,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "train_end": args.train_end,
            "vol_lookback": args.vol_lookback,
            "target_clip": args.target_clip,
            "seed": args.seed,
        },
        "excluded_features": EXCLUDED_FEATURES,
        "dataset": {
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "target_mean": float(y.mean()),
            "target_std": float(y.std()),
        },
        "shap_values_shape": list(shap_values.shape),
        "selected_features": selected,
        "all_feature_names": feature_names,
        "top_20_rankings": importance_df.head(20).to_dict(orient="records"),
    }
    with open(cache_paths["summary"], "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save charts
    save_shap_charts(importance_df, shap_values, feature_names, cache_dir, args.top_k)

    # Print code snippet
    _print_code_snippet(selected)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"DONE — Results cached to: {cache_dir}")
    logger.info(f"Re-run without --force to load from cache instantly.")
    logger.info(f"{'=' * 60}")


def _print_code_snippet(selected: List[str]):
    """Print a copy-pasteable Python list for use in experiments."""
    logger.info(f"\n{'=' * 60}")
    logger.info("COPY-PASTE INTO regression_v1.py:")
    logger.info(f"{'=' * 60}")

    # Format as Python list
    lines = ["SHAP_TOP20_FEATURES = ["]
    for i in range(0, len(selected), 5):
        batch = selected[i:i + 5]
        items = ", ".join(f'"{f}"' for f in batch)
        lines.append(f"    {items},")
    lines.append("]")
    snippet = "\n".join(lines)

    logger.info(f"\n{snippet}\n")


if __name__ == "__main__":
    main()
