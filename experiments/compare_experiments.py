"""
Compare experiment results across multiple runs.

Reads config.json + summary.csv from reports/ directories and generates
a side-by-side comparison table.

Usage:
    # Compare specific experiments
    python experiments/compare_experiments.py reports/ultimate_model_* 

    # Compare all experiments
    python experiments/compare_experiments.py reports/*/

    # Save to CSV
    python experiments/compare_experiments.py reports/*/ --output comparison.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import pandas as pd
from typing import List


def load_experiment(report_dir: Path) -> dict:
    """Load config + summary from a report directory."""
    result = {"directory": report_dir.name}
    
    # Load config
    config_path = report_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        result.update(config)
    
    # Load summary
    summary_path = report_dir / "summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        if len(summary) > 0:
            result.update(summary.iloc[0].to_dict())
    
    return result


def compare_experiments(report_dirs: List[Path], output_path: str = None):
    """Load and compare multiple experiments."""
    results = []
    
    for d in sorted(report_dirs):
        if not d.is_dir():
            continue
        if not (d / "summary.csv").exists():
            continue
        results.append(load_experiment(d))
    
    if not results:
        print("No valid experiment directories found.")
        return
    
    df = pd.DataFrame(results)
    
    # Select display columns (prefer these, fall back to what's available)
    display_cols = [
        "directory",
        "experiment_name",
        "model_type",
        "frequency",
        "n_params",
        "train_stocks",
        "sequence_length",
        "threshold_low",
        "threshold_high",
        # Pooled metrics
        "pooled_accuracy",
        "pooled_lift",
        "pooled_mcc",
        "pooled_n_samples",
        # Per-stock metrics
        "per_stock_avg_accuracy",
        "per_stock_avg_lift",
        "per_stock_avg_mcc",
        # Other
        "train_time",
        "dropout",
        "label_smoothing",
    ]
    
    # Also support older experiment formats
    alt_cols = {
        "accuracy": "pooled_accuracy",
        "accuracy_lift": "pooled_lift",
        "mcc": "pooled_mcc",
        "avg_accuracy": "per_stock_avg_accuracy",
        "avg_lift": "per_stock_avg_lift",
        "avg_mcc": "per_stock_avg_mcc",
    }
    
    for old_name, new_name in alt_cols.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]
    
    available = [c for c in display_cols if c in df.columns]
    display_df = df[available]
    
    # Format percentages
    pct_cols = [c for c in available if any(x in c for x in ["accuracy", "lift", "zero_rule"])]
    for col in pct_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.2%}" if pd.notna(x) else ""
            )
    
    mcc_cols = [c for c in available if "mcc" in c]
    for col in mcc_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.4f}" if pd.notna(x) else ""
            )
    
    # Print table
    print("\n" + "=" * 100)
    print("EXPERIMENT COMPARISON")
    print("=" * 100)
    print(display_df.to_string(index=False))
    print("=" * 100)
    
    # Save if requested
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\nFull results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare experiment results")
    parser.add_argument("dirs", nargs="+", type=Path, help="Report directories to compare")
    parser.add_argument("--output", "-o", type=str, default=None, help="Save comparison to CSV")
    args = parser.parse_args()
    
    # Expand glob patterns
    all_dirs = []
    for d in args.dirs:
        if d.exists():
            all_dirs.append(d)
        else:
            # Try as glob from parent
            parent = d.parent
            pattern = d.name
            all_dirs.extend(parent.glob(pattern))
    
    compare_experiments(all_dirs, args.output)


if __name__ == "__main__":
    main()
