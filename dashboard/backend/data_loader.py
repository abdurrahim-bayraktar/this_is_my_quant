"""
Dashboard data loader — scans reports/ and parses experiment data.

Discovers all valid experiment directories, reads their CSV/JSON outputs,
and provides structured access for the FastAPI endpoints.
"""

import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from functools import lru_cache

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent.parent.parent / "reports"


def _parse_csv_row(filepath: Path) -> Optional[Dict[str, Any]]:
    """Parse a single-row CSV (summary.csv) into a dict."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric strings
                parsed = {}
                for k, v in row.items():
                    if v in ("True", "true"):
                        parsed[k] = True
                    elif v in ("False", "false"):
                        parsed[k] = False
                    else:
                        try:
                            if "." in v:
                                parsed[k] = float(v)
                            else:
                                parsed[k] = int(v)
                        except (ValueError, TypeError):
                            parsed[k] = v
                return parsed
    except Exception as e:
        logger.warning(f"Failed to parse {filepath}: {e}")
        return None


def _parse_csv_rows(filepath: Path) -> List[Dict[str, Any]]:
    """Parse a multi-row CSV into a list of dicts."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                parsed = {}
                for k, v in row.items():
                    if v in ("True", "true"):
                        parsed[k] = True
                    elif v in ("False", "false"):
                        parsed[k] = False
                    else:
                        try:
                            if "." in v:
                                parsed[k] = float(v)
                            else:
                                parsed[k] = int(v)
                        except (ValueError, TypeError):
                            parsed[k] = v
                rows.append(parsed)
            return rows
    except Exception as e:
        logger.warning(f"Failed to parse {filepath}: {e}")
        return []


def _parse_config(filepath: Path) -> Optional[Dict[str, Any]]:
    """Parse config.json."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to parse {filepath}: {e}")
        return None


def _detect_task_type(summary: Dict[str, Any]) -> str:
    """Detect whether experiment is classification or regression."""
    task = summary.get("task", "")
    if task:
        return task
    # Fallback heuristics
    if "accuracy" in summary or "wf_agg_accuracy" in summary:
        return "classification"
    if "mse" in summary or "wf_agg_mse" in summary or "ic" in summary:
        return "regression"
    return "unknown"


def _get_headline_metric(summary: Dict[str, Any], task_type: str) -> tuple:
    """Get the headline metric name and value for sorting."""
    if task_type == "classification":
        for key in ["wf_agg_accuracy", "accuracy", "per_stock_avg_accuracy"]:
            if key in summary and isinstance(summary[key], (int, float)):
                return "Accuracy", summary[key]
        return "Accuracy", 0.0
    else:
        for key in ["cross_sectional_ic", "wf_agg_ic", "ic", "per_stock_avg_ic"]:
            if key in summary and isinstance(summary[key], (int, float)):
                return "IC", summary[key]
        return "IC", 0.0


def _list_strategies(report_dir: Path) -> List[str]:
    """List available backtesting strategies from *_trades.csv files."""
    strategies = []
    for f in report_dir.glob("*_trades.csv"):
        name = f.stem.replace("_trades", "")
        strategies.append(name)
    return sorted(strategies)


def _get_trade_ticker_stats(report_dir: Path, strategy: str) -> List[Dict]:
    """Get per-ticker trade statistics for a strategy, sorted by trade count."""
    trades_file = report_dir / f"{strategy}_trades.csv"
    if not trades_file.exists():
        return []

    rows = _parse_csv_rows(trades_file)
    if not rows:
        return []

    # Aggregate per ticker
    ticker_stats = {}
    for row in rows:
        ticker = row.get("Ticker", "")
        if not ticker:
            continue
        if ticker not in ticker_stats:
            ticker_stats[ticker] = {
                "ticker": ticker,
                "total_trades": 0,
                "buys": 0,
                "sells": 0,
                "total_weight": 0.0,
            }
        ticker_stats[ticker]["total_trades"] += 1

        direction = str(row.get("Direction", row.get("Action", ""))).lower()
        weight = float(row.get("Trade_Weight", row.get("Weight", 0)))

        if "buy" in direction:
            ticker_stats[ticker]["buys"] += 1
        elif "sell" in direction:
            ticker_stats[ticker]["sells"] += 1

        ticker_stats[ticker]["total_weight"] += abs(weight)

    # Compute derived stats
    result = []
    for stats in ticker_stats.values():
        total = stats["total_trades"]
        stats["buy_pct"] = round(stats["buys"] / total * 100, 1) if total > 0 else 0
        stats["sell_pct"] = round(stats["sells"] / total * 100, 1) if total > 0 else 0
        stats["avg_weight"] = round(stats["total_weight"] / total, 6) if total > 0 else 0
        stats["net_bias"] = "long" if stats["buys"] > stats["sells"] else "short" if stats["sells"] > stats["buys"] else "neutral"
        result.append(stats)

    # Sort by most traded
    result.sort(key=lambda x: x["total_trades"], reverse=True)
    return result


def scan_reports(reports_dir: Path = None) -> List[Dict[str, Any]]:
    """
    Scan the reports directory and return metadata for all valid experiments.

    Returns a list of experiment dicts sorted by headline metric (best first).
    """
    rdir = reports_dir or REPORTS_DIR
    if not rdir.exists():
        logger.warning(f"Reports directory not found: {rdir}")
        return []

    experiments = []

    for entry in sorted(rdir.iterdir()):
        if not entry.is_dir():
            continue

        summary_path = entry / "summary.csv"
        config_path = entry / "config.json"
        has_summary = summary_path.exists()
        has_config = config_path.exists()
        has_backtest = (entry / "backtest_results.csv").exists()

        if not has_summary and not has_config and not has_backtest:
            continue

        summary = _parse_csv_row(summary_path) if has_summary else {}
        if summary is None:
            summary = {}

        config = _parse_config(config_path) if has_config else {}
        if config is None:
            config = {}
            
        task_type = _detect_task_type(summary)
        headline_name, headline_value = _get_headline_metric(summary, task_type)

        has_folds = (entry / "per_fold_results.csv").exists()
        has_stocks = (entry / "per_stock_results.csv").exists()
        has_backtest = (entry / "backtest_results.csv").exists()
        has_ic = (entry / "cross_sectional_ic.csv").exists()
        strategies = _list_strategies(entry)

        experiments.append({
            "name": entry.name,
            "task": task_type,
            "architecture": summary.get("architecture", config.get("architecture", "unknown")),
            "model_type": summary.get("model_type", config.get("model_type", "unknown")),
            "use_sentiment": summary.get("use_sentiment", config.get("use_sentiment", False)),
            "n_params": summary.get("n_params", config.get("n_params", 0)),
            "n_features": summary.get("n_features", config.get("n_features", 0)),
            "frequency": summary.get("frequency", config.get("frequency", "daily")),
            "headline_metric_name": headline_name,
            "headline_metric_value": headline_value,
            "has_folds": has_folds,
            "has_stocks": has_stocks,
            "has_backtest": has_backtest,
            "has_ic": has_ic,
            "strategies": strategies,
            "n_folds": summary.get("n_folds", summary.get("n_valid_folds", 0)),
            "summary": summary,
        })

    # Sort by headline metric (best first within each task type)
    experiments.sort(key=lambda x: x["headline_metric_value"], reverse=True)
    return experiments


def get_experiment_detail(name: str, reports_dir: Path = None) -> Optional[Dict[str, Any]]:
    """Get full experiment details including all sub-data."""
    rdir = reports_dir or REPORTS_DIR
    exp_dir = rdir / name

    if not exp_dir.exists():
        return None

    summary = _parse_csv_row(exp_dir / "summary.csv") if (exp_dir / "summary.csv").exists() else {}
    if summary is None:
        summary = {}

    config = _parse_config(exp_dir / "config.json") if (exp_dir / "config.json").exists() else {}
    if config is None:
        config = {}

    if not summary and not config and not (exp_dir / "backtest_results.csv").exists():
        return None

    result = {
        "name": name,
        "summary": summary,
        "config": config,
        "folds": [],
        "stocks": [],
        "backtest": [],
        "ic": [],
        "strategies": _list_strategies(exp_dir),
    }

    # Per-fold results
    folds_path = exp_dir / "per_fold_results.csv"
    if folds_path.exists():
        result["folds"] = _parse_csv_rows(folds_path)

    # Per-stock results
    stocks_path = exp_dir / "per_stock_results.csv"
    if stocks_path.exists():
        result["stocks"] = _parse_csv_rows(stocks_path)

    # Backtest results
    bt_path = exp_dir / "backtest_results.csv"
    if bt_path.exists():
        result["backtest"] = _parse_csv_rows(bt_path)

    # IC time series
    ic_path = exp_dir / "cross_sectional_ic.csv"
    if ic_path.exists():
        result["ic"] = _parse_csv_rows(ic_path)

    return result


def get_trades(name: str, strategy: str, reports_dir: Path = None) -> List[Dict]:
    """Get trade data for a specific strategy."""
    rdir = reports_dir or REPORTS_DIR
    trades_file = rdir / name / f"{strategy}_trades.csv"
    if not trades_file.exists():
        return []
    return _parse_csv_rows(trades_file)


def get_holdings(name: str, strategy: str, reports_dir: Path = None) -> List[Dict]:
    """Get holdings data for a specific strategy."""
    rdir = reports_dir or REPORTS_DIR
    holdings_file = rdir / name / f"{strategy}_holdings.csv"
    if not holdings_file.exists():
        return []
    return _parse_csv_rows(holdings_file)


def get_ticker_rankings(name: str, strategy: str, reports_dir: Path = None) -> List[Dict]:
    """Get tickers ranked by trade frequency for a strategy."""
    rdir = reports_dir or REPORTS_DIR
    return _get_trade_ticker_stats(rdir / name, strategy)
