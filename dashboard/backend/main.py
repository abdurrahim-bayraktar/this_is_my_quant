"""
Quant Dashboard — FastAPI Backend

Serves experiment data, price data, and trade signals to the React frontend.
Run with: uvicorn dashboard.backend.main:app --reload --port 8000
"""

import logging
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .data_loader import (
    scan_reports,
    get_experiment_detail,
    get_trades,
    get_holdings,
    get_ticker_rankings,
)
from .price_loader import get_ohlcv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Quant Dashboard API",
    description="Serves experiment results for the Sentiment-Driven Stock Trend Prediction dashboard.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Experiment endpoints ──────────────────────────────────────────────────

@app.get("/api/experiments")
def list_experiments():
    """List all experiments sorted by headline metric (best first)."""
    experiments = scan_reports()
    return {"experiments": experiments, "total": len(experiments)}


@app.get("/api/experiment/{name}")
def get_experiment(name: str):
    """Get full experiment details."""
    detail = get_experiment_detail(name)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Experiment '{name}' not found")
    return detail


@app.get("/api/experiment/{name}/folds")
def get_folds(name: str):
    """Get per-fold results."""
    detail = get_experiment_detail(name)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Experiment '{name}' not found")
    return {"folds": detail.get("folds", [])}


@app.get("/api/experiment/{name}/backtest")
def get_backtest(name: str):
    """Get backtest results."""
    detail = get_experiment_detail(name)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Experiment '{name}' not found")
    backtest = detail.get("backtest", [])
    if not backtest:
        return {"available": False, "message": "No backtest data. Run evaluate_hybrid.py for this experiment."}
    return {"available": True, "strategies": backtest}


@app.get("/api/experiment/{name}/ic")
def get_ic_timeseries(name: str):
    """Get cross-sectional IC time series."""
    detail = get_experiment_detail(name)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Experiment '{name}' not found")
    ic = detail.get("ic", [])
    if not ic:
        return {"available": False, "message": "No IC time series data available."}
    return {"available": True, "data": ic}


@app.get("/api/experiment/{name}/trades/{strategy}")
def get_strategy_trades(name: str, strategy: str):
    """Get trade signals for a specific strategy."""
    trades = get_trades(name, strategy)
    if not trades:
        raise HTTPException(status_code=404, detail=f"No trades found for strategy '{strategy}'")
    return {"trades": trades, "total": len(trades)}


@app.get("/api/experiment/{name}/holdings/{strategy}")
def get_strategy_holdings(name: str, strategy: str):
    """Get holdings data for a specific strategy."""
    holdings = get_holdings(name, strategy)
    if not holdings:
        raise HTTPException(status_code=404, detail=f"No holdings found for strategy '{strategy}'")
    return {"holdings": holdings, "total": len(holdings)}


@app.get("/api/experiment/{name}/ticker-rankings/{strategy}")
def get_strategy_ticker_rankings(name: str, strategy: str):
    """Get tickers ranked by trade frequency for a strategy."""
    rankings = get_ticker_rankings(name, strategy)
    return {"rankings": rankings, "total": len(rankings)}


# ── Compare endpoint ──────────────────────────────────────────────────────

@app.get("/api/compare")
def compare_experiments(
    exp1: str = Query(..., description="First experiment name"),
    exp2: str = Query(..., description="Second experiment name"),
):
    """Compare two experiments side by side."""
    detail1 = get_experiment_detail(exp1)
    detail2 = get_experiment_detail(exp2)

    if not detail1:
        raise HTTPException(status_code=404, detail=f"Experiment '{exp1}' not found")
    if not detail2:
        raise HTTPException(status_code=404, detail=f"Experiment '{exp2}' not found")

    return {"exp1": detail1, "exp2": detail2}


# ── Price endpoint ────────────────────────────────────────────────────────

@app.get("/api/price/{ticker}")
def get_price_data(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
):
    """Get OHLCV candlestick data for a ticker."""
    data = get_ohlcv(ticker.upper(), start, end)
    if not data:
        raise HTTPException(status_code=404, detail=f"No price data for '{ticker}'")
    return {"ticker": ticker.upper(), "data": data, "total": len(data)}


# ── Health check ──────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    experiments = scan_reports()
    return {
        "status": "ok",
        "experiments_found": len(experiments),
    }
