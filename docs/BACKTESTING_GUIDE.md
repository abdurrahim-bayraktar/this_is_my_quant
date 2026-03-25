# Backtesting Framework Guide

This guide explains the components of the vector-based backtesting framework added to the project, how to run evaluations for experiments (like the `ultimate_model`), and how to extend the framework with custom strategies.

## Overview

The backtesting framework is located in the `backtesting/` directory at the project root. It is designed to be completely decoupled from the training and experiment codebase (`src/` and `experiments/`), ensuring no unintended modifications are made to the core logic.

It consists of two primary files:
1. **`framework.py`**: The core vector-based engine (`Backtester`). It accepts a DataFrame of predictions and simulate trades day-by-day.
2. **`evaluate_ultimate_model.py`**: A specialized runner script that connects the output of the `ultimate_model` experiment to the `Backtester`.

## 1. Running an Evaluation

You can run a backtest on any previously completed `ultimate_model` experiment by passing the path to its report directory.

**Command:**
```bash
python backtesting/evaluate_ultimate_model.py --report-dir reports/ultimate_model_YYYYMMDD_HHMMSS
```

### What happens during execution?
1. **Reconstruction**: The script reads `config.json` inside the report directory to determine the training and testing sets. Crucially, it re-loads the training stocks up to `2022-01-01` and **refits the `StandardScaler`**. This guarantees that the test set features are scaled using *only* the data the model historically learned from, completely preventing data leakage.
2. **Prediction Gathering**: The script iterates through the `test_stocks`, creating temporal sequences from the test partition (`2023-06-01` onward). It runs inference in batches to produce daily probabilities (`Prob_Up`, `Prob_Down`).
3. **Simulation**: The generated probabilities, along with exact prediction dates and actual `Return_Next` values, are passed to the `Backtester` to run a battery of predefined strategies.
4. **Metrics Generation**: A comparison table is printed to the console, and a `backtest_results.csv` is saved in the same report directory.

## 2. Built-in Strategies

The framework currently simulates trading based on the 3-way prediction schema (Up > 0.5%, Neutral, Down < -0.5%). The following strategies are included:

- **Long_Top_N (Conviction Trading)**: Selects the Top $N$ stocks with the highest strictly predicted `Prob_Up` each day. Weights them equally.
- **Long_Short_Neutral (Market Neutral)**: Buys the Top $N$ predicted "Up" stocks and short-sells the Top $N$ predicted "Down" stocks simultaneously, aiming for beta neutrality.
- **Threshold_Long (Confidence)**: Buys any stock that crosses an absolute probability threshold (e.g., `> 60%`). If Neural Network probabilities are uncalibrated (highly clustered around 33-40%), this strategy may rarely trigger.
- **Daily_Rebalanced_Universe\":** Forces the portfolio back to equal weight every day by actively trading.
- **Buy_Hold_Universe (True Baseline)**: Buys equal weight on day 1 and lets the allocations naturally drift strictly tracking raw market trajectory. Effectively $0$ continuous turnover.
- **Random_Allocation (Sanity Check)**: Selects $N$ random stocks daily. Crucial to ensure your Conviction Trading actually has mathematical edge over throwing darts.

## 3. Metrics

The `Backtester` outputs standard quant metrics:
- **Total Return**: Cumulative return over the test period.
- **Annualized Return**: Scaled to a 252 trading-day calendar.
- **Max Drawdown**: The deepest peak-to-trough drop in equity during the backtest.
- **Annualized Volatility**: Risk metric; standard deviation of daily returns annualized.
- **Sharpe Ratio**: Risk-adjusted return measure (Annualized Return / Annualized Volatility). A score > 1.0 is considered good.
- **Sortino Ratio**: Similar to Sharpe, but only penalizes downside volatility.
- **Daily Win Rate**: The percentage of days where the simulated portfolio generated a strictly positive return.

## 4. Extending the Framework

To implement your own custom allocation strategies, modify the `run_strategy` method inside `backtesting/framework.py`:

```python
# Inside Backtester.run_strategy() loop for 'date'
elif strategy_name == "My_Custom_Strategy":
    # Retrieve the day's predictions (index is Date)
    # day_data columns include: 'Ticker', 'Prob_Up', 'Prob_Down', 'Return_Next'
    
    # 1. Implement logic...
    selected = day_data[(day_data['Prob_Up'] > 0.45) & (day_data['Prob_Down'] < 0.20)]['Ticker']
    
    # 2. Assign target weights (weights should preferably sum to 1.0 or less)
    if not selected.empty:
        weights[selected] = 1.0 / len(selected)
```

The `Backtester` engine will automatically handle aligning the target `weights` against the `Return_Next` outcomes and subtract standard transaction costs (configurable via `transaction_cost_bps` during `Backtester` initialization) based on daily turnover.
