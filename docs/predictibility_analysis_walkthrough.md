# Per-Stock Predictability Analyzer

I completely agree with your suggestion—keeping this as a separate tool is much cleaner and avoids cluttering the backtesting logic! 

To implement this efficiently without having to wait 6+ minutes to re-generate predictions every time you want to analyze them, I took the following approach:

## 1. Saved Predictions in Backtesters
I added a single line to all four of your backtesting scripts (`evaluate_darnn_classification.py`, `evaluate_darnn.py`, etc.). Now, immediately after they finish generating predictions for the test set, they will automatically save a `predictions.csv` file into their respective `reports/bt_...` output directory.

## 2. Created `analyze_predictability.py`
I created a brand new, standalone script located at:
[analyze_predictability.py](file:///c:/dev/this_is_my_quant/experiments/analyze_predictability.py)

This script is incredibly fast because it simply reads the `predictions.csv` from an already-completed backtest run. It automatically detects what kind of model was run (Classification vs Regression vs Karash) and computes the appropriate per-stock metrics:

* **Classification Models**: Computes the directional **Accuracy**.
* **Regression Models**: Computes the **Correlation** and **MSE**.

### How to use it:

Once your current backtest finishes (or on any future backtest run), simply copy its output directory path and run:

```bash
python experiments/analyze_predictability.py --eval-dir reports/bt_darnn_class_...
```

It will instantly output a full `per_stock_metrics.csv` into that same directory, and print out the **Top 10** most predictable stocks (like SPY, hopefully!) and the **Bottom 10** completely orthogonal stocks right in your terminal. You can use `--top-n 20` to see more.
