# Class Imbalance and Threshold Optimization

## Overview
In the context of predicting next-day stock trends (Up, Neutral, Down), static boundaries (e.g., $\pm 0.5\%$) often create severe class imbalances. A $0.5\%$ move is normal noise for high-beta stocks (like TSLA) but represents a significant directional move for low-beta stocks or index ETFs.

This documentation outlines the methodology and results for discovering optimal classification boundaries to achieve a balanced target distribution (~33.3% per class).

## Optimization Approaches

The script `experiments/class_balance_optimization.py` tests two primary methodologies across a pooled dataset of 400 stocks:

### 1. Fixed Global Thresholds
Tests a range of fixed percentages (from $0.1\%$ to $2.0\%$).
- **Advantage**: Simple to implement, computationally cheap.
- **Disadvantage**: Punishes diverse stock pools; invariably categorizes volatile stocks mostly as Up/Down and stable stocks entirely as Neutral.

### 2. Dynamic Volatility-Adjusted Thresholds
Scales the threshold dynamically based on the recent volatility of the individual asset. The script evaluates two volatility metrics:
- **ATR Percentage (Average True Range)**: The 14-day ATR normalized by the closing price.
- **20-Day Standard Deviation**: The rolling 20-day standard deviation of daily returns.

The model tests various multipliers (from $0.1\times$ to $1.5\times$ the volatility metric) to find the sweet spot that distributes classes evenly.
- **Advantage**: Normalizes the definition of a "significant move" across all assets.
- **Disadvantage**: Requires calculating ATR/Vol for every target during inference.

## Results & Optimal Parameters
*Note: Run `python experiments/class_balance_optimization.py --stocks 400` to regenerate these metrics.*

### The Importance of Stock-to-Stock Coherence 
Initially, optimization focused on **Global MSE**—concatenating all returns and aiming for a 33% split. However, this permitted a dangerous bias: a global threshold might classify high-beta stocks (TSLA) entirely as Up/Down, and low-beta stocks (JNJ) entirely as Neutral, seemingly achieving "global balance" while destroying "local balance". 

To correct this, the metric was updated to **Per-Stock MSE**: calculating the distribution error for *each individual stock*, and then averaging those errors. The results conclusively prove that dynamic thresholds are mandatory for model health.

**1. Dynamic Volatility (20d) Threshold (OVERALL BEST)**:
- Best Multiplier: 0.35$\times$ 20-day Rolling Standard Deviation
- **Per-Stock MSE: 0.0009** (Best overall)
- Global MSE: 0.0004
- Target Distribution: Down: 31.8%, Neutral: 32.0%, Up: 36.2%

**2. Dynamic ATR Threshold:**
- Best Multiplier: 0.25$\times$ ATR(14) Percentage
- **Per-Stock MSE: 0.0013** 
- Global MSE: 0.0003
- Target Distribution: Down: 31.4%, Neutral: 32.8%, Up: 35.7%

**3. Global Fixed Threshold (FLAWED LOCALLY):**
- Best Threshold: 0.006 ($0.6\%$)
- **Per-Stock MSE: 0.0088** (10x worse than Volatility!)
- Global MSE: 0.0003
- Target Distribution: Down: 30.9%, Neutral: 34.5%, Up: 34.6%

## Market Microstructure & LSTM Learning Dynamics

To determine the most mathematically sound approach for a neural network, we must analyze how financial markets behave and how an LSTM processes information.

### 1. The Global Fixed Threshold
**Concept**: A rigid boundary (e.g., ±0.6%) applied universally to all stocks across all timeframes.

**Pros:**
- Computationally trivial.
- Maintains absolute percentage equivalence (a 0.6% move in AAPL is mathematically identical to a 0.6% move in TSLA in terms of portfolio compounding).

**Cons (Why it fails in ML):**
- **Heteroskedasticity Ignored**: Financial returns are heteroskedastic (their variance changes over time). A 0.6% move during a VIX=12 bull market is a massive outlier. A 0.6% move during a VIX=40 crash is statistical noise. A fixed threshold forces the LSTM to treat anomalies and noise as the exact same target class depending on the year.
- **Beta Bias Injection**: A fixed threshold heavily biases the target distribution. The LSTM will trivially learn to predict "Neutral" for low-beta stocks (like Utilities) and "Up/Down" for high-beta stocks (like Tech), rather than learning actual directional edge. It learns identity (the stock's inherent volatility) instead of alpha (predictive direction).

### 2. The Dynamic Volatility Threshold
**Concept**: A flexible boundary scaled to the asset's recent localized volatility (e.g., $0.35 \times \sigma_{20}$).

**Pros (Why it succeeds in ML):**
- **Stationarity & Normalization**: LSTMs require stationary, normalized data to learn mapping functions. By dividing the return by its local standard deviation (effectively creating a local Z-score cutoff), the target classes are completely normalized. An "Up" day represents an equivalent standard-deviation shock (+0.35$\sigma$) for any stock in any year.
- **Regime Agnosticism**: In a high-volatility regime, the absolute threshold widens; the model isn't penalized by random noise triggering false "Up/Down" classifications. In a low-volatility regime, the threshold narrows; the model isn't starved of "Up/Down" examples, preventing catastrophic forgetting of directional patterns.

**Cons:**
- Increases target complexity during inference (requires projecting the abstract volatility percentage back into an absolute price point for the trader).

---

### Conclusion
From a strictly mathematical and machine learning perspective, the **Dynamic Volatility Threshold is objectively superior**.

Financial markets are non-stationary and heteroskedastic. A global fixed threshold violates the core ML principle of target normalization, turning the LSTM into a flawed beta-classifier that merely memorizes which stocks are naturally volatile. 

By using the **Dynamic Volatility Threshold (Multiplier: 0.35)**, we force the LSTM to solve the much harder—and much more valuable—problem: predicting when a stock will experience a directional move that is statistically significant *relative to its own current behavior*. This is the definition of true alpha in quantitative finance.

## Usage Guide
When preparing data for training/validation, apply the optimized threshold as follows:

```python
# Assuming optimal strategy is Dynamic Volatility with multiplier M
df['returns'] = df['Close'].pct_change()
df['volatility_20d'] = df['returns'].rolling(20).std()
df['dynamic_threshold'] = df['volatility_20d'] * M

df['return_next'] = df['returns'].shift(-1)

# Categorize
conditions = [
    df['return_next'] < -df['dynamic_threshold'],
    df['return_next'] > df['dynamic_threshold']
]
choices = [0, 2]  # Down=0, Up=2
df['trend'] = np.select(conditions, choices, default=1)  # Neutral=1
```
