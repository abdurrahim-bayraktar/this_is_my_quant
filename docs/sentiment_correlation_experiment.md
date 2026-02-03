# Sentiment Correlation Experiment Report

## 1. Objective
The goal of this experiment was to quantify the correlation between **FinBERT-extracted sentiment scores** from financial news and **stock price returns** for the most covered stocks in the FNSPID dataset. Specifically, we aimed to determine:
1.  Does news sentiment correlate with price movements on the **same day**?
2.  Does news sentiment have predictive power for the **next day**'s price movements?

## 2. Methodology

### Data Source
-   **News**: FNSPID dataset (Top 40 most covered tickers).
-   **Sentiment Model**: FinBERT (ProsusAI), utilizing GPU acceleration.
-   **Price Data**: Daily Close prices via `yfinance`.

### Processing Pipeline
1.  **Sentiment Extraction**: Used `SentimentExtractor` (FinBERT) to assign a sentiment score (-1 to 1) and label (positive, negative, neutral) to each headline.
2.  **Filtering**: Removed all news items labeled as **"neutral"** to avoid signal dilution.
3.  **Aggregation**: Aggregated sentiment to a daily level using a simple mean (`SentimentAggregator`).
4.  **Alignment**:
    -   **Same-Day Return**: $R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$ aligned with Sentiment on Day $t$.
    -   **Next-Day Return**: $R_{t+1} = \frac{P_{t+1} - P_t}{P_t}$ aligned with Sentiment on Day $t$.
    -   **Filtering**: Used an **Inner Join** matching strategy. Days with **no news** (or only neutral news) were strictly excluded from the analysis.

## 3. Results

### Top Performers (Same-Day Correlation)
Stocks like **MSFT** showed significant positive correlation on the same day, indicating valid sentiment signal capture.

| Ticker | Same-Day Correlation | Next-Day Correlation | Observations (Days) | News Count |
| :--- | :--- | :--- | :--- | :--- |
| **MSFT** | **0.347** | -0.003 | 406 | 3367 |
| **DIS** | 0.087 | -0.026 | 931 | 2881 |
| **WMT** | 0.079 | -0.040 | 999 | 3096 |

*Note: The broader "Top 40" experiment (N=32) showed an average Same-Day correlation of **0.177**.*

### Visualization Analysis
Scatter plots with trend lines were generated to visualize these relationships:
-   **Same-Day Plots**: `reports/top40_scatters/`
-   **Next-Day Plots**: `reports/top40_scatters_next_day/`

**Observation**: MSFT's same-day scatter plot shows a clear positive slope, confirming that positive news generally co-occurs with positive price action.

## 4. Addressing the "Obvious Difference"
A discrepancy was noted between the plots generated in this experiment and those from a previous debugging script (`debug_distribution.py`).

**Investigation Findings**:
-   `debug_distribution.py` labeled its Y-Axis as "Next Day Return".
-   **However**, the code actually plotted **Same-Day Return** (Return at T vs Sentiment at T).
-   This explains why the "Next Day" plots in the debug script looked good (showing ~0.35 correlation for MSFT)—they were actually showing the **Same Day** correlation.
-   **This Experiment's** "Next Day" correlation is ~0.00, which accurately reflects the lack of simple linear predictability from daily aggregated sentiment to the next day's close.

## 5. Key Takeaways & Recommendations
1.  **News Moves Markets Instantly**: The strong Same-Day correlation (especially for tech stocks like MSFT) confirms that the market prices in news immediately (or intrabay).
2.  **No Simple "Alpha"**: The near-zero Next-Day correlation suggests that a simple "Buy if Sentiment > 0 yesterday" strategy will likely fail.
3.  **Future Directions**:
    -   **Intraday Analysis**: Move to hourly or minute-level aggregation to capture the reaction *before* the market close.
    -   **Momentum/Accumulation**: Instead of single-day sentiment, use a rolling window (e.g., 3-day sentiment momentum) as a feature.
    -   **Event-Based**: Focus only on "high dispersion" or "high volume" news days rather than every day with news.
