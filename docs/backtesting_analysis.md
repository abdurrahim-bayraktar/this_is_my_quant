# Backtesting Analysis: Why Long_Top_N Returns Are Unrealistic

## Summary

After a thorough code review of [evaluate_ultimate_model.py](file:///c:/dev/this_is_my_quant/backtesting/evaluate_ultimate_model.py), [framework.py](file:///c:/dev/this_is_my_quant/backtesting/framework.py), [ultimate_model.py](file:///c:/dev/this_is_my_quant/experiments/ultimate_model.py), and [utils.py](file:///c:/dev/this_is_my_quant/src/data/utils.py):

**The return calculation, date alignment, scaler reconstruction, and label indexing are all technically correct.** There is no single smoking-gun bug. However, the results are inflated by **experimental design flaws** that make the backtest unreliable as a measure of real-world performance.

---

## ✅ What Was Verified (No Bugs Found)

| Check | Result |
|-------|--------|
| `return_next = Close.pct_change().shift(-1)` | ✅ Correctly computes forward return from T → T+1 |
| `ret_next` at `iloc[i + seq_length - 1]` | ✅ Matches the label position used in training |
| `trade_date` = last day of feature window | ✅ Consistent with prediction timing |
| `target_date` used for train/test filtering | ✅ Matches [create_sequences()](file:///c:/dev/this_is_my_quant/src/data/utils.py#138-191) logic in [utils.py](file:///c:/dev/this_is_my_quant/src/data/utils.py) |
| Scaler fitted only on training data | ✅ No data leakage |
| Train/test stock separation | ✅ Test stocks excluded from training pool |
| Feature indicators (SHAP_TOP20) | ✅ No leaky indicators (no ichimoku/dpo) |
| `dropna` preserving `return_next` values | ✅ Values are row-level, not positional — surviving rows retain correct forward returns |

---

## 🔴 The Real Problems (Experimental Design Flaws)

### 1. Extreme Concentration Amplification (Root Cause of 15x Returns)

`Long_Top_N[3]` picks **3 stocks out of 900** each day, equal-weighted at 33% each. This is an insanely concentrated portfolio that compounds daily:

- Even **random** stock selection from this universe yields 45% total return (Random_Allocation[5])
- With 900 stocks, the model's "top 3 by Prob_Up" are the **most extreme tail predictions**
- Even noise in predictions creates a selection effect: you're systematically picking the stocks the model is most confident about, and at 53-54% accuracy, the lucky hits on volatile stocks compound dramatically

The math: with 172% annualized volatility (10.8% daily σ), a daily return of ~0.73% compounds to 482% annualized. This is mathematically plausible for a hyper-concentrated, hyper-volatile strategy — but it doesn't represent real alpha.

### 2. Universe Contains Leveraged ETFs and Non-Stock Instruments

The [ranked_tickers.py](file:///c:/dev/this_is_my_quant/experiments/ranked_tickers.py) list includes:
- **Leveraged ETFs**: `TQQQ` (3x QQQ), `SQQQ` (-3x QQQ), `NVDL` (2x NVDA), `TSLL` (2x TSLA), `NVDU`, `AMZU`, etc.
- **Inverse ETFs**: `SARK`, `SQQQ`, `TSLQ`
- **Warrants**: Many tickers ending in `W` (e.g., `CRMLW`, `HTZWW`, etc.)
- **SPACs and shells**: Various blank-check companies

> [!CAUTION]
> When the model's top prediction happens to be a 3x leveraged ETF, a 2% move in the underlying becomes a 6% return — contributing 2% to your portfolio on a 33% weight. This dramatically inflates concentrated strategy returns without any real stock-picking skill.

### 3. Survivorship Bias in Static Ticker List

The `EXTENDED_TICKERS` list is a static, hand-curated list. Any stock in this list that went to zero, was delisted, or suffered catastrophic losses during the backtest period (2023-2025) may have been excluded when the list was compiled. This creates a universe with an upward return bias baked in.

Evidence: the **Buy & Hold** of this universe returns 61% over ~1.5 years (≈35% annualized), which is significantly above the S&P 500's long-term average of ~10%.

### 4. Closing Price Execution Assumption

The backtester assumes you can observe day T's closing price, compute predictions, and execute trades **at that same closing price**. In reality, you'd trade at T+1's open, losing the overnight move. While this is common in academic backtests, it creates a systematic upward bias, especially for volatile stocks where the gap between close and next-day open can be significant.

### 5. No Slippage or Market Impact Modeling

With only 5 bps transaction costs and no slippage, the model freely rotates between 900 stocks daily. Real-world execution of concentrated positions in micro-cap and low-liquidity stocks would incur:
- Significant bid-ask spread costs
- Market impact on entry/exit
- Inability to execute at stated prices for illiquid warrants/SPACs

---

## 📊 Diagnostic Evidence

| Metric | Long_Top_N[3] | Random[5] | Buy_Hold | Interpretation |
|--------|--------------|-----------|----------|----------------|
| Total Return | 1502% | 45% | 61% | 33x random — impossible without a bug or design flaw |
| Ann. Vol | 172% | 24.5% | 16.5% | 10x higher vol = extreme concentration, not skill |
| Win Rate | 53.9% | 54.4% | 57.2% | **Worse than random allocation and buy-hold** |
| Sharpe | 2.80 | 1.09 | 2.15 | High only because return is high; risk-adjusted it's not extraordinary |
| Max DD | -36.9% | -24.7% | -12.8% | Deeper drawdown = higher risk, not better strategy |

> [!IMPORTANT]
> The **win rate** tells the real story: at 53.9%, the model predicts daily direction *worse* than buy-and-hold (57.2%) and essentially the same as random (54.4%). The 15x return is driven entirely by concentration × compounding × volatile universe, not predictive skill.

---

## 🛠 Recommended Fixes

1. **Remove leveraged ETFs, inverse ETFs, and warrants** from [ranked_tickers.py](file:///c:/dev/this_is_my_quant/experiments/ranked_tickers.py) — keep only common stocks
2. **Add a "Random_Top_N" benchmark** that picks 3 random stocks each day to directly compare against `Long_Top_N[3]` with perfectly matched concentration
3. **Use open-to-open returns** instead of close-to-close for more realistic execution timing
4. **Add slippage modeling** (e.g., 10-20 bps for mid-caps, 50+ bps for micro-caps)
5. **Cap position concentration** — instead of 3 out of 900, use Long_Top_Pct (top 5-10% of universe)
6. **Verify with a shuffled-prediction test**: randomize the `Prob_Up` assignments while keeping the same universe and rebalancing frequency to measure how much return comes from concentration alone
