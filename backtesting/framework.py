import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class Backtester:
    """
    A vector-based backtesting engine.
    Given a DataFrame of predictions and forward returns per ticker per day,
    simulates trading equity over time.
    """
    def __init__(self, initial_capital: float = 100000.0, transaction_cost_bps: float = 5.0):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost_bps / 10000.0
        
    def _compute_metrics(self, portfolio_returns: pd.Series) -> dict:
        """Compute standard performance metrics."""
        cum_ret = (1 + portfolio_returns).prod() - 1
        
        # Annualization factor: Use exact calendar days to handle skipped/dropped days gracefully
        if len(portfolio_returns) > 1:
            calendar_days = (portfolio_returns.index[-1] - portfolio_returns.index[0]).days
            years = calendar_days / 365.25 if calendar_days > 0 else 0
        else:
            years = 0
            
        ann_ret = (1 + cum_ret) ** (1 / years) - 1 if years > 0 else 0
        
        ann_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        
        # Sortino (downside deviation)
        downside = portfolio_returns[portfolio_returns < 0]
        downside_std = downside.std() * np.sqrt(252)
        sortino = ann_ret / downside_std if downside_std > 0 else 0
        
        # Max Drawdown
        cum_idx = (1 + portfolio_returns).cumprod()
        running_max = cum_idx.cummax()
        drawdown = (cum_idx - running_max) / running_max
        mdd = drawdown.min()
        
        win_rate = (portfolio_returns > 0).mean()
        
        return {
            "Total Return": cum_ret,
            "Annualized Return": ann_ret,
            "Annualized Volatility": ann_vol,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Max Drawdown": mdd,
            "Daily Win Rate": win_rate,
        }

    def run_strategy(self, data: pd.DataFrame, strategy_name: str, **kwargs) -> tuple:
        """
        data: pd.DataFrame containing: ['Date', 'Ticker', 'Prob_Up', 'Prob_Down', 'Return_Next']
        Returns a tuple of (metrics_dict, portfolio_history_df).
        """
        df = data.copy()
        df = df.sort_values(['Date', 'Ticker'])
        dates = df['Date'].unique()
        
        df.set_index('Date', inplace=True)
        
        port_returns = []
        turnover_list = []
        
        holdings = pd.Series(dtype=float)
        
        # New tracking for holdings and trades
        daily_holdings = []
        daily_trades = []
        
        for date in dates:
            day_data = df.loc[[date]] if isinstance(df.loc[date], pd.DataFrame) else df.loc[[date]].to_frame().T
            
            # Determine target weights based on strategy
            weights = pd.Series(0.0, index=day_data['Ticker'].values)
            
            if strategy_name == "Long_Top_N":
                n = kwargs.get('n', 3)
                top_tickers = day_data.nlargest(n, 'Prob_Up')['Ticker']
                if not top_tickers.empty:
                    weights[top_tickers] = 1.0 / len(top_tickers)
                    
            elif strategy_name == "Long_Short_Neutral":
                n = kwargs.get('n', 3)
                # Buy top N Up
                top_up = day_data.nlargest(n, 'Prob_Up')['Ticker']
                # Short top N Down
                top_down = day_data.nlargest(n, 'Prob_Down')['Ticker']
                
                # Make sure no overlap (in case of weird probs)
                overlap = set(top_up).intersection(set(top_down))
                top_up = top_up[~top_up.isin(overlap)]
                top_down = top_down[~top_down.isin(overlap)]
                
                if len(top_up) > 0 and len(top_down) > 0:
                    weights[top_up] = 0.5 / len(top_up)
                    weights[top_down] = -0.5 / len(top_down)
                    
            elif strategy_name == "Threshold_Long":
                threshold = kwargs.get('threshold', 0.60)
                selected = day_data[day_data['Prob_Up'] > threshold]['Ticker']
                if len(selected) > 0:
                    weights[selected] = 1.0 / len(selected)
            
            elif strategy_name == "Daily_Rebalanced_Universe":
                selected = day_data['Ticker']
                weights[selected] = 1.0 / len(selected)
                
            elif strategy_name == "Buy_Hold_Universe":
                if holdings.empty:
                    # Initialize equal weights on the very first day
                    selected = day_data['Ticker']
                    weights[selected] = 1.0 / len(selected)
                else:
                    # Let weights drift naturally; target exactly what we already hold
                    weights = holdings.copy()
                
            elif strategy_name == "Random_Allocation":
                n = kwargs.get('n', 3)
                # Fix seed for reproducibility but vary per day
                np.random.seed(hash(str(date)) % 2**32)
                selected = np.random.choice(day_data['Ticker'], min(n, len(day_data)), replace=False)
                weights[selected] = 1.0 / len(selected)

            # === REGRESSION STRATEGIES (use 'Pred_Return' column) ===
            # These strategies rank stocks by predicted continuous return.
            # long_pct / short_pct control what % of the universe to trade (default 20%).

            elif strategy_name == "Regression_Long_Top_Pct":
                # Long the top X% of stocks by predicted return
                pct = kwargs.get('long_pct', 0.20)
                n = max(1, int(len(day_data) * pct))
                top = day_data.nlargest(n, 'Pred_Return')['Ticker']
                if not top.empty:
                    weights[top] = 1.0 / len(top)

            elif strategy_name == "Regression_Long_Short":
                # Long top X%, short bottom X% — market neutral
                long_pct = kwargs.get('long_pct', 0.20)
                short_pct = kwargs.get('short_pct', 0.20)
                n_long = max(1, int(len(day_data) * long_pct))
                n_short = max(1, int(len(day_data) * short_pct))
                top = day_data.nlargest(n_long, 'Pred_Return')['Ticker']
                bottom = day_data.nsmallest(n_short, 'Pred_Return')['Ticker']
                # Remove any overlap
                overlap = set(top).intersection(set(bottom))
                top = top[~top.isin(overlap)]
                bottom = bottom[~bottom.isin(overlap)]
                if len(top) > 0 and len(bottom) > 0:
                    weights[top] = 0.5 / len(top)
                    weights[bottom] = -0.5 / len(bottom)

            elif strategy_name == "Regression_Quantile_Spread":
                # Long top quintile, short bottom quintile (20% each)
                # This is essentially Long_Short with fixed 20/20 — used for
                # tracking the quantile spread as a portfolio return.
                n_q = max(1, len(day_data) // 5)
                top = day_data.nlargest(n_q, 'Pred_Return')['Ticker']
                bottom = day_data.nsmallest(n_q, 'Pred_Return')['Ticker']
                overlap = set(top).intersection(set(bottom))
                top = top[~top.isin(overlap)]
                bottom = bottom[~bottom.isin(overlap)]
                if len(top) > 0 and len(bottom) > 0:
                    weights[top] = 0.5 / len(top)
                    weights[bottom] = -0.5 / len(bottom)

            elif strategy_name == "Regression_Threshold_Long":
                # Long any stock whose predicted return exceeds a threshold
                threshold = kwargs.get('threshold', 0.005)  # default 0.5%
                selected = day_data[day_data['Pred_Return'] > threshold]['Ticker']
                if len(selected) > 0:
                    weights[selected] = 1.0 / len(selected)

            # === KARASH RULE-BASED STRATEGIES (use 'Karash_Score' column) ===
            # These strategies rank stocks by the composite Karash score (-100 to +100).

            elif strategy_name == "Karash_Threshold_Long":
                # Long any stock whose Karash score exceeds a threshold
                threshold = kwargs.get('threshold', 50)
                selected = day_data[day_data['Karash_Score'] > threshold]['Ticker']
                if len(selected) > 0:
                    weights[selected] = 1.0 / len(selected)

            elif strategy_name == "Karash_Long_Top_Pct":
                # Long the top X% of stocks by Karash score
                pct = kwargs.get('long_pct', 0.20)
                n = max(1, int(len(day_data) * pct))
                top = day_data.nlargest(n, 'Karash_Score')['Ticker']
                if not top.empty:
                    weights[top] = 1.0 / len(top)

            elif strategy_name == "Karash_Long_Short":
                # Long top X%, short bottom X% — market neutral
                long_pct = kwargs.get('long_pct', 0.20)
                short_pct = kwargs.get('short_pct', 0.20)
                n_long = max(1, int(len(day_data) * long_pct))
                n_short = max(1, int(len(day_data) * short_pct))
                top = day_data.nlargest(n_long, 'Karash_Score')['Ticker']
                bottom = day_data.nsmallest(n_short, 'Karash_Score')['Ticker']
                # Remove any overlap
                overlap = set(top).intersection(set(bottom))
                top = top[~top.isin(overlap)]
                bottom = bottom[~bottom.isin(overlap)]
                if len(top) > 0 and len(bottom) > 0:
                    weights[top] = 0.5 / len(top)
                    weights[bottom] = -0.5 / len(bottom)

            else:
                raise ValueError(f"Unknown strategy: {strategy_name}")
                
            # Align target weights and current holdings to compute turnover
            all_tickers = set(weights.index).union(set(holdings.index))
            w_target = pd.Series({t: weights.get(t, 0.0) for t in all_tickers})
            w_hold = pd.Series({t: holdings.get(t, 0.0) for t in all_tickers})
            
            # Record Trades
            trade_weights = w_target - w_hold
            for t, tw in trade_weights.items():
                if abs(tw) > 1e-6:
                    direction = "Buy" if tw > 0 else "Sell"
                    daily_trades.append({
                        'Date': date, 'Ticker': t, 'Trade_Weight': tw, 'Direction': direction
                    })
                    
            # Record Holdings
            for t, w in w_target.items():
                if abs(w) > 1e-6:
                    daily_holdings.append({'Date': date, 'Ticker': t, 'Weight': w})
            
            # Simplified turnover: sum of absolute weight changes to reach target
            turnover = (w_target - w_hold).abs().sum() / 2.0
            turnover_list.append(turnover)
            
            # Compute portfolio return for this day
            # Reindex day_data to match weights
            day_returns = day_data.set_index('Ticker')['Return_Next']
            # Only count returns where we have targeted weights
            valid_tickers = [t for t in weights.index[weights != 0] if t in day_returns.index]
            
            w_norm = weights[valid_tickers]
            r_valid = day_returns[valid_tickers]
            
            # Portfolio return is sum of (weight * return) minus transaction costs
            port_ret = (w_norm * r_valid).sum() - (turnover * self.transaction_cost)
            port_returns.append(port_ret)
            
            # Update holdings for next step (market drift)
            new_holdings = w_target.copy()
            for t in valid_tickers:
                new_holdings[t] *= (1 + r_valid[t])
            
            # Normalize by total portfolio equity change to handle Long/Short correctly
            equity_multiplier = 1.0 + port_ret
            if equity_multiplier > 0:
                holdings = new_holdings / equity_multiplier
            else:
                holdings = pd.Series(0.0, index=new_holdings.index) # bankrupt
        
        # Build history dataframe
        history = pd.DataFrame({
            "Date": dates,
            "Portfolio_Return": port_returns,
            "Turnover": turnover_list
        })
        history.set_index("Date", inplace=True)
        history["Equity"] = self.initial_capital * (1 + history["Portfolio_Return"]).cumprod()
        
        # Build holdings and trades dataframes
        holdings_df = pd.DataFrame(daily_holdings) if daily_holdings else pd.DataFrame(columns=['Date', 'Ticker', 'Weight'])
        trades_df = pd.DataFrame(daily_trades) if daily_trades else pd.DataFrame(columns=['Date', 'Ticker', 'Trade_Weight', 'Direction'])
        
        # Filter 0s to avoid div/0 in metrics
        metrics = self._compute_metrics(history["Portfolio_Return"])
        
        total_turnover = np.sum(turnover_list)
        metrics["Total Turnover"] = total_turnover
        metrics["Total Tx Cost (%)"] = total_turnover * self.transaction_cost
        
        return metrics, history, holdings_df, trades_df
