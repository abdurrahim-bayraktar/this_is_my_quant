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
        
        prev_weights = pd.Series(dtype=float)
        
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
            
            elif strategy_name == "Buy_Hold_Universe":
                selected = day_data['Ticker']
                weights[selected] = 1.0 / len(selected)
                
            elif strategy_name == "Random_Allocation":
                n = kwargs.get('n', 3)
                # Fix seed for reproducibility but vary per day
                np.random.seed(hash(str(date)) % 2**32)
                selected = np.random.choice(day_data['Ticker'], min(n, len(day_data)), replace=False)
                weights[selected] = 1.0 / len(selected)
            else:
                raise ValueError(f"Unknown strategy: {strategy_name}")
                
            # Align weights and prev_weights to compute turnover
            all_tickers = set(weights.index).union(set(prev_weights.index))
            w_curr = pd.Series({t: weights.get(t, 0.0) for t in all_tickers})
            w_prev = pd.Series({t: prev_weights.get(t, 0.0) for t in all_tickers})
            
            # Simplified turnover: sum of absolute weight changes
            turnover = (w_curr - w_prev).abs().sum() / 2.0
            turnover_list.append(turnover)
            
            # Compute portfolio return for this day
            # Reindex day_data to match weights
            day_returns = day_data.set_index('Ticker')['Return_Next']
            # Only count returns where we have weights
            valid_tickers = [t for t in weights.index[weights != 0] if t in day_returns.index]
            
            w_norm = weights[valid_tickers]
            r_valid = day_returns[valid_tickers]
            
            # Portfolio return is sum of (weight * return) minus transaction costs
            port_ret = (w_norm * r_valid).sum() - (turnover * self.transaction_cost)
            port_returns.append(port_ret)
            
            # Update prev_weights for next step
            # Actually next day weights will drift based on return, but for daily rebalancing we 
            # assume we rebalance completely.
            prev_weights = weights
        
        # Build history dataframe
        history = pd.DataFrame({
            "Date": dates,
            "Portfolio_Return": port_returns,
            "Turnover": turnover_list
        })
        history.set_index("Date", inplace=True)
        history["Equity"] = self.initial_capital * (1 + history["Portfolio_Return"]).cumprod()
        
        # Filter 0s to avoid div/0 in metrics
        metrics = self._compute_metrics(history["Portfolio_Return"])
        
        total_turnover = np.sum(turnover_list)
        metrics["Total Turnover"] = total_turnover
        metrics["Total Tx Cost (%)"] = total_turnover * self.transaction_cost
        
        return metrics, history
