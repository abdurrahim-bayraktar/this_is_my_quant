import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)

def plot_portfolio_composition(holdings_df: pd.DataFrame, strategy_name: str, output_path: str, top_n_tickers: int = 15):
    """
    Creates a stacked area chart showing the portfolio allocation to different tickers over time.
    Colors top N tickers individually and groups the rest into 'Other'.
    """
    if holdings_df.empty:
        logger.warning(f"No holdings data to plot for {strategy_name}")
        return
        
    try:
        # Pivot the dataframe to get dates as index and tickers as columns
        # Fill missing with 0 (since weight is 0 if not held)
        pivot_df = holdings_df.pivot(index='Date', columns='Ticker', values='Weight').fillna(0)
        
        # Ensure index is datetime for proper plotting
        pivot_df.index = pd.to_datetime(pivot_df.index)
        
        # Find the most systematically held tickers to assign distinct colors
        # We can rank them by their average absolute weight over time
        avg_weights = pivot_df.abs().mean().sort_values(ascending=False)
        top_tickers = avg_weights.head(top_n_tickers).index.tolist()
        
        # Group remaining into 'Other'
        other_tickers = [c for c in pivot_df.columns if c not in top_tickers]
        
        plot_df = pivot_df[top_tickers].copy()
        if other_tickers:
            # For 'Other', we need to be careful with Long/Short (negative weights).
            # We'll sum positive and negative separately to avoid them canceling out in stacked plots
            pos_others = pivot_df[other_tickers].clip(lower=0).sum(axis=1)
            neg_others = pivot_df[other_tickers].clip(upper=0).sum(axis=1)
            
            if pos_others.abs().sum() > 1e-4:
                plot_df['Other (Long)'] = pos_others
            if neg_others.abs().sum() > 1e-4:
                plot_df['Other (Short)'] = neg_others

        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Separate long and short allocations for stacked area chart
        long_df = plot_df.clip(lower=0)
        short_df = plot_df.clip(upper=0)
        
        # Use a nice colormap
        colors = sns.color_palette("husl", len(plot_df.columns))
        
        # Stacked area plot for Longs
        if long_df.abs().sum().sum() > 0:
            ax.stackplot(long_df.index, long_df.values.T, labels=long_df.columns, colors=colors, alpha=0.8)
            
        # Stacked area plot for Shorts
        if short_df.abs().sum().sum() > 0:
            ax.stackplot(short_df.index, short_df.values.T, labels=[], colors=colors, alpha=0.8)
            
        ax.set_title(f"Portfolio Composition Over Time - {strategy_name}", fontsize=14)
        ax.set_ylabel("Portfolio Weight", fontsize=12)
        ax.set_xlabel("Date", fontsize=12)
        
        # Formatting X-axis
        fig.autofmt_xdate()
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Add legend outside the plot
        handles, labels = ax.get_legend_handles_labels()
        if handles: # ensure we don't error if empty
            ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0.)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
    except Exception as e:
        logger.error(f"Failed to plot portfolio composition for {strategy_name}: {e}")

def plot_trade_activity(trades_df: pd.DataFrame, strategy_name: str, output_path: str):
    """
    Plots the daily trading activity (number of trades and absolute turnover volume).
    """
    if trades_df.empty:
        logger.warning(f"No trades data to plot for {strategy_name}")
        return
        
    try:
        trades_df = trades_df.copy()
        trades_df['Date'] = pd.to_datetime(trades_df['Date'])
        
        # Aggregate daily metrics
        daily_activity = trades_df.groupby('Date').agg(
            num_trades=('Ticker', 'count'),
            buy_volume=('Trade_Weight', lambda x: x[x > 0].sum()),
            sell_volume=('Trade_Weight', lambda x: x[x < 0].abs().sum()),
            total_turnover=('Trade_Weight', lambda x: x.abs().sum() / 2.0)
        )
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Plot 1: Buy/Sell Volume (Stacked)
        ax1.bar(daily_activity.index, daily_activity['buy_volume'], color='green', alpha=0.7, label='Buy Weight', width=1)
        ax1.bar(daily_activity.index, -daily_activity['sell_volume'], color='red', alpha=0.7, label='Sell Weight', width=1)
        
        ax1.set_title(f"Daily Trade Weight Allocation - {strategy_name}", fontsize=14)
        ax1.set_ylabel("Traded Weight", fontsize=12)
        ax1.axhline(0, color='black', linewidth=1)
        ax1.legend(loc='upper left')
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # Plot 2: Number of Distinct Tickers Traded
        ax2.plot(daily_activity.index, daily_activity['num_trades'], color='blue', linewidth=1.5, marker='o', markersize=3, alpha=0.8)
        ax2.fill_between(daily_activity.index, 0, daily_activity['num_trades'], color='blue', alpha=0.1)
        
        ax2.set_title("Number of Tickers Traded per Day", fontsize=14)
        ax2.set_ylabel("Ticker Count", fontsize=12)
        ax2.set_xlabel("Date", fontsize=12)
        ax2.set_ylim(bottom=0)
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        
    except Exception as e:
        logger.error(f"Failed to plot trade activity for {strategy_name}: {e}")
