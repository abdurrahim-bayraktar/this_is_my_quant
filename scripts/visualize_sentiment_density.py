import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import os

def main():
    cache_dir = Path(r"c:\dev\this_is_my_quant\data\cache\sentiment")
    output_dir = Path(r"c:\dev\this_is_my_quant\sentiment_graphs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = list(cache_dir.glob("*_sentiment.parquet"))
    print(f"Found {len(files)} sentiment files.")

    start_date = pd.Timestamp("2012-01-01")
    end_date = pd.Timestamp("2019-12-31")
    full_idx = pd.bdate_range(start_date, end_date)
    total_trading_days = len(full_idx)
    
    exclude_tickers = {
        "FDX", "BHI", "AET", "CHN", "SLB", "VNQ", "ESRX", "XPP", "DGAZ", "UGAZ", 
        "WFM", "ERO", "JD", "VGK", "FOXA", "YINN", "ABX", "FXP", "PGJ", "GILD", 
        "GXC", "XLY", "XLU",
        "MYL", "GME", "MON", "EWU", "TSN", "JCP", "POT", "CI", "WBA", "BRRY", 
        "SPLS", "FCX", "BTU", "GRUB", "REGN", "XLK", "AXP", "DISH", "TWC", "TJX", 
        "PBR", "DZZ", "KR", "MDT", "PANW", "CSX", "AVGO", "KSS", "SHLD", "OXY", 
        "EWJ", "ANTM", "COH", "TXN", "FCAU", "BABA", "WDC", "HAL", "EA",
        "GMCR", "DIS", "RSP", "WMT", "YHOO", "INTC", "JPM", "DOW", "SLV", "GRPN", 
        "LEN", "LLY", "VMW", "GPRO", "CMG", "HD", "LUV", "AEO", "TWX", "JWN", 
        "BRCM", "DD", "STZ", "NFLX", "BLK", "BBRY", "WDAY"
    }

    # 1. Rank tickers by density in 2012-2019
    print("Calculating density...")
    densities = []
    daily_data_dict = {}
    
    for f in tqdm(files):
        ticker = f.stem.replace("_sentiment", "")
        if ticker in exclude_tickers:
            continue
        try:
            raw = pd.read_parquet(f)
            if 'date' not in raw.columns:
                continue
                
            raw['date'] = pd.to_datetime(raw['date'], utc=True).dt.tz_localize(None)
            
            # Filter to 2012-2019
            mask = (raw['date'] >= start_date) & (raw['date'] <= end_date)
            period_data = raw[mask].copy()
            
            if len(period_data) == 0:
                continue
                
            period_data['trading_date'] = period_data['date'].dt.normalize()
            
            # Aggregate to daily
            daily = period_data.groupby('trading_date').agg(
                sent_mean=('sentiment_value', 'mean'),
                sent_count=('sentiment_value', 'count')
            )
            
            days_with_news = len(daily)
            density = days_with_news / total_trading_days
            
            densities.append({
                "ticker": ticker,
                "days_with_news": days_with_news,
                "density_pct": density * 100,
                "total_news_in_period": daily['sent_count'].sum()
            })
            daily_data_dict[ticker] = daily
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            
    density_df = pd.DataFrame(densities).sort_values('density_pct', ascending=False)
    
    # Save the ranking report
    ranking_path = output_dir / "sentiment_density_ranking_2012_2019.csv"
    density_df.to_csv(ranking_path, index=False)
    print(f"\nTop 10 Tickers by Density (2012-2019):")
    print(density_df.head(10).to_string(index=False))
    
    top_150_tickers = density_df.head(150)['ticker'].tolist()
    
    # 2. Visualize and apply mathematical filling techniques
    print("\nGenerating graphs for top 150 tickers...")
    
    for rank, ticker in enumerate(tqdm(top_150_tickers)):
        daily = daily_data_dict[ticker]
        
        # Reindex to full business days to expose gaps
        ts = daily[['sent_mean']].reindex(full_idx)
        
        # Mathematical Technique 1: Forward-fill with 0.95 decay
        # Vectorized implementation of exponential decay for missing days
        decay_factor = 0.95
        ffill_ts = ts['sent_mean'].ffill()
        is_valid = ts['sent_mean'].notna()
        groups = is_valid.cumsum()
        days_since_valid = groups.groupby(groups).cumcount()
        decay_multiplier = decay_factor ** days_since_valid
        decay_filled = ffill_ts * decay_multiplier
        
        # Mathematical Technique 2: 30-Day Rolling Cumulative Sum
        # Instead of infinite cumulative sum which drifts to infinity, we sum the last 30 days.
        # Fill NA with 0 for the rolling sum calculation
        ts_zero_filled = ts['sent_mean'].fillna(0)
        rolling_cumsum = ts_zero_filled.rolling(window=30, min_periods=1).sum()
        
        # Mathematical Technique 3: 30-Day EMA
        # A slower moving average to capture longer-term sentiment momentum
        ema_30_filled = ts['sent_mean'].ewm(span=30, min_periods=1).mean()
        
        # Plotting
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
        fig.suptitle(f"{ticker} Sentiment Analysis (Rank {rank+1}, Density: {density_df.iloc[rank]['density_pct']:.1f}%)", fontsize=16)
        
        # Plot 1: Raw Scatter
        scatter_idx = daily.index
        scatter_vals = daily['sent_mean']
        ax1.scatter(scatter_idx, scatter_vals, alpha=0.5, s=10, color='blue')
        ax1.set_title("Raw Daily Sentiment (Mean)")
        ax1.set_ylabel("Sentiment Score")
        ax1.axhline(0, color='black', linewidth=1, linestyle='--')
        
        # Plot 2: Forward-fill with 0.95 decay
        ax2.plot(full_idx, decay_filled, color='orange', linewidth=1.5)
        ax2.set_title("Forward-Fill with 0.95 Daily Decay")
        ax2.set_ylabel("Decaying Sentiment")
        ax2.axhline(0, color='black', linewidth=1, linestyle='--')
        
        # Plot 3: 30-Day Rolling Cumulative Sum
        ax3.plot(full_idx, rolling_cumsum, color='green', linewidth=1.5)
        ax3.set_title("30-Day Rolling Cumulative Sum (Bounded Conviction)")
        ax3.set_ylabel("Rolling Sum")
        ax3.axhline(0, color='black', linewidth=1, linestyle='--')
        
        # Plot 4: 30-Day EMA
        ax4.plot(full_idx, ema_30_filled, color='purple', linewidth=1.5)
        ax4.set_title("30-Day Exponential Moving Average (Slow Momentum)")
        ax4.set_ylabel("30-Day EMA")
        ax4.axhline(0, color='black', linewidth=1, linestyle='--')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        save_path = output_dir / f"{rank+1:03d}_{ticker}_sentiment.png"
        plt.savefig(save_path, dpi=100)
        plt.close(fig)

    print(f"\nSaved 150 graphs to {output_dir}")

if __name__ == "__main__":
    main()
