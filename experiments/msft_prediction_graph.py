"""
MSFT 2-Year Prediction Visualization.

Generates a graph showing model predictions overlaid on MSFT price data.
Uses the trained V5 model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from sklearn.preprocessing import StandardScaler

from config import REPORTS_DIR, MODELS_DIR
from src.models import AttentionLSTM
from src.features import ComprehensiveIndicators  # Use centralized module

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_best_model():
    """Load the best V5 model."""
    # Find latest model directory
    model_dirs = sorted(MODELS_DIR.glob("20260124_*"), reverse=True)
    
    for model_dir in model_dirs:
        best_model_path = model_dir / "best_model.pt"
        if best_model_path.exists():
            logger.info(f"Loading model from {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)
            
            # Get input size from checkpoint
            state_dict = checkpoint['model_state_dict']
            # LSTM input size from weight_ih_l0
            input_size = state_dict['lstm.weight_ih_l0'].shape[1]
            
            model = AttentionLSTM(
                input_size=input_size,
                hidden_size=128,
                num_layers=2,
                dropout=0.4,
            )
            model.load_state_dict(state_dict)
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = model.to(device)
            model.eval()
            
            return model
    
    raise FileNotFoundError("No trained model found!")


def prepare_msft_data(start_date: str, end_date: str, sequence_length: int = 20):
    """Download and prepare MSFT data."""
    logger.info(f"Downloading MSFT data from {start_date} to {end_date}")
    
    df = yf.download("MSFT", start=start_date, end=end_date, progress=False)
    df.columns = [c.capitalize() if isinstance(c, str) else c[0].capitalize() for c in df.columns]
    
    # Compute indicators
    indicators = ComprehensiveIndicators()
    df = indicators.compute_all(df)
    feature_cols = indicators.get_indicator_columns(df)
    
    # Actual returns for comparison
    df['actual_return'] = df['Close'].pct_change().shift(-1)
    df['actual_trend'] = pd.cut(
        df['actual_return'],
        bins=[-np.inf, -0.005, 0.005, np.inf],
        labels=[0, 1, 2]
    ).astype(float)
    
    df = df.dropna()
    
    # Normalize features
    scaler = StandardScaler()
    feature_data = df[feature_cols].values
    feature_data = scaler.fit_transform(feature_data)
    feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Create sequences
    dates = []
    prices = []
    actual_trends = []
    sequences = []
    
    for i in range(len(feature_data) - sequence_length):
        sequences.append(feature_data[i:i + sequence_length])
        dates.append(df.index[i + sequence_length])
        prices.append(df['Close'].iloc[i + sequence_length])
        actual_trends.append(df['actual_trend'].iloc[i + sequence_length])
    
    return (
        np.array(sequences, dtype=np.float32),
        dates,
        np.array(prices),
        np.array(actual_trends),
    )


def predict_batched(model, X, batch_size=512):
    """Batched predictions."""
    device = next(model.parameters()).device
    all_preds = []
    all_probs = []
    
    for i in range(0, len(X), batch_size):
        batch = X[i:i + batch_size]
        with torch.no_grad():
            batch_tensor = torch.FloatTensor(batch).to(device)
            predictions = model.predict(batch_tensor)
            all_preds.append(predictions["trend_class"].cpu().numpy())
            all_probs.append(predictions["trend_probs"].cpu().numpy())
    
    return np.concatenate(all_preds), np.concatenate(all_probs)


def create_prediction_graph(dates, prices, predictions, actual_trends, probs, output_path):
    """Create the 2-year prediction visualization."""
    
    # Calculate accuracy
    accuracy = (predictions == actual_trends).mean()
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), height_ratios=[3, 1, 1])
    fig.suptitle(f'MSFT Price with Model Predictions (Accuracy: {accuracy:.1%})', fontsize=14, fontweight='bold')
    
    # Color map for predictions
    colors = {0: '#ef4444', 1: '#a3a3a3', 2: '#22c55e'}  # Red, Gray, Green
    labels = {0: 'Down', 1: 'Neutral', 2: 'Up'}
    
    # Top plot: Price chart with prediction markers
    ax1 = axes[0]
    ax1.plot(dates, prices, color='#3b82f6', linewidth=1.5, label='MSFT Price', alpha=0.8)
    
    # Add prediction markers (sample every 5 days to avoid clutter)
    for i in range(0, len(dates), 5):
        pred = predictions[i]
        is_correct = predictions[i] == actual_trends[i]
        marker = 'o' if is_correct else 'x'
        ax1.scatter(dates[i], prices[i], c=colors[pred], s=30, marker=marker, alpha=0.7, zorder=5)
    
    ax1.set_ylabel('Price ($)', fontsize=11)
    ax1.set_title('Price Chart with Model Predictions (○ = Correct, × = Wrong)', fontsize=11)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    # Middle plot: Prediction vs Actual
    ax2 = axes[1]
    
    # Create rolling accuracy (30-day window)
    window = 30
    rolling_correct = pd.Series([1 if predictions[i] == actual_trends[i] else 0 for i in range(len(predictions))])
    rolling_acc = rolling_correct.rolling(window=window, min_periods=1).mean()
    
    ax2.plot(dates, rolling_acc, color='#8b5cf6', linewidth=2, label=f'{window}-day Rolling Accuracy')
    ax2.axhline(y=0.333, color='#ef4444', linestyle='--', alpha=0.7, label='Random Chance (33%)')
    ax2.axhline(y=accuracy, color='#22c55e', linestyle='--', alpha=0.7, label=f'Overall Accuracy ({accuracy:.1%})')
    ax2.fill_between(dates, 0.333, rolling_acc, where=rolling_acc > 0.333, alpha=0.3, color='#22c55e')
    ax2.fill_between(dates, 0.333, rolling_acc, where=rolling_acc <= 0.333, alpha=0.3, color='#ef4444')
    
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.set_ylim(0, 0.8)
    ax2.set_title('Rolling Prediction Accuracy', fontsize=11)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    # Bottom plot: Prediction distribution
    ax3 = axes[2]
    
    pred_series = pd.Series(predictions, index=dates)
    for cls, color in colors.items():
        mask = pred_series == cls
        ax3.bar(pred_series.index[mask], [1]*mask.sum(), color=color, alpha=0.7, width=2, label=labels[cls])
    
    ax3.set_ylabel('Prediction', fontsize=11)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_title('Daily Predictions (Red=Down, Gray=Neutral, Green=Up)', fontsize=11)
    ax3.legend(loc='upper left', fontsize=9)
    ax3.set_yticks([])
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Saved prediction graph to {output_path}")
    
    # Print summary stats
    print("\n" + "="*60)
    print("MSFT PREDICTION SUMMARY")
    print("="*60)
    print(f"Date Range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    print(f"Total Trading Days: {len(dates)}")
    print(f"Overall Accuracy: {accuracy:.2%}")
    print(f"Random Chance: 33.33%")
    print(f"Improvement over Random: +{(accuracy - 0.333)*100:.1f}pp")
    print()
    print("Prediction Distribution:")
    for cls, label in labels.items():
        count = (predictions == cls).sum()
        pct = count / len(predictions) * 100
        correct = ((predictions == cls) & (actual_trends == cls)).sum()
        cls_acc = correct / count if count > 0 else 0
        print(f"  {label:8s}: {count:4d} ({pct:5.1f}%) - Accuracy: {cls_acc:.1%}")
    print("="*60)


def main():
    # 2-year date range
    start_date = "2023-01-01"
    end_date = "2024-12-31"
    
    # Load model
    model = load_best_model()
    
    # Prepare data
    X, dates, prices, actual_trends = prepare_msft_data(start_date, end_date)
    
    # Make predictions
    predictions, probs = predict_batched(model, X)
    
    # Create output directory
    output_dir = REPORTS_DIR / f"msft_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate graph
    output_path = output_dir / "msft_2year_predictions.png"
    create_prediction_graph(dates, prices, predictions, actual_trends, probs, output_path)
    
    print(f"\nGraph saved to: {output_path}")


if __name__ == "__main__":
    main()
