"""
MSFT 2-Year Prediction Visualization.

Generates a graph showing model predictions overlaid on MSFT price data.
Uses the trained V7 model with domain knowledge features.
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
from src.features.indicators_v7 import ComprehensiveIndicatorsV7  # V7 indicators with domain features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_best_model():
    """Load the best V7 model and its feature list."""
    # 1. Find the latest V7 report directory for features
    report_dirs = sorted(REPORTS_DIR.glob("pooled_v7_*"), reverse=True)
    features_path = None
    feature_names = []
    
    for report_dir in report_dirs:
        # Try features_v7.txt first, then features.txt
        f7 = report_dir / "features_v7.txt"
        f = report_dir / "features.txt"
        
        if f7.exists():
            features_path = f7
            break
        elif f.exists():
            features_path = f
            break
            
    if not features_path:
        raise FileNotFoundError("No features.txt or features_v7.txt found in any pooled_v7_* report directory!")
        
    logger.info(f"Loading features from {features_path}")
    with open(features_path, 'r') as f:
        feature_names = [line.strip() for line in f if line.strip()]
    
    # 2. Find the latest model in MODELS_DIR
    # User specified "latest model under models folder"
    # We look for YYYYMMDD_HHMMSS directories in models/
    model_dirs = sorted([d for d in MODELS_DIR.iterdir() if d.is_dir() and d.name[0].isdigit()], reverse=True)
    
    model = None
    for model_dir in model_dirs:
        # Check for best_model.pt
        best_model_path = model_dir / "best_model.pt"
        if best_model_path.exists():
            logger.info(f"Loading model from {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)
            
            # Get input size from checkpoint
            state_dict = checkpoint['model_state_dict']
            input_size = state_dict['lstm.weight_ih_l0'].shape[1]
            
            # Verify input size matches feature count
            if input_size != len(feature_names):
                logger.warning(f"Mismatch: Model expects {input_size} features, but found {len(feature_names)} in features file.")
                logger.warning("Continuing, but this may cause shape errors if not handled.")
            
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
            
            logger.info(f"Model loaded with {input_size} input features")
            return model, feature_names
            
    raise FileNotFoundError("No best_model.pt found in any recent models directory!")


def prepare_msft_data(start_date: str, end_date: str, feature_names: list, sequence_length: int = 20):
    """Download and prepare MSFT data with V7 features."""
    logger.info(f"Downloading MSFT data from {start_date} to {end_date}")
    
    df = yf.download("MSFT", start=start_date, end=end_date, progress=False)
    if df.empty:
        raise ValueError("No data downloaded for MSFT")
        
    df.columns = [c.capitalize() if isinstance(c, str) else c[0].capitalize() for c in df.columns]
    
    # Compute V7 technical indicators (includes domain knowledge features)
    indicators = ComprehensiveIndicatorsV7()
    df = indicators.compute_all(df)
    
    # Verify all features exist
    missing_cols = [col for col in feature_names if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing {len(missing_cols)} features: {missing_cols[:5]}...")
        # Fill missing with 0 to prevent crash, though optimal is to fix computation
        for col in missing_cols:
            df[col] = 0.0
            
    # Select exactly the features expected by the model, in order
    feature_data = df[feature_names].values
    
    # Actual returns for comparison
    df['actual_return'] = df['Close'].pct_change().shift(-1)
    df['actual_trend'] = pd.cut(
        df['actual_return'],
        bins=[-np.inf, -0.005, 0.005, np.inf],
        labels=[0, 1, 2]
    ).astype(float)
    
    # We need to align the feature_data with the targets after dropna
    # The original code dropped na then got features. 
    # Here we need to be careful.
    
    # Create a clean dataframe for sequences
    # We only drop rows if the *target* or *price* is missing, or if we have critical missingness
    # But for features, we follow the training script's approach: 0-fill NaNs.
    
    # 1. Ensure we have valid targets and prices first
    # Drop rows where we can't calculate return/trend (usually the last row)
    df = df.dropna(subset=['actual_trend', 'Close'])
    
    feature_data = df[feature_names].values
    
    # 2. MATCH TRAINING LOGIC: Replace NaNs/Infs with 0.0 BEFORE scaling
    # This matches pooled_price_baseline_v7.py lines 370-373
    feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 3. Scale
    scaler = StandardScaler()
    feature_data = scaler.fit_transform(feature_data)
    
    # 4. MATCH TRAINING LOGIC: Replace NaNs/Infs with 0.0 AFTER scaling
    # (Scaling might introduce NaNs if a column has variance 0)
    feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Create sequences
    dates = []
    prices = []
    actual_trends = []
    sequences = []
    
    # Use the index from the processed df
    valid_dates = df.index
    
    for i in range(len(feature_data) - sequence_length):
        sequences.append(feature_data[i:i + sequence_length])
        dates.append(valid_dates[i + sequence_length])
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
    
    # Load model and features
    model, feature_names = load_best_model()
    
    # Prepare data
    X, dates, prices, actual_trends = prepare_msft_data(start_date, end_date, feature_names)
    
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
