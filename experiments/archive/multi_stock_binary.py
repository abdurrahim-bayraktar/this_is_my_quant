"""
Multi-Stock Binary LSTM Experiment

Config:
- Binary classification (Up/Down)
- 5 tech stocks (MSFT, AAPL, GOOGL, AMZN, META)
- 10 years of data (2015-2025)
- Best 12 indicators selected by Random Forest
- LSTM only (64 hidden, 2 layers)

This script follows the established pattern:
1. Load data per stock
2. Compute indicators per stock
3. Concatenate all data
4. Split features/metadata cleanly
5. Feature selection on training split
6. Normalize features using training statistics
7. Create sequences per ticker
8. Train/Evaluate
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from src.models import BaselineLSTM
from src.training import Trainer
from config import training_config

def compute_indicators(g):
    """Compute indicators for a single stock group."""
    g = g.copy()
    
    # Returns
    g['return_1d'] = g['Close'].pct_change()
    g['return_3d'] = g['Close'].pct_change(3)
    g['return_5d'] = g['Close'].pct_change(5)
    g['return_10d'] = g['Close'].pct_change(10)
    g['return_20d'] = g['Close'].pct_change(20)
    
    # SMA ratios
    g['sma_5_ratio'] = g['Close'] / g['Close'].rolling(5).mean() - 1
    g['sma_20_ratio'] = g['Close'] / g['Close'].rolling(20).mean() - 1
    g['sma_50_ratio'] = g['Close'] / g['Close'].rolling(50).mean() - 1
    
    # EMA
    g['ema_12_ratio'] = g['Close'] / g['Close'].ewm(span=12).mean() - 1
    g['ema_26_ratio'] = g['Close'] / g['Close'].ewm(span=26).mean() - 1
    
    # RSI
    delta = g['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    g['rsi'] = (100 - 100 / (1 + gain / loss.replace(0, np.nan))) / 100 - 0.5
    
    # MACD
    ema12 = g['Close'].ewm(span=12).mean()
    ema26 = g['Close'].ewm(span=26).mean()
    g['macd'] = (ema12 - ema26) / g['Close']
    g['macd_signal'] = g['macd'].ewm(span=9).mean()
    g['macd_hist'] = g['macd'] - g['macd_signal']
    
    # Bollinger
    sma20 = g['Close'].rolling(20).mean()
    std20 = g['Close'].rolling(20).std()
    g['bb_pctb'] = (g['Close'] - (sma20 - 2*std20)) / (4*std20)
    g['bb_width'] = (4 * std20) / g['Close']
    
    # ATR
    high_low = g['High'] - g['High'] # Placeholder for TR calc correction? 
    # Actually TR logic:
    high_low = g['High'] - g['Low']
    high_close = (g['High'] - g['Close'].shift()).abs()
    low_close = (g['Low'] - g['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    g['atr'] = tr.rolling(14).mean() / g['Close']
    
    # Volume
    g['volume_ratio'] = g['Volume'] / g['Volume'].rolling(20).mean()
    
    # Volatility
    g['volatility'] = g['return_1d'].rolling(20).std()
    
    # Momentum
    g['momentum_10'] = g['Close'] / g['Close'].shift(10) - 1
    
    # Binary target
    g['future_return'] = g['Close'].pct_change().shift(-1)
    g['target'] = (g['future_return'] > 0).astype(int)
    
    return g

def create_sequences(data, feature_cols, seq_len=20):
    """Create sequences from DataFrame."""
    X, y = [], []
    
    # Iterate by ticker segment
    for ticker in data['ticker'].unique():
        ticker_data = data[data['ticker'] == ticker]
        features = ticker_data[feature_cols].values
        labels = ticker_data['target'].values
        
        if len(features) <= seq_len:
            continue
            
        for i in range(len(features) - seq_len):
            X.append(features[i:i+seq_len])
            y.append(labels[i+seq_len])
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def main():
    print('='*80)
    print('BINARY 5-STOCK LSTM EXPERIMENT')
    print('10 years, 12 best indicators, LSTM only')
    print('='*80)

    # 5 stocks
    stocks = ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'META']

    # Load data
    print('\nLoading data...')
    all_dfs = []
    
    for ticker in stocks:
        try:
            df = yf.Ticker(ticker).history(start='2015-01-01', end='2025-01-01')
            df['ticker'] = ticker  # Add ticker column immediately
            
            # Compute indicators per stock independently to avoid cross-contamination
            df = compute_indicators(df)
            df = df.dropna()
            
            print(f'  {ticker}: {len(df)} days')
            all_dfs.append(df)
        except Exception as e:
            print(f"Error loading {ticker}: {e}")

    # Combine all stocks
    full_df = pd.concat(all_dfs)
    print(f'Total: {len(full_df)} samples')
    
    # Reset index to simple range index to avoid duplicate datetime index issues
    full_df = full_df.reset_index() 
    # Note: 'Date' is now a column, 'ticker' is a column. Perfect.

    # Features list
    all_features = ['return_1d', 'return_3d', 'return_5d', 'return_10d', 'return_20d',
                    'sma_5_ratio', 'sma_20_ratio', 'sma_50_ratio', 
                    'ema_12_ratio', 'ema_26_ratio',
                    'rsi', 'macd', 'macd_signal', 'macd_hist',
                    'bb_pctb', 'bb_width', 'atr', 'volume_ratio', 
                    'volatility', 'momentum_10']

    # Feature selection on training split
    # Use 'Date' column for splitting
    if 'Date' in full_df.columns:
        date_col = 'Date'
        # Ensure date is tz-naive for comparison
        full_df[date_col] = pd.to_datetime(full_df[date_col]).dt.tz_localize(None)
    else:
        print("Date column missing, using index?")
        # Should be there after reset_index if history returned DatetimeIndex
        
    print('\nSelecting best 12 features with Random Forest...')
    train_mask = full_df[date_col] < '2023-01-01'
    
    X_rf = full_df.loc[train_mask, all_features].values
    y_rf = full_df.loc[train_mask, 'target'].values

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_rf, y_rf)

    importance = pd.DataFrame({
        'feature': all_features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    best_12 = importance.head(12)['feature'].tolist()
    print('Selected features:')
    for f in best_12:
        imp = importance[importance['feature'] == f]['importance'].values[0]
        print(f'  {f}: {imp:.4f}')

    # Normalization
    print('\nNormalizing features...')
    scaler = StandardScaler()
    
    # Fit on training data only
    train_df = full_df[full_df[date_col] < '2023-01-01'].copy()
    val_df = full_df[(full_df[date_col] >= '2023-01-01') & (full_df[date_col] < '2023-07-01')].copy()
    test_df = full_df[full_df[date_col] >= '2023-07-01'].copy()
    
    print(f'Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}')
    
    scaler.fit(train_df[best_12])
    
    # Apply to all
    train_df[best_12] = scaler.transform(train_df[best_12])
    val_df[best_12] = scaler.transform(val_df[best_12])
    test_df[best_12] = scaler.transform(test_df[best_12])
    
    # Create sequences
    print('Creating sequences...')
    X_train, y_train = create_sequences(train_df, best_12)
    X_val, y_val = create_sequences(val_df, best_12)
    X_test, y_test = create_sequences(test_df, best_12)
    
    print(f'Sequences - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')
    
    n_down = (y_train == 0).sum()
    n_up = (y_train == 1).sum()
    weight_down = n_up / (n_down + n_up) * 2
    weight_up = n_down / (n_down + n_up) * 2
    
    # Cast weights to float32 explicitly to match model parameters
    class_weights = torch.tensor([weight_down, weight_up], dtype=torch.float32)
    print(f'Class weights: Down={weight_down:.2f}, Up={weight_up:.2f}')

    # Model
    model = BaselineLSTM(
        input_size=12,
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
        num_trend_classes=2,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {n_params:,}')
    print(f'Samples/params ratio: {len(X_train)/n_params:.1f}x')
    
    # Training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    trainer = Trainer(model, loss_fn, learning_rate=5e-4, weight_decay=1e-3)

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=64, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
        batch_size=64
    )

    training_config.early_stopping_patience = 20
    history = trainer.train(train_loader, val_loader, epochs=100)
    
    # Evaluate
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        outputs = model(X_test_tensor)
        y_pred = outputs[0].argmax(dim=-1).cpu().numpy()

    accuracy = (y_pred == y_test).mean()

    print('\n' + '='*80)
    print('RESULTS')
    print('='*80)
    print(f'Test Accuracy: {accuracy:.2%}')
    print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))

if __name__ == "__main__":
    main()
