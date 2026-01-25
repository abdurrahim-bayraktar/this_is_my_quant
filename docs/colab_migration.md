# Migrating to Google Colab

This guide explains how to run the price prediction experiments on Google Colab for access to better hardware (T4/A100 GPUs, more RAM).

---

## Quick Start

### 1. Create a New Colab Notebook

Go to [Google Colab](https://colab.research.google.com/) and create a new notebook.

### 2. Setup Cell (Run First)

```python
# Install dependencies
!pip install pandas_ta yfinance torch scikit-learn tqdm

# Clone or upload the repository
# Option A: Clone from GitHub (if you have it there)
# !git clone https://github.com/YOUR_USERNAME/this_is_my_quant.git
# %cd this_is_my_quant

# Option B: Upload files manually (recommended for private repos)
# Use the file upload button in Colab to upload:
# - src/features/indicators_v7.py
# - src/models/baseline_lstm.py
# - src/training/trainer.py
# - experiments/pooled_price_baseline_v7.py

# Create directory structure
!mkdir -p /content/this_is_my_quant/src/features
!mkdir -p /content/this_is_my_quant/src/models
!mkdir -p /content/this_is_my_quant/src/training
!mkdir -p /content/this_is_my_quant/experiments
!mkdir -p /content/this_is_my_quant/data/price_cache
!mkdir -p /content/this_is_my_quant/reports
!mkdir -p /content/this_is_my_quant/models
```

### 3. Upload Core Files

Upload these files to Colab (drag and drop or use the file browser):

| Local Path | Colab Path |
|------------|------------|
| `src/features/indicators_v7.py` | `/content/this_is_my_quant/src/features/` |
| `src/models/baseline_lstm.py` | `/content/this_is_my_quant/src/models/` |
| `src/training/trainer.py` | `/content/this_is_my_quant/src/training/` |
| `config/settings.py` | `/content/this_is_my_quant/config/` |

### 4. Create `__init__.py` Files

```python
# Create __init__.py files
!touch /content/this_is_my_quant/src/__init__.py
!touch /content/this_is_my_quant/src/features/__init__.py
!touch /content/this_is_my_quant/src/models/__init__.py
!touch /content/this_is_my_quant/src/training/__init__.py
!touch /content/this_is_my_quant/config/__init__.py
```

### 5. Run the Experiment

```python
%cd /content/this_is_my_quant

import sys
sys.path.insert(0, '/content/this_is_my_quant')

# Import and run V7
from experiments.pooled_price_baseline_v7 import PooledExperimentV7, EXTENDED_TICKERS

# Run with 1000 stocks on Colab (more than local can handle)
experiment = PooledExperimentV7(
    stocks=EXTENDED_TICKERS[:1000],
    epochs=150,
    batch_size=2048,  # Larger batch for Colab GPU
    colab_mode=True,
)
experiment.run()
```

---

## Alternative: Self-Contained Colab Notebook

Copy this entire cell into a Colab notebook for a self-contained V7 experiment:

```python
#@title V7 Experiment - Self-Contained Colab Version
#@markdown Run this cell to execute the full V7 experiment

!pip install -q pandas_ta yfinance torch scikit-learn tqdm

import sys
import logging
import pickle
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Check GPU
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("No GPU available!")

# Stock list
STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "UNH", "JNJ",
    "V", "XOM", "JPM", "WMT", "PG", "MA", "HD", "CVX", "LLY", "MRK",
    # ... add more as needed
][:800]

# === MODEL ===
class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.4, num_heads=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        out = self.fc(attn_out[:, -1, :])
        return out
    
    def predict(self, x):
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        return {"trend_class": probs.argmax(dim=-1), "probs": probs}

# === INDICATORS ===
def compute_indicators(df):
    import pandas_ta as ta
    df = df.copy()
    
    # Moving Averages
    df['sma_20'] = ta.sma(df['Close'], 20)
    df['sma_50'] = ta.sma(df['Close'], 50)
    df['sma_200'] = ta.sma(df['Close'], 200)
    df['ema_12'] = ta.ema(df['Close'], 12)
    df['ema_26'] = ta.ema(df['Close'], 26)
    
    # Momentum
    df['rsi_14'] = ta.rsi(df['Close'], 14)
    macd = ta.macd(df['Close'])
    if macd is not None:
        for col in macd.columns:
            df[f'macd_{col}'] = macd[col]
    df['cci'] = ta.cci(df['High'], df['Low'], df['Close'])
    
    # Volume
    df['mfi'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'])
    df['obv'] = ta.obv(df['Close'], df['Volume'])
    df['volume_sma'] = ta.sma(df['Volume'], 20)
    df['volume_ratio'] = df['Volume'] / df['volume_sma'].replace(0, np.nan)
    
    # Volatility
    bbands = ta.bbands(df['Close'])
    if bbands is not None:
        for col in bbands.columns:
            df[f'bb_{col}'] = bbands[col]
    df['atr'] = ta.atr(df['High'], df['Low'], df['Close'])
    
    # Trend
    adx = ta.adx(df['High'], df['Low'], df['Close'])
    if adx is not None:
        for col in adx.columns:
            df[f'adx_{col}'] = adx[col]
    
    # Price features
    df['return_1d'] = df['Close'].pct_change()
    df['return_5d'] = df['Close'].pct_change(5)
    
    # Binary domain features
    df['above_sma_20'] = (df['Close'] > df['sma_20']).astype(float)
    df['rsi_overbought'] = (df['rsi_14'] > 70).astype(float)
    df['rsi_oversold'] = (df['rsi_14'] < 30).astype(float)
    
    # Cleanup
    df = df.drop(columns=['volume_sma'], errors='ignore')
    return df

# === DATA LOADING ===
def load_stocks(stocks, start='2014-01-01', end='2024-12-31'):
    logger.info(f"Downloading {len(stocks)} stocks...")
    data = yf.download(stocks, start=start, end=end, group_by='ticker', threads=True, progress=True)
    
    stock_data = {}
    for ticker in tqdm(stocks, desc="Processing"):
        try:
            if len(stocks) == 1:
                df = data.copy()
            else:
                df = data[ticker].copy()
            df = df.dropna()
            if len(df) < 100:
                continue
            df.columns = [c.capitalize() for c in df.columns]
            stock_data[ticker] = df
        except:
            pass
    
    logger.info(f"Loaded {len(stock_data)} stocks")
    return stock_data

# === MAIN ===
def run_experiment(stocks, epochs=100, batch_size=2048):
    stock_data = load_stocks(stocks)
    
    all_X, all_y = [], []
    feature_cols = None
    
    for ticker, df in tqdm(stock_data.items(), desc="Features"):
        df = compute_indicators(df)
        exclude = ['Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']
        cols = [c for c in df.columns if c not in exclude]
        
        df['return_next'] = df['Close'].pct_change().shift(-1)
        df['trend'] = pd.cut(df['return_next'], bins=[-np.inf, -0.005, 0.005, np.inf], labels=[0,1,2]).astype(float)
        df = df.dropna()
        
        if len(df) < 30:
            continue
        
        scaler = StandardScaler()
        feat = scaler.fit_transform(df[cols].values)
        feat = np.nan_to_num(feat, 0, 0, 0)
        
        seq_len = 20
        for i in range(len(feat) - seq_len):
            all_X.append(feat[i:i+seq_len])
            all_y.append(df['trend'].iloc[i+seq_len])
        
        if feature_cols is None:
            feature_cols = cols
    
    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int64)
    logger.info(f"Dataset: {len(X):,} samples, {X.shape[-1]} features")
    
    # Split
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    n = len(X)
    X_train, y_train = X[:int(0.8*n)], y[:int(0.8*n)]
    X_val, y_val = X[int(0.8*n):int(0.9*n)], y[int(0.8*n):int(0.9*n)]
    X_test, y_test = X[int(0.9*n):], y[int(0.9*n):]
    
    # Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AttentionLSTM(X.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)), batch_size=batch_size)
    
    # Train
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                out = model(X_batch)
                val_loss += criterion(out, y_batch).item()
                correct += (out.argmax(1) == y_batch).sum().item()
                total += len(y_batch)
        
        val_acc = correct / total
        val_loss /= len(val_loader)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}: val_loss={val_loss:.4f}, val_acc={val_acc:.2%}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 15:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Test
    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), 2048):
            batch = torch.FloatTensor(X_test[i:i+2048]).to(device)
            preds = model(batch).argmax(1).cpu().numpy()
            all_preds.extend(preds)
    
    accuracy = (np.array(all_preds) == y_test).mean()
    logger.info(f"\n=== RESULTS ===")
    logger.info(f"Test Accuracy: {accuracy:.2%}")
    logger.info(f"Stocks: {len(stock_data)}")
    logger.info(f"Samples: {len(X):,}")
    
    return accuracy

# Run it!
accuracy = run_experiment(STOCKS, epochs=100, batch_size=2048)
```

---

## Hardware Comparison

| Environment | GPU | VRAM | Expected Speed |
|-------------|-----|------|----------------|
| Your Local | RTX 3050 | 6GB | ~48 it/s |
| Colab Free | T4 | 16GB | ~100 it/s |
| Colab Pro | A100 | 40GB | ~300 it/s |

---

## Tips for Colab

1. **Enable GPU**: `Runtime` → `Change runtime type` → `T4 GPU`
2. **Save checkpoints**: Mount Google Drive to save models
3. **Larger batches**: Colab has more VRAM, use batch_size=2048 or 4096
4. **More stocks**: Can handle 1000+ stocks easily
5. **Disconnect warning**: Colab disconnects after ~12h idle, save frequently

### Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Save results to Drive
!cp -r /content/reports /content/drive/MyDrive/quant_results/
```
