"""
DIAGNOSTIC TEST: Single Feature Sanity Check

If the model achieves high accuracy with ONLY return_1d (yesterday's return),
then there's definitely data leakage. This is impossible in real markets.

Usage:
    python experiments/single_feature_diagnostic.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import logging

from config import DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test parameters
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "JPM", "WMT", "JNJ", "V"]
TRAIN_END = pd.Timestamp("2022-01-01")
VAL_END = pd.Timestamp("2023-06-01")
SEQUENCE_LENGTH = 20


class SimpleLSTM(nn.Module):
    """Minimal LSTM for diagnostic."""
    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)
    
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


def load_stock(ticker: str) -> pd.DataFrame:
    """Load stock data."""
    df = yf.download(ticker, start="2014-01-01", end="2024-12-31", progress=False)
    df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
    return df


def create_single_feature_data(df: pd.DataFrame, feature_name: str = 'return_1d'):
    """Create sequences using ONLY the specified feature."""
    
    # Create the single feature
    df = df.copy()
    df['return_1d'] = df['close'].pct_change()
    
    # Create target - SAME AS V9
    df['return_next'] = df['close'].pct_change().shift(-1)
    df['trend'] = pd.cut(
        df['return_next'],
        bins=[-np.inf, -0.005, 0.005, np.inf],
        labels=[0, 1, 2]
    ).astype(float)
    
    df = df.dropna()
    
    # Extract feature and labels
    feature_data = df[[feature_name]].values
    labels = df['trend'].values
    dates = df.index
    
    # Create sequences with temporal split
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []
    
    for i in range(len(feature_data) - SEQUENCE_LENGTH):
        target_date = dates[i + SEQUENCE_LENGTH]
        seq = feature_data[i:i + SEQUENCE_LENGTH]
        label = labels[i + SEQUENCE_LENGTH]
        
        if target_date < TRAIN_END:
            X_train.append(seq)
            y_train.append(label)
        elif target_date < VAL_END:
            X_val.append(seq)
            y_val.append(label)
        else:
            X_test.append(seq)
            y_test.append(label)
    
    return (
        np.array(X_train, dtype=np.float32) if X_train else np.array([]),
        np.array(y_train, dtype=np.int64) if y_train else np.array([]),
        np.array(X_val, dtype=np.float32) if X_val else np.array([]),
        np.array(y_val, dtype=np.int64) if y_val else np.array([]),
        np.array(X_test, dtype=np.float32) if X_test else np.array([]),
        np.array(y_test, dtype=np.int64) if y_test else np.array([]),
    )


def main():
    logger.info("=" * 60)
    logger.info("DIAGNOSTIC: Single Feature Test (return_1d only)")
    logger.info("=" * 60)
    
    # Collect data from all stocks
    all_train_X, all_train_y = [], []
    all_val_X, all_val_y = [], []
    all_test_X, all_test_y = [], []
    
    for ticker in tqdm(TICKERS, desc="Loading stocks"):
        try:
            df = load_stock(ticker)
            X_tr, y_tr, X_val, y_val, X_te, y_te = create_single_feature_data(df)
            
            if len(X_tr) > 0:
                all_train_X.append(X_tr)
                all_train_y.append(y_tr)
            if len(X_val) > 0:
                all_val_X.append(X_val)
                all_val_y.append(y_val)
            if len(X_te) > 0:
                all_test_X.append(X_te)
                all_test_y.append(y_te)
        except Exception as e:
            logger.warning(f"Failed {ticker}: {e}")
    
    X_train = np.concatenate(all_train_X)
    y_train = np.concatenate(all_train_y)
    X_val = np.concatenate(all_val_X)
    y_val = np.concatenate(all_val_y)
    X_test = np.concatenate(all_test_X)
    y_test = np.concatenate(all_test_y)
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Scale
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, 1)
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
    
    # Train simple model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleLSTM(input_size=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train)),
        batch_size=256, shuffle=True
    )
    
    # Train for just 20 epochs
    for epoch in range(20):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        test_output = model(torch.FloatTensor(X_test_scaled).to(device))
        pred = test_output.argmax(dim=1).cpu().numpy()
    
    accuracy = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='macro')
    
    # Baseline
    from collections import Counter
    baseline = Counter(y_test).most_common(1)[0][1] / len(y_test)
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC RESULTS")
    print("=" * 60)
    print(f"Feature used:         return_1d ONLY (1 feature)")
    print(f"Test Accuracy:        {accuracy:.2%}")
    print(f"Zero-Rule Baseline:   {baseline:.2%}")
    print(f"Accuracy Lift:        {accuracy - baseline:+.2%}")
    print(f"F1 (macro):           {f1:.4f}")
    print("=" * 60)
    
    if accuracy > 0.60:
        print("\n⚠️  WARNING: High accuracy with single feature suggests DATA LEAKAGE!")
    elif accuracy > baseline + 0.05:
        print("\n✓  Model shows modest improvement over baseline - might be learning something.")
    else:
        print("\n✓  Model performs near baseline - as expected with a single weak feature.")


if __name__ == "__main__":
    main()
