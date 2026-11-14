"""
Overfitting Sanity Check

If the model CAN'T overfit a tiny dataset, there's a bug.
If it CAN overfit, the model works but signal is weak.

Usage:
    python experiments/overfit_test.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import logging

from config import lstm_config
from src.models import BaselineLSTM, CombinedLoss
from src.training import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_synthetic_data(n_samples=200, seq_len=20, n_features=25):
    """
    Create synthetic data with a clear learnable pattern.
    
    Pattern: If mean of last 5 timesteps > 0, label = UP (2)
             If mean < 0, label = DOWN (0)
             Else label = NEUTRAL (1)
    """
    X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    
    # Create labels based on a simple pattern
    y = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        # Use mean of last 5 timesteps of first feature
        signal = X[i, -5:, 0].mean()
        if signal > 0.3:
            y[i] = 2  # UP
        elif signal < -0.3:
            y[i] = 0  # DOWN
        else:
            y[i] = 1  # NEUTRAL
    
    logger.info(f"Label distribution: Down={np.sum(y==0)}, Neutral={np.sum(y==1)}, Up={np.sum(y==2)}")
    return X, y


def test_overfit():
    """Test if model can overfit a tiny dataset."""
    
    print("=" * 60)
    print("OVERFITTING SANITY CHECK")
    print("=" * 60)
    print("If model CAN'T reach ~100% train accuracy, there's a BUG.")
    print("=" * 60)
    
    # Create tiny synthetic dataset with learnable pattern
    X, y = create_synthetic_data(n_samples=200, seq_len=20, n_features=25)
    
    # Use same data for train and val (we WANT to overfit)
    train_dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(train_dataset, batch_size=32)  # Same data!
    
    # Create model
    model = BaselineLSTM(
        input_size=25,
        hidden_size=lstm_config.hidden_size,
        num_layers=lstm_config.num_layers,
        dropout=0.0,  # No dropout for overfitting
    )
    
    logger.info(f"Model parameters: {model.get_num_parameters():,}")
    
    # Train with NO early stopping, many epochs
    loss_fn = CombinedLoss(trend_weight=1.0, confidence_weight=0.0)  # Disable confidence loss
    trainer = Trainer(model, loss_fn)
    
    # Override early stopping
    trainer.train(train_loader, val_loader, epochs=50, early_stopping_patience=100)
    
    # Final evaluation
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        predictions = model.predict(X_tensor)
    
    pred_classes = predictions['trend_class'].cpu().numpy()
    train_accuracy = (pred_classes == y).mean()
    
    print("\n" + "=" * 60)
    print("OVERFITTING TEST RESULTS")
    print("=" * 60)
    print(f"Train Accuracy: {train_accuracy:.2%}")
    
    if train_accuracy > 0.90:
        print("[PASS] Model CAN overfit - no obvious bug in architecture")
        print("   The issue is likely weak signal in real data")
    elif train_accuracy > 0.60:
        print("[WARN] Model partially overfits - possible minor issue")
    else:
        print("[FAIL] Model CANNOT overfit - THERE IS A BUG!")
        print("   Check: loss function, data loading, label encoding")
    
    print("=" * 60)
    
    return train_accuracy


if __name__ == "__main__":
    test_overfit()
