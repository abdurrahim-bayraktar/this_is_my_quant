"""
Evaluation metrics for stock trend prediction.

Provides standardized metric computation used across all experiments.

Usage:
    from src.evaluation.metrics import compute_metrics, evaluate_model_on_data
    
    metrics = compute_metrics(y_true, y_pred)
    # {'accuracy': 0.445, 'zero_rule': 0.354, 'lift': 0.091, 'f1_macro': 0.39, 'mcc': 0.17}
"""

import logging
from typing import Dict
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import f1_score, matthews_corrcoef

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute standard classification metrics.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        
    Returns:
        Dict with: accuracy, zero_rule, lift, f1_macro, f1_weighted, mcc
    """
    class_counts = Counter(y_true)
    most_common_count = class_counts.most_common(1)[0][1]
    zero_rule = most_common_count / len(y_true)
    
    accuracy = (y_pred == y_true).mean()
    
    return {
        "accuracy": float(accuracy),
        "zero_rule": float(zero_rule),
        "lift": float(accuracy - zero_rule),
        "f1_macro": float(f1_score(y_true, y_pred, average='macro')),
        "f1_weighted": float(f1_score(y_true, y_pred, average='weighted')),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "n_samples": int(len(y_true)),
        "class_distribution": {int(k): int(v) for k, v in class_counts.items()},
    }


def evaluate_model_on_data(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device = None,
    batch_size: int = 2048,
) -> Dict[str, object]:
    """
    Run model predictions on data and compute metrics.
    
    Args:
        model: PyTorch model with .predict() method.
        X: Input array of shape (n_samples, seq_len, n_features).
        y: Ground truth labels of shape (n_samples,).
        device: Device to run on.
        batch_size: Batch size for prediction.
        
    Returns:
        Dict with metrics + 'predictions' array.
    """
    if len(X) == 0:
        return {"error": "No data", "n_samples": 0}
    
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    all_preds = []
    
    for i in range(0, len(X), batch_size):
        batch = X[i:i + batch_size]
        with torch.no_grad():
            X_tensor = torch.FloatTensor(batch).to(device)
            predictions = model.predict(X_tensor)
            all_preds.append(predictions["trend_class"].cpu().numpy())
    
    pred_classes = np.concatenate(all_preds)
    metrics = compute_metrics(y, pred_classes)
    metrics["predictions"] = pred_classes
    
    return metrics
