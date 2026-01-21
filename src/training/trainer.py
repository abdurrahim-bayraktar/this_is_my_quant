"""
Model trainer with walk-forward validation.

This module provides:
- Training loop with early stopping
- Learning rate scheduling
- Walk-forward validation for time series
- TensorBoard logging
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from tqdm import tqdm
import json
from datetime import datetime

from config import training_config, MODELS_DIR, LOGS_DIR

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for LSTM-based stock prediction models.
    
    Features:
    - Mixed precision training (FP16) for memory efficiency
    - Gradient clipping for RNN stability
    - Walk-forward validation for proper time series evaluation
    - Early stopping based on validation F1
    - Checkpoint saving and loading
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        learning_rate: float = None,
        weight_decay: float = None,
        device: str = None,
        mixed_precision: bool = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: The model to train.
            loss_fn: Loss function.
            learning_rate: Learning rate. Default from config.
            weight_decay: Weight decay for regularization.
            device: Device for training.
            mixed_precision: Use FP16 training.
        """
        self.device = device or training_config.device
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate or training_config.learning_rate
        self.weight_decay = weight_decay or training_config.weight_decay
        self.mixed_precision = mixed_precision if mixed_precision is not None else training_config.mixed_precision
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        # Setup scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6,
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.mixed_precision else None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.training_history: List[Dict] = []
        
        # Experiment tracking
        self.experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = MODELS_DIR / self.experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Trainer initialized on {self.device}")
        logger.info(f"Mixed precision: {self.mixed_precision}")
    
    def train_epoch(
        self,
        train_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader.
            
        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (X, y) in enumerate(pbar):
            X = X.to(self.device)
            y = y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass (with mixed precision if enabled)
            if self.mixed_precision:
                with autocast():
                    trend_logits, confidence, _ = self.model(X)
                    loss, loss_dict = self.loss_fn(trend_logits, confidence, y)
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    training_config.gradient_clip,
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                trend_logits, confidence, _ = self.model(X)
                loss, loss_dict = self.loss_fn(trend_logits, confidence, y)
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    training_config.gradient_clip,
                )
                
                self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * X.size(0)
            total_correct += (trend_logits.argmax(dim=-1) == y).sum().item()
            total_samples += X.size(0)
            
            pbar.set_postfix({
                "loss": loss.item(),
                "acc": loss_dict.get("accuracy", 0),
            })
        
        self.scheduler.step()
        
        metrics = {
            "train_loss": total_loss / total_samples,
            "train_accuracy": total_correct / total_samples,
        }
        
        return metrics
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader.
            
        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_confidences = []
        
        for X, y in val_loader:
            X = X.to(self.device)
            y = y.to(self.device)
            
            trend_logits, confidence_logits, _ = self.model(X)
            loss, _ = self.loss_fn(trend_logits, confidence_logits, y)
            
            total_loss += loss.item() * X.size(0)
            all_preds.extend(trend_logits.argmax(dim=-1).cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            # Apply sigmoid to get probabilities from logits
            all_confidences.extend(torch.sigmoid(confidence_logits).squeeze(-1).cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_confidences = np.array(all_confidences)
        
        # Compute metrics
        accuracy = (all_preds == all_targets).mean()
        
        # Per-class accuracy
        class_acc = {}
        for cls in [0, 1, 2]:
            mask = all_targets == cls
            if mask.sum() > 0:
                class_acc[f"class_{cls}_acc"] = (all_preds[mask] == cls).mean()
        
        # Confidence calibration
        high_conf_mask = all_confidences > 0.5
        if high_conf_mask.sum() > 0:
            high_conf_accuracy = (all_preds[high_conf_mask] == all_targets[high_conf_mask]).mean()
        else:
            high_conf_accuracy = 0.0
        
        metrics = {
            "val_loss": total_loss / len(all_preds),
            "val_accuracy": accuracy,
            "val_high_conf_accuracy": high_conf_accuracy,
            "mean_confidence": all_confidences.mean(),
            **class_acc,
        }
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = None,
        early_stopping_patience: int = None,
    ) -> Dict[str, List]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Number of epochs.
            early_stopping_patience: Patience for early stopping.
            
        Returns:
            Dictionary with training history.
        """
        epochs = epochs or training_config.epochs
        patience = early_stopping_patience or training_config.early_stopping_patience
        
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            self.current_epoch = epoch + 1
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics, "epoch": self.current_epoch}
            self.training_history.append(epoch_metrics)
            
            # Log
            logger.info(
                f"Epoch {self.current_epoch}: "
                f"train_loss={train_metrics['train_loss']:.4f}, "
                f"val_loss={val_metrics['val_loss']:.4f}, "
                f"val_acc={val_metrics['val_accuracy']:.2%}"
            )
            
            # Early stopping check
            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.patience_counter = 0
                self.save_checkpoint("best_model.pt")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {self.current_epoch}")
                    break
        
        # Save final checkpoint
        self.save_checkpoint("final_model.pt")
        
        # Save training history (convert numpy types to native Python for JSON)
        history_path = self.checkpoint_dir / "training_history.json"
        
        def convert_to_serializable(obj):
            """Convert numpy types to native Python types."""
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        with open(history_path, "w") as f:
            json.dump(convert_to_serializable(self.training_history), f, indent=2)
        
        return {"history": self.training_history}
    
    def walk_forward_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = None,
        train_ratio: float = 0.7,
    ) -> Dict[str, List]:
        """
        Walk-forward validation for time series.
        
        Maintains temporal order: train on past, validate on future.
        
        Args:
            X: Feature array (n_samples, seq_len, n_features).
            y: Target array (n_samples,).
            n_folds: Number of folds.
            train_ratio: Ratio of training data in each fold.
            
        Returns:
            Dictionary with fold results.
        """
        n_folds = n_folds or training_config.walk_forward_folds
        n_samples = len(X)
        
        fold_results = []
        
        # Calculate fold sizes
        fold_size = n_samples // n_folds
        
        logger.info(f"Walk-forward validation with {n_folds} folds")
        
        for fold in range(1, n_folds + 1):
            # Define train/val split for this fold
            val_end = fold * fold_size
            val_start = int(val_end - fold_size * (1 - train_ratio))
            train_end = val_start
            
            # Ensure minimum training data
            if train_end < fold_size:
                train_end = fold_size
                val_start = train_end
            
            logger.info(f"Fold {fold}: train=[0:{train_end}], val=[{val_start}:{val_end}]")
            
            # Create data loaders
            X_train = torch.FloatTensor(X[:train_end])
            y_train = torch.LongTensor(y[:train_end])
            X_val = torch.FloatTensor(X[val_start:val_end])
            y_val = torch.LongTensor(y[val_start:val_end])
            
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=training_config.batch_size,
                shuffle=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=training_config.batch_size,
            )
            
            # Reset model and optimizer
            self._reset_model()
            
            # Train
            self.train(train_loader, val_loader, epochs=training_config.epochs // 2)
            
            # Final validation
            val_metrics = self.validate(val_loader)
            val_metrics["fold"] = fold
            fold_results.append(val_metrics)
        
        # Aggregate results
        avg_metrics = {}
        for key in fold_results[0].keys():
            if key != "fold":
                values = [r[key] for r in fold_results]
                avg_metrics[f"{key}_mean"] = np.mean(values)
                avg_metrics[f"{key}_std"] = np.std(values)
        
        logger.info(f"Walk-forward results: {avg_metrics}")
        
        return {
            "fold_results": fold_results,
            "aggregate": avg_metrics,
        }
    
    def _reset_model(self):
        """Reset model weights and optimizer for new fold."""
        # Re-initialize model weights
        for layer in self.model.modules():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        
        # Reset optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6,
        )
        
        self.best_val_loss = float("inf")
        self.patience_counter = 0
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "training_history": self.training_history,
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.training_history = checkpoint["training_history"]
        
        logger.info(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")


def main():
    """Test trainer."""
    logging.basicConfig(level=logging.INFO)
    
    from src.models import BaselineLSTM, CombinedLoss
    
    # Create model
    model = BaselineLSTM(input_size=25)
    loss_fn = CombinedLoss()
    
    trainer = Trainer(model, loss_fn)
    
    # Create dummy data
    n_samples = 1000
    seq_len = 20
    n_features = 25
    
    X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    y = np.random.randint(0, 3, n_samples)
    
    # Create data loaders
    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Train for a few epochs
    trainer.train(train_loader, val_loader, epochs=5)


if __name__ == "__main__":
    main()
