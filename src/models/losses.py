"""
Custom loss functions for multi-task learning.

Handles:
- Combined trend classification + confidence regression
- Confidence-weighted loss (higher weight when confidence is high)
- Focal loss for imbalanced classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CombinedLoss(nn.Module):
    """
    Combined loss for trend prediction and meta-prediction.
    
    Loss = α * CrossEntropyLoss(trend) + β * BCELoss(confidence)
    
    The confidence target is computed as: 1 if prediction was correct, else 0.
    This teaches the model to predict its own accuracy.
    
    Attributes:
        trend_weight: Weight for trend classification loss.
        confidence_weight: Weight for confidence prediction loss.
        class_weights: Optional weights for imbalanced trend classes.
    """
    
    def __init__(
        self,
        trend_weight: float = 1.0,
        confidence_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.1,
    ):
        """
        Initialize combined loss.
        
        Args:
            trend_weight: Weight for trend loss.
            confidence_weight: Weight for confidence loss.
            class_weights: Optional class weights for imbalanced data.
            label_smoothing: Label smoothing for classification.
        """
        super().__init__()
        
        self.trend_weight = trend_weight
        self.confidence_weight = confidence_weight
        self.label_smoothing = label_smoothing
        
        # Trend classification loss
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
        
        # Confidence regression loss (BCE with logits for autocast safety)
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        logger.info(
            f"CombinedLoss: trend_weight={trend_weight}, "
            f"confidence_weight={confidence_weight}, "
            f"label_smoothing={label_smoothing}"
        )
    
    def forward(
        self,
        trend_logits: torch.Tensor,
        confidence: torch.Tensor,
        trend_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.
        
        Args:
            trend_logits: Predicted trend logits (batch, num_classes).
            confidence: Predicted confidence (batch, 1).
            trend_targets: True trend labels (batch,).
            
        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains
            individual loss components for logging.
            
        Note:
            confidence should be RAW LOGITS (before sigmoid) for autocast safety.
        """
        # Trend classification loss
        trend_loss = self.ce_loss(trend_logits, trend_targets)
        
        # Compute confidence target: 1 if correct, 0 otherwise
        with torch.no_grad():
            predicted_trends = trend_logits.argmax(dim=-1)
            correct = (predicted_trends == trend_targets).float()
            confidence_targets = correct.unsqueeze(-1)
        
        # Confidence regression loss
        confidence_loss = self.bce_loss(confidence, confidence_targets)
        
        # Combined loss
        total_loss = (
            self.trend_weight * trend_loss +
            self.confidence_weight * confidence_loss
        )
        
        loss_dict = {
            "total_loss": total_loss.item(),
            "trend_loss": trend_loss.item(),
            "confidence_loss": confidence_loss.item(),
            "accuracy": correct.mean().item(),
        }
        
        return total_loss, loss_dict


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Where:
    - p_t is the probability of the correct class
    - α_t is class weight
    - γ is focusing parameter (higher = more focus on hard examples)
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Initialize focal loss.
        
        Args:
            alpha: Class weights tensor of shape (num_classes,).
            gamma: Focusing parameter. Default: 2.0.
            reduction: Reduction method ("mean", "sum", "none").
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predicted logits (batch, num_classes).
            targets: True class labels (batch,).
            
        Returns:
            Focal loss value.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # Probability of correct class
        
        # Focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha[targets]
            focal_weight = focal_weight * alpha_t
        
        loss = focal_weight * ce_loss
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class ConfidenceWeightedLoss(nn.Module):
    """
    Loss that weights samples by predicted confidence.
    
    Encourages the model to be more confident on easier examples
    and less confident on harder examples.
    """
    
    def __init__(
        self,
        base_loss: nn.Module = None,
        confidence_penalty: float = 0.1,
    ):
        """
        Initialize confidence-weighted loss.
        
        Args:
            base_loss: Base loss function (default: CrossEntropyLoss).
            confidence_penalty: Penalty for low confidence on correct predictions.
        """
        super().__init__()
        self.base_loss = base_loss or nn.CrossEntropyLoss(reduction="none")
        self.confidence_penalty = confidence_penalty
    
    def forward(
        self,
        trend_logits: torch.Tensor,
        confidence: torch.Tensor,
        trend_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute confidence-weighted loss.
        
        High confidence + wrong = high loss
        Low confidence + wrong = moderate loss
        High confidence + correct = low loss
        Low confidence + correct = moderate loss (encourages confidence)
        """
        # Base classification loss
        base_loss = self.base_loss(trend_logits, trend_targets)
        
        # Check if prediction is correct
        with torch.no_grad():
            predicted = trend_logits.argmax(dim=-1)
            correct = (predicted == trend_targets).float()
        
        confidence = confidence.squeeze(-1)
        
        # Weight loss by confidence
        # If wrong: higher confidence = higher loss
        # If correct: lower confidence = penalty
        weight = torch.where(
            correct.bool(),
            1 - confidence * (1 - self.confidence_penalty),  # Correct: encourage confidence
            confidence,  # Wrong: penalize confidence
        )
        
        weighted_loss = base_loss * weight
        
        return weighted_loss.mean()


def main():
    """Test loss functions."""
    logging.basicConfig(level=logging.INFO)
    
    # Sample data
    batch_size = 32
    num_classes = 3
    
    trend_logits = torch.randn(batch_size, num_classes)
    confidence = torch.sigmoid(torch.randn(batch_size, 1))
    trend_targets = torch.randint(0, num_classes, (batch_size,))
    
    # Test CombinedLoss
    combined_loss = CombinedLoss()
    loss, loss_dict = combined_loss(trend_logits, confidence, trend_targets)
    print("CombinedLoss:")
    print(f"  Total: {loss_dict['total_loss']:.4f}")
    print(f"  Trend: {loss_dict['trend_loss']:.4f}")
    print(f"  Confidence: {loss_dict['confidence_loss']:.4f}")
    print(f"  Accuracy: {loss_dict['accuracy']:.2%}")
    
    # Test FocalLoss
    focal_loss = FocalLoss(gamma=2.0)
    fl = focal_loss(trend_logits, trend_targets)
    print(f"\nFocalLoss: {fl.item():.4f}")
    
    # Test ConfidenceWeightedLoss
    cw_loss = ConfidenceWeightedLoss()
    cwl = cw_loss(trend_logits, confidence, trend_targets)
    print(f"ConfidenceWeightedLoss: {cwl.item():.4f}")


if __name__ == "__main__":
    main()
