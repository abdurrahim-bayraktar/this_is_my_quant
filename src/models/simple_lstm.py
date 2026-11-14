"""
Simplified Trend LSTM Model - Cleaned up architecture.

Fixes applied from architecture analysis:
1. Removed unused confidence head (saves ~2K params, clearer optimization)
2. Removed multi-head attention (overkill for 20-step sequences)
3. Reduced dropout chain (single dropout layer instead of 4)
4. Cleaner forward pass returning only what's needed

For use with standard CrossEntropyLoss - no CombinedLoss needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class SimpleTrendLSTM(nn.Module):
    """
    Simplified LSTM for trend prediction.
    
    Architecture:
    - Input: [batch, sequence_length, input_size]
    - LSTM: Process temporal sequence
    - Classifier: Dense layers → 3-class logits
    
    No attention, no confidence head - pure classification.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        num_classes: int = 3,
    ):
        """
        Initialize simplified model.
        
        Args:
            input_size: Number of input features per timestep.
            hidden_size: LSTM hidden state size.
            num_layers: Number of LSTM layers.
            dropout: Dropout probability (applied once, not 4x).
            num_classes: Number of trend classes (default: 3).
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Optional input normalization
        self.input_norm = nn.LayerNorm(input_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Single classifier head (no confidence head)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),  # Single dropout, not 4 layers
            nn.Linear(32, num_classes),
        )
        
        # Initialize weights
        self._init_weights()
        
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"SimpleTrendLSTM: input={input_size}, hidden={hidden_size}, "
                    f"layers={num_layers}, params={n_params:,}")
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_size)
            
        Returns:
            Tuple of (trend_logits, None, None) for compatibility with Trainer
        """
        # Optional: normalize input
        x = self.input_norm(x)
        
        # LSTM forward - use last hidden state
        _, (h_n, _) = self.lstm(x)
        final_hidden = h_n[-1]  # (batch, hidden_size)
        
        # Classification
        logits = self.classifier(final_hidden)
        
        # Return tuple for Trainer compatibility (trend, confidence, hidden)
        # Confidence is None since we removed that head
        return logits, None, None
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Make predictions.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_size)
            threshold: Unused, kept for API compatibility
            
        Returns:
            Dictionary with trend_class, trend_probs, confidence, should_trade
        """
        self.eval()
        with torch.no_grad():
            logits, _, _ = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            classes = probs.argmax(dim=-1)
            
            # Confidence is just the max probability (no meta-prediction)
            confidence = probs.max(dim=-1).values
            
            return {
                "trend_class": classes,
                "trend_probs": probs,
                "confidence": confidence,
                "should_trade": confidence > threshold,
            }
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MediumTrendLSTM(nn.Module):
    """
    Medium complexity LSTM - keeps attention but removes confidence head.
    
    Use this to isolate whether attention helps or hurts.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        num_classes: int = 3,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_size)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Simple temporal attention (not multi-head)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )
        
        self._init_weights()
        
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"MediumTrendLSTM: input={input_size}, hidden={hidden_size}, "
                    f"layers={num_layers}, params={n_params:,}")
    
    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> tuple:
        x = self.input_norm(x)
        
        # LSTM forward
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)
        
        # Simple attention pooling
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden)
        
        # Classification
        logits = self.classifier(context)
        
        return logits, None, attn_weights
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            logits, _, _ = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            classes = probs.argmax(dim=-1)
            confidence = probs.max(dim=-1).values
            
            return {
                "trend_class": classes,
                "trend_probs": probs,
                "confidence": confidence,
                "should_trade": confidence > threshold,
            }
    
    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def main():
    """Test simplified models."""
    logging.basicConfig(level=logging.INFO)
    
    batch_size = 32
    seq_len = 20
    n_features = 20  # SHAP top 20
    
    x = torch.randn(batch_size, seq_len, n_features)
    
    # Test SimpleTrendLSTM
    model1 = SimpleTrendLSTM(input_size=n_features, hidden_size=64, num_layers=1)
    logits1, _, _ = model1(x)
    print(f"SimpleTrendLSTM output: {logits1.shape}")
    print(f"Parameters: {model1.get_num_parameters():,}")
    
    # Test MediumTrendLSTM
    model2 = MediumTrendLSTM(input_size=n_features, hidden_size=64, num_layers=1)
    logits2, _, attn = model2(x)
    print(f"\nMediumTrendLSTM output: {logits2.shape}")
    print(f"Attention shape: {attn.shape}")
    print(f"Parameters: {model2.get_num_parameters():,}")
    
    # Compare with AttentionLSTM
    try:
        from src.models import AttentionLSTM
        model3 = AttentionLSTM(input_size=n_features, hidden_size=64, num_layers=1)
        print(f"\nAttentionLSTM parameters: {model3.get_num_parameters():,}")
    except Exception as e:
        print(f"Could not load AttentionLSTM: {e}")


if __name__ == "__main__":
    main()
