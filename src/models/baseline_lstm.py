"""
Baseline LSTM model for stock trend prediction with meta-prediction.

This model predicts:
1. Trend direction (Up/Down/Neutral) - classification head
2. Prediction confidence - meta-prediction head (reliability score)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import logging

from config import lstm_config, training_config

logger = logging.getLogger(__name__)


class BaselineLSTM(nn.Module):
    """
    LSTM-based stock trend prediction model with meta-prediction.
    
    Architecture:
    - Input: [sequence_length, input_size] - price + sentiment features
    - LSTM: 2-layer bidirectional LSTM with dropout
    - Trend Head: Dense layers → 3-class softmax (Down/Neutral/Up)
    - Confidence Head: Dense layers → sigmoid (meta-prediction)
    
    The confidence head learns to predict when the trend prediction
    is likely to be correct, enabling the model to "know what it knows".
    
    Attributes:
        lstm: The LSTM layers
        fc_shared: Shared fully-connected layer
        trend_head: Classification head for trend prediction
        confidence_head: Regression head for meta-prediction
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = None,
        num_layers: int = None,
        dropout: float = None,
        bidirectional: bool = None,
        num_trend_classes: int = None,
    ):
        """
        Initialize the model.
        
        Args:
            input_size: Number of input features per timestep.
            hidden_size: LSTM hidden state size. Default: 128.
            num_layers: Number of LSTM layers. Default: 2.
            dropout: Dropout probability. Default: 0.3.
            bidirectional: Use bidirectional LSTM. Default: False.
            num_trend_classes: Number of trend classes. Default: 3.
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size or lstm_config.hidden_size
        self.num_layers = num_layers or lstm_config.num_layers
        self.dropout = dropout or lstm_config.dropout
        self.bidirectional = bidirectional if bidirectional is not None else lstm_config.bidirectional
        self.num_trend_classes = num_trend_classes or lstm_config.num_trend_classes
        
        # Direction multiplier for bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
        )
        
        # Shared representation layer
        lstm_output_size = self.hidden_size * self.num_directions
        self.fc_shared = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        
        # Trend prediction head (classification)
        self.trend_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(self.dropout / 2),
            nn.Linear(32, self.num_trend_classes),
        )
        
        # Confidence prediction head (meta-prediction)
        # Outputs RAW LOGITS (sigmoid applied at inference, not training)
        # This is required for BCEWithLogitsLoss + mixed precision
        self.confidence_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(self.dropout / 2),
            nn.Linear(32, 1),
            # NO Sigmoid here - applied at inference only
        )
        
        # Initialize weights
        self._init_weights()
        
        logger.info(
            f"BaselineLSTM initialized: input={input_size}, hidden={self.hidden_size}, "
            f"layers={self.num_layers}, bidirectional={self.bidirectional}"
        )
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        
        for module in [self.fc_shared, self.trend_head, self.confidence_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        return_hidden: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_size).
            return_hidden: Whether to return LSTM hidden state.
            
        Returns:
            Tuple of:
            - trend_logits: (batch, num_trend_classes)
            - confidence: (batch, 1)
            - hidden: (batch, hidden_size) if return_hidden else None
        """
        # LSTM forward
        # lstm_out: (batch, seq_len, hidden_size * num_directions)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last timestep output
        # For bidirectional, concatenate forward and backward
        if self.bidirectional:
            # h_n: (num_layers * 2, batch, hidden_size)
            # Get last layer forward and backward
            h_forward = h_n[-2, :, :]  # Forward last layer
            h_backward = h_n[-1, :, :]  # Backward last layer
            final_hidden = torch.cat([h_forward, h_backward], dim=-1)
        else:
            final_hidden = h_n[-1, :, :]  # (batch, hidden_size)
        
        # Shared representation
        shared = self.fc_shared(final_hidden)  # (batch, 64)
        
        # Trend prediction
        trend_logits = self.trend_head(shared)  # (batch, num_classes)
        
        # Confidence prediction (meta-prediction) - raw logits
        confidence_logits = self.confidence_head(shared)  # (batch, 1)
        
        if return_hidden:
            return trend_logits, confidence_logits, final_hidden
        
        return trend_logits, confidence_logits, None
    
    def predict(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with confidence filtering.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_size).
            threshold: Minimum confidence to make a prediction.
            
        Returns:
            Dictionary with:
            - trend_class: Predicted class indices
            - trend_probs: Class probabilities
            - confidence: Confidence scores
            - should_trade: Boolean mask for high-confidence predictions
        """
        self.eval()
        with torch.no_grad():
            trend_logits, confidence_logits, _ = self.forward(x)
            trend_probs = F.softmax(trend_logits, dim=-1)
            trend_class = trend_probs.argmax(dim=-1)
            confidence = torch.sigmoid(confidence_logits)  # Apply sigmoid at inference
            
            return {
                "trend_class": trend_class,
                "trend_probs": trend_probs,
                "confidence": confidence.squeeze(-1),
                "should_trade": confidence.squeeze(-1) > threshold,
            }
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SentimentAttention(nn.Module):
    """
    Attention mechanism over sentiment features in the lookback window.
    
    This is an Iteration 3 enhancement. The idea is to learn which days
    in the lookback window are most important for prediction (typically
    days with strong sentiment signals).
    
    Uses multi-head self-attention with residual connection and layer norm.
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply self-attention over sequence.
        
        Args:
            x: (batch, seq_len, hidden_size)
            
        Returns:
            attended: (batch, seq_len, hidden_size) - attention-weighted output
            weights: (batch, seq_len, seq_len) - attention weights for visualization
        """
        attended, weights = self.attention(x, x, x)
        attended = self.layer_norm(attended + x)  # Residual connection
        return attended, weights


class AttentionLSTM(nn.Module):
    """
    LSTM with attention mechanism for sentiment weighting.
    
    This extends BaselineLSTM by adding attention over the LSTM outputs
    to focus on important timesteps (high-sentiment days).
    
    Architecture:
    - LSTM processes the full sequence
    - Self-attention weights the LSTM outputs
    - Weighted representation fed to classification heads
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_heads: int = 4,
        num_trend_classes: int = 3,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_trend_classes = num_trend_classes
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Attention over LSTM outputs
        self.attention = SentimentAttention(hidden_size, num_heads, dropout)
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.trend_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(32, num_trend_classes),
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(32, 1),
        )
        
        self._init_weights()
        
        logger.info(f"AttentionLSTM initialized: input={input_size}, hidden={hidden_size}, "
                    f"layers={num_layers}, heads={num_heads}")
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        
        for module in [self.fc, self.trend_head, self.confidence_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_size)
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of:
            - trend_logits: (batch, num_trend_classes)
            - confidence_logits: (batch, 1)
            - attn_weights: (batch, seq_len, seq_len) if return_attention else None
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        
        # Apply attention
        attended, attn_weights = self.attention(lstm_out)
        
        # Use attended representation from last timestep
        final = attended[:, -1, :]  # (batch, hidden_size)
        
        # Shared representation
        shared = self.fc(final)
        
        # Output heads
        trend_logits = self.trend_head(shared)
        confidence_logits = self.confidence_head(shared)
        
        if return_attention:
            return trend_logits, confidence_logits, attn_weights
        return trend_logits, confidence_logits, None
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """Make predictions with confidence filtering."""
        self.eval()
        with torch.no_grad():
            trend_logits, confidence_logits, _ = self.forward(x)
            trend_probs = F.softmax(trend_logits, dim=-1)
            trend_class = trend_probs.argmax(dim=-1)
            confidence = torch.sigmoid(confidence_logits)
            
            return {
                "trend_class": trend_class,
                "trend_probs": trend_probs,
                "confidence": confidence.squeeze(-1),
                "should_trade": confidence.squeeze(-1) > threshold,
            }
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DualBranchLSTM(nn.Module):
    """
    Dual-branch LSTM for separate price and sentiment processing.
    
    This is an advanced architecture for iteration 2.
    
    Branch A: Processes price + technical features
    Branch B: Processes sentiment features
    Fusion: Attention-based fusion of both branches
    """
    
    def __init__(
        self,
        price_input_size: int,
        sentiment_input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
        num_trend_classes: int = 3,
    ):
        """
        Initialize dual-branch model.
        
        Args:
            price_input_size: Number of price/technical features.
            sentiment_input_size: Number of sentiment features.
            hidden_size: Hidden size for each branch.
            num_layers: Number of LSTM layers per branch.
            dropout: Dropout probability.
            num_trend_classes: Number of trend classes.
        """
        super().__init__()
        
        # Price branch
        self.price_lstm = nn.LSTM(
            input_size=price_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Sentiment branch
        self.sentiment_lstm = nn.LSTM(
            input_size=sentiment_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Attention for fusion
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2),
            nn.Softmax(dim=-1),
        )
        
        # Combined representation
        # Weighted (1) + Price (1) + Sentiment (1) = 3 * hidden_size
        self.fc_combined = nn.Sequential(
            nn.Linear(hidden_size * 3, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Output heads
        self.trend_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_trend_classes),
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        price_features: torch.Tensor,
        sentiment_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            price_features: (batch, seq_len, price_input_size)
            sentiment_features: (batch, seq_len, sentiment_input_size)
            
        Returns:
            Tuple of (trend_logits, confidence, attention_weights).
        """
        # Process each branch
        _, (h_price, _) = self.price_lstm(price_features)
        _, (h_sentiment, _) = self.sentiment_lstm(sentiment_features)
        
        h_price = h_price[-1]  # (batch, hidden_size)
        h_sentiment = h_sentiment[-1]  # (batch, hidden_size)
        
        # Concatenate for attention
        combined = torch.cat([h_price, h_sentiment], dim=-1)  # (batch, hidden_size*2)
        
        # Compute attention weights
        attn_weights = self.attention(combined)  # (batch, 2)
        
        # Weighted combination
        h_weighted = (
            attn_weights[:, 0:1] * h_price + 
            attn_weights[:, 1:2] * h_sentiment
        )
        
        # Combine both representations
        h_fused = torch.cat([h_weighted, combined], dim=-1)
        h_fused = self.fc_combined(h_fused)
        
        # Output heads
        trend_logits = self.trend_head(h_fused)
        confidence = self.confidence_head(h_fused)
        
        return trend_logits, confidence, attn_weights


def main():
    """Test the LSTM model."""
    logging.basicConfig(level=logging.INFO)
    
    # Create model
    model = BaselineLSTM(
        input_size=25,  # Example: 5 price + 15 technical + 5 sentiment
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        bidirectional=False,
    )
    
    print(f"Model parameters: {model.get_num_parameters():,}")
    
    # Test forward pass
    batch_size = 32
    seq_len = 20
    x = torch.randn(batch_size, seq_len, 25)
    
    trend_logits, confidence, _ = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Trend logits shape: {trend_logits.shape}")
    print(f"Confidence shape: {confidence.shape}")
    
    # Test prediction
    predictions = model.predict(x)
    print(f"\nPredictions:")
    print(f"  Trend classes: {predictions['trend_class'][:5]}")
    print(f"  Confidence: {predictions['confidence'][:5]}")
    print(f"  Should trade: {predictions['should_trade'].sum().item()}/{batch_size}")


if __name__ == "__main__":
    main()
