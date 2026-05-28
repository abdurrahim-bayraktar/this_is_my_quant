"""
Hybrid Attention LSTM — Dual-Branch Architecture.

Splits the input feature vector into two branches:

Branch 1 (Technical): Multi-layer LSTM with multi-head self-attention
    over the full lookback window.  Processes the 20 SHAP-selected
    technical indicators to extract structural market momentum.

Branch 2 (Sentiment): Lightweight single-layer GRU.  Processes the
    2 RF-selected sentiment features (sent_surprise, sent_ema_20d)
    through a smaller temporal model that respects their lower
    dimensionality and noisier nature.

Fusion: The final hidden states of both branches are concatenated
    and passed through shared Dense layers before the classification
    head.  This lets the model learn technical trends independently,
    using sentiment as a learned modifier rather than a foundational
    input.

The model accepts a *single* concatenated input tensor and splits
it internally by feature index, so the existing data pipeline,
scaler, and walk-forward loop require no changes.

Design choices:
    - GRU (not LSTM) for sentiment: fewer parameters (no cell state),
      less prone to overfitting on only 2 input features.
    - Sentiment hidden_size=32 vs technical hidden_size=128: keeps
      the sentiment branch intentionally small (~3K params) so it
      cannot dominate gradient updates.
    - Attention applied only to the technical branch: the LSTM needs
      help choosing *which* timesteps matter for price dynamics.
      Sentiment is already a derived summary statistic.
    - No confidence head: follows the v2+ experiment convention of
      using CrossEntropyLoss only (confidence output intentionally
      disabled).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

from src.models.baseline_lstm import SentimentAttention

logger = logging.getLogger(__name__)


class HybridAttentionLSTM(nn.Module):
    """
    Dual-branch architecture: LSTM+Attention for technicals, GRU for sentiment.

    Architecture overview::

        Input: [batch, seq_len, n_tech + n_sent]
                    │
            ┌───────┴───────┐
            │               │
        x[:,:,:n_tech]  x[:,:,n_tech:]
            │               │
        LayerNorm(n_tech) LayerNorm(n_sent)
            │               │
        LSTM(2-layer,    GRU(1-layer,
             hidden=128)     hidden=32)
            │               │
        SentimentAttn       │
        (4 heads)           │
            │               │
        attended[:,-1,:]  h_n[-1]
        (batch, 128)     (batch, 32)
            │               │
            └───────┬───────┘
                    │
            Concat → (batch, 160)
                    │
            FC: 160 → 64, ReLU, Dropout
                    │
            Trend Head: 64 → 32 → 3
                    │
                trend_logits

    Forward returns ``(trend_logits, None, None)`` for Trainer compatibility.
    """

    def __init__(
        self,
        n_technical_features: int = 20,
        n_sentiment_features: int = 2,
        # Technical branch hyperparameters (match AttentionLSTM defaults)
        tech_hidden_size: int = 128,
        tech_num_layers: int = 2,
        tech_num_heads: int = 4,
        # Sentiment branch hyperparameters
        sent_hidden_size: int = 32,
        sent_num_layers: int = 1,
        # Shared
        dropout: float = 0.3,
        num_classes: int = 3,
    ):
        """
        Initialize dual-branch model.

        Args:
            n_technical_features: Number of technical indicator features
                (first N columns of the input tensor).
            n_sentiment_features: Number of sentiment features
                (last M columns of the input tensor).
            tech_hidden_size: Hidden size for the technical LSTM branch.
            tech_num_layers: Number of LSTM layers in the technical branch.
            tech_num_heads: Number of attention heads for the technical branch.
            sent_hidden_size: Hidden size for the sentiment GRU branch.
            sent_num_layers: Number of GRU layers in the sentiment branch.
            dropout: Dropout probability.
            num_classes: Number of output classes (default: 3).
        """
        super().__init__()

        self.n_technical = n_technical_features
        self.n_sentiment = n_sentiment_features
        self.tech_hidden_size = tech_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.num_classes = num_classes

        # ── Technical Branch ───────────────────────────────────────────
        self.tech_norm = nn.LayerNorm(n_technical_features)

        self.tech_lstm = nn.LSTM(
            input_size=n_technical_features,
            hidden_size=tech_hidden_size,
            num_layers=tech_num_layers,
            batch_first=True,
            dropout=dropout if tech_num_layers > 1 else 0,
        )

        # Multi-head self-attention over LSTM outputs (reuses existing class)
        self.tech_attention = SentimentAttention(
            hidden_size=tech_hidden_size,
            num_heads=tech_num_heads,
            dropout=dropout,
        )

        # ── Sentiment Branch (skipped when n_sentiment_features=0) ────
        self.has_sentiment_branch = n_sentiment_features > 0

        if self.has_sentiment_branch:
            self.sent_norm = nn.LayerNorm(n_sentiment_features)

            self.sent_gru = nn.GRU(
                input_size=n_sentiment_features,
                hidden_size=sent_hidden_size,
                num_layers=sent_num_layers,
                batch_first=True,
                dropout=dropout if sent_num_layers > 1 else 0,
            )
            fusion_input_size = tech_hidden_size + sent_hidden_size
        else:
            fusion_input_size = tech_hidden_size

        # ── Fusion ─────────────────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Classification Head ────────────────────────────────────────
        self.trend_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(32, num_classes),
        )

        # Initialize weights
        self._init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        tech_params = (
            sum(p.numel() for p in self.tech_lstm.parameters())
            + sum(p.numel() for p in self.tech_attention.parameters())
            + sum(p.numel() for p in self.tech_norm.parameters())
        )
        sent_params = (
            (
                sum(p.numel() for p in self.sent_gru.parameters())
                + sum(p.numel() for p in self.sent_norm.parameters())
            ) if self.has_sentiment_branch else 0
        )
        fusion_params = (
            sum(p.numel() for p in self.fusion.parameters())
            + sum(p.numel() for p in self.trend_head.parameters())
        )
        logger.info(
            f"HybridAttentionLSTM: "
            f"tech={n_technical_features}→LSTM({tech_hidden_size}×{tech_num_layers})+Attn({tech_num_heads}h) [{tech_params:,}p], "
            f"sent={n_sentiment_features}→GRU({sent_hidden_size}×{sent_num_layers}) [{sent_params:,}p], "
            f"fusion [{fusion_params:,}p], total={n_params:,}p"
        )

    def _init_weights(self):
        """Initialize weights with proven initializations for recurrent nets."""
        # Technical LSTM
        for name, param in self.tech_lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        # Sentiment GRU
        if self.has_sentiment_branch:
            for name, param in self.sent_gru.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

        # Dense layers
        for module in [self.fusion, self.trend_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass — splits input into technical and sentiment branches.

        Args:
            x: Input tensor of shape (batch, seq_len, n_tech + n_sent).
                The first ``n_technical`` columns are technical indicators,
                the remaining ``n_sentiment`` columns are sentiment features.

        Returns:
            Tuple of (trend_logits, None, None) for Trainer compatibility.
        """
        # ── Split input by feature index ──────────────────────────────
        x_tech = x[:, :, : self.n_technical]   # (batch, seq_len, n_tech)
        x_sent = x[:, :, self.n_technical :]   # (batch, seq_len, n_sent)

        # ── Technical Branch ──────────────────────────────────────────
        x_tech = self.tech_norm(x_tech)
        tech_out, _ = self.tech_lstm(x_tech)            # (batch, seq_len, tech_hidden)
        tech_attended, _ = self.tech_attention(tech_out) # (batch, seq_len, tech_hidden)
        h_tech = tech_attended[:, -1, :]                 # (batch, tech_hidden)

        # ── Sentiment Branch (skipped when no sentiment features) ─────
        if self.has_sentiment_branch:
            x_sent = self.sent_norm(x_sent)
            _, h_sent = self.sent_gru(x_sent)  # h_sent: (n_layers, batch, sent_hidden)
            h_sent = h_sent[-1]                 # (batch, sent_hidden)
            h_fused = torch.cat([h_tech, h_sent], dim=-1)  # (batch, tech_h + sent_h)
        else:
            h_fused = h_tech  # (batch, tech_hidden)

        # ── Fusion ────────────────────────────────────────────────────
        shared = self.fusion(h_fused)                    # (batch, 64)

        # ── Classification ────────────────────────────────────────────
        trend_logits = self.trend_head(shared)  # (batch, num_classes)

        return trend_logits, None, None

    def predict(
        self, x: torch.Tensor, threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with probability outputs.

        Args:
            x: Input tensor of shape (batch, seq_len, n_tech + n_sent).
            threshold: Confidence threshold for should_trade flag.

        Returns:
            Dict with trend_class, trend_probs, confidence, should_trade.
        """
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
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def main():
    """Standalone test: shape, gradient flow, and no-sentiment mode."""
    logging.basicConfig(level=logging.INFO)

    batch_size = 32
    seq_len = 20
    n_tech = 20
    n_sent = 2
    n_features = n_tech + n_sent

    model = HybridAttentionLSTM(
        n_technical_features=n_tech,
        n_sentiment_features=n_sent,
    )

    # --- Shape test ---
    x = torch.randn(batch_size, seq_len, n_features)
    logits, _, _ = model(x)
    assert logits.shape == (batch_size, 3), f"Expected (32, 3), got {logits.shape}"
    print(f"[OK] Shape test passed: {logits.shape}")

    # --- Gradient flow test ---
    model.train()
    x = torch.randn(batch_size, seq_len, n_features)
    logits, _, _ = model(x)
    loss = logits.sum()
    loss.backward()

    dead_params = []
    for name, param in model.named_parameters():
        if param.grad is None:
            dead_params.append(name)
        elif param.grad.abs().max().item() == 0:
            dead_params.append(f"{name} (zero grad)")

    if dead_params:
        print(f"[FAIL] Dead parameters: {dead_params}")
    else:
        print(f"[OK] Gradient flow test passed: all {model.get_num_parameters():,} params have gradients")

    # --- No-sentiment mode (all-zero sentiment) ---
    model.eval()
    x_nosent = torch.randn(batch_size, seq_len, n_features)
    x_nosent[:, :, n_tech:] = 0.0  # Zero out sentiment features
    logits_nosent, _, _ = model(x_nosent)
    assert not torch.isnan(logits_nosent).any(), "NaN in no-sentiment output"
    assert not torch.isinf(logits_nosent).any(), "Inf in no-sentiment output"
    print(f"[OK] No-sentiment mode test passed: no NaN/Inf")

    # --- Parameter breakdown ---
    print(f"\nTotal parameters: {model.get_num_parameters():,}")

    # --- Prediction test ---
    preds = model.predict(x)
    print(f"[OK] Predict test: classes={preds['trend_class'].shape}, "
          f"probs={preds['trend_probs'].shape}, "
          f"confidence range=[{preds['confidence'].min():.3f}, {preds['confidence'].max():.3f}]")


if __name__ == "__main__":
    main()
