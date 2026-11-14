"""
DA-RNN — Dual-Stage Attention-Based Recurrent Neural Network.

Implements the architecture from Qin et al. (2017):
  "A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction"

Three classes:
  - InputAttentionEncoder: LSTM with adaptive input-feature attention
  - TemporalAttentionDecoder: LSTM with temporal attention over encoder states
  - DARNN: Wrapper combining encoder + decoder with regression head
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class InputAttentionEncoder(nn.Module):
    """
    Encoder with input attention mechanism.

    At each time step t, computes attention weights α over all n input features
    using the encoder's previous hidden/cell state and the full input matrix X.
    Produces weighted input x̃_t = α_t ⊙ x_t, fed into the LSTM cell.
    Returns all encoder hidden states H = [h_1, …, h_T].

    Following the paper (Eq. 8-9):
      e^k_t = v_e^T * tanh(W_e [h_{t-1}; s_{t-1}] + U_e x^k)
    where x^k is the k-th driving series (feature) across all T time steps,
    i.e., x^k has shape (T,). We compute all n_features scores simultaneously.
    """

    def __init__(self, input_size: int, hidden_size: int, seq_length: int, dropout: float = 0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)

        # Input attention layers following the paper:
        # W_e: maps [h_{t-1}; s_{t-1}] (2*hidden) → T
        # U_e: maps x^k (T values for each feature) → T
        # v_e: maps T → 1 (per-feature score)
        # We need seq_length to define these dimensions
        self.attn_W = nn.Linear(2 * hidden_size, seq_length)
        self.attn_U = nn.Linear(seq_length, seq_length, bias=False)
        self.attn_v = nn.Linear(seq_length, 1, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, T, input_size) — driving series

        Returns:
            encoder_outputs: (batch, T, hidden_size) — all hidden states
            input_attn_weights: (batch, T, input_size) — attention weights per step
        """
        batch_size, T, n_features = x.shape

        # Initialize hidden and cell states
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        s = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # Transpose x for feature-wise attention: (batch, n_features, T)
        x_transposed = x.permute(0, 2, 1)

        encoder_outputs = []
        input_attn_weights = []

        for t in range(T):
            # State projection: (batch, 2*hidden) → (batch, T)
            state_proj = self.attn_W(torch.cat([h, s], dim=1))  # (batch, T)

            # Expand for broadcasting: (batch, 1, T)
            state_proj = state_proj.unsqueeze(1)  # (batch, 1, T)

            # Feature projection: (batch, n_features, T) → (batch, n_features, T)
            feat_proj = self.attn_U(x_transposed)  # (batch, n_features, T)

            # Score per feature: v^T * tanh(state + feat)
            # (batch, n_features, T) → (batch, n_features, 1) → (batch, n_features)
            e = self.attn_v(torch.tanh(state_proj + feat_proj)).squeeze(-1)  # (batch, n_features)

            alpha = F.softmax(e, dim=1)  # (batch, n_features)
            input_attn_weights.append(alpha)

            # Weighted input
            x_tilde = alpha * x[:, t, :]  # (batch, n_features)

            # LSTM cell step
            h, s = self.lstm_cell(x_tilde, (h, s))

            encoder_outputs.append(h)

        # Stack along time dimension
        encoder_outputs = torch.stack(encoder_outputs, dim=1)  # (batch, T, hidden)
        input_attn_weights = torch.stack(input_attn_weights, dim=1)  # (batch, T, n_features)

        encoder_outputs = self.dropout(encoder_outputs)

        return encoder_outputs, input_attn_weights


class TemporalAttentionDecoder(nn.Module):
    """
    Decoder with temporal attention mechanism.

    At each step t, computes attention weights β over all encoder hidden states
    using the decoder's previous state (d_{t-1}, s'_{t-1}).
    Produces context vector c_t = Σ β_t^i · h_i.
    Context is combined with the scaled historical target y_{t-1} and fed into
    the decoder LSTM cell.
    """

    def __init__(self, encoder_hidden: int, decoder_hidden: int, dropout: float = 0.2):
        super().__init__()
        self.encoder_hidden = encoder_hidden
        self.decoder_hidden = decoder_hidden

        self.lstm_cell = nn.LSTMCell(encoder_hidden + 1, decoder_hidden)

        # Temporal attention: maps (d_{t-1}, s'_{t-1}, h_i) → scalar score per encoder step
        self.attn_linear = nn.Linear(2 * decoder_hidden, encoder_hidden)
        self.attn_encoder = nn.Linear(encoder_hidden, encoder_hidden, bias=False)
        self.attn_v = nn.Linear(encoder_hidden, 1, bias=False)

        # Small linear embedding for historical target value (paper: y_{t-1} → ỹ_{t-1})
        self.target_embedding = nn.Linear(1, 1)

        self.dropout = nn.Dropout(dropout)

        # Regression head: final hidden state → scalar prediction
        self.fc = nn.Sequential(
            nn.Linear(decoder_hidden + encoder_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        y_history: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_outputs: (batch, T, encoder_hidden) — from InputAttentionEncoder
            y_history: (batch, T) — scaled historical target values

        Returns:
            prediction: (batch,) — scalar prediction
            temporal_attn_weights: (batch, T, T) — attention weights per decoder step
        """
        batch_size, T, _ = encoder_outputs.shape

        # Initialize decoder hidden and cell states
        d = torch.zeros(batch_size, self.decoder_hidden, device=encoder_outputs.device)
        s_prime = torch.zeros(batch_size, self.decoder_hidden, device=encoder_outputs.device)

        # Pre-compute encoder projections for attention (shared across decoder steps)
        # (batch, T, encoder_hidden)
        encoder_proj = self.attn_encoder(encoder_outputs)

        temporal_attn_weights = []

        for t in range(T):
            # Compute temporal attention weights
            # (batch, 2*decoder_hidden) → (batch, encoder_hidden)
            state_part = self.attn_linear(torch.cat([d, s_prime], dim=1))

            # (batch, encoder_hidden) → (batch, 1, encoder_hidden) for broadcasting
            state_part = state_part.unsqueeze(1)

            # Attention scores over all encoder time steps
            # (batch, T, encoder_hidden) + (batch, 1, encoder_hidden) → (batch, T, encoder_hidden)
            l = self.attn_v(torch.tanh(encoder_proj + state_part)).squeeze(-1)
            # (batch, T)

            beta = F.softmax(l, dim=1)  # (batch, T)
            temporal_attn_weights.append(beta)

            # Context vector: weighted sum of encoder hidden states
            # (batch, T) → (batch, 1, T) @ (batch, T, encoder_hidden) → (batch, 1, encoder_hidden)
            c_t = torch.bmm(beta.unsqueeze(1), encoder_outputs).squeeze(1)
            # (batch, encoder_hidden)

            # Embed historical target value
            y_t = y_history[:, t].unsqueeze(1)  # (batch, 1)
            y_tilde = self.target_embedding(y_t)  # (batch, 1)

            # Decoder LSTM input: context + embedded target
            decoder_input = torch.cat([c_t, y_tilde], dim=1)  # (batch, encoder_hidden + 1)

            # LSTM cell step
            d, s_prime = self.lstm_cell(decoder_input, (d, s_prime))

        # Final prediction from last decoder state + last context
        d = self.dropout(d)
        # Compute final context from last temporal attention
        c_final = torch.bmm(
            temporal_attn_weights[-1].unsqueeze(1), encoder_outputs
        ).squeeze(1)

        prediction = self.fc(torch.cat([d, c_final], dim=1)).squeeze(-1)  # (batch,)

        temporal_attn_weights = torch.stack(temporal_attn_weights, dim=1)  # (batch, T, T)

        return prediction, temporal_attn_weights


class DARNN(nn.Module):
    """
    Dual-Stage Attention-Based Recurrent Neural Network (DA-RNN).

    Combines InputAttentionEncoder and TemporalAttentionDecoder.

    forward(x_features, y_history) returns (prediction, None, None)
    for compatibility with RegressionTrainer / DARNNTrainer.
    """

    def __init__(
        self,
        input_size: int,
        seq_length: int = 20,
        encoder_hidden: int = 64,
        decoder_hidden: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = InputAttentionEncoder(input_size, encoder_hidden, seq_length, dropout)
        self.decoder = TemporalAttentionDecoder(encoder_hidden, decoder_hidden, dropout)

        self._init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"DARNN: input={input_size}, enc_hidden={encoder_hidden}, "
            f"dec_hidden={decoder_hidden}, params={n_params:,}"
        )

    def _init_weights(self):
        """Xavier/Orthogonal initialization for LSTM cells and linear layers."""
        for module in [self.encoder, self.decoder]:
            for name, param in module.named_parameters():
                if "lstm_cell" in name:
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)
                elif "weight" in name and param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

    def forward(
        self,
        x_features: torch.Tensor,
        y_history: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, None, None]:
        """
        Args:
            x_features: (batch, T, n_features) — driving series
            y_history: (batch, T) — scaled historical target values
            return_attention: if True, return attention weights instead of None

        Returns:
            prediction: (batch,) — scalar prediction
            input_attn_weights or None
            temporal_attn_weights or None
        """
        encoder_outputs, input_attn_weights = self.encoder(x_features)
        prediction, temporal_attn_weights = self.decoder(encoder_outputs, y_history)

        if return_attention:
            return prediction, input_attn_weights, temporal_attn_weights

        return prediction, None, None

    def predict(
        self,
        x_features: torch.Tensor,
        y_history: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Inference interface matching RegressionAttentionLSTM.predict()."""
        self.eval()
        with torch.no_grad():
            prediction, _, _ = self.forward(x_features, y_history)
            return {"prediction": prediction}
