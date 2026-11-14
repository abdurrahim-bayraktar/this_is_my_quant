"""
DA-RNN Classifier — Classification variant of the Dual-Stage Attention-Based RNN.

Copies the encoder/decoder architecture from darnn.py but modifies the decoder's
final FC head to output `output_size` class logits instead of a single scalar.

This file exists separately from darnn.py to guarantee backward compatibility
with the regression DA-RNN pipeline.

Key differences from darnn.py:
  - TemporalAttentionDecoderClassifier: final FC outputs (batch, output_size)
  - DARNNClassifier: wrapper with output_size parameter (default 3)
  - predict() returns {"logits": ..., "prediction": argmax}
"""

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class InputAttentionEncoder(nn.Module):
    """
    Encoder with input attention mechanism.

    Identical to darnn.InputAttentionEncoder — duplicated here to keep
    the classifier fully self-contained and avoid any coupling to the
    regression model.

    At each time step t, computes attention weights α over all n input features
    using the encoder's previous hidden/cell state and the full input matrix X.
    Produces weighted input x̃_t = α_t ⊙ x_t, fed into the LSTM cell.
    Returns all encoder hidden states H = [h_1, …, h_T].
    """

    def __init__(self, input_size: int, hidden_size: int, seq_length: int, dropout: float = 0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)

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

        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        s = torch.zeros(batch_size, self.hidden_size, device=x.device)

        x_transposed = x.permute(0, 2, 1)  # (batch, n_features, T)

        encoder_outputs = []
        input_attn_weights = []

        for t in range(T):
            state_proj = self.attn_W(torch.cat([h, s], dim=1))  # (batch, T)
            state_proj = state_proj.unsqueeze(1)  # (batch, 1, T)

            feat_proj = self.attn_U(x_transposed)  # (batch, n_features, T)

            e = self.attn_v(torch.tanh(state_proj + feat_proj)).squeeze(-1)  # (batch, n_features)

            alpha = F.softmax(e, dim=1)  # (batch, n_features)
            input_attn_weights.append(alpha)

            x_tilde = alpha * x[:, t, :]  # (batch, n_features)

            h, s = self.lstm_cell(x_tilde, (h, s))
            encoder_outputs.append(h)

        encoder_outputs = torch.stack(encoder_outputs, dim=1)  # (batch, T, hidden)
        input_attn_weights = torch.stack(input_attn_weights, dim=1)  # (batch, T, n_features)

        encoder_outputs = self.dropout(encoder_outputs)

        return encoder_outputs, input_attn_weights


class TemporalAttentionDecoderClassifier(nn.Module):
    """
    Decoder with temporal attention mechanism — classification variant.

    Same temporal attention as darnn.TemporalAttentionDecoder, but the final FC
    head outputs `output_size` logits instead of a single scalar.

    No Softmax is applied — nn.CrossEntropyLoss expects raw logits.
    """

    def __init__(
        self,
        encoder_hidden: int,
        decoder_hidden: int,
        output_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder_hidden = encoder_hidden
        self.decoder_hidden = decoder_hidden
        self.output_size = output_size

        self.lstm_cell = nn.LSTMCell(encoder_hidden + 1, decoder_hidden)

        # Temporal attention layers
        self.attn_linear = nn.Linear(2 * decoder_hidden, encoder_hidden)
        self.attn_encoder = nn.Linear(encoder_hidden, encoder_hidden, bias=False)
        self.attn_v = nn.Linear(encoder_hidden, 1, bias=False)

        # Historical target embedding (continuous y_history → small projection)
        self.target_embedding = nn.Linear(1, 1)

        self.dropout = nn.Dropout(dropout)

        # Classification head: outputs `output_size` logits (no softmax)
        self.fc = nn.Sequential(
            nn.Linear(decoder_hidden + encoder_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(32, output_size),
        )

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        y_history: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_outputs: (batch, T, encoder_hidden) — from InputAttentionEncoder
            y_history: (batch, T) — scaled continuous historical target values

        Returns:
            logits: (batch, output_size) — raw class logits
            temporal_attn_weights: (batch, T, T) — attention weights per decoder step
        """
        batch_size, T, _ = encoder_outputs.shape

        d = torch.zeros(batch_size, self.decoder_hidden, device=encoder_outputs.device)
        s_prime = torch.zeros(batch_size, self.decoder_hidden, device=encoder_outputs.device)

        encoder_proj = self.attn_encoder(encoder_outputs)

        temporal_attn_weights = []

        for t in range(T):
            state_part = self.attn_linear(torch.cat([d, s_prime], dim=1))
            state_part = state_part.unsqueeze(1)

            l = self.attn_v(torch.tanh(encoder_proj + state_part)).squeeze(-1)
            beta = F.softmax(l, dim=1)
            temporal_attn_weights.append(beta)

            c_t = torch.bmm(beta.unsqueeze(1), encoder_outputs).squeeze(1)

            y_t = y_history[:, t].unsqueeze(1)
            y_tilde = self.target_embedding(y_t)

            decoder_input = torch.cat([c_t, y_tilde], dim=1)
            d, s_prime = self.lstm_cell(decoder_input, (d, s_prime))

        d = self.dropout(d)
        c_final = torch.bmm(
            temporal_attn_weights[-1].unsqueeze(1), encoder_outputs
        ).squeeze(1)

        logits = self.fc(torch.cat([d, c_final], dim=1))  # (batch, output_size)

        temporal_attn_weights = torch.stack(temporal_attn_weights, dim=1)

        return logits, temporal_attn_weights


class DARNNClassifier(nn.Module):
    """
    Dual-Stage Attention-Based RNN for Classification.

    Combines InputAttentionEncoder and TemporalAttentionDecoderClassifier.
    Outputs raw logits of shape (batch, output_size) — no softmax applied.

    forward(x_features, y_history) returns (logits, None, None)
    for compatibility with DARNNClassificationTrainer.
    """

    def __init__(
        self,
        input_size: int,
        seq_length: int = 20,
        encoder_hidden: int = 64,
        decoder_hidden: int = 64,
        output_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.output_size = output_size
        self.encoder = InputAttentionEncoder(input_size, encoder_hidden, seq_length, dropout)
        self.decoder = TemporalAttentionDecoderClassifier(
            encoder_hidden, decoder_hidden, output_size, dropout
        )

        self._init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"DARNNClassifier: input={input_size}, enc_hidden={encoder_hidden}, "
            f"dec_hidden={decoder_hidden}, output_size={output_size}, params={n_params:,}"
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
            y_history: (batch, T) — scaled continuous historical target values
            return_attention: if True, return attention weights instead of None

        Returns:
            logits: (batch, output_size) — raw class logits
            input_attn_weights or None
            temporal_attn_weights or None
        """
        encoder_outputs, input_attn_weights = self.encoder(x_features)
        logits, temporal_attn_weights = self.decoder(encoder_outputs, y_history)

        if return_attention:
            return logits, input_attn_weights, temporal_attn_weights

        return logits, None, None

    def predict(
        self,
        x_features: torch.Tensor,
        y_history: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Inference interface returning both logits and predicted class."""
        self.eval()
        with torch.no_grad():
            logits, _, _ = self.forward(x_features, y_history)
            prediction = logits.argmax(dim=-1)
            return {"logits": logits, "prediction": prediction}
