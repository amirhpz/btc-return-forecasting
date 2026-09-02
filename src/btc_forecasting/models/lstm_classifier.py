from __future__ import annotations

import torch
from torch import nn


class LSTMClassifier(nn.Module):
    """Frozen unidirectional LSTM backbone with a three-logit classification head."""

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        configured_dropout: float,
        class_count: int = 3,
    ) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0 or num_layers <= 0:
            raise ValueError("LSTM dimensions must be positive")
        if class_count != 3:
            raise ValueError("E06-C-WF4 requires exactly three output classes")
        if not 0.0 <= configured_dropout < 1.0:
            raise ValueError("configured_dropout must be in [0, 1)")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.class_count = class_count
        self.configured_dropout = configured_dropout
        self.effective_lstm_dropout = configured_dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=self.effective_lstm_dropout,
            bidirectional=False,
        )
        self.classification_head = nn.Linear(hidden_size, class_count)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("LSTM input must have shape [batch, time, features]")
        if features.shape[2] != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} features per timestep, got {features.shape[2]}"
            )
        _, (hidden_state, _) = self.lstm(features)
        return self.classification_head(hidden_state[-1])
