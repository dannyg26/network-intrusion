"""
models.py
PyTorch model definitions:
  - MLPModel      : Baseline fully-connected network (256 → 128 → 64)
  - CNNLSTMModel  : Hybrid 1D-Conv + LSTM model
"""

import torch
import torch.nn as nn


# ── Shared helper ─────────────────────────────────────────────────────────────
class _BNDropBlock(nn.Module):
    """BatchNorm → ReLU → Dropout in one reusable block."""
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


# ── Baseline MLP ──────────────────────────────────────────────────────────────
class MLPModel(nn.Module):
    """
    Fully-connected MLP baseline.
    Architecture: Input → 256 → 128 → 64 → num_classes
    Each hidden layer has BatchNorm + ReLU + Dropout(0.3).
    """

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 256),
            _BNDropBlock(256, dropout),
            # Layer 2
            nn.Linear(256, 128),
            _BNDropBlock(128, dropout),
            # Layer 3
            nn.Linear(128, 64),
            _BNDropBlock(64, dropout),
            # Output
            nn.Linear(64, num_classes),
        )

    def forward(self, x):          # x: (B, input_dim)
        return self.net(x)


# ── CNN-LSTM Hybrid ───────────────────────────────────────────────────────────
class CNNLSTMModel(nn.Module):
    """
    Hybrid CNN-LSTM model.
    Treats the feature vector as a 1-D sequence of length `input_dim` with
    channel size 1, allowing Conv1D layers to extract local patterns and the
    LSTM to model sequential dependencies.

    Architecture:
      (B, input_dim) → reshape → (B, 1, input_dim)
      Conv1d(1→64, k=3) + BN + ReLU + MaxPool(2)
      Conv1d(64→128, k=3) + BN + ReLU + MaxPool(2)
      LSTM(128, 128, batch_first=True)
      Dense 128 → 64 → num_classes
    Optional residual skip from the flattened conv output to the dense layer.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        lstm_hidden: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim

        # ── CNN blocks ───────────────────────────────────────────────────────
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),          # L/2
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),          # L/4
        )

        # ── LSTM ─────────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )

        # ── Classifier head ──────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):          # x: (B, input_dim)
        B = x.size(0)
        # Reshape to (B, channels=1, length=input_dim)
        out = x.unsqueeze(1)               # (B, 1, input_dim)
        out = self.conv1(out)              # (B, 64,  input_dim//2)
        out = self.conv2(out)              # (B, 128, input_dim//4)
        # Permute for LSTM: (B, seq_len, features)
        out = out.permute(0, 2, 1)        # (B, input_dim//4, 128)
        out, _ = self.lstm(out)            # (B, input_dim//4, lstm_hidden)
        out = out[:, -1, :]               # take last time-step (B, lstm_hidden)
        return self.classifier(out)        # (B, num_classes)


# ── Factory ───────────────────────────────────────────────────────────────────
def build_model(name: str, input_dim: int, num_classes: int, **kwargs) -> nn.Module:
    """Return an MLP or CNN-LSTM model by name string."""
    name = name.lower()
    if name == "mlp":
        return MLPModel(input_dim, num_classes, **kwargs)
    elif name in ("cnn_lstm", "cnnlstm"):
        return CNNLSTMModel(input_dim, num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model '{name}'. Choose 'mlp' or 'cnn_lstm'.")
