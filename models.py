"""
models.py
PyTorch model definitions:
  - MLPModel      : Baseline fully-connected network (256 → 128 → 64)
  - ResMLPModel   : Wider MLP with residual skip connections (512 → 256 → 128)
  - CNNLSTMModel  : Hybrid 1D-Conv + LSTM model (optional self-attention)
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


# ── Residual MLP ─────────────────────────────────────────────────────────────
class ResMLPModel(nn.Module):
    """
    Wider MLP with residual skip connections.
    Architecture: Input → 512 → (512 residual) → (256 residual) → (128 residual) → num_classes
    Skip connections help gradient flow and allow deeper training without degradation.
    """

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        # Residual block 1: 512 → 512
        self.block1 = nn.Sequential(
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 512), nn.BatchNorm1d(512),
        )
        self.relu1 = nn.ReLU(inplace=True)
        # Residual block 2: 512 → 256
        self.block2 = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 256), nn.BatchNorm1d(256),
        )
        self.proj2  = nn.Linear(512, 256, bias=False)
        self.relu2  = nn.ReLU(inplace=True)
        # Residual block 3: 256 → 128
        self.block3 = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 128), nn.BatchNorm1d(128),
        )
        self.proj3  = nn.Linear(256, 128, bias=False)
        self.relu3  = nn.ReLU(inplace=True)

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.input_proj(x)

        residual = x
        x = self.relu1(self.block1(x) + residual)

        residual = self.proj2(x)
        x = self.relu2(self.block2(x) + residual)

        residual = self.proj3(x)
        x = self.relu3(self.block3(x) + residual)

        return self.classifier(x)


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
        use_attention: bool = False,
    ):
        super().__init__()
        self.input_dim    = input_dim
        self.use_attention = use_attention

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

        # ── LSTM (2 layers) ──────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        # ── Self-attention over LSTM timesteps ───────────────────────────────
        if use_attention:
            self.attn = nn.Linear(lstm_hidden, 1)

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
        out = x.unsqueeze(1)               # (B, 1, input_dim)
        out = self.conv1(out)              # (B, 64,  input_dim//2)
        out = self.conv2(out)              # (B, 128, input_dim//4)
        out = out.permute(0, 2, 1)        # (B, seq_len, 128)
        out, _ = self.lstm(out)            # (B, seq_len, lstm_hidden)
        if self.use_attention:
            # Weighted sum across timesteps instead of taking only last
            weights = torch.softmax(self.attn(out), dim=1)  # (B, seq_len, 1)
            out = (out * weights).sum(dim=1)                 # (B, lstm_hidden)
        else:
            out = out[:, -1, :]                              # (B, lstm_hidden)
        return self.classifier(out)        # (B, num_classes)


# ── Factory ───────────────────────────────────────────────────────────────────
def build_model(name: str, input_dim: int, num_classes: int, **kwargs) -> nn.Module:
    """Return a model by name string."""
    name = name.lower()
    if name == "mlp":
        return MLPModel(input_dim, num_classes, **kwargs)
    elif name == "res_mlp":
        return ResMLPModel(input_dim, num_classes, **kwargs)
    elif name in ("cnn_lstm", "cnnlstm"):
        return CNNLSTMModel(input_dim, num_classes, **kwargs)
    elif name == "cnn_lstm_attn":
        return CNNLSTMModel(input_dim, num_classes, use_attention=True, **kwargs)
    else:
        raise ValueError(f"Unknown model '{name}'. Choose 'mlp', 'res_mlp', 'cnn_lstm', or 'cnn_lstm_attn'.")
