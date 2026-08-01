import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .base_model import BaseMatchModel

logger = logging.getLogger(__name__)


class _LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)


class LSTMMatchModel(BaseMatchModel):
    """LSTM model that consumes sequences of past matches."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        lr: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 30,
        device: Optional[str] = None,
        random_state: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.random_state = random_state
        self.model: Optional[_LSTMClassifier] = None
        self.input_dim: int = 0

    def fit_sequences(self, X_seq: np.ndarray, y: np.ndarray) -> "LSTMMatchModel":
        torch.manual_seed(self.random_state)
        self.input_dim = X_seq.shape[2]
        dataset = TensorDataset(
            torch.from_numpy(X_seq).float(),
            torch.from_numpy(y).long(),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = _LSTMClassifier(self.input_dim, self.hidden_dim, self.num_layers, self.dropout).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(xb)
            if (epoch + 1) % 10 == 0:
                logger.info("LSTM epoch %d/%d - loss %.4f", epoch + 1, self.epochs, total_loss / len(dataset))
        return self

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight=None):
        raise NotImplementedError("Use fit_sequences() with sequence data")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("Use predict_proba_sequences()")

    def predict_proba_sequences(self, X_seq: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted")
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X_seq).float().to(self.device))
            return torch.softmax(logits, dim=1).cpu().numpy()

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
            },
            path,
        )

    def load(self, path: str) -> "LSTMMatchModel":
        ckpt = torch.load(path, map_location=self.device)
        self.input_dim = ckpt["input_dim"]
        self.hidden_dim = ckpt["hidden_dim"]
        self.num_layers = ckpt["num_layers"]
        self.dropout = ckpt["dropout"]
        self.model = _LSTMClassifier(self.input_dim, self.hidden_dim, self.num_layers, self.dropout).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        return self
