"""
Lightweight momentum LSTM that focuses on recent form sequences.
Same interface as LSTMMatchModel but with smaller architecture.
"""

from .lstm_model import LSTMMatchModel


class LSTMMomentumModel(LSTMMatchModel):
    def __init__(self, **kwargs):
        defaults = {
            "hidden_dim": 32,
            "num_layers": 1,
            "dropout": 0.2,
            "epochs": 25,
        }
        defaults.update(kwargs)
        super().__init__(**defaults)
