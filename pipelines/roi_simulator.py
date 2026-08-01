import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ROISimulator:
    """Simple Kelly-criterion bankroll simulator for model evaluation."""

    def __init__(self, initial_bankroll: float = 1000.0, fraction: float = 0.25):
        self.initial_bankroll = initial_bankroll
        self.fraction = fraction  # fractional Kelly

    def simulate(
        self,
        predictions: pd.DataFrame,
        odds: pd.DataFrame,
        results: pd.Series,
    ) -> dict:
        """
        predictions: columns prob_home, prob_draw, prob_away
        odds: columns odds_home, odds_draw, odds_away
        results: 0=home, 1=draw, 2=away
        """
        bankroll = self.initial_bankroll
        history = []

        for i in range(len(predictions)):
            p = predictions.iloc[i][["prob_home", "prob_draw", "prob_away"]].values
            o = odds.iloc[i][["odds_home", "odds_draw", "odds_away"]].values
            true = int(results.iloc[i])

            # Expected value
            ev = p * o - 1
            best = int(np.argmax(ev))
            if ev[best] <= 0:
                history.append(bankroll)
                continue

            # Fractional Kelly
            kelly = (p[best] * o[best] - 1) / (o[best] - 1)
            stake = bankroll * self.fraction * max(kelly, 0)
            stake = min(stake, bankroll * 0.05)  # hard cap 5%

            if true == best:
                bankroll += stake * (o[best] - 1)
            else:
                bankroll -= stake

            history.append(bankroll)

        return {
            "final_bankroll": bankroll,
            "roi": (bankroll - self.initial_bankroll) / self.initial_bankroll,
            "history": history,
        }
