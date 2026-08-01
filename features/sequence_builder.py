import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SequenceBuilder:
    """
    Builds fixed-length sequences of past matches for LSTM models.
    Strictly causal: only previous matches of the same team are used.
    """

    def __init__(self, sequence_length: int = 10, feature_cols: Optional[List[str]] = None):
        self.sequence_length = sequence_length
        self.feature_cols = feature_cols or [
            "gf", "ga", "gd", "points", "is_home"
        ]

    def build(self, matches: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        df = matches.sort_values("start_time").copy()
        sequences = []
        targets = []
        match_ids = []

        for team_id in pd.unique(df[["home_id", "away_id"]].values.ravel("K")):
            team_matches = []
            for _, row in df.iterrows():
                if row["home_id"] == team_id:
                    team_matches.append({
                        "match_id": row["match_id"],
                        "start_time": row["start_time"],
                        "gf": row["home_score"],
                        "ga": row["away_score"],
                        "gd": row["home_score"] - row["away_score"],
                        "points": 3 if row["home_score"] > row["away_score"] else (1 if row["home_score"] == row["away_score"] else 0),
                        "is_home": 1.0,
                        "result": 0 if row["home_score"] > row["away_score"] else (1 if row["home_score"] == row["away_score"] else 2),
                    })
                elif row["away_id"] == team_id:
                    team_matches.append({
                        "match_id": row["match_id"],
                        "start_time": row["start_time"],
                        "gf": row["away_score"],
                        "ga": row["home_score"],
                        "gd": row["away_score"] - row["home_score"],
                        "points": 3 if row["away_score"] > row["home_score"] else (1 if row["away_score"] == row["home_score"] else 0),
                        "is_home": 0.0,
                        "result": 2 if row["away_score"] > row["home_score"] else (1 if row["away_score"] == row["home_score"] else 0),
                    })

            team_df = pd.DataFrame(team_matches).sort_values("start_time").reset_index(drop=True)
            if len(team_df) < self.sequence_length + 1:
                continue

            for i in range(self.sequence_length, len(team_df)):
                seq = team_df.iloc[i - self.sequence_length:i][self.feature_cols].values.astype(np.float32)
                target = team_df.iloc[i]["result"]
                sequences.append(seq)
                targets.append(target)
                match_ids.append(team_df.iloc[i]["match_id"])

        if not sequences:
            return np.array([]), np.array([]), []

        return np.stack(sequences), np.array(targets), match_ids
