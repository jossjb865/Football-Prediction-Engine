import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RollingMetricsCalculator:
    """
    Causal rolling features for football matches.
    All metrics are computed using only past information (shift(1)).
    """

    def __init__(self, windows: List[int] = None):
        self.windows = windows or [3, 5, 10, 20]

    def compute(self, matches: pd.DataFrame) -> pd.DataFrame:
        df = matches.sort_values("start_time").copy()
        df = df.reset_index(drop=True)

        home_stats = self._team_perspective(df, is_home=True)
        away_stats = self._team_perspective(df, is_home=False)

        home_stats = home_stats.add_prefix("home_")
        away_stats = away_stats.add_prefix("away_")

        result = pd.concat([df[["match_id"]], home_stats, away_stats], axis=1)
        return result

    def _team_perspective(self, df: pd.DataFrame, is_home: bool) -> pd.DataFrame:
        team_col = "home_id" if is_home else "away_id"
        opp_col = "away_id" if is_home else "home_id"
        goals_for = "home_score" if is_home else "away_score"
        goals_against = "away_score" if is_home else "home_score"

        records = []
        for team_id, group in df.groupby(team_col):
            g = group.sort_values("start_time").copy()
            g["gf"] = g[goals_for]
            g["ga"] = g[goals_against]
            g["gd"] = g["gf"] - g["ga"]
            g["win"] = (g["gf"] > g["ga"]).astype(float)
            g["draw"] = (g["gf"] == g["ga"]).astype(float)
            g["loss"] = (g["gf"] < g["ga"]).astype(float)
            g["points"] = g["win"] * 3 + g["draw"]

            for w in self.windows:
                g[f"gf_roll{w}"] = g["gf"].shift(1).rolling(w, min_periods=1).mean()
                g[f"ga_roll{w}"] = g["ga"].shift(1).rolling(w, min_periods=1).mean()
                g[f"gd_roll{w}"] = g["gd"].shift(1).rolling(w, min_periods=1).mean()
                g[f"pts_roll{w}"] = g["points"].shift(1).rolling(w, min_periods=1).mean()
                g[f"win_rate{w}"] = g["win"].shift(1).rolling(w, min_periods=1).mean()

            records.append(g)

        if not records:
            return pd.DataFrame()

        out = pd.concat(records).sort_index()
        feature_cols = [c for c in out.columns if any(x in c for x in ["gf_roll", "ga_roll", "gd_roll", "pts_roll", "win_rate"])]
        return out[feature_cols]
