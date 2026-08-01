import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from .rolling_metrics import RollingMetricsCalculator
from config.settings import settings

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Full feature pipeline for Liga MX matches.
    Produces a clean feature matrix ready for gradient boosting / DNN.
    """

    def __init__(self, windows: Optional[List[int]] = None):
        self.windows = windows or settings.ROLLING_WINDOWS
        self.rolling = RollingMetricsCalculator(windows=self.windows)
        self.feature_columns_: List[str] = []

    def transform(self, matches: pd.DataFrame) -> pd.DataFrame:
        if matches.empty:
            return matches

        df = matches.copy()
        df = df.sort_values("start_time").reset_index(drop=True)

        # Basic derived columns
        df["total_goals"] = df["home_score"] + df["away_score"]
        df["goal_diff"] = df["home_score"] - df["away_score"]

        # Rolling features
        rolling_df = self.rolling.compute(df)
        df = df.merge(rolling_df, on="match_id", how="left")

        # Simple form features
        df["home_form_pts"] = df.get("home_pts_roll5", 0)
        df["away_form_pts"] = df.get("away_pts_roll5", 0)
        df["form_diff"] = df["home_form_pts"] - df["away_form_pts"]

        # Fill NaNs with 0 (new teams / early season)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        # Final feature list (exclude leakage columns)
        exclude = {
            "match_id", "season_id", "competition_id", "competition_name",
            "season_name", "start_time", "home_id", "away_id",
            "home_name", "away_name", "home_score", "away_score",
            "status", "match_status", "venue_id", "venue_name", "referee_id",
            "result_1x2", "total_goals", "goal_diff"
        }
        self.feature_columns_ = [c for c in df.columns if c not in exclude]
        return df

    def get_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.feature_columns_].copy()
