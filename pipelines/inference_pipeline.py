import logging
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from config.settings import settings
from ensemble.super_stacking import SuperStackingEnsemble
from features.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class InferencePipeline:
    """Load trained ensemble and produce 1X2 probabilities for new Liga MX matches."""

    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = models_dir or settings.MODELS_DIR
        self.ensemble = SuperStackingEnsemble()
        self.feature_engineer = FeatureEngineer()
        self._load()

    def _load(self) -> None:
        ensemble_path = os.path.join(self.models_dir, "liga_mx_ensemble.joblib")
        fe_path = os.path.join(self.models_dir, "feature_engineer.joblib")

        if not os.path.exists(ensemble_path):
            raise FileNotFoundError(f"Ensemble not found at {ensemble_path}. Run training first.")

        self.ensemble.load(ensemble_path)
        if os.path.exists(fe_path):
            self.feature_engineer = joblib.load(fe_path)
        logger.info("Models loaded from %s", self.models_dir)

    def predict_matches(self, matches: pd.DataFrame) -> pd.DataFrame:
        if matches.empty:
            return matches

        # Minimal feature engineering for upcoming matches (no scores yet)
        featured = matches.copy()
        featured["home_score"] = 0
        featured["away_score"] = 0
        featured = self.feature_engineer.transform(featured)
        X = self.feature_engineer.get_feature_matrix(featured)

        # Align columns
        for col in self.ensemble.feature_names:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self.ensemble.feature_names]

        proba = self.ensemble.predict_proba(X)

        result = matches[["match_id", "home_name", "away_name", "start_time"]].copy()
        result["prob_home"] = proba[:, 0]
        result["prob_draw"] = proba[:, 1]
        result["prob_away"] = proba[:, 2]
        result["prediction"] = np.argmax(proba, axis=1).map({0: "Home", 1: "Draw", 2: "Away"})
        return result
