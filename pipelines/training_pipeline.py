import logging
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from config.settings import settings
from data_loader.data_processor import DataProcessor
from features.feature_engineering import FeatureEngineer
from ensemble.super_stacking import SuperStackingEnsemble

logger = logging.getLogger(__name__)


@dataclass
class TrainingReport:
    n_matches: int
    accuracy: float
    log_loss: float
    models_path: str

    def to_dict(self) -> dict:
        return asdict(self)


class TrainingPipeline:
    """End-to-end training pipeline for Liga MX (iSportsAPI only)."""

    def __init__(
        self,
        competition_ids: Optional[List[str]] = None,
        min_season_year: int = 2019,
    ):
        self.competition_ids = competition_ids or [settings.LIGA_MX_LEAGUE_ID]
        self.min_season_year = min_season_year
        self.processor = DataProcessor()
        self.feature_engineer = FeatureEngineer()
        self.ensemble = SuperStackingEnsemble(
            n_splits=settings.N_TIME_SERIES_SPLITS,
            random_state=settings.RANDOM_SEED,
        )

    def run(self, max_matches: Optional[int] = None) -> TrainingReport:
        logger.info("=== Liga MX Training Pipeline started ===")

        # 1. Fetch data
        matches = self.processor.fetch_historical_matches(
            competition_ids=self.competition_ids,
            min_season_year=self.min_season_year,
            max_matches=max_matches,
        )
        if matches.empty:
            raise RuntimeError("No matches retrieved from iSportsAPI for Liga MX")

        # 2. Feature engineering
        featured = self.feature_engineer.transform(matches)
        X = self.feature_engineer.get_feature_matrix(featured)
        y = featured["result_1x2"]

        # 3. Train ensemble
        self.ensemble.fit(X, y, matches_df=matches)

        # 4. Quick evaluation on last 15%
        split = int(len(X) * 0.85)
        X_test, y_test = X.iloc[split:], y.iloc[split:]
        proba = self.ensemble.predict_proba(X_test)
        preds = np.argmax(proba, axis=1)
        acc = accuracy_score(y_test, preds)
        ll = log_loss(y_test, proba)

        # 5. Persist
        os.makedirs(settings.MODELS_DIR, exist_ok=True)
        model_path = os.path.join(settings.MODELS_DIR, "liga_mx_ensemble.joblib")
        self.ensemble.save(model_path)
        joblib.dump(self.feature_engineer, os.path.join(settings.MODELS_DIR, "feature_engineer.joblib"))

        report = TrainingReport(
            n_matches=len(matches),
            accuracy=float(acc),
            log_loss=float(ll),
            models_path=model_path,
        )
        logger.info("Training finished → Acc=%.3f | LogLoss=%.3f | saved to %s", acc, ll, model_path)
        return report
