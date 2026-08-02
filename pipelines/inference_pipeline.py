import logging
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from config.settings import settings
from ensemble.super_stacking import SuperStackingEnsemble
from features.feature_engineering import FeatureEngineer
from evaluation.calibration import ProbabilityCalibrator
from monitoring.drift_detector import DriftDetector, DriftReport

logger = logging.getLogger(__name__)


class InferencePipeline:
    """Pipeline de inferencia con calibración y detección de drift."""

    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = models_dir or settings.MODELS_DIR
        self.ensemble = SuperStackingEnsemble()
        self.feature_engineer = FeatureEngineer()
        self.calibrator: Optional[ProbabilityCalibrator] = None
        self.drift_detector: Optional[DriftDetector] = None
        self._load()

    def _load(self) -> None:
        """Carga todos los artefactos."""
        ensemble_path = os.path.join(self.models_dir, "liga_mx_ensemble.joblib")
        fe_path = os.path.join(self.models_dir, "feature_engineer.joblib")
        calib_path = os.path.join(self.models_dir, "calibrator.joblib")
        drift_path = os.path.join(self.models_dir, "drift_detector.joblib")

        if not os.path.exists(ensemble_path):
            raise FileNotFoundError(f"Ensemble not found at {ensemble_path}")

        self.ensemble.load(ensemble_path)
        
        if os.path.exists(fe_path):
            self.feature_engineer = joblib.load(fe_path)
        
        if os.path.exists(calib_path):
            self.calibrator = joblib.load(calib_path)
            logger.info("✅ Calibrator loaded")
        
        if os.path.exists(drift_path):
            self.drift_detector = joblib.load(drift_path)
            logger.info("✅ Drift detector loaded")
        
        logger.info(f"✅ Models loaded from {self.models_dir}")

    def predict_matches(
        self,
        matches: pd.DataFrame,
        detect_drift: bool = True
    ) -> tuple[pd.DataFrame, Optional[DriftReport]]:
        """
        Predice partidos con calibración y detección de drift.
        
        Returns:
            (predictions_df, drift_report)
        """
        if matches.empty:
            return matches, None

        # Feature engineering
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

        # Drift detection
        drift_report = None
        if detect_drift and self.drift_detector is not None:
            try:
                drift_report = self.drift_detector.detect_drift(
                    X_prod=X,
                    y_prod=None,
                    proba_prod=None
                )
                logger.info(str(drift_report))
                
                if drift_report.severity in ['high', 'medium']:
                    logger.warning("⚠️  SIGNIFICANT DRIFT DETECTED - Consider retraining!")
            except Exception as e:
                logger.error(f"Drift detection failed: {e}")

        # Predictions
        proba = self.ensemble.predict_proba(X)

        # Calibration
        if self.calibrator is not None:
            proba = self.calibrator.transform(proba)
            logger.info("✅ Probabilities calibrated")

        # Format results
        result = matches[["match_id", "home_name", "away_name", "start_time"]].copy()
        result["prob_home"] = proba[:, 0]
        result["prob_draw"] = proba[:, 1]
        result["prob_away"] = proba[:, 2]
        result["prediction"] = np.argmax(proba, axis=1).map({0: "Home", 1: "Draw", 2: "Away"})
        result["confidence"] = proba.max(axis=1)
        
        # Drift flag
        if drift_report:
            result["drift_severity"] = drift_report.severity

        return result, drift_report
