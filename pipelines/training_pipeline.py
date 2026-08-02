import logging
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from config.settings import settings
from data_loader.data_processor import DataProcessor
from features.feature_engineering import FeatureEngineer
from ensemble.super_stacking import SuperStackingEnsemble
from evaluation.metrics_calculator import MetricsCalculator, ComprehensiveMetrics
from evaluation.calibration import ProbabilityCalibrator
from monitoring.drift_detector import DriftDetector
from mlops.experiment_tracker import ExperimentTracker

logger = logging.getLogger(__name__)


@dataclass
class TrainingReport:
    """Reporte extendido de entrenamiento."""
    n_matches: int
    comprehensive_metrics: Dict
    models_path: str
    calibration_applied: bool = False
    drift_baseline_saved: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class TrainingPipeline:
    """Pipeline de entrenamiento completo con mejores prácticas MLOps."""

    def __init__(
        self,
        competition_ids: Optional[List[str]] = None,
        min_season_year: int = 2019,
        use_mlflow: bool = False,
    ):
        self.competition_ids = competition_ids or [settings.LIGA_MX_LEAGUE_ID]
        self.min_season_year = min_season_year
        self.processor = DataProcessor()
        self.feature_engineer = FeatureEngineer()
        self.ensemble = SuperStackingEnsemble(
            n_splits=settings.N_TIME_SERIES_SPLITS,
            random_state=settings.RANDOM_SEED,
        )
        self.metrics_calculator = MetricsCalculator()
        self.calibrator = ProbabilityCalibrator(method='isotonic')
        self.drift_detector = DriftDetector()
        
        # Experiment tracking
        self.tracker = ExperimentTracker(
            experiment_name="football-liga-mx",
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI") if use_mlflow else None
        )

    def run(self, max_matches: Optional[int] = None) -> TrainingReport:
        logger.info("="*70)
        logger.info("🚀 STARTING ENHANCED TRAINING PIPELINE")
        logger.info("="*70)

        with self.tracker:
            # Log experiment params
            self.tracker.log_params({
                'min_season_year': self.min_season_year,
                'n_splits': self.ensemble.n_splits,
                'random_seed': settings.RANDOM_SEED,
                'rolling_windows': str(settings.ROLLING_WINDOWS),
            })

            # 1. Fetch data
            matches = self.processor.fetch_historical_matches(
                competition_ids=self.competition_ids,
                min_season_year=self.min_season_year,
                max_matches=max_matches,
            )
            if matches.empty:
                raise RuntimeError("No matches retrieved")

            # 2. Feature engineering
            featured = self.feature_engineer.transform(matches)
            X = self.feature_engineer.get_feature_matrix(featured)
            y = featured["result_1x2"]

            # 3. Train/test split temporal
            split = int(len(X) * 0.85)
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]

            logger.info(f"📊 Dataset: Train={len(X_train)} | Test={len(X_test)}")

            # 4. Train ensemble
            logger.info("🔧 Training ensemble...")
            self.ensemble.fit(X_train, y_train, matches_df=matches)

            # 5. Predictions
            proba_train = self.ensemble.predict_proba(X_train)
            proba_test_raw = self.ensemble.predict_proba(X_test)

            # 6. Calibration
            logger.info("⚖️  Calibrating probabilities...")
            self.calibrator.fit(y_train.values, proba_train)
            proba_test = self.calibrator.transform(proba_test_raw)
            
            # Save calibration plot
            calib_plot_path = os.path.join(settings.ARTIFACTS_DIR, "calibration_curve.png")
            self.calibrator.plot_calibration_curve(
                y_test.values, proba_test_raw, proba_test, save_path=calib_plot_path
            )
            self.tracker.log_artifact(calib_plot_path)

            # 7. Comprehensive evaluation
            logger.info("📈 Calculating comprehensive metrics...")
            metrics = self.metrics_calculator.calculate_comprehensive(
                y_test.values, proba_test, odds=None
            )
            
            logger.info(str(metrics))
            
            # Log metrics to tracker
            self.tracker.log_metrics({
                'accuracy': metrics.accuracy,
                'log_loss': metrics.log_loss,
                'brier_score': metrics.brier_score,
                'roc_auc_ovr': metrics.roc_auc_ovr,
                'f1_home': metrics.f1_home,
                'f1_draw': metrics.f1_draw,
                'f1_away': metrics.f1_away,
                'calibration_error': metrics.calibration_error or 0.0,
            })

            # 8. Drift baseline
            logger.info("📊 Establishing drift detection baseline...")
            self.drift_detector.fit_reference(X_train, y_train.values, proba_train)

            # 9. Persist everything
            os.makedirs(settings.MODELS_DIR, exist_ok=True)
            
            ensemble_path = os.path.join(settings.MODELS_DIR, "liga_mx_ensemble.joblib")
            self.ensemble.save(ensemble_path)
            
            joblib.dump(self.feature_engineer, os.path.join(settings.MODELS_DIR, "feature_engineer.joblib"))
            joblib.dump(self.calibrator, os.path.join(settings.MODELS_DIR, "calibrator.joblib"))
            joblib.dump(self.drift_detector, os.path.join(settings.MODELS_DIR, "drift_detector.joblib"))
            
            self.tracker.log_model(self.ensemble, "ensemble")

            report = TrainingReport(
                n_matches=len(matches),
                comprehensive_metrics=metrics.to_dict(),
                models_path=ensemble_path,
                calibration_applied=True,
                drift_baseline_saved=True,
            )

            logger.info("✅ Training pipeline completed successfully")
            return report
