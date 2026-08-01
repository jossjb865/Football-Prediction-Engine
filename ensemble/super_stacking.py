import logging
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

from models.base_model import BaseMatchModel
from models.poisson_dixon_coles import DixonColesPoisson
from models.xgboost_model import XGBoostMatchModel
from models.catboost_model import CatBoostMatchModel
from models.dnn_model import DNNMatchModel

logger = logging.getLogger(__name__)


class SuperStackingEnsemble:
    """
    Two-level stacking ensemble:
    Level-0: Dixon-Coles, XGBoost, CatBoost, DNN
    Level-1: Logistic Regression on out-of-fold predictions
    """

    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.base_models: Dict[str, BaseMatchModel] = {}
        self.meta_model: Optional[LogisticRegression] = None
        self.feature_names: List[str] = []
        self.is_fitted = False

    def _init_base_models(self) -> Dict[str, BaseMatchModel]:
        return {
            "xgboost": XGBoostMatchModel(random_state=self.random_state),
            "catboost": CatBoostMatchModel(random_seed=self.random_state),
            "dnn": DNNMatchModel(random_state=self.random_state),
        }

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        matches_df: Optional[pd.DataFrame] = None,
    ) -> "SuperStackingEnsemble":
        self.feature_names = list(X.columns)
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        oof_preds = np.zeros((len(X), 3 * 3))  # 3 models × 3 classes
        model_names = ["xgboost", "catboost", "dnn"]

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info("Stacking fold %d/%d", fold + 1, self.n_splits)
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train = y.iloc[train_idx]

            fold_models = self._init_base_models()
            for name, model in fold_models.items():
                model.fit(X_train, y_train)

            for i, name in enumerate(model_names):
                preds = fold_models[name].predict_proba(X_val)
                oof_preds[val_idx, i * 3:(i + 1) * 3] = preds

        # Train meta-model on OOF predictions
        self.meta_model = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=1000,
            random_state=self.random_state,
        )
        # Only use rows that received OOF predictions
        mask = oof_preds.sum(axis=1) > 0
        self.meta_model.fit(oof_preds[mask], y[mask])

        # Retrain base models on full data
        self.base_models = self._init_base_models()
        for name, model in self.base_models.items():
            model.fit(X, y)

        # Optional Dixon-Coles (needs raw matches)
        if matches_df is not None and not matches_df.empty:
            dc = DixonColesPoisson()
            try:
                dc.fit_from_matches(matches_df)
                self.base_models["dixon_coles"] = dc
            except Exception as e:
                logger.warning("Dixon-Coles could not be fitted: %s", e)

        self.is_fitted = True
        logger.info("SuperStackingEnsemble fitted successfully")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted")

        model_names = ["xgboost", "catboost", "dnn"]
        base_preds = []
        for name in model_names:
            preds = self.base_models[name].predict_proba(X)
            base_preds.append(preds)

        stacked = np.hstack(base_preds)
        meta_probs = self.meta_model.predict_proba(stacked)
        return meta_probs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "base_models": self.base_models,
                "meta_model": self.meta_model,
                "feature_names": self.feature_names,
                "n_splits": self.n_splits,
                "random_state": self.random_state,
            },
            path,
        )

    def load(self, path: str) -> "SuperStackingEnsemble":
        data = joblib.load(path)
        self.base_models = data["base_models"]
        self.meta_model = data["meta_model"]
        self.feature_names = data["feature_names"]
        self.n_splits = data["n_splits"]
        self.random_state = data["random_state"]
        self.is_fitted = True
        return self
