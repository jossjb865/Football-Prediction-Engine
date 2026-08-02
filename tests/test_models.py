"""
Tests unitarios para modelos individuales y ensemble.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from models.xgboost_model import XGBoostMatchModel
from models.catboost_model import CatBoostMatchModel
from models.dnn_model import DNNMatchModel
from ensemble.super_stacking import SuperStackingEnsemble


@pytest.fixture
def sample_data():
    """Genera datos sintéticos para testing."""
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=15,
        n_classes=3,
        random_state=42
    )
    X_df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(20)])
    y_series = pd.Series(y)
    
    split = int(0.8 * len(X_df))
    return {
        'X_train': X_df[:split],
        'y_train': y_series[:split],
        'X_test': X_df[split:],
        'y_test': y_series[split:]
    }


class TestXGBoostModel:
    def test_fit_and_predict(self, sample_data):
        model = XGBoostMatchModel(n_estimators=50)
        model.fit(sample_data['X_train'], sample_data['y_train'])
        proba = model.predict_proba(sample_data['X_test'])
        
        assert proba.shape == (len(sample_data['X_test']), 3)
        assert np.allclose(proba.sum(axis=1), 1.0)
    
    def test_save_load(self, sample_data, tmp_path):
        model = XGBoostMatchModel(n_estimators=50)
        model.fit(sample_data['X_train'], sample_data['y_train'])
        
        save_path = tmp_path / "xgb_test.json"
        model.save(str(save_path))
        
        loaded_model = XGBoostMatchModel()
        loaded_model.load(str(save_path))
        
        proba_original = model.predict_proba(sample_data['X_test'])
        proba_loaded = loaded_model.predict_proba(sample_data['X_test'])
        
        assert np.allclose(proba_original, proba_loaded)


class TestCatBoostModel:
    def test_fit_and_predict(self, sample_data):
        model = CatBoostMatchModel(params={'iterations': 50, 'verbose': False})
        model.fit(sample_data['X_train'], sample_data['y_train'])
        proba = model.predict_proba(sample_data['X_test'])
        
        assert proba.shape == (len(sample_data['X_test']), 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


class TestDNNModel:
    def test_fit_and_predict(self, sample_data):
        model = DNNMatchModel(epochs=5, batch_size=32)
        model.fit(sample_data['X_train'], sample_data['y_train'])
        proba = model.predict_proba(sample_data['X_test'])
        
        assert proba.shape == (len(sample_data['X_test']), 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


class TestSuperStackingEnsemble:
    def test_ensemble_predictions_sum_to_one(self, sample_data):
        ensemble = SuperStackingEnsemble(n_splits=3, random_state=42)
        ensemble.fit(
            sample_data['X_train'],
            sample_data['y_train'],
            matches_df=None  # Skip Dixon-Coles
        )
        
        proba = ensemble.predict_proba(sample_data['X_test'])
        
        assert proba.shape == (len(sample_data['X_test']), 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    
    def test_ensemble_better_than_individual(self, sample_data):
        """Verificar que ensemble no empeora performance."""
        from sklearn.metrics import log_loss
        
        # Entrenar modelo individual
        xgb = XGBoostMatchModel(n_estimators=50, random_state=42)
        xgb.fit(sample_data['X_train'], sample_data['y_train'])
        proba_xgb = xgb.predict_proba(sample_data['X_test'])
        ll_xgb = log_loss(sample_data['y_test'], proba_xgb)
        
        # Entrenar ensemble
        ensemble = SuperStackingEnsemble(n_splits=3, random_state=42)
        ensemble.fit(sample_data['X_train'], sample_data['y_train'])
        proba_ensemble = ensemble.predict_proba(sample_data['X_test'])
        ll_ensemble = log_loss(sample_data['y_test'], proba_ensemble)
        
        # Ensemble debe ser al menos comparable
        assert ll_ensemble <= ll_xgb * 1.1  # Permitir 10% peor (por splits pequeños)
    
    def test_save_load_ensemble(self, sample_data, tmp_path):
        ensemble = SuperStackingEnsemble(n_splits=3, random_state=42)
        ensemble.fit(sample_data['X_train'], sample_data['y_train'])
        
        save_path = tmp_path / "ensemble_test.joblib"
        ensemble.save(str(save_path))
        
        loaded = SuperStackingEnsemble()
        loaded.load(str(save_path))
        
        proba_original = ensemble.predict_proba(sample_data['X_test'])
        proba_loaded = loaded.predict_proba(sample_data['X_test'])
        
        assert np.allclose(proba_original, proba_loaded, atol=1e-5)
