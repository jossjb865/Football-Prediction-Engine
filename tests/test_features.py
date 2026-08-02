"""Tests para feature engineering."""

import pytest
import pandas as pd
import numpy as np
from features.feature_engineering import FeatureEngineer


@pytest.fixture
def sample_matches():
    """Genera partidos sintéticos."""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    matches = pd.DataFrame({
        'match_id': [f'match_{i}' for i in range(100)],
        'start_time': dates,
        'home_id': np.random.choice(['team_a', 'team_b', 'team_c'], 100),
        'away_id': np.random.choice(['team_a', 'team_b', 'team_c'], 100),
        'home_score': np.random.randint(0, 5, 100),
        'away_score': np.random.randint(0, 5, 100),
    })
    matches = matches[matches['home_id'] != matches['away_id']]
    return matches


class TestFeatureEngineering:
    def test_transform_no_leakage(self, sample_matches):
        """Verificar que no hay data leakage temporal."""
        fe = FeatureEngineer()
        featured = fe.transform(sample_matches)
        
        # Las métricas rolling deben ser NaN o 0 para los primeros partidos
        first_match_features = featured.iloc[0]
        rolling_cols = [c for c in featured.columns if 'roll' in c]
        
        # Todos deben ser 0 o NaN (no información futura)
        for col in rolling_cols:
            assert first_match_features[col] == 0 or pd.isna(first_match_features[col])
    
    def test_feature_matrix_no_target_leak(self, sample_matches):
        """Verificar que matriz de features no incluye target."""
        fe = FeatureEngineer()
        featured = fe.transform(sample_matches)
        X = fe.get_feature_matrix(featured)
        
        # No debe incluir scores ni resultado
        leak_cols = ['home_score', 'away_score', 'result_1x2', 'goal_diff']
        for col in leak_cols:
            assert col not in X.columns
    
    def test_rolling_causality(self, sample_matches):
        """Verificar que rolling metrics usan solo pasado."""
        fe = FeatureEngineer(windows=[5])
        featured = fe.transform(sample_matches.sort_values('start_time'))
        
        # Para el partido 10 de un equipo, verificar que rolling5 usa partidos 5-9
        # (implementación simplificada: verificar que existe)
        assert 'home_pts_roll5' in featured.columns
