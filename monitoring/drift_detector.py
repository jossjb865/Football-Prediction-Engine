"""
Detección de data drift y concept drift para modelos en producción.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Reporte de drift detectado."""
    has_feature_drift: bool
    has_prediction_drift: bool
    has_target_drift: bool
    drifted_features: List[str]
    drift_scores: Dict[str, float]
    severity: str  # 'none', 'low', 'medium', 'high'
    
    def __str__(self):
        return (
            f"\n🚨 DRIFT DETECTION REPORT\n"
            f"{'='*50}\n"
            f"Feature Drift:     {'⚠️ DETECTED' if self.has_feature_drift else '✅ OK'}\n"
            f"Prediction Drift:  {'⚠️ DETECTED' if self.has_prediction_drift else '✅ OK'}\n"
            f"Target Drift:      {'⚠️ DETECTED' if self.has_target_drift else '✅ OK'}\n"
            f"Severity:          {self.severity.upper()}\n"
            f"\nDrifted Features: {', '.join(self.drifted_features) if self.drifted_features else 'None'}\n"
            f"{'='*50}\n"
        )


class DriftDetector:
    """
    Detector de drift multimodal:
    - Feature drift (distribuciones de entrada)
    - Prediction drift (distribuciones de salida)
    - Target drift (cambios en etiquetas reales)
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Args:
            significance_level: Umbral p-value para Kolmogorov-Smirnov test
        """
        self.significance_level = significance_level
        self.reference_stats: Dict = {}
    
    def fit_reference(self, X_ref: pd.DataFrame, y_ref: np.ndarray, proba_ref: np.ndarray):
        """
        Establece distribuciones de referencia (datos de entrenamiento).
        
        Args:
            X_ref: Features de referencia
            y_ref: Target de referencia
            proba_ref: Probabilidades predichas de referencia
        """
        self.reference_stats = {
            'X_mean': X_ref.mean().to_dict(),
            'X_std': X_ref.std().to_dict(),
            'y_distribution': np.bincount(y_ref, minlength=3) / len(y_ref),
            'proba_mean': proba_ref.mean(axis=0),
            'proba_std': proba_ref.std(axis=0),
            'feature_distributions': {col: X_ref[col].values for col in X_ref.columns}
        }
        logger.info("Reference distributions fitted")
    
    def detect_drift(
        self,
        X_prod: pd.DataFrame,
        y_prod: Optional[np.ndarray] = None,
        proba_prod: Optional[np.ndarray] = None
    ) -> DriftReport:
        """
        Detecta drift comparando datos de producción con referencia.
        
        Returns:
            DriftReport con diagnóstico completo
        """
        if not self.reference_stats:
            raise RuntimeError("Must call fit_reference() first")
        
        drifted_features = []
        drift_scores = {}
        
        # 1. Feature drift (Kolmogorov-Smirnov test)
        for col in X_prod.columns:
            if col not in self.reference_stats['feature_distributions']:
                continue
            
            ref_dist = self.reference_stats['feature_distributions'][col]
            prod_dist = X_prod[col].values
            
            stat, p_value = ks_2samp(ref_dist, prod_dist)
            drift_scores[col] = p_value
            
            if p_value < self.significance_level:
                drifted_features.append(col)
                logger.warning(f"Feature drift detected in '{col}' (p={p_value:.4f})")
        
        has_feature_drift = len(drifted_features) > 0
        
        # 2. Prediction drift
        has_prediction_drift = False
        if proba_prod is not None:
            prod_proba_mean = proba_prod.mean(axis=0)
            ref_proba_mean = self.reference_stats['proba_mean']
            
            # MSE entre distribuciones de probabilidades
            proba_mse = mean_squared_error(ref_proba_mean, prod_proba_mean)
            drift_scores['prediction_distribution_mse'] = proba_mse
            
            if proba_mse > 0.01:  # Umbral arbitrario
                has_prediction_drift = True
                logger.warning(f"Prediction drift detected (MSE={proba_mse:.4f})")
        
        # 3. Target drift (Chi-square test)
        has_target_drift = False
        if y_prod is not None:
            prod_y_dist = np.bincount(y_prod, minlength=3) / len(y_prod)
            ref_y_dist = self.reference_stats['y_distribution']
            
            # Chi-square test
            observed = np.bincount(y_prod, minlength=3)
            expected = ref_y_dist * len(y_prod)
            
            chi2, p_value = chi2_contingency([observed, expected])[:2]
            drift_scores['target_distribution_chi2_p'] = p_value
            
            if p_value < self.significance_level:
                has_target_drift = True
                logger.warning(f"Target drift detected (p={p_value:.4f})")
        
        # Determinar severidad
        severity = self._calculate_severity(
            has_feature_drift, has_prediction_drift, has_target_drift, len(drifted_features), len(X_prod.columns)
        )
        
        return DriftReport(
            has_feature_drift=has_feature_drift,
            has_prediction_drift=has_prediction_drift,
            has_target_drift=has_target_drift,
            drifted_features=drifted_features,
            drift_scores=drift_scores,
            severity=severity
        )
    
    def _calculate_severity(
        self, feat_drift: bool, pred_drift: bool, tgt_drift: bool,
        n_drifted: int, n_total: int
    ) -> str:
        """Calcula nivel de severidad del drift."""
        score = 0
        if feat_drift:
            score += (n_drifted / n_total) * 3
        if pred_drift:
            score += 2
        if tgt_drift:
            score += 3
        
        if score == 0:
            return 'none'
        elif score < 2:
            return 'low'
        elif score < 5:
            return 'medium'
        else:
            return 'high'
