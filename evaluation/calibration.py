"""
Calibración de probabilidades usando Isotonic Regression y Platt Scaling.
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """Calibrador de probabilidades para modelos multiclase."""
    
    def __init__(self, method: Literal['isotonic', 'sigmoid'] = 'isotonic'):
        """
        Args:
            method: 'isotonic' (Isotonic Regression) o 'sigmoid' (Platt Scaling)
        """
        self.method = method
        self.calibrators = []  # Uno por clase
        self.is_fitted = False
    
    def fit(self, y_true: np.ndarray, y_proba: np.ndarray):
        """
        Entrena calibradores para cada clase.
        
        Args:
            y_true: Ground truth (0, 1, 2)
            y_proba: Probabilidades sin calibrar (n_samples, 3)
        """
        self.calibrators = []
        
        for class_idx in range(3):
            y_true_binary = (y_true == class_idx).astype(int)
            probs = y_proba[:, class_idx]
            
            if self.method == 'isotonic':
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(probs, y_true_binary)
            else:  # sigmoid
                from sklearn.linear_model import LogisticRegression
                calibrator = LogisticRegression()
                calibrator.fit(probs.reshape(-1, 1), y_true_binary)
            
            self.calibrators.append(calibrator)
        
        self.is_fitted = True
        logger.info(f"Probability calibrators fitted using {self.method}")
        return self
    
    def transform(self, y_proba: np.ndarray) -> np.ndarray:
        """
        Aplica calibración a nuevas probabilidades.
        
        Returns:
            Probabilidades calibradas (n_samples, 3)
        """
        if not self.is_fitted:
            raise RuntimeError("Calibrator not fitted")
        
        calibrated = np.zeros_like(y_proba)
        
        for class_idx, calibrator in enumerate(self.calibrators):
            probs = y_proba[:, class_idx]
            
            if self.method == 'isotonic':
                calibrated[:, class_idx] = calibrator.predict(probs)
            else:
                calibrated[:, class_idx] = calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]
        
        # Normalizar para que sumen 1
        calibrated /= calibrated.sum(axis=1, keepdims=True)
        
        return calibrated
    
    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_proba_before: np.ndarray,
        y_proba_after: np.ndarray,
        save_path: str = "calibration_curve.png"
    ):
        """Genera gráficos de calibración antes/después."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        class_names = ['Home', 'Draw', 'Away']
        
        for idx, (ax, name) in enumerate(zip(axes, class_names)):
            y_true_binary = (y_true == idx).astype(int)
            
            # Before calibration
            self._plot_reliability_diagram(
                ax, y_true_binary, y_proba_before[:, idx],
                label='Before', color='red', alpha=0.5
            )
            
            # After calibration
            self._plot_reliability_diagram(
                ax, y_true_binary, y_proba_after[:, idx],
                label='After', color='blue', alpha=0.5
            )
            
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Fraction of Positives')
            ax.set_title(f'Calibration Curve - {name}')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        logger.info(f"Calibration curve saved to {save_path}")
        plt.close()
    
    def _plot_reliability_diagram(self, ax, y_true, y_prob, n_bins=10, **kwargs):
        """Helper para graficar diagrama de confiabilidad."""
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        bin_indices = np.digitize(y_prob, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        fraction_positives = []
        mean_predicted = []
        
        for b in range(n_bins):
            mask = bin_indices == b
            if mask.sum() > 0:
                fraction_positives.append(y_true[mask].mean())
                mean_predicted.append(y_prob[mask].mean())
        
        ax.plot(mean_predicted, fraction_positives, 'o-', **kwargs)
