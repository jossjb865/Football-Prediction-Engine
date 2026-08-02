"""
Evaluación comprehensiva de modelos de predicción de fútbol.
Incluye métricas estándar, de negocio (EV) y análisis por clase.
"""

import logging
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)


@dataclass
class ComprehensiveMetrics:
    """Reporte completo de métricas del modelo."""
    accuracy: float
    log_loss: float
    brier_score: float
    roc_auc_ovr: float
    roc_auc_ovo: float
    
    # Métricas por clase
    precision_home: float
    precision_draw: float
    precision_away: float
    recall_home: float
    recall_draw: float
    recall_away: float
    f1_home: float
    f1_draw: float
    f1_away: float
    
    # Métricas de negocio
    expected_value: Optional[float] = None
    kelly_criterion: Optional[float] = None
    roi: Optional[float] = None
    
    # Diagnóstico
    calibration_error: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def __str__(self) -> str:
        return (
            f"\n{'='*60}\n"
            f"📊 COMPREHENSIVE MODEL EVALUATION\n"
            f"{'='*60}\n"
            f"🎯 Overall Metrics:\n"
            f"   Accuracy:        {self.accuracy:.4f}\n"
            f"   Log Loss:        {self.log_loss:.4f}\n"
            f"   Brier Score:     {self.brier_score:.4f}\n"
            f"   ROC-AUC (OvR):   {self.roc_auc_ovr:.4f}\n"
            f"   ROC-AUC (OvO):   {self.roc_auc_ovo:.4f}\n"
            f"\n📈 Per-Class Performance:\n"
            f"   HOME: Prec={self.precision_home:.3f} | Rec={self.recall_home:.3f} | F1={self.f1_home:.3f}\n"
            f"   DRAW: Prec={self.precision_draw:.3f} | Rec={self.recall_draw:.3f} | F1={self.f1_draw:.3f}\n"
            f"   AWAY: Prec={self.precision_away:.3f} | Rec={self.recall_away:.3f} | F1={self.f1_away:.3f}\n"
            f"\n💰 Business Metrics:\n"
            f"   Expected Value:  {self.expected_value if self.expected_value else 'N/A'}\n"
            f"   ROI:             {self.roi if self.roi else 'N/A'}\n"
            f"\n🔧 Calibration Error: {self.calibration_error if self.calibration_error else 'N/A'}\n"
            f"{'='*60}\n"
        )


class MetricsCalculator:
    """Calculadora avanzada de métricas para sistemas de predicción."""
    
    def __init__(self):
        self.class_labels = ["Home", "Draw", "Away"]
    
    def calculate_comprehensive(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        odds: Optional[pd.DataFrame] = None,
    ) -> ComprehensiveMetrics:
        """
        Calcula todas las métricas del sistema.
        
        Args:
            y_true: Ground truth labels (0=Home, 1=Draw, 2=Away)
            y_proba: Predicted probabilities shape (n_samples, 3)
            odds: DataFrame con columnas ['home_odds', 'draw_odds', 'away_odds'] (opcional)
        """
        y_pred = np.argmax(y_proba, axis=1)
        
        # Métricas básicas
        acc = accuracy_score(y_true, y_pred)
        ll = log_loss(y_true, y_proba)
        
        # Brier Score (promedio de las 3 clases)
        brier_scores = []
        for i in range(3):
            y_true_binary = (y_true == i).astype(int)
            brier_scores.append(brier_score_loss(y_true_binary, y_proba[:, i]))
        brier_avg = np.mean(brier_scores)
        
        # ROC-AUC
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
        try:
            roc_ovr = roc_auc_score(y_true_bin, y_proba, multi_class='ovr', average='weighted')
            roc_ovo = roc_auc_score(y_true_bin, y_proba, multi_class='ovo', average='weighted')
        except Exception as e:
            logger.warning(f"Could not calculate ROC-AUC: {e}")
            roc_ovr = roc_ovo = 0.0
        
        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_labels,
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred).tolist()
        
        # Calibration error
        calib_error = self._calculate_calibration_error(y_true, y_proba)
        
        # Business metrics
        ev, roi, kelly = None, None, None
        if odds is not None:
            ev, roi, kelly = self._calculate_business_metrics(y_true, y_proba, odds)
        
        return ComprehensiveMetrics(
            accuracy=acc,
            log_loss=ll,
            brier_score=brier_avg,
            roc_auc_ovr=roc_ovr,
            roc_auc_ovo=roc_ovo,
            precision_home=report['Home']['precision'],
            precision_draw=report['Draw']['precision'],
            precision_away=report['Away']['precision'],
            recall_home=report['Home']['recall'],
            recall_draw=report['Draw']['recall'],
            recall_away=report['Away']['recall'],
            f1_home=report['Home']['f1-score'],
            f1_draw=report['Draw']['f1-score'],
            f1_away=report['Away']['f1-score'],
            expected_value=ev,
            roi=roi,
            kelly_criterion=kelly,
            calibration_error=calib_error,
            confusion_matrix=cm,
        )
    
    def _calculate_calibration_error(self, y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
        """Expected Calibration Error (ECE)."""
        ece = 0.0
        for class_idx in range(3):
            y_true_binary = (y_true == class_idx).astype(int)
            probs = y_proba[:, class_idx]
            
            bins = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(probs, bins) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            
            for b in range(n_bins):
                mask = bin_indices == b
                if mask.sum() > 0:
                    avg_confidence = probs[mask].mean()
                    avg_accuracy = y_true_binary[mask].mean()
                    ece += mask.sum() * abs(avg_confidence - avg_accuracy)
        
        return ece / len(y_true)
    
    def _calculate_business_metrics(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        odds: pd.DataFrame
    ) -> tuple:
        """
        Calcula Expected Value (EV), ROI y Kelly Criterion.
        
        Args:
            odds: DataFrame con ['home_odds', 'draw_odds', 'away_odds']
        """
        if odds is None or len(odds) != len(y_true):
            return None, None, None
        
        odds_array = odds[['home_odds', 'draw_odds', 'away_odds']].values
        
        # Evitar odds inválidas
        odds_array = np.where(odds_array <= 1.01, 1.01, odds_array)
        
        # Probabilidades implícitas del mercado
        market_probs = 1.0 / odds_array
        market_probs /= market_probs.sum(axis=1, keepdims=True)
        
        # Expected Value
        ev = np.sum((y_proba - market_probs) * (odds_array - 1))
        
        # ROI (simulación de apostar siempre a la clase con mayor EV)
        stake = 1.0
        total_profit = 0.0
        for i in range(len(y_true)):
            best_bet = np.argmax(y_proba[i] * odds_array[i])
            if y_true[i] == best_bet:
                total_profit += stake * (odds_array[i, best_bet] - 1)
            else:
                total_profit -= stake
        
        roi = (total_profit / (len(y_true) * stake)) * 100
        
        # Kelly Criterion (fracción óptima para apostar)
        kelly_fractions = []
        for i in range(len(y_true)):
            for c in range(3):
                edge = y_proba[i, c] - market_probs[i, c]
                kelly = edge / (odds_array[i, c] - 1) if odds_array[i, c] > 1.01 else 0
                kelly_fractions.append(max(0, kelly))
        
        avg_kelly = np.mean(kelly_fractions) if kelly_fractions else 0.0
        
        return float(ev), float(roi), float(avg_kelly)
    
    def generate_detailed_report(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save_path: Optional[str] = None
    ) -> str:
        """Genera reporte textual detallado."""
        metrics = self.calculate_comprehensive(y_true, y_proba)
        report = str(metrics)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Detailed report saved to {save_path}")
        
        return report
