"""
Análisis de explicabilidad usando SHAP values.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelExplainer:
    """Explicabilidad de modelos usando SHAP."""
    
    def __init__(self, model, model_type: str = 'tree'):
        """
        Args:
            model: Modelo entrenado (XGBoost, CatBoost, etc)
            model_type: 'tree' para tree-based, 'deep' para DNN
        """
        if not SHAP_AVAILABLE:
            raise ImportError("shap library not installed. Run: pip install shap")
        
        self.model = model
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None
    
    def fit(self, X: pd.DataFrame):
        """Crea el explainer."""
        if self.model_type == 'tree':
            self.explainer = shap.TreeExplainer(self.model.model)
        else:
            self.explainer = shap.DeepExplainer(self.model.model, X.values)
        
        logger.info(f"SHAP explainer created ({self.model_type})")
    
    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Calcula SHAP values."""
        if self.explainer is None:
            raise RuntimeError("Must call fit() first")
        
        self.shap_values = self.explainer.shap_values(X)
        return self.shap_values
    
    def plot_summary(self, X: pd.DataFrame, save_path: str = "shap_summary.png"):
        """Genera summary plot."""
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, X, show=False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"SHAP summary plot saved to {save_path}")
        plt.close()
    
    def plot_feature_importance(self, X: pd.DataFrame, save_path: str = "shap_importance.png"):
        """Feature importance basado en SHAP."""
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        # Para multiclase, promediar importancia sobre clases
        if isinstance(self.shap_values, list):
            mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in self.shap_values], axis=0)
        else:
            mean_shap = np.abs(self.shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': mean_shap
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'][:20], importance_df['importance'][:20])
        plt.xlabel('Mean |SHAP value|')
        plt.title('Top 20 Feature Importance (SHAP)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"SHAP importance plot saved to {save_path}")
        plt.close()
        
        return importance_df
    
    def explain_prediction(self, X_sample: pd.DataFrame, save_path: Optional[str] = None):
        """Explica una predicción individual."""
        if self.shap_values is None:
            self.compute_shap_values(X_sample)
        
        shap.force_plot(
            self.explainer.expected_value[0],
            self.shap_values[0][0],
            X_sample.iloc[0],
            matplotlib=True,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"SHAP force plot saved to {save_path}")
            plt.close()
