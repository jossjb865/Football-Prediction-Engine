"""
Tracking de experimentos compatible con MLflow.
"""

import logging
import os
import json
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Wrapper para tracking de experimentos.
    Usa MLflow si está disponible, sino registra localmente.
    """
    
    def __init__(self, experiment_name: str = "football-prediction", tracking_uri: Optional[str] = None):
        self.experiment_name = experiment_name
        self.local_mode = not MLFLOW_AVAILABLE or tracking_uri is None
        
        if self.local_mode:
            self.log_dir = Path("artifacts/experiments")
            self.log_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Experiment tracking in LOCAL mode")
        else:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow tracking enabled: {tracking_uri}")
    
    def start_run(self, run_name: Optional[str] = None):
        """Inicia un run de experimento."""
        if self.local_mode:
            self.current_run = {
                'run_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'run_name': run_name or 'unnamed_run',
                'params': {},
                'metrics': {},
                'artifacts': []
            }
        else:
            mlflow.start_run(run_name=run_name)
    
    def log_params(self, params: Dict[str, Any]):
        """Registra hiperparámetros."""
        if self.local_mode:
            self.current_run['params'].update(params)
        else:
            mlflow.log_params(params)
        
        logger.info(f"Logged params: {params}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Registra métricas."""
        if self.local_mode:
            self.current_run['metrics'].update(metrics)
        else:
            mlflow.log_metrics(metrics, step=step)
        
        logger.info(f"Logged metrics: {metrics}")
    
    def log_artifact(self, artifact_path: str):
        """Registra un artefacto (modelo, gráfico, etc)."""
        if self.local_mode:
            self.current_run['artifacts'].append(artifact_path)
        else:
            mlflow.log_artifact(artifact_path)
        
        logger.info(f"Logged artifact: {artifact_path}")
    
    def log_model(self, model, artifact_path: str):
        """Registra un modelo entrenado."""
        if self.local_mode:
            # Guardar referencia
            self.current_run['artifacts'].append(f"model:{artifact_path}")
        else:
            mlflow.sklearn.log_model(model, artifact_path)
    
    def end_run(self):
        """Finaliza el run."""
        if self.local_mode:
            # Guardar JSON local
            run_file = self.log_dir / f"{self.current_run['run_id']}.json"
            with open(run_file, 'w') as f:
                json.dump(self.current_run, f, indent=2)
            logger.info(f"Run saved to {run_file}")
        else:
            mlflow.end_run()
    
    def __enter__(self):
        self.start_run()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run()
