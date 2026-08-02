#!/usr/bin/env python3
"""
Entry point mejorado con tracking completo.
"""

import argparse
import logging
import sys

from config.logging_config import setup_logging
from config.settings import settings
from pipelines.training_pipeline import TrainingPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Liga MX Prediction Engine – Enhanced Training")
    parser.add_argument("--min-year", type=int, default=2019, help="Earliest season year")
    parser.add_argument("--max-matches", type=int, default=None, help="Limit matches (debug)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--use-mlflow", action="store_true", help="Enable MLflow tracking")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(level=args.log_level)
    logger = logging.getLogger("main")

    try:
        settings.validate()
    except EnvironmentError as exc:
        logger.error(str(exc))
        sys.exit(1)

    pipeline = TrainingPipeline(
        competition_ids=[settings.LIGA_MX_LEAGUE_ID],
        min_season_year=args.min_year,
        use_mlflow=args.use_mlflow,
    )
    
    try:
        report = pipeline.run(max_matches=args.max_matches)
        logger.info("="*70)
        logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        logger.info(f"📦 Models saved to: {report.models_path}")
        logger.info(f"📊 Accuracy: {report.comprehensive_metrics['accuracy']:.4f}")
        logger.info(f"📉 Log Loss: {report.comprehensive_metrics['log_loss']:.4f}")
        logger.info(f"🎯 Brier Score: {report.comprehensive_metrics['brier_score']:.4f}")
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
