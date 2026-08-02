#!/usr/bin/env python3
"""
Entry point for the Liga MX Prediction Engine (iSportsAPI only).
Usage:
    export ISPORTS_API_KEY=your_key
    python main.py --min-year 2019
"""

import argparse
import logging
import os
import sys

from config.logging_config import setup_logging
from config.settings import settings
from pipelines.training_pipeline import TrainingPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Liga MX Prediction Engine – Super Ensemble Trainer")
    parser.add_argument("--min-year", type=int, default=2019, help="Earliest season year to include")
    parser.add_argument("--max-matches", type=int, default=None, help="Limit number of matches (debug)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
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
    )
    report = pipeline.run(max_matches=args.max_matches)
    logger.info("Pipeline finished. Metrics: %s", report.to_dict())

    # Asegurar que la carpeta de artefactos exista y guardar el modelo
    os.makedirs("artifacts/models", exist_ok=True)
    
    if hasattr(pipeline, "save"):
        pipeline.save("artifacts/models/model.joblib")
        logger.info("Models saved successfully to artifacts/models/")
    elif hasattr(pipeline, "ensemble") and hasattr(pipeline.ensemble, "save"):
        pipeline.ensemble.save("artifacts/models/ensemble.joblib")
        logger.info("Ensemble saved successfully to artifacts/models/")


if __name__ == "__main__":
    main()
