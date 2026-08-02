#!/usr/bin/env python3
"""
Predicción mejorada con drift detection.

ERRORES CORREGIDOS:
- Falta import de pandas al inicio del archivo
- Import estaba incorrectamente al final
"""

import argparse
import logging
import os
from datetime import datetime, timedelta

import pandas as pd

from config.logging_config import setup_logging
from config.settings import settings
from data_loader.data_processor import DataProcessor
from data_loader.isports_client import iSportsClient
from pipelines.inference_pipeline import InferencePipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced Prediction Engine")
    parser.add_argument("--days-ahead", type=int, default=7, help="Days to predict")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--no-drift-check", action="store_true", help="Skip drift detection")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(level=args.log_level)
    logger = logging.getLogger("predict")

    try:
        settings.validate()
    except EnvironmentError as exc:
        logger.error(str(exc))
        return

    client = iSportsClient()
    processor = DataProcessor(client)
    pipeline = InferencePipeline()

    # Fetch upcoming matches
    today = datetime.now()
    upcoming = []
    for offset in range(args.days_ahead):
        date = (today + timedelta(days=offset)).strftime("%Y-%m-%d")
        try:
            matches = client.get_schedule_by_date(date)
            upcoming.extend(matches)
        except Exception as e:
            logger.warning(f"Could not fetch {date}: {e}")

    if not upcoming:
        logger.info("No upcoming Liga MX matches found")
        return

    # Format data
    records = []
    for m in upcoming:
        records.append({
            "match_id": str(m.get("matchId")),
            "home_name": m.get("homeName", ""),
            "away_name": m.get("awayName", ""),
            "start_time": datetime.fromtimestamp(int(m.get("matchTime", 0))),
        })

    matches_df = pd.DataFrame(records)
    logger.info(f"🔮 Predicting {len(matches_df)} matches...")

    # Predict with drift detection
    predictions, drift_report = pipeline.predict_matches(
        matches_df,
        detect_drift=not args.no_drift_check
    )

    # Save results
    os.makedirs("artifacts/predictions", exist_ok=True)
    output_path = f"artifacts/predictions/predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    predictions.to_csv(output_path, index=False)

    logger.info(f"✅ Predictions saved to {output_path}")
    print(predictions[["home_name", "away_name", "prediction", "confidence"]].to_string(index=False))


if __name__ == "__main__":
    main()
