#!/usr/bin/env python3
"""
CLI for live / batch inference – Liga MX only (iSportsAPI).
Example:
    python predict.py --date 2026-08-01
"""

import argparse
import logging
from datetime import datetime

import pandas as pd

from config.logging_config import setup_logging
from config.settings import settings
from data_loader.isports_client import iSportsClient
from pipelines.inference_pipeline import InferencePipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Liga MX Prediction Engine – Inference")
    parser.add_argument("--date", type=str, default=None, help="YYYY-MM-DD (default = today)")
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(level=args.log_level)
    logger = logging.getLogger("predict")

    settings.validate()
    date_str = args.date or datetime.utcnow().strftime("%Y-%m-%d")

    client = iSportsClient()
    schedules = client.get_schedule_by_date(date_str)
    if not schedules:
        schedules = client.get_livescores_today()

    records = []
    for item in schedules:
        match_time = item.get("matchTime")
        try:
            start_time = pd.to_datetime(int(match_time), unit="s", utc=True) if match_time else pd.NaT
        except Exception:
            start_time = pd.NaT
        records.append(
            {
                "match_id": str(item.get("matchId") or ""),
                "start_time": start_time,
                "home_id": str(item.get("homeId") or ""),
                "away_id": str(item.get("awayId") or ""),
                "home_name": item.get("homeName") or item.get("home") or "",
                "away_name": item.get("awayName") or item.get("away") or "",
                "competition_id": str(item.get("leagueId") or settings.LIGA_MX_LEAGUE_ID),
            }
        )

    if not records:
        logger.warning("No Liga MX matches found for %s", date_str)
        return

    matches = pd.DataFrame(records)
    matches = matches[matches["match_id"] != ""].reset_index(drop=True)

    if matches.empty:
        logger.warning("No matches left after filtering")
        return

    pipeline = InferencePipeline(models_dir=args.models_dir)
    preds = pipeline.predict_matches(matches)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print("\n=== Liga MX Predictions ===")
    print(preds.to_string(index=False))
    preds.to_csv(f"predictions_{date_str}.csv", index=False)
    logger.info("Saved predictions_%s.csv", date_str)


if __name__ == "__main__":
    main()
