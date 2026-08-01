import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from .isports_client import iSportsClient

logger = logging.getLogger(__name__)


class DataProcessor:
    """Transforms raw iSportsAPI payloads into clean tabular datasets for Liga MX only."""

    def __init__(self, client: Optional[iSportsClient] = None):
        self.client = client or iSportsClient()

    def fetch_historical_matches(
        self,
        competition_ids: Optional[List[str]] = None,
        min_season_year: int = 2019,
        max_matches: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Pull Liga MX schedule/results. competition_ids is ignored (hard-locked to Liga MX).
        """
        records: List[Dict[str, Any]] = []
        seasons_to_try = [None, "2025-2026", "2024-2025", "2023-2024", "2022-2023"]
        seen_ids = set()

        for season in seasons_to_try:
            try:
                schedules = self.client.get_liga_mx_schedule(season=season)
            except Exception as exc:
                logger.warning("Could not fetch schedule for season %s: %s", season, exc)
                continue

            for item in schedules:
                mid = str(item.get("matchId") or item.get("match_id") or "")
                if not mid or mid in seen_ids:
                    continue
                home_score = item.get("homeScore")
                away_score = item.get("awayScore")
                if home_score is None or away_score is None:
                    continue
                status = item.get("status")
                if status is not None and str(status).upper() not in ("-1", "FT", "FINISHED", "1"):
                    continue

                match_time = item.get("matchTime")
                if match_time:
                    try:
                        start_time = pd.to_datetime(int(match_time), unit="s", utc=True)
                    except Exception:
                        start_time = pd.NaT
                else:
                    start_time = pd.NaT

                year = start_time.year if pd.notna(start_time) else 0
                if year and year < min_season_year:
                    continue

                records.append(
                    {
                        "match_id": mid,
                        "season_id": item.get("season") or season or "current",
                        "competition_id": str(item.get("leagueId") or self.client.league_id),
                        "competition_name": item.get("leagueName") or "Liga MX",
                        "season_name": item.get("season") or season or "current",
                        "start_time": start_time,
                        "home_id": str(item.get("homeId") or item.get("home_id") or ""),
                        "away_id": str(item.get("awayId") or item.get("away_id") or ""),
                        "home_name": item.get("homeName") or item.get("home") or "",
                        "away_name": item.get("awayName") or item.get("away") or "",
                        "home_score": int(home_score),
                        "away_score": int(away_score),
                        "status": status,
                        "match_status": status,
                        "venue_id": None,
                        "venue_name": item.get("venue") or None,
                        "referee_id": None,
                    }
                )
                seen_ids.add(mid)
                if max_matches and len(records) >= max_matches:
                    break
            if max_matches and len(records) >= max_matches:
                break

        df = pd.DataFrame(records)
        if df.empty:
            logger.warning("No historical Liga MX matches retrieved.")
            return df

        df = df.dropna(subset=["start_time"]).sort_values("start_time").reset_index(drop=True)
        df["result_1x2"] = df.apply(self._encode_1x2, axis=1)
        df["total_goals"] = df["home_score"] + df["away_score"]
        logger.info("Loaded %d Liga MX historical matches", len(df))
        return df

    @staticmethod
    def _encode_1x2(row: pd.Series) -> int:
        if row["home_score"] > row["away_score"]:
            return 0
        if row["home_score"] < row["away_score"]:
            return 2
        return 1

    def enrich_with_match_statistics(self, matches_df: pd.DataFrame, sample_size: Optional[int] = None) -> pd.DataFrame:
        if matches_df.empty:
            return matches_df

        target = matches_df if sample_size is None else matches_df.sample(
            n=min(sample_size, len(matches_df)), random_state=42
        )
        extra_rows = []
        for _, row in target.iterrows():
            try:
                summary = self.client.get_match_stats(row["match_id"])
                stats_list = summary.get("stats", summary) if isinstance(summary, dict) else summary
                if not isinstance(stats_list, list):
                    continue

                def _val(type_code: int, side: str = "home"):
                    for s in stats_list:
                        if s.get("type") == type_code:
                            return s.get(side)
                    return None

                extra = {
                    "match_id": row["match_id"],
                    "home_shots_on_target": _val(4, "home"),
                    "away_shots_on_target": _val(4, "away"),
                    "home_possession": _val(14, "home"),
                    "away_possession": _val(14, "away"),
                    "home_yellow_cards": _val(11, "home"),
                    "away_yellow_cards": _val(11, "away"),
                    "home_corners": _val(6, "home"),
                    "away_corners": _val(6, "away"),
                    "home_shots": _val(3, "home"),
                    "away_shots": _val(3, "away"),
                }
                extra_rows.append(extra)
            except Exception as exc:
                logger.debug("Could not enrich %s: %s", row["match_id"], exc)
                continue

        if not extra_rows:
            return matches_df
        extra_df = pd.DataFrame(extra_rows)
        return matches_df.merge(extra_df, on="match_id", how="left")
