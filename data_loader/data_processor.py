"""
Data processor for transforming raw API data into training-ready DataFrames.
"""

import logging
from datetime import datetime
from typing import List, Optional

import pandas as pd

from .isports_client import iSportsClient

logger = logging.getLogger(__name__)


class DataProcessor:
    """Processes raw iSportsAPI data into structured DataFrames for ML."""

    def __init__(self, client: Optional[iSportsClient] = None):
        """
        Initialize data processor.

        Args:
            client: iSportsClient instance (creates new if None)
        """
        self.client = client or iSportsClient()

    def fetch_historical_matches(
        self,
        competition_ids: List[str],
        min_season_year: int = 2019,
        max_matches: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch and process historical match data.

        Args:
            competition_ids: List of competition IDs (typically just Liga MX)
            min_season_year: Earliest season to include
            max_matches: Limit number of matches (for debugging)

        Returns:
            DataFrame with processed match data
        """
        logger.info(f"Fetching matches from {min_season_year} onwards...")

        all_matches = []
        for comp_id in competition_ids:
            try:
                raw_matches = self.client.get_liga_mx_schedule()
                all_matches.extend(raw_matches)
            except Exception as e:
                logger.error(f"Failed to fetch matches for competition {comp_id}: {e}")

        if not all_matches:
            logger.warning("No matches retrieved")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_matches)

        # Standardize column names
        column_mapping = {
            'matchId': 'match_id',
            'leagueId': 'competition_id',
            'seasonId': 'season_id',
            'homeName': 'home_name',
            'awayName': 'away_name',
            'homeId': 'home_id',
            'awayId': 'away_id',
            'homeScore': 'home_score',
            'awayScore': 'away_score',
            'matchTime': 'start_time',
            'statusId': 'status',
        }
        df = df.rename(columns=column_mapping)

        # Convert timestamps
        if 'start_time' in df.columns:
            df['start_time'] = pd.to_datetime(
                df['start_time'].astype(int),
                unit='s',
                errors='coerce'
            )

        # Filter by year
        if 'start_time' in df.columns:
            df = df[df['start_time'].dt.year >= min_season_year]

        # Filter completed matches only (status 3 = finished)
        if 'status' in df.columns:
            df = df[df['status'] == 3]

        # Ensure scores are numeric
        df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce').fillna(0).astype(int)
        df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce').fillna(0).astype(int)

        # Create target variable (1=Home, X=Draw, 2=Away)
        df['result_1x2'] = df.apply(
            lambda row: 0 if row['home_score'] > row['away_score']
            else (1 if row['home_score'] == row['away_score'] else 2),
            axis=1
        )

        # Sort by date
        df = df.sort_values('start_time').reset_index(drop=True)

        if max_matches:
            df = df.head(max_matches)

        logger.info(f"✅ Processed {len(df)} matches")
        return df
