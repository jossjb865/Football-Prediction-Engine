"""
Rolling window metrics calculator for team performance tracking.
"""

import logging
from typing import List

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class RollingMetricsCalculator:
    """Calculate rolling statistics for team performance."""

    def __init__(self, windows: List[int] = None):
        """
        Initialize calculator.

        Args:
            windows: List of rolling window sizes (e.g., [3, 5, 10])
        """
        self.windows = windows or [3, 5, 10, 20]

    def compute(self, matches: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling metrics for both home and away teams.

        Args:
            matches: DataFrame with match data

        Returns:
            DataFrame with rolling metrics per match
        """
        if matches.empty or 'home_id' not in matches.columns:
            return pd.DataFrame()

        df = matches.copy()
        results = []

        for _, match in df.iterrows():
            match_metrics = {'match_id': match['match_id']}

            # Home team rolling stats
            home_history = df[
                (df['start_time'] < match['start_time']) &
                ((df['home_id'] == match['home_id']) | (df['away_id'] == match['home_id']))
            ].copy()

            for window in self.windows:
                recent = home_history.tail(window)
                if len(recent) > 0:
                    match_metrics[f'home_pts_roll{window}'] = self._calculate_points(
                        recent, match['home_id']
                    )
                    match_metrics[f'home_goals_scored_roll{window}'] = self._goals_scored(
                        recent, match['home_id']
                    )
                    match_metrics[f'home_goals_conceded_roll{window}'] = self._goals_conceded(
                        recent, match['home_id']
                    )
                else:
                    match_metrics[f'home_pts_roll{window}'] = 0.0
                    match_metrics[f'home_goals_scored_roll{window}'] = 0.0
                    match_metrics[f'home_goals_conceded_roll{window}'] = 0.0

            # Away team rolling stats
            away_history = df[
                (df['start_time'] < match['start_time']) &
                ((df['home_id'] == match['away_id']) | (df['away_id'] == match['away_id']))
            ].copy()

            for window in self.windows:
                recent = away_history.tail(window)
                if len(recent) > 0:
                    match_metrics[f'away_pts_roll{window}'] = self._calculate_points(
                        recent, match['away_id']
                    )
                    match_metrics[f'away_goals_scored_roll{window}'] = self._goals_scored(
                        recent, match['away_id']
                    )
                    match_metrics[f'away_goals_conceded_roll{window}'] = self._goals_conceded(
                        recent, match['away_id']
                    )
                else:
                    match_metrics[f'away_pts_roll{window}'] = 0.0
                    match_metrics[f'away_goals_scored_roll{window}'] = 0.0
                    match_metrics[f'away_goals_conceded_roll{window}'] = 0.0

            results.append(match_metrics)

        return pd.DataFrame(results)

    def _calculate_points(self, matches: pd.DataFrame, team_id: str) -> float:
        """Calculate total points for a team in given matches."""
        points = 0
        for _, m in matches.iterrows():
            if m['home_id'] == team_id:
                if m['home_score'] > m['away_score']:
                    points += 3
                elif m['home_score'] == m['away_score']:
                    points += 1
            elif m['away_id'] == team_id:
                if m['away_score'] > m['home_score']:
                    points += 3
                elif m['away_score'] == m['home_score']:
                    points += 1
        return float(points)

    def _goals_scored(self, matches: pd.DataFrame, team_id: str) -> float:
        """Calculate total goals scored by team."""
        total = 0
        for _, m in matches.iterrows():
            if m['home_id'] == team_id:
                total += m['home_score']
            elif m['away_id'] == team_id:
                total += m['away_score']
        return float(total)

    def _goals_conceded(self, matches: pd.DataFrame, team_id: str) -> float:
        """Calculate total goals conceded by team."""
        total = 0
        for _, m in matches.iterrows():
            if m['home_id'] == team_id:
                total += m['away_score']
            elif m['away_id'] == team_id:
                total += m['home_score']
        return float(total)
