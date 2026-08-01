import pandas as pd

class LigaMXDataProcessor:
    def __init__(self):
        pass

    def process_raw_data(self, raw_fixtures):
        """Limpia y estructura los datos crudos devueltos por iSportsAPI para la Liga MX."""
        processed_data = []
        for match in raw_fixtures:
            match_row = {
                "match_id": match.get("matchId"),
                "date": match.get("matchTime"),
                "home_team": match.get("homeName"),
                "away_team": match.get("awayName"),
                "home_goals": match.get("homeScore", 0),
                "away_goals": match.get("awayScore", 0),
            }
            processed_data.append(match_row)
        
        df = pd.DataFrame(processed_data)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df.fillna(0, inplace=True)
        return df
