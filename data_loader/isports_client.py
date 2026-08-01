import requests
from config.settings import settings

class ISportsClient:
    def __init__(self):
        self.api_key = settings.ISPORTS_API_KEY
        self.base_url = settings.ISPORTS_BASE_URL

    def get_fixtures(self):
        """Obtiene los partidos y el calendario actual de la Liga MX."""
        endpoint = f"{self.base_url}/schedule"
        params = {
            "api_key": self.api_key,
            "leagueId": settings.LEAGUE_ID
        }
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        return response.json().get("data", [])

    def get_match_stats(self, match_id):
        """Obtiene estadísticas detalladas (remates, posesión, córners) de un partido específico."""
        endpoint = f"{self.base_url}/match/stats"
        params = {
            "api_key": self.api_key,
            "matchId": match_id
        }
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        return response.json().get("data", [])
