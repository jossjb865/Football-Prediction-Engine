"""
iSportsAPI client – exclusively for Liga MX.
Base: http://api.isportsapi.com  (or api2 / api-asia)
Auth: ?api_key=ISPORTS_API_KEY
"""

import logging
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config.settings import settings
from .rate_limiter import TokenBucketRateLimiter
from .cache_manager import CacheManager

logger = logging.getLogger(__name__)


class iSportsClient:
    """
    Production client for iSportsAPI focused solely on Liga MX.
    All endpoints use the api_key query parameter.
    """

    def __init__(self):
        settings.validate()
        self.api_key = settings.ISPORTS_API_KEY
        self.base_url = settings.ISPORTS_BASE_URL.rstrip("/") + "/"
        self.league_id = settings.LIGA_MX_LEAGUE_ID

        self.rate_limiter = TokenBucketRateLimiter(
            rate=settings.MAX_QPS,
            capacity=settings.RATE_LIMIT_BURST,
        )
        self.cache = CacheManager(settings.CACHE_DIR, settings.CACHE_TTL_SECONDS)

        self.session = requests.Session()
        retry_strategy = Retry(
            total=settings.MAX_RETRIES,
            backoff_factor=settings.BACKOFF_FACTOR,
            status_forcelist=settings.RETRY_STATUS_CODES,
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=20)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "LigaMX-PredictionEngine/1.0",
            }
        )

    def _request(self, endpoint: str, params: Optional[Dict] = None, use_cache: bool = True) -> Any:
        params = dict(params or {})
        params["api_key"] = self.api_key
        url = urljoin(self.base_url, endpoint.lstrip("/"))
        cache_key = f"{url}?{sorted(params.items())}"

        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug("Cache hit: %s", endpoint)
                return cached

        self.rate_limiter.acquire()
        logger.info("GET %s params=%s", endpoint, {k: v for k, v in params.items() if k != "api_key"})
        response = self.session.get(url, params=params, timeout=30)

        if response.status_code == 429:
            retry_after = float(response.headers.get("Retry-After", 2.0))
            logger.warning("Rate limited. Sleeping %.1fs", retry_after)
            time.sleep(retry_after)
            self.rate_limiter.acquire()
            response = self.session.get(url, params=params, timeout=30)

        response.raise_for_status()
        data = response.json()

        # iSportsAPI wraps most responses as {"code": 0, "message": "...", "data": ...}
        if isinstance(data, dict) and "code" in data:
            if data.get("code") != 0:
                raise RuntimeError(f"iSportsAPI error {data.get('code')}: {data.get('message')}")
            payload = data.get("data", data)
        else:
            payload = data

        if use_cache:
            self.cache.set(cache_key, payload)
        return payload

    # ------------------------------------------------------------------
    # Liga MX focused endpoints
    # ------------------------------------------------------------------

    def get_leagues(self) -> List[Dict]:
        """Return all leagues (used once to discover Liga MX id if needed)."""
        data = self._request("sport/football/league/basic")
        if isinstance(data, list):
            return data
        return data.get("leagues", data) if isinstance(data, dict) else []

    def get_liga_mx_schedule(self, season: Optional[str] = None) -> List[Dict]:
        """
        Schedule & Results for Liga MX.
        Path: /sport/football/schedule?leagueId=...
        """
        params = {"leagueId": self.league_id}
        if season:
            params["season"] = season
        data = self._request("sport/football/schedule", params=params)
        if isinstance(data, list):
            return data
        return []

    def get_schedule_by_date(self, date_str: str) -> List[Dict]:
        """Matches on a given date (yyyy-MM-dd). Filtered later to Liga MX."""
        data = self._request("sport/football/schedule", params={"date": date_str})
        if isinstance(data, list):
            return [m for m in data if str(m.get("leagueId")) == str(self.league_id)]
        return []

    def get_match_stats(self, match_id: str) -> Dict:
        """Team technical stats for a match (shots, corners, cards, possession...)."""
        data = self._request("sport/football/stats", params={"matchId": match_id})
        if isinstance(data, list):
            for item in data:
                if str(item.get("matchId")) == str(match_id):
                    return item
            return data[0] if data else {}
        return data if isinstance(data, dict) else {}

    def get_standing(self) -> Dict:
        """Current Liga MX standings."""
        return self._request("sport/football/standing/league", params={"leagueId": self.league_id})

    def get_top_scorers(self, season: Optional[str] = None) -> List[Dict]:
        params = {"leagueId": self.league_id}
        if season:
            params["season"] = season
        data = self._request("sport/football/topscorer", params=params)
        return data if isinstance(data, list) else []

    def get_livescores_today(self) -> List[Dict]:
        data = self._request("sport/football/livescores")
        if isinstance(data, list):
            return [m for m in data if str(m.get("leagueId")) == str(self.league_id)]
        return []

    def get_match_analysis(self, match_id: str) -> Dict:
        """H2H, form, odds stats, goals distribution."""
        return self._request("sport/football/analysis", params={"matchId": match_id})
