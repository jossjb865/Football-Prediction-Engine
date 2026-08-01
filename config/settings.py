import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Settings:
    """Central configuration – Liga MX only, iSportsAPI only (GitHub Secrets compatible)."""

    # iSportsAPI
    ISPORTS_API_KEY: str = field(default_factory=lambda: os.environ.get("ISPORTS_API_KEY", ""))
    ISPORTS_BASE_URL: str = field(
        default_factory=lambda: os.environ.get("ISPORTS_BASE_URL", "http://api.isportsapi.com")
    )
    # Liga MX leagueId – override via secret/env if the default is outdated.
    # Discover via GET /sport/football/league/basic and filter name containing "Liga MX" or "Mexico".
    LIGA_MX_LEAGUE_ID: str = field(
        default_factory=lambda: os.environ.get("LIGA_MX_LEAGUE_ID", "130")
    )

    # Rate limiting
    MAX_QPS: float = field(default_factory=lambda: float(os.environ.get("ISPORTS_MAX_QPS", "1.0")))
    RATE_LIMIT_BURST: int = field(default_factory=lambda: int(os.environ.get("ISPORTS_RATE_BURST", "3")))

    # Retry / backoff
    MAX_RETRIES: int = field(default_factory=lambda: int(os.environ.get("ISPORTS_MAX_RETRIES", "5")))
    BACKOFF_FACTOR: float = field(default_factory=lambda: float(os.environ.get("ISPORTS_BACKOFF_FACTOR", "1.5")))
    RETRY_STATUS_CODES: List[int] = field(default_factory=lambda: [429, 500, 502, 503, 504])

    # Local cache
    CACHE_DIR: str = field(default_factory=lambda: os.environ.get("CACHE_DIR", "./data_cache"))
    CACHE_TTL_SECONDS: int = field(default_factory=lambda: int(os.environ.get("CACHE_TTL_SECONDS", "86400")))

    # Model / training
    RANDOM_SEED: int = field(default_factory=lambda: int(os.environ.get("RANDOM_SEED", "42")))
    N_TIME_SERIES_SPLITS: int = field(default_factory=lambda: int(os.environ.get("N_TIME_SERIES_SPLITS", "5")))
    MIN_TRAIN_SIZE: int = field(default_factory=lambda: int(os.environ.get("MIN_TRAIN_SIZE", "300")))

    # Feature windows
    ROLLING_WINDOWS: List[int] = field(default_factory=lambda: [3, 5, 10, 20])

    # Paths
    ARTIFACTS_DIR: str = field(default_factory=lambda: os.environ.get("ARTIFACTS_DIR", "./artifacts"))
    MODELS_DIR: str = field(default_factory=lambda: os.path.join(os.environ.get("ARTIFACTS_DIR", "./artifacts"), "models"))
    FEATURES_DIR: str = field(default_factory=lambda: os.path.join(os.environ.get("ARTIFACTS_DIR", "./artifacts"), "features"))

    def validate(self) -> None:
        if not self.ISPORTS_API_KEY:
            raise EnvironmentError(
                "ISPORTS_API_KEY environment variable is required. "
                "Inject it via GitHub Secrets or local .env."
            )
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        os.makedirs(self.ARTIFACTS_DIR, exist_ok=True)
        os.makedirs(self.MODELS_DIR, exist_ok=True)
        os.makedirs(self.FEATURES_DIR, exist_ok=True)


settings = Settings()
