import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = "Football Prediction Engine - Liga MX"
    LEAGUE_ID: str = os.getenv("LEAGUE_ID", "mx_liga")  # Identificador de la Liga MX en iSportsAPI
    ISPORTS_API_KEY: str = os.getenv("ISPORTS_API_KEY", "")
    ISPORTS_BASE_URL: str = "https://api.isportsapi.com/routine/soccer"
    
    MODELS_DIR: str = "models/"
    FEATURE_STORE_DIR: str = "features/store/"

settings = Settings()
