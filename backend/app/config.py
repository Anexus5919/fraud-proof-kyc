from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache

# Resolve .env path relative to this file, not CWD
# This file is at backend/app/config.py → .env is at backend/.env
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql+asyncpg://localhost/kyc_db"

    # Cloudinary
    cloudinary_cloud_name: str = ""
    cloudinary_api_key: str = ""
    cloudinary_api_secret: str = ""

    # CORS
    cors_origins: str = "http://localhost:5173,http://localhost:5174"

    # Thresholds
    spoof_threshold: float = 0.55  # Stricter threshold for anti-spoofing
    duplicate_threshold: float = 0.55  # Cosine distance threshold (catches same person with slight pose changes)

    class Config:
        env_file = str(_ENV_FILE)
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
