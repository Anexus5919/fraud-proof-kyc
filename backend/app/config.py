from pydantic_settings import BaseSettings
from functools import lru_cache


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
    duplicate_threshold: float = 0.4  # Cosine distance threshold

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
