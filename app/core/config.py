from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    APP_NAME: str = "QRypt"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # API KEYS  
    OPENAI_API_KEY: str = ""
    VIRUSTOTAL_API_KEY: str = ""
    GROQ_API_KEY: str = ""

    # MONGO_DB
    MONGODB_URI: str = ""
    MONGODB_DB_NAME: str = "qrypt"

    # RATE LIMITING
    RATE_LIMIT_PER_MINUTE: int = 30

    # TIMEOUTS
    HTTP_TIMEOUT: int = 8
    VT_POLL_WAIT: int = 3

    WEIGHT_PHYSICAL: float = 0.30
    WEIGHT_THREAT_INTEL: float = 0.30
    WEIGHT_AI_CONTEXT: float = 0.40

    THRESHOLD_HIGH_RISK: int = 60
    THRESHOLD_SUSPICIOUS: int = 30

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",          # ignore unknown env vars
    )

    @property
    def verdict_thresholds(self) -> dict:
        return {
            "HIGH_RISK":   self.THRESHOLD_HIGH_RISK,
            "SUSPICIOUS":  self.THRESHOLD_SUSPICIOUS,
            "SAFE":        0,
        }


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings instance.
    lru_cache ensures .env is read exactly once at startup.
    """
    return Settings()


# Module-level singleton — import this everywhere
settings = get_settings()