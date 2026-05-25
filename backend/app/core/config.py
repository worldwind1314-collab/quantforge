"""Application configuration, loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "QuantForge"
    CHINESE_NAME: str = "熔量"
    VERSION: str = "0.1.0"
    DEBUG: bool = False

    # Database — same PG instance as BeyondFate, separate database
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/quantforge"

    # JWT
    SECRET_KEY: str = ""  # MUST be set via .env in production
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    # DeepSeek API (for financial NLP)
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    AI_MODEL: str = "deepseek-chat"

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:8000", "http://127.0.0.1:8000"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
