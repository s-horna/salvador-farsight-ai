"""Application settings loaded from environment variables and the project .env file."""

from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Credentials and connection details
    openai_api_key: SecretStr
    pinecone_api_key: SecretStr
    pinecone_index: str
    postgres_host: str
    postgres_port: int = 5432
    postgres_db_name: str
    # The provided .env names the username POSTGRES; POSTGRES_USER is accepted too.
    postgres_user: str = Field(validation_alias=AliasChoices("POSTGRES_USER", "POSTGRES"))
    postgres_password: SecretStr

    # Models
    embedding_model: str = "text-embedding-3-large"
    insight_model: str = "gpt-4.1-mini"

    # Sector resolution: keep matches scoring at least
    # max(min_similarity, top_score - similarity_window).
    # Defaults are the configuration recommended by the evals/ sweep.
    pinecone_top_k: int = Field(default=1000, ge=1, le=1000)
    min_similarity: float = Field(default=0.28, ge=0.0, le=1.0)
    similarity_window: float = Field(default=0.16, ge=0.0, le=1.0)

    # Resilience
    openai_timeout_seconds: float = 20.0
    openai_max_retries: int = 2
    pinecone_timeout_seconds: float = 20.0
    postgres_command_timeout_seconds: float = 30.0
    postgres_pool_min_size: int = 1
    postgres_pool_max_size: int = 10
    insight_timeout_seconds: float = 15.0

    # Response shaping
    top_results_limit: int = Field(default=10, ge=1, le=100)
    sector_cache_size: int = Field(default=256, ge=0)
