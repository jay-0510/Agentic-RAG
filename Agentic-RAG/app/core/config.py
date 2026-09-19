from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration is loaded from environment variables or a local .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    retriever_backend: str = "local"
    llm_backend: str = "local"
    max_retrieval_retries: int = 2
    max_verification_retries: int = 1
    aws_region: str = "us-east-1"
    nova_model_id: str = "amazon.nova-micro-v1:0"
    embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    opensearch_url: str | None = None
    opensearch_index: str = "sports-knowledge"
    opensearch_vector_field: str = "embedding"


@lru_cache
def get_settings() -> Settings:
    return Settings()
