import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    model_config = SettingsConfigDict(
        env_file='.env',
        extra='ignore' # Ignores other env variables not specified here
    )

    GEMINI_API_KEY: str
    N8N_WEBHOOK_URL: str
    ELEVENLABS_API_KEY: str

    if not all([GEMINI_API_KEY, N8N_WEBHOOK_URL, ELEVENLABS_API_KEY]):
        raise ValueError("Missing one or more required environment variables.")

# Create a single instance of the settings for the entire application
settings = Settings()
