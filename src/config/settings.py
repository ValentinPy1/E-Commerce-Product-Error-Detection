"""Centralized configuration management using environment variables."""

import os
from pathlib import Path
from typing import Optional

from .constants import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_RANDOM_STATE,
)


class Settings:
    """Application settings loaded from environment variables."""

    def __init__(self):
        """Initialize settings from environment variables."""
        # Paths
        self.artifacts_dir = Path(
            os.getenv("ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR)
        ).resolve()
        self.data_dir = Path(os.getenv("DATA_DIR", DEFAULT_DATA_DIR)).resolve()

        # Random seed
        self.random_state = int(
            os.getenv("RANDOM_STATE", str(DEFAULT_RANDOM_STATE))
        )

        # API keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        # Device settings
        self.cuda_device = os.getenv("CUDA_VISIBLE_DEVICES")

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get the global settings instance.

    Returns:
        Settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

