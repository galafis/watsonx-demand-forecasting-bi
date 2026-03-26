"""Application configuration loaded from environment variables and settings.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


def _load_yaml_config() -> dict[str, Any]:
    """Load YAML configuration from config/settings.yaml."""
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


_yaml = _load_yaml_config()


class WatsonxSettings(BaseSettings):
    """IBM Watsonx connection settings."""

    api_key: str = Field(default="", alias="WATSONX_API_KEY")
    project_id: str = Field(default="", alias="WATSONX_PROJECT_ID")
    url: str = Field(
        default="https://us-south.ml.cloud.ibm.com",
        alias="WATSONX_URL",
    )
    generation_model: str = Field(default="ibm/granite-13b-chat-v2")
    ts_model: str = Field(default="ibm/granite-tsfm")


class AppSettings(BaseSettings):
    """Application server settings."""

    host: str = Field(default="0.0.0.0", alias="APP_HOST")
    port: int = Field(default=8080, alias="APP_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    environment: str = Field(default="development", alias="ENVIRONMENT")


class ForecastSettings(BaseSettings):
    """Forecast-specific settings."""

    horizon: int = Field(default=30, alias="FORECAST_HORIZON")
    backtest_folds: int = Field(default=5, alias="BACKTEST_FOLDS")


class Settings:
    """Aggregated application settings."""

    def __init__(self) -> None:
        self.watsonx = WatsonxSettings()
        self.app = AppSettings()
        self.forecast = ForecastSettings()
        self.yaml = _yaml

    @property
    def forecast_config(self) -> dict[str, Any]:
        """Forecast configuration from YAML."""
        return self.yaml.get("forecast", {})

    @property
    def features_config(self) -> dict[str, Any]:
        """Feature engineering configuration from YAML."""
        return self.yaml.get("features", {})

    @property
    def models_config(self) -> dict[str, Any]:
        """Model configuration from YAML."""
        return self.yaml.get("models", {})

    @property
    def evaluation_config(self) -> dict[str, Any]:
        """Evaluation and backtesting configuration from YAML."""
        return self.yaml.get("evaluation", {})

    @property
    def governance_config(self) -> dict[str, Any]:
        """Governance configuration from YAML."""
        return self.yaml.get("governance", {})

    @property
    def generation_params(self) -> dict[str, Any]:
        """Watsonx generation parameters from YAML."""
        return self.yaml.get("watsonx", {}).get("generation", {}).get("parameters", {})


settings = Settings()
