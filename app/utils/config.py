"""
Configuration management for the prompt optimization pipeline.
Loads and merges YAML configs with environment overrides.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

load_dotenv()

CONFIG_DIR = Path(__file__).parent.parent.parent / "configs"


# ---------------------------------------------------------------------------
# Pydantic config models
# ---------------------------------------------------------------------------

class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    timeout: int = 120
    max_retries: int = 3
    retry_delay: float = 2.0


class DatabaseConfig(BaseModel):
    url: str = "sqlite:///./runs/experiments.db"
    echo: bool = False


class CacheConfig(BaseModel):
    enabled: bool = True
    dir: str = "./data/cache"
    ttl_hours: int = 168


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "rich"
    file_enabled: bool = True


class ExperimentConfig(BaseModel):
    name: str = "default_run"
    random_seed: int = 42
    output_dir: str = "./runs"
    reports_dir: str = "./reports"
    logs_dir: str = "./logs"


class PipelineConfig(BaseModel):
    mode: str = "optimize"
    dataset: str = "synthetic_mini"
    model: str = "mistral"
    optimizer: str = "beam"
    max_budget: int = 20
    validation_split: float = 0.2
    test_split: float = 0.1
    batch_size: int = 5


class AppConfig(BaseModel):
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)

    # Raw sections from sub-configs
    datasets: dict[str, Any] = Field(default_factory=dict)
    models: dict[str, Any] = Field(default_factory=dict)
    optimizer_strategies: dict[str, Any] = Field(default_factory=dict)
    mutation_params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def apply_env_overrides(cls, values: dict) -> dict:
        """Apply environment variable overrides to config values."""
        if "ollama" not in values:
            values["ollama"] = {}
        env_url = os.getenv("OLLAMA_BASE_URL")
        if env_url:
            values["ollama"]["base_url"] = env_url

        if "database" not in values:
            values["database"] = {}
        env_db = os.getenv("DATABASE_URL")
        if env_db:
            values["database"]["url"] = env_db

        return values


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file, returning empty dict if not found."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def load_config(config_dir: str | None = None) -> AppConfig:
    """
    Load and merge all YAML configs.
    Results are cached so config is a singleton per process.
    """
    base = Path(config_dir) if config_dir else CONFIG_DIR

    base_cfg = _load_yaml(base / "base.yaml")
    datasets_cfg = _load_yaml(base / "datasets.yaml")
    models_cfg = _load_yaml(base / "models.yaml")
    optimizer_cfg = _load_yaml(base / "optimizer.yaml")

    merged = {
        **base_cfg,
        "datasets": datasets_cfg.get("datasets", {}),
        "models": models_cfg.get("models", {}),
        "optimizer_strategies": optimizer_cfg.get("optimizer", {}),
        "mutation_params": optimizer_cfg.get("mutation_params", {}),
    }

    return AppConfig(**merged)


def get_dataset_config(cfg: AppConfig, dataset_key: str) -> dict[str, Any]:
    """Retrieve a specific dataset config by key."""
    ds = cfg.datasets.get(dataset_key)
    if ds is None:
        raise KeyError(f"Dataset '{dataset_key}' not found in datasets.yaml")
    return ds


def get_model_config(cfg: AppConfig, model_key: str) -> dict[str, Any]:
    """Retrieve a specific model config by key."""
    model = cfg.models.get(model_key)
    if model is None:
        raise KeyError(f"Model '{model_key}' not found in models.yaml")
    return model


def get_optimizer_config(cfg: AppConfig, optimizer_key: str) -> dict[str, Any]:
    """Retrieve a specific optimizer strategy config by key."""
    opt = cfg.optimizer_strategies.get(optimizer_key)
    if opt is None:
        raise KeyError(f"Optimizer '{optimizer_key}' not found in optimizer.yaml")
    return opt


def ensure_dirs(cfg: AppConfig) -> None:
    """Create required output directories if they don't exist."""
    dirs = [
        cfg.experiment.output_dir,
        cfg.experiment.reports_dir,
        cfg.experiment.logs_dir,
        cfg.cache.dir,
        "./data/raw",
        "./data/processed",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
