
"""Shared configuration primitives for the papersys package."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypeVar
import tomllib

from pydantic import BaseModel, ConfigDict


class BaseConfig(BaseModel):
    """Base config with strict validation rules."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    @classmethod
    def from_toml(cls: "type[C]", path: Path) -> "C":
        """Construct the config object from a TOML file."""
        data = _load_toml(path)
        return cls.model_validate(data)


C = TypeVar("C", bound="BaseConfig")


def load_config(config_cls: type[C], path: Path) -> C:
    """Parse the given TOML file into the provided config class."""
    return config_cls.from_toml(path)


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    if not isinstance(data, dict):
        raise ValueError("Root TOML value must be a table")
    return data


__all__ = ["BaseConfig", "load_config"]


class DatabaseConfig(BaseConfig):
    """Database configuration."""

    uri: str  # 数据库URI，可以是本地路径或 S3/R2 路径
    
class EmbeddingConfig(BaseConfig):
    """Embedding configuration."""

    model: str
    dim: int

class PaperConfig(BaseConfig):
    """Paper configuration."""
    
    categories: list[str]

class LogisticRegressionConfig(BaseConfig):
    """Logistic regression configuration."""
    
    C: float = 1.0
    max_iter: int = 1000

class ConfidenceWeightedSamplingConfig(BaseConfig):
    """Confidence weighted sampling configuration."""
    
    enable: bool = False
    high_conf_threshold: float = 0.9
    high_conf_weight: float = 2.0

class AdaptiveDifficultySamplingConfig(BaseConfig):
    """Adaptive difficulty sampling configuration."""
    
    enable: bool = False
    n_neighbors: int = 5
    pos_sampling_ratio: float = 2.0
    synthetic_ratio: float = 0.5
    k_smote: int = 16

class PredictConfig(BaseConfig):
    """Prediction configuration."""
    
    last_n_days: int = 7
    sample_rate: float = 0.15
    high_threshold: float = 0.95
    boundary_threshold: float = 0.5

class RecommendConfig(BaseConfig):
    """Recommendation configuration."""
    
    neg_sample_ratio: float = 5.0
    seed: int = 42
    logistic_regression: LogisticRegressionConfig
    confidence_weighted_sampling: ConfidenceWeightedSamplingConfig
    adaptive_difficulty_sampling: AdaptiveDifficultySamplingConfig
    predict: PredictConfig
    
class AppConfig(BaseConfig):
    """Application configuration."""

    database: DatabaseConfig
    embedding: EmbeddingConfig
    paper: PaperConfig
    recommend: RecommendConfig