
"""Shared configuration primitives for the papersys package."""

from __future__ import annotations

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


class HuggingFaceDatasetConfig(BaseConfig):
    """HuggingFace dataset configuration."""

    hf_repo: str
    shard_prefix: str
    revision: str | None = None


class MetadataConfig(HuggingFaceDatasetConfig):
    """Metadata specific settings."""


class EmbeddingConfig(HuggingFaceDatasetConfig):
    """Embedding specific settings."""

    model: str | None = None
    dim: int


class PaperConfig(BaseConfig):
    """Paper category configuration."""

    categories: list[str]


class LogisticRegressionConfig(BaseConfig):
    """Logistic regression hyper-parameters."""

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
    limit: int | None = None


class RecommendConfig(BaseConfig):
    """Recommendation behaviour configuration."""

    neg_sample_ratio: float = 5.0
    seed: int = 42
    logistic_regression: LogisticRegressionConfig
    confidence_weighted_sampling: ConfidenceWeightedSamplingConfig
    adaptive_difficulty_sampling: AdaptiveDifficultySamplingConfig
    predict: PredictConfig


class GitStoreConfig(BaseConfig):
    """External Git repository used to persist JSONL artefacts."""

    repo_url: str
    branch: str = "main"
    summary_dir: Path
    preference_file: Path


class SummaryConfig(BaseConfig):
    """Summary generation configuration."""

    model: str = "gemini-2.5-flash"
    use_batch: bool = True
    poll_interval: int = 30


class OCRConfig(BaseConfig):
    """OCR pipeline configuration."""

    ar5iv: bool = True


class NotionConfig(BaseConfig):
    """Notion sync configuration."""

    database: str


class AppConfig(BaseConfig):
    """Application configuration."""

    metadata: MetadataConfig
    embedding: EmbeddingConfig
    paper: PaperConfig
    recommend: RecommendConfig
    summary: SummaryConfig
    ocr: OCRConfig = OCRConfig()
    git_store: GitStoreConfig
    notion: NotionConfig


__all__ += [
    "HuggingFaceDatasetConfig",
    "MetadataConfig",
    "EmbeddingConfig",
    "PaperConfig",
    "RecommendConfig",
    "SummaryConfig",
    "OCRConfig",
    "GitStoreConfig",
    "NotionConfig",
    "AppConfig",
]
