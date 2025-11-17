"""Shared dataclasses and interfaces for recommendation algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import polars as pl

from ...config import RecommendConfig


@dataclass(slots=True)
class RecommendTrainingData:
    """Dataset slices needed by algorithms during fit()."""

    dataset: pl.DataFrame
    labeled: pl.DataFrame
    positive: pl.DataFrame


@dataclass(slots=True)
class RecommendPredictData:
    """Dataset slice used by algorithms during predict()."""

    dataset: pl.DataFrame
    limit: int | None = None


class BaseRecommendAlgorithm(ABC):
    """Abstract strategy for scoring candidate papers."""

    def __init__(self, config: RecommendConfig) -> None:
        self.config = config

    @abstractmethod
    def fit(self, data: RecommendTrainingData) -> None:
        """Train internal state from prepared datasets."""

    @abstractmethod
    def predict(self, data: RecommendPredictData) -> pl.DataFrame:
        """Score prepared dataset and annotate with `score`/`show` columns."""


__all__ = [
    "BaseRecommendAlgorithm",
    "RecommendPredictData",
    "RecommendTrainingData",
]
