"""Algorithm factory and exports."""

from __future__ import annotations

from ...config import RecommendConfig
from .base import BaseRecommendAlgorithm, RecommendPredictData, RecommendTrainingData
from .cluster_uct import ClusterUctAlgorithm
from .logistic_regression import LogisticRegressionAlgorithm

_ALGORITHMS: dict[str, type[BaseRecommendAlgorithm]] = {
    "logistic_regression": LogisticRegressionAlgorithm,
    "cluster_uct": ClusterUctAlgorithm,
}


def create_algorithm(config: RecommendConfig) -> BaseRecommendAlgorithm:
    """Instantiate recommendation strategy from config."""
    try:
        algo_cls = _ALGORITHMS[config.algorithm]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"未知的推荐算法：{config.algorithm}") from exc
    return algo_cls(config)


__all__ = [
    "BaseRecommendAlgorithm",
    "ClusterUctAlgorithm",
    "LogisticRegressionAlgorithm",
    "RecommendPredictData",
    "RecommendTrainingData",
    "create_algorithm",
]
