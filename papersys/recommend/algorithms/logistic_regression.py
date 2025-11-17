"""Logistic-regression-based recommendation strategy."""

from __future__ import annotations

import numpy as np
import polars as pl
from loguru import logger

from ...config import LogisticAlgorithmConfig, RecommendConfig
from ...fields import EMBEDDING_VECTOR, ID, SCORE
from ..sampler import adaptive_sample
from ..trainer import train_model
from .base import BaseRecommendAlgorithm, RecommendPredictData, RecommendTrainingData


class LogisticRegressionAlgorithm(BaseRecommendAlgorithm):
    """Wrap the historical logistic regression workflow into a strategy."""

    def __init__(self, config: RecommendConfig) -> None:
        super().__init__(config)
        strategy = config.logistic_regression
        if strategy is None:
            raise ValueError("logistic_regression 配置缺失，无法初始化推荐算法。")
        self.strategy: LogisticAlgorithmConfig = strategy
        self._model = None

    def fit(self, data: RecommendTrainingData) -> None:
        labeled = data.labeled
        dataset = data.dataset
        if labeled.is_empty():
            raise ValueError("没有有效的偏好数据，无法训练逻辑回归。")

        background = dataset.filter(~pl.col(ID).is_in(labeled[ID]))
        if background.is_empty():
            raise ValueError("缺少负样本候选，无法训练模型。")

        prefered_df = labeled.rename({EMBEDDING_VECTOR: "embedding"})
        background_df = background.rename({EMBEDDING_VECTOR: "embedding"})

        self._model = train_model(
            prefered_df=prefered_df,
            remaining_df=background_df,
            embedding_columns=["embedding"],
            config=self.strategy,
            seed=self.config.seed,
        )
        logger.info("逻辑回归算法训练完成。")

    def predict(self, data: RecommendPredictData) -> pl.DataFrame:
        if self._model is None:
            raise ValueError("尚未训练逻辑回归模型，请先调用 fit()。")

        frame = data.dataset
        if frame.is_empty():
            return frame

        embeddings = np.vstack(frame[EMBEDDING_VECTOR].to_numpy())
        embeddings = np.nan_to_num(embeddings, nan=0.0)
        scores = self._model.predict_proba(embeddings)[:, 1]

        predict_cfg = self.config.predict
        show_flags = adaptive_sample(
            scores,
            target_sample_rate=predict_cfg.sample_rate,
            high_threshold=predict_cfg.high_threshold,
            boundary_threshold=predict_cfg.boundary_threshold,
            random_state=self.config.seed,
        )

        result = frame.with_columns(
            pl.Series(SCORE, scores),
            pl.Series("show", show_flags.astype(np.int8)),
        )
        return result.sort(SCORE, descending=True)


__all__ = ["LogisticRegressionAlgorithm"]
