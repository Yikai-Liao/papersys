"""Recommendation pipeline using HuggingFace datasets and JSONL preferences."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable, Sequence, Set

import numpy as np
import polars as pl
from loguru import logger

from ..config import RecommendConfig
from ..fields import CATEGORIES, EMBEDDING_VECTOR, ID, PREFERENCE, UPDATE_DATE
from .algorithms import (
    BaseRecommendAlgorithm,
    RecommendPredictData,
    RecommendTrainingData,
    create_algorithm,
)


@dataclass(slots=True)
class RecommendationResult:
    """Simple container for prediction outputs."""

    frame: pl.DataFrame


class Recommender:
    """Train and score recommendations from in-memory datasets."""

    def __init__(
        self,
        *,
        metadata: pl.DataFrame | pl.LazyFrame,
        embeddings: pl.DataFrame | pl.LazyFrame,
        preferences: pl.DataFrame,
        excluded_ids: Set[str],
        config: RecommendConfig,
    ) -> None:
        self.config = config
        self._metadata = self._to_lazy(metadata)
        self._embeddings = self._to_lazy(embeddings)
        self._preferences = preferences
        self._preference_ids = set(preferences[ID].to_list()) if ID in preferences.columns else set()
        self._excluded_ids = set(excluded_ids) | self._preference_ids
        self._algorithm: BaseRecommendAlgorithm = create_algorithm(config)

        logger.info(
            "Loaded datasets (lazy metadata={}, lazy embeddings={}, preferences={})",
            isinstance(self._metadata, pl.LazyFrame),
            isinstance(self._embeddings, pl.LazyFrame),
            preferences.height,
        )

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def fit(self, categories: Sequence[str]) -> "Recommender":
        """Prepare dataset slices and delegate to configured algorithm."""

        dataset = self._prepare_dataset(categories)
        if dataset.is_empty():
            raise ValueError("没有符合类别筛选的候选论文。")

        labeled = self._prepare_preferences(dataset)
        if labeled.is_empty():
            raise ValueError("没有有效的偏好数据，无法训练。")

        positive = labeled.filter(pl.col(PREFERENCE) == "like")
        if positive.is_empty():
            raise ValueError("偏好数据中缺少正向样本（like）。")

        logger.info(
            "Training dataset prepared: labeled={}, positives={}",
            labeled.height,
            positive.height,
        )

        training_data = RecommendTrainingData(
            dataset=dataset,
            labeled=labeled,
            positive=positive,
        )
        self._algorithm.fit(training_data)
        return self

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #

    def predict(
        self,
        categories: Sequence[str],
        *,
        last_n_days: int | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int | None = None,
    ) -> RecommendationResult:
        """Score candidate papers via configured algorithm."""

        start_date, end_date = self._resolve_date_range(last_n_days, start_date, end_date)
        dataset = self._prepare_dataset(categories, start_date=start_date, end_date=end_date)

        if dataset.is_empty():
            logger.warning("筛选后无可推荐的候选论文。")
            return RecommendationResult(pl.DataFrame())

        if self._preference_ids:
            dataset = dataset.filter(~pl.col(ID).is_in(list(self._preference_ids)))
        if self._excluded_ids:
            dataset = dataset.filter(~pl.col(ID).is_in(list(self._excluded_ids)))

        if dataset.is_empty():
            logger.warning("候选论文全部在已处理列表中。")
            return RecommendationResult(pl.DataFrame())

        dataset = self._filter_nan_embeddings(dataset)
        if dataset.is_empty():
            logger.warning("过滤非法向量后无候选论文。")
            return RecommendationResult(pl.DataFrame())

        prediction = self._algorithm.predict(
            RecommendPredictData(dataset=dataset, limit=limit),
        )
        if prediction.is_empty():
            logger.warning("算法未返回任何推荐结果。")
            return RecommendationResult(prediction)

        recommended = int(prediction["show"].sum()) if "show" in prediction.columns else 0

        logger.info(
            "Predicted {} papers, recommended {} ({}%).",
            prediction.height,
            recommended,
            (
                float(recommended) / float(prediction.height) * 100
                if prediction.height
                else 0.0
            ),
        )

        return RecommendationResult(prediction)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _prepare_dataset(
        self,
        categories: Sequence[str],
        *,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pl.DataFrame:
        meta = self._metadata
        if categories:
            checks = [pl.element().str.starts_with(cat) for cat in categories]
            meta = meta.filter(
                pl.col(CATEGORIES)
                .list.eval(pl.any_horizontal(checks))
                .list.any()
            )

        if start_date is not None:
            meta = meta.filter(pl.col(UPDATE_DATE) >= start_date)
        if end_date is not None:
            meta = meta.filter(pl.col(UPDATE_DATE) <= end_date)

        dataset_lazy = meta.join(self._embeddings, on=ID, how="inner")
        return dataset_lazy.collect(streaming=True)

    def _prepare_preferences(self, dataset: pl.DataFrame) -> pl.DataFrame:
        if self._preferences.is_empty():
            return pl.DataFrame()
        labeled = self._preferences.join(dataset, on=ID, how="inner")
        return labeled

    def _filter_nan_embeddings(self, frame: pl.DataFrame) -> pl.DataFrame:
        vectors = frame[EMBEDDING_VECTOR].to_list()
        mask = np.array(
            [
                vec is None
                or (isinstance(vec, (list, np.ndarray)) and np.isnan(vec).any())
                for vec in vectors
            ],
            dtype=bool,
        )
        removed = int(mask.sum())
        if removed:
            logger.warning(
                "移除包含 NaN 的向量 {} 条，占比 {:.2f}%",
                removed,
                removed / len(mask) * 100,
            )
            frame = (
                frame.with_row_index("__idx__")
                .filter(pl.col("__idx__").is_in(np.where(~mask)[0].tolist()))
                .drop("__idx__")
            )
        return frame

    def _resolve_date_range(
        self,
        last_n_days: int | None,
        start_date: date | None,
        end_date: date | None,
    ) -> tuple[date | None, date | None]:
        if last_n_days is not None:
            end = date.today()
            start = end - timedelta(days=last_n_days)
            return start, end
        return start_date, end_date

    @staticmethod
    def _to_lazy(frame: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
        if isinstance(frame, pl.LazyFrame):
            return frame
        return frame.lazy()


__all__ = ["Recommender", "RecommendationResult"]
