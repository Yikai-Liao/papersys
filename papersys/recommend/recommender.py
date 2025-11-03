"""Recommendation pipeline using HuggingFace datasets and JSONL preferences."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable, Sequence, Set

import numpy as np
import polars as pl
from loguru import logger

from ..config import RecommendConfig
from ..fields import (
    AUTHORS,
    CATEGORIES,
    EMBEDDING_VECTOR,
    ID,
    PREFERENCE,
    PUBLISH_DATE,
    SCORE,
    TITLE,
    UPDATE_DATE,
)
from .sampler import adaptive_sample
from .trainer import train_model


@dataclass(slots=True)
class RecommendationResult:
    """Simple container for prediction outputs."""

    frame: pl.DataFrame


class Recommender:
    """Train and score recommendations from in-memory datasets."""

    def __init__(
        self,
        *,
        metadata: pl.DataFrame,
        embeddings: pl.DataFrame,
        preferences: pl.DataFrame,
        excluded_ids: Set[str],
        config: RecommendConfig,
    ) -> None:
        self.config = config
        self._metadata = metadata
        self._embeddings = embeddings
        self._preferences = preferences
        self._preference_ids = set(preferences[ID].to_list()) if ID in preferences.columns else set()
        self._excluded_ids = set(excluded_ids) | self._preference_ids
        self.model = None

        logger.info(
            "Loaded datasets: metadata={}, embeddings={}, preferences={}",
            metadata.height,
            embeddings.height,
            preferences.height,
        )

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def fit(self, categories: Sequence[str]) -> "Recommender":
        """Train logistic regression model."""

        dataset = self._prepare_dataset(categories)
        if dataset.is_empty():
            raise ValueError("没有符合类别筛选的候选论文。")

        labeled = self._prepare_preferences(dataset)
        if labeled.is_empty():
            raise ValueError("没有有效的偏好数据，无法训练。")

        positive = labeled.filter(pl.col(PREFERENCE) == "like")
        if positive.is_empty():
            raise ValueError("偏好数据中缺少正向样本（like）。")

        candidate_background = dataset.filter(
            ~pl.col(ID).is_in(labeled[ID])
        )
        if candidate_background.is_empty():
            raise ValueError("缺少负样本候选，无法训练模型。")

        logger.info(
            "Training dataset prepared: positives={}, background={}",
            positive.height,
            candidate_background.height,
        )

        prefered_df = labeled.rename({EMBEDDING_VECTOR: "embedding"})
        background_df = candidate_background.rename({EMBEDDING_VECTOR: "embedding"})

        self.model = train_model(
            prefered_df=prefered_df,
            remaining_df=background_df,
            embedding_columns=["embedding"],
            config=self.config,
        )
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
    ) -> RecommendationResult:
        """Score candidate papers and return recommendation dataframe."""

        if self.model is None:
            raise ValueError("尚未训练模型，请先调用 fit()。")

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

        embeddings = np.vstack(dataset[EMBEDDING_VECTOR].to_numpy())
        embeddings = np.nan_to_num(embeddings, nan=0.0)

        scores = self.model.predict_proba(embeddings)[:, 1]
        predict_cfg = self.config.predict

        show_flags = adaptive_sample(
            scores,
            target_sample_rate=predict_cfg.sample_rate,
            high_threshold=predict_cfg.high_threshold,
            boundary_threshold=predict_cfg.boundary_threshold,
            random_state=self.config.seed,
        )

        result = dataset.with_columns(
            pl.Series(SCORE, scores),
            pl.Series("show", show_flags.astype(np.int8)),
        )
        result = result.sort(SCORE, descending=True)

        logger.info(
            "Predicted {} papers, recommended {} ({}%).",
            result.height,
            int(show_flags.sum()),
            (
                float(show_flags.sum()) / float(len(show_flags)) * 100
                if len(show_flags)
                else 0.0
            ),
        )

        return RecommendationResult(result)

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

        dataset = meta.join(self._embeddings, on=ID, how="inner")
        return dataset

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


__all__ = ["Recommender", "RecommendationResult"]
