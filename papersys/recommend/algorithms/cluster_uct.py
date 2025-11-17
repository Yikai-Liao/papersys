"""Cluster + UCB based recommendation strategy."""

from __future__ import annotations

from datetime import date, datetime
from typing import Sequence

import numpy as np
import polars as pl
from loguru import logger

from ...config import ClusterUctConfig, RecommendConfig
from ...fields import EMBEDDING_VECTOR, ID, PREFERENCE_DATE, SCORE
from ..cluster_utils import (
    compute_ucb_allocations,
    maybe_reduce_for_clustering,
    run_hdbscan,
    to_normalized_vectors,
)
from .base import BaseRecommendAlgorithm, RecommendPredictData, RecommendTrainingData


class ClusterUctAlgorithm(BaseRecommendAlgorithm):
    """Cluster preference prototypes and allocate quotas via UCB."""

    def __init__(self, config: RecommendConfig) -> None:
        super().__init__(config)
        strategy = config.cluster_uct
        if strategy is None:
            raise ValueError("cluster_uct configuration is missing; cannot initialize recommender.")
        self.strategy: ClusterUctConfig = strategy
        self._prototypes: np.ndarray | None = None
        self._cluster_labels: np.ndarray | None = None
        self._positive_assignments: pl.DataFrame | None = None

    # ------------------------------------------------------------------ #
    # BaseRecommendAlgorithm API
    # ------------------------------------------------------------------ #

    def fit(self, data: RecommendTrainingData) -> None:
        positive = self._prepare_positive_samples(data.positive)
        if positive.is_empty():
            raise ValueError("No positive feedback samples available to build prototypes.")

        normalized = to_normalized_vectors(positive[EMBEDDING_VECTOR].to_list())
        cluster_vectors = maybe_reduce_for_clustering(
            normalized,
            dim=self.strategy.cluster_dim,
            n_neighbors=self.strategy.cluster_n_neighbors,
            random_state=self.config.seed,
        )

        labels, probabilities, _ = run_hdbscan(
            cluster_vectors,
            min_cluster_size=self.strategy.min_cluster_size,
            min_samples=self.strategy.min_cluster_size,
            metric=self.strategy.cluster_metric,
        )
        cluster_ids = sorted({int(label) for label in labels if label >= 0})
        if not cluster_ids:
            raise ValueError("HDBSCAN returned only noise; cannot construct interest prototypes.")

        centroids = self._build_centroids(normalized, labels, cluster_ids)
        self._prototypes = np.vstack(centroids)
        self._cluster_labels = np.asarray(cluster_ids, dtype=np.int32)
        self._positive_assignments = positive.with_columns(
            pl.Series("cluster_label", labels.astype(np.int32)),
            pl.Series("cluster_probability", probabilities),
        )

        logger.info(
            "Cluster-UCT fitting complete: clusters={} likes={}",
            len(cluster_ids),
            positive.height,
        )
        self._log_cluster_stats()

    def predict(self, data: RecommendPredictData) -> pl.DataFrame:
        if self._prototypes is None or self._cluster_labels is None:
            raise ValueError("Prototypes are not ready; call fit() first.")
        frame = data.dataset
        if frame.is_empty():
            return frame

        vectors = to_normalized_vectors(frame[EMBEDDING_VECTOR].to_list())
        sims = vectors @ self._prototypes.T
        scores, weights = self._aggregate_scores(sims, return_weights=True)
        primary_indices = np.argmax(sims, axis=1)
        primary_labels = self._cluster_labels[primary_indices]

        scored = frame.with_columns(
            pl.Series("cluster_label", primary_labels.astype(np.int32)),
            pl.Series(SCORE, scores),
        )
        self._log_score_contributions(scored, weights)

        budget = self._compute_budget(scored.height, data.limit)
        if budget <= 0:
            logger.info("Quota budget is zero; returning empty recommendations.")
            result = self._mark_top_n(scored, 0)
        else:
            allocations = self._compute_allocations(budget)
            if allocations:
                result = self._apply_cluster_quotas(scored, allocations)
            else:
                logger.warning("Could not compute UCB allocation; falling back to global top-N.")
                result = self._mark_top_n(scored, budget)

        return result.sort(SCORE, descending=True)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _prepare_positive_samples(self, frame: pl.DataFrame) -> pl.DataFrame:
        if frame.is_empty():
            return frame

        df = frame
        if PREFERENCE_DATE not in df.columns:
            df = df.with_columns(pl.lit(date.today()).alias(PREFERENCE_DATE))
        else:
            dtype = df.schema.get(PREFERENCE_DATE)
            if dtype == pl.Date:
                series = pl.col(PREFERENCE_DATE).fill_null(date.today())
            elif dtype == pl.Datetime:
                series = (
                    pl.col(PREFERENCE_DATE)
                    .cast(pl.Date)
                    .fill_null(date.today())
                )
            else:
                series = (
                    pl.col(PREFERENCE_DATE)
                    .cast(pl.Utf8)
                    .str.strptime(pl.Date, strict=False)
                    .fill_null(date.today())
                )
            df = df.with_columns(series.alias(PREFERENCE_DATE))

        df = df.filter(pl.col(EMBEDDING_VECTOR).is_not_null())
        df = (
            df.sort([PREFERENCE_DATE, ID])
            .unique(subset=[ID], keep="last")
            .sort([PREFERENCE_DATE, ID])
        )
        return df

    def _build_centroids(
        self,
        normalized_vectors: np.ndarray,
        labels: np.ndarray,
        cluster_ids: Sequence[int],
    ) -> list[np.ndarray]:
        centroids: list[np.ndarray] = []
        for label in cluster_ids:
            mask = labels == label
            vectors = normalized_vectors[mask]
            centroid = vectors.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            centroids.append(centroid.astype(np.float32, copy=False))
        return centroids

    def _aggregate_scores(
        self, sims: np.ndarray, *, return_weights: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        tau = max(self.strategy.prototype_temperature, 1e-6)
        scaled = sims * tau
        max_vals = np.max(scaled, axis=1, keepdims=True)
        stable = scaled - max_vals
        exp_vals = np.exp(stable)
        sum_exp = np.clip(exp_vals.sum(axis=1, keepdims=True), a_min=1e-12, a_max=None)
        logs = np.log(sum_exp)
        aggregated = (max_vals + logs) / tau
        weights = exp_vals / sum_exp
        if return_weights:
            return (
                aggregated.squeeze().astype(np.float32, copy=False),
                weights.astype(np.float32, copy=False),
            )
        return aggregated.squeeze().astype(np.float32, copy=False)

    def _apply_cluster_quotas(
        self,
        frame: pl.DataFrame,
        allocations: list[dict[str, int]],
    ) -> pl.DataFrame:
        quota_df = pl.DataFrame(
            {
                "cluster_label": [int(row["cluster_label"]) for row in allocations],
                "quota": [int(row["quota"]) for row in allocations],
            }
        )
        ranked = (
            frame.sort(["cluster_label", SCORE], descending=[False, True])
            .join(quota_df, on="cluster_label", how="left")
            .with_columns(pl.col("quota").fill_null(0))
        )
        partitions = ranked.partition_by("cluster_label", maintain_order=True, as_dict=False)
        annotated: list[pl.DataFrame] = []
        for part in partitions:
            quota = int(part["quota"][0]) if part["quota"].len() else 0
            limit = max(min(quota, part.height), 0)
            if limit <= 0:
                annotated.append(
                    part.with_columns(pl.lit(0, dtype=pl.Int8).alias("show"))
                )
                continue
            show = np.zeros(part.height, dtype=np.int8)
            show[:limit] = 1
            annotated.append(part.with_columns(pl.Series("show", show)))
        combined = pl.concat(annotated) if annotated else ranked
        return combined.drop("quota")

    def _mark_top_n(self, frame: pl.DataFrame, limit: int | None) -> pl.DataFrame:
        if limit is None:
            return frame.with_columns(pl.lit(1, dtype=pl.Int8).alias("show"))
        effective = max(0, min(limit, frame.height))
        ranked = (
            frame.sort(SCORE, descending=True)
            .with_row_count("_rank")
            .with_columns(
                pl.when(pl.col("_rank") < effective)
                .then(pl.lit(1, dtype=pl.Int8))
                .otherwise(pl.lit(0, dtype=pl.Int8))
                .alias("show")
            )
            .drop("_rank")
        )
        return ranked

    def _compute_allocations(self, budget: int) -> list[dict[str, int]]:
        if self._positive_assignments is None or budget <= 0:
            return []
        allocations, _ = compute_ucb_allocations(
            self._positive_assignments,
            recency_days=self.strategy.ucb_recency_days,
            coef=self.strategy.ucb_coef,
            epsilon=self.strategy.ucb_epsilon,
            budget=budget,
            min_quota=0,
        )
        return allocations

    def _compute_budget(self, candidate_count: int, limit: int | None) -> int:
        if candidate_count <= 0:
            return 0
        predict_cfg = self.config.predict
        target = max(int(candidate_count * predict_cfg.sample_rate), 1)
        if limit is not None:
            target = min(target, limit)
        target = min(target, candidate_count)
        return max(target, 0)

    @staticmethod
    def _coerce_date(value: object, fallback: date) -> date:
        if isinstance(value, date):
            return value
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, str):
            try:
                return date.fromisoformat(value[:10])
            except ValueError:
                return fallback
        return fallback

    def _log_cluster_stats(self) -> None:
        if self._positive_assignments is None:
            return
        stats = (
            self._positive_assignments.group_by("cluster_label")
            .agg(pl.len().alias("size"))
            .sort("cluster_label")
        )
        total = stats["size"].sum()
        summary = [
            {
                "cluster": int(row[0]),
                "size": int(row[1]),
                "ratio": round(row[1] / total, 4) if total else 0.0,
            }
            for row in stats.iter_rows()
        ]
        logger.info("Cluster size summary: {}", summary)

    def _log_score_contributions(
        self, scored: pl.DataFrame, weights: np.ndarray
    ) -> None:
        if scored.is_empty():
            return
        top_k = min(5, scored.height)
        if top_k <= 0:
            return
        score_array = scored[SCORE].to_numpy()
        top_idx = np.argsort(-score_array)[:top_k]
        for idx, row_idx in enumerate(top_idx, start=1):
            row = scored.row(int(row_idx), named=True)
            contributions = sorted(
                zip(self._cluster_labels, weights[row_idx]),
                key=lambda item: item[1],
                reverse=True,
            )
            top_contribs = [
                {"cluster": int(label), "weight": round(float(weight), 4)}
                for label, weight in contributions[:3]
            ]
            logger.info(
                "Top#{}/{} id={} score={:.4f} contributions={}",
                idx,
                top_k,
                row.get(ID, "<unknown>"),
                float(row[SCORE]),
                top_contribs,
            )


__all__ = ["ClusterUctAlgorithm"]
