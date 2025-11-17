"""Shared utilities for prototype-based recommendation pipelines."""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any, Iterable

import hdbscan
import numpy as np
import polars as pl
import umap
from loguru import logger

from ..fields import PREFERENCE_DATE


def to_normalized_vectors(rows: Iterable[list[float]]) -> np.ndarray:
    """Convert embedding rows into L2-normalized numpy matrix."""
    rows_list = list(rows)
    if not rows_list:
        raise RuntimeError("没有可用嵌入。")
    matrix = np.asarray(rows_list, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-8, a_max=None)
    return matrix / norms


def maybe_reduce_for_clustering(
    vectors: np.ndarray,
    *,
    dim: int,
    n_neighbors: int,
    random_state: int = 42,
) -> np.ndarray:
    """Apply UMAP dimensionality reduction when it can aid clustering."""
    if dim <= 0 or vectors.shape[1] <= dim:
        return vectors
    effective_neighbors = max(5, min(n_neighbors, len(vectors) - 1))
    if effective_neighbors < 5:
        logger.warning("样本太少，跳过聚类降维。")
        return vectors
    reducer = umap.UMAP(
        n_neighbors=effective_neighbors,
        n_components=dim,
        min_dist=0.0,
        metric="cosine",
        random_state=random_state,
    )
    return reducer.fit_transform(vectors)


def run_hdbscan(
    vectors: np.ndarray,
    *,
    min_cluster_size: int,
    min_samples: int,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, hdbscan.HDBSCAN]:
    """Cluster normalized vectors with HDBSCAN."""
    if len(vectors) < min_cluster_size:
        raise RuntimeError(
            f"样本只有 {len(vectors)}，比 min_cluster_size={min_cluster_size} 还小。"
        )
    working_vectors = (
        vectors.astype(np.float64, copy=False) if metric != "euclidean" else vectors
    )
    algorithm = "best" if metric == "euclidean" else "generic"
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        algorithm=algorithm,
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(working_vectors)
    probabilities = clusterer.probabilities_
    return labels, probabilities, clusterer


def compute_ucb_allocations(
    df: pl.DataFrame,
    *,
    recency_days: int,
    coef: float,
    epsilon: float,
    budget: int,
    min_quota: int,
) -> tuple[list[dict[str, Any]], int]:
    """Compute UCB scores and quota allocations per cluster."""
    valid = df.filter(pl.col("cluster_label") >= 0)
    if valid.is_empty():
        return [], min_quota

    cutoff = date.today() - timedelta(days=max(recency_days, 1))
    stats = (
        valid.group_by("cluster_label")
        .agg(
            pl.len().alias("total_likes"),
            pl.when(pl.col(PREFERENCE_DATE) >= cutoff)
            .then(1)
            .otherwise(0)
            .sum()
            .alias("recent_likes"),
            pl.col("cluster_probability").mean().alias("mean_probability"),
        )
        .sort("cluster_label")
    )
    rows = stats.to_dicts()
    total_events = sum(row["total_likes"] for row in rows)
    log_total = math.log(max(total_events, 1) + 1.0) if total_events else 0.0

    for row in rows:
        total = row["total_likes"]
        recent = row["recent_likes"]
        ratio = recent / total if total else 0.0
        explore = coef * math.sqrt(log_total / (total + epsilon)) if total else coef
        row["recent_like_ratio"] = ratio
        row["ucb_score"] = ratio + explore
    allocated, effective_min = allocate_candidate_quota(
        rows,
        budget=budget,
        min_quota=min_quota,
    )
    return allocated, effective_min


def allocate_candidate_quota(
    rows: list[dict[str, Any]],
    *,
    budget: int,
    min_quota: int,
) -> tuple[list[dict[str, Any]], int]:
    """Distribute candidate budget across clusters according to UCB scores."""
    if not rows:
        return rows, min_quota
    cluster_count = len(rows)
    if budget <= 0:
        for row in rows:
            row["raw_quota"] = 0.0
            row["quota"] = 0
            row["quota_share"] = 0.0
            row["quota_fraction"] = 0.0
        return rows, 0

    effective_min = min_quota
    min_total = min_quota * cluster_count
    if min_total > budget:
        effective_min = max(budget // cluster_count, 0)

    sum_scores = sum(row["ucb_score"] for row in rows)
    if sum_scores <= 0:
        raw_quota = budget / cluster_count
        for row in rows:
            row["raw_quota"] = raw_quota
    else:
        for row in rows:
            row["raw_quota"] = budget * row["ucb_score"] / sum_scores

    for row in rows:
        floor_val = math.floor(row["raw_quota"])
        row["_floor_quota"] = floor_val
        row["quota_fraction"] = row["raw_quota"] - floor_val
        row["quota"] = max(effective_min, floor_val)

    total_alloc = sum(row["quota"] for row in rows)
    if total_alloc > budget:
        overflow = total_alloc - budget
        adjustable = sorted(
            rows,
            key=lambda item: item["quota"] - effective_min,
            reverse=True,
        )
        for row in adjustable:
            reducible = row["quota"] - effective_min
            if reducible <= 0:
                continue
            take = min(reducible, overflow)
            row["quota"] -= take
            overflow -= take
            if overflow <= 0:
                break
    elif total_alloc < budget:
        remainder = budget - total_alloc
        if remainder > 0:
            candidates = sorted(
                rows,
                key=lambda item: item["quota_fraction"],
                reverse=True,
            )
            idx = 0
            while remainder > 0 and candidates:
                row = candidates[idx % len(candidates)]
                row["quota"] += 1
                remainder -= 1
                idx += 1

    for row in rows:
        row["quota_share"] = row["quota"] / budget if budget > 0 else 0.0
        row["quota_fraction"] = float(row["quota_fraction"])
        row.pop("_floor_quota", None)
    return rows, effective_min


__all__ = [
    "allocate_candidate_quota",
    "compute_ucb_allocations",
    "maybe_reduce_for_clustering",
    "run_hdbscan",
    "to_normalized_vectors",
]
