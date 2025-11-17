"""Cluster preference embeddings with HDBSCAN and produce visualization/report."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import hdbscan
import matplotlib
import numpy as np
import polars as pl
import umap
from loguru import logger

from papersys.config import AppConfig
from papersys.const import DEFAULT_CONFIG_PATH
from papersys.data_sources import load_embeddings as load_hf_embeddings
from papersys.data_sources import load_metadata as load_hf_metadata
from papersys.fields import (
    CATEGORIES,
    EMBEDDING_VECTOR,
    ID,
    PREFERENCE,
    PREFERENCE_DATE,
    TITLE,
)
from papersys.recommend.cluster_utils import (
    allocate_candidate_quota,
    compute_ucb_allocations,
    maybe_reduce_for_clustering,
    run_hdbscan,
    to_normalized_vectors,
)
from papersys.storage.git_store import GitStore

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(slots=True)
class Args:
    config: Path
    preference_label: str
    min_cluster_size: int
    min_samples: int | None
    cluster_dim: int
    cluster_n_neighbors: int
    cluster_metric: str
    viz_n_neighbors: int
    sample_size: int
    plot_path: Path
    report_path: Path
    cluster_cache: Path | None
    load_cache: bool
    save_cache: bool
    ucb_coef: float
    ucb_recency_days: int
    ucb_epsilon: float
    candidate_budget: int
    min_quota: int


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="对偏好样本做 HDBSCAN 聚类，可视化并输出代表样本"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="配置文件路径 (默认: config.toml)",
    )
    parser.add_argument(
        "--preference-label",
        default="like",
        help="只聚类该偏好标签的样本 (默认: like)",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=4,
        help="HDBSCAN 的 min_cluster_size (默认: 4，推荐用于 10+ 簇)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=2,
        help="HDBSCAN 的 min_samples (默认: 2)",
    )
    parser.add_argument(
        "--cluster-dim",
        type=int,
        default=50,
        help="聚类前的 UMAP 降维维度，<=0 表示不降维 (默认: 50)",
    )
    parser.add_argument(
        "--cluster-n-neighbors",
        type=int,
        default=40,
        help="聚类用 UMAP 的 n_neighbors (默认: 40)",
    )
    parser.add_argument(
        "--cluster-metric",
        choices=("cosine", "euclidean"),
        default="cosine",
        help="HDBSCAN 距离度量 (默认: cosine)",
    )
    parser.add_argument(
        "--viz-n-neighbors",
        type=int,
        default=20,
        help="可视化 UMAP 的 n_neighbors (默认: 20)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="每个簇输出多少代表样本 (默认: 5)",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("docs/images/preference_clusters.png"),
        help="UMAP 可视化图片输出路径",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("docs/reports/preference_clusters.json"),
        help="聚类结果 JSON 报告路径",
    )
    parser.add_argument(
        "--cluster-cache",
        type=Path,
        default=None,
        help="聚类结果缓存路径（.parquet），供多脚本共享",
    )
    parser.add_argument(
        "--load-cache",
        action="store_true",
        help="从 --cluster-cache 读取现有聚类结果，跳过重新聚类",
    )
    parser.add_argument(
        "--save-cache",
        action="store_true",
        help="将本次聚类的 enriched 数据写入 --cluster-cache",
    )
    parser.add_argument(
        "--ucb-coef",
        type=float,
        default=0.7,
        help="UCB 探索系数 c (默认: 0.7)",
    )
    parser.add_argument(
        "--ucb-recency-days",
        type=int,
        default=30,
        help="最近点赞窗口天数，用于估计 UCB 成功率 (默认: 30)",
    )
    parser.add_argument(
        "--ucb-epsilon",
        type=float,
        default=1.0,
        help="UCB 分母平滑常数 epsilon (默认: 1.0)",
    )
    parser.add_argument(
        "--candidate-budget",
        type=int,
        default=200,
        help="单轮候选预算 B，用于根据 UCB 分配配额 (默认: 200)",
    )
    parser.add_argument(
        "--min-quota",
        type=int,
        default=10,
        help="每个聚类最少候选数 (默认: 10)",
    )
    raw = parser.parse_args()
    min_samples = raw.min_samples if raw.min_samples is not None else raw.min_cluster_size
    return Args(
        config=raw.config,
        preference_label=raw.preference_label,
        min_cluster_size=raw.min_cluster_size,
        min_samples=min_samples,
        cluster_dim=raw.cluster_dim,
        cluster_n_neighbors=raw.cluster_n_neighbors,
        cluster_metric=raw.cluster_metric,
        viz_n_neighbors=raw.viz_n_neighbors,
        sample_size=raw.sample_size,
        plot_path=raw.plot_path,
        report_path=raw.report_path,
        cluster_cache=raw.cluster_cache,
        load_cache=raw.load_cache,
        save_cache=raw.save_cache,
        ucb_coef=raw.ucb_coef,
        ucb_recency_days=raw.ucb_recency_days,
        ucb_epsilon=raw.ucb_epsilon,
        candidate_budget=raw.candidate_budget,
        min_quota=raw.min_quota,
    )


def main() -> None:
    args = parse_args()
    app_config = AppConfig.from_toml(args.config)

    git_store = GitStore(app_config.git_store)
    git_store.ensure_local_copy()

    (
        enriched,
        vectors,
        labels,
        probabilities,
        clusterer,
    ) = load_or_cluster(app_config, git_store, args)

    plot_clusters(enriched, args.plot_path)
    cluster_summaries = build_cluster_summaries(
        enriched,
        vectors,
        labels,
        probabilities,
        sample_size=args.sample_size,
    )
    ucb_allocations, effective_min_quota = compute_ucb_allocations(
        enriched,
        recency_days=args.ucb_recency_days,
        coef=args.ucb_coef,
        epsilon=args.ucb_epsilon,
        budget=args.candidate_budget,
        min_quota=args.min_quota,
    )
    cluster_summaries = merge_ucb_stats(cluster_summaries, ucb_allocations)
    report = build_report_dict(
        args=args,
        total_points=enriched.height,
        noise_count=int((labels == -1).sum()),
        cluster_summaries=cluster_summaries,
        clusterer=clusterer,
        effective_min_quota=effective_min_quota,
    )
    write_report(args.report_path, report)
    print_human_summary(report)
    logger.info(
        "完成聚类：clusters={} noise={} plot={} report={}",
        len(report["clusters"]),
        report["noise_count"],
        args.plot_path,
        args.report_path,
    )


def load_preferences(path: Path, label: str) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Preference CSV 不存在：{path}")
    df = pl.read_csv(
        path,
        schema_overrides={ID: pl.String, PREFERENCE: pl.String, PREFERENCE_DATE: pl.String},
    )
    if df.is_empty():
        raise RuntimeError("preference.csv 为空，没法聚类。")
    df = df.filter(pl.col(PREFERENCE) == label)
    if df.is_empty():
        raise RuntimeError(f"没有偏好标签 '{label}' 的记录。")
    today = date.today()
    if PREFERENCE_DATE not in df.columns:
        logger.warning("偏好文件缺少 {} 列，使用今天 {} 作为占位。", PREFERENCE_DATE, today)
        df = df.with_columns(pl.lit(today).cast(pl.Date).alias(PREFERENCE_DATE))
    else:
        df = df.with_columns(
            pl.col(PREFERENCE_DATE)
            .str.strptime(pl.Date, strict=False)
            .alias(PREFERENCE_DATE)
        )
    df = (
        df.sort([PREFERENCE_DATE, ID])
        .unique(subset=[ID], keep="last")
        .sort([PREFERENCE_DATE, ID])
    )
    return df


def collect_embeddings(config, id_list: list[str]) -> pl.DataFrame:
    if not id_list:
        raise RuntimeError("没有 ID 可用于加载嵌入。")
    lf = load_hf_embeddings(
        config,
        lazy=True,
        columns=[ID, EMBEDDING_VECTOR],
    )
    return (
        lf.filter(pl.col(ID).is_in(id_list))
        .select([ID, EMBEDDING_VECTOR])
        .collect()
    )


def collect_metadata(config, id_list: list[str]) -> pl.DataFrame:
    if not id_list:
        raise RuntimeError("没有 ID 可用于加载元数据。")
    lf = load_hf_metadata(config, lazy=True)
    return (
        lf.filter(pl.col(ID).is_in(id_list))
        .select([ID, TITLE, "abstract", CATEGORIES])
        .collect()
    )


def load_or_cluster(
    app_config: AppConfig,
    git_store: GitStore,
    args: Args,
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray, np.ndarray, hdbscan.HDBSCAN | None]:
    if args.load_cache:
        if args.cluster_cache is None or not args.cluster_cache.exists():
            raise FileNotFoundError("--load-cache 需要有效的 --cluster-cache 路径。")
        enriched = pl.read_parquet(args.cluster_cache)
        vectors = to_normalized_vectors(enriched[EMBEDDING_VECTOR].to_list())
        labels = enriched["cluster_label"].to_numpy()
        probabilities = enriched["cluster_probability"].to_numpy()
        return enriched, vectors, labels, probabilities, None

    preference_df = load_preferences(git_store.preference_path, args.preference_label)
    embeddings_df = collect_embeddings(app_config.embedding, preference_df[ID].to_list())

    df = preference_df.join(embeddings_df, on=ID, how="inner")
    if df.is_empty():
        raise RuntimeError("找不到和偏好匹配的嵌入，检查数据目录。")

    vectors = to_normalized_vectors(df[EMBEDDING_VECTOR].to_list())
    cluster_vectors = maybe_reduce_for_clustering(
        vectors,
        dim=args.cluster_dim,
        n_neighbors=args.cluster_n_neighbors,
    )
    labels, probabilities, clusterer = run_hdbscan(
        cluster_vectors,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric=args.cluster_metric,
    )

    umap_points = project_umap(vectors, n_neighbors=args.viz_n_neighbors)
    context_df = collect_metadata(app_config.metadata, df[ID].to_list())
    enriched = (
        df.join(context_df, on=ID, how="left")
        .with_columns(
            pl.Series("cluster_label", labels),
            pl.Series("cluster_probability", probabilities),
            pl.Series("umap_x", umap_points[:, 0]),
            pl.Series("umap_y", umap_points[:, 1]),
        )
    )
    if args.save_cache and args.cluster_cache is not None:
        args.cluster_cache.parent.mkdir(parents=True, exist_ok=True)
        enriched.write_parquet(args.cluster_cache)
    return enriched, vectors, labels, probabilities, clusterer


def project_umap(vectors: np.ndarray, *, n_neighbors: int) -> np.ndarray:
    if len(vectors) < 2:
        return np.zeros((len(vectors), 2), dtype=np.float32)
    neighbors = max(2, min(n_neighbors, len(vectors) - 1))
    reducer = umap.UMAP(
        n_neighbors=neighbors,
        n_components=2,
        min_dist=0.15,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(vectors)


def plot_clusters(df: pl.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = df["cluster_label"].to_numpy()
    points = df.select(["umap_x", "umap_y"]).to_numpy()
    unique_labels = sorted(set(labels))

    cmap = matplotlib.colormaps["tab20"].resampled(max(len(unique_labels), 1))
    fig, ax = plt.subplots(figsize=(8, 6))

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        color = "lightgray" if label == -1 else cmap(idx)
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            s=36,
            c=[color],
            alpha=0.75,
            label="noise" if label == -1 else f"cluster {label}",
            edgecolors="none",
        )

    ax.set_title("Preference Clusters via HDBSCAN + UMAP")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.2, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_cluster_summaries(
    df: pl.DataFrame,
    vectors: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    sample_size: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    unique_labels = [label for label in sorted(set(labels)) if label >= 0]
    for label in unique_labels:
        mask = labels == label
        cluster_indices = np.nonzero(mask)[0]
        cluster_vectors = vectors[cluster_indices]
        centroid = cluster_vectors.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid /= centroid_norm
        sims = cluster_vectors @ centroid
        order = cluster_indices[np.argsort(-sims)[:sample_size]]

        samples = [row_payload(df.row(idx, named=True)) for idx in order]
        cat_tokens = collect_tokens(df[CATEGORIES].gather(cluster_indices).to_list())
        mean_prob = float(probabilities[cluster_indices].mean())
        results.append(
            {
                "label": int(label),
                "size": int(mask.sum()),
                "mean_probability": mean_prob,
                "top_categories": cat_tokens[:5],
                "representative_ids": [sample["id"] for sample in samples],
                "samples": samples,
            }
        )
    return results


def row_payload(row: dict[str, Any]) -> dict[str, Any]:
    title = row.get(TITLE) or "未知标题"
    summary = truncate_text(row.get("abstract", ""))
    categories = normalize_list(row.get(CATEGORIES))
    return {
        "id": row[ID],
        "title": title,
        "summary": summary,
        "categories": categories,
        "cluster_probability": float(row.get("cluster_probability", 0.0)),
    }


def truncate_text(text: str | None, limit: int = 240) -> str:
    if not text:
        return ""
    clean = text.replace("\n", " ").strip()
    if len(clean) <= limit:
        return clean
    return f"{clean[:limit].rstrip()}…"


def normalize_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def collect_tokens(values: list[Any]) -> list[str]:
    counter: Counter[str] = Counter()
    for value in values:
        if not value:
            continue
        tokens = value if isinstance(value, list) else [value]
        for token in tokens:
            token = str(token).strip()
            if token:
                counter[token] += 1
    return [token for token, _ in counter.most_common()]


def merge_ucb_stats(
    summaries: list[dict[str, Any]],
    allocations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    mapping = {row["cluster_label"]: row for row in allocations}
    for summary in summaries:
        stats = mapping.get(summary["label"])
        if not stats:
            continue
        summary.update(
            {
                "total_likes": stats["total_likes"],
                "recent_likes": stats["recent_likes"],
                "recent_like_ratio": stats["recent_like_ratio"],
                "ucb_score": stats["ucb_score"],
                "candidate_quota": stats["quota"],
                "quota_share": stats["quota_share"],
            }
        )
    return summaries


def build_report_dict(
    *,
    args: Args,
    total_points: int,
    noise_count: int,
    cluster_summaries: list[dict[str, Any]],
    clusterer: hdbscan.HDBSCAN,
    effective_min_quota: int,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preference_label": args.preference_label,
        "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples,
        "cluster_dim": args.cluster_dim,
        "cluster_n_neighbors": args.cluster_n_neighbors,
        "cluster_metric": args.cluster_metric,
        "viz_n_neighbors": args.viz_n_neighbors,
        "sample_size": args.sample_size,
        "total_points": total_points,
        "noise_count": noise_count,
        "cluster_persistence": getattr(clusterer, "cluster_persistence_", []).tolist()
        if hasattr(clusterer, "cluster_persistence_")
        else [],
        "plot_path": str(args.plot_path),
        "report_path": str(args.report_path),
        "ucb": {
            "coef": args.ucb_coef,
            "recency_days": args.ucb_recency_days,
            "epsilon": args.ucb_epsilon,
            "budget": args.candidate_budget,
            "min_quota": args.min_quota,
            "effective_min_quota": effective_min_quota,
        },
        "clusters": cluster_summaries,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def print_human_summary(report: dict[str, Any]) -> None:
    header = (
        f"[{report['preference_label']}] 样本 {report['total_points']} 条 → "
        f"{len(report['clusters'])} 个簇 (noise={report['noise_count']})"
    )
    print(header)
    ucb_info = report.get("ucb")
    if ucb_info:
        print(
            "UCB: "
            f"c={ucb_info['coef']} "
            f"window={ucb_info['recency_days']}d "
            f"B={ucb_info['budget']} "
            f"min={ucb_info['effective_min_quota']}"
        )
    for cluster in report["clusters"]:
        quota = cluster.get("candidate_quota")
        ucb_score = cluster.get("ucb_score")
        ratio = cluster.get("recent_like_ratio")
        ucb_text = f"{ucb_score:.3f}" if isinstance(ucb_score, (int, float)) else "n/a"
        ratio_text = f"{ratio:.2f}" if isinstance(ratio, (int, float)) else "n/a"
        quota_text = quota if isinstance(quota, int) else 0
        print(
            f"- cluster {cluster['label']} "
            f"(n={cluster['size']}, mean_p={cluster['mean_probability']:.2f}, "
            f"ucb={ucb_text}, recent={ratio_text}, quota={quota_text}) "
            f"cats={', '.join(cluster['top_categories'][:3])}"
        )
        for sample in cluster["samples"]:
            summary = sample["summary"] or ""
            print(f"  · {sample['id']} {sample['title']} :: {summary[:120]}")


if __name__ == "__main__":
    main()
