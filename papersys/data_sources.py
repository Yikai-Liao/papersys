from __future__ import annotations

from pathlib import Path
from typing import Iterable

import polars as pl
from huggingface_hub import snapshot_download
from loguru import logger

from .config import EmbeddingConfig, HuggingFaceDatasetConfig, MetadataConfig
from .fields import ID

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "hf_cache"


def _ensure_cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def _repo_cache_dir(config: HuggingFaceDatasetConfig) -> Path:
    """Return the local cache directory for a HF dataset repo."""

    safe_name = config.hf_repo.replace("/", "__")
    repo_dir = _ensure_cache_dir() / safe_name
    repo_dir.mkdir(parents=True, exist_ok=True)
    return repo_dir


def _snapshot_shards(config: HuggingFaceDatasetConfig) -> str:
    """Download shard files and return a glob pattern for them."""

    repo_dir = _repo_cache_dir(config)
    filename_pattern = f"{config.shard_prefix}_*.parquet"

    snapshot_download(
        repo_id=config.hf_repo,
        repo_type="dataset",
        revision=config.revision,
        local_dir=repo_dir,
        allow_patterns=filename_pattern,
    )

    shard_files = sorted(repo_dir.glob(filename_pattern))
    if not shard_files:
        message = (
            f"在 HuggingFace 仓库 {config.hf_repo} 中找不到 {filename_pattern} 文件，"
            "请确认已经按年份上传。"
        )
        logger.error(message)
        raise FileNotFoundError(message)

    years = [int(part) for path in shard_files if (part := path.stem.split("_")[-1]).isdigit()]
    if years:
        logger.info(
            "加载 {} 至 {} 的 {} 个年度分片",
            min(years),
            max(years),
            len(shard_files),
        )

    return (repo_dir / filename_pattern).as_posix()


def _load_sharded_dataset(
    config: HuggingFaceDatasetConfig,
    *,
    columns: Iterable[str] | None = None,
    lazy: bool = False,
) -> pl.DataFrame | pl.LazyFrame:
    glob_pattern = _snapshot_shards(config)
    scan = pl.scan_parquet(glob_pattern, low_memory=True)
    if columns:
        scan = scan.select(list(columns))
    if lazy:
        return scan
    return scan.collect(streaming=True)


def load_metadata(config: MetadataConfig, *, lazy: bool = False) -> pl.DataFrame | pl.LazyFrame:
    """Load sharded metadata from HuggingFace."""

    return _load_sharded_dataset(config, lazy=lazy)


def load_embeddings(
    config: EmbeddingConfig,
    *,
    lazy: bool = False,
    columns: Iterable[str] | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """Load sharded embeddings from HuggingFace."""

    return _load_sharded_dataset(config, columns=columns, lazy=lazy)


__all__ = ["load_metadata", "load_embeddings"]
