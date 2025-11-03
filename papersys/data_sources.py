from __future__ import annotations

from pathlib import Path
from typing import Iterable

import polars as pl
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from loguru import logger

from .config import EmbeddingConfig, MetadataConfig
from .fields import ID

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "hf_cache"


def _ensure_cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def _download_parquet(config_repo: str, filename: str, revision: str | None) -> Path | None:
    try:
        path = hf_hub_download(
            repo_id=config_repo,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            cache_dir=_ensure_cache_dir(),
        )
    except HfHubHTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            logger.warning("文件 {} 在仓库 {} 未找到 (revision={}), 返回空数据。", filename, config_repo, revision)
            return None
        raise
    return Path(path)


def load_metadata(config: MetadataConfig, *, lazy: bool = False) -> pl.DataFrame | pl.LazyFrame:
    """Load metadata parquet from HuggingFace."""
    parquet_path = _download_parquet(config.hf_repo, config.parquet, config.revision)
    if parquet_path is None:
        schema = {ID: pl.String}
        df = pl.DataFrame(schema=schema)
        return df.lazy() if lazy else df
    scan = pl.scan_parquet(parquet_path, low_memory=True)
    if lazy:
        return scan
    return scan.collect(streaming=True)


def load_embeddings(
    config: EmbeddingConfig,
    *,
    lazy: bool = False,
    columns: Iterable[str] | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """Load embedding parquet from HuggingFace."""
    parquet_path = _download_parquet(config.hf_repo, config.parquet, config.revision)
    if parquet_path is None:
        schema = {ID: pl.String, "embedding": pl.List(pl.Float32)}
        df = pl.DataFrame(schema=schema)
        return df.lazy() if lazy else df
    scan = pl.scan_parquet(parquet_path, low_memory=True)
    if columns:
        scan = scan.select(list(columns))
    if lazy:
        return scan
    return scan.collect(streaming=True)


__all__ = ["load_metadata", "load_embeddings"]
