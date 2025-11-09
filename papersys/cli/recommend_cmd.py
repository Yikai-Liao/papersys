"""Recommend papers command."""

from datetime import date
from pathlib import Path

import polars as pl
import typer
from loguru import logger

from ..config import AppConfig, load_config
from ..const import BASE_DIR
from ..data_sources import load_embeddings, load_metadata
from ..fields import EMBEDDING_VECTOR, ID, SCORE, TITLE
from ..recommend import Recommender
from ..storage.git_store import GitStore


def _parse_date(value: str | None, name: str) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter(f"{name} 日期格式错误，应为 YYYY-MM-DD。") from exc


def recommend(
    config: Path = typer.Option(
        BASE_DIR / "config.toml",
        "--config",
        "-c",
        help="配置文件路径",
    ),
    categories: list[str] | None = typer.Option(
        None,
        "--category",
        "-g",
        help="覆盖配置中的分类（可重复指定）。",
    ),
    last_n_days: int | None = typer.Option(
        None,
        "--last-n-days",
        "-n",
        help="筛选最近 N 天的论文（与 --start/--end 互斥）。",
    ),
    start: str | None = typer.Option(
        None,
        "--start",
        help="起始日期 (YYYY-MM-DD)。",
    ),
    end: str | None = typer.Option(
        None,
        "--end",
        help="结束日期 (YYYY-MM-DD)。",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        "-l",
        help="限制输出数量。",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="导出结果路径（支持 .csv / .parquet）。",
    ),
) -> None:
    if last_n_days is not None and (start or end):
        raise typer.BadParameter("--last-n-days 不可与 --start/--end 同时使用。", param_name="last_n_days")

    start_date = _parse_date(start, "start") if start else None
    end_date = _parse_date(end, "end") if end else None

    logger.info("加载配置：{}", config)
    app_config = load_config(AppConfig, config)

    selected_categories = categories or app_config.paper.categories
    if not selected_categories:
        raise typer.BadParameter("未提供分类，无法进行推荐。", param_name="category")

    git_store = GitStore(app_config.git_store)
    git_store.ensure_local_copy()

    metadata_df = load_metadata(app_config.metadata, lazy=True)
    embedding_df = load_embeddings(app_config.embedding, columns=[ID, EMBEDDING_VECTOR], lazy=True)
    preference_df = git_store.load_preferences()
    summary_ids = git_store.summary_store.existing_ids()

    def _frame_info(frame: pl.DataFrame | pl.LazyFrame) -> str:
        return str(frame.height) if isinstance(frame, pl.DataFrame) else "lazy"

    logger.info(
        "数据加载完成：metadata={} rows, embedding={} rows, preferences={} rows, existing summaries={}",
        _frame_info(metadata_df),
        _frame_info(embedding_df),
        preference_df.height,
        len(summary_ids),
    )

    recommender = Recommender(
        metadata=metadata_df,
        embeddings=embedding_df,
        preferences=preference_df,
        excluded_ids=summary_ids,
        config=app_config.recommend,
    )

    try:
        recommender.fit(selected_categories)
    except ValueError as exc:
        logger.error("训练失败：{}", exc)
        raise typer.Exit(code=1) from exc

    effective_last_n = last_n_days if last_n_days is not None else app_config.recommend.predict.last_n_days

    result = recommender.predict(
        selected_categories,
        last_n_days=effective_last_n,
        start_date=start_date,
        end_date=end_date,
    ).frame

    if result.is_empty():
        logger.warning("没有产生推荐结果。")
        return

    recommended = result.filter(pl.col("show") == 1)
    if recommended.is_empty():
        logger.warning("模型未选择任何推荐论文。")
        return

    drop_columns: list[str] = ["show"]
    if EMBEDDING_VECTOR in recommended.columns:
        drop_columns.append(EMBEDDING_VECTOR)

    recommended = recommended.drop(drop_columns).with_columns(pl.col(SCORE).round(6))

    if limit is not None:
        recommended = recommended.head(limit)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        suffix = output.suffix.lower()
        if suffix in {".jsonl", ".ndjson"}:
            recommended.write_ndjson(str(output))
        elif suffix == ".csv":
            recommended.write_csv(str(output))
        else:
            raise typer.BadParameter("仅支持输出为 .jsonl/.ndjson 或 .csv 文件。", param_name="output")
        logger.info("推荐结果已写入 {}", output)

    display_df = recommended.select([ID, TITLE, SCORE]).head(min(20, recommended.height))
    typer.echo(display_df)


__all__ = ["recommend"]
