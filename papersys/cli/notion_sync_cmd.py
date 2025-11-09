"""CLI command for syncing summary snapshots to Notion."""

from __future__ import annotations

from pathlib import Path

import typer
from loguru import logger

from ..config import AppConfig, load_config
from ..const import BASE_DIR
from ..notion.summary_sync import sync_snapshot_to_notion
from ..storage.git_store import GitStore


def notion_sync(
    config: Path = typer.Option(
        BASE_DIR / "config.toml",
        "--config",
        "-c",
        help="Path to the configuration TOML file.",
    ),
    database: str | None = typer.Option(
        None,
        "--database",
        "-d",
        help="Notion database URL or ID for summary ingestion (falls back to config).",
    ),
    snapshot: Path | None = typer.Option(
        None,
        "--snapshot",
        help="Optional override for the summary last.jsonl snapshot path.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Optionally cap how many records to upload in this run.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run/--no-dry-run",
        help="If enabled, log planned changes without touching Notion.",
    ),
) -> None:
    """Upload the latest summary batch into Notion."""

    logger.info("Loading config from {}", config)
    app_config = load_config(AppConfig, config)
    git_store = GitStore(app_config.git_store)
    database_ref = database or app_config.notion.database
    if not database_ref:
        raise typer.BadParameter("--database must be provided if config lacks a default.")

    if snapshot is None:
        git_store.ensure_local_copy()
        snapshot_path = git_store.summary_dir / "last.jsonl"
    else:
        snapshot_path = snapshot
        logger.info("Using explicit snapshot {}; skipping git fetch", snapshot_path)

    try:
        report = sync_snapshot_to_notion(
            snapshot_path=snapshot_path,
            database_ref=database_ref,
            limit=limit,
            dry_run=dry_run,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to sync summaries to Notion: {}", exc)
        raise typer.Exit(code=1) from exc

    logger.success(
        "Notion sync finished: total={}, created={}, updated={}",
        report.total,
        report.created,
        report.updated,
    )


__all__ = ["notion_sync"]
