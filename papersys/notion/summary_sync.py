"""Sync summary snapshots into a Notion database via ultimate-notion."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TypeVar, Callable, Any
from collections.abc import Sequence

import polars as pl
from loguru import logger
from tqdm import tqdm
import ultimate_notion as uno
from ultimate_notion import schema as notion_schema
from ultimate_notion.database import Database
from ultimate_notion.option import Option
from ultimate_notion.page import Page

from ..fields import (
    AUTHORS,
    EXPERIMENT,
    FURTHER_THOUGHTS,
    ID,
    INSTITUTION,
    KEYWORDS,
    METHOD,
    ONE_SENTENCE_SUMMARY,
    PROBLEM_BACKGROUND,
    PUBLISH_DATE,
    REASONING_STEP,
    SCORE,
    SLUG,
    SUMMARY_DATE,
    SUMMARY_MODEL,
    TITLE,
    UPDATE_DATE,
)
from ..storage.summary_schema import SUMMARY_RECORD_SCHEMA
from .md2notion import MarkdownToNotionConverter


PREFERENCE_OPTIONS = ("like", "dislike", "neutral")
PAPER_TITLE_FIELD = "paper_title"

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2.0  # seconds
RETRY_BACKOFF = 2.0  # exponential backoff multiplier

T = TypeVar('T')


def retry_on_502(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Retry a function call if it fails with a 502 error."""
    last_exception = None
    
    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e).lower()
            # Check for 502 errors or other transient Notion API errors
            is_retryable = any(
                indicator in error_msg 
                for indicator in ["502", "bad gateway", "503", "service unavailable", "timeout"]
            )
            
            if is_retryable:
                last_exception = e
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY * (RETRY_BACKOFF ** attempt)
                    logger.warning(
                        "Notion API error (attempt {}/{}): {} - Retrying in {:.1f}s...",
                        attempt + 1,
                        MAX_RETRIES,
                        str(e),
                        delay
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.error(
                        "Notion API error persists after {} attempts: {}",
                        MAX_RETRIES,
                        str(e)
                    )
            # If it's not a retryable error, or we've exhausted retries, re-raise
            raise
    
    # If we get here, all retries failed
    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected retry loop exit")


@dataclass(slots=True)
class NotionSyncReport:
    """Stats about the sync run."""

    created: int = 0
    updated: int = 0
    total: int = 0


def sync_snapshot_to_notion(
    snapshot_path: Path,
    database_ref: str,
    *,
    limit: int | None = None,
    dry_run: bool = False,
) -> NotionSyncReport:
    """Upload summary snapshot records to a Notion database."""

    if not snapshot_path.exists():
        msg = f"Snapshot file not found: {snapshot_path}"
        raise FileNotFoundError(msg)

    token = os.getenv("NOTION_TOKEN")
    if not token:
        msg = "NOTION_TOKEN environment variable is missing"
        raise RuntimeError(msg)

    records = _load_snapshot(snapshot_path)
    if limit is not None:
        records = records[:limit]

    if not records:
        raise ValueError("Snapshot does not contain any summary records")

    logger.info("Connecting to Notion database {}", database_ref)
    session = uno.Session.get_or_create()
    report = NotionSyncReport(total=len(records))

    try:
        database = retry_on_502(session.get_db, database_ref)
        retry_on_502(_ensure_schema, database)
        title_attr_name = PAPER_TITLE_FIELD
        converter = MarkdownToNotionConverter(session=session)

        for record in tqdm(records, desc="Creating pages in Notion", unit="paper"):
            record_id = record.get(ID)
            if not record_id:
                logger.warning("Skipping record without id: {}", record)
                continue

            if dry_run:
                logger.info("[dry-run] Would create Notion page for {}", record_id)
                continue

            # Step 1: Prepare blocks offline (convert markdown)
            blocks = _prepare_page_blocks(record, converter)
            
            # Step 2: Create page with all properties and content in one go
            page_kwargs = {title_attr_name: record.get(PAPER_TITLE_FIELD) or record_id}
            page = retry_on_502(database.create_page, **page_kwargs)
            
            # Step 3: Set properties
            _apply_properties(page, record)
            
            # Step 4: Append all blocks in batch (already in Notion, so will upload immediately)
            if blocks:
                retry_on_502(page.append, blocks)
            
            report.created += 1

    finally:
        session.close()

    logger.info(
        "Sync complete: total={}, created={}, updated={}",
        report.total,
        report.created,
        report.updated,
    )
    return report


def _load_snapshot(path: Path) -> list[dict[str, object]]:
    df = pl.read_ndjson(path, schema=SUMMARY_RECORD_SCHEMA)
    if TITLE in df.columns:
        df = df.rename({TITLE: PAPER_TITLE_FIELD})
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias(PAPER_TITLE_FIELD))
    return df.to_dicts()


def _ensure_schema(database: Database) -> None:
    schema = database.schema
    needed_props: dict[str, notion_schema.Property] = {
        ID: notion_schema.Text(),
        "authors": notion_schema.Text(),
        PUBLISH_DATE: notion_schema.Date(),
        UPDATE_DATE: notion_schema.Date(),
        SUMMARY_DATE: notion_schema.Date(),
        SUMMARY_MODEL: notion_schema.Text(),
        SCORE: notion_schema.Number(),
        SLUG: notion_schema.Text(),
        "paper_url": notion_schema.Text(),
        "preference": notion_schema.Select(options=[Option(opt) for opt in PREFERENCE_OPTIONS]),
    }

    for name, prop in needed_props.items():
        if not schema.has_prop(name):
            schema[name] = prop

    _ensure_select_options(schema.get_prop("preference"), PREFERENCE_OPTIONS)


def _ensure_select_options(prop: notion_schema.Select, options: Iterable[str]) -> None:
    existing = {opt.name for opt in prop.options}
    missing = [Option(opt) for opt in options if opt not in existing]
    if not missing:
        return
    prop.options = prop.options + missing  # type: ignore[operator]


def _apply_properties(page: Page, record: dict[str, object]) -> None:
    record_id = record.get(ID)
    page.title = str(record_id or "")

    display_title = record.get(PAPER_TITLE_FIELD) or ""

    text_fields = {
        ID: record_id,
        PAPER_TITLE_FIELD: display_title,
        "authors": record.get(AUTHORS),
        INSTITUTION: record.get(INSTITUTION),
        "keywords": record.get(KEYWORDS),
        SUMMARY_MODEL: record.get(SUMMARY_MODEL),
        SLUG: record.get(SLUG),
        "paper_url": _build_arxiv_url(record.get(ID)),
    }

    for prop, value in text_fields.items():
        _set_property_value(page, prop, value)

    date_fields = {
        PUBLISH_DATE: record.get(PUBLISH_DATE),
        UPDATE_DATE: record.get(UPDATE_DATE),
        SUMMARY_DATE: record.get(SUMMARY_DATE),
    }

    for prop, value in date_fields.items():
        page.props[prop] = value if value else None

    score = record.get(SCORE)
    if isinstance(score, float) and math.isnan(score):
        score_value = None
    else:
        score_value = score
    page.props[SCORE] = score_value


def _build_arxiv_url(value: object) -> str | None:
    if not value:
        return None
    return f"https://arxiv.org/abs/{value}"


def _prepare_page_blocks(
    record: dict[str, object],
    converter: MarkdownToNotionConverter,
) -> list[uno.Block]:
    """Prepare all blocks for a page from markdown content."""
    sections = [
        ("One-sentence Summary", record.get(ONE_SENTENCE_SUMMARY)),
        ("Problem Background", record.get(PROBLEM_BACKGROUND)),
        ("Method", record.get(METHOD)),
        ("Experiment", record.get(EXPERIMENT)),
        ("Further Thoughts", record.get(FURTHER_THOUGHTS)),
        ("Reasoning Steps", record.get(REASONING_STEP)),
    ]

    blocks: list[uno.Block] = []

    for heading, content in sections:
        if not content:
            continue
        blocks.append(uno.Heading2(heading))
        section_blocks = converter.convert(str(content))
        blocks.extend(section_blocks)
        blocks.append(uno.Paragraph(""))

    # Remove trailing spacer if present to avoid empty block at EOF
    if blocks and isinstance(blocks[-1], uno.Paragraph) and not str(blocks[-1]).strip():
        blocks.pop()
    
    return blocks



def _format_text_property(value: object) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, Sequence) and not isinstance(value, str):
        filtered = [str(item) for item in value if item]
        return ", ".join(filtered) if filtered else None
    return str(value)


def _set_property_value(page: Page, prop_name: str, value: object) -> None:
    db = page.parent_db
    prop_schema = None
    if db is not None:
        try:
            prop_schema = db.schema.get_prop(prop_name, default=None)
        except Exception:
            prop_schema = None

    if isinstance(prop_schema, notion_schema.MultiSelect):
        page.props[prop_name] = _to_multi_select_values(value)
        return

    page.props[prop_name] = _format_text_property(value)


def _to_multi_select_values(value: object) -> list[str] | None:
    if not value:
        return None
    if isinstance(value, str):
        candidates = [item.strip() for item in value.split(",")]
    elif isinstance(value, Sequence):
        candidates = [str(item).strip() for item in value if item]
    else:
        candidates = [str(value).strip()]

    cleaned = [item for item in candidates if item]
    return cleaned or None


__all__ = ["sync_snapshot_to_notion", "NotionSyncReport"]
