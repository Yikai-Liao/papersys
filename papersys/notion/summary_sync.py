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
from ultimate_notion.props import Title, Text, Date, Number, MultiSelect, PropertyValue

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
            
            # Step 2: Prepare all properties
            properties = _prepare_properties(record, database)
            
            # Step 3: Create page with title, blocks, and properties in ONE API call
            page = retry_on_502(_create_page_with_properties, session, database, properties, blocks)
            
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


def _prepare_properties(record: dict[str, object], database: Database) -> dict[str, PropertyValue]:
    """Prepare all properties for a page in Notion API format."""
    record_id = record.get(ID)
    display_title = record.get(PAPER_TITLE_FIELD) or record_id or ""

    # Prepare all property values
    properties: dict[str, PropertyValue] = {}
    
    # Title property - use paper_title as the title field
    # In Notion database, there's always one "title" property field
    properties["title"] = Title(display_title)
    
    # Text fields
    text_fields = {
        ID: record_id,
        "authors": record.get(AUTHORS),
        SUMMARY_MODEL: record.get(SUMMARY_MODEL),
        SLUG: record.get(SLUG),
        "paper_url": _build_arxiv_url(record.get(ID)),
    }
    
    for prop_name, value in text_fields.items():
        formatted_value = _format_text_property(value)
        if formatted_value is not None:
            properties[prop_name] = Text(formatted_value)
    
    # MultiSelect fields (institution and keywords)
    multi_select_fields = {
        INSTITUTION: record.get(INSTITUTION),
        "keywords": record.get(KEYWORDS),
    }
    
    for prop_name, value in multi_select_fields.items():
        multi_select_value = _to_multi_select_values(value)
        if multi_select_value:
            properties[prop_name] = MultiSelect(multi_select_value)
    
    # Date fields
    date_fields = {
        PUBLISH_DATE: record.get(PUBLISH_DATE),
        UPDATE_DATE: record.get(UPDATE_DATE),
        SUMMARY_DATE: record.get(SUMMARY_DATE),
    }
    
    for prop_name, value in date_fields.items():
        if value:
            properties[prop_name] = Date(value)
    
    # Score (Number field)
    score = record.get(SCORE)
    if score is not None and not (isinstance(score, float) and math.isnan(score)):
        properties[SCORE] = Number(score)
    
    return properties


def _append_nested_children(
    session: uno.Session,
    parent_block_id: str,
    children: list,  # List of Block objects (from md2notion converter)
) -> None:
    """Recursively append children to a parent block.
    
    Args:
        session: The Notion session
        parent_block_id: The ID of the parent block
        children: List of child blocks to append (Block objects with obj_ref attribute)
    """
    # Convert children to obj_ref format for the API
    children_obj_refs = [child.obj_ref for child in children]
    
    # Append all children to the parent block
    appended_blocks, _ = session.api.blocks.children.append(parent_block_id, children_obj_refs)
    
    # For each appended block, if it has nested children, recursively append them
    for i, child in enumerate(children):
        if hasattr(child, '_children') and child._children:
            # Get the block_id of the appended child
            appended_block = appended_blocks[i]
            if hasattr(appended_block, 'id') and appended_block.id:
                _append_nested_children(session, str(appended_block.id), child._children)


def _create_page_with_properties(
    session: uno.Session,
    database: Database,
    properties: dict[str, PropertyValue],
    blocks: list[uno.Block] | None,
) -> Page:
    """Create a page with properties and blocks using the low-level API.
    
    This creates the page with properties and top-level blocks in a single API call.
    Nested children are appended separately due to Notion API limitations.
    """
    from ultimate_notion.obj_api.objects import DatabaseRef
    from ultimate_notion.obj_api.blocks import Database as ObjAPIDatabase
    
    # Build the request manually
    # Convert high-level Database to obj_api Database if needed
    if hasattr(database, 'obj_ref'):
        db_obj = database.obj_ref
    else:
        db_obj = database
    
    # Build DatabaseRef from the obj_api Database
    if isinstance(db_obj, ObjAPIDatabase):
        parent_ref = DatabaseRef.build(db_obj)
    else:
        msg = f'Unsupported database type: {type(database)}'
        raise TypeError(msg)
    
    request = {
        'parent': parent_ref.serialize_for_api(),
        'properties': {name: prop.obj_ref.serialize_for_api() for name, prop in properties.items()},
    }
    
    # Serialize blocks WITHOUT nested children (Notion API doesn't support nested children in create)
    blocks_with_children = []
    if blocks:
        request['children'] = []
        for block in blocks:
            serialized = block.obj_ref.serialize_for_api()
            request['children'].append(serialized)
            
            # Track blocks that have children for later processing
            if hasattr(block, '_children') and block._children:
                blocks_with_children.append(block)
    
    # Create the page with top-level blocks
    data = session.api.pages.raw_api.create(**request)
    
    # Wrap the response in a Page object
    # Note: The raw API returns a dict, we need to convert it to obj_api Page first
    from ultimate_notion.obj_api.blocks import Page as ObjAPIPage
    page_obj = ObjAPIPage.model_validate(data)
    page = Page.wrap_obj_ref(page_obj)
    session.cache[page.id] = page
    
    # Now append nested children to blocks that have them
    if blocks_with_children:
        # Retrieve the page's children to get their block IDs
        created_blocks = list(session.api.blocks.children.list(str(page.id)))
        
        # Match created blocks with our original blocks (in order)
        for i, block in enumerate(blocks):
            if hasattr(block, '_children') and block._children and i < len(created_blocks):
                _append_nested_children(session, str(created_blocks[i].id), block._children)
    
    return page


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
