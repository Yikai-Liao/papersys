"""Upload summary parquet data into a Notion database.

This script bypasses the Ultimate Notion session layer so it can coexist with
newer Notion property types (e.g., Place) that the library does not yet model.
It supports dry runs, optional row limits, and basic Markdown-to-rich-text
conversion for rich_text properties.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import polars as pl
from dotenv import load_dotenv
from markdown_it import MarkdownIt
from notion_client import Client as NotionClient
import os

from papersys.notion.md2notion import MarkdownToNotionConverter

# Mapping from rich_text property names to parquet column names.
RICH_TEXT_FIELDS: dict[str, str] = {
    "slug": "slug",
    "paper_title": "title",
    "method": "method",
    "reasoning_step": "reasoning_step",
    "one_senetence_summary": "one_sentence_summary",
    "experiment": "experiment",
    "further_thoughts": "further_thoughts",
    "problem_background": "problem_background",
}

# Multi-select properties.
MULTI_SELECT_FIELDS: dict[str, str] = {
    "institution": "institution",
    "keywords": "keywords",
}

# Optional date mapping (keep empty unless requested).
DATE_FIELDS: dict[str, str] = {
    # "日期": "publish_date",
}


@dataclass(frozen=True)
class NotionConfig:
    db_id: str
    parquet_path: Path
    dry_run: bool
    limit: int | None
    project_root: Path


class RichTextRenderer:
    """Convert Markdown strings into Notion rich_text payloads."""

    def __init__(self) -> None:
        self._converter = MarkdownToNotionConverter()
        self._md: MarkdownIt = self._converter._build_markdown()

    def render(self, markdown: str | None) -> list[dict[str, Any]]:
        if not markdown:
            return []
        tokens = self._md.parse(markdown)
        segments: list[dict[str, Any]] = []
        self._consume_blocks(tokens, 0, segments)
        # Trim trailing newline segments
        while segments and self._is_newline(segments[-1]):
            segments.pop()
        return segments

    # ------------------------------------------------------------------ #
    # Token walkers
    # ------------------------------------------------------------------ #
    def _consume_blocks(self, tokens: list[Any], index: int, out: list[dict[str, Any]]) -> int:
        length = len(tokens)
        while index < length:
            token = tokens[index]
            t_type = token.type
            if t_type in {"paragraph_open", "heading_open"}:
                inline_token = tokens[index + 1]
                prefix = ""
                if t_type == "heading_open":
                    level = int(token.tag[1]) if token.tag and token.tag[1].isdigit() else 1
                    prefix = "#" * level + " "
                self._extend_inline(out, inline_token.children or [], prefix=prefix)
                out.append(self._newline_segment())
                index += 3
                continue

            if t_type == "bullet_list_open":
                index = self._consume_list(tokens, index + 1, out, ordered=False)
                continue

            if t_type == "ordered_list_open":
                start = token.attrGet("start")
                try:
                    number = int(start)
                except (TypeError, ValueError):
                    number = 1
                index = self._consume_list(tokens, index + 1, out, ordered=True, start=number)
                continue

            if t_type == "fence":
                content = token.content.rstrip("\n")
                out.extend(self._text_segments(content, code=True))
                out.append(self._newline_segment())
                index += 1
                continue

            if t_type == "code_block":
                content = token.content.rstrip("\n")
                out.extend(self._text_segments(content, code=True))
                out.append(self._newline_segment())
                index += 1
                continue

            if t_type == "math_block":
                expression = token.content.strip()
                if expression:
                    out.append(
                        {
                            "type": "equation",
                            "equation": {"expression": expression},
                            "annotations": self._default_annotations(),
                        },
                    )
                    out.append(self._newline_segment())
                index += 1
                continue

            if t_type in {"blockquote_open", "blockquote_close"}:
                # For now, flatten blockquotes into plain text.
                index += 1
                continue

            if t_type.endswith("_close"):
                index += 1
                continue

            # Fallback: skip unsupported token types.
            index += 1
        return index

    def _consume_list(
        self,
        tokens: list[Any],
        index: int,
        out: list[dict[str, Any]],
        *,
        ordered: bool,
        start: int = 1,
    ) -> int:
        number = start
        while index < len(tokens):
            token = tokens[index]
            if token.type == ("ordered_list_close" if ordered else "bullet_list_close"):
                return index + 1
            if token.type != "list_item_open":
                index += 1
                continue

            prefix = f"{number}. " if ordered else "• "
            number += 1
            index += 1
            while index < len(tokens) and tokens[index].type != "list_item_close":
                inner = tokens[index]
                if inner.type in {"paragraph_open", "heading_open"}:
                    inline_token = tokens[index + 1]
                    self._extend_inline(out, inline_token.children or [], prefix=prefix)
                    out.append(self._newline_segment())
                    prefix = "  " if ordered else "  "
                    index += 3
                    continue
                if inner.type in {"bullet_list_open", "ordered_list_open"}:
                    index = self._consume_list(
                        tokens,
                        index + 1,
                        out,
                        ordered=inner.type == "ordered_list_open",
                        start=int(inner.attrGet("start") or "1"),
                    )
                    continue
                index += 1  # Skip unknown tokens within the list item
            # Skip the list_item_close token
            index += 1
        return index

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _extend_inline(self, out: list[dict[str, Any]], tokens: Iterable[Any], *, prefix: str = "") -> None:
        if prefix:
            out.extend(self._text_segments(prefix))
        result = self._converter._render_inline(list(tokens))
        if result.text:
            out.extend(self._rich_texts_to_api(result.text.rich_texts))

    def _text_segments(self, content: str, *, code: bool = False) -> list[dict[str, Any]]:
        if not content:
            return []
        annotations = self._default_annotations(code=code)
        return [
            {
                "type": "text",
                "text": {"content": content, "link": None},
                "annotations": annotations,
            },
        ]

    def _rich_texts_to_api(self, items: Iterable[Any]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for item in items:
            ref = item.obj_ref
            payload = ref.model_dump(mode="json")
            annotations = payload.get("annotations")
            if annotations:
                color = annotations.get("color")
                if isinstance(color, dict):
                    annotations["color"] = "default"
                elif hasattr(color, "value"):
                    annotations["color"] = getattr(color, "value")
            else:
                annotations = self._default_annotations()
                payload["annotations"] = annotations

            if payload["type"] == "text":
                text_content = payload["text"]["content"]
                if payload["annotations"].get("code") and self._looks_like_equation(text_content):
                    converted.append(
                        {
                            "type": "equation",
                            "equation": {"expression": text_content},
                            "annotations": self._default_annotations(),
                        },
                    )
                    continue
                converted.append(
                    {
                        "type": "text",
                        "text": {
                            "content": text_content,
                            "link": payload["text"].get("link"),
                        },
                        "annotations": annotations,
                    },
                )
            elif payload["type"] == "equation":
                converted.append(
                    {
                        "type": "equation",
                        "equation": {"expression": payload["equation"]["expression"]},
                        "annotations": annotations,
                    },
                )
            elif payload["type"] == "mention":
                converted.append(
                    {
                        "type": "mention",
                        "mention": payload["mention"],
                        "annotations": annotations,
                    },
                )
            else:
                # As a fallback, coerce to plain text.
                converted.append(
                    {
                        "type": "text",
                        "text": {"content": payload.get("plain_text", ""), "link": None},
                        "annotations": annotations,
                    },
                )

        return converted

    @staticmethod
    def _looks_like_equation(text: str) -> bool:
        equation_markers = set("=+−*/^_<>")
        greek_chars = set("αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ")
        if any(char in equation_markers for char in text):
            return True
        if any(char in greek_chars for char in text):
            return True
        if any(char.isdigit() for char in text) and any(char.isalpha() for char in text):
            return True
        if " " in text and any(token.isdigit() for token in text.split()):
            return True
        return False

    @staticmethod
    def _default_annotations(*, code: bool = False) -> dict[str, Any]:
        return {
            "bold": False,
            "italic": False,
            "strikethrough": False,
            "underline": False,
            "code": code,
            "color": "default",
        }

    @staticmethod
    def _newline_segment() -> dict[str, Any]:
        return {
            "type": "text",
            "text": {"content": "\n", "link": None},
            "annotations": RichTextRenderer._default_annotations(),
        }

    @staticmethod
    def _is_newline(segment: dict[str, Any]) -> bool:
        return segment.get("type") == "text" and segment.get("text", {}).get("content") == "\n"


def _parse_args(argv: list[str]) -> NotionConfig:
    parser = argparse.ArgumentParser(
        description="Upload summaries from a parquet file into a Notion database.",
    )
    parser.add_argument(
        "--db-id",
        required=True,
        help="Target Notion database ID (UUID with or without dashes).",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path("example/summaries.parquet"),
        help="Path to the parquet file containing summary data.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of rows to upload.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not perform writes; just print the actions that would be taken.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root used to locate the .env file (defaults to repo root).",
    )

    args = parser.parse_args(argv)
    return NotionConfig(
        db_id=_normalize_db_id(args.db_id),
        parquet_path=args.parquet,
        dry_run=bool(args.dry_run),
        limit=args.limit,
        project_root=args.project_root,
    )


def _normalize_db_id(value: str) -> str:
    stripped = value.replace("-", "")
    if len(stripped) != 32:
        raise ValueError(f"Unexpected database ID format: {value}")
    return f"{stripped[0:8]}-{stripped[8:12]}-{stripped[12:16]}-{stripped[16:20]}-{stripped[20:32]}"


def _load_records(parquet_path: Path, limit: int | None) -> list[dict[str, Any]]:
    df = pl.read_parquet(parquet_path)
    if limit is not None:
        df = df.head(limit)
    return df.to_dicts()


def _load_existing_pages(client: NotionClient, db_id: str) -> dict[str, str]:
    existing: dict[str, str] = {}
    cursor: str | None = None
    while True:
        response = client.databases.query(
            **{"database_id": db_id, "page_size": 100, **({"start_cursor": cursor} if cursor else {})}
        )
        for page in response.get("results", []):
            props = page.get("properties", {})
            id_prop = props.get("id", {})
            title_items = id_prop.get("title", [])
            if title_items:
                plain_id = title_items[0].get("plain_text", "").strip()
                if plain_id:
                    existing[plain_id] = page["id"]
        cursor = response.get("next_cursor")
        if not response.get("has_more"):
            break
    return existing


def _build_rich_text_props(renderer: RichTextRenderer, record: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    props: dict[str, list[dict[str, Any]]] = {}
    for prop_name, column in RICH_TEXT_FIELDS.items():
        value = record.get(column)
        if isinstance(value, str):
            source = value
        elif value is None:
            source = ""
        else:
            source = str(value)
        segments = renderer.render(source)
        props[prop_name] = segments
    return props


def _build_multi_select_props(record: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    props: dict[str, list[dict[str, str]]] = {}

    def _normalize_option(option: str) -> str:
        cleaned = option.strip()
        for sep in (",", "，", ";", "；"):
            cleaned = cleaned.replace(sep, " / ")
        return " ".join(cleaned.split())

    for prop_name, column in MULTI_SELECT_FIELDS.items():
        value = record.get(column)
        if value is None:
            props[prop_name] = []
            continue
        if isinstance(value, str):
            separators = [",", "，", ";", "；"]
            if any(sep in value for sep in separators):
                candidates = []
                current = value
                for sep in separators:
                    current = current.replace(sep, ",")
                candidates = [part.strip() for part in current.split(",") if part.strip()]
            else:
                candidates = [value.strip()] if value.strip() else []
        else:
            candidates = [item for item in value if isinstance(item, str) and item.strip()]

        normalized: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            cleaned = _normalize_option(candidate)
            if cleaned and cleaned not in seen:
                normalized.append(cleaned)
                seen.add(cleaned)

        props[prop_name] = [{"name": option} for option in normalized]
    return props


def _build_date_props(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    props: dict[str, dict[str, Any]] = {}
    for prop_name, column in DATE_FIELDS.items():
        value = record.get(column)
        if value:
            props[prop_name] = {"start": str(value)}
        else:
            props[prop_name] = None  # Clear the date if value missing
    return props


def _build_properties(
    renderer: RichTextRenderer,
    record: dict[str, Any],
    schema: dict[str, dict[str, Any]],
    title_prop: str,
) -> dict[str, Any]:
    record_id = str(record.get("id", "")).strip()
    if not record_id:
        raise ValueError("Each record must include an 'id' column with a non-empty value.")

    rich_text_props = _build_rich_text_props(renderer, record)
    multi_select_props = _build_multi_select_props(record)
    date_props = _build_date_props(record)

    properties: dict[str, Any] = {
        title_prop: {"title": [{"type": "text", "text": {"content": record_id}}]},
    }

    for prop_name, segments in rich_text_props.items():
        meta = schema.get(prop_name)
        if meta and meta.get("type") == "rich_text":
            properties[prop_name] = {"rich_text": segments}

    for prop_name, options in multi_select_props.items():
        meta = schema.get(prop_name)
        if meta and meta.get("type") == "multi_select":
            properties[prop_name] = {"multi_select": options}

    for prop_name, date_payload in date_props.items():
        meta = schema.get(prop_name)
        if meta and meta.get("type") == "date":
            properties[prop_name] = {"date": date_payload}

    return properties


def _create_page(
    client: NotionClient,
    db_id: str,
    properties: dict[str, Any],
    record_id: str,
    *,
    property_labels: dict[str, str],
    dry_run: bool,
) -> None:
    deferred_title = properties.get("title")
    create_payload = {name: payload for name, payload in properties.items() if name != "title"}

    if dry_run:
        prop_names = ", ".join(sorted(property_labels.get(name, name) for name in properties))
        print(f"[DRY-RUN] Would create page '{record_id}' with: {prop_names}")
        return

    # Debugging helper: print create payload keys when failure occurs.
    # Not printed normally to avoid noisy logs.
    try:
        page = client.pages.create(parent={"database_id": db_id}, properties=create_payload)
    except Exception:
        debug_keys = ", ".join(sorted(create_payload.keys()))
        print(f"Create payload keys: {debug_keys}", file=sys.stderr)
        raise
    if deferred_title:
        client.pages.update(page_id=page["id"], properties={"title": deferred_title})


def _update_page(
    client: NotionClient,
    page_id: str,
    properties: dict[str, Any],
    record_id: str,
    *,
    property_labels: dict[str, str],
    dry_run: bool,
) -> None:
    if dry_run:
        prop_names = ", ".join(sorted(property_labels.get(name, name) for name in properties))
        print(f"[DRY-RUN] Would update page {record_id} ({page_id}) with: {prop_names}")
        return

    client.pages.update(page_id=page_id, properties=properties)


def main(argv: list[str] | None = None) -> int:
    cfg = _parse_args(argv or sys.argv[1:])
    load_dotenv(cfg.project_root / ".env")

    if not cfg.parquet_path.exists():
        print(f"Parquet file not found: {cfg.parquet_path}", file=sys.stderr)
        return 1

    records = _load_records(cfg.parquet_path, cfg.limit)
    if not records:
        print("No records found in parquet file.", file=sys.stderr)
        return 1

    renderer = RichTextRenderer()
    token = os.environ.get("NOTION_TOKEN")
    if not token:
        print("NOTION_TOKEN is not set; aborting.", file=sys.stderr)
        return 1

    client = NotionClient(auth=token)
    db_meta = client.databases.retrieve(cfg.db_id)
    schema = dict(db_meta.get("properties", {}))
    try:
        title_prop = next(name for name, meta in schema.items() if meta.get("type") == "title")
    except StopIteration:
        print("Database schema does not define a title property.", file=sys.stderr)
        return 1
    title_conflict = "title" in schema and schema["title"].get("type") == "rich_text" and title_prop != "title"
    if title_conflict:
        print(
            "WARNING: Database contains a rich_text property named 'title'.\n"
            "         Notion's API refuses to set the database title column when that name is taken.\n"
            "         Existing rows will still be updated (title column will be left untouched),\n"
            "         but new rows cannot be created until the column is renamed.",
            file=sys.stderr,
        )
    property_labels = {name: name for name in schema}
    existing_pages = _load_existing_pages(client, cfg.db_id)

    created = 0
    updated = 0

    for record in records:
        properties = _build_properties(renderer, record, schema, title_prop)
        record_id = str(record.get("id", "")).strip()
        page_id = existing_pages.get(record_id)
        if page_id:
            update_props = {name: payload for name, payload in properties.items() if name != title_prop}
            try:
                _update_page(
                    client,
                    page_id,
                    update_props,
                    record_id,
                    property_labels=property_labels,
                    dry_run=cfg.dry_run,
                )
            except Exception as exc:  # pragma: no cover - diagnostic aid
                print(f"Failed to update record {record_id}: {exc}", file=sys.stderr)
                raise
            updated += 1
        else:
            if title_conflict:
                print(
                    f"Skipping creation of '{record_id}' because the database has a conflicting 'title' rich_text column.",
                    file=sys.stderr,
                )
                continue
            try:
                _create_page(
                    client,
                    cfg.db_id,
                    properties,
                    record_id,
                    property_labels=property_labels,
                    dry_run=cfg.dry_run,
                )
            except Exception as exc:  # pragma: no cover - diagnostic aid
                prop_names = ", ".join(sorted(property_labels.get(pid, pid) for pid in properties))
                print(
                    f"Failed to create record {record_id} with properties: {prop_names}\nError: {exc}",
                    file=sys.stderr,
                )
                raise
            created += 1

    suffix = " (dry-run)" if cfg.dry_run else ""
    print(f"Created {created} pages, updated {updated} pages{suffix}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
