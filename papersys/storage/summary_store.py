from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Set

import polars as pl
from loguru import logger

from ..fields import (
    ID,
    PUBLISH_DATE,
    SUMMARY_DATE,
    UPDATE_DATE,
)
from .summary_schema import SUMMARY_RECORD_SCHEMA

SNAPSHOT_FILENAME = "last.jsonl"


@dataclass(slots=True)
class SummaryWriteReport:
    batch_size: int
    partition_paths: list[Path]
    partition_slugs: list[str]
    duplicate_ids: list[str]
    snapshot_path: Path


class SummaryStore:
    """Persist summaries to JSONL files partitioned by summary month."""

    def __init__(self, root: Path, *, partition_by_publish_year: bool = True) -> None:
        # partition_by_publish_year is kept only for backward compatibility
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def upsert_many(self, records: Iterable[Mapping[str, Any]]) -> SummaryWriteReport:
        """Append summary records into month-partitioned JSONL files."""

        prepared: list[dict[str, Any]] = []
        partitions: dict[str, list[dict[str, Any]]] = {}
        duplicate_ids: list[str] = []
        seen_ids: set[str] = set()

        for record in records:
            serialised = self._serialise_record(record)
            record_id = serialised.get(ID)
            if not record_id:
                logger.warning("Skip summary record without id: {}", record)
                continue

            if record_id in seen_ids:
                duplicate_ids.append(record_id)
            else:
                seen_ids.add(record_id)

            slug = self._resolve_month_slug(serialised)
            partitions.setdefault(slug, []).append(serialised)
            prepared.append(serialised)

        if not prepared:
            raise ValueError("No summary records to write")

        touched: list[tuple[str, Path]] = []
        for slug, batch in partitions.items():
            path = self._partition_path(slug)
            self._validate_existing_file(path)
            self._append_records(path, batch)
            touched.append((slug, path))
            logger.debug("Wrote {} summaries into {}", len(batch), path)

        snapshot_path = self.root / SNAPSHOT_FILENAME
        self._write_last_snapshot(snapshot_path, prepared)

        partition_slugs = [slug for slug, _ in touched]
        partition_paths = [path for _, path in touched]
        return SummaryWriteReport(
            batch_size=len(prepared),
            partition_paths=partition_paths,
            partition_slugs=partition_slugs,
            duplicate_ids=list(dict.fromkeys(duplicate_ids)),
            snapshot_path=snapshot_path,
        )

    # Internal helpers -----------------------------------------------------

    def _partition_path(self, slug: str) -> Path:
        return self.root / f"{slug}.jsonl"

    def _resolve_month_slug(self, record: Mapping[str, Any]) -> str:
        for field in (SUMMARY_DATE, PUBLISH_DATE, UPDATE_DATE):
            if field not in record or record[field] in (None, ""):
                continue
            slug = self._extract_month_slug(record[field])
            if slug:
                return slug
        return "unknown_month"

    def _extract_month_slug(self, value: Any) -> str | None:
        if isinstance(value, date):
            return value.strftime("%Y-%m")
        if isinstance(value, datetime):
            return value.strftime("%Y-%m")
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value)
            except ValueError:
                return None
            return parsed.strftime("%Y-%m")
        return None

    def _serialise_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        serialised: dict[str, Any] = {}
        for key, value in record.items():
            serialised[key] = self._serialise_value(value)
        return serialised

    def _serialise_value(self, value: Any) -> Any:
        if isinstance(value, date):
            return value.isoformat()
        if isinstance(value, datetime):
            return value.isoformat()
        return value

    def _validate_existing_file(self, path: Path) -> None:
        if not path.exists():
            return

        try:
            pl.scan_ndjson(str(path), schema=SUMMARY_RECORD_SCHEMA).select(pl.first()).collect(streaming=True)
        except Exception as exc:
            raise ValueError(f"Existing JSONL file {path} is corrupted; fix before writing") from exc

    def _append_records(self, path: Path, records: Iterable[Mapping[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as wf:
            for record in records:
                wf.write(json.dumps(record, ensure_ascii=False))
                wf.write("\n")

    def _write_last_snapshot(self, path: Path, records: Iterable[Mapping[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as wf:
            for record in records:
                wf.write(json.dumps(record, ensure_ascii=False))
                wf.write("\n")

    # Public helpers -------------------------------------------------------

    def iter_records(self) -> Iterator[dict[str, Any]]:
        """Iterate over all stored summary records."""
        if not self.root.exists():
            return iter(())

        def _iter() -> Iterator[dict[str, Any]]:
            for file_path in sorted(self.root.glob("*.jsonl")):
                if file_path.name == SNAPSHOT_FILENAME:
                    continue
                yield from self._iter_file(file_path)

        return _iter()

    def _iter_file(self, file_path: Path) -> Iterator[dict[str, Any]]:
        try:
            df = pl.scan_ndjson(str(file_path), schema=SUMMARY_RECORD_SCHEMA).collect(streaming=True)
        except Exception as exc:
            logger.warning("Failed to parse summary shard {}: {}", file_path, exc)
            return

        for row in df.iter_rows(named=True):
            yield dict(row)

    def existing_ids(self) -> Set[str]:
        """Return a set of all IDs already summarised."""
        ids: set[str] = set()
        for record in self.iter_records():
            record_id = record.get(ID)
            if record_id:
                ids.add(record_id)
        return ids
