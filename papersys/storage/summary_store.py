from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Set

from loguru import logger

from ..fields import (
    ID,
    PUBLISH_DATE,
    SUMMARY_DATE,
    UPDATE_DATE,
)


class SummaryStore:
    """Persist summaries to JSONL files partitioned by publish year."""

    def __init__(self, root: Path, *, partition_by_publish_year: bool = True) -> None:
        self.root = root
        self.partition_by_publish_year = partition_by_publish_year
        self.root.mkdir(parents=True, exist_ok=True)

    def upsert_many(self, records: Iterable[Mapping[str, Any]]) -> None:
        """Upsert multiple summary records into the JSONL store."""
        grouped: dict[Path, dict[str, dict[str, Any]]] = defaultdict(dict)

        for record in records:
            serialised = self._serialise_record(record)
            record_id = serialised.get(ID)
            if not record_id:
                logger.warning("跳过缺少 id 的摘要记录：{}", record)
                continue

            target_path = self._target_file(serialised)
            grouped[target_path][record_id] = serialised

        for path, updates in grouped.items():
            existing = self._load_existing(path)
            existing.update(updates)
            self._write_records(path, existing.values())
            logger.debug("写入 {} 条摘要到 {}", len(updates), path)

    # Internal helpers -----------------------------------------------------

    def _target_file(self, record: Mapping[str, Any]) -> Path:
        if not self.partition_by_publish_year:
            return self.root / "summaries.jsonl"

        for key in (PUBLISH_DATE, UPDATE_DATE, SUMMARY_DATE):
            if key not in record or record[key] is None:
                continue
            year = self._extract_year(record[key])
            if year is not None:
                return self.root / f"{year}.jsonl"

        return self.root / "unknown_year.jsonl"

    def _extract_year(self, value: Any) -> int | None:
        if isinstance(value, date):
            return value.year
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value).year
            except ValueError:
                return None
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

    def _load_existing(self, path: Path) -> dict[str, dict[str, Any]]:
        if not path.exists():
            return {}

        records: dict[str, dict[str, Any]] = {}
        try:
            with path.open("r", encoding="utf-8") as rf:
                for line in rf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("跳过无法解析的 JSON 行：{} -> {}", path, line[:80])
                        continue

                    record_id = payload.get(ID)
                    if record_id is None:
                        continue
                    records[record_id] = payload
        except FileNotFoundError:
            return {}

        return records

    def _write_records(self, path: Path, records: Iterable[Mapping[str, Any]]) -> None:
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
                with file_path.open("r", encoding="utf-8") as rf:
                    for line in rf:
                        if not line.strip():
                            continue
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            logger.warning("跳过损坏的摘要记录: {} -> {}", file_path, line[:80])
                            continue

        return _iter()

    def existing_ids(self) -> Set[str]:
        """Return a set of all IDs already summarised."""
        ids: set[str] = set()
        for record in self.iter_records():
            record_id = record.get(ID)
            if record_id:
                ids.add(record_id)
        return ids
