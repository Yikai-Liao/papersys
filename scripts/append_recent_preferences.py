"""临时脚本：把 Notion 最近更新的偏好同步到本地 preference.csv。"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable

import polars as pl
import ultimate_notion as uno
from dotenv import load_dotenv
from ultimate_notion.option import Option
from ultimate_notion.page import Page as NotionPage
from ultimate_notion.query import prop

from papersys.config import AppConfig
from papersys.const import DEFAULT_CONFIG_PATH
from papersys.fields import ID, PREFERENCE, PREFERENCE_DATE
from papersys.storage.git_store import GitStore


PREFERENCE_SCHEMA = {
    ID: pl.String,
    PREFERENCE: pl.String,
    PREFERENCE_DATE: pl.String,
}


def _empty_preferences_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            col: pl.Series(name=col, values=[], dtype=dtype)
            for col, dtype in PREFERENCE_SCHEMA.items()
        }
    )


@dataclass(slots=True)
class Args:
    config: Path
    days: float
    limit: int | None
    dry_run: bool
    skip_git_sync: bool


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="抓取 Notion 中最近编辑的 preference, 并附加到本地 CSV"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="配置文件路径 (默认: config.toml)",
    )
    parser.add_argument(
        "--days",
        type=float,
        default=2.0,
        help="回溯的天数窗口，默认 2 天",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="最多处理多少条 Notion 记录，默认不限",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印结果，不写回 CSV",
    )
    parser.add_argument(
        "--skip-git-sync",
        action="store_true",
        help="跳过 GitStore.ensure_local_copy，避免误拉远端",
    )
    raw = parser.parse_args()
    return Args(
        config=raw.config,
        days=raw.days,
        limit=raw.limit,
        dry_run=raw.dry_run,
        skip_git_sync=raw.skip_git_sync,
    )


def main() -> None:
    args = parse_args()
    load_dotenv()
    config = AppConfig.from_toml(args.config)

    notion_rows = fetch_recent_preferences(
        database_ref=config.notion.database,
        lookback_days=args.days,
        limit=args.limit,
    )

    if not notion_rows:
        print("没捞到符合条件的记录，别问，Notion 就这表现。")
        return

    git_store = GitStore(config.git_store)
    if not args.skip_git_sync:
        git_store.ensure_local_copy()

    preference_path = git_store.preference_path
    updated_df, new_df = merge_and_update_csv(
        preference_path=preference_path,
        existing_df=load_existing_preferences(preference_path),
        new_rows=notion_rows,
        dry_run=args.dry_run,
    )

    show_summary(preference_path, updated_df, new_df, args.dry_run)


def fetch_recent_preferences(
    *, database_ref: str, lookback_days: float, limit: int | None
) -> list[dict[str, str]]:
    token = os.getenv("NOTION_TOKEN")
    if not token:
        raise RuntimeError("NOTION_TOKEN 没配好，老王没法连 Notion。")

    session = uno.Session.get_or_create()
    try:
        database = session.get_db(database_ref)
        cutoff = datetime.now(tz=UTC) - timedelta(days=lookback_days)

        condition = prop("preference").is_not_empty() & (prop("last_edit_time") >= cutoff)
        query = database.query.filter(condition).sort(prop("last_edit_time").desc())
        view = query.execute()

        pages = view.to_pages()
        if limit is not None:
            pages = pages[:limit]

        collected: list[dict[str, str]] = []
        for page in pages:
            preference = _extract_preference(page)
            if not preference:
                continue

            last_edit = page.last_edited_time
            if last_edit.tzinfo is None:
                last_edit = last_edit.replace(tzinfo=UTC)
            else:
                last_edit = last_edit.astimezone(UTC)

            record_id = _extract_record_id(page)
            collected.append(
                {
                    ID: record_id,
                    PREFERENCE: preference,
                    PREFERENCE_DATE: last_edit.date().isoformat(),
                }
            )

        return collected
    finally:
        session.close()


def _extract_preference(page: NotionPage) -> str | None:
    try:
        raw_value = page.props["preference"]
    except AttributeError:
        return None

    if isinstance(raw_value, Option):
        return raw_value.name
    if raw_value is None:
        return None

    text = str(raw_value).strip()
    if not text or text.lower() == "none":
        return None
    return text


def _extract_record_id(page: NotionPage) -> str:
    try:
        value = page.props[ID]
    except AttributeError as exc:
        msg = f"Notion 页面缺少 '{ID}' 属性: {page.id}"
        raise RuntimeError(msg) from exc

    if not value:
        msg = f"Notion 页面 '{page.id}' 的 '{ID}' 属性为空"
        raise RuntimeError(msg)

    return str(value).strip()


def load_existing_preferences(path: Path) -> pl.DataFrame:
    if not path.exists():
        return _empty_preferences_df()
    return pl.read_csv(path, schema_overrides=PREFERENCE_SCHEMA, infer_schema_length=0)


def merge_and_update_csv(
    *,
    preference_path: Path,
    existing_df: pl.DataFrame,
    new_rows: Iterable[dict[str, str]],
    dry_run: bool,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if not new_rows:
        return existing_df, _empty_preferences_df()

    new_df = pl.DataFrame(new_rows, schema=PREFERENCE_SCHEMA)
    merged = pl.concat([existing_df, new_df], how="vertical") if len(existing_df) else new_df
    merged = merged.with_row_index(name="_row_order")
    last_rows = merged.group_by(ID).agg(pl.col("_row_order").max().alias("_last_row"))
    updated = (
        merged.join(last_rows, on=ID, how="inner")
        .filter(pl.col("_row_order") == pl.col("_last_row"))
        .sort("_row_order")
        .drop(["_row_order", "_last_row"])
    )

    if not dry_run:
        preference_path.parent.mkdir(parents=True, exist_ok=True)
        updated.write_csv(preference_path)

    return updated, new_df


def show_summary(path: Path, updated: pl.DataFrame, appended: pl.DataFrame, dry_run: bool) -> None:
    unique_ids = set(updated[ID]) if ID in updated.columns else set()
    print("-------------")
    print(f"preference.csv: {path}")
    print(f"Notion 命中新记录: {len(appended)} 条")
    print(f"去重后总量: {len(updated)} 条 (覆盖 {len(unique_ids)} 个 id)")
    print("模式: DRY-RUN" if dry_run else "模式: 实写")
    print("-------------")


if __name__ == "__main__":
    main()
