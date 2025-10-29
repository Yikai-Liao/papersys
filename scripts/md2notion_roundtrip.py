"""Upload Markdown fixtures to Notion and read them back for inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

import ultimate_notion as uno

from papersys.notion.md2notion import (
    MarkdownToNotionConverter,
    create_page_from_markdown,
)


def _iter_markdown_files(root: Path) -> Iterable[Path]:
    yield from sorted(path for path in root.glob("**/*.md") if path.is_file())


def upload_and_roundtrip(
    *,
    parent: str,
    markdown_files: Iterable[Path],
    session: uno.Session | None = None,
) -> list[dict[str, str]]:
    """Upload Markdown files to Notion and return roundtrip results."""

    session = session or uno.Session.get_or_create()
    converter = MarkdownToNotionConverter(session=session)

    results: list[dict[str, str]] = []

    for markdown_path in markdown_files:
        page = create_page_from_markdown(
            markdown_path,
            parent=parent,
            session=session,
            converter=converter,
        )
        roundtrip_md = page.to_markdown()
        results.append(
            {
                "source": str(markdown_path),
                "page_url": page.url,
                "title": page.title,
                "roundtrip_markdown": roundtrip_md,
            }
        )

    return results


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent",
        required=True,
        help="Parent Notion page ID or URL used when uploading markdown files.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=project_root / "example" / "ocr_responses",
        help="Directory containing Markdown files to upload (defaults to example fixtures).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of Markdown files to upload.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to dump roundtrip Markdown results as JSON for manual review.",
    )
    args = parser.parse_args()

    markdown_files = list(_iter_markdown_files(args.root))
    if args.limit is not None:
        markdown_files = markdown_files[: args.limit]

    if not markdown_files:
        raise SystemExit(f"No Markdown files found under {args.root}")

    results = upload_and_roundtrip(parent=args.parent, markdown_files=markdown_files)

    for result in results:
        print("-" * 80)
        print(f"源文件: {result['source']}")
        print(f"页面标题: {result['title']}")
        print(f"Notion 链接: {result['page_url']}")
        print("回转 Markdown:")
        print(result["roundtrip_markdown"])

    if args.output is not None:
        payload = json.dumps(results, ensure_ascii=False, indent=2)
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
