"""Upload Markdown fixtures to Notion and read them back for inspection."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import httpx
from dotenv import load_dotenv
from notion_client import Client as NotionClient
from notion_client.errors import RequestTimeoutError

import ultimate_notion as uno
from ultimate_notion.config import get_or_create_cfg

from papersys.notion.md2notion import MarkdownToNotionConverter


def _iter_markdown_files(root: Path) -> Iterable[Path]:
    yield from sorted(path for path in root.glob("**/*.md") if path.is_file())


@contextmanager
def _deadline(seconds: float | None) -> Iterable[None]:
    """Context manager enforcing a soft timeout via SIGALRM."""

    if seconds is None or seconds <= 0:
        yield
        return

    if not hasattr(signal, "setitimer"):
        yield
        return

    def _handle_timeout(signum: int, frame: object) -> None:  # pragma: no cover - relies on signals
        raise TimeoutError(f"Operation exceeded {seconds} seconds")

    previous_handler = signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _ensure_session(timeout: float | None) -> uno.Session:
    """Create or retrieve a Notion session with explicit HTTP timeout."""

    if timeout is None:
        return uno.Session.get_or_create()

    token = os.environ.get("NOTION_TOKEN")
    if not token:
        cfg = get_or_create_cfg()
        token = cfg.ultimate_notion.token
        if not token:
            raise RuntimeError(
                "NOTION_TOKEN environment variable not set; unable to build Notion session with timeout.",
            )

    timeout_obj = httpx.Timeout(timeout, connect=timeout, read=timeout, write=timeout)
    http_client = httpx.Client(timeout=timeout_obj)
    notion = NotionClient(auth=token, client=http_client)

    cfg = get_or_create_cfg()
    if cfg.ultimate_notion.token is None:
        cfg.ultimate_notion.token = token

    return uno.Session(cfg=cfg, client=notion)


async def _upload_local_assets_async(
    session: uno.Session,
    paths: Iterable[Path],
    *,
    concurrency: int = 4,
) -> dict[Path, uno.AnyFile]:  # type: ignore[attr-defined]
    """Upload local assets concurrently using asyncio threads."""

    paths = [path for path in paths if path.exists()]
    if not paths:
        return {}

    semaphore = asyncio.Semaphore(max(1, concurrency))
    results: dict[Path, uno.AnyFile] = {}  # type: ignore[attr-defined]

    async def _worker(path: Path) -> None:
        async with semaphore:
            def _do_upload() -> uno.AnyFile:  # type: ignore[attr-defined]
                with path.open("rb") as stream:
                    return session.upload(stream, file_name=path.name)

            uploaded = await asyncio.to_thread(_do_upload)
            results[path] = uploaded

    await asyncio.gather(*(_worker(path) for path in paths))
    return results


def _upload_local_assets(
    session: uno.Session,
    paths: Iterable[Path],
    *,
    concurrency: int = 4,
) -> dict[Path, uno.AnyFile]:  # type: ignore[attr-defined]
    """Synchronous helper to upload assets via asyncio."""

    return asyncio.run(_upload_local_assets_async(session, paths, concurrency=concurrency))


def upload_and_roundtrip(
    *,
    parent: str,
    markdown_files: Iterable[Path],
    session: uno.Session | None = None,
    timeout: float | None = None,
    asset_concurrency: int = 4,
) -> list[dict[str, str]]:
    """Upload Markdown files to Notion and return roundtrip results."""

    session = session or _ensure_session(timeout)
    parent_page = session.get_page(parent)

    results: list[dict[str, str]] = []
    files = list(markdown_files)
    total = len(files)

    for index, markdown_path in enumerate(files, start=1):
        start_time = time.perf_counter()
        print(f"[{index}/{total}] Uploading {markdown_path} …", flush=True)
        markdown = markdown_path.read_text(encoding="utf-8")
        converter = MarkdownToNotionConverter(session=session)
        asset_paths = converter.collect_local_asset_paths(markdown, asset_root=markdown_path.parent)
        if asset_paths:
            uploaded_assets = _upload_local_assets(session, asset_paths, concurrency=asset_concurrency)
            converter.asset_cache.update(uploaded_assets)
        convert_start = time.perf_counter()
        blocks = converter.convert(markdown, asset_root=markdown_path.parent)
        convert_elapsed = time.perf_counter() - convert_start
        page_title = markdown_path.stem
        try:
            upload_start = time.perf_counter()
            with _deadline(timeout):
                page = session.create_page(
                    parent=parent_page,
                    title=page_title,
                    blocks=blocks,
                )
        except (TimeoutError, httpx.TimeoutException, RequestTimeoutError) as exc:
            elapsed = time.perf_counter() - start_time
            print(
                f"[{index}/{total}] FAILED after {elapsed:.1f}s → {exc}",
                flush=True,
            )
            raise
        upload_elapsed = time.perf_counter() - upload_start
        roundtrip_start = time.perf_counter()
        elapsed = time.perf_counter() - start_time
        print(f"[{index}/{total}] Uploaded → {page.url} ({elapsed:.1f}s)", flush=True)
        print(
            f"    convert: {convert_elapsed:.1f}s | upload: {upload_elapsed:.1f}s",
            flush=True,
        )
        roundtrip_md = page.to_markdown()
        roundtrip_elapsed = time.perf_counter() - roundtrip_start
        print(
            f"    to_markdown: {roundtrip_elapsed:.1f}s | total: {elapsed:.1f}s",
            flush=True,
        )
        results.append(
            {
                "source": str(markdown_path),
                "page_url": page.url,
                "title": page.title,
                "roundtrip_markdown": roundtrip_md,
                "timings": {
                    "convert_seconds": convert_elapsed,
                    "upload_seconds": upload_elapsed,
                    "roundtrip_seconds": roundtrip_elapsed,
                    "total_seconds": elapsed,
                },
                "block_count": len(blocks),
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
        "--timeout",
        type=float,
        default=120.0,
        help="Per-file timeout in seconds for Notion interactions (set <= 0 to disable).",
    )
    parser.add_argument(
        "--asset-workers",
        type=int,
        default=4,
        help="同时上传本地资源的协程数量 (默认 4)。",
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

    timeout = args.timeout if args.timeout and args.timeout > 0 else None
    asset_workers = args.asset_workers if args.asset_workers and args.asset_workers > 0 else 1
    session = _ensure_session(timeout)

    try:
        results = upload_and_roundtrip(
            parent=args.parent,
            markdown_files=markdown_files,
            session=session,
            timeout=timeout,
            asset_concurrency=asset_workers,
        )
    except (TimeoutError, httpx.TimeoutException, RequestTimeoutError) as exc:
        raise SystemExit(f"Timed out while syncing with Notion: {exc}") from exc
    finally:
        session.close()

    output_path = args.output
    if output_path is None:
        output_path = Path.cwd() / "md2notion_roundtrip.md"

    if output_path.suffix.lower() == ".json":
        payload = json.dumps(results, ensure_ascii=False, indent=2)
        output_path.write_text(payload, encoding="utf-8")
        print(f"Wrote JSON results to {output_path}")
    else:
        lines = ["# Notion Roundtrip Results", ""]
        lines.append(f"- Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- Parent page: {args.parent}")
        lines.append(f"- Timeout: {timeout if timeout is not None else 'disabled'} seconds")
        lines.append(f"- Asset workers: {asset_workers}")
        lines.append("")
        for result in results:
            lines.append(f"## {result['title']}")
            lines.append(f"- Source: `{result['source']}`")
            lines.append(f"- Notion: {result['page_url']}")
            timings = result["timings"]
            lines.append(
                "- Timings (s): "
                f"convert={timings['convert_seconds']:.1f}, "
                f"upload={timings['upload_seconds']:.1f}, "
                f"to_markdown={timings['roundtrip_seconds']:.1f}, "
                f"total={timings['total_seconds']:.1f}"
            )
            lines.append(f"- Block count: {result['block_count']}")
            lines.append("")
            lines.append("```markdown")
            lines.append(result["roundtrip_markdown"])
            lines.append("```")
            lines.append("")
        output_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote markdown report to {output_path}")


if __name__ == "__main__":
    main()
