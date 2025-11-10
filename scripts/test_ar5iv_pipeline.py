#!/usr/bin/env python3
"""Quick-and-dirty harness to verify ar5iv HTML → Markdown conversion."""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
import urllib.parse

import requests
import pypandoc


SCRIPT_DIR = pathlib.Path(__file__).parent
FILTER_PATH = SCRIPT_DIR / "strip-wrapper.lua"
DEFAULT_OUTPUT = SCRIPT_DIR / "ar5iv-sample.md"
USER_AGENT = "PaperSys-ar5iv-prober/0.1 (+paperops@example.com)"
MARKDOWN_FORMAT = "commonmark_x+tex_math_dollars"
REFERENCE_HEADING_RE = re.compile(
    r"(?im)^\s*#{1,6}\s+.*?(references?|bibliography)\b.*$"
)


def fetch_ar5iv_html(arxiv_id: str, timeout: int = 30) -> tuple[str, str] | tuple[None, None]:
    """Fetch rendered HTML from ar5iv if available."""
    url = f"https://ar5iv.org/html/{arxiv_id}"
    headers = {"User-Agent": USER_AGENT}

    try:
        resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    except requests.RequestException as exc:
        print(f"[warn] request error for {url}: {exc}", file=sys.stderr)
        return None, None

    final_host = urllib.parse.urlparse(resp.url).hostname or ""
    if resp.status_code == 200 and "ar5iv" in final_host:
        return resp.text, resp.url

    if "arxiv" in final_host:
        print(f"[info] {arxiv_id} missing on ar5iv (redirected to {resp.url})", file=sys.stderr)

    return None, None


def html_to_markdown(html: str) -> str:
    """Convert ar5iv HTML to GitHub-flavored Markdown via pandoc."""
    extra_args = [
        "--lua-filter",
        str(FILTER_PATH),
        "--wrap=none",
        "--markdown-headings=atx",
    ]
    return pypandoc.convert_text(
        html,
        to=MARKDOWN_FORMAT,
        format="html",
        extra_args=extra_args,
    )


def strip_reference_section(markdown: str) -> str:
    """Remove reference/bibliography section from markdown."""
    match = REFERENCE_HEADING_RE.search(markdown)
    if not match:
        return markdown
    cutoff = match.start()
    return markdown[:cutoff].rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arxiv_id", help="e.g. 2510.26641")
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    html, source_url = fetch_ar5iv_html(args.arxiv_id)
    if not html:
        print(f"[error] failed to fetch HTML for {args.arxiv_id}", file=sys.stderr)
        return 1

    print(f"[info] fetched ar5iv HTML from {source_url}")

    markdown = html_to_markdown(html)
    markdown = strip_reference_section(markdown)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(f"[ok] wrote markdown to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
