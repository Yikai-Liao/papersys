"""Utilities for converting Markdown documents into Ultimate Notion pages."""

from __future__ import annotations

import base64
import logging
import mimetypes
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence
from markdown_it import MarkdownIt
from markdown_it.token import Token
from mdit_py_plugins.footnote import footnote_plugin
from mdit_py_plugins.texmath import texmath_plugin

import ultimate_notion as uno
from ultimate_notion.rich_text import Text, math as make_math, text as make_text

logger = logging.getLogger(__name__)


def _attr_dict(token: Token) -> dict[str, str]:
    """Utility to convert a token's attribute list to a dictionary."""

    return dict(token.attrs or [])


def _combine_texts(left: Text | None, right: Text | None) -> Text | None:
    """Combine two Text objects while being robust to None."""

    if left is None:
        return right
    if right is None:
        return left
    return left + right


_HIGHLIGHT_PATTERN = re.compile(r"==(.+?)==", re.DOTALL)


def _split_highlight_segments(text: str) -> list[tuple[str, bool]]:
    """Split text into (segment, is_highlighted) pairs."""

    segments: list[tuple[str, bool]] = []
    last = 0
    for match in _HIGHLIGHT_PATTERN.finditer(text):
        start, end = match.span()
        if start > last:
            segments.append((text[last:start], False))
        segments.append((match.group(1), True))
        last = end
    if last < len(text):
        segments.append((text[last:], False))
    return segments


@dataclass(frozen=True)
class InlineStyle:
    """Track inline formatting state while walking markdown-it tokens."""

    bold: bool = False
    italic: bool = False
    underline: bool = False
    strikethrough: bool = False
    color: uno.Color | uno.BGColor | None = None
    href: str | None = None

    def evolve(self, **kwargs: object) -> "InlineStyle":
        """Return a copy with updated styling flags."""

        return replace(self, **kwargs)


@dataclass
class InlineRenderResult:
    """Container for inline conversion results."""

    text: Text | None = None
    attachments: list[uno.Block] = field(default_factory=list)

    def extend(self, other: "InlineRenderResult") -> None:
        """Merge another inline result into this instance."""

        self.text = _combine_texts(self.text, other.text)
        self.attachments.extend(other.attachments)


class MarkdownToNotionConverter:
    """Convert Markdown strings to Ultimate Notion block objects."""

    def __init__(
        self,
        *,
        session: uno.Session | None = None,
        highlight_color: uno.BGColor = uno.BGColor.YELLOW,
    ) -> None:
        self.session = session
        self.highlight_color = highlight_color
        self._md = self._build_markdown()
        self._asset_root = Path.cwd()

    @staticmethod
    def _build_markdown() -> MarkdownIt:
        md = MarkdownIt("commonmark", {"html": True, "linkify": True})
        md.use(texmath_plugin, delimiters="dollars")
        md.use(footnote_plugin)
        md.enable("strikethrough")
        md.enable("table")
        return md

    def convert(self, markdown: str, *, asset_root: Path | None = None) -> list[uno.Block]:
        """Convert Markdown content into Notion block instances."""

        if asset_root is not None:
            self._asset_root = asset_root
        tokens = self._md.parse(markdown)
        blocks, _ = self._consume_blocks(tokens, 0)
        return blocks

    def convert_file(self, path: Path) -> list[uno.Block]:
        """Load a Markdown file and convert it."""

        markdown = path.read_text(encoding="utf-8")
        return self.convert(markdown, asset_root=path.parent)

    # ------------------------------------------------------------------
    # Block level processing
    # ------------------------------------------------------------------
    def _consume_blocks(
        self,
        tokens: Sequence[Token],
        index: int,
        stop_types: set[str] | None = None,
    ) -> tuple[list[uno.Block], int]:
        blocks: list[uno.Block] = []
        stop_types = stop_types or set()
        length = len(tokens)

        while index < length:
            token = tokens[index]
            if token.type in stop_types:
                index += 1
                break

            if token.type == "heading_open":
                inline_token = tokens[index + 1]
                result = self._render_inline(inline_token.children or [])
                level = int(token.tag[1]) if token.tag and token.tag[1].isdigit() else 1
                block_cls = {1: uno.Heading1, 2: uno.Heading2}.get(level, uno.Heading3)
                text = result.text or make_text("")
                blocks.append(block_cls(text))
                blocks.extend(result.attachments)
                index += 3
                continue

            if token.type == "paragraph_open":
                inline_token = tokens[index + 1]
                result = self._render_inline(inline_token.children or [])
                index += 3
                if result.text:
                    blocks.append(uno.Paragraph(result.text))
                if result.attachments:
                    blocks.extend(result.attachments)
                continue

            if token.type == "fence":
                language = (token.info or "plain text").strip() or "plain text"
                blocks.append(uno.Code(token.content, language=language))
                index += 1
                continue

            if token.type == "code_block":
                blocks.append(uno.Code(token.content, language="plain text"))
                index += 1
                continue

            if token.type == "math_block":
                blocks.append(uno.Equation(token.content.strip()))
                index += 1
                continue

            if token.type == "bullet_list_open":
                items, index = self._consume_list(tokens, index, ordered=False)
                blocks.extend(items)
                continue

            if token.type == "ordered_list_open":
                items, index = self._consume_list(tokens, index, ordered=True)
                blocks.extend(items)
                continue

            if token.type == "blockquote_open":
                quote, index = self._consume_blockquote(tokens, index)
                if quote is not None:
                    blocks.append(quote)
                continue

            if token.type == "table_open":
                table, attachments, index = self._consume_table(tokens, index)
                if table is not None:
                    blocks.append(table)
                if attachments:
                    blocks.extend(attachments)
                continue

            # Skip any tokens we don't explicitly handle
            index += 1

        return blocks, index

    def _consume_list(
        self,
        tokens: Sequence[Token],
        index: int,
        *,
        ordered: bool,
    ) -> tuple[list[uno.Block], int]:
        items: list[uno.Block] = []
        index += 1  # skip list_open

        while index < len(tokens):
            token = tokens[index]
            if token.type == "list_item_open":
                item, index = self._consume_list_item(tokens, index, ordered=ordered)
                if item is not None:
                    items.append(item)
                continue
            if token.type in {"bullet_list_close", "ordered_list_close"}:
                index += 1
                break
            index += 1

        return items, index

    def _consume_list_item(
        self,
        tokens: Sequence[Token],
        index: int,
        *,
        ordered: bool,
    ) -> tuple[uno.Block | None, int]:
        index += 1  # skip list_item_open
        primary: InlineRenderResult | None = None
        children: list[uno.Block] = []

        while index < len(tokens):
            token = tokens[index]
            if token.type == "paragraph_open":
                inline_token = tokens[index + 1]
                result = self._render_inline(inline_token.children or [])
                if primary is None:
                    primary = result
                else:
                    if result.text:
                        children.append(uno.Paragraph(result.text))
                    if result.attachments:
                        children.extend(result.attachments)
                index += 3
                continue

            if token.type == "bullet_list_open":
                nested, index = self._consume_list(tokens, index, ordered=False)
                children.extend(nested)
                continue

            if token.type == "ordered_list_open":
                nested, index = self._consume_list(tokens, index, ordered=True)
                children.extend(nested)
                continue

            if token.type == "math_block":
                children.append(uno.Equation(token.content.strip()))
                index += 1
                continue

            if token.type == "list_item_close":
                index += 1
                break

            index += 1

        text = primary.text if primary and primary.text else make_text("")
        block_cls = uno.NumberedItem if ordered else uno.BulletedItem
        block = block_cls(text)

        if primary and primary.attachments:
            for attachment in primary.attachments:
                block.append(attachment)

        for child in children:
            block.append(child)

        return block, index

    def _consume_blockquote(
        self, tokens: Sequence[Token], index: int
    ) -> tuple[uno.Quote | None, int]:
        index += 1  # skip blockquote_open
        primary: InlineRenderResult | None = None
        children: list[uno.Block] = []

        while index < len(tokens):
            token = tokens[index]
            if token.type == "paragraph_open":
                inline_token = tokens[index + 1]
                result = self._render_inline(inline_token.children or [])
                if primary is None:
                    primary = result
                else:
                    if result.text:
                        children.append(uno.Paragraph(result.text))
                    if result.attachments:
                        children.extend(result.attachments)
                index += 3
                continue

            if token.type == "bullet_list_open":
                nested, index = self._consume_list(tokens, index, ordered=False)
                children.extend(nested)
                continue

            if token.type == "ordered_list_open":
                nested, index = self._consume_list(tokens, index, ordered=True)
                children.extend(nested)
                continue

            if token.type == "blockquote_close":
                index += 1
                break

            index += 1

        if primary is None:
            return None, index

        quote = uno.Quote(primary.text or make_text(""))
        for attachment in primary.attachments:
            quote.append(attachment)
        for child in children:
            quote.append(child)
        return quote, index

    def _consume_table(
        self, tokens: Sequence[Token], index: int
    ) -> tuple[uno.Table | None, list[uno.Block], int]:
        """Convert a Markdown table into a Notion table block."""

        index += 1  # skip table_open
        header_cells: list[Text] = []
        body_rows: list[list[Text]] = []
        attachments: list[uno.Block] = []

        while index < len(tokens):
            token = tokens[index]
            if token.type == "thead_open":
                header_cells, index, extra = self._consume_table_section(tokens, index, header=True)
                attachments.extend(extra)
                continue
            if token.type == "tbody_open":
                rows, index, extra = self._consume_table_section(tokens, index, header=False)
                body_rows.extend(rows)
                attachments.extend(extra)
                continue
            if token.type == "table_close":
                index += 1
                break
            index += 1

        if not header_cells and not body_rows:
            return None, attachments, index

        rows_to_write: list[list[Text]] = []
        if header_cells:
            rows_to_write.append(header_cells)
        rows_to_write.extend(body_rows)

        n_rows = len(rows_to_write)
        n_cols = max((len(row) for row in rows_to_write), default=0)
        if n_cols == 0:
            return None, attachments, index

        table = uno.Table(n_rows=max(1, n_rows), n_cols=n_cols, header_row=bool(header_cells))

        for idx, cells in enumerate(rows_to_write):
            normalized = self._normalize_table_cells(cells, expected_cols=n_cols)
            table[idx] = normalized

        return table, attachments, index

    def _consume_table_section(
        self,
        tokens: Sequence[Token],
        index: int,
        *,
        header: bool,
    ) -> tuple[list[Text] | list[list[Text]], int, list[uno.Block]]:
        rows: list[list[Text]] = []
        attachments: list[uno.Block] = []
        index += 1  # skip section_open

        while index < len(tokens):
            token = tokens[index]
            if token.type == "tr_open":
                cells, index, extra = self._consume_table_row(tokens, index)
                rows.append(cells)
                attachments.extend(extra)
                continue
            if token.type in {"thead_close", "tbody_close"}:
                index += 1
                break
            index += 1

        if header and rows:
            return rows[0], index, attachments
        return rows, index, attachments

    def _consume_table_row(
        self, tokens: Sequence[Token], index: int
    ) -> tuple[list[Text], int, list[uno.Block]]:
        cells: list[Text] = []
        attachments: list[uno.Block] = []
        index += 1  # skip tr_open

        while index < len(tokens):
            token = tokens[index]
            if token.type in {"th_open", "td_open"}:
                inline_token = tokens[index + 1]
                result = self._render_inline(inline_token.children or [])
                cells.append(result.text or make_text(""))
                attachments.extend(result.attachments)
                index += 3
                continue
            if token.type == "tr_close":
                index += 1
                break
            index += 1

        return cells, index, attachments

    @staticmethod
    def _normalize_table_cells(cells: Sequence[Text], *, expected_cols: int) -> list[Text]:
        padded = list(cells)
        if len(padded) < expected_cols:
            padded.extend(make_text("") for _ in range(expected_cols - len(padded)))
        return list(padded[:expected_cols])

    # ------------------------------------------------------------------
    # Inline level processing
    # ------------------------------------------------------------------
    def _render_inline(self, children: Sequence[Token]) -> InlineRenderResult:
        result = InlineRenderResult()
        style_stack: list[InlineStyle] = [InlineStyle()]

        for child in children:
            ctype = child.type

            if ctype == "text":
                if child.content:
                    for segment, highlighted in _split_highlight_segments(child.content):
                        if not segment:
                            continue
                        style = style_stack[-1]
                        if highlighted:
                            style = style.evolve(color=self.highlight_color)
                        result.text = _combine_texts(
                            result.text,
                            self._make_styled_text(segment, style),
                        )
                continue

            if ctype in {"softbreak", "hardbreak"}:
                result.text = _combine_texts(result.text, make_text("\n"))
                continue

            if ctype == "code_inline":
                result.text = _combine_texts(
                    result.text, make_text(child.content, code=True)
                )
                continue

            if ctype == "math_inline":
                result.text = _combine_texts(result.text, make_math(child.content))
                continue

            if ctype == "strong_open":
                style_stack.append(style_stack[-1].evolve(bold=True))
                continue

            if ctype == "strong_close":
                if len(style_stack) > 1:
                    style_stack.pop()
                continue

            if ctype == "em_open":
                style_stack.append(style_stack[-1].evolve(italic=True))
                continue

            if ctype == "em_close":
                if len(style_stack) > 1:
                    style_stack.pop()
                continue

            if ctype == "s_open":
                style_stack.append(style_stack[-1].evolve(strikethrough=True))
                continue

            if ctype == "s_close":
                if len(style_stack) > 1:
                    style_stack.pop()
                continue

            if ctype == "link_open":
                attrs = _attr_dict(child)
                style_stack.append(style_stack[-1].evolve(href=attrs.get("href")))
                continue

            if ctype == "link_close":
                if len(style_stack) > 1:
                    style_stack.pop()
                continue

            if ctype == "image":
                image_block = self._make_image_block(child)
                if image_block is not None:
                    result.attachments.append(image_block)
                continue

            if ctype == "footnote_ref":
                label = child.meta.get("label") if isinstance(child.meta, dict) else None
                display = f"[^{label}]" if label else "[†]"
                result.text = _combine_texts(
                    result.text, self._make_styled_text(display, style_stack[-1])
                )
                continue

            if child.content:
                result.text = _combine_texts(
                    result.text,
                    self._make_styled_text(child.content, style_stack[-1]),
                )

        return result

    def _make_styled_text(self, content: str, style: InlineStyle) -> Text:
        return make_text(
            content,
            bold=style.bold,
            italic=style.italic,
            underline=style.underline,
            strikethrough=style.strikethrough,
            color=style.color,
            href=style.href,
        )

    def _make_image_block(self, token: Token) -> uno.Image | None:
        attrs = _attr_dict(token)
        src = attrs.get("src") or ""
        alt = attrs.get("alt") or token.content or ""
        file_obj = self._resolve_file(src)
        if file_obj is None:
            logger.warning("Unable to resolve image '%s' relative to %s", src, self._asset_root)
            return None
        caption = alt.strip() or None
        return uno.Image(file_obj, caption=caption)

    def _resolve_file(self, source: str) -> uno.AnyFile | None:  # type: ignore[attr-defined]
        source = source.strip()
        if not source:
            return None
        if source.startswith("http://") or source.startswith("https://"):
            return uno.url(source)

        path = (self._asset_root / source).resolve()
        if not path.exists():
            logger.warning("Image path '%s' does not exist", path)
            return None

        if self.session is not None:
            with path.open("rb") as stream:
                uploaded = self.session.upload(stream, file_name=path.name)
            return uploaded

        data_uri = _file_to_data_uri(path)
        return uno.url(data_uri)


def _file_to_data_uri(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "application/octet-stream"
    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def create_page_from_markdown(
    markdown_path: Path,
    *,
    parent: str,
    title: str | None = None,
    session: uno.Session | None = None,
    converter: MarkdownToNotionConverter | None = None,
) -> uno.Page:
    """Upload a Markdown document to Notion as a brand-new page."""

    if session is None:
        session = uno.Session.get_or_create()

    parent_page = session.get_page(parent)

    if converter is None:
        converter = MarkdownToNotionConverter(session=session)

    blocks = converter.convert_file(markdown_path)
    page_title = title.strip() if title else markdown_path.stem

    page = session.create_page(parent=parent_page, title=page_title, blocks=blocks)
    return page

