"""Summarize papers command."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping

import polars as pl
import typer
from loguru import logger

from ..config import AppConfig, load_config
from ..const import BASE_DIR
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
from ..storage.git_store import GitStore
from ..storage.summary_schema import (
    RECOMMEND_INPUT_SCHEMA,
    SUMMARY_RECORD_SCHEMA,
    align_dataframe_to_schema,
)


def summary(
    config: Path = typer.Option(
        BASE_DIR / "config.toml",
        "--config",
        "-c",
        help="Path to the configuration TOML file.",
    ),
    input_path: Path | None = typer.Option(
        None,
        "--input",
        "-i",
        help="Path to recommend output (parquet/csv/jsonl/ndjson).",
    ),
    model: str = typer.Option(
        "gemini-2.5-flash",
        "--model",
        "-m",
        help="Gemini model to use for summarization.",
    ),
    use_batch: bool = typer.Option(
        True,
        "--no-batch/--batch",
        help="Whether to use Batch API (faster and cheaper for large batches). Enabled by default.",
    ),
    poll_interval: int = typer.Option(
        30,
        "--poll-interval",
        help="Polling interval for batch job status (seconds, default: 30).",
    ),
    ocr_output_dir: Path = typer.Option(
        BASE_DIR / "data" / "ocr_responses",
        "--ocr-output",
        help="Directory to save OCR outputs (markdown + images).",
    ),
    output: Path = typer.Option(
        BASE_DIR / "data" / "summaries.last.jsonl",
        "--output",
        "-o",
        help="Path to save latest batch snapshot (JSONL, same as last.jsonl).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run/--no-dry-run",
        help="Skip JSONL/Git writes; only emit --output snapshot.",
    ),
) -> None:
    """
    Complete pipeline: OCR papers → Generate summaries → Save to JSONL store.

    This command performs:
    1. Batch OCR processing of papers (converts PDFs to markdown + images)
    2. Batch AI summarization using Gemini API
    3. Saves results to the JSONL summary store with latest snapshot file

    Examples:
        # Process papers from recommend output
        papersys summary --input data/rec.jsonl

        # Custom output paths
        papersys summary --input data/rec.parquet --ocr-output data/ocr --output data/last.jsonl
    """
    from papersys.ocr import ocr_by_id_batch, response2md
    from papersys.summary import summarize_from_path_map

    logger.info("Loading config from {}", config)
    app_config = load_config(AppConfig, config)
    git_store = GitStore(app_config.git_store)
    git_store.ensure_local_copy()
    summary_store = git_store.summary_store

    if input_path is None:
        logger.error("--input is required")
        raise typer.Exit(code=1)

    if not input_path.exists():
        logger.error("Input file not found: {}", input_path)
        raise typer.Exit(code=1)

    try:
        input_df = _load_input_table(input_path)
    except Exception as exc:
        logger.error("Failed to load input file {}: {}", input_path, exc)
        raise typer.Exit(code=1) from exc

    if ID not in input_df.columns:
        logger.error("Input file must contain '{}' column", ID)
        raise typer.Exit(code=1)

    unique_ids_series = (
        input_df.select(pl.col(ID).cast(pl.Utf8()).unique())
        .to_series()
        .drop_nulls()
    )
    paper_ids_list = sorted(unique_ids_series.to_list())

    if not paper_ids_list:
        logger.error("Input file {} does not contain valid paper IDs", input_path)
        raise typer.Exit(code=1)

    logger.info("Total {} unique paper IDs to process", len(paper_ids_list))

    metadata_fields = [TITLE, AUTHORS, PUBLISH_DATE, UPDATE_DATE, SCORE]
    available_fields = [field for field in metadata_fields if field in input_df.columns]
    metadata_by_id: dict[str, dict[str, object]] = {}
    if available_fields:
        metadata_df = (
            input_df
            .select([pl.col(ID).cast(pl.Utf8())] + [pl.col(field) for field in available_fields])
            .unique(subset=[ID], keep="first")
        )
        metadata_by_id = {
            row[ID]: {field: row.get(field) for field in available_fields}
            for row in metadata_df.iter_rows(named=True)
        }

    ocr_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Step 1: Batch OCR processing")
    logger.info("=" * 80)

    papers_to_ocr = []
    papers_skipped = []

    for arxiv_id in paper_ids_list:
        output_paper_dir = ocr_output_dir / arxiv_id
        md_file = output_paper_dir / f"{arxiv_id}.md"

        if md_file.exists():
            logger.debug("Skip {}, OCR result already exists at {}", arxiv_id, md_file)
            papers_skipped.append(arxiv_id)
        else:
            papers_to_ocr.append(arxiv_id)

    if papers_skipped:
        logger.info("Detected {} papers with existing OCR results", len(papers_skipped))

    if papers_to_ocr:
        logger.info("Starting OCR for {} papers...", len(papers_to_ocr))

        try:
            ocr_results = ocr_by_id_batch(
                arxiv_ids=papers_to_ocr,
                wait_for_completion=True,
                poll_interval=10,
            )
            logger.success("OCR batch processing completed for {} papers", len(ocr_results))
        except Exception as exc:
            logger.error("OCR batch processing failed: {}", exc)
            raise typer.Exit(code=1) from exc

        logger.info("Saving OCR results to markdown files...")
        for arxiv_id, ocr_response in ocr_results.items():
            try:
                output_paper_dir = ocr_output_dir / arxiv_id
                response2md(
                    ocr_response=ocr_response,
                    output_dir=output_paper_dir,
                    filename=f"{arxiv_id}.md",
                )
            except Exception as exc:
                logger.error("Failed to save OCR results for {}: {}", arxiv_id, exc)

        logger.success("All OCR results saved to {}", ocr_output_dir)
    else:
        logger.info("All papers already have OCR results, skipping OCR API calls")

    logger.info("=" * 80)
    logger.info("Step 2: Batch AI summarization")
    logger.info("=" * 80)

    path_map = {pid: ocr_output_dir / pid for pid in paper_ids_list}

    try:
        summary_results = summarize_from_path_map(
            path_map=path_map,
            model=model,
            api_key=None,  # Will read from env
            use_batch=use_batch,
            poll_interval=poll_interval,
            max_wait_minutes=None,
        )
        logger.success("Summarization completed for {} papers", len(summary_results))
    except Exception as exc:
        logger.error("Summarization failed: {}", exc)
        raise typer.Exit(code=1) from exc

    logger.info("=" * 80)
    logger.info("Step 3: Saving results")
    logger.info("=" * 80)

    today_iso = date.today().isoformat()

    def _coerce_date(value: object) -> date | None:
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return date.fromisoformat(value)
            except ValueError:
                return None
        return None

    def _coerce_date_str(value: object) -> str | None:
        extracted = _coerce_date(value)
        return extracted.isoformat() if extracted else None

    records: list[dict[str, object]] = []
    for arxiv_id, summary in summary_results.items():
        meta = metadata_by_id.get(arxiv_id, {})

        score_value = meta.get(SCORE) if meta else None
        if score_value is not None:
            try:
                score_value = float(score_value)
            except (TypeError, ValueError):
                score_value = None

        record = {
            ID: arxiv_id,
            TITLE: meta.get(TITLE) if meta else None,
            INSTITUTION: list(summary.institution or []),
            AUTHORS: meta.get(AUTHORS) if meta else None,
            PUBLISH_DATE: _coerce_date_str(meta.get(PUBLISH_DATE)) if meta else None,
            UPDATE_DATE: _coerce_date_str(meta.get(UPDATE_DATE)) if meta else None,
            SUMMARY_DATE: today_iso,
            SUMMARY_MODEL: model,
            SCORE: score_value,
            REASONING_STEP: summary.reasoning_step,
            PROBLEM_BACKGROUND: summary.problem_background,
            METHOD: summary.method,
            EXPERIMENT: summary.experiment,
            ONE_SENTENCE_SUMMARY: summary.one_sentence_summary,
            SLUG: summary.slug,
            KEYWORDS: list(summary.keywords or []),
            FURTHER_THOUGHTS: summary.further_thoughts,
        }
        records.append(record)

    if not records:
        logger.error("No summaries to save")
        raise typer.Exit(code=1)

    summary_df = align_dataframe_to_schema(
        pl.DataFrame(records),
        SUMMARY_RECORD_SCHEMA,
        drop_extra=True,
    )
    typed_records = summary_df.to_dicts()

    try:
        _write_snapshot_file(output, typed_records)
        logger.success("Snapshot saved to {} ({} rows)", output, len(typed_records))
    except Exception as exc:
        logger.error("Failed to write snapshot file {}: {}", output, exc)
        raise typer.Exit(code=1) from exc

    if dry_run:
        logger.info("Dry-run enabled: skipped SummaryStore writes")
    else:
        try:
            report = summary_store.upsert_many(typed_records)
        except Exception as exc:
            logger.error("Failed to update SummaryStore: {}", exc)
            raise typer.Exit(code=1) from exc

        touched = ", ".join(sorted(report.partition_slugs)) if report.partition_slugs else "(none)"
        logger.info("Touched monthly shards: {}", touched)
        logger.info("Summary snapshot path: {} ({} rows)", report.snapshot_path, report.batch_size)
        if report.duplicate_ids:
            logger.warning("Duplicate summary IDs in batch: {}", ", ".join(report.duplicate_ids))

        try:
            git_store.commit_and_push(
                "Update paper summaries",
                paths=[summary_store.root],
            )
            logger.success("Synced SummaryStore to git repo at {}", summary_store.root)
        except Exception as exc:
            logger.error("Failed to push SummaryStore changes: {}", exc)
            raise typer.Exit(code=1) from exc

    logger.info("=" * 80)
    logger.info("Summary Statistics")
    logger.info("=" * 80)
    logger.info("Total papers processed: {}", len(typed_records))
    logger.info("OCR output directory: {}", ocr_output_dir)
    logger.info("Snapshot copy path: {}", output)
    logger.info("Summary store root: {}", summary_store.root)
    logger.info("=" * 80)

    logger.info("Sample results (first 3 papers):")
    for i, (arxiv_id, summary) in enumerate(list(summary_results.items())[:3]):
        typer.echo(f"\n{'='*80}")
        typer.echo(f"Paper {i+1}: {arxiv_id}")
        typer.echo(f"{'='*80}")

        if summary.institution:
            typer.echo(f"\nInstitutions: {', '.join(summary.institution)}")

        if summary.one_sentence_summary:
            typer.echo(f"\nOne-sentence summary:\n{summary.one_sentence_summary}")

        if summary.keywords:
            typer.echo(f"\nKeywords: {', '.join(summary.keywords)}")

    typer.echo(f"\n{'='*80}")
    logger.success("Summary command completed successfully!")


def _load_input_table(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pl.read_parquet(path)
    elif suffix == ".csv":
        df = pl.read_csv(path, schema_overrides=RECOMMEND_INPUT_SCHEMA, infer_schema_length=0)
    elif suffix in {".jsonl", ".ndjson"}:
        df = pl.read_ndjson(path)
    elif suffix == ".json":
        df = pl.read_json(path)
    else:
        raise ValueError(f"Unsupported input format: {path.suffix}")

    return align_dataframe_to_schema(df, RECOMMEND_INPUT_SCHEMA)


def _write_snapshot_file(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as wf:
        for record in records:
            wf.write(json.dumps(record, ensure_ascii=False))
            wf.write("\n")


__all__ = ["summary"]
