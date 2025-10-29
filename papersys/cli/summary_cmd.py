"""Summarize papers command."""

from pathlib import Path
from datetime import date

import pyarrow as pa
import typer
from loguru import logger

from ..const import BASE_DIR
from ..config import AppConfig, load_config
from ..database.manager import PaperManager, upsert
from ..database.name import (
    AUTHORS,
    EXPERIMENT,
    FURTHER_THOUGHTS,
    ID,
    INSTITUTION,
    KEYWORDS,
    METHOD,
    ONE_SENTENCE_SUMMARY,
    PAPER_SUMMARY_TABLE,
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
from ..database.schema import PAPER_SUMMARY_SCHEMA


def summary(
    config: Path = typer.Option(
        BASE_DIR / "config.toml",
        "--config",
        "-c",
        help="Path to the configuration TOML file.",
    ),
    parquet: Path = typer.Option(
        ...,
        "--parquet",
        "-p",
        help="Path to parquet file (e.g., output from recommend command).",
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
        BASE_DIR / "data" / "summaries.parquet",
        "--output",
        "-o",
        help="Path to save summary results as parquet file.",
    ),
) -> None:
    """
    Complete pipeline: OCR papers → Generate summaries → Save to parquet and database.

    This command performs:
    1. Batch OCR processing of papers (converts PDFs to markdown + images)
    2. Batch AI summarization using Gemini API
    3. Saves results to parquet and upserts into the summary table

    Examples:
        # Process papers from recommend output
        papersys summary --parquet recommendations.parquet

        # Custom output paths
        papersys summary --parquet recs.parquet --ocr-output data/ocr --output summaries.parquet
    """
    import polars as pl

    from papersys.ocr import ocr_by_id_batch, response2md
    from papersys.summary import summarize_from_path_map

    logger.info("Loading config from {}", config)
    app_config = load_config(AppConfig, config)

    if not parquet.exists():
        logger.error("Parquet file not found: {}", parquet)
        raise typer.Exit(code=1)

    try:
        input_df = pl.read_parquet(parquet)
    except Exception as exc:
        logger.error("Failed to read parquet file: {}", exc)
        raise typer.Exit(code=1) from exc

    if ID not in input_df.columns:
        logger.error("Parquet file must contain '{}' column", ID)
        raise typer.Exit(code=1)

    unique_ids_series = (
        input_df.select(pl.col(ID).cast(pl.Utf8()).unique())
        .to_series()
        .drop_nulls()
    )
    paper_ids_list = sorted(unique_ids_series.to_list())

    if not paper_ids_list:
        logger.error("No paper IDs found in parquet file {}", parquet)
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
        )
        logger.success("Summarization completed for {} papers", len(summary_results))
    except Exception as exc:
        logger.error("Summarization failed: {}", exc)
        raise typer.Exit(code=1) from exc

    logger.info("=" * 80)
    logger.info("Step 3: Saving results")
    logger.info("=" * 80)

    today = date.today()

    def _coerce_date(value: object) -> date | None:
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return date.fromisoformat(value)
            except ValueError:
                return None
        return None

    records = []
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
            PUBLISH_DATE: _coerce_date(meta.get(PUBLISH_DATE)) if meta else None,
            UPDATE_DATE: _coerce_date(meta.get(UPDATE_DATE)) if meta else None,
            SUMMARY_DATE: today,
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

    try:
        summary_arrow = pa.Table.from_pylist(records, schema=PAPER_SUMMARY_SCHEMA)
        summary_df = pl.from_arrow(summary_arrow)

        output.parent.mkdir(parents=True, exist_ok=True)
        summary_df.write_parquet(str(output))
        logger.success("Summary results saved to {}", output)
        logger.info("Saved {} summaries", len(records))
    except Exception as exc:
        logger.error("Failed to save parquet file: {}", exc)
        raise typer.Exit(code=1) from exc

    try:
        manager = PaperManager(uri=app_config.database.uri)
        summary_table = manager.summary_table
        upsert(summary_table, summary_arrow, primary_key=ID)
        logger.success(
            "Upserted {} summaries into table '{}'",
            len(records),
            PAPER_SUMMARY_TABLE,
        )
    except Exception as exc:
        logger.error("Failed to upsert summaries into database: {}", exc)
        raise typer.Exit(code=1) from exc

    logger.info("=" * 80)
    logger.info("Summary Statistics")
    logger.info("=" * 80)
    logger.info("Total papers processed: {}", len(records))
    logger.info("OCR output directory: {}", ocr_output_dir)
    logger.info("Summary parquet file: {}", output)
    logger.info("=" * 80)

    logger.info("Sample results (first 3 papers):")
    for i, (arxiv_id, summary) in enumerate(list(summary_results.items())[:3]):
        typer.echo(f"\n{'='*80}")
        typer.echo(f"Paper {i+1}: {arxiv_id}")
        typer.echo(f"{'='*80}")

        if summary.institution:
            typer.echo(f"\n机构: {', '.join(summary.institution)}")

        if summary.one_sentence_summary:
            typer.echo(f"\n一句话总结:\n{summary.one_sentence_summary}")

        if summary.keywords:
            typer.echo(f"\n关键词: {', '.join(summary.keywords)}")

    typer.echo(f"\n{'='*80}")
    logger.success("Summary command completed successfully!")


__all__ = ["summary"]
