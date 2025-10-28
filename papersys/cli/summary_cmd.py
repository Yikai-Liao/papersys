"""Summarize papers command."""

from pathlib import Path
from typing import Dict

import typer
from loguru import logger

from ..const import BASE_DIR
from ..config import AppConfig, load_config


def summary(
    config: Path = typer.Option(
        BASE_DIR / "config.toml",
        "--config",
        "-c",
        help="Path to the configuration TOML file.",
    ),
    parquet: Path | None = typer.Option(
        None,
        "--parquet",
        "-p",
        help="Path to parquet file (e.g., output from recommend command).",
    ),
    ids: list[str] | None = typer.Option(
        None,
        "--id",
        "-i",
        help="Additional paper IDs to summarize (repeat for multiple IDs).",
    ),
    model: str = typer.Option(
        "gemini-2.5-flash",
        "--model",
        "-m",
        help="Gemini model to use for summarization.",
    ),
    use_batch: bool = typer.Option(
        True,
        "--batch/--no-batch",
        help="Whether to use Batch API (faster and cheaper for large batches).",
    ),
    max_wait_time: int = typer.Option(
        600,
        "--max-wait",
        help="Maximum wait time for batch job completion (seconds).",
    ),
    ocr_dir: Path | None = typer.Option(
        None,
        "--ocr-dir",
        help="Directory containing OCR markdown files (e.g., data/ocr_responses_example).",
    ),
) -> None:
    """
    Summarize papers using Gemini API.
    
    Inputs can come from:
    1. A parquet file (--parquet) - typically output from recommend command
    2. Direct paper IDs (--id) - can be repeated multiple times
    
    The command will fetch paper content from the database or OCR directory,
    then call Gemini API to generate structured summaries.
    
    Examples:
        # Summarize papers from recommend output
        papersys summary --parquet recommendations.parquet
        
        # Summarize specific papers
        papersys summary --id 2510.20766 --id 2510.20810
        
        # Use both sources
        papersys summary --parquet recommendations.parquet --id 2510.20766
        
        # Use OCR directory for paper content
        papersys summary --id 2510.20766 --ocr-dir data/ocr_responses_example
    """
    import polars as pl
    from papersys.database.manager import PaperManager
    from papersys.summary import summarize_texts, summarize_from_path_map

    logger.info("Loading config from {}", config)
    app_config = load_config(AppConfig, config)

    # Collect all paper IDs from parquet and CLI args
    paper_ids = set()
    
    if parquet is not None:
        if not parquet.exists():
            logger.error("Parquet file not found: {}", parquet)
            raise typer.Exit(code=1)
        
        try:
            df = pl.read_parquet(parquet)
            if "id" not in df.columns:
                logger.error("Parquet file must contain 'id' column")
                raise typer.Exit(code=1)
            
            parquet_ids = df.select("id").to_series().to_list()
            paper_ids.update(parquet_ids)
            logger.info("Loaded {} paper IDs from parquet file", len(parquet_ids))
        except Exception as exc:
            logger.error("Failed to read parquet file: {}", exc)
            raise typer.Exit(code=1) from exc
    
    if ids:
        paper_ids.update(ids)
        logger.info("Added {} paper IDs from command line", len(ids))
    
    if not paper_ids:
        logger.error("No paper IDs provided. Use --parquet or --id to specify papers.")
        raise typer.Exit(code=1)
    
    logger.info("Total {} unique paper IDs to summarize", len(paper_ids))
    
    # If OCR directory is provided, use summarize_from_path_map
    if ocr_dir is not None:
        if not ocr_dir.exists() or not ocr_dir.is_dir():
            logger.error("OCR directory not found or not a directory: {}", ocr_dir)
            raise typer.Exit(code=1)
        
        logger.info("Using OCR directory for paper content: {}", ocr_dir)
        path_map = {pid: ocr_dir / pid for pid in paper_ids}
        
        try:
            results = summarize_from_path_map(
                path_map=path_map,
                model=model,
                api_key=None,  # Will read from env
                use_batch=use_batch,
                max_wait_time=max_wait_time,
            )
        except Exception as exc:
            logger.error("Summarization failed: {}", exc)
            raise typer.Exit(code=1) from exc
    else:
        # Fetch paper content from database
        logger.info("Fetching paper content from database...")
        manager = PaperManager(uri=app_config.database.uri)
        
        try:
            # Fetch metadata (title + abstract as content)
            import pyarrow as pa
            import pyarrow.compute as pc
            from papersys.database.name import ID, TITLE, ABSTRACT
            
            meta_ds = manager.metadata_table.to_lance()
            ids_array = pa.array(list(paper_ids), type=pa.string())
            filter_expr = pc.is_in(pc.field("id"), ids_array)
            
            meta_table = meta_ds.to_table(
                columns=[ID, TITLE, ABSTRACT],
                filter=filter_expr,
            )
            meta_df = pl.from_arrow(meta_table)
            
            if meta_df.is_empty():
                logger.error("No papers found in database for given IDs")
                raise typer.Exit(code=1)
            
            # Build inputs dict (id -> title + abstract)
            inputs: Dict[str, str] = {}
            for row in meta_df.iter_rows(named=True):
                pid = row["id"]
                title = row.get("title", "")
                abstract = row.get("abstract", "")
                content = f"# {title}\n\n{abstract}"
                inputs[pid] = content
            
            logger.info("Fetched content for {} papers", len(inputs))
            
        except Exception as exc:
            logger.error("Failed to fetch paper content from database: {}", exc)
            raise typer.Exit(code=1) from exc
        
        # Call summarize_texts
        try:
            results = summarize_texts(
                inputs=inputs,
                model=model,
                api_key=None,  # Will read from env
                use_batch=use_batch,
                max_wait_time=max_wait_time,
            )
        except Exception as exc:
            logger.error("Summarization failed: {}", exc)
            raise typer.Exit(code=1) from exc
    
    # Display results
    logger.success("Successfully summarized {} papers", len(results))
    
    for pid, summary in results.items():
        typer.echo(f"\n{'='*80}")
        typer.echo(f"Paper ID: {pid}")
        typer.echo(f"{'='*80}")
        
        if summary.institution:
            typer.echo(f"\n机构: {', '.join(summary.institution)}")
        
        if summary.one_sentence_summary:
            typer.echo(f"\n一句话总结:\n{summary.one_sentence_summary}")
        
        if summary.keywords:
            typer.echo(f"\n关键词: {', '.join(summary.keywords)}")
        
        if summary.problem_background:
            typer.echo(f"\n问题背景:\n{summary.problem_background}")
        
        if summary.method:
            typer.echo(f"\n方法:\n{summary.method}")
        
        if summary.experiment:
            typer.echo(f"\n实验:\n{summary.experiment}")
        
        if summary.further_thoughts:
            typer.echo(f"\n进一步思考:\n{summary.further_thoughts}")
        
        if summary.slug:
            typer.echo(f"\nSlug: {summary.slug}")
    
    typer.echo(f"\n{'='*80}")
    logger.info("Summary command completed")


__all__ = ["summary"]
