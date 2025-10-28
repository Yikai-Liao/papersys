"""Summarize papers command."""

from pathlib import Path

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
        "-o",
        help="Directory to save OCR outputs (markdown + images).",
    ),
    output: Path = typer.Option(
        BASE_DIR / "data" / "summaries.parquet",
        "--output",
        help="Path to save summary results as parquet file.",
    ),
    skip_ocr: bool = typer.Option(
        False,
        "--skip-ocr",
        help="Skip OCR step and use existing markdown files in ocr_output_dir.",
    ),
) -> None:
    """
    Complete pipeline: OCR papers → Generate summaries → Save as parquet.
    
    This command performs:
    1. Batch OCR processing of papers (converts PDFs to markdown + images)
    2. Batch AI summarization using Gemini API
    3. Saves results to parquet file
    
    Examples:
        # Process papers from recommend output
        papersys summary --parquet recommendations.parquet
        
        # Process specific papers
        papersys summary --id 2510.20766 --id 2510.20810
        
        # Skip OCR if markdown files already exist
        papersys summary --parquet recommendations.parquet --skip-ocr
        
        # Custom output paths
        papersys summary --parquet recs.parquet --ocr-output data/ocr --output summaries.parquet
    """
    import polars as pl
    from papersys.ocr import ocr_by_id_batch, response2md
    from papersys.summary import summarize_from_path_map

    logger.info("Loading config from {}", config)
    app_config = load_config(AppConfig, config)

    # Collect all paper IDs from parquet and CLI args
    paper_ids = []
    
    if parquet is not None:
        if not parquet.exists():
            logger.error("Parquet file not found: {}", parquet)
            raise typer.Exit(code=1)
        
        try:
            df = pl.read_parquet(parquet)
            if "id" not in df.columns:
                logger.error("Parquet file must contain 'id' column")
                raise typer.Exit(code=1)
            
            paper_ids = df.unique("id").select("id").to_series().to_list()
            logger.info("Loaded {} paper IDs from parquet file", len(paper_ids))
        except Exception as exc:
            logger.error("Failed to read parquet file: {}", exc)
            raise typer.Exit(code=1) from exc
    
    if ids:
        paper_ids.extend(ids)
        logger.info("Added {} paper IDs from command line", len(ids))
    
    if not paper_ids:
        logger.error("No paper IDs provided. Use --parquet or --id to specify papers.")
        raise typer.Exit(code=1)
    
    # 去重并排序
    paper_ids_list = sorted(set(paper_ids))
    logger.info("Total {} unique paper IDs to process", len(paper_ids_list))
    
    # Step 1: OCR processing (unless skipped)
    ocr_output_dir.mkdir(parents=True, exist_ok=True)
    
    if not skip_ocr:
        logger.info("=" * 80)
        logger.info("Step 1: Batch OCR processing")
        logger.info("=" * 80)
        
        # 检查哪些论文已经有 OCR 结果
        papers_to_ocr = []
        papers_skipped = []
        
        for arxiv_id in paper_ids_list:
            output_paper_dir = ocr_output_dir / arxiv_id
            md_file = output_paper_dir / f"{arxiv_id}.md"
            
            if md_file.exists():
                logger.debug(f"跳过 {arxiv_id}，OCR 结果已存在: {md_file}")
                papers_skipped.append(arxiv_id)
            else:
                papers_to_ocr.append(arxiv_id)
        
        if papers_skipped:
            logger.info(f"跳过 {len(papers_skipped)} 篇已有 OCR 结果的论文")
        
        if papers_to_ocr:
            logger.info(f"开始 OCR 处理 {len(papers_to_ocr)} 篇论文...")
            
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
            
            # Save OCR results to markdown + images
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
            logger.info("所有论文的 OCR 结果均已存在，跳过 OCR 步骤")
    else:
        logger.info("Skipping OCR step, using existing files in {}", ocr_output_dir)
    
    # Step 2: AI Summarization
    logger.info("=" * 80)
    logger.info("Step 2: Batch AI summarization")
    logger.info("=" * 80)
    
    # Build path map for summarization
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
    
    # Step 3: Save results to parquet
    logger.info("=" * 80)
    logger.info("Step 3: Saving results to parquet")
    logger.info("=" * 80)
    
    # Convert summary results to DataFrame
    records = []
    for arxiv_id, summary in summary_results.items():
        record = {
            "id": arxiv_id,
            "institution": summary.institution,
            "problem_background": summary.problem_background,
            "method": summary.method,
            "experiment": summary.experiment,
            "one_sentence_summary": summary.one_sentence_summary,
            "slug": summary.slug,
            "keywords": summary.keywords,
            "further_thoughts": summary.further_thoughts,
            "reasoning_step": summary.reasoning_step,
        }
        records.append(record)
    
    if not records:
        logger.error("No summaries to save")
        raise typer.Exit(code=1)
    
    try:
        # Create DataFrame and save to parquet
        df = pl.DataFrame(records)
        output.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(output)
        logger.success("Summary results saved to {}", output)
        logger.info("Saved {} summaries", len(records))
    except Exception as exc:
        logger.error("Failed to save parquet file: {}", exc)
        raise typer.Exit(code=1) from exc
    
    # Display summary statistics
    logger.info("=" * 80)
    logger.info("Summary Statistics")
    logger.info("=" * 80)
    logger.info("Total papers processed: {}", len(records))
    logger.info("OCR output directory: {}", ocr_output_dir)
    logger.info("Summary parquet file: {}", output)
    logger.info("=" * 80)
    
    # Display sample results
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
