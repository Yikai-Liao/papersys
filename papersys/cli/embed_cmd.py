"""Embed papers command."""

import time
import typer
from pathlib import Path
from loguru import logger

from ..const import BASE_DIR, DATA_DIR
from ..config import AppConfig, load_config


def embed(
    config: Path = typer.Option(
        BASE_DIR / "config.toml",
        "--config",
        "-c",
        help="Path to the configuration TOML file.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        "-l",
        help="Limit the number of papers to embed. If not set, embed all papers without embeddings.",
    ),
    # force: bool = typer.Option(
    #     False,
    #     "--force",
    #     "-f",
    #     help="Force re-embedding of all papers, even if they already have embeddings.",
    # ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-d",
        help="Perform a dry run without actually writing embeddings to the database.",
    ),
    start: str | None = typer.Option(
        None,
        "--start",
        "-s",
        help="Start date for filtering papers (format: YYYY-MM-DD). Papers with publish_date or update_date >= start will be included.",
    ),
    end: str | None = typer.Option(
        None,
        "--end",
        "-e",
        help="End date for filtering papers (format: YYYY-MM-DD). Papers with publish_date or update_date <= end will be included.",
    ),
):
    """
    Embed papers in the database using the specified embedding model.
    
    Parameters:
        config (Path): Path to the configuration TOML file.
        limit (int | None): Limit the number of papers to embed.
        dry_run (bool): Perform a dry run without writing to database.
        start (str | None): Start date for filtering papers (YYYY-MM-DD).
        end (str | None): End date for filtering papers (YYYY-MM-DD).
    """
    import pyarrow as pa
    import numpy as np
    from datetime import datetime
    from papersys.database.manager import PaperManager, upsert, add
    from papersys.embedding import google_batch_embedding_with_rate_limit, collect_content
    from papersys.database.name import ID, TITLE, ABSTRACT
    
    # Load config
    logger.info("Loading config from {}", config)
    config = load_config(AppConfig, config)
    logger.debug("Loaded config: {}", config)
    
    # Initialize database manager
    manager = PaperManager(uri=config.database.uri)
    
    # Parse date parameters
    start_date = None
    end_date = None
    if start is not None:
        try:
            start_date = datetime.strptime(start, "%Y-%m-%d").date()
            logger.info("Filtering papers with dates >= {}", start_date)
        except ValueError:
            logger.error("Invalid start date format: {}. Expected YYYY-MM-DD.", start)
            raise typer.Exit(code=1)
    if end is not None:
        try:
            end_date = datetime.strptime(end, "%Y-%m-%d").date()
            logger.info("Filtering papers with dates <= {}", end_date)
        except ValueError:
            logger.error("Invalid end date format: {}. Expected YYYY-MM-DD.", end)
            raise typer.Exit(code=1)
    
    # Filtering out papers to embed
    logger.info("Starting to collect papers to embed.")
    start_time = time.time()
    df = manager.unembeded_papers(
        config.paper.categories, 
        columns=[ID, TITLE, ABSTRACT],
        start=start_date,
        end=end_date
    )
    if limit is not None:
        df = df.head(limit)
    end_time = time.time()
    logger.info("Collected {} unembedded papers in {:.2f} seconds", len(df), end_time - start_time)
    
    contents = collect_content(df)
    
    logger.info("Embedding {} papers using model {}", len(contents), config.embedding.model)
    
    time_start = time.time()
    embeddings: np.ndarray = google_batch_embedding_with_rate_limit(
        model=config.embedding.model,
        inputs=contents,
        output_dimensionality=config.embedding.dim,
        dtype = np.float16,
        batch_size=5000,
        tokens_per_minute=500_000,
    )
    time_end = time.time()
    
    logger.info("Generated embeddings in {:.2f} seconds", time_end - time_start)
    if dry_run:
        logger.info("Dry run enabled, skipping database update.")
        return
    
    logger.debug("Before upsert, embedding table has {} records", len(manager.embedding_table))
    upsert(
        manager.embedding_table,
        pa.Table.from_arrays(
            [
                pa.array(df[ID].to_list()),
                pa.array(embeddings.tolist(), type=pa.list_(pa.float16(), len(embeddings[0]))),
            ],
            names=[ID, "embedding"],
        ),
        ID
    )
    logger.debug("After upsert, embedding table has {} records", len(manager.embedding_table))
