"""Fetch papers command."""

import typer
from pathlib import Path
from datetime import date, timedelta
from loguru import logger

from ..const import BASE_DIR
from ..config import AppConfig, load_config


def fetch(
    config: Path = typer.Option(
        BASE_DIR / "config.toml",
        "--config",
        "-c",
        help="Path to the configuration TOML file.",
    ),
    start: str | None = typer.Option(
        None,
        "--start",
        "-s",
        help="Start date for fetching papers (format: YYYY-MM-DD). Default: yesterday.",
    ),
    end: str | None = typer.Option(
        None,
        "--end",
        "-e",
        help="End date for fetching papers (format: YYYY-MM-DD). Default: tomorrow.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-d",
        help="Perform a dry run without actually writing to the database.",
    ),
):
    """
    Fetch new papers from arXiv OAI-PMH API and upsert them into the database.
    
    Parameters:
        config (Path): Path to the configuration TOML file.
        start (str | None): Start date for fetching papers (YYYY-MM-DD). Default: yesterday.
        end (str | None): End date for fetching papers (YYYY-MM-DD). Default: tomorrow.
        categories (str | None): Comma-separated categories to fetch. Uses config if not specified.
        dry_run (bool): Perform a dry run without writing to database.
    """
    from papersys.database.manager import PaperManager, upsert
    from papersys.database.migrate import fetch_arxiv_oai
    
    # Load config
    logger.info("Loading config from {}", config)
    config_obj = load_config(AppConfig, config)
    logger.debug("Loaded config: {}", config_obj)
    
    # Parse dates
    today = date.today()
    if start is None:
        start_date = today - timedelta(days=2)
        logger.info("Start date not specified, using default: {}", start_date)
    else:
        try:
            start_date = date.fromisoformat(start)
            logger.info("Using specified start date: {}", start_date)
        except ValueError:
            logger.error("Invalid start date format: {}. Expected YYYY-MM-DD.", start)
            raise typer.Exit(code=1)
    
    end_date = end
    # Fetch papers from arXiv
    logger.info("Fetching papers from arXiv OAI-PMH API from {} to {}", 
                start_date, end_date)
    
    try:
        fetch_results = fetch_arxiv_oai(
            categories=config_obj.paper.categories,
            start=start_date,
            end=end_date
        )
        logger.info("Fetched {} papers from arXiv", len(fetch_results))
    except Exception as e:
        logger.error("Failed to fetch papers from arXiv: {}", e)
        raise typer.Exit(code=1)
    
    if dry_run:
        logger.info("Dry run enabled, skipping database update.")
        logger.info("Would have upserted {} papers to the database.", len(fetch_results))
        return
    
    # Initialize database manager
    logger.info("Initializing database manager at {}", config_obj.database.uri)
    manager = PaperManager(uri=config_obj.database.uri)
    
    # Upsert papers into database
    logger.info("Upserting {} papers into database", len(fetch_results))
    try:
        t = manager.db["paper_metadata"]
        upsert(t, fetch_results)
        logger.info("Successfully upserted papers into database")
    except Exception as e:
        logger.error("Failed to upsert papers into database: {}", e)
        raise typer.Exit(code=1)
    
    logger.info("Fetch process completed successfully.")
