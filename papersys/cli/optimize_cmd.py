"""Optimize command for compacting and optimizing database tables."""

from pathlib import Path
import time
import typer
from loguru import logger

from ..config import AppConfig, load_config
from ..const import BASE_DIR
from ..database.manager import PaperManager


def optimize(
    config: Path = typer.Option(
        BASE_DIR / "config.toml",
        "--config",
        "-c",
        help="Path to the configuration file.",
    ),
) -> None:
    """Optimize all database tables.
    
    This command calls the optimize() method on each table to compact files
    and improve query performance.
    
    Examples:
        # Optimize all tables
        papersys optimize
    """
    # Load config
    logger.info("Loading config from {}", config)
    config = load_config(AppConfig, config)
    
    # Initialize database manager
    logger.info("Connecting to database: {}", config.database.uri)
    manager = PaperManager(uri=config.database.uri)
    logger.info("Starting optimization of database tables...")
    start = time.time()
    manager.optimize()
    end = time.time()
    logger.info("Optimization took {:.2f} seconds.", end - start)
    

if __name__ == "__main__":
    typer.run(optimize)
