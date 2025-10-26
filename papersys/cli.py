import typer
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from loguru import logger

from .const import BASE_DIR, DATA_DIR
from .config import AppConfig, load_config

app = typer.Typer(help="CLI entry point for papersys.")

@app.callback()
def main() -> None:
    """Management commands for papersys."""
    
    
@app.command(help="Iinit Database from Arxiv OAI File host on Kaggle")
def init(
    config: Path = typer.Option(
        BASE_DIR / "config.toml",
        "--config",
        "-c",
        help="Path to the configuration TOML file.",
    ),
    oai_file: str = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to the Arxiv OAI file hosted on Kaggle.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-initialization even if the database already exists.",
    ),
):
    from papersys.database.manager import PaperManager, upsert, add
    from papersys.database.migrate import load_arxiv_oai_snapshot
    
    # Load config
    logger.info("Loading config from {}", config)
    config = load_config(AppConfig, config)
    logger.debug("Loaded config: {}", config)
    
    # Check OAI file existence
    oai_path = Path(oai_file)
    if not oai_path.exists():
        raise FileNotFoundError(f"OAI file not found: {oai_file}")
    # Load OAI file
    logger.info("Loading Arxiv OAI snapshot from {}", oai_path)
    df = load_arxiv_oai_snapshot(oai_path)
    logger.info("Loaded {} records from OAI file", df.height)
    
    # Initialize database
    manager = PaperManager(uri=str(DATA_DIR / config.database.name))
    if force:
        logger.info("Force flag set, dropping existing tables if any.")
        manager.drop_metadata_table()
        manager.drop_embedding_table()
    
    # Create tables
    manager.create_metadata_table()
    manager.create_embedding_table(config.embedding.dim)
    
    # Upsert data
    logger.debug("Before upsert, metadata table has {} records", len(manager.metadata_table))
    if force or len(manager.metadata_table) == 0:
        add(manager.metadata_table, df)
    else:
        upsert(manager.metadata_table, df)
    logger.debug("After upsert, metadata table has {} records", len(manager.metadata_table))
    # Optimize tables
    logger.info("Optimizing database tables.")
    manager.metadata_table.optimize()
    manager.embedding_table.optimize()
    logger.info("Database initialization complete.")
    

if __name__ == "__main__":
    app()
