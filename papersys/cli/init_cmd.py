"""Initialize database tables command."""

import typer
from pathlib import Path
from loguru import logger

from ..const import BASE_DIR, DATA_DIR
from ..config import AppConfig, load_config
from ..database.manager import PaperManager


def _init_metadata_table(
    manager: PaperManager,
    oai_file: Path,
    force: bool = False,
):
    """
    Initialize the metadata table from the Arxiv OAI snapshot file.
    
    Parameters:
        manager: PaperManager instance
        oai_file: Path to the Arxiv OAI file
        force: Whether to force re-initialization of the metadata table
    """
    from papersys.database.manager import upsert, add
    from papersys.database.migrate import load_arxiv_oai_snapshot
    
    # Drop table if force is set
    if force:
        logger.info("Force flag set, dropping metadata table if exists.")
        manager.drop_metadata_table()
    
    # Create metadata table
    manager.create_metadata_table()
    
    # Load OAI file
    logger.info("Loading Arxiv OAI snapshot from {}", oai_file)
    df = load_arxiv_oai_snapshot(oai_file)
    logger.info("Loaded {} records from OAI file", df.height)
    
    # Upsert data
    logger.debug("Before upsert, metadata table has {} records", len(manager.metadata_table))
    if force or len(manager.metadata_table) == 0:
        add(manager.metadata_table, df)
    else:
        upsert(manager.metadata_table, df)
    logger.debug("After upsert, metadata table has {} records", len(manager.metadata_table))
    
    # Optimize table
    logger.info("Optimizing metadata table.")
    manager.metadata_table.optimize()
    logger.info("Metadata table initialization complete.")


def _init_embedding_table(
    manager: PaperManager,
    embedding_file: Path,
    embedding_dim: int,
    force: bool = False,
):
    """
    Initialize the embedding table from an embedding file.
    
    Parameters:
        manager: PaperManager instance
        embedding_file: Path to the embedding file
        embedding_dim: Dimension of the embedding vectors
        force: Whether to force re-initialization of the embedding table
    """
    # TODO: Implement embedding file loading logic
    
    # Drop table if force is set
    if force:
        logger.info("Force flag set, dropping embedding table if exists.")
        manager.drop_embedding_table()
    
    # Create embedding table
    manager.create_embedding_table(embedding_dim)
    
    # Load embedding data if file is provided
    logger.warning("Embedding file loading not yet implemented.")
    # Add embedding loading logic here when available
    
    # Optimize table
    logger.info("Optimizing embedding table.")
    manager.embedding_table.optimize()
    logger.info("Embedding table initialization complete.")


def _init_preference_table(
    manager: PaperManager,
    preference_file: Path,
    force: bool = False,
):
    """
    Initialize the preference table from a preference file.
    
    Parameters:
        manager: PaperManager instance
        preference_file: Path to the user preference file
        force: Whether to force re-initialization of the preference table
    """
    from papersys.database.manager import upsert, add
    from papersys.database.migrate import load_preference_csv
    df = load_preference_csv(preference_file)
       
    # Drop table if force is set
    if force:
        logger.info("Force flag set, dropping preference table if exists.")
        manager.drop_preference_table()
    
    # Create preference table
    manager.create_preference_table()
    upsert(manager.preference_table, df)

    # Add preference loading logic here when available
    
    logger.info("Preference table initialization complete.")


def init(
    config: Path = typer.Option(
        BASE_DIR / "config.toml",
        "--config",
        "-c",
        help="Path to the configuration TOML file.",
    ),
    oai_file: str | None = typer.Option(
        None,
        "--arxiv",
        help="Path to the Arxiv OAI file hosted on Kaggle. If provided, initializes metadata table.",
    ),
    embedding_file: str | None = typer.Option(
        None,
        "--embedding",
        help="Path to the embedding file. If provided, initializes embedding table.",
    ),
    preference_file: str | None = typer.Option(
        None,
        "--preference",
        help="Path to the user preference file. If provided, initializes preference table.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-initialization of the tables corresponding to the provided files.",
    ),
):
    """
    Initialize database tables from specified files.

    This function loads data from various sources and initializes the corresponding
    database tables. Only tables with provided files will be initialized.
    Use --force to drop and recreate tables for the provided files.

    Parameters:
        config (Path): Path to the configuration TOML file, defaults to BASE_DIR / "config.toml".
        oai_file (str | None): Path to the Arxiv OAI file. If provided, initializes metadata table.
        embedding_file (str | None): Path to the embedding file. If provided, initializes embedding table.
        preference_file (str | None): Path to the user preference file. If provided, initializes preference table.
        force (bool): Force re-initialization of the tables corresponding to the provided files.
    """
    from papersys.database.manager import PaperManager
    
    # Load config
    logger.info("Loading config from {}", config)
    config = load_config(AppConfig, config)
    logger.debug("Loaded config: {}", config)
    
    # Initialize database manager
    manager = PaperManager(uri=config.database.uri)
    
    # Initialize metadata table if OAI file is provided
    if oai_file is not None:
        oai_path = Path(oai_file)
        if not oai_path.exists():
            raise FileNotFoundError(f"OAI file not found: {oai_file}")
        _init_metadata_table(manager, oai_path, force=force)
    
    # Initialize embedding table if embedding file is provided
    if embedding_file is not None:
        embed_path = Path(embedding_file)
        if not embed_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {embedding_file}")
    _init_embedding_table(manager, None, config.embedding.dim, force=force)
    
    # Initialize preference table if preference file is provided
    if preference_file is not None:
        pref_path = Path(preference_file)
        if not pref_path.exists():
            raise FileNotFoundError(f"Preference file not found: {preference_file}")
        _init_preference_table(manager, pref_path, force=force)
    
    # Check if at least one file was provided
    if oai_file is None and embedding_file is None and preference_file is None:
        logger.warning("No files provided. Please specify at least one of: --arxiv, --embedding, or --preference")
        return
    
    logger.info("Database initialization complete.")
