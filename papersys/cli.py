import typer
import time
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
    
    
def _init_metadata_table(
    manager,
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
    manager,
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
    manager,
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
    
    # Load preference data
    logger.warning("Preference file loading not yet implemented.")
    # Add preference loading logic here when available
    
    logger.info("Preference table initialization complete.")


@app.command(help="Initialize Database from Arxiv OAI File hosted on Kaggle")
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
    manager = PaperManager(uri=str(DATA_DIR / config.database.name))
    
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
        _init_embedding_table(manager, embed_path, config.embedding.dim, force=force)
    
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
    
@app.command(help="Embed papers in the database using the specified embedding model.")
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
    import pyarrow as pa
    import numpy as np
    from datetime import datetime
    from papersys.database.manager import PaperManager, upsert, add
    from papersys.embedding import google_batch_embedding, collect_content
    from papersys.database.name import ID, TITLE, ABSTRACT
    
    # Load config
    logger.info("Loading config from {}", config)
    config = load_config(AppConfig, config)
    logger.debug("Loaded config: {}", config)
    
    # Initialize database manager
    manager = PaperManager(uri=str(DATA_DIR / config.database.name))
    
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
    embeddings: np.ndarray = google_batch_embedding(
        model=config.embedding.model,
        inputs=contents,
        output_dimensionality=config.embedding.dim,
        dtype = np.float16,
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


@app.command(help="Display statistics and status for database tables.")
def stat(
    config: Path = typer.Option(
        BASE_DIR / "config.toml",
        "--config",
        "-c",
        help="Path to the configuration TOML file.",
    ),
    head: int | None = typer.Option(
        None,
        "--head",
        "-n",
        help="Number of rows to display from each table. If not set, only show statistics.",
    ),
):
    """
    Display statistics and status for all database tables.
    
    This command shows:
    - Table name and existence status
    - Number of records
    - Schema information
    - Index information
    - Optional: First N rows of data (with --head option)
    
    Parameters:
        config (Path): Path to the configuration TOML file.
        head (int | None): Number of rows to display from each table.
    """
    import polars as pl
    from papersys.database.manager import PaperManager
    from papersys.database.name import (
        PAPER_METADATA_TABLE, EMBEDDING_TABLE, PREFERENCE_TABLE,
        ID, EMBEDDING_VECTOR
    )
    
    # Load config
    config = load_config(AppConfig, config)
    
    # Initialize database manager
    manager = PaperManager(uri=str(DATA_DIR / config.database.name))
    
    # Define tables to check
    tables_info = [
        (PAPER_METADATA_TABLE, "Paper Metadata Table"),
        (EMBEDDING_TABLE, "Paper Embedding Table"),
        (PREFERENCE_TABLE, "User Preference Table"),
    ]
    
    typer.echo("=" * 80)
    typer.echo(f"Database Statistics: {DATA_DIR / config.database.name}")
    typer.echo("=" * 80)
    typer.echo()
    
    for table_name, display_name in tables_info:
        typer.echo(f"📊 {display_name} ({table_name})")
        typer.echo("-" * 80)
        
        # Check if table exists
        if table_name not in manager.db.table_names():
            typer.echo(f"❌ Table does not exist")
            typer.echo()
            continue
        
        try:
            table = manager.db[table_name]
            
            # Get basic statistics
            count = len(table)
            typer.echo(f"✅ Table exists")
            typer.echo(f"📈 Total records: {count:,}")
            
            # Show schema
            schema = table.schema
            typer.echo(f"📋 Schema:")
            for field in schema:
                typer.echo(f"   - {field.name}: {field.type}")
            
            # Show index information
            typer.echo(f"🔍 Indices:")
            try:
                indices = table.list_indices()
                if indices:
                    for idx in indices:
                        typer.echo(f"   - {idx}")
                else:
                    typer.echo(f"   - No indices found")
            except Exception as e:
                typer.echo(f"   - Unable to retrieve indices: {e}")
            
            # Show embedding dimension if it's the embedding table
            if table_name == EMBEDDING_TABLE:
                try:
                    dim = manager.embedding_dim
                    typer.echo(f"🎯 Embedding dimension: {dim}")
                except Exception as e:
                    typer.echo(f"⚠️  Could not retrieve embedding dimension: {e}")
            
            # Show sample data if head is specified
            if head is not None and head > 0 and count > 0:
                typer.echo(f"\n📄 First {head} rows:")
                typer.echo("-" * 80)
                try:
                    # Use LanceDB search with limit to get first N rows
                    df_pl = table.search().limit(head).to_polars()
                    
                    # Truncate long fields for better display
                    display_df = df_pl
                    for col in df_pl.columns:
                        # Truncate string columns
                        if df_pl[col].dtype == pl.Utf8 or df_pl[col].dtype == pl.String:
                            display_df = display_df.with_columns(
                                pl.when(pl.col(col).str.len_bytes() > 80)
                                .then(pl.col(col).str.slice(0, 77) + "...")
                                .otherwise(pl.col(col))
                                .alias(col)
                            )
                        # Don't display embedding vectors (too large)
                        elif col == EMBEDDING_VECTOR:
                            display_df = display_df.drop(col)
                            typer.echo(f"   (Column '{EMBEDDING_VECTOR}' hidden for brevity)")
                    
                    typer.echo(display_df)
                except Exception as e:
                    typer.echo(f"⚠️  Error displaying data: {e}")
            
        except Exception as e:
            typer.echo(f"❌ Error accessing table: {e}")
        
        typer.echo()
    
    typer.echo("=" * 80)


if __name__ == "__main__":
    app()
