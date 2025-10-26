"""Display database statistics command."""

import typer
from pathlib import Path

from ..const import BASE_DIR, DATA_DIR
from ..config import AppConfig, load_config


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
