"""CLI commands for papersys."""

import typer
from .init_cmd import init
from .embed_cmd import embed
from .stat_cmd import stat

app = typer.Typer(help="CLI entry point for papersys.")

# Register commands
app.command(help="Initialize Database from Arxiv OAI File hosted on Kaggle")(init)
app.command(help="Embed papers in the database using the specified embedding model.")(embed)
app.command(help="Display statistics and status for database tables.")(stat)


@app.callback()
def main() -> None:
    """Management commands for papersys."""
    pass


__all__ = ["app", "init", "embed", "stat"]
