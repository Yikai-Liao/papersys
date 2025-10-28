"""CLI commands for papersys."""

import typer
from dotenv import load_dotenv
from ..const import BASE_DIR
from .init_cmd import init
from .embed_cmd import embed
from .stat_cmd import stat
from .optimize_cmd import optimize
from .fetch_cmd import fetch
from .recommend_cmd import recommend
from .summary_cmd import summary

# Load environment variables before creating the app
load_dotenv(BASE_DIR / ".env")

app = typer.Typer(help="CLI entry point for papersys.")

# Register commands
app.command(help="Initialize Database from Arxiv OAI File hosted on Kaggle")(init)
app.command(help="Embed papers in the database using the specified embedding model.")(embed)
app.command(help="Display statistics and status for database tables.")(stat)
app.command(help="Optimize database tables ")(optimize)
app.command(help="Fetch new papers from arXiv OAI-PMH API and upsert into database.")(fetch)
app.command(help="Train the recommender and show recommended papers.")(recommend)
app.command(help="Summarize papers using Gemini API with structured output.")(summary)


@app.callback()
def main() -> None:
    """Management commands for papersys."""
    pass


__all__ = ["app", "init", "embed", "stat", "optimize", "fetch", "recommend", "summary"]
