"""CLI commands for papersys."""

import typer
from dotenv import load_dotenv

from ..const import BASE_DIR
from .notion_sync_cmd import notion_sync
from .recommend_cmd import recommend
from .summary_cmd import summary

# Load environment variables before creating the app
load_dotenv(BASE_DIR / ".env")

app = typer.Typer(help="CLI entry point for papersys.")

# Register commands
app.command(help="Train the recommender and show recommended papers.")(recommend)
app.command(help="Summarize papers using Gemini API with structured output.")(summary)
app.command(help="Sync summary snapshots into a Notion database.")(notion_sync)


@app.callback()
def main() -> None:
    """Management commands for papersys."""
    pass


__all__ = ["app", "recommend", "summary", "notion_sync"]
