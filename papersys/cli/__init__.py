"""CLI commands for papersys."""

import typer
from dotenv import load_dotenv

from ..const import BASE_DIR
from .recommend_cmd import recommend
from .summary_cmd import summary

# Load environment variables before creating the app
load_dotenv(BASE_DIR / ".env")

app = typer.Typer(help="CLI entry point for papersys.")

# Register commands
app.command(help="Train the recommender and show recommended papers.")(recommend)
app.command(help="Summarize papers using Gemini API with structured output.")(summary)


@app.callback()
def main() -> None:
    """Management commands for papersys."""
    pass


__all__ = ["app", "recommend", "summary"]
