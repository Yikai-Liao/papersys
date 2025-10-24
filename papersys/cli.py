import typer

app = typer.Typer(help="CLI entry point for papersys.")


@app.callback()
def main() -> None:
    """Management commands for papersys."""


@app.command()
def hello() -> None:
    """Print a friendly greeting."""
    typer.echo("Hello World")


if __name__ == "__main__":
    app()
