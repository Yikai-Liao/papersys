"""Recommend papers command."""

from pathlib import Path
from datetime import date

import typer
from loguru import logger

from ..const import BASE_DIR
from ..config import AppConfig, load_config


def _parse_date(value: str | None, name: str) -> date | None:
    """Parse ISO format date arguments."""
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter(
            f"Invalid {name} date '{value}'. Expected format: YYYY-MM-DD."
        ) from exc


def _ensure_output_dir(path: Path) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def _enrich_with_metadata(manager, df):
    """Attach metadata columns for readability."""
    import polars as pl
    import pyarrow as pa
    import pyarrow.compute as pc

    from ..database.name import ID, TITLE, PUBLISH_DATE, UPDATE_DATE
    if df.is_empty():
        return df

    ids = df.select("id").to_series().to_list()
    if not ids:
        return df

    try:
        meta_ds = manager.metadata_table.to_lance()
    except Exception as exc:
        logger.warning("Unable to load metadata table: {}", exc)
        return df

    ids_array = pa.array(ids, type=pa.string())
    filter_expr = pc.is_in(pc.field("id"), ids_array)

    meta_table = meta_ds.to_table(
        columns=[ID, TITLE, PUBLISH_DATE, UPDATE_DATE],
        filter=filter_expr,
        use_scalar_index=len(ids) < 10_000,
    )
    meta_df = pl.from_arrow(meta_table)
    if meta_df.is_empty():
        return df

    return (
        df.join(meta_df, on="id", how="left")
        .with_columns(
            [
                pl.col("score").round(6),
            ]
        )
        .sort("score", descending=True)
    )


def recommend(
    config: Path = typer.Option(
        BASE_DIR / "config.toml",
        "--config",
        "-c",
        help="Path to the configuration TOML file.",
    ),
    categories: list[str] | None = typer.Option(
        None,
        "--category",
        "-g",
        help="Override paper categories (repeat the option for multiple values).",
    ),
    last_n_days: int | None = typer.Option(
        None,
        "--last-n-days",
        "-n",
        help="Restrict prediction to papers updated within the last N days.",
    ),
    start: str | None = typer.Option(
        None,
        "--start",
        help="Start date (YYYY-MM-DD) for filtering papers.",
    ),
    end: str | None = typer.Option(
        None,
        "--end",
        help="End date (YYYY-MM-DD) for filtering papers.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        "-l",
        help=(
            "Maximum number of recommendations to retain (for display and saving). "
            "When omitted, no additional cap is applied."
        ),
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional path to save recommendations (supports .csv or .parquet).",
    ),
) -> None:
    """Train the recommender and display/saves recommended papers."""
    import polars as pl

    from papersys.database.manager import PaperManager
    from papersys.recommend import Recommender

    if last_n_days is not None and (start or end):
        raise typer.BadParameter(
            "--last-n-days cannot be used together with --start/--end.",
            param_name="last_n_days",
        )

    start_date = _parse_date(start, "start")
    end_date = _parse_date(end, "end")

    logger.info("Loading config from {}", config)
    app_config = load_config(AppConfig, config)

    selected_categories = categories or app_config.paper.categories
    if not selected_categories:
        raise typer.BadParameter(
            "No categories provided and config is empty.",
            param_name="category",
        )

    manager = PaperManager(uri=app_config.database.uri)

    logger.info("Initializing recommender with categories {}", selected_categories)
    recommender = Recommender(manager, app_config.recommend)

    try:
        recommender.fit(selected_categories)
        logger.info("Model training complete.")
    except ValueError as exc:
        logger.error("Training failed: {}", exc)
        raise typer.Exit(code=1) from exc

    try:
        predictions = recommender.predict(
            categories=selected_categories,
            last_n_days=last_n_days,
            start_date=start_date,
            end_date=end_date,
        )
    except ValueError as exc:
        logger.error("Prediction failed: {}", exc)
        raise typer.Exit(code=1) from exc

    if predictions.is_empty():
        logger.warning("No predictions generated with the current filters.")
        return

    recommended = predictions.filter(pl.col("show") == 1)
    if recommended.is_empty():
        logger.warning("No papers met the recommendation criteria.")
        return

    if "show" in recommended.columns:
        recommended = recommended.drop("show")

    ranked = recommended.sort("score", descending=True)

    if limit is not None:
        if limit <= 0:
            raise typer.BadParameter(
                "--limit must be a positive integer.", param_name="limit"
            )
        ranked = ranked.head(limit)

    enriched = _enrich_with_metadata(manager, ranked)

    if output is not None:
        _ensure_output_dir(output)
        if output.suffix.lower() == ".parquet":
            enriched.write_parquet(str(output))
        else:
            enriched.write_csv(str(output))
        logger.info("Saved {} recommendations to {}", enriched.height, output)

    display_limit = 20 if limit is None else min(limit, enriched.height)
    display_df = enriched
    if display_limit is not None and display_limit < enriched.height:
        display_df = display_df.head(display_limit)

    typer.echo(display_df)


__all__ = ["recommend"]
