from __future__ import annotations

from typing import Mapping

import polars as pl

from ..fields import (
    AUTHORS,
    EXPERIMENT,
    FURTHER_THOUGHTS,
    ID,
    INSTITUTION,
    KEYWORDS,
    METHOD,
    ONE_SENTENCE_SUMMARY,
    PROBLEM_BACKGROUND,
    PUBLISH_DATE,
    REASONING_STEP,
    SCORE,
    SLUG,
    SUMMARY_DATE,
    SUMMARY_MODEL,
    TITLE,
    UPDATE_DATE,
)


SummarySchema = Mapping[str, pl.DataType]


SUMMARY_RECORD_SCHEMA: dict[str, pl.DataType] = {
    ID: pl.Utf8,
    TITLE: pl.Utf8,
    AUTHORS: pl.Utf8,
    INSTITUTION: pl.List(pl.Utf8),
    PUBLISH_DATE: pl.Utf8,
    UPDATE_DATE: pl.Utf8,
    SUMMARY_DATE: pl.Utf8,
    SUMMARY_MODEL: pl.Utf8,
    SCORE: pl.Float64,
    REASONING_STEP: pl.Utf8,
    PROBLEM_BACKGROUND: pl.Utf8,
    METHOD: pl.Utf8,
    EXPERIMENT: pl.Utf8,
    ONE_SENTENCE_SUMMARY: pl.Utf8,
    SLUG: pl.Utf8,
    KEYWORDS: pl.List(pl.Utf8),
    FURTHER_THOUGHTS: pl.Utf8,
}


RECOMMEND_INPUT_SCHEMA: dict[str, pl.DataType] = {
    ID: pl.Utf8,
    TITLE: pl.Utf8,
    AUTHORS: pl.Utf8,
    PUBLISH_DATE: pl.Utf8,
    UPDATE_DATE: pl.Utf8,
    SCORE: pl.Float64,
}


def align_dataframe_to_schema(
    df: pl.DataFrame,
    schema: SummarySchema,
    *,
    drop_extra: bool = False,
) -> pl.DataFrame:
    """Cast/augment dataframe columns so they match the provided schema."""

    exprs = []
    for name, dtype in schema.items():
        if name in df.columns:
            exprs.append(pl.col(name).cast(dtype, strict=False).alias(name))
        else:
            exprs.append(pl.lit(None, dtype=dtype).alias(name))

    df = df.with_columns(exprs)

    if drop_extra:
        df = df.select(list(schema.keys()))

    return df


__all__ = [
    "SUMMARY_RECORD_SCHEMA",
    "RECOMMEND_INPUT_SCHEMA",
    "align_dataframe_to_schema",
]
