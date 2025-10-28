"""Model training utilities for recommendation system."""

import numpy as np
import polars as pl
import sklearn.linear_model
from loguru import logger

from ..config import RecommendConfig
from .sampler import (
    adaptive_difficulty_sampling,
    confidence_weighted_sampling,
)


def train_model(
    prefered_df: pl.DataFrame,
    remaining_df: pl.DataFrame,
    embedding_columns: list[str],
    config: RecommendConfig,
) -> sklearn.linear_model.LogisticRegression:
    """Train the recommendation model."""
    logger.info("Training logistic regression model")

    # Convert labels: 'like' -> 1, 'dislike' -> 0
    prefered_df = prefered_df.with_columns(
        pl.when(pl.col("preference") == "like").then(1).otherwise(0).alias("label")
    ).select("label", *embedding_columns)

    remaining_df = remaining_df.select(*embedding_columns)

    positive_sample_num = prefered_df.filter(pl.col("label") == 1).height
    neg_sample_num = int(config.neg_sample_ratio * positive_sample_num)
    logger.info(
        f"Sampling negatives: {positive_sample_num} positive samples -> {neg_sample_num} negatives"
    )

    pesudo_neg_df = remaining_df.sample(n=neg_sample_num, seed=config.seed)
    pesudo_neg_df = pesudo_neg_df.with_columns(pl.lit(0).alias("label")).select(
        "label", *embedding_columns
    )

    combined_df = pl.concat([prefered_df, pesudo_neg_df], how="vertical")
    logger.info(
        f"Training dataset ready: {combined_df.height} rows "
        f"({positive_sample_num} positive, {neg_sample_num} negative)"
    )

    nan_mask = np.zeros(combined_df.height, dtype=bool)
    for col in embedding_columns:
        col_data = combined_df[col].to_list()
        for i, vec in enumerate(col_data):
            if vec is None or (
                isinstance(vec, (list, np.ndarray)) and np.isnan(vec).any()
            ):
                nan_mask[i] = True

    removed_count = nan_mask.sum()
    if removed_count > 0:
        removal_rate = removed_count / len(combined_df) * 100 if len(combined_df) else 0
        logger.warning(
            f"Removed {removed_count}/{len(combined_df)} ({removal_rate:.2f}%) samples containing NaN values"
        )
        combined_df = combined_df.with_row_index("__idx__")
        valid_indices = np.where(~nan_mask)[0]
        combined_df = combined_df.filter(pl.col("__idx__").is_in(valid_indices)).drop(
            "__idx__"
        )
    else:
        logger.info("No NaN vectors detected in training dataset")

    arrays = []
    for col in embedding_columns:
        col_arr = np.vstack(combined_df[col].to_numpy())
        nan_count = np.isnan(col_arr).sum()
        if nan_count > 0:
            logger.warning(
                f"Column '{col}' contains {nan_count} NaN values; filling with 0"
            )
        arrays.append(col_arr)

    x = np.hstack(arrays)
    y = combined_df.select("label").to_numpy().ravel()

    samples_with_nan = np.isnan(x).any(axis=1).sum()
    if samples_with_nan > 0:
        logger.warning(f"Replacing NaN values in {samples_with_nan} samples with 0")
        x = np.nan_to_num(x, nan=0.0)

    logger.info(f"Training matrix shape: {x.shape}, labels shape: {y.shape}")

    cws_config = config.confidence_weighted_sampling
    if cws_config.enable:
        logger.info("Applying confidence-weighted sampling")
        tmp_model = sklearn.linear_model.LogisticRegression(
            C=config.logistic_regression.C,
            max_iter=config.logistic_regression.max_iter,
            random_state=config.seed,
            class_weight="balanced",
        ).fit(x, y)

        new_positive_embedding = confidence_weighted_sampling(
            x[y == 1],
            tmp_model,
            high_conf_threshold=cws_config.high_conf_threshold,
            high_conf_weight=cws_config.high_conf_weight,
            random_state=config.seed,
        )

        x = np.concatenate((x[y == 0], new_positive_embedding))
        y = np.concatenate((y[y == 0], np.ones(new_positive_embedding.shape[0])))
        logger.info(
            f"Post confidence-weighted sampling: features {x.shape}, labels {y.shape}"
        )

    ads_config = config.adaptive_difficulty_sampling
    if ads_config.enable:
        logger.info("Applying adaptive-difficulty sampling")
        unlabeled_data = np.hstack(
            [np.vstack(remaining_df[col].to_numpy()) for col in embedding_columns]
        )
        x_pos = adaptive_difficulty_sampling(
            x[y == 1],
            unlabeled_data,
            n_neighbors=ads_config.n_neighbors,
            sampling_ratio=ads_config.pos_sampling_ratio,
            random_state=config.seed,
            synthetic_ratio=ads_config.synthetic_ratio,
            k_smote=ads_config.k_smote,
        )
        x = np.concatenate((x[y == 0], x_pos))
        y = np.concatenate((y[y == 0], np.ones(x_pos.shape[0])))
        logger.info(
            f"Post adaptive sampling: features {x.shape}, labels {y.shape}"
        )

    final_model = sklearn.linear_model.LogisticRegression(
        C=config.logistic_regression.C,
        max_iter=config.logistic_regression.max_iter,
        random_state=config.seed,
        class_weight="balanced",
    ).fit(x, y)

    logger.info("Model training complete")
    return final_model
