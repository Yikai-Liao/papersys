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
    """训练推荐模型。
    
    Args:
        prefered_df: 偏好数据，包含 preference 列（'like' 或 'dislike'）
        remaining_df: 背景数据
        embedding_columns: 嵌入列名列表
        config: 推荐配置
    
    Returns:
        训练好的逻辑回归模型
    """
    logger.info("开始训练模型...")

    # 转换标签：'like' -> 1, 'dislike' -> 0
    prefered_df = prefered_df.with_columns(
        pl.when(pl.col("preference") == "like").then(1).otherwise(0).alias("label")
    ).select("label", *embedding_columns)

    remaining_df = remaining_df.select(*embedding_columns)

    # 计算正样本数量
    positive_sample_num = prefered_df.filter(pl.col("label") == 1).height
    logger.debug(f"正样本数量: {positive_sample_num}")

    # 采样负样本
    neg_sample_num = int(config.neg_sample_ratio * positive_sample_num)
    logger.debug(f"负样本数量: {neg_sample_num}")

    pesudo_neg_df = remaining_df.sample(n=neg_sample_num, seed=config.seed)
    pesudo_neg_df = pesudo_neg_df.with_columns(pl.lit(0).alias("label")).select(
        "label", *embedding_columns
    )

    # 合并数据
    combined_df = pl.concat([prefered_df, pesudo_neg_df], how="vertical")
    logger.info(f"合并后的DataFrame大小: {combined_df.height} 行")

    # 过滤向量内部包含 NaN 的样本
    logger.info("开始过滤向量内部包含 NaN 的样本...")
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
        logger.warning(
            f"过滤了 {removed_count}/{len(combined_df)} "
            f"({removed_count/len(combined_df)*100:.2f}%) 个含 NaN 的样本"
        )
        combined_df = combined_df.with_row_index("__idx__")
        valid_indices = np.where(~nan_mask)[0]
        combined_df = combined_df.filter(pl.col("__idx__").is_in(valid_indices)).drop(
            "__idx__"
        )
        logger.info(f"过滤后的DataFrame大小: {combined_df.height} 行")
    else:
        logger.info("✅ 没有向量内部包含 NaN 的样本")

    # 转换为 numpy 数组
    arrays = []
    for col in embedding_columns:
        col_arr = np.vstack(combined_df[col].to_numpy())
        nan_count = np.isnan(col_arr).sum()
        if nan_count > 0:
            logger.warning(f"列 '{col}' 中有 {nan_count} 个 NaN，将替换为 0")
        arrays.append(col_arr)

    x = np.hstack(arrays)
    y = combined_df.select("label").to_numpy().ravel()

    # 处理 NaN 值
    samples_with_nan = np.isnan(x).any(axis=1).sum()
    if samples_with_nan > 0:
        logger.warning(f"将 {samples_with_nan} 个样本中的 NaN 值替换为 0")
        x = np.nan_to_num(x, nan=0.0)

    logger.info(f"特征矩阵: {x.shape}, 标签: {y.shape}")

    # 置信度加权采样
    cws_config = config.confidence_weighted_sampling
    if cws_config.enable:
        logger.info("使用置信度加权采样...")
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
        logger.info(f"新的特征矩阵: {x.shape}, 新的标签: {y.shape}")

    # 自适应难度采样
    ads_config = config.adaptive_difficulty_sampling
    if ads_config.enable:
        logger.info("使用自适应难度采样...")
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
        logger.info(f"采样后的特征矩阵: {x.shape}, 标签: {y.shape}")

    # 训练最终模型
    final_model = sklearn.linear_model.LogisticRegression(
        C=config.logistic_regression.C,
        max_iter=config.logistic_regression.max_iter,
        random_state=config.seed,
        class_weight="balanced",
    ).fit(x, y)

    logger.info("模型训练完成")
    return final_model
