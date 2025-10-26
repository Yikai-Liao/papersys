"""Sampling utilities for recommendation system."""

import numpy as np
from loguru import logger
from sklearn.neighbors import NearestNeighbors


def confidence_weighted_sampling(
    x: np.ndarray,
    model,
    high_conf_threshold: float = 0.9,
    high_conf_weight: float = 2.0,
    random_state: int = 42,
) -> np.ndarray:
    """对正样本进行置信度加权采样。
    
    Args:
        x: 正样本特征矩阵
        model: 已训练的模型
        high_conf_threshold: 高置信度阈值
        high_conf_weight: 高置信度样本的权重
        random_state: 随机种子
    
    Returns:
        采样后的特征矩阵
    """
    n_samples = x.shape[0]
    confidences = model.predict_proba(x)[:, 1]

    # 设置权重
    weights = np.ones_like(confidences)
    high_conf_indices = np.where(confidences >= high_conf_threshold)[0]
    weights[high_conf_indices] = high_conf_weight

    # 记录高置信度样本数量
    n_high_conf = len(high_conf_indices)
    logger.info(
        f"发现{n_high_conf}个高置信度样本 (置信度 >= {high_conf_threshold})。"
        f"权重设置：高置信度样本={high_conf_weight}，其他=1.0"
    )

    # 归一化权重
    sampling_probs = weights / weights.sum()

    # 加权采样
    np.random.seed(random_state)
    sampled_indices = np.random.choice(
        np.arange(n_samples), size=n_samples, replace=True, p=sampling_probs
    )
    sampled_x = x[sampled_indices]

    return sampled_x


def adaptive_difficulty_sampling(
    x_pos: np.ndarray,
    unlabeled_data: np.ndarray,
    n_neighbors: int = 5,
    sampling_ratio: float = 1.0,
    random_state: int = 42,
    synthetic_ratio: float = 0.5,
    k_smote: int = 5,
) -> np.ndarray:
    """基于正样本与背景数据分布的相对关系进行自适应难度加权采样，并合成新样本。
    
    Args:
        x_pos: 正样本特征矩阵
        unlabeled_data: 未标记的背景数据特征矩阵
        n_neighbors: 计算难度时考虑的邻居数量
        sampling_ratio: 采样比例，相对于正样本数量的倍数
        random_state: 随机种子
        synthetic_ratio: 合成数据的比例(0-1)，0表示全部重采样，1表示全部合成
        k_smote: 用于SMOTE合成的近邻数量
    
    Returns:
        采样后的正样本特征矩阵（包含原始样本和合成样本）
    """
    # 设置随机种子
    np.random.seed(random_state)

    n_pos = x_pos.shape[0]

    if n_pos == 0:
        logger.warning("没有正样本，无法进行自适应采样")
        return x_pos

    # 计算目标采样数量
    n_samples = int(n_pos * sampling_ratio)
    if n_samples <= 0:
        logger.warning("计算的采样数量为0，保持原始数据不变")
        return x_pos

    # 计算重采样和合成的数量
    n_synthetic = int(n_samples * synthetic_ratio)
    n_resample = n_samples - n_synthetic

    logger.info(
        f"自适应难度采样: 正样本数={n_pos}, 采样比例={sampling_ratio}, "
        f"目标采样数量={n_samples} (重采样={n_resample}, 合成={n_synthetic})"
    )

    try:
        # 检查并处理 NaN 值
        nan_in_xpos = np.isnan(x_pos).sum()
        if nan_in_xpos > 0:
            logger.warning(f"正样本数据中发现 {nan_in_xpos} 个 NaN 值，将替换为 0")
            x_pos = np.nan_to_num(x_pos, nan=0.0)

        nan_in_unlabeled = np.isnan(unlabeled_data).sum()
        if nan_in_unlabeled > 0:
            logger.warning(f"背景数据中发现 {nan_in_unlabeled} 个 NaN 值，将替换为 0")
            unlabeled_data = np.nan_to_num(unlabeled_data, nan=0.0)

        # 1. 对背景数据建立KNN模型
        nn_background = NearestNeighbors(
            n_neighbors=min(n_neighbors, unlabeled_data.shape[0])
        )
        nn_background.fit(unlabeled_data)

        # 2. 计算每个正样本到最近的n_neighbors个背景数据点的平均距离
        distances, _ = nn_background.kneighbors(x_pos)
        avg_distances = np.mean(distances, axis=1)

        # 3. 转换距离为难度分数（距离越小，难度越大）
        if np.max(avg_distances) - np.min(avg_distances) < 1e-10:
            logger.warning("所有样本到背景数据的距离几乎相同，使用均匀采样")
            difficulties = np.ones(n_pos) / n_pos
        else:
            # 反转距离并归一化
            difficulties = 1.0 / (avg_distances + 1e-6)
            difficulties = difficulties / np.sum(difficulties)

        # 4. 记录难度分布情况
        logger.info(
            f"距离统计: 最小={np.min(avg_distances):.4f}, 最大={np.max(avg_distances):.4f}, "
            f"平均={np.mean(avg_distances):.4f}, 标准差={np.std(avg_distances):.4f}"
        )

        # 5. 基于难度进行重采样
        if n_resample > 0:
            resampled_indices = np.random.choice(
                np.arange(n_pos), size=n_resample, replace=True, p=difficulties
            )
            resampled_x = x_pos[resampled_indices]
        else:
            resampled_x = np.empty((0, x_pos.shape[1]))

        # 6. SMOTE合成新样本
        if n_synthetic > 0:
            synthetic_per_sample = np.zeros(n_pos, dtype=int)

            # 根据难度分配每个样本需要合成的数量
            for _ in range(n_synthetic):
                idx = np.random.choice(np.arange(n_pos), p=difficulties)
                synthetic_per_sample[idx] += 1

            # 7. 创建基于正样本的KNN模型
            k_nn = min(k_smote, n_pos - 1)
            if k_nn <= 0:
                logger.warning(f"正样本数量过少({n_pos})，无法执行SMOTE合成")
                synthetic_x = np.empty((0, x_pos.shape[1]))
            else:
                nn_pos = NearestNeighbors(n_neighbors=k_nn + 1).fit(x_pos)

                synthetic_x = []

                # 8. 为每个正样本生成对应数量的合成样本
                for i in range(n_pos):
                    n_to_generate = synthetic_per_sample[i]
                    if n_to_generate == 0:
                        continue

                    # 找到当前样本的k个最近邻
                    distances_i, indices_i = nn_pos.kneighbors([x_pos[i]])
                    nn_indices = indices_i[0][1:]  # 排除自身

                    # 生成合成样本
                    for _ in range(n_to_generate):
                        nn_idx = np.random.choice(nn_indices)
                        alpha = np.random.random()
                        synthetic_sample = x_pos[i] + alpha * (x_pos[nn_idx] - x_pos[i])
                        synthetic_x.append(synthetic_sample)

                if synthetic_x:
                    synthetic_x = np.array(synthetic_x)
                else:
                    synthetic_x = np.empty((0, x_pos.shape[1]))
        else:
            synthetic_x = np.empty((0, x_pos.shape[1]))

        # 9. 合并重采样样本和合成样本
        final_samples = (
            np.vstack([resampled_x, synthetic_x])
            if synthetic_x.size > 0
            else resampled_x
        )

        logger.info(
            f"最终样本数量: {final_samples.shape[0]} = "
            f"{resampled_x.shape[0]}(重采样) + {synthetic_x.shape[0]}(合成)"
        )

        return final_samples

    except Exception as e:
        logger.error(f"自适应难度采样过程中发生错误: {e}")
        logger.exception("详细错误信息:")
        logger.warning("回退到简单随机采样")
        sampled_indices = np.random.choice(np.arange(n_pos), size=n_samples, replace=True)
        return x_pos[sampled_indices]


def adaptive_sample(
    scores: np.ndarray,
    target_sample_rate: float = 0.15,
    high_threshold: float = 0.95,
    boundary_threshold: float = 0.5,
    random_state: int = 42,
) -> np.ndarray:
    """自适应采样算法，根据分数决定哪些样本应该被推荐。
    
    策略:
    1. 所有高于high_threshold的样本被优先推荐
    2. 如果高分样本数量超过目标数量，随机抽取一部分
    3. 如果高分样本不足，从boundary_threshold到high_threshold之间的分数中按权重采样
    
    Args:
        scores: 每个样本的预测分数
        target_sample_rate: 目标推荐比例
        high_threshold: 高置信度阈值
        boundary_threshold: 边界阈值
        random_state: 随机种子
    
    Returns:
        布尔数组，标记哪些样本被推荐
    """
    np.random.seed(random_state)
    n_samples = len(scores)
    target_count = int(n_samples * target_sample_rate)

    if target_count <= 0:
        target_count = 1

    # 初始化推荐标记数组
    show_flags = np.zeros(n_samples, dtype=bool)

    # 找出所有高分样本
    high_score_mask = scores >= high_threshold
    high_score_indices = np.where(high_score_mask)[0]
    high_score_count = len(high_score_indices)

    logger.info(f"目标推荐数量: {target_count} / {n_samples} ({target_sample_rate*100:.2f}%)")
    logger.info(
        f"高分样本(>={high_threshold:.4f})数量: {high_score_count} "
        f"({high_score_count/n_samples*100:.2f}%)"
    )

    # 情况A: 高分样本足够或超过目标数量
    if high_score_count >= target_count:
        if high_score_count > target_count:
            selected_indices = np.random.choice(
                high_score_indices, target_count, replace=False
            )
            show_flags[selected_indices] = True
            logger.info(f"高分样本超过目标数量，随机选择了{target_count}个")
        else:
            show_flags[high_score_indices] = True
            logger.info(f"高分样本数量恰好等于目标数量")

        return show_flags

    # 情况B: 高分样本不足，需要从中等分数样本中补充
    show_flags[high_score_indices] = True
    remaining_count = target_count - high_score_count

    # 找出边界区域的样本
    boundary_mask = (scores >= boundary_threshold) & (scores < high_threshold)
    boundary_indices = np.where(boundary_mask)[0]
    boundary_count = len(boundary_indices)

    logger.info(
        f"边界样本({boundary_threshold:.4f}-{high_threshold:.4f})数量: {boundary_count} "
        f"({boundary_count/n_samples*100:.2f}%)"
    )

    if boundary_count == 0:
        # 如果没有边界样本，从所有剩余样本中随机选择
        remaining_indices = np.where(~high_score_mask)[0]
        if len(remaining_indices) > 0:
            if len(remaining_indices) > remaining_count:
                selected_indices = np.random.choice(
                    remaining_indices, remaining_count, replace=False
                )
            else:
                selected_indices = remaining_indices

            show_flags[selected_indices] = True
            logger.info(f"无边界样本，从所有剩余样本中随机选择了{len(selected_indices)}个")
    else:
        # 从边界区域按权重采样
        boundary_scores = scores[boundary_indices]
        min_score = boundary_threshold
        max_score = high_threshold
        normalized_scores = (boundary_scores - min_score) / (max_score - min_score)
        weights = np.exp(normalized_scores * 2)
        weights = weights / np.sum(weights)

        # 加权采样
        sample_size = min(remaining_count, boundary_count)
        selected_indices = np.random.choice(
            boundary_indices, sample_size, replace=False, p=weights
        )
        show_flags[selected_indices] = True

        logger.info(f"从边界区域加权采样了{len(selected_indices)}个样本")

        # 如果边界区域样本数量仍不足，从低分区域随机采样补足
        if sample_size < remaining_count:
            still_remaining = remaining_count - sample_size
            low_score_mask = scores < boundary_threshold
            low_score_indices = np.where(low_score_mask)[0]

            if len(low_score_indices) > 0:
                sample_size_low = min(still_remaining, len(low_score_indices))
                selected_indices_low = np.random.choice(
                    low_score_indices, sample_size_low, replace=False
                )
                show_flags[selected_indices_low] = True
                logger.info(f"从低分区域随机采样了{len(selected_indices_low)}个样本")

    return show_flags
