"""Sampling utilities for the recommendation system."""

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
    """Perform confidence-weighted resampling on positive samples."""
    n_samples = x.shape[0]
    confidences = model.predict_proba(x)[:, 1]

    weights = np.ones_like(confidences)
    high_conf_indices = np.where(confidences >= high_conf_threshold)[0]
    weights[high_conf_indices] = high_conf_weight

    n_high_conf = len(high_conf_indices)
    logger.info(
        f"Confidence-weighted sampling: {n_high_conf} high-confidence samples "
        f"(>={high_conf_threshold:.2f}); weights {high_conf_weight:.2f} vs 1.0"
    )

    sampling_probs = weights / weights.sum()

    np.random.seed(random_state)
    sampled_indices = np.random.choice(
        np.arange(n_samples), size=n_samples, replace=True, p=sampling_probs
    )
    return x[sampled_indices]


def adaptive_difficulty_sampling(
    x_pos: np.ndarray,
    unlabeled_data: np.ndarray,
    n_neighbors: int = 5,
    sampling_ratio: float = 1.0,
    random_state: int = 42,
    synthetic_ratio: float = 0.5,
    k_smote: int = 5,
) -> np.ndarray:
    """Adaptive-difficulty sampling with optional SMOTE-style synthesis."""
    np.random.seed(random_state)

    n_pos = x_pos.shape[0]
    if n_pos == 0:
        logger.warning("Adaptive difficulty sampling skipped: no positive samples")
        return x_pos

    n_samples = int(n_pos * sampling_ratio)
    if n_samples <= 0:
        logger.warning("Adaptive difficulty sampling target is 0; keeping original data")
        return x_pos

    n_synthetic = int(n_samples * synthetic_ratio)
    n_resample = n_samples - n_synthetic

    logger.info(
        f"Adaptive difficulty sampling: positives={n_pos}, target={n_samples} "
        f"(resample={n_resample}, synthetic={n_synthetic})"
    )

    try:
        nan_in_xpos = np.isnan(x_pos).sum()
        if nan_in_xpos > 0:
            logger.warning(
                f"Positive samples contain {nan_in_xpos} NaN entries; filling with 0"
            )
            x_pos = np.nan_to_num(x_pos, nan=0.0)

        nan_in_unlabeled = np.isnan(unlabeled_data).sum()
        if nan_in_unlabeled > 0:
            logger.warning(
                f"Background data contain {nan_in_unlabeled} NaN entries; filling with 0"
            )
            unlabeled_data = np.nan_to_num(unlabeled_data, nan=0.0)

        nn_background = NearestNeighbors(
            n_neighbors=min(n_neighbors, unlabeled_data.shape[0])
        ).fit(unlabeled_data)

        distances, _ = nn_background.kneighbors(x_pos)
        avg_distances = np.mean(distances, axis=1)

        if np.max(avg_distances) - np.min(avg_distances) < 1e-10:
            logger.warning("Distance distribution is almost uniform; using uniform sampling")
            difficulties = np.ones(n_pos) / n_pos
        else:
            difficulties = 1.0 / (avg_distances + 1e-6)
            difficulties = difficulties / np.sum(difficulties)

        min_dist = float(np.min(avg_distances))
        max_dist = float(np.max(avg_distances))
        mean_dist = float(np.mean(avg_distances))
        std_dist = float(np.std(avg_distances))
        logger.debug(
            f"Distance stats (min={min_dist:.4f}, max={max_dist:.4f}, "
            f"mean={mean_dist:.4f}, std={std_dist:.4f})"
        )

        if n_resample > 0:
            resampled_indices = np.random.choice(
                np.arange(n_pos), size=n_resample, replace=True, p=difficulties
            )
            resampled_x = x_pos[resampled_indices]
        else:
            resampled_x = np.empty((0, x_pos.shape[1]))

        if n_synthetic > 0:
            synthetic_per_sample = np.zeros(n_pos, dtype=int)
            for _ in range(n_synthetic):
                idx = np.random.choice(np.arange(n_pos), p=difficulties)
                synthetic_per_sample[idx] += 1

            k_nn = min(k_smote, n_pos - 1)
            if k_nn <= 0:
                logger.warning(
                    f"Not enough positive samples ({n_pos}) to perform SMOTE synthesis"
                )
                synthetic_x = np.empty((0, x_pos.shape[1]))
            else:
                nn_pos = NearestNeighbors(n_neighbors=k_nn + 1).fit(x_pos)
                synthetic_x = []

                for i in range(n_pos):
                    n_to_generate = synthetic_per_sample[i]
                    if n_to_generate == 0:
                        continue

                    _, indices_i = nn_pos.kneighbors([x_pos[i]])
                    nn_indices = indices_i[0][1:]

                    for _ in range(n_to_generate):
                        nn_idx = np.random.choice(nn_indices)
                        alpha = np.random.random()
                        synthetic_sample = x_pos[i] + alpha * (x_pos[nn_idx] - x_pos[i])
                        synthetic_x.append(synthetic_sample)

                synthetic_x = (
                    np.array(synthetic_x) if synthetic_x else np.empty((0, x_pos.shape[1]))
                )
        else:
            synthetic_x = np.empty((0, x_pos.shape[1]))

        final_samples = (
            np.vstack([resampled_x, synthetic_x])
            if synthetic_x.size > 0
            else resampled_x
        )

        logger.info(
            f"Adaptive difficulty sampling produced {final_samples.shape[0]} samples "
            f"({resampled_x.shape[0]} resampled, {synthetic_x.shape[0]} synthetic)"
        )

        return final_samples

    except Exception as exc:
        logger.error(f"Adaptive difficulty sampling failed: {exc}")
        logger.exception("Adaptive difficulty sampling traceback")
        logger.warning("Falling back to simple random sampling")
        sampled_indices = np.random.choice(np.arange(n_pos), size=n_samples, replace=True)
        return x_pos[sampled_indices]


def adaptive_sample(
    scores: np.ndarray,
    target_sample_rate: float = 0.15,
    high_threshold: float = 0.95,
    boundary_threshold: float = 0.5,
    random_state: int = 42,
) -> np.ndarray:
    """Adaptive sampling strategy for converting scores to recommendation flags."""
    np.random.seed(random_state)

    n_samples = len(scores)
    target_count = max(int(n_samples * target_sample_rate), 1)

    high_mask = scores >= high_threshold
    boundary_mask = (scores >= boundary_threshold) & (scores < high_threshold)
    low_mask = scores < boundary_threshold

    high_indices = np.where(high_mask)[0]
    boundary_indices = np.where(boundary_mask)[0]
    low_indices = np.where(low_mask)[0]

    logger.info(
        f"Adaptive sampling target: {target_count} of {n_samples} "
        f"samples ({target_sample_rate * 100:.2f}%)"
    )
    logger.info(
        f"Score distribution: high>={high_threshold:.2f} -> {len(high_indices)}, "
        f"boundary {boundary_threshold:.2f}-{high_threshold:.2f} -> {len(boundary_indices)}, "
        f"low -> {len(low_indices)}"
    )

    show_flags = np.zeros(n_samples, dtype=bool)

    if len(high_indices) >= target_count:
        np.random.shuffle(high_indices)
        show_flags[high_indices[:target_count]] = True
        logger.info(f"Selected {target_count} high-score samples (above threshold)")
        return show_flags

    show_flags[high_indices] = True
    remaining = target_count - len(high_indices)

    if remaining <= 0:
        logger.info("High-score samples exactly meet the target count")
        return show_flags

    if len(boundary_indices) == 0:
        if len(low_indices) == 0:
            logger.warning("No candidates available for remaining quota; returning current selection")
            return show_flags
        np.random.shuffle(low_indices)
        show_flags[low_indices[:remaining]] = True
        logger.info(
            f"Selected {min(remaining, len(low_indices))} additional samples from low-score pool"
        )
        return show_flags

    boundary_scores = scores[boundary_indices]
    boundary_probs = (boundary_scores - boundary_threshold) / (
        high_threshold - boundary_threshold + 1e-8
    )
    boundary_probs = boundary_probs / boundary_probs.sum()

    sample_size = min(remaining, len(boundary_indices))
    selected_boundary = np.random.choice(
        boundary_indices, size=sample_size, replace=False, p=boundary_probs
    )
    show_flags[selected_boundary] = True

    remaining -= sample_size
    if remaining > 0:
        if len(low_indices) == 0:
            logger.warning(
                "Boundary pool exhausted and no low-score samples available; using boundary picks only"
            )
            return show_flags
        np.random.shuffle(low_indices)
        selected_low = low_indices[:remaining]
        show_flags[selected_low] = True
        logger.info(
            f"Selected {sample_size} boundary samples (weighted) and {len(selected_low)} "
            "low-score samples (fallback)"
        )
    else:
        logger.info(f"Selected {sample_size} boundary samples (weighted)")

    return show_flags
