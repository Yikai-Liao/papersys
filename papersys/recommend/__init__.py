"""Recommendation system for paper discovery."""

from .recommender import Recommender
from .sampler import (
    adaptive_difficulty_sampling,
    adaptive_sample,
    confidence_weighted_sampling,
)
from .trainer import train_model

__all__ = [
    "Recommender",
    "adaptive_difficulty_sampling",
    "adaptive_sample",
    "confidence_weighted_sampling",
    "train_model",
]
