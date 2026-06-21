"""
Shared utilities for the isolated encoder system.
"""

from .base_types import (
    EmbeddingResult,
    EvaluationResult,
    SimilarityScore,
    EmbeddingEvaluationResult,
    ModelConfig,
    EmbeddingModelPaths
)

from .result_manager import SharedResultManager
from .config_loader import EmbeddingModelConfigLoader

__all__ = [
    'EmbeddingResult',
    'EvaluationResult',
    'SimilarityScore',
    'EmbeddingEvaluationResult',
    'ModelConfig',
    'EmbeddingModelPaths',
    'SharedResultManager',
    'EmbeddingModelConfigLoader'
]