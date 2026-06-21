"""
Shared Base Types for Isolated EmbeddingModel System

Common data structures and types used across all embedding_model modules.
"""

import torch
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class SimilarityScore:
    """Container for similarity computation results."""
    # Coarse-grained metrics
    cosine_similarity: float
    normalized_cosine: float

    # Fine-grained metrics
    fine_grained_precision: Optional[float] = None
    fine_grained_recall: Optional[float] = None
    fine_grained_f1: Optional[float] = None

    # Hybrid metric: harmonic mean of coarse and fine F1
    hm_cf: Optional[float] = None

    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EmbeddingResult:
    """Result container for embedding operations."""
    embeddings: torch.Tensor
    input_texts: List[str]
    model_name: str
    device: str
    success: bool = True
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EvaluationResult:
    """Result container for single evaluation operations."""
    video_id: str
    ground_truth_text: str
    prediction_text: str
    similarity_score: SimilarityScore
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class EmbeddingEvaluationResult:
    """Result container for embedding_model evaluation of a single video."""
    video_id: str
    model_name: str
    ground_truth_text: str
    prediction_text: str
    embedding_model_scores: Dict[str, SimilarityScore]
    timestamp: str
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class ModelConfig:
    """Configuration for an individual embedding_model model."""
    name: str
    path: str
    type: str
    batch_size: int
    max_length: int
    device_map: str
    trust_remote_code: bool
    additional_params: Optional[Dict[str, Any]] = None


class EmbeddingModelPaths:
    """Standardized path management for embedding_model modules."""

    def __init__(self, base_dir: Path, embedding_model_name: str):
        self.base_dir = Path(base_dir)
        self.embedding_model_name = embedding_model_name

        # Module-specific directories (flattened structure)
        self.configs_dir = self.base_dir / "configs"
        self.scripts_dir = self.base_dir / "scripts"
        self.utils_dir = self.base_dir / "utils"

        self.results_base_dir = self.base_dir / "results"

        # Result directories (shared across all embedding_models)
        self.individual_csv_dir = self.results_base_dir / "embedding_models" / "individual_results" / "csv"
        self.individual_json_dir = self.results_base_dir / "embedding_models" / "individual_results" / "json"
        self.aggregated_results_dir = self.results_base_dir / "embedding_models" / "aggregated_results"
        self.logs_dir = self.results_base_dir / "embedding_models" / "logs"
        self.cache_dir = self.results_base_dir / "embedding_models" / "cache"

        # Data directories
        self.data_dir = self.base_dir / "data"
        self.ground_truth_file = self.data_dir / "ground_truth" / "clip_cc_dataset.json"
        self.predictions_dir = self.data_dir / "models"

    def ensure_directories(self):
        """Create all necessary directories."""
        for dir_path in [
            self.configs_dir, self.scripts_dir, self.utils_dir,
            self.individual_csv_dir, self.individual_json_dir,
            self.aggregated_results_dir, self.logs_dir, self.cache_dir
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_config_file(self) -> Path:
        """Get the embedding_model-specific config file path."""
        return self.configs_dir / f"{self.embedding_model_name}.yaml"

    def get_requirements_file(self) -> Path:
        """Get the embedding_model-specific requirements file path."""
        return self.configs_dir / "requirements" / f"{self.embedding_model_name}.txt"

    def get_run_script(self) -> Path:
        """Get the embedding_model-specific run script path."""
        return self.scripts_dir / f"run_{self.embedding_model_name}_evaluation.py"
