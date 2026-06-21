"""
Shared Path Management System

Centralized path management with configurable base directory for portability.
All paths are relative to a single configurable base directory.
"""

import os
from pathlib import Path
from typing import Dict, Any


class ProjectPaths:
    """Centralized path management for the entire project."""

    # Default base directory - can be easily changed for different systems
    # Default base directory - relative to this file
    # This file is in src/utils/paths.py, so we go up 3 levels to get to the project root
    DEFAULT_BASE_DIR = Path(__file__).resolve().parents[2]

    def __init__(self, base_dir: str = None):
        """Initialize with base directory."""
        self.base_dir = Path(base_dir or os.environ.get('CLIP_CC_BASE_DIR', self.DEFAULT_BASE_DIR))

    def get_base_dir(self) -> Path:
        """Get the base directory."""
        return self.base_dir

    def get_src_dir(self) -> Path:
        """Get src directory."""
        return self.base_dir / "src"

    def get_data_dir(self) -> Path:
        """Get data directory."""
        return self.base_dir / "data"

    def get_results_dir(self) -> Path:
        """Get results directory."""
        return self.base_dir / "results"

    def get_configs_dir(self) -> Path:
        """Get configs directory."""
        return self.get_src_dir() / "configs"

    def get_utils_dir(self) -> Path:
        """Get utils directory."""
        return self.get_src_dir() / "utils"

    def get_scripts_dir(self) -> Path:
        """Get scripts directory."""
        return self.get_src_dir() / "scripts"

    def get_ground_truth_file(self) -> Path:
        """Get ground truth dataset file."""
        return self.get_data_dir() / "ground_truth" / "clip_cc_dataset.json"

    def get_predictions_dir(self) -> Path:
        """Get model predictions directory."""
        return self.get_data_dir() / "models"

    def get_embedding_model_results_dir(self) -> Path:
        """Get embedding_model results base directory."""
        return self.get_results_dir() / "embedding_models"

    def get_embedding_model_logs_dir(self) -> Path:
        """Get embedding_model logs directory."""
        return self.get_embedding_model_results_dir() / "logs"

    def get_individual_results_dir(self) -> Path:
        """Get individual results directory."""
        return self.get_embedding_model_results_dir() / "individual_results"

    def get_aggregated_results_dir(self) -> Path:
        """Get aggregated results directory."""
        return self.get_embedding_model_results_dir() / "aggregated_results"

    def get_embedding_model_config_file(self, embedding_model_name: str) -> Path:
        """Get embedding_model-specific config file (flattened structure)."""
        return self.get_configs_dir() / f"{embedding_model_name}.yaml"

    def get_embedding_models_dir(self) -> Path:
        """Get embedding_model models directory (in project root)."""
        return self.base_dir / "embedding_models"

    def to_dict(self) -> Dict[str, str]:
        """Convert paths to dictionary for config files."""
        return {
            'base_dir': str(self.base_dir),
            'src_dir': str(self.get_src_dir()),
            'data_dir': str(self.get_data_dir()),
            'results_dir': str(self.get_results_dir()),
            'configs_dir': str(self.get_configs_dir()),
            'ground_truth_file': str(self.get_ground_truth_file()),
            'predictions_dir': str(self.get_predictions_dir()),
            'embedding_model_results_dir': str(self.get_embedding_model_results_dir()),
            'embedding_model_logs_dir': str(self.get_embedding_model_logs_dir()),
            'individual_results_dir': str(self.get_individual_results_dir()),
            'aggregated_results_dir': str(self.get_aggregated_results_dir())
        }


# Global instance - can be reconfigured by setting environment variable
project_paths = ProjectPaths()


def set_base_directory(base_dir: str):
    """Set a new base directory for all paths."""
    global project_paths
    project_paths = ProjectPaths(base_dir)


def get_project_paths() -> ProjectPaths:
    """Get the global project paths instance."""
    return project_paths
