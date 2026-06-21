"""
Shared Configuration Loader for Isolated EmbeddingModel System

Handles loading and validation of embedding_model-specific configurations.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List
import torch

from base_types import ModelConfig, EmbeddingModelPaths
from paths import get_project_paths


class EmbeddingModelConfigLoader:
    """Configuration loader for individual isolated embedding_models."""

    def __init__(self, embedding_model_name: str, base_dir: Path = None):
        self.embedding_model_name = embedding_model_name
        self.project_paths = get_project_paths()

        # If base_dir is provided, use it; otherwise use centralized path system
        if base_dir:
            self.base_dir = Path(base_dir)
            self.paths = EmbeddingModelPaths(base_dir / "src", embedding_model_name)
        else:
            self.base_dir = self.project_paths.get_base_dir()
            self.paths = EmbeddingModelPaths(self.base_dir / "src", embedding_model_name)

        self.logger = logging.getLogger(f'config_loader.{embedding_model_name}')

    def load_config(self) -> Dict[str, Any]:
        """Load embedding_model-specific configuration."""
        config_file = self.paths.get_config_file()

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Resolve relative paths to absolute paths FIRST
        self._resolve_paths(config)

        # Then validate configuration
        self._validate_config(config)

        # Add computed paths
        config['paths'] = self._get_computed_paths()

        return config

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate embedding_model configuration."""
        required_sections = ['embedding_model', 'processing', 'data_paths']

        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section '{section}' in configuration")

        # Validate embedding_model section
        embedding_model_config = config['embedding_model']
        required_embedding_model_fields = ['name', 'path', 'type', 'batch_size', 'max_length']

        for field in required_embedding_model_fields:
            if field not in embedding_model_config:
                raise ValueError(f"Missing required embedding_model field: {field}")

        # Validate embedding_model path (skip validation for HuggingFace repo names)
        embedding_model_path = embedding_model_config['path']
        if '/' in embedding_model_path and not embedding_model_path.startswith('/'):
            # This looks like a HuggingFace repo name, skip path validation
            self.logger.info(f"Using HuggingFace repository: {embedding_model_path}")
        else:
            # This is a local path, validate it exists
            path_obj = Path(embedding_model_path)
            if not path_obj.exists():
                raise ValueError(f"EmbeddingModel model path does not exist: {embedding_model_path}")

        # Validate processing settings
        processing = config['processing']

        # Check GPU availability and set device accordingly
        if processing.get('use_gpu', True):
            if not torch.cuda.is_available():
                self.logger.warning("CUDA not available, falling back to CPU")
                processing['use_gpu'] = False
                processing['device'] = 'cpu'
            else:
                if 'device' not in processing:
                    processing['device'] = 'cuda:0'

        # Validate data paths
        data_paths = config['data_paths']
        required_paths = ['ground_truth_file', 'predictions_dir', 'results_base_dir']

        for path_key in required_paths:
            if path_key not in data_paths:
                raise ValueError(f"Missing required data path: {path_key}")

            path_obj = Path(data_paths[path_key])
            if path_key == 'results_base_dir':
                # Results directory can be created if it doesn't exist
                path_obj.mkdir(parents=True, exist_ok=True)
            elif not path_obj.exists():
                raise ValueError(f"Data path does not exist: {path_obj}")

        self.logger.info("✅ Configuration validation passed")

    def _resolve_paths(self, config: Dict[str, Any]) -> None:
        """Resolve relative paths to absolute paths using project base directory."""
        base_dir = self.project_paths.get_base_dir()

        # Resolve data paths
        if 'data_paths' in config:
            data_paths = config['data_paths']
            for key, path in data_paths.items():
                if isinstance(path, str) and not Path(path).is_absolute():
                    data_paths[key] = str(base_dir / path)

        # Resolve logging paths
        if 'logging' in config and 'log_dir' in config['logging']:
            log_dir = config['logging']['log_dir']
            if isinstance(log_dir, str) and not Path(log_dir).is_absolute():
                config['logging']['log_dir'] = str(base_dir / log_dir)

        # Resolve embedding_model model paths
        if 'embedding_model' in config and 'path' in config['embedding_model']:
            embedding_model_path = config['embedding_model']['path']
            if isinstance(embedding_model_path, str) and not Path(embedding_model_path).is_absolute():
                # embedding_models directory is in project root
                config['embedding_model']['path'] = str(base_dir / embedding_model_path)

    def _get_computed_paths(self) -> Dict[str, str]:
        """Get computed paths for the embedding_model."""
        return {
            'configs_dir': str(self.paths.configs_dir),
            'scripts_dir': str(self.paths.scripts_dir),
            'utils_dir': str(self.paths.utils_dir),
            'individual_csv_dir': str(self.paths.individual_csv_dir),
            'individual_json_dir': str(self.paths.individual_json_dir),
            'aggregated_results_dir': str(self.paths.aggregated_results_dir),
            'logs_dir': str(self.paths.logs_dir),
            'cache_dir': str(self.paths.cache_dir)
        }

    def create_model_config(self, config: Dict[str, Any]) -> ModelConfig:
        """Create ModelConfig object from configuration."""
        embedding_model_config = config['embedding_model']

        return ModelConfig(
            name=embedding_model_config['name'],
            path=embedding_model_config['path'],
            type=embedding_model_config['type'],
            batch_size=embedding_model_config['batch_size'],
            max_length=embedding_model_config['max_length'],
            device_map=embedding_model_config.get('device_map', 'auto'),
            trust_remote_code=embedding_model_config.get('trust_remote_code', True),
            additional_params=embedding_model_config.get('additional_params', {})
        )

    def get_models_to_evaluate(self, config: Dict[str, Any]) -> List[str]:
        """Get list of text generation models to evaluate."""
        return config.get('models_to_evaluate', [])

    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
        """Get GPU information."""
        gpu_info = {
            'cuda_available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'devices': []
        }

        if gpu_info['cuda_available']:
            for i in range(gpu_info['device_count']):
                device_props = torch.cuda.get_device_properties(i)
                gpu_info['devices'].append({
                    'id': i,
                    'name': device_props.name,
                    'memory_total': device_props.total_memory,
                    'memory_free': torch.cuda.mem_get_info(i)[0] if torch.cuda.is_available() else 0
                })

        return gpu_info

    def create_default_config(self, embedding_model_name: str, embedding_model_path: str, embedding_model_type: str) -> Dict[str, Any]:
        """Create a default configuration for a embedding_model."""
        return {
            'embedding_model': {
                'name': embedding_model_name,
                'path': embedding_model_path,
                'type': embedding_model_type,
                'batch_size': 16,
                'max_length': 4096,
                'device_map': 'auto',
                'trust_remote_code': True,
                'additional_params': {},
                'fine_grained': {
                    'enabled': True,
                    'chunking_method': 'nltk'
                }
            },
            'processing': {
                'use_gpu': True,
                'device': 'cuda:0',
                'clear_cache_interval': 25,
                'force_gc_interval': 50,
                'save_intermediate_results': True
            },
            'data_paths': {
                'ground_truth_file': str(self.base_dir / "data" / "ground_truth" / "clip_cc_dataset.json"),
                'predictions_dir': str(self.base_dir / "data" / "models"),
                'results_base_dir': str(self.base_dir / "results")
            },
            'logging': {
                'level': 'INFO',
                'log_dir': f'results/embedding_models/logs',
                'log_prefix': f'{embedding_model_name}_evaluation'
            },
            'embedding': {
                'cache_embeddings': True,
                'normalize_text': False,
                'truncate_strategy': 'none',
                'similarity_metric': 'cosine',
                'normalize_embeddings': True
            }
        }
