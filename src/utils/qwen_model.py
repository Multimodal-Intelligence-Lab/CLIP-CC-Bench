"""
Qwen3-Embedding-8B Model Implementation

Implementation for Qwen3-Embedding-8B embedding_model with fine-grained evaluation.
Uses sentence-transformers with GAS-style cosine similarity.
"""

import torch
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import gc
import time

# Import shared types and utilities
import sys
from base_types import EmbeddingResult, EvaluationResult, SimilarityScore
from text_chunking import chunk_text_into_sentences


class QwenModel:
    """Qwen3-Embedding-8B model implementation with fine-grained evaluation support."""

    def __init__(self, model_path: str, device: str = "cuda:0", **kwargs):
        self.model_path = model_path
        self.device = device
        self.batch_size = kwargs.get('batch_size', 8)
        self.max_length = kwargs.get('max_length', 32768)
        self.attn_implementation = kwargs.get('attn_implementation', 'eager')

        # Fine-grained settings
        self.fine_grained_enabled = kwargs.get('fine_grained_enabled', True)

        self.logger = logging.getLogger('qwen_model')
        self.model = None

        # Performance tracking
        self.embed_times = []
        self.memory_usage = []

    def load_model(self):
        """Load Qwen3 model with sentence-transformers."""
        try:
            self.logger.info(f"Loading Qwen3 model from {self.model_path}")

            # Load the model with sentence-transformers
            self.model = SentenceTransformer(
                self.model_path,
                model_kwargs={
                    "attn_implementation": self.attn_implementation,
                    "device_map": "auto"
                },
                tokenizer_kwargs={"padding_side": "left"},
                device=self.device
            )

            self.logger.info("✅ Qwen3 model loaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load Qwen3 model: {e}")
            return False

    def encode_texts(self, texts: List[str], prompt_name: str = None) -> np.ndarray:
        """Encode texts to embeddings.

        Args:
            texts: List of texts to encode
            prompt_name: Optional prompt name for task-specific encoding (e.g., "query")
        """
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.time()

        try:
            # Generate embeddings using sentence-transformers
            with torch.no_grad():
                if prompt_name:
                    embeddings = self.model.encode(
                        texts,
                        prompt_name=prompt_name,
                        batch_size=self.batch_size,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
                else:
                    embeddings = self.model.encode(
                        texts,
                        batch_size=self.batch_size,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )

            # Track performance
            embed_time = time.time() - start_time
            self.embed_times.append(embed_time)

            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
                self.memory_usage.append(memory_used)

            self.logger.debug(f"Encoded {len(texts)} texts in {embed_time:.2f}s")
            return embeddings

        except Exception as e:
            self.logger.error(f"Error encoding texts: {e}")
            raise

    def compute_fine_grained_similarity(
        self,
        gt_chunks: List[str],
        pred_chunks: List[str]
    ) -> Tuple[float, float, float]:
        """
        Compute fine-grained precision, recall, and F1 using raw cosine similarity (GAS-style).

        Args:
            gt_chunks: Ground truth sentence chunks
            pred_chunks: Prediction sentence chunks

        Returns:
            (precision, recall, f1_score) using raw cosine similarity (range: [-1, 1])
        """
        if not gt_chunks or not pred_chunks:
            return 0.0, 0.0, 0.0

        try:
            # Encode all chunks (batched for efficiency)
            gt_embeddings = self.encode_texts(gt_chunks)
            pred_embeddings = self.encode_texts(pred_chunks)

            # Precision: For each prediction chunk, find best GT match
            precision_scores = []
            for pred_emb in pred_embeddings:
                similarities = []
                for gt_emb in gt_embeddings:
                    cosine_sim = float(np.dot(pred_emb, gt_emb))
                    similarities.append(cosine_sim)  # GAS-style: raw cosine
                precision_scores.append(max(similarities))

            precision = np.mean(precision_scores) if precision_scores else 0.0

            # Recall: For each GT chunk, find best prediction match
            recall_scores = []
            for gt_emb in gt_embeddings:
                similarities = []
                for pred_emb in pred_embeddings:
                    cosine_sim = float(np.dot(gt_emb, pred_emb))
                    similarities.append(cosine_sim)  # GAS-style: raw cosine
                recall_scores.append(max(similarities))

            recall = np.mean(recall_scores) if recall_scores else 0.0

            # F1 Score (harmonic mean of precision and recall)
            # Note: With raw cosine, precision/recall could be negative
            if precision > 0 and recall > 0 and (precision + recall) > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score = 0.0

            return float(precision), float(recall), float(f1_score)

        except Exception as e:
            self.logger.error(f"Error computing fine-grained similarity: {e}")
            return 0.0, 0.0, 0.0

    def compute_similarity(self, ground_truth_text: str, prediction_text: str) -> SimilarityScore:
        """Compute both coarse and fine-grained similarity between ground truth and prediction texts.

        Uses GAS-style implementation: raw cosine similarity without [0,1] normalization.
        """
        try:
            start_time = time.time()

            # Coarse-grained: Full text embeddings (GAS-style)
            gt_embedding = self.encode_texts([ground_truth_text])
            pred_embedding = self.encode_texts([prediction_text])

            # GAS implementation: raw cosine similarity (range: [-1, 1])
            cosine_sim = float(np.dot(gt_embedding[0], pred_embedding[0]))

            # For backward compatibility, keep normalized_cosine but set it equal to raw cosine
            # This matches GAS behavior
            normalized_cosine = cosine_sim

            # Fine-grained: Chunk-level embeddings
            fine_precision, fine_recall, fine_f1 = None, None, None
            hm_cf = None
            num_gt_chunks, num_pred_chunks = 0, 0

            if self.fine_grained_enabled:
                gt_chunks = chunk_text_into_sentences(ground_truth_text)
                pred_chunks = chunk_text_into_sentences(prediction_text)
                num_gt_chunks = len(gt_chunks)
                num_pred_chunks = len(pred_chunks)

                if gt_chunks and pred_chunks:
                    fine_precision, fine_recall, fine_f1 = self.compute_fine_grained_similarity(
                        gt_chunks, pred_chunks
                    )

                    # Compute hm-cf (harmonic mean of coarse and fine F1)
                    # Note: Using raw cosine (may be negative), need to handle carefully
                    if cosine_sim > 0 and fine_f1 > 0:
                        hm_cf = 2 * (cosine_sim * fine_f1) / (cosine_sim + fine_f1)
                    else:
                        hm_cf = 0.0

            # Create metadata
            computation_time = time.time() - start_time
            metadata = {
                'embedding_model_name': 'qwen3-8b',
                'ground_truth_length': len(ground_truth_text),
                'prediction_length': len(prediction_text),
                'num_gt_chunks': num_gt_chunks,
                'num_pred_chunks': num_pred_chunks,
                'embedding_dim': gt_embedding.shape[1] if len(gt_embedding.shape) > 1 else len(gt_embedding),
                'computation_time': computation_time,
                'implementation': 'GAS-style'  # Mark as GAS implementation
            }

            return SimilarityScore(
                cosine_similarity=cosine_sim,
                normalized_cosine=normalized_cosine,
                fine_grained_precision=fine_precision,
                fine_grained_recall=fine_recall,
                fine_grained_f1=fine_f1,
                hm_cf=hm_cf,
                metadata=metadata
            )

        except Exception as e:
            self.logger.error(f"Error computing similarity: {e}")
            # Return fallback similarity score on error (should be marked as failure)
            return SimilarityScore(
                cosine_similarity=0.0,
                normalized_cosine=0.0,  # Changed from 0.5 to match GAS behavior
                metadata={'error': str(e), 'fallback_score': True}
            )

    def clear_cache(self):
        """Clear GPU cache and perform garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self.logger.debug("Cache cleared")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.embed_times:
            return {}

        stats = {
            'total_embeddings': len(self.embed_times),
            'avg_embed_time': np.mean(self.embed_times),
            'total_embed_time': np.sum(self.embed_times),
            'min_embed_time': np.min(self.embed_times),
            'max_embed_time': np.max(self.embed_times)
        }

        if self.memory_usage:
            stats.update({
                'avg_memory_usage_gb': np.mean(self.memory_usage),
                'max_memory_usage_gb': np.max(self.memory_usage),
                'min_memory_usage_gb': np.min(self.memory_usage)
            })

        return stats

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'model') and self.model:
            del self.model
        self.clear_cache()


class QwenEvaluator:
    """Isolated evaluator for Qwen3 model with fine-grained support."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger('qwen_evaluator')
        self.model = None

        # Initialize model
        embedding_model_config = config['embedding_model']
        fine_grained_config = embedding_model_config.get('fine_grained', {})
        additional_params = embedding_model_config.get('additional_params', {})

        self.model = QwenModel(
            model_path=embedding_model_config['path'],
            device=config['processing']['device'],
            batch_size=embedding_model_config['batch_size'],
            max_length=embedding_model_config['max_length'],
            attn_implementation=additional_params.get('attn_implementation', 'eager'),
            fine_grained_enabled=fine_grained_config.get('enabled', True)
        )

    def initialize(self) -> bool:
        """Initialize the evaluator."""
        return self.model.load_model()

    def evaluate_single(self, ground_truth_text: str, prediction_text: str,
                       video_id: str) -> EvaluationResult:
        """Evaluate a single ground truth vs prediction pair."""
        try:
            similarity_score = self.model.compute_similarity(ground_truth_text, prediction_text)

            return EvaluationResult(
                video_id=video_id,
                ground_truth_text=ground_truth_text,
                prediction_text=prediction_text,
                similarity_score=similarity_score,
                success=True,
                error_message=None
            )

        except Exception as e:
            self.logger.error(f"Evaluation failed for {video_id}: {e}")
            return EvaluationResult(
                video_id=video_id,
                ground_truth_text=ground_truth_text,
                prediction_text=prediction_text,
                similarity_score=SimilarityScore(0.0, 0.0, {'error': str(e), 'fallback_score': True}),
                success=False,
                error_message=str(e)
            )

    def cleanup(self):
        """Cleanup resources."""
        if self.model:
            self.model.clear_cache()

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if self.model:
            return self.model.get_performance_stats()
        return {}
