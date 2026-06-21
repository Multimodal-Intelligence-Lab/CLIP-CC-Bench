"""
Shared Result Management System

Thread-safe result management for concurrent embedding_model evaluations.
Handles CSV and JSON file updates with proper locking.
"""

import csv
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from base_types import EmbeddingEvaluationResult, SimilarityScore


class SharedResultManager:
    """Thread-safe result manager for concurrent embedding_model evaluations."""

    def __init__(self, results_base_dir: Path, embedding_model_name: str):
        self.results_base_dir = Path(results_base_dir)
        self.embedding_model_name = embedding_model_name
        self.logger = logging.getLogger(f'result_manager.{embedding_model_name}')

        # Result directories - embedding_model-isolated structure
        self.individual_csv_dir = self.results_base_dir / "embedding_models" / "individual_results" / "csv" / self.embedding_model_name
        self.individual_json_dir = self.results_base_dir / "embedding_models" / "individual_results" / "json" / self.embedding_model_name
        self.aggregated_results_dir = self.results_base_dir / "embedding_models" / "aggregated_results"
        self.embedding_model_aggregated_dir = self.aggregated_results_dir / self.embedding_model_name
        self.logs_dir = self.results_base_dir / "embedding_models" / "logs" / self.embedding_model_name

        # Ensure directories exist
        for dir_path in [self.individual_csv_dir, self.individual_json_dir, self.aggregated_results_dir, self.embedding_model_aggregated_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def save_individual_result(self, result: EmbeddingEvaluationResult) -> bool:
        """Save individual evaluation result to both CSV and JSON."""
        try:
            # Save to CSV (thread-safe)
            self._save_individual_result_csv(result)

            # Save to JSON (thread-safe)
            self._save_individual_result_json(result)

            return True
        except Exception as e:
            self.logger.error(f"Failed to save individual result for {result.video_id}: {e}")
            return False

    def _save_individual_result_csv(self, result: EmbeddingEvaluationResult):
        """Save individual result to CSV file in embedding_model-specific directory with fine-grained metrics."""
        model_csv_file = self.individual_csv_dir / f"{result.model_name}.csv"
        score = result.embedding_model_scores.get(self.embedding_model_name, SimilarityScore(0.5, 0.5))

        # Read existing data if file exists
        data = []
        if model_csv_file.exists():
            with open(model_csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                data = list(reader)

        # Initialize with header if empty
        if not data:
            data = [[
                'video_id',
                'coarse_similarity',
                'fine_precision',
                'fine_recall',
                'fine_f1',
                'hm_cf'
            ]]

        # Find existing row or add new one
        video_row_idx = None
        for i in range(1, len(data)):
            if len(data[i]) > 0 and data[i][0] == result.video_id:
                video_row_idx = i
                break

        row_data = [
            result.video_id,
            round(score.normalized_cosine, 4),
            round(score.fine_grained_precision or 0.0, 4),
            round(score.fine_grained_recall or 0.0, 4),
            round(score.fine_grained_f1 or 0.0, 4),
            round(score.hm_cf or 0.0, 4)
        ]

        if video_row_idx is not None:
            # Update existing row
            data[video_row_idx] = row_data
        else:
            # Add new row
            data.append(row_data)

        # Write updated data
        with open(model_csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(data)

    def _save_individual_result_json(self, result: EmbeddingEvaluationResult):
        """Save individual result to JSON file in embedding_model-specific directory with fine-grained metrics."""
        model_json_file = self.individual_json_dir / f"{result.model_name}.json"

        # Load existing data if file exists
        data = {}
        if model_json_file.exists():
            try:
                with open(model_json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                self.logger.warning(f"JSON file corrupted for {result.model_name}, starting fresh: {e}")
                data = {}

        # Ensure video_id entry exists
        if result.video_id not in data:
            data[result.video_id] = {
                'reference': result.ground_truth_text,
                'candidate': result.prediction_text,
                'reference_length': len(result.ground_truth_text),
                'candidate_length': len(result.prediction_text)
            }

        # Update with embedding_model's results
        if self.embedding_model_name in result.embedding_model_scores:
            sim_score = result.embedding_model_scores[self.embedding_model_name]

            # Check if this is a fallback score and mark as failure
            is_fallback = sim_score.metadata.get('fallback_score', False) if sim_score.metadata else False
            final_success = result.success and not is_fallback

            # Add fallback information to error message if applicable
            error_msg = result.error_message or ""
            if is_fallback and not error_msg:
                error_msg = "Fallback score used due to evaluation failure"

            data[result.video_id]['similarity_data'] = {
                'coarse_grained': {
                    'cosine_similarity': sim_score.cosine_similarity,
                    'normalized_cosine': sim_score.normalized_cosine
                },
                'fine_grained': {
                    'precision': sim_score.fine_grained_precision,
                    'recall': sim_score.fine_grained_recall,
                    'f1_score': sim_score.fine_grained_f1,
                    'num_gt_chunks': sim_score.metadata.get('num_gt_chunks', 0) if sim_score.metadata else 0,
                    'num_pred_chunks': sim_score.metadata.get('num_pred_chunks', 0) if sim_score.metadata else 0
                },
                'hybrid': {
                    'hm_cf': sim_score.hm_cf
                },
                'metadata': {
                    'success': final_success,
                    'error_message': error_msg,
                    'computation_time': sim_score.metadata.get('computation_time', 0.0) if sim_score.metadata else 0.0,
                    'is_fallback_score': is_fallback
                }
            }

        # Write updated content
        with open(model_json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def create_embedding_model_summary(self, all_results: List[EmbeddingEvaluationResult]) -> Dict[str, Any]:
        """Create aggregated summary for this embedding_model."""
        if not all_results:
            return {}

        # Calculate aggregate statistics
        similarities = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        hm_cf_scores = []
        successful_evaluations = 0

        for result in all_results:
            if result.success and self.embedding_model_name in result.embedding_model_scores:
                sim_score = result.embedding_model_scores[self.embedding_model_name]
                # Check if this is a fallback score - if so, don't count as successful
                is_fallback = sim_score.metadata.get('fallback_score', False) if sim_score.metadata else False
                if not is_fallback:
                    similarities.append(sim_score.normalized_cosine)
                    if sim_score.fine_grained_precision is not None:
                        precision_scores.append(sim_score.fine_grained_precision)
                    if sim_score.fine_grained_recall is not None:
                        recall_scores.append(sim_score.fine_grained_recall)
                    if sim_score.fine_grained_f1 is not None:
                        f1_scores.append(sim_score.fine_grained_f1)
                    if sim_score.hm_cf is not None:
                        hm_cf_scores.append(sim_score.hm_cf)
                    successful_evaluations += 1

        if not similarities:
            return {
                'embedding_model_name': self.embedding_model_name,
                'total_evaluations': len(all_results),
                'successful_evaluations': 0,
                'error_rate': 1.0,
                'timestamp': datetime.now().isoformat()
            }

        # Calculate statistics
        summary = {
            'embedding_model_name': self.embedding_model_name,
            'total_evaluations': len(all_results),
            'successful_evaluations': successful_evaluations,
            'error_rate': (len(all_results) - successful_evaluations) / len(all_results),
            'coarse_grained_stats': {
                'mean': np.mean(similarities),
                'min': np.min(similarities),
                'max': np.max(similarities),
                'std': np.std(similarities)
            },
            'timestamp': datetime.now().isoformat()
        }

        # Add fine-grained stats if available
        if precision_scores:
            summary['fine_grained_stats'] = {
                'precision': {
                    'mean': np.mean(precision_scores),
                    'std': np.std(precision_scores),
                    'min': np.min(precision_scores),
                    'max': np.max(precision_scores)
                },
                'recall': {
                    'mean': np.mean(recall_scores),
                    'std': np.std(recall_scores),
                    'min': np.min(recall_scores),
                    'max': np.max(recall_scores)
                },
                'f1': {
                    'mean': np.mean(f1_scores),
                    'std': np.std(f1_scores),
                    'min': np.min(f1_scores),
                    'max': np.max(f1_scores)
                }
            }

        # Add hybrid stats if available
        if hm_cf_scores:
            summary['hybrid_stats'] = {
                'hm_cf': {
                    'mean': np.mean(hm_cf_scores),
                    'std': np.std(hm_cf_scores),
                    'min': np.min(hm_cf_scores),
                    'max': np.max(hm_cf_scores)
                }
            }

        return summary

    def get_all_model_results(self) -> List[str]:
        """Get list of all model names that have result files."""
        model_names = set()

        # Check CSV files in embedding_model-specific directory
        for csv_file in self.individual_csv_dir.glob("*.csv"):
            model_names.add(csv_file.stem)

        # Check JSON files in embedding_model-specific directory
        for json_file in self.individual_json_dir.glob("*.json"):
            model_names.add(json_file.stem)

        return sorted(list(model_names))

    def load_model_results(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Load existing results for a model from embedding_model-specific directory."""
        json_file = self.individual_json_dir / f"{model_name}.json"

        if json_file.exists():
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load results for {model_name}: {e}")

        return None

    def create_per_model_summary(self, model_results: List[EmbeddingEvaluationResult], model_name: str) -> Dict[str, Any]:
        """Create summary statistics for a specific model with fine-grained metrics."""
        if not model_results:
            return {}

        # Calculate per-model statistics
        similarities = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        hm_cf_scores = []
        successful_evaluations = 0

        for result in model_results:
            if result.success and self.embedding_model_name in result.embedding_model_scores:
                sim_score = result.embedding_model_scores[self.embedding_model_name]
                # Check if this is a fallback score - if so, don't count as successful
                is_fallback = sim_score.metadata.get('fallback_score', False) if sim_score.metadata else False
                if not is_fallback:
                    similarities.append(sim_score.normalized_cosine)
                    if sim_score.fine_grained_precision is not None:
                        precision_scores.append(sim_score.fine_grained_precision)
                    if sim_score.fine_grained_recall is not None:
                        recall_scores.append(sim_score.fine_grained_recall)
                    if sim_score.fine_grained_f1 is not None:
                        f1_scores.append(sim_score.fine_grained_f1)
                    if sim_score.hm_cf is not None:
                        hm_cf_scores.append(sim_score.hm_cf)
                    successful_evaluations += 1

        if not similarities:
            return {
                'embedding_model_name': self.embedding_model_name,
                'model_name': model_name,
                'total_evaluations': len(model_results),
                'successful_evaluations': 0,
                'error_rate': 1.0,
                'timestamp': datetime.now().isoformat()
            }

        # Calculate statistics
        summary = {
            'embedding_model_name': self.embedding_model_name,
            'model_name': model_name,
            'total_evaluations': len(model_results),
            'successful_evaluations': successful_evaluations,
            'error_rate': (len(model_results) - successful_evaluations) / len(model_results),
            'coarse_grained_stats': {
                'mean': round(np.mean(similarities), 6),
                'min': round(np.min(similarities), 6),
                'max': round(np.max(similarities), 6),
                'std': round(np.std(similarities), 6)
            },
            'timestamp': datetime.now().isoformat()
        }

        # Add fine-grained stats if available
        if precision_scores:
            summary['fine_grained_stats'] = {
                'precision': {
                    'mean': round(np.mean(precision_scores), 6),
                    'std': round(np.std(precision_scores), 6),
                    'min': round(np.min(precision_scores), 6),
                    'max': round(np.max(precision_scores), 6)
                },
                'recall': {
                    'mean': round(np.mean(recall_scores), 6),
                    'std': round(np.std(recall_scores), 6),
                    'min': round(np.min(recall_scores), 6),
                    'max': round(np.max(recall_scores), 6)
                },
                'f1': {
                    'mean': round(np.mean(f1_scores), 6),
                    'std': round(np.std(f1_scores), 6),
                    'min': round(np.min(f1_scores), 6),
                    'max': round(np.max(f1_scores), 6)
                }
            }

        # Add hybrid stats if available
        if hm_cf_scores:
            summary['hybrid_stats'] = {
                'hm_cf': {
                    'mean': round(np.mean(hm_cf_scores), 6),
                    'std': round(np.std(hm_cf_scores), 6),
                    'min': round(np.min(hm_cf_scores), 6),
                    'max': round(np.max(hm_cf_scores), 6)
                }
            }

        return summary

    def save_per_model_summary(self, model_results: List[EmbeddingEvaluationResult], model_name: str):
        """Save per-model summary to embedding_model-specific aggregated results."""
        summary = self.create_per_model_summary(model_results, model_name)

        if not summary:
            return

        # Save to embedding_model-specific aggregated directory
        model_summary_file = self.embedding_model_aggregated_dir / f"{model_name}.json"
        with open(model_summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Update aggregated CSV
        self.update_aggregated_csv(model_name, summary)

        # Update cross-embedding_model stats
        self.update_cross_embedding_model_stats(model_name, summary)

    def update_aggregated_csv(self, model_name: str, summary: Dict[str, Any]):
        """Update the aggregated results CSV file by reading from all embedding_model directories."""
        csv_file = self.aggregated_results_dir / "aggregated_results.csv"

        # Discover all embedding_model directories
        embedding_models_base_dir = self.results_base_dir / "embedding_models" / "aggregated_results"
        embedding_model_dirs = [d for d in embedding_models_base_dir.iterdir() if d.is_dir() and d.name != '__pycache__']

        # Collect data from all embedding_model directories
        data = {}
        embedding_models = set()

        for embedding_model_dir in embedding_model_dirs:
            embedding_model_name = embedding_model_dir.name
            embedding_models.add(embedding_model_name)

            # Read all model summaries for this embedding_model
            for model_file in embedding_model_dir.glob("*.json"):
                model = model_file.stem
                try:
                    with open(model_file, 'r', encoding='utf-8') as f:
                        model_summary = json.load(f)
                        if model not in data:
                            data[model] = {}
                        
                        # Store metrics with avg±std format
                        # Coarse
                        coarse_mean = model_summary['coarse_grained_stats']['mean']
                        coarse_std = model_summary['coarse_grained_stats']['std']
                        data[model][f"{embedding_model_name}_coarse_grained"] = f"{coarse_mean:.2f}±{coarse_std:.2f}"

                        # Fine-grained
                        if 'fine_grained_stats' in model_summary:
                            fine_stats = model_summary['fine_grained_stats']
                            for metric in ['precision', 'recall', 'f1']:
                                mean = fine_stats[metric]['mean']
                                std = fine_stats[metric]['std']
                                data[model][f"{embedding_model_name}_fine_grained_{metric}"] = f"{mean:.2f}±{std:.2f}"

                        # Hybrid
                        if 'hybrid_stats' in model_summary:
                            hm_mean = model_summary['hybrid_stats']['hm_cf']['mean']
                            hm_std = model_summary['hybrid_stats']['hm_cf']['std']
                            data[model][f"{embedding_model_name}_harmonic_mean_cf"] = f"{hm_mean:.2f}±{hm_std:.2f}"

                except Exception as e:
                    self.logger.warning(f"Could not read {model_file}: {e}")

        # Build column names (Hierarchical structure flattened for CSV)
        # Row 1: Embedding model names
        # Row 2: Metric names
        # We'll write this as a standard CSV but with specific column ordering
        
        sorted_embedding_models = sorted(list(embedding_models))
        metrics = [
            'coarse_grained',
            'fine_grained_precision',
            'fine_grained_recall',
            'fine_grained_f1',
            'harmonic_mean_cf'
        ]
        
        # Create header rows
        header_row1 = ['vlm']
        header_row2 = ['']
        
        for embed in sorted_embedding_models:
            for _ in metrics:
                header_row1.append(embed)
            header_row2.extend(metrics)

        # Write updated CSV
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header_row1)
            writer.writerow(header_row2)

            for model in sorted(data.keys()):
                row = [model]
                for embed in sorted_embedding_models:
                    for metric in metrics:
                        key = f"{embed}_{metric}"
                        row.append(data[model].get(key, ''))
                writer.writerow(row)

    def update_cross_embedding_model_stats(self, model_name: str, summary: Dict[str, Any]):
        """Update cross-embedding_model comparison statistics (vlm_performance_stats.json)."""
        stats_file = self.aggregated_results_dir / "vlm_performance_stats.json"

        # Load existing stats
        stats = {}
        if stats_file.exists():
            try:
                with open(stats_file, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load vlm_performance_stats: {e}")

        # Initialize structure if empty
        if 'vlm_performance' not in stats:
            stats['vlm_performance'] = {}

        # Helper to format mean±std
        def fmt_ms(mean, std):
            return f"{mean:.2f}±{std:.2f}"

        # Update VLM performance data
        vlm_data = {
            "metrics": {
                "coarse_grained": {
                    "mean": summary['coarse_grained_stats']['mean'],
                    "std": summary['coarse_grained_stats']['std'],
                    "min": summary['coarse_grained_stats']['min'],
                    "max": summary['coarse_grained_stats']['max'],
                    "mean±std": fmt_ms(summary['coarse_grained_stats']['mean'], summary['coarse_grained_stats']['std'])
                }
            },
            "embedding_model": self.embedding_model_name,
            "num_videos_evaluated": summary['total_evaluations'],
            "timestamp": datetime.now().isoformat()
        }

        # Add fine-grained stats
        if 'fine_grained_stats' in summary:
            vlm_data['metrics']['fine_grained'] = {}
            for metric in ['precision', 'recall', 'f1']:
                s = summary['fine_grained_stats'][metric]
                vlm_data['metrics']['fine_grained'][metric] = {
                    "mean": s['mean'],
                    "std": s['std'],
                    "min": s['min'],
                    "max": s['max'],
                    "mean±std": fmt_ms(s['mean'], s['std'])
                }

        # Add hybrid stats
        if 'hybrid_stats' in summary:
            vlm_data['metrics']['hybrid'] = {
                "hm_cf": {
                    "mean": summary['hybrid_stats']['hm_cf']['mean'],
                    "std": summary['hybrid_stats']['hm_cf']['std'],
                    "min": summary['hybrid_stats']['hm_cf']['min'],
                    "max": summary['hybrid_stats']['hm_cf']['max'],
                    "mean±std": fmt_ms(summary['hybrid_stats']['hm_cf']['mean'], summary['hybrid_stats']['hm_cf']['std'])
                }
            }

        # Update the VLM entry
        stats['vlm_performance'][model_name] = vlm_data

        # Update summary
        stats['summary'] = self._calculate_summary_stats(stats)

        # Save updated stats
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

    def _calculate_summary_stats(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics across all VLMs."""
        vlm_perf = stats.get('vlm_performance', {})
        
        if not vlm_perf:
            return {
                'total_vlms_evaluated': 0,
                'embedding_model': self.embedding_model_name,
                'timestamp': datetime.now().isoformat()
            }

        # Collect metrics across VLMs
        hm_scores = []
        coarse_scores = []
        fine_f1_scores = []

        for vlm_data in vlm_perf.values():
            metrics = vlm_data.get('metrics', {})
            
            # HM
            if 'hybrid' in metrics and 'hm_cf' in metrics['hybrid']:
                hm_scores.append(metrics['hybrid']['hm_cf']['mean'])
            
            # Coarse
            if 'coarse_grained' in metrics:
                coarse_scores.append(metrics['coarse_grained']['mean'])
                
            # Fine F1
            if 'fine_grained' in metrics and 'f1' in metrics['fine_grained']:
                fine_f1_scores.append(metrics['fine_grained']['f1']['mean'])

        def calc_dist(scores):
            if not scores:
                return {}
            mean_val = np.mean(scores)
            std_val = np.std(scores)
            return {
                "mean_across_vlms": round(mean_val, 4),
                "std_across_vlms": round(std_val, 4),
                "min": round(np.min(scores), 4),
                "max": round(np.max(scores), 4),
                "mean±std": f"{mean_val:.2f}±{std_val:.2f}"
            }

        return {
            "total_vlms_evaluated": len(vlm_perf),
            "embedding_model": self.embedding_model_name,
            "performance_distribution": {
                "hm_cf": calc_dist(hm_scores),
                "coarse_grained": calc_dist(coarse_scores),
                "fine_f1": calc_dist(fine_f1_scores)
            },
            "timestamp": datetime.now().isoformat()
        }
