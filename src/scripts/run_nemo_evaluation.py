#!/usr/bin/env python3
"""
NVIDIA Llama-Embed-Nemotron-8B Isolated Evaluation Script

Standalone evaluation script for Nemo embedding_model with fine-grained evaluation.
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add paths for imports (flattened structure)
script_dir = Path(__file__).parent
src_dir = script_dir.parent
sys.path.append(str(src_dir / "utils"))

from nemo_model import NemoEvaluator
from config_loader import EmbeddingModelConfigLoader
from result_manager import SharedResultManager
from base_types import EmbeddingEvaluationResult, SimilarityScore


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging configuration."""
    log_config = config.get('logging', {})
    log_level = log_config.get('level', 'INFO')

    # Use embedding_model name for consistency with result_manager
    embedding_model_name = config['embedding_model']['name']
    results_base_dir = Path(config['data_paths']['results_base_dir'])
    log_dir = results_base_dir / "embedding_models" / "logs" / embedding_model_name

    log_dir.mkdir(parents=True, exist_ok=True)

    log_prefix = log_config.get('log_prefix', 'nemo_evaluation')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{log_prefix}_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger('nemo_main')
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def load_ground_truth_data(ground_truth_file: Path) -> List[Dict[str, Any]]:
    """Load ground truth dataset."""
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_predictions(predictions_dir: Path, model_name: str) -> Dict[str, str]:
    """Load predictions for a specific model."""
    pred_file = predictions_dir / f"{model_name}.json"

    if not pred_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_file}")

    with open(pred_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_evaluation(config: Dict[str, Any], target_models: List[str] = None) -> Dict[str, Any]:
    """Run Nemo evaluation for specified models."""
    logger = logging.getLogger('nemo_evaluation')

    # Load ground truth data
    logger.info("Loading ground truth data...")
    ground_truth_file = Path(config['data_paths']['ground_truth_file'])
    ground_truth_data = load_ground_truth_data(ground_truth_file)
    logger.info(f"Loaded {len(ground_truth_data)} ground truth samples")

    # Initialize result manager
    results_base_dir = Path(config['data_paths']['results_base_dir'])
    result_manager = SharedResultManager(results_base_dir, config['embedding_model']['name'])

    # Initialize evaluator
    logger.info("Initializing Nemo evaluator...")
    evaluator = NemoEvaluator(config)

    if not evaluator.initialize():
        logger.error("Failed to initialize Nemo evaluator")
        return {'success': False, 'error': 'Evaluator initialization failed'}

    logger.info("✅ Nemo evaluator initialized successfully")

    # Get models to evaluate
    models_to_evaluate = target_models or config.get('models_to_evaluate', [])
    predictions_dir = Path(config['data_paths']['predictions_dir'])

    # Track results
    all_results = []
    model_results = {}  # Group results by model
    total_evaluations = 0
    successful_evaluations = 0
    processing_config = config.get('processing', {})
    clear_cache_interval = processing_config.get('clear_cache_interval', 25)

    logger.info(f"Starting evaluation for {len(models_to_evaluate)} models")

    for model_idx, model_name in enumerate(models_to_evaluate, 1):
        logger.info(f"[{model_idx}/{len(models_to_evaluate)}] Processing model: {model_name}")

        try:
            # Load predictions for this model
            predictions = load_predictions(predictions_dir, model_name)
            logger.info(f"Loaded {len(predictions)} predictions for {model_name}")

            model_successful = 0
            model_results[model_name] = []  # Initialize results for this model

            # Process each video
            for video_idx, gt_item in enumerate(ground_truth_data, 1):
                video_id = gt_item['id']  # Ground truth uses 'id', not 'video_id'
                ground_truth_text = gt_item['summary']  # Ground truth uses 'summary', not 'text'

                # Get prediction for this video
                if video_id not in predictions:
                    logger.warning(f"No prediction found for video {video_id} in model {model_name}")
                    continue

                prediction_text = predictions[video_id]
                total_evaluations += 1

                # Run evaluation
                try:
                    embedding_result = evaluator.evaluate_single(
                        ground_truth_text, prediction_text, video_id
                    )

                    # Create embedding_model evaluation result
                    embedding_model_result = EmbeddingEvaluationResult(
                        video_id=video_id,
                        model_name=model_name,
                        ground_truth_text=ground_truth_text,
                        prediction_text=prediction_text,
                        embedding_model_scores={config['embedding_model']['name']: embedding_result.similarity_score},
                        success=embedding_result.success,
                        error_message=embedding_result.error_message,
                        timestamp=datetime.now().isoformat()
                    )

                    # Save result
                    if result_manager.save_individual_result(embedding_model_result):
                        all_results.append(embedding_model_result)
                        model_results[model_name].append(embedding_model_result)  # Add to model-specific results
                        if embedding_model_result.success:
                            successful_evaluations += 1
                            model_successful += 1

                    # Progress logging
                    if video_idx % processing_config.get('progress_interval', 10) == 0:
                        success_rate = (model_successful / video_idx) * 100
                        logger.info(f"  Progress: {video_idx}/{len(ground_truth_data)} "
                                   f"({success_rate:.1f}% success)")

                    # Clear cache periodically
                    if video_idx % clear_cache_interval == 0:
                        evaluator.model.clear_cache()
                        logger.debug(f"Cache cleared after {video_idx} evaluations")

                except Exception as e:
                    logger.error(f"Failed to evaluate {video_id} for model {model_name}: {e}")
                    continue

            logger.info(f"✅ Completed {model_name}: {model_successful}/{len(ground_truth_data)} successful")

            # Create per-model summary
            if model_results[model_name]:
                result_manager.save_per_model_summary(model_results[model_name], model_name)
                logger.info(f"📊 Saved per-model summary for {model_name}")

        except FileNotFoundError as e:
            logger.error(f"Skipping model {model_name}: {e}")
            continue
        except Exception as e:
            logger.error(f"Error processing model {model_name}: {e}")
            continue

    # Create overall summary (now just for logging purposes)
    logger.info("Creating evaluation summary...")
    summary = result_manager.create_embedding_model_summary(all_results)

    # Get performance stats
    performance_stats = evaluator.get_stats()

    # Cleanup
    evaluator.cleanup()

    # Final results
    final_results = {
        'success': True,
        'embedding_model_name': 'nemo',
        'total_evaluations': total_evaluations,
        'successful_evaluations': successful_evaluations,
        'success_rate': (successful_evaluations / total_evaluations) * 100 if total_evaluations > 0 else 0,
        'models_processed': len(models_to_evaluate),
        'summary': summary,
        'performance_stats': performance_stats,
        'timestamp': datetime.now().isoformat()
    }

    logger.info(f"🎉 Evaluation completed successfully!")
    logger.info(f"   Total evaluations: {total_evaluations}")
    logger.info(f"   Successful: {successful_evaluations} ({final_results['success_rate']:.1f}%)")
    logger.info(f"   Models processed: {len(models_to_evaluate)}")

    return final_results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Nemo Isolated Evaluation')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--models', nargs='+', help='Specific models to evaluate')
    parser.add_argument('--base-dir', type=str, help='Base directory path')

    args = parser.parse_args()

    try:
        # Determine base directory
        if args.base_dir:
            base_dir = Path(args.base_dir)
        else:
            # Default to project root (2 levels up from script - flattened structure)
            base_dir = Path(__file__).parent.parent.parent

        # Load configuration
        config_loader = EmbeddingModelConfigLoader('nemo', base_dir)
        config = config_loader.load_config()

        # Setup logging
        logger = setup_logging(config)
        logger.info("Starting Nemo isolated evaluation")
        logger.info(f"Base directory: {base_dir}")
        logger.info(f"Config: {config_loader.paths.get_config_file()}")

        # Run evaluation
        results = run_evaluation(config, args.models)

        if results['success']:
            logger.info("✅ Evaluation completed successfully")
            return 0
        else:
            logger.error("❌ Evaluation failed")
            return 1

    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
