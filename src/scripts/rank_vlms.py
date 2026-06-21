#!/usr/bin/env python3
"""
VLM Ranking Algorithm using Borda Count and Mean Judge Score
Based on multi-embedding evaluation results
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


# Resolve project root relative to this script
# script is in src/scripts/ -> project root is ../../
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

def load_evaluation_results(base_dir: Path = None) -> Dict:
    """
    Load all evaluation results from the aggregated results directory.

    Args:
        base_dir: Directory containing aggregated results. Defaults to project_root/results/embedding_models/aggregated_results

    Returns:
        Dictionary with structure: {embedding_model: {vlm_name: stats}}
    """
    if base_dir is None:
        base_dir = DEFAULT_RESULTS_DIR / "embedding_models" / "aggregated_results"
    
    base_path = Path(base_dir)
    results = {}

    # Get all embedding model directories
    embedding_dirs = [d for d in base_path.iterdir() if d.is_dir()]

    for embed_dir in embedding_dirs:
        embedding_name = embed_dir.name
        results[embedding_name] = {}

        # Load all VLM JSON files in this embedding directory
        for json_file in embed_dir.glob("*.json"):
            vlm_name = json_file.stem  # filename without .json

            with open(json_file, 'r') as f:
                data = json.load(f)

                # Extract the metrics we need
                results[embedding_name][vlm_name] = {
                    'coarse': data['coarse_grained_stats']['mean'],
                    'coarse_std': data['coarse_grained_stats']['std'],
                    'fine': data['fine_grained_stats']['f1']['mean'],
                    'fine_std': data['fine_grained_stats']['f1']['std'],
                    'hm': data['hybrid_stats']['hm_cf']['mean'],
                    'hm_std': data['hybrid_stats']['hm_cf']['std'],
                    'total_evals': data['total_evaluations']
                }

    return results


def compute_borda_scores(results: Dict) -> Tuple[Dict[str, int], Dict[str, float], Dict[str, float]]:
    """
    Compute Borda scores and MeanJudge scores for each VLM.

    Args:
        results: Dictionary with structure {embedding_model: {vlm_name: stats}}

    Returns:
        Tuple of (borda_scores, mean_judge_scores, mean_judge_std)
        - borda_scores: {vlm_name: total_borda_points}
        - mean_judge_scores: {vlm_name: mean_hm_across_judges}
        - mean_judge_std: {vlm_name: std_hm_across_judges}
    """
    # Get list of all VLMs (assuming all embedding models evaluated the same VLMs)
    embedding_models = list(results.keys())
    first_embedding = embedding_models[0]
    vlm_names = list(results[first_embedding].keys())

    N_vlms = len(vlm_names)
    N_embeds = len(embedding_models)

    print(f"Found {N_vlms} VLMs and {N_embeds} embedding models")
    print(f"VLMs: {vlm_names}")
    print(f"Embedding models: {embedding_models}")
    print()

    # Initialize Borda scores
    borda_scores = {vlm: 0 for vlm in vlm_names}

    # Store all HM scores for computing MeanJudge
    hm_scores = {vlm: [] for vlm in vlm_names}

    # Step 2: Compute Borda points per judge (embedding model)
    for embedding_model in embedding_models:
        # Collect S[j, k] = μ_hm[j, k] for all VLMs under this judge
        vlm_scores = []
        for vlm in vlm_names:
            hm_score = results[embedding_model][vlm]['hm']
            vlm_scores.append((vlm, hm_score))
            hm_scores[vlm].append(hm_score)

        # Sort VLMs by HM score in descending order
        vlm_scores.sort(key=lambda x: x[1], reverse=True)

        # Assign Borda points
        print(f"Judge: {embedding_model}")
        for rank, (vlm, score) in enumerate(vlm_scores, start=1):
            borda_points = N_vlms - rank
            borda_scores[vlm] += borda_points
            print(f"  Rank {rank}: {vlm} (HM={score:.6f}) → +{borda_points} Borda points")
        print()

    # Step 4: Compute MeanJudge score and std per VLM
    mean_judge_scores = {
        vlm: np.mean(hm_scores[vlm])
        for vlm in vlm_names
    }

    mean_judge_std = {
        vlm: np.std(hm_scores[vlm], ddof=1)  # Sample standard deviation
        for vlm in vlm_names
    }

    return borda_scores, mean_judge_scores, mean_judge_std


def generate_overall_vlm_ranking_table(borda_scores: Dict[str, int],
                                       mean_judge_scores: Dict[str, float],
                                       mean_judge_std: Dict[str, float]) -> pd.DataFrame:
    """
    Generate overall VLM ranking table.

    Returns:
        DataFrame with columns: [Rank, VLM, Borda, MeanJudge, StdJudge]
    """
    # Create list of (vlm, borda, mean_judge, std_judge)
    data = []
    for vlm in borda_scores.keys():
        data.append({
            'VLM': vlm,
            'Borda': borda_scores[vlm],
            'MeanJudge': round(mean_judge_scores[vlm], 2),
            'StdJudge': round(mean_judge_std[vlm], 2)
        })

    # Sort by Borda (primary), then MeanJudge (secondary)
    df = pd.DataFrame(data)
    df = df.sort_values(by=['Borda', 'MeanJudge'], ascending=[False, False])
    df.insert(0, 'Rank', range(1, len(df) + 1))

    return df


def generate_per_judge_vlm_metrics_table(results: Dict) -> pd.DataFrame:
    """
    Generate per-judge VLM metrics table (vlm_per_judge_metrics.csv).

    Creates a hierarchical column structure with:
    - Top level: Embedding model names
    - Second level: Coarse, Fine F1, HM (formatted as avg±std)

    Returns:
        DataFrame with MultiIndex columns
    """
    embedding_models = sorted(results.keys())
    vlm_names = sorted(list(results[embedding_models[0]].keys()))

    # Build data dictionary
    data = []
    for vlm in vlm_names:
        row = {'VLM': vlm}
        for embed in embedding_models:
            stats = results[embed][vlm]
            
            # Format as avg±std
            coarse = f"{stats['coarse']:.2f}±{stats['coarse_std']:.2f}"
            fine_f1 = f"{stats['fine']:.2f}±{stats['fine_std']:.2f}"
            hm = f"{stats['hm']:.2f}±{stats['hm_std']:.2f}"
            
            row[f'{embed}_coarse_grained'] = coarse
            row[f'{embed}_fine_grained_f1'] = fine_f1
            row[f'{embed}_harmonic_mean_cf'] = hm
            
        data.append(row)

    df = pd.DataFrame(data)

    # Create MultiIndex columns
    columns = [('', 'vlm')]
    metrics = ['coarse_grained', 'fine_grained_f1', 'harmonic_mean_cf']
    
    for embed in embedding_models:
        for metric in metrics:
            columns.append((embed, metric))

    # Reorder dataframe columns to match the MultiIndex structure
    ordered_data = {'VLM': df['VLM']}
    for embed in embedding_models:
        for metric in metrics:
            ordered_data[f'{embed}_{metric}'] = df[f'{embed}_{metric}']

    df_ordered = pd.DataFrame(ordered_data)
    df_ordered.columns = pd.MultiIndex.from_tuples(columns)

    return df_ordered


def save_results(overall_vlm_ranking: pd.DataFrame, per_judge_vlm_metrics: pd.DataFrame, output_dir: Path = None):
    """
    Save the ranking tables to CSV files.
    """
    if output_dir is None:
        output_dir = DEFAULT_RESULTS_DIR / "ranking"
        
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save overall VLM ranking table
    overall_ranking_file = output_path / "vlm_overall_ranking.csv"
    overall_vlm_ranking.to_csv(overall_ranking_file, index=False)
    print(f"Overall VLM ranking table saved to: {overall_ranking_file}")

    # Save per-judge VLM metrics table
    per_judge_metrics_file = output_path / "vlm_per_judge_metrics.csv"
    per_judge_vlm_metrics.to_csv(per_judge_metrics_file)
    print(f"Per-judge VLM metrics table saved to: {per_judge_metrics_file}")




def print_summary(overall_vlm_ranking: pd.DataFrame):
    """
    Print a summary of the ranking results.
    """
    print("=" * 80)
    print("OVERALL VLM RANKING SUMMARY")
    print("=" * 80)
    print(overall_vlm_ranking.to_string(index=False))
    print()

    print("=" * 80)
    print("TOP 5 VLMs:")
    print("=" * 80)
    top5 = overall_vlm_ranking.head(5)
    for idx, row in top5.iterrows():
        print(f"{row['Rank']}. {row['VLM']}")
        print(f"   Borda Score: {row['Borda']}")
        print(f"   MeanJudge: {row['MeanJudge']:.2f} (±{row['StdJudge']:.2f})")
        print()


def main():
    """
    Main execution function.
    """
    print("=" * 80)
    print("VLM RANKING ALGORITHM")
    print("Based on Borda Count and Mean Judge Score")
    print("=" * 80)
    print()

    # Step 1: Load evaluation results
    print("Step 1: Loading evaluation results...")
    results = load_evaluation_results()
    print(f"Loaded results for {len(results)} embedding models")
    print()

    # Step 2-4: Compute Borda scores and MeanJudge
    print("Step 2-4: Computing Borda scores and MeanJudge scores...")
    borda_scores, mean_judge_scores, mean_judge_std = compute_borda_scores(results)
    print()

    # Print Borda scores summary
    print("=" * 80)
    print("BORDA SCORES SUMMARY")
    print("=" * 80)
    for vlm in sorted(borda_scores.keys(), key=lambda x: borda_scores[x], reverse=True):
        print(f"{vlm}: {borda_scores[vlm]} (MeanJudge: {mean_judge_scores[vlm]:.2f} ±{mean_judge_std[vlm]:.2f})")
    print()

    # Step 5: Generate tables
    print("Step 5: Generating ranking tables...")
    overall_vlm_ranking = generate_overall_vlm_ranking_table(borda_scores, mean_judge_scores, mean_judge_std)
    per_judge_vlm_metrics = generate_per_judge_vlm_metrics_table(results)
    print()

    # Print summary
    print_summary(overall_vlm_ranking)

    # Save results
    print("Step 6: Saving results...")
    save_results(overall_vlm_ranking, per_judge_vlm_metrics)
    print()

    print("=" * 80)
    print("ALGORITHM COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
