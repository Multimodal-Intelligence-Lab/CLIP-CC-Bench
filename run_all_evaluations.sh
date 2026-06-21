#!/bin/bash

# CLIP-CC-Bench: Run All Embedding Model Evaluations
# This script runs all 5 embedding model evaluations sequentially

set -e  # Exit on error

# Get the directory where this script is located and save it
# NOTE: Activation scripts will overwrite SCRIPT_DIR, so we need to preserve it
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_CONFIGS_DIR="${SCRIPT_DIR}/venv_configs"

# Save the original script directory (activation scripts will corrupt SCRIPT_DIR)
readonly ORIGINAL_SCRIPT_DIR="${SCRIPT_DIR}"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "CLIP-CC-Bench: All Embedding Model Evaluations"
echo "========================================"
echo ""

# Function to run evaluation
run_evaluation() {
    local model_name=$1
    local activate_script=$2
    local eval_script=$3
    local config_file=$4

    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${model_name} evaluation...${NC}"
    echo "----------------------------------------"

    # Paths are already absolute (using ${SCRIPT_DIR}), just store them in local vars
    # to protect against variable corruption from activation script
    local EVAL_SCRIPT="${eval_script}"
    local CONFIG_FILE="${config_file}"

    # Source the activation script (may overwrite variables and change directory)
    source "${activate_script}"

    # Restore our script directory and change to it
    cd "${ORIGINAL_SCRIPT_DIR}"

    # Run the evaluation with the saved paths
    python "${EVAL_SCRIPT}" --config "${CONFIG_FILE}"

    # Deactivate the environment
    deactivate

    # CRITICAL: Restore SCRIPT_DIR for next evaluation call
    # (The activation script overwrites it, breaking subsequent ${SCRIPT_DIR} expansions)
    SCRIPT_DIR="${ORIGINAL_SCRIPT_DIR}"

    # Return to script directory for next iteration
    cd "${ORIGINAL_SCRIPT_DIR}"

    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ${model_name} evaluation completed!${NC}"
    echo ""
}

# Track start time
START_TIME=$(date +%s)

# 1. NV-Embed Evaluation
run_evaluation \
    "NV-Embed-v2" \
    "${VENV_CONFIGS_DIR}/activate_nv-embed_env.sh" \
    "${SCRIPT_DIR}/src/scripts/run_nv_embed_evaluation.py" \
    "${SCRIPT_DIR}/src/configs/nv-embed.yaml"

# 2. GTE Evaluation
run_evaluation \
    "gte-Qwen2-7B-instruct" \
    "${VENV_CONFIGS_DIR}/activate_gte_env.sh" \
    "${SCRIPT_DIR}/src/scripts/run_gte_evaluation.py" \
    "${SCRIPT_DIR}/src/configs/gte.yaml"

# 3. Nemo Evaluation
run_evaluation \
    "llama-embed-nemotron-8b" \
    "${VENV_CONFIGS_DIR}/activate_nemo_env.sh" \
    "${SCRIPT_DIR}/src/scripts/run_nemo_evaluation.py" \
    "${SCRIPT_DIR}/src/configs/nemo.yaml"

# 4. Qwen Evaluation
run_evaluation \
    "Qwen3-Embedding-8B" \
    "${VENV_CONFIGS_DIR}/activate_qwen_env.sh" \
    "${SCRIPT_DIR}/src/scripts/run_qwen_evaluation.py" \
    "${SCRIPT_DIR}/src/configs/qwen.yaml"

# 5. KaLM Evaluation
run_evaluation \
    "KaLM-Embedding-Gemma3-12B-2511" \
    "${VENV_CONFIGS_DIR}/activate_kalm_env.sh" \
    "${SCRIPT_DIR}/src/scripts/run_kalm_evaluation.py" \
    "${SCRIPT_DIR}/src/configs/kalm.yaml"

# 6. Run VLM Ranking
echo "========================================"
echo "Running VLM Ranking Algorithm..."
echo "========================================"
python "${SCRIPT_DIR}/src/scripts/rank_vlms.py"
echo -e "${GREEN}Ranking completed!${NC}"
echo ""

# Calculate total time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))
SECONDS=$((TOTAL_TIME % 60))

echo "========================================"
echo -e "${GREEN}All evaluations and ranking completed successfully!${NC}"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "========================================"
echo ""
echo "Results are stored in:"
echo "  - results/embedding_models/logs/NV-Embed-v2/"
echo "  - results/embedding_models/logs/gte-Qwen2-7B-instruct/"
echo "  - results/embedding_models/logs/llama-embed-nemotron-8b/"
echo "  - results/embedding_models/logs/Qwen3-Embedding-8B/"
echo "  - results/embedding_models/logs/KaLM-Embedding-Gemma3-12B-2511/"
echo "  - results/ranking/vlm_overall_ranking.csv"
echo "  - results/ranking/vlm_per_judge_metrics.csv"
