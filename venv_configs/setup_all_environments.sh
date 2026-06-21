#!/bin/bash

# Master Setup Script - Creates all encoder/CLIP-CC-Bench model environments
# This will take 1-3 hours depending on network speed

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Virtual Environment Setup"
echo "=========================================="
echo ""
echo "This script will create 5 virtual environments:"
echo "  1. NV-Embed (~7 GB) - CLIP-CC-Bench encoder"
echo "  2. Qwen (~7 GB) - CLIP-CC-Bench with Qwen"
echo "  3. Nemo (~7 GB) - CLIP-CC-Bench with Nemotron"
echo "  4. GTE (~7 GB) - CLIP-CC-Bench with GTE"
echo "  5. KaLM (~7 GB) - CLIP-CC-Bench with KaLM"
echo ""
echo "Estimated time: 1-3 hours"
echo ""
echo "Locations:"
echo "  <project_root>/venv_nv-embed"
echo "  <project_root>/venv_qwen"
echo "  <project_root>/venv_nemo"
echo "  <project_root>/venv_gte"
echo "  <project_root>/venv_kalm"
echo ""

# Ask for confirmation
read -p "Do you want to continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Setup cancelled"
    exit 1
fi

# Start time
START_TIME=$(date +%s)

# Setup each environment
MODELS=("nv-embed" "qwen" "nemo" "gte" "kalm")
TOTAL=${#MODELS[@]}
CURRENT=0
FAILED=()

for model in "${MODELS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "=========================================="
    echo "[$CURRENT/$TOTAL] Setting up: $model"
    echo "=========================================="

    if "$SCRIPT_DIR/setup_${model}_env.sh"; then
        echo "✅ $model setup completed successfully"
    else
        echo "❌ $model setup failed"
        FAILED+=("$model")
    fi
done

# End time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo "Time taken: ${MINUTES}m ${SECONDS}s"
echo ""

if [ ${#FAILED[@]} -eq 0 ]; then
    echo "✅ All environments created successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Test each environment (see README.md)"
    echo "  2. Activate environments as needed for your tasks"
else
    echo "⚠️  Some environments failed to create:"
    for model in "${FAILED[@]}"; do
        echo "   - $model"
    done
    echo ""
    echo "Please check the error messages above and retry failed setups manually"
fi

echo ""
echo "To activate an environment:"
echo "  NV-Embed: source venv_configs/activate_nv-embed_env.sh"
echo "  Qwen:     source venv_configs/activate_qwen_env.sh"
echo "  Nemo:     source venv_configs/activate_nemo_env.sh"
echo "  GTE:      source venv_configs/activate_gte_env.sh"
echo "  KaLM:     source venv_configs/activate_kalm_env.sh"
echo ""
