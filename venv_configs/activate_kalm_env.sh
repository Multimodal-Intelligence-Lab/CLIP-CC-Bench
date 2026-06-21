#!/bin/bash

# KaLM CLIP-CC-Bench Environment Activation Script
# Activates the centralized KaLM virtual environment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${SCRIPT_DIR}/../venv_kalm"

if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found at: $VENV_PATH"
    echo "Please run setup_kalm_env.sh first to create the environment."
    exit 1
fi

echo "🔧 Activating KaLM CLIP-CC-Bench environment..."
source "$VENV_PATH/bin/activate"

if [ "$VIRTUAL_ENV" = "" ]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

echo "✅ KaLM environment activated"
echo "📁 Location: $VIRTUAL_ENV"
echo ""
