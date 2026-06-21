#!/bin/bash

# Nemo (Nemotron) CLIP-CC-Bench Environment Activation Script
# Activates the centralized Nemo virtual environment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${SCRIPT_DIR}/../venv_nemo"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found: $VENV_PATH"
    echo "Please run setup_nemo_env.sh first to create it."
    return 1 2>/dev/null || exit 1
fi

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Verify activation
if [ "$VIRTUAL_ENV" = "" ]; then
    echo "❌ Failed to activate virtual environment"
    return 1 2>/dev/null || exit 1
fi

echo "✅ Nemo (Nemotron) CLIP-CC-Bench environment activated"
echo "📁 Location: $VIRTUAL_ENV"
echo ""
echo "To deactivate, run: deactivate"
