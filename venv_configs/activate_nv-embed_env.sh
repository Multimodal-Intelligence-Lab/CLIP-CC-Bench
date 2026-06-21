#!/bin/bash

# NV-Embed Environment Activation Script (Centralized)
# Activates the centralized NV-Embed virtual environment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${SCRIPT_DIR}/../venv_nv-embed"

echo "⚡ Activating NV-Embed environment..."

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Virtual environment not found at: $VENV_PATH"
    echo "   Please run setup_nv-embed_env.sh first"
    exit 1
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Verify activation
if [ "$VIRTUAL_ENV" != "" ]; then
    echo "✅ NV-Embed environment activated"
    echo "📁 Virtual environment: $VIRTUAL_ENV"
    echo "Python: $(python --version 2>&1)"
    echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
    echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Unknown')"
    echo ""
    echo "🚀 Ready to run NV-Embed evaluation!"
    echo "   Deactivate: deactivate"
else
    echo "❌ Failed to activate virtual environment"
    exit 1
fi
