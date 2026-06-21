#!/bin/bash

# GTE CLIP-CC-Bench Environment Setup Script (Centralized)
# Creates isolated virtual environment in centralized location

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${SCRIPT_DIR}/../venv_gte"
REQUIREMENTS_FILE="${SCRIPT_DIR}/gte-requirements.txt"

echo "🔧 Setting up GTE CLIP-CC-Bench environment..."

# Check if requirements.txt exists
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "❌ Requirements file not found: $REQUIREMENTS_FILE"
    exit 1
fi

# Remove existing virtual environment if it exists
if [ -d "$VENV_PATH" ]; then
    echo "🗑️  Removing existing virtual environment..."
    rm -rf "$VENV_PATH"
fi

# Create new virtual environment
echo "📦 Creating new virtual environment..."
python3 -m venv "$VENV_PATH"

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Verify activation
if [ "$VIRTUAL_ENV" = "" ]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

echo "✅ Virtual environment created and activated"
echo "📁 Location: $VIRTUAL_ENV"

# Upgrade pip, setuptools, and wheel first
echo "⬆️  Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

# Install from frozen requirements (exact versions)
echo "📚 Installing dependencies from frozen requirements..."
pip install -r "$REQUIREMENTS_FILE"

# Verify installation
echo "🔍 Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import nltk; print(f'NLTK: {nltk.__version__}')"
python -c "print('✅ GTE CLIP-CC-Bench dependencies installed successfully')"

echo ""
echo "🎉 GTE CLIP-CC-Bench environment setup completed!"
echo ""
echo "To activate the environment:"
echo "  source $VENV_PATH/bin/activate"
echo ""
echo "To deactivate:"
echo "  deactivate"
echo ""
