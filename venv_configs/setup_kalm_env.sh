#!/bin/bash

# KaLM CLIP-CC-Bench Environment Setup Script (Centralized)
# Creates isolated virtual environment in centralized location

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${SCRIPT_DIR}/../venv_kalm"
REQUIREMENTS_FILE="${SCRIPT_DIR}/kalm-requirements.txt"

echo "🔧 Setting up KaLM CLIP-CC-Bench environment..."

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

# Install base dependencies first (without flash-attn)
echo "📚 Installing base dependencies..."
pip install torch>=2.0.0 transformers>=4.40.0 tokenizers>=0.15.0 sentence-transformers>=2.7.0 accelerate>=0.20.0 numpy>=1.24.0 pandas>=2.0.0 datasets>=4.0.0 scikit-learn scipy matplotlib seaborn Pillow PyYAML>=6.0 tqdm psutil nltk>=3.8.1 huggingface-hub>=0.20.0 black flake8 isort mypy

# Install flash-attn separately (optional - will use eager attention as fallback if this fails)
echo "📚 Installing flash-attn (optional, may take a while)..."
pip install flash-attn>=2.5.6 || echo "⚠️  Flash-attention installation failed, will use eager attention as fallback"

# Verify installation
echo "🔍 Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import nltk; print(f'NLTK: {nltk.__version__}')"
python -c "import sentence_transformers; print(f'Sentence-Transformers: {sentence_transformers.__version__}')"
python -c "print('✅ KaLM CLIP-CC-Bench dependencies installed successfully')"

echo ""
echo "🎉 KaLM CLIP-CC-Bench environment setup completed!"
echo ""
echo "To activate the environment:"
echo "  source $VENV_PATH/bin/activate"
echo ""
echo "To deactivate:"
echo "  deactivate"
echo ""
