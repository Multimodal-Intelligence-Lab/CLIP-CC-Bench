#!/bin/bash

# Cleanup Script - Removes old virtual environments from src/config and src/scripts
# Run this ONLY after verifying that new centralized venvs work!

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIP_CC_BENCH="${SCRIPT_DIR}/.."

echo "=========================================="
echo "Old Virtual Environment Cleanup"
echo "=========================================="
echo ""
echo "⚠️  WARNING: This will DELETE the following directories:"
echo ""
echo "From src/config/:"
echo "  - $CLIP_CC_BENCH/src/config/bge-icl/venv (~9.0 GB)"
echo "  - $CLIP_CC_BENCH/src/config/e5-mistral-7b-instruct/venv (~7.6 GB)"
echo "  - $CLIP_CC_BENCH/src/config/envision/venv (~6.7 GB)"
echo "  - $CLIP_CC_BENCH/src/config/nv-embed/venv (~7.3 GB)"
echo "  - $CLIP_CC_BENCH/src/config/stella-en-1.5b-v5/venv (~7.2 GB)"
echo ""
echo "From src/scripts/:"
echo "  - $CLIP_CC_BENCH/src/scripts/jina-v4/venv (~9.2 GB)"
echo "  - $CLIP_CC_BENCH/src/scripts/nomic-embed-text-v1.5/venv (~9.8 GB)"
echo ""
echo "Total space to reclaim: ~56.8 GB"
echo ""
echo "❗ Make sure you have:"
echo "  1. Created all new centralized venvs"
echo "  2. Tested them to ensure they work"
echo "  3. Updated any scripts that reference old venv paths"
echo ""

# Ask for confirmation
read -p "Have you verified that new venvs work? (yes/no) " -r
echo ""
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "❌ Cleanup cancelled - verify new venvs first!"
    exit 1
fi

echo "Last chance! This action cannot be undone."
read -p "Type 'DELETE' to confirm: " -r
echo ""
if [[ $REPLY != "DELETE" ]]; then
    echo "❌ Cleanup cancelled"
    exit 1
fi

echo "🗑️  Starting cleanup..."
echo ""

# Function to remove a venv directory
remove_venv() {
    local path=$1
    local name=$2

    if [ -d "$path" ]; then
        echo "Removing $name..."
        du -sh "$path" 2>/dev/null
        rm -rf "$path"
        if [ ! -d "$path" ]; then
            echo "✅ $name removed"
        else
            echo "❌ Failed to remove $name"
        fi
    else
        echo "⏭️  $name not found (already removed?)"
    fi
    echo ""
}

# Remove config venvs
echo "Cleaning up src/config venvs..."
remove_venv "$CLIP_CC_BENCH/src/config/bge-icl/venv" "bge-icl"
remove_venv "$CLIP_CC_BENCH/src/config/e5-mistral-7b-instruct/venv" "e5-mistral-7b-instruct"
remove_venv "$CLIP_CC_BENCH/src/config/envision/venv" "envision"
remove_venv "$CLIP_CC_BENCH/src/config/nv-embed/venv" "nv-embed"
remove_venv "$CLIP_CC_BENCH/src/config/stella-en-1.5b-v5/venv" "stella-en-1.5b-v5"

# Remove scripts venvs
echo "Cleaning up src/scripts venvs..."
remove_venv "$CLIP_CC_BENCH/src/scripts/jina-v4/venv" "jina-v4"
remove_venv "$CLIP_CC_BENCH/src/scripts/nomic-embed-text-v1.5/venv" "nomic-embed-text-v1.5"

echo "=========================================="
echo "Cleanup Complete!"
echo "=========================================="
echo ""
echo "Check disk space reclaimed:"
echo "  df -h <project_root>/"
echo ""
echo "New venvs are located at:"
echo "  <project_root>/venv_*"
echo ""
