#!/bin/bash
# ============================================================================
# Setup autoresearch-kazakh on a vast.ai A100 80GB instance
#
# Usage:
#   1. Create a vast.ai instance: 1x A100 80GB, pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel
#   2. SSH into the instance
#   3. Run this script: bash setup_vastai.sh
#   4. Then run Claude Code: claude --print program.md
# ============================================================================
set -e

echo "=== Setting up autoresearch-kazakh ==="

# Install uv
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Go to autoresearch directory
cd /root/autoresearch

# Install dependencies
echo "Installing Python dependencies..."
uv sync

# Prepare data (download from HuggingFace)
echo "Downloading Kazakh pretokenized data..."
uv run prepare.py

# Initialize git repo for experiment tracking
if [ ! -d ".git" ]; then
    git init
    git add -A
    git commit -m "initial: autoresearch-kazakh setup"
fi

# Create experiment branch
BRANCH="autoresearch/kazakh-exp019"
git checkout -b "$BRANCH" 2>/dev/null || git checkout "$BRANCH"

# Initialize results.tsv
if [ ! -f "results.tsv" ]; then
    printf "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n" > results.tsv
fi

echo ""
echo "=== Setup complete ==="
echo "Branch: $BRANCH"
echo "Data: ~/.cache/autoresearch-kazakh/data/"
echo ""
echo "Quick test (manual run):"
echo "  uv run train.py"
echo ""
echo "Start autonomous research with Claude Code:"
echo "  claude --print program.md"
echo ""
echo "Or run a 1-hour session (12 × 5-min slots):"
echo "  claude 'Read program.md and run 12 experiments (1 hour total). Start with baseline, then optimize val_bpb systematically.'"
