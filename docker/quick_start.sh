#!/bin/bash
# Build and run in one command

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "========================================="
echo "Building Docker image..."
echo "========================================="
docker build -t leo-fuzzy-fairness -f docker/Dockerfile .

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Build successful! Starting container..."
    echo "========================================="
    docker run --gpus all -it --name fairness-dev \
      -v "$PROJECT_ROOT/data:/workspace/data" \
      -v "$PROJECT_ROOT/experiments:/workspace/experiments" \
      -v "$PROJECT_ROOT/results:/workspace/results" \
      leo-fuzzy-fairness bash
else
    echo ""
    echo "========================================="
    echo "Build failed. Check errors above."
    echo "========================================="
    exit 1
fi

