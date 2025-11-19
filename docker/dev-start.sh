#!/bin/bash
# Development container startup script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "========================================="
echo "Starting Development Environment"
echo "========================================="

# Check if base image exists
if ! docker images | grep -q "leo-fuzzy-fairness.*latest"; then
    echo "Base image not found. Building base image..."
    docker build -t leo-fuzzy-fairness:latest -f docker/Dockerfile .
fi

# Build dev image
echo "Building development image..."
docker build -t leo-fuzzy-fairness:dev -f docker/Dockerfile.dev .

# Start with docker-compose
cd docker
echo "Starting development container..."
docker-compose -f docker-compose.dev.yaml up -d --build

echo ""
echo "========================================="
echo "Development container started!"
echo "========================================="
echo ""
echo "Available services:"
echo "  - Container: fairness-dev"
echo "  - Jupyter Lab: http://localhost:8888"
echo "  - Jupyter Notebook: http://localhost:8889"
echo ""
echo "Commands:"
echo "  docker exec -it fairness-dev bash          # Enter container"
echo "  docker-compose -f docker/docker-compose.dev.yaml logs -f  # View logs"
echo "  docker-compose -f docker/docker-compose.dev.yaml stop     # Stop container"
echo "  docker-compose -f docker/docker-compose.dev.yaml down     # Stop and remove"
echo ""

