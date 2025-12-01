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

# Start with docker compose (modern Docker Compose V2)
cd docker
echo "Starting development container..."

# Determine which compose command to use (try V2 first, fallback to V1)
if docker compose version &> /dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

# Remove existing container if it exists (stopped or running)
if docker ps -a --format '{{.Names}}' | grep -q '^fairness-dev$'; then
    echo "Removing existing fairness-dev container..."
    $COMPOSE_CMD -f docker-compose.dev.yaml down 2>/dev/null || true
    docker rm -f fairness-dev 2>/dev/null || true
fi

# Start the container
echo "Starting new development container..."
$COMPOSE_CMD -f docker-compose.dev.yaml up -d --build

# Wait a moment for container to start
sleep 2

# Check if container is running
if docker ps --format '{{.Names}}' | grep -q '^fairness-dev$'; then
    echo ""
    echo "========================================="
    echo "✓ Development container started!"
    echo "========================================="
    echo ""
    echo "Available services:"
    echo "  - Container: fairness-dev"
    echo "  - Jupyter Lab: http://localhost:8888"
    echo "  - Jupyter Notebook: http://localhost:8889"
    echo "  - TensorBoard: http://localhost:6006"
    echo ""
    echo "Quick commands:"
    echo "  docker exec -it fairness-dev bash                    # Enter container"
    echo "  $COMPOSE_CMD -f docker/docker-compose.dev.yaml logs -f    # View logs"
    echo "  $COMPOSE_CMD -f docker/docker-compose.dev.yaml stop        # Stop container"
    echo "  $COMPOSE_CMD -f docker/docker-compose.dev.yaml down        # Stop and remove"
    echo ""
    echo "To start Jupyter Lab inside container:"
    echo "  docker exec -it fairness-dev jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
    echo ""
else
    echo ""
    echo "⚠ Warning: Container may not have started properly."
    echo "Check logs with: $COMPOSE_CMD -f docker/docker-compose.dev.yaml logs"
    echo ""
fi

