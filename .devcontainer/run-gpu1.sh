#!/bin/bash
# Run Docker container with only GPU1

echo "========================================="
echo "Running Container with GPU1 Only"
echo "========================================="
echo ""

# Check if image exists
if ! docker images | grep -q "fuzzy-fairness-dss.*dev"; then
    echo "❌ Image not found. Building first..."
    docker build -f docker/Dockerfile.final -t fuzzy-fairness-dss:dev .
fi

echo "Starting container with GPU1..."
echo ""

docker run --gpus device=1 -it \
  -v $(pwd):/workspace \
  -p 8888:8888 \
  --name fuzzy-dss-gpu1 \
  fuzzy-fairness-dss:dev \
  /bin/bash

