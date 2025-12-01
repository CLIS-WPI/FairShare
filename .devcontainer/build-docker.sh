#!/bin/bash
# Build Docker image for dev container
# Run this BEFORE opening the dev container to avoid certificate issues

set -e

echo "========================================="
echo "Building Docker Image for Dev Container"
echo "========================================="
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ ERROR: Docker is not running"
    echo "   Please start Docker and try again"
    exit 1
fi

# Build the image
echo "Building image: fuzzy-fairness-dss:dev"
echo "Dockerfile: docker/Dockerfile.final"
echo ""

docker build -f docker/Dockerfile.final -t fuzzy-fairness-dss:dev .

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ Build successful!"
    echo "========================================="
    echo ""
    echo "Image built: fuzzy-fairness-dss:dev"
    echo ""
    echo "Now you can:"
    echo "  1. Open dev container in VS Code"
    echo "     F1 → 'Dev Containers: Reopen in Container'"
    echo ""
    echo "  2. Or use the image directly:"
    echo "     docker run --gpus all -it \\"
    echo "       -v \$(pwd):/workspace \\"
    echo "       fuzzy-fairness-dss:dev \\"
    echo "       /bin/bash"
else
    echo ""
    echo "========================================="
    echo "❌ Build failed!"
    echo "========================================="
    echo ""
    echo "Common issues:"
    echo "  1. Certificate error: Run 'sudo bash .devcontainer/fix-docker-certificates.sh'"
    echo "  2. Network issues: Check internet connection"
    echo "  3. NVIDIA registry access: May need to login"
    echo ""
    echo "See .devcontainer/CERTIFICATE_FIX.md for troubleshooting"
    exit 1
fi

