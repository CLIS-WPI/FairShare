# Build and Run Docker Container

## Prerequisites

- Docker installed and running
- NVIDIA Docker runtime (for GPU support)
- At least 20GB free disk space

## Build the Docker Image

### From Project Root

```bash
cd /home/tanglab/Desktop/fuzzy-fairness-dss-leo
docker build -t leo-fuzzy-fairness -f docker/Dockerfile .
```

### Build Process

The build will:
1. Pull NVIDIA TensorFlow base image (25.02-tf2-py3)
2. Install Python 3.12
3. Install Sionna 1.0+
4. Clone and install OpenNTN
5. Install all Python dependencies from requirements.txt

**Expected build time**: 10-30 minutes (depending on network speed)

## Run the Container

### Interactive Mode (Recommended for Development)

```bash
docker run --gpus all -it --name fairness-dev \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/experiments:/workspace/experiments \
  -v $(pwd)/results:/workspace/results \
  leo-fuzzy-fairness bash
```

This will:
- Mount data, experiments, and results directories
- Give you an interactive bash shell
- Enable GPU access

### Run Simulation Directly

```bash
docker run --gpus all --rm \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/experiments:/workspace/experiments \
  -v $(pwd)/results:/workspace/results \
  leo-fuzzy-fairness \
  python3.12 src/main.py --scenario experiments/scenarios/rural_coverage.yaml
```

### Using Docker Compose

```bash
cd docker
docker-compose up
```

## Verify Installation

Once inside the container:

```bash
# Check Python version
python3.12 --version

# Check Sionna
python3.12 -c "import sionna; print(f'Sionna: {sionna.__version__}')"

# Check OpenNTN
python3.12 -c "from sionna.channel import tr38811; print('OpenNTN: OK')"

# Check GPU
nvidia-smi

# Check TensorFlow GPU
python3.12 -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

## Common Commands

### Stop Container
```bash
docker stop fairness-dev
```

### Remove Container
```bash
docker rm fairness-dev
```

### View Logs
```bash
docker logs fairness-dev
```

### Execute Command in Running Container
```bash
docker exec -it fairness-dev bash
```

## Troubleshooting

### Certificate Issues

If you get certificate errors during build, the base image might not be accessible. In that case, you may need to:
1. Fix system certificates (see `TROUBLESHOOTING.md`)
2. Or use a cached base image if available

### GPU Not Detected

```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.5.0-base-ubuntu22.04 nvidia-smi

# If this fails, install nvidia-docker2
```

### Build Fails on OpenNTN Installation

The OpenNTN installation might fail if:
- Sionna is not properly installed first
- Network issues during git clone
- install.sh script fails

Check the build logs for specific errors.

## Quick Start Script

Create a file `docker/quick_start.sh`:

```bash
#!/bin/bash
# Build and run in one command

echo "Building Docker image..."
docker build -t leo-fuzzy-fairness -f docker/Dockerfile .

if [ $? -eq 0 ]; then
    echo "Build successful! Starting container..."
    docker run --gpus all -it --name fairness-dev \
      -v $(pwd)/data:/workspace/data \
      -v $(pwd)/experiments:/workspace/experiments \
      -v $(pwd)/results:/workspace/results \
      leo-fuzzy-fairness bash
else
    echo "Build failed. Check errors above."
fi
```

Make it executable and run:
```bash
chmod +x docker/quick_start.sh
./docker/quick_start.sh
```

