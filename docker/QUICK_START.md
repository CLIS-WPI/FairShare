# Docker Quick Start Guide

## Build Image

```bash
# From project root
make docker-build

# Or directly
docker build -f docker/Dockerfile.final -t fuzzy-fairness-dss:latest .
```

## Run Container

### Interactive Mode (Recommended for Development)

```bash
docker run --gpus all -it \
  -v $(pwd):/workspace \
  --name fuzzy-dss-dev \
  fuzzy-fairness-dss:latest \
  /bin/bash
```

### Run Simulation Directly

```bash
docker run --gpus all --rm \
  -v $(pwd)/results:/workspace/results \
  fuzzy-fairness-dss:latest \
  python -m src.main --scenario urban_congestion_phase4 --policy fuzzy --duration-s 30
```

### With Jupyter Lab

```bash
docker run --gpus all -it \
  -v $(pwd):/workspace \
  -p 8888:8888 \
  --name fuzzy-dss-jupyter \
  fuzzy-fairness-dss:latest \
  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Then open: `http://localhost:8888`

## Docker Compose

```bash
# From docker/ directory
docker compose -f compose.yaml up -d

# Or from project root
docker compose -f docker/compose.yaml up -d
```

## Common Commands

```bash
# List images
docker images | grep fuzzy-fairness

# Remove old container
docker rm -f fuzzy-dss-dev

# View logs
docker logs fuzzy-dss-dev

# Execute command in running container
docker exec -it fuzzy-dss-dev bash

# Stop container
docker stop fuzzy-dss-dev
```

## Troubleshooting

### Image not found
Make sure you use the correct image name: `fuzzy-fairness-dss:latest`

### GPU not available
```bash
# Check GPU
nvidia-smi

# Run without GPU (CPU mode)
docker run -it fuzzy-fairness-dss:latest /bin/bash
```

### Permission denied
```bash
# Run with user permissions
docker run --gpus all -it --user $(id -u):$(id -g) \
  -v $(pwd):/workspace \
  fuzzy-fairness-dss:latest /bin/bash
```

