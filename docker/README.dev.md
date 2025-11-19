# Development Environment

This directory contains files for setting up a development environment with Docker.

## Quick Start

### Option 1: Using the startup script (Recommended)

```bash
bash docker/dev-start.sh
```

### Option 2: Manual setup

```bash
# Build base image (if not already built)
docker build -t leo-fuzzy-fairness:latest -f docker/Dockerfile .

# Build dev image
docker build -t leo-fuzzy-fairness:dev -f docker/Dockerfile.dev .

# Start development container
cd docker
docker-compose -f docker-compose.dev.yaml up -d
```

## Accessing the Development Environment

### Enter the container

```bash
docker exec -it fairness-dev bash
```

### Start Jupyter Lab

Inside the container:
```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Or use the entrypoint:
```bash
docker exec -it fairness-dev /usr/local/bin/entrypoint.sh jupyter
```

Access at: http://localhost:8888

### Start Jupyter Notebook

```bash
docker exec -it fairness-dev /usr/local/bin/entrypoint.sh notebook
```

Access at: http://localhost:8889

## Volume Mounts

The following directories are mounted for live editing:

- `src/` → `/workspace/src` - Source code
- `experiments/` → `/workspace/experiments` - Experiment configurations
- `data/` → `/workspace/data` - Data files
- `results/` → `/workspace/results` - Results output
- `notebooks/` → `/workspace/notebooks` - Jupyter notebooks
- `tests/` → `/workspace/tests` - Test files

**Changes made in these directories are immediately reflected in the container!**

## Useful Commands

### View logs
```bash
docker-compose -f docker/docker-compose.dev.yaml logs -f
```

### Stop container
```bash
docker-compose -f docker/docker-compose.dev.yaml stop
```

### Start container
```bash
docker-compose -f docker/docker-compose.dev.yaml start
```

### Restart container
```bash
docker-compose -f docker/docker-compose.dev.yaml restart
```

### Stop and remove
```bash
docker-compose -f docker/docker-compose.dev.yaml down
```

### Rebuild and restart
```bash
docker-compose -f docker/docker-compose.dev.yaml up -d --build
```

## Development Tools Included

- **Jupyter Lab**: Interactive development environment
- **Jupyter Notebook**: Classic notebook interface
- **vim/nano**: Text editors
- **htop**: Process monitor
- **git**: Version control
- **ipython**: Enhanced Python shell

## Testing

Run tests inside the container:

```bash
docker exec -it fairness-dev bash
cd /workspace
python3 -m pytest tests/ -v
```

## GPU Access

The container has full access to all GPUs. Verify with:

```bash
docker exec -it fairness-dev nvidia-smi
docker exec -it fairness-dev python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Troubleshooting

### Container won't start
```bash
docker-compose -f docker/docker-compose.dev.yaml logs
```

### Rebuild everything
```bash
docker-compose -f docker/docker-compose.dev.yaml down
docker build -t leo-fuzzy-fairness:latest -f docker/Dockerfile .
docker build -t leo-fuzzy-fairness:dev -f docker/Dockerfile.dev .
docker-compose -f docker/docker-compose.dev.yaml up -d
```

### Clear volumes (if needed)
```bash
docker-compose -f docker/docker-compose.dev.yaml down -v
```

## Ports

- **8888**: Jupyter Lab
- **8889**: Jupyter Notebook
- **6006**: TensorBoard (if you add it)

## Notes

- Source code changes are live - no need to rebuild
- Jupyter notebooks are saved in `notebooks/` directory
- Results are saved in `results/` directory
- All data persists between container restarts

