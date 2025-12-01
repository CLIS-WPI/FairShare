# Quick Start: Development Container

The easiest way to get started with the development environment.

## 🚀 One-Command Start

```bash
bash docker/dev-start.sh
```

That's it! The script will:
- Build the development image (if needed)
- Start the container
- Show you how to access it

## 📋 What You Get

- **Container name**: `fairness-dev`
- **Jupyter Lab**: http://localhost:8888
- **Jupyter Notebook**: http://localhost:8889
- **TensorBoard**: http://localhost:6006
- **Live code editing**: All changes in `src/`, `experiments/`, `notebooks/`, etc. are immediately available

## 🔧 Common Commands

### Enter the container
```bash
docker exec -it fairness-dev bash
```

### Start Jupyter Lab (from host)
```bash
docker exec -it fairness-dev jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```
Then open: http://localhost:8888

### View container logs
```bash
cd docker
docker compose -f docker-compose.dev.yaml logs -f
```

### Stop the container
```bash
cd docker
docker compose -f docker-compose.dev.yaml stop
```

### Start the container (if stopped)
```bash
cd docker
docker compose -f docker-compose.dev.yaml start
```

### Remove the container
```bash
cd docker
docker compose -f docker-compose.dev.yaml down
```

## ✅ Verify Everything Works

```bash
# Check container is running
docker ps | grep fairness-dev

# Test Python dependencies
docker exec fairness-dev python3 -c "import sionna; import tensorflow as tf; print('✓ All dependencies working!')"

# Check GPU access (if you have GPUs)
docker exec fairness-dev nvidia-smi
```

## 📁 Volume Mounts

These directories are automatically synced:
- `src/` → `/workspace/src`
- `experiments/` → `/workspace/experiments`
- `data/` → `/workspace/data`
- `results/` → `/workspace/results`
- `notebooks/` → `/workspace/notebooks`
- `tests/` → `/workspace/tests`

**Changes are live - no rebuild needed!**

## 🐛 Troubleshooting

### Container won't start
```bash
cd docker
docker compose -f docker-compose.dev.yaml logs
```

### Rebuild from scratch
```bash
cd docker
docker compose -f docker-compose.dev.yaml down
docker build -t leo-fuzzy-fairness:dev -f Dockerfile.dev ..
docker compose -f docker-compose.dev.yaml up -d
```

### Port already in use
If ports 8888, 8889, or 6006 are already in use, edit `docker/docker-compose.dev.yaml` and change the port mappings.

## 📚 More Information

See `docker/README.dev.md` for detailed documentation.

