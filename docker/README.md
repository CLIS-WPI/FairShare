# Docker Development Environment

Complete guide for using Docker in this project.

## 🚀 Quick Start

```bash
bash docker/dev-start.sh
```

This command will:
- Build the development image (if needed)
- Start the container
- Show useful commands

## 📋 Docker Files

- **`Dockerfile.dev`** - Development image (with Jupyter, dev tools)
- **`Dockerfile.final`** - Production image (for devcontainer)
- **`docker-compose.dev.yaml`** - Docker Compose configuration
- **`dev-start.sh`** - Script to start the container

## 🔧 Useful Commands

### Enter Container
```bash
docker exec -it fairness-dev bash
```

### Run Tests in Docker
```bash
bash run_verification_in_docker.sh
```

### View Logs
```bash
cd docker
docker compose -f docker-compose.dev.yaml logs -f
```

### Stop Container
```bash
cd docker
docker compose -f docker-compose.dev.yaml stop
```

### Stop and Remove
```bash
cd docker
docker compose -f docker-compose.dev.yaml down
```

## 📁 Volume Mounts

These directories are mounted to the container:
- `src/` → `/workspace/src`
- `experiments/` → `/workspace/experiments`
- `data/` → `/workspace/data`
- `results/` → `/workspace/results`
- `notebooks/` → `/workspace/notebooks`
- `tests/` → `/workspace/tests`
- `verify_framework.py` → `/workspace/verify_framework.py`

**Changes are immediately reflected!**

## 🎯 Services

- **Container:** `fairness-dev`
- **Jupyter Lab:** http://localhost:8888
- **Jupyter Notebook:** http://localhost:8889
- **TensorBoard:** http://localhost:6006

## ⚠️ Important Notes

1. **Always run in Docker** - not on Linux host
2. Container must be running
3. GPU must be available (`--gpus all`)
4. Volume mounts ensure changes are immediately reflected

---

**For more information:** See `README.md` in the root directory
