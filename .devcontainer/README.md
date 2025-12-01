# Dev Container Configuration

## Overview

This dev container configuration provides a complete development environment for the Fuzzy-Fairness DSS LEO project with:

- **Python 3.12** (from NVIDIA TensorFlow base image)
- **TensorFlow 2.16+** with GPU support
- **Sionna 1.2.1** for channel modeling
- **OpenNTN** for NTN channel models
- **Jupyter Lab** for interactive development
- **All project dependencies** pre-installed

## Configuration Details

### Dockerfile
- **Path**: `docker/Dockerfile.final`
- **Base Image**: `nvcr.io/nvidia/tensorflow:24.12-tf2-py3`
- **Working Directory**: `/workspace`

### Features
- ✅ GPU support (`--gpus=all`)
- ✅ Workspace mounted at `/workspace`
- ✅ Python 3.12 interpreter
- ✅ Jupyter Lab on port 8888
- ✅ Git support
- ✅ VS Code extensions pre-installed

### VS Code Extensions
- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- YAML (redhat.vscode-yaml)
- Docker (ms-azuretools.vscode-docker)
- GitLens (eamodio.gitlens)
- Jupyter (ms-toolsai.jupyter)

### Environment Variables
- `PYTHONPATH=/workspace`
- `TZ=UTC`

### Post-Create Command
Automatically runs `python3 -m pip install -e .` to install the project in editable mode.

## Usage

### Opening in VS Code
1. Open VS Code
2. Install the "Dev Containers" extension
3. Press `F1` → "Dev Containers: Reopen in Container"
4. VS Code will build and start the container

### Manual Build
```bash
# Build the container
docker build -f docker/Dockerfile.final -t fuzzy-fairness-dss:dev .

# Run manually
docker run --gpus all -it \
  -v $(pwd):/workspace \
  -p 8888:8888 \
  fuzzy-fairness-dss:dev \
  /bin/bash
```

### Accessing Jupyter Lab
Once the container is running:
- Jupyter Lab will be available at `http://localhost:8888`
- Token will be shown in VS Code terminal

## Verification

After opening the container, verify installation:

```bash
# Check Python
python3 --version  # Should show Python 3.12.x

# Check TensorFlow
python3 -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"

# Check Sionna
python3 -c "import sionna; print(f'Sionna: {sionna.__version__}')"

# Check GPU
nvidia-smi

# Check project installation
python3 -c "import src; print('Project installed')"
```

## Troubleshooting

### Container won't start
- Check Docker is running: `docker ps`
- Check GPU support: `nvidia-smi`
- Check Dockerfile exists: `ls docker/Dockerfile.final`

### Python not found
- Verify Python path: `which python3`
- Check VS Code settings: Python interpreter should be `/usr/bin/python3`

### Package import errors
- Re-run post-create: `python3 -m pip install -e .`
- Check PYTHONPATH: `echo $PYTHONPATH` (should be `/workspace`)

### GPU not available
- Verify `--gpus=all` in runArgs
- Check NVIDIA Docker runtime: `docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi`

## Recent Fixes

### Fixed Issues (2024)
1. ✅ Changed `pip install -e .` to `python3 -m pip install -e .` for consistency
2. ✅ Added `workspaceFolder` configuration
3. ✅ Added `containerEnv` for PYTHONPATH and TZ
4. ✅ Verified JSON validity

## Notes

- The container runs as `root` user (default for NVIDIA TensorFlow images)
- All workspace files are mounted, so changes persist
- The container includes all dependencies from `requirements.txt`
- Jupyter Lab is pre-installed and ready to use

