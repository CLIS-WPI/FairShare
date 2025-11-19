# Phase 0 - Build & Run Docker Environment

## 🎯 Objective
Setup complete environment including TensorFlow + Sionna + OpenNTN + Nsight + Python stack

## Prerequisites

1. **NVIDIA Docker Runtime**: Ensure `nvidia-docker2` is installed
   ```bash
   # Check if nvidia-docker is available
   docker run --rm --gpus all nvidia/cuda:12.5.0-base-ubuntu22.04 nvidia-smi
   ```

2. **Nsight Systems** (Optional): Download `.deb` file from [NVIDIA Developer](https://developer.nvidia.com/nsight-systems)
   - Place the `.deb` file in `docker/` directory
   - Example: `docker/nsight-systems-2025.5.1_2025.5.1.121-1_amd64.deb`

## Quick Start

### Option 1: Automated Build & Test

```bash
cd docker
chmod +x build_and_test.sh
./build_and_test.sh
```

### Option 2: Manual Build

```bash
# Build Docker image
docker build -t leo-fuzzy-fairness -f docker/Dockerfile .

# Run interactive container
docker run --gpus all -it --name fairness-dev leo-fuzzy-fairness bash
```

### Option 3: Docker Compose

```bash
cd docker
docker-compose up --build
```

## Verification

### Inside Container

```bash
# Test Python imports
python3 -c "import sionna, openntn; print('OK')"

# Check GPU visibility
nvidia-smi

# Check TensorFlow GPU
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Run full test suite
bash /workspace/docker/test_installation.sh
```

### Expected Outputs

1. **Sionna Import**: Should show version (e.g., `1.2.1`)
2. **OpenNTN Import**: Should import without error (or show optional warning)
3. **nvidia-smi**: Should show available GPUs (H100 × 2)
4. **TensorFlow GPU**: Should detect both GPUs

## Docker Image Details

- **Base Image**: `nvcr.io/nvidia/tensorflow:25.02-tf2-py3`
- **CUDA**: 12.5
- **Python**: 3.x (from TensorFlow image)
- **Working Directory**: `/workspace`

## Installed Packages

- TensorFlow 2.x (from base image)
- Sionna 1.2.1
- OpenNTN (main branch)
- sgp4, skyfield
- numpy, scipy, matplotlib, pandas
- All dependencies from `requirements.txt`

## Troubleshooting

### Issue: GPU not detected

```bash
# Check nvidia-docker installation
docker run --rm --gpus all nvidia/cuda:12.5.0-base-ubuntu22.04 nvidia-smi

# If fails, install nvidia-docker2
# Ubuntu/Debian:
sudo apt-get install nvidia-docker2
sudo systemctl restart docker
```

### Issue: OpenNTN import fails

OpenNTN is optional. The system will work with fallback channel models. To install manually:

```bash
# Inside container
pip install git+https://github.com/nvidia/openntn.git
```

### Issue: Nsight Systems not found

This is optional. The build will continue without it. To install:

1. Download `.deb` from NVIDIA Developer website
2. Place in `docker/` directory
3. Rebuild image

## Done When

✅ Container runs on both H100 GPUs without errors  
✅ TensorFlow recognizes both GPUs  
✅ `python3 -c "import sionna, openntn; print('OK')"` succeeds  
✅ `nvidia-smi` shows both GPUs inside container  

## Next Steps

After Phase 0 is complete, proceed to:
- **Phase 1**: LEO Geometry + Channel (OpenNTN + Sionna)
- **Phase 2**: Multi-GPU Support

