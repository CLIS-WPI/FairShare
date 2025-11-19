# Docker Setup

## Building the Docker Image

### Prerequisites

1. **NVIDIA Docker Runtime**: Ensure you have nvidia-docker2 installed
2. **Nsight Systems** (Optional): For profiling, download the `.deb` file from NVIDIA Developer website and place it in the `docker/` directory as `nsight-systems-*.deb`

### Building

```bash
cd docker
docker-compose build
```

### Running

```bash
docker-compose up
```

## Nsight Systems Installation

If you want to use Nsight Systems for profiling:

1. Download the `.deb` file from: https://developer.nvidia.com/nsight-systems
2. Place it in the `docker/` directory
3. The Dockerfile will automatically install it during build

If the `.deb` file is not found, the build will continue without Nsight Systems.

## Multi-GPU Support

The Docker image is configured for multi-GPU usage with:
- TensorFlow GPU placement enabled
- XLA compilation enabled
- Automatic Mixed Precision (AMP) ready

For H100 × 2 systems, the container will automatically detect and use both GPUs.

## Environment Variables

- `TF_FORCE_GPU_ALLOW_GROWTH=true`: Allow GPU memory growth
- `TF_XLA_FLAGS`: XLA compilation flags
- `PYTHONPATH=/workspace`: Python path for imports

