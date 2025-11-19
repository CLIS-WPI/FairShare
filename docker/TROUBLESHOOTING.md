# Docker Build Troubleshooting

## Issue: Certificate Expired Error

If you see an error like:
```
tls: failed to verify certificate: x509: certificate has expired or is not yet valid
current time 2025-11-18 is after 2025-05-07
```

### Solutions (in order of preference):

1. **Use Local Dockerfile (RECOMMENDED - No registry needed)**
   ```bash
   docker build -t leo-fuzzy-fairness -f docker/Dockerfile.local .
   ```
   This builds from Ubuntu and installs everything manually, bypassing registry issues.

2. **Fix System Clock**
   ```bash
   # Check current date
   date
   
   # Sync time (requires root)
   sudo ntpdate -s time.nist.gov
   # or
   sudo timedatectl set-ntp true
   
   # Or use the fix script
   bash docker/fix_certificate.sh
   ```

3. **Configure Docker to Skip Certificate Verification (NOT RECOMMENDED)**
   
   Edit `/etc/docker/daemon.json` (requires root):
   ```json
   {
     "insecure-registries": ["nvcr.io", "registry-1.docker.io"]
   }
   ```
   Then restart Docker:
   ```bash
   sudo systemctl restart docker
   ```
   **Warning**: This reduces security. Only use if absolutely necessary.

4. **Use Alternative Dockerfile**
   ```bash
   docker build -t leo-fuzzy-fairness -f docker/Dockerfile.alternative .
   ```

5. **Login to NVIDIA NGC (if required)**
   ```bash
   docker login nvcr.io
   # Use your NVIDIA NGC API key
   ```

## Issue: Image Tag Not Found

If `25.02-tf2-py3` doesn't exist:

1. **Check Available Tags**
   Visit: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow

2. **Use Latest Stable Version**
   ```dockerfile
   FROM nvcr.io/nvidia/tensorflow:24.08-tf2-py3
   ```

## Issue: OpenNTN Installation Fails

OpenNTN is optional. The system will work with fallback channel models.

To install manually inside container:
```bash
pip install git+https://github.com/nvidia/openntn.git
```

## Issue: GPU Not Detected

1. **Check nvidia-docker2**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.5.0-base-ubuntu22.04 nvidia-smi
   ```

2. **Install NVIDIA Container Toolkit**
   ```bash
   # Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

## Quick Fix Commands

```bash
# Try with alternative Dockerfile
docker build -t leo-fuzzy-fairness -f docker/Dockerfile.alternative .

# Or try with different tag (edit Dockerfile first)
# Change FROM line to: FROM nvcr.io/nvidia/tensorflow:24.08-tf2-py3
docker build -t leo-fuzzy-fairness -f docker/Dockerfile .

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.5.0-base-ubuntu22.04 nvidia-smi
```

