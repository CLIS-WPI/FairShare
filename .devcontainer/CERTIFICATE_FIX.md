# Certificate Expiration Fix

## Problem
When opening the dev container, you may see:
```
Failed to install Cursor server: Failed to run devcontainer command: 1. 
{"outcome":"error","message":"certificate has expired","description":"An error occurred setting up the container."}
```

**This error occurs when Docker tries to pull the base image from NVIDIA's registry (nvcr.io) during the build phase.**

## Quick Fix (Start Here!)

**Run this first:**
```bash
sudo bash .devcontainer/fix-docker-certificates.sh
```

This script will:
1. Fix system time
2. Update CA certificates  
3. Test NVIDIA registry access
4. Optionally configure Docker to handle certificate issues

Then rebuild the dev container: `F1` → "Dev Containers: Rebuild Container"

## Solutions

### Solution 1: Fix Host System Certificates (Recommended)
The certificate error happens during Docker build when pulling the base image. Fix it on the host:

```bash
sudo bash .devcontainer/fix-docker-certificates.sh
```

Or manually:
```bash
# Sync system time
sudo timedatectl set-ntp true
# or
sudo ntpdate -s time.nist.gov

# Update CA certificates
sudo update-ca-certificates

# Restart Docker
sudo systemctl restart docker
```

### Solution 2: Pre-build the Image
Build the image manually first, then use the dev container:

```bash
# Build the image
docker build -f docker/Dockerfile.final -t fuzzy-fairness-dss:dev .

# Then open dev container (it will use the pre-built image)
```

### Solution 3: Automatic Fix in Container
The dev container automatically updates certificates on startup via `post-create.sh` (but this only helps after the image is built).

**After fixing host certificates, rebuild the container:**
1. In VS Code: `F1` → "Dev Containers: Rebuild Container"
2. The post-create script will update certificates automatically

### Solution 2: Fix Host System Certificates
If the issue persists, fix certificates on your host system:

```bash
# Run on host (may need sudo)
sudo bash .devcontainer/fix-certificates.sh
```

Or manually:
```bash
# Update system time
sudo timedatectl set-ntp true
# or
sudo ntpdate -s time.nist.gov

# Update CA certificates
sudo update-ca-certificates

# Restart Docker
sudo systemctl restart docker
```

### Solution 3: Use Alternative Base Image
If NVIDIA registry (nvcr.io) is still having issues, you can modify the Dockerfile to use a different base:

Edit `docker/Dockerfile.final`:
```dockerfile
# Instead of:
FROM nvcr.io/nvidia/tensorflow:24.12-tf2-py3

# Use:
FROM tensorflow/tensorflow:2.16.0-gpu
```

Then install CUDA and other dependencies manually.

### Solution 4: Build Without Dev Container
Build the image manually first:

```bash
# Build the image
docker build -f docker/Dockerfile.final -t fuzzy-fairness-dss:dev .

# Then use the dev container (it will use the pre-built image)
```

### Solution 5: Skip Certificate Verification (NOT RECOMMENDED)
Only use this as a last resort:

Edit `/etc/docker/daemon.json` (requires root):
```json
{
  "insecure-registries": ["nvcr.io"]
}
```

Then restart Docker:
```bash
sudo systemctl restart docker
```

**Warning**: This reduces security. Only use if absolutely necessary.

## What Was Fixed

1. **Dockerfile.final**: Added CA certificate update step
2. **post-create.sh**: Automatically updates certificates when container starts
3. **fix-certificates.sh**: Host system certificate fix script

## Verification

After fixing, verify certificates work:

```bash
# Inside container
curl -I https://github.com
# Should return HTTP 200, not certificate error
```

## Still Having Issues?

1. Check system date/time: `date`
2. Check Docker version: `docker --version`
3. Check if base image can be pulled: `docker pull nvcr.io/nvidia/tensorflow:24.12-tf2-py3`
4. Check Docker logs: `journalctl -u docker -n 50`

