# Quick Fix for Certificate Expiration Error

## The Problem
```
Failed to install Cursor server: Failed to run devcontainer command: 1. 
{"outcome":"error","message":"certificate has expired"}
```

This happens when Docker tries to pull the base image from NVIDIA's registry (nvcr.io).

## Quick Solution (Choose One)

### Option 1: Fix Host System Certificates (Recommended)
```bash
sudo bash .devcontainer/fix-docker-certificates.sh
```

This will:
- Sync system time
- Update CA certificates
- Test NVIDIA registry access
- Optionally add nvcr.io to insecure registries (if needed)

### Option 2: Pre-build the Image Manually
Build the image first, then use the dev container:

```bash
# Build the image manually
docker build -f docker/Dockerfile.final -t fuzzy-fairness-dss:dev .

# Then open dev container (it will use the pre-built image)
```

### Option 3: Use Alternative Base Image
If NVIDIA registry is completely inaccessible, modify `docker/Dockerfile.final`:

Change line 5 from:
```dockerfile
FROM nvcr.io/nvidia/tensorflow:24.12-tf2-py3
```

To:
```dockerfile
FROM tensorflow/tensorflow:2.16.0-gpu
```

Then rebuild.

### Option 4: Add Insecure Registry (Last Resort)
Only if nothing else works:

```bash
sudo nano /etc/docker/daemon.json
```

Add:
```json
{
  "insecure-registries": ["nvcr.io"]
}
```

Then:
```bash
sudo systemctl restart docker
```

**Warning**: This reduces security.

## After Fixing

1. Rebuild the dev container:
   - VS Code: `F1` → "Dev Containers: Rebuild Container"

2. Verify it works:
   - Container should build and start successfully
   - No certificate errors in the output

## Still Having Issues?

Check:
- System date/time: `date` (should be correct)
- Docker version: `docker --version`
- Test pull: `docker pull nvcr.io/nvidia/tensorflow:24.12-tf2-py3`
- Docker logs: `journalctl -u docker -n 50`

