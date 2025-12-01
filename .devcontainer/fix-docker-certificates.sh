#!/bin/bash
# Fix Docker certificate issues for pulling images from NVIDIA registry
# Run this on the host system BEFORE building the dev container

set -e

echo "========================================="
echo "Fixing Docker Certificate Issues"
echo "========================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "This script needs root privileges."
    echo "Please run with: sudo bash .devcontainer/fix-docker-certificates.sh"
    exit 1
fi

echo "1. Checking system date/time..."
date
echo ""

echo "2. Updating system time..."
# Sync system time
if command -v timedatectl &> /dev/null; then
    timedatectl set-ntp true
    echo "✓ Time sync enabled"
elif command -v ntpdate &> /dev/null; then
    ntpdate -s time.nist.gov
    echo "✓ Time synced"
else
    echo "⚠ No time sync tool found"
fi

echo ""
echo "3. Updating CA certificates..."
# Update CA certificates
if [ -f /etc/ssl/certs/ca-certificates.crt ]; then
    update-ca-certificates
    echo "✓ CA certificates updated"
else
    echo "⚠ CA certificates file not found"
fi

echo ""
echo "4. Checking Docker daemon configuration..."
DOCKER_DAEMON_JSON="/etc/docker/daemon.json"
if [ ! -f "$DOCKER_DAEMON_JSON" ]; then
    echo "Creating Docker daemon.json..."
    mkdir -p /etc/docker
    cat > "$DOCKER_DAEMON_JSON" << 'EOF'
{
  "insecure-registries": []
}
EOF
    echo "✓ Created $DOCKER_DAEMON_JSON"
else
    echo "✓ Docker daemon.json exists"
fi

echo ""
echo "5. Testing NVIDIA registry access..."
# Test if we can access NVIDIA registry
if docker pull nvcr.io/nvidia/tensorflow:24.12-tf2-py3 2>&1 | grep -q "certificate"; then
    echo "⚠ Certificate error detected when pulling NVIDIA image"
    echo ""
    echo "OPTION A: Add nvcr.io to insecure registries (less secure)"
    read -p "Do you want to add nvcr.io to insecure registries? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Backup existing config
        cp "$DOCKER_DAEMON_JSON" "${DOCKER_DAEMON_JSON}.backup.$(date +%Y%m%d_%H%M%S)"
        
        # Add insecure registry
        python3 << 'PYTHON_SCRIPT'
import json
import sys

daemon_json = "/etc/docker/daemon.json"
try:
    with open(daemon_json, 'r') as f:
        config = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    config = {}

if "insecure-registries" not in config:
    config["insecure-registries"] = []

if "nvcr.io" not in config["insecure-registries"]:
    config["insecure-registries"].append("nvcr.io")
    print("Added nvcr.io to insecure registries")
else:
    print("nvcr.io already in insecure registries")

with open(daemon_json, 'w') as f:
    json.dump(config, f, indent=2)

print("✓ Docker daemon.json updated")
PYTHON_SCRIPT
        
        echo ""
        echo "6. Restarting Docker..."
        systemctl restart docker
        echo "✓ Docker restarted"
        echo ""
        echo "⚠ WARNING: nvcr.io is now in insecure registries"
        echo "   This reduces security but allows pulling images"
    else
        echo "Skipping insecure registry configuration"
    fi
else
    echo "✓ NVIDIA registry accessible"
fi

echo ""
echo "========================================="
echo "Certificate fix complete!"
echo "========================================="
echo ""
echo "Now try building the dev container again:"
echo "  F1 → Dev Containers: Rebuild Container"

