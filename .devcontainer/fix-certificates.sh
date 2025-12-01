#!/bin/bash
# Fix certificate issues for dev container
# Run this on the host system before building the container

set -e

echo "========================================="
echo "Fixing Certificate Issues"
echo "========================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "This script needs root privileges."
    echo "Please run with: sudo bash .devcontainer/fix-certificates.sh"
    exit 1
fi

echo "1. Updating system time..."
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
echo "2. Updating CA certificates..."
# Update CA certificates
if [ -f /etc/ssl/certs/ca-certificates.crt ]; then
    update-ca-certificates
    echo "✓ CA certificates updated"
else
    echo "⚠ CA certificates file not found"
fi

echo ""
echo "3. Restarting Docker (if needed)..."
# Restart Docker to pick up certificate changes
if systemctl is-active --quiet docker; then
    systemctl restart docker
    echo "✓ Docker restarted"
else
    echo "⚠ Docker not running as a service"
fi

echo ""
echo "========================================="
echo "Certificate fix complete!"
echo "========================================="
echo ""
echo "Now try building the dev container again."

