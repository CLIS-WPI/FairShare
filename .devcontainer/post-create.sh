#!/bin/bash
# Post-create script for dev container
# Updates certificates and installs the project

set -e

echo "========================================="
echo "Dev Container Post-Create Setup"
echo "========================================="

# Update CA certificates (fix certificate expiration issues)
echo "Updating CA certificates..."
apt-get update -qq && \
apt-get install -y --no-install-recommends ca-certificates -qq && \
update-ca-certificates && \
rm -rf /var/lib/apt/lists/*

# Install project in editable mode
echo "Installing project in editable mode..."
python3 -m pip install --no-cache-dir -e .

echo "========================================="
echo "Setup complete!"
echo "========================================="

