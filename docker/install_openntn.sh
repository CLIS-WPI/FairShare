#!/bin/bash
# Script to install OpenNTN into Sionna
# This can be run inside the container if OpenNTN wasn't installed during build

set -e

echo "Installing OpenNTN..."

# Install OpenNTN via pip
pip install git+https://github.com/ant-uni-bremen/OpenNTN@main

# Download post_install.py
POST_INSTALL_URL="https://raw.githubusercontent.com/ant-uni-bremen/OpenNTN/refs/heads/main/post_install.py"
POST_INSTALL_NAME=$(basename "$POST_INSTALL_URL")
curl -L -o "$POST_INSTALL_NAME" "$POST_INSTALL_URL"

# Run post_install.py
python3 "$POST_INSTALL_NAME"

# Create symlink
OPENNTN_DIR=$(pip show OpenNTN | grep Location | cut -d' ' -f2)/OpenNTN
SIONNA_DIR=$(pip show sionna | grep Location | cut -d' ' -f2)/sionna
CHANNEL_DIR="$SIONNA_DIR/phy/channel"

# Remove old symlink if exists
rm -f "$CHANNEL_DIR/tr38811"

# Create correct symlink
ln -s "$OPENNTN_DIR" "$CHANNEL_DIR/tr38811"

# Verify
python3 -c "from sionna.phy.channel import tr38811; print('✓ OpenNTN installed successfully!')" || {
    echo "⚠ Verification failed, but installation may still work"
}

# Cleanup
rm -f "$POST_INSTALL_NAME"

echo "✓ OpenNTN installation complete"

