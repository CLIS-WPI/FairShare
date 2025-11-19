# Use the official NVIDIA TensorFlow image as the base
#FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3
FROM nvcr.io/nvidia/tensorflow:24.12-tf2-py3
# Set the working directory inside the container
WORKDIR /workspace

# Set the timezone to America/New_York (for Worcester, MA)
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install system dependencies
# Note: Base image already has Python, but we'll use the system Python
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    wget \
    unzip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Verify Python version (base image should have Python 3.x)
RUN python3 --version

# Copy the requirements file and install additional packages
COPY requirements.txt .

# Install Sionna (version 1.0+ for main branch compatibility)
# OpenNTN main branch requires Sionna 1.0+
RUN python3 -m pip install --no-cache-dir sionna

# Verify Sionna installation before installing OpenNTN
RUN python3 -c "import sionna; print(f'Sionna version: {sionna.__version__}')" || \
    echo "Warning: Sionna import failed"

# Install OpenNTN properly using install.sh script
# According to OpenNTN docs: pip install sionna, then run . install.sh
# Note: If repository requires authentication, build will continue without OpenNTN
# The system will use fallback channel models from src/channel/channel_model.py
RUN cd /tmp && \
    # Disable git credential prompts
    export GIT_TERMINAL_PROMPT=0 && \
    export GIT_ASKPASS=echo && \
    git config --global credential.helper store || true && \
    # Try to clone OpenNTN repository (main branch for Sionna 1.0+)
    # Correct repository: ant-uni-bremen/openntn (not nvidia/openntn)
    echo "Attempting to clone OpenNTN repository..." && \
    (git clone --depth 1 https://github.com/ant-uni-bremen/openntn.git openntn 2>&1 || \
     git clone --depth 1 https://github.com/nvidia/openntn.git openntn 2>&1 || \
     git clone --depth 1 https://github.com/NVlabs/openntn.git openntn 2>&1 || \
     (echo "Git clone failed, trying to download as ZIP..." && \
      (wget -q https://github.com/ant-uni-bremen/openntn/archive/refs/heads/main.zip -O openntn.zip 2>&1 || \
       wget -q https://github.com/nvidia/openntn/archive/refs/heads/main.zip -O openntn.zip 2>&1 || \
       wget -q https://github.com/NVlabs/openntn/archive/refs/heads/main.zip -O openntn.zip 2>&1 || \
       true) && \
      if [ -f "openntn.zip" ]; then \
          unzip -q openntn.zip && \
          mv openntn-main openntn && \
          echo "✓ Downloaded OpenNTN as ZIP"; \
      else \
          echo "Warning: OpenNTN download failed - repository may require authentication"; \
      fi) || true) && \
    if [ -d "openntn" ] && [ "$(ls -A openntn 2>/dev/null)" ]; then \
        cd openntn && \
        echo "OpenNTN repository cloned. Contents:" && \
        ls -la . && \
        # Get Sionna path
        SIONNA_PATH=$(python3 -c "import sionna; import os; print(os.path.dirname(sionna.__file__))") && \
        CHANNEL_DIR="$SIONNA_PATH/phy/channel" && \
        echo "Sionna path: $SIONNA_PATH" && \
        echo "Channel dir: $CHANNEL_DIR" && \
        # Try install.sh first (it uses pip install and creates symlink)
        if [ -f "install.sh" ]; then \
            echo "Running install.sh..." && \
            chmod +x install.sh && \
            bash install.sh 2>&1 && \
            echo "✓ OpenNTN installed via install.sh"; \
        else \
            echo "Warning: install.sh not found, trying manual installation..." && \
            # Manual installation: install via pip
            python3 -m pip install git+https://github.com/ant-uni-bremen/OpenNTN@main && \
            # Download and run post_install.py
            POST_INSTALL_URL="https://raw.githubusercontent.com/ant-uni-bremen/OpenNTN/refs/heads/main/post_install.py" && \
            curl -L -o /tmp/post_install.py "$POST_INSTALL_URL" && \
            python3 /tmp/post_install.py && \
            # Create symlink
            OPENNTN_DIR=$(python3 -m pip show OpenNTN | grep Location | cut -d' ' -f2)/OpenNTN && \
            ln -s "$OPENNTN_DIR" "$CHANNEL_DIR/tr38811" && \
            echo "✓ OpenNTN manually installed"; \
        fi && \
        # Update __init__.py
        if [ -d "$CHANNEL_DIR/tr38811" ]; then \
            echo "Updating __init__.py..." && \
            if [ -f "$CHANNEL_DIR/__init__.py" ]; then \
                if ! grep -q "tr38811" "$CHANNEL_DIR/__init__.py"; then \
                    echo "from . import tr38811" >> "$CHANNEL_DIR/__init__.py" && \
                    echo "✓ __init__.py updated"; \
                else \
                    echo "✓ tr38811 already in __init__.py"; \
                fi; \
            fi && \
            echo "Verifying installation..." && \
            ls -la "$CHANNEL_DIR/tr38811" | head -5; \
        else \
            echo "Warning: tr38811 directory not found after installation"; \
        fi && \
        cd /tmp && \
        rm -rf openntn; \
    else \
        echo "=========================================" && \
        echo "WARNING: OpenNTN installation skipped" && \
        echo "Reason: Repository requires authentication or is not accessible" && \
        echo "Solution: The system will use fallback channel models from" && \
        echo "          src/channel/channel_model.py which implements TR38.811" && \
        echo "========================================="; \
    fi || \
    (echo "=========================================" && \
     echo "WARNING: OpenNTN installation failed" && \
     echo "Build will continue with fallback channel models" && \
     echo "=========================================" && \
     true)

# Verify OpenNTN installation - try multiple import paths
RUN (python3 -c "from sionna.phy.channel import tr38811; print('OpenNTN: OK (sionna.phy.channel)')" || \
     python3 -c "from sionna.channel import tr38811; print('OpenNTN: OK (sionna.channel)')" || \
     python3 -c "import sionna.phy.channel.tr38811; print('OpenNTN: OK (direct import)')" || \
     echo "Warning: OpenNTN import failed - check installation")

# Install other Python dependencies
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Set Python path
ENV PYTHONPATH=/workspace

# Enable TensorFlow GPU placement and XLA
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_XLA_FLAGS="--tf_xla_enable_xla_devices --tf_xla_auto_jit=2"

# Default command
CMD ["python3", "src/main.py", "--scenario", "experiments/scenarios/rural_coverage.yaml"]
