#!/bin/bash
# Setup script for cloud GPU training environment
# Usage: ./setup_cloud_gpu.sh <ssh_host> [ssh_port]
#
# If ssh_port is omitted, uses port from ~/.ssh/config
# Examples:
#   ./setup_cloud_gpu.sh Vast-Ruhollah              # uses config
#   ./setup_cloud_gpu.sh user@gpu.example.com 22    # explicit port

set -e  # Exit on error

if [ $# -lt 1 ]; then
    echo "Usage: $0 <ssh_host> [ssh_port]"
    echo "Examples:"
    echo "  $0 Vast-Ruhollah              # uses ~/.ssh/config"
    echo "  $0 user@gpu.example.com 22    # explicit port"
    exit 1
fi

SSH_HOST=$1
SSH_SOCKET="/tmp/ssh-cloud-gpu-$$"

# If port is explicitly provided, use it. Otherwise rely on SSH config.
if [ $# -ge 2 ]; then
    SSH_PORT=$2
    SSH_OPTS="-p $SSH_PORT"
    echo "========================================="
    echo "Setting up Boolformer MCTS on cloud GPU"
    echo "Host: $SSH_HOST"
    echo "Port: $SSH_PORT"
    echo "========================================="
else
    SSH_OPTS=""
    echo "========================================="
    echo "Setting up Boolformer MCTS on cloud GPU"
    echo "Host: $SSH_HOST"
    echo "Port: (from SSH config)"
    echo "========================================="
fi

# Helper: run command over SSH, abort on failure
run_remote() {
    local step_name="$1"
    shift
    if ! ssh $SSH_OPTS $SSH_HOST "$@"; then
        echo "ERROR: Step '$step_name' failed!"
        # Close persistent connection before exiting
        ssh -S "$SSH_SOCKET" -O exit $SSH_HOST 2>/dev/null || true
        exit 1
    fi
}

# Establish persistent SSH connection
echo -e "\n[1/8] Establishing persistent SSH connection..."
ssh -M -S "$SSH_SOCKET" $SSH_OPTS -o ControlPersist=10m $SSH_HOST -N -f
echo "✓ Persistent connection established"

# Update SSH_OPTS to use control socket
SSH_OPTS="-S $SSH_SOCKET"

# Test connection
echo -e "\n[2/8] Testing connection..."
run_remote "Test connection" "echo 'SSH connection successful'"

# Install system dependencies
echo -e "\n[3/8] Installing system dependencies..."
run_remote "Install system dependencies" "set -e; apt-get update && apt-get install -y git python3 python3-venv python3-pip"

# Clone repositories
echo -e "\n[4/8] Cloning repositories..."
run_remote "Clone repositories" 'set -e
cd ~
git clone https://github.com/arthurenard/Boolformer.git
echo "✓ Cloned Boolformer"
cd ~/Boolformer
git clone -b boolformer-example https://github.com/Majdoddin/mctx.git
echo "✓ Cloned mctx (boolformer-example branch)"
git clone -b rope-rmsnorm https://github.com/Majdoddin/flax.git
echo "✓ Cloned flax (rope-rmsnorm branch)"'

# Create virtual environment and install dependencies
echo -e "\n[5/8] Creating virtual environment..."
run_remote "Create virtual environment" 'set -e
cd ~/Boolformer
python3 -m venv .venv
echo "✓ Created .venv"
source .venv/bin/activate
pip install --no-cache-dir --upgrade pip
echo "✓ Virtual environment ready"'

# Install JAX with bundled CUDA/CuDNN.
# jax[cuda12_local] would avoid ~2GB download but only works with exact CUDA 12
# system libs (not CUDA 13) and CuDNN >= 9.8. Too fragile for varying cloud images.
echo -e "\n[6/8] Installing JAX with CUDA support..."
run_remote "Install JAX" 'set -e
cd ~/Boolformer
source .venv/bin/activate
pip install --no-cache-dir -U "jax[cuda12]"
echo "✓ JAX with CUDA installed"'

# Install Python dependencies
echo -e "\n[7/8] Installing Python dependencies..."
run_remote "Install Python dependencies" 'set -e
cd ~/Boolformer
source .venv/bin/activate
cd flax && pip install --no-cache-dir -e . && cd ..
echo "✓ Installed flax"
cd mctx && pip install --no-cache-dir -e . && cd ..
echo "✓ Installed mctx"
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
echo "✓ Installed PyTorch (CPU-only)"
pip install --no-cache-dir -r requirements.txt
echo "✓ All Python packages installed"'

# Verify installation
echo -e "\n[8/8] Verifying installation..."
run_remote "Verify installation" 'set -e
cd ~/Boolformer
source .venv/bin/activate
echo "Checking JAX GPU support..."
python3 -c "import jax; print(f'"'"'JAX version: {jax.__version__}'"'"'); print(f'"'"'Devices: {jax.devices()}'"'"'); print(f'"'"'Default backend: {jax.default_backend()}'"'"')"
echo ""
echo "Checking installed packages..."
pip list | grep -E "(jax|flax|mctx|numpy|optax)"
echo ""
echo "✓ Installation verification complete"'

# Disable auto-tmux for future logins
echo -e "\nDisabling auto-tmux..."
run_remote "Disable auto-tmux" "touch ~/.no_auto_tmux"
echo "✓ Auto-tmux disabled"

# Close persistent SSH connection
echo -e "\nClosing persistent SSH connection..."
ssh -S "$SSH_SOCKET" -O exit $SSH_HOST 2>/dev/null || true
echo "✓ Connection closed"

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "To start training:"
echo "1. SSH to the server: ssh $SSH_HOST"
echo "2. Activate venv: source ~/Boolformer/.venv/bin/activate"
echo "3. Navigate to: cd ~/Boolformer/mctx/examples/boolformer"
echo "4. Run training: python train.py"
echo ""
echo "To update code on cloud (after pushing local changes):"
echo "ssh $SSH_HOST 'cd ~/Boolformer/mctx && git pull'"
echo "ssh $SSH_HOST 'cd ~/Boolformer/flax && git pull'"
echo ""
