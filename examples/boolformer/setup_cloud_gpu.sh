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

# Establish persistent SSH connection
echo -e "\n[1/8] Establishing persistent SSH connection..."
ssh -M -S "$SSH_SOCKET" $SSH_OPTS -o ControlPersist=10m $SSH_HOST -N -f
echo "✓ Persistent connection established"

# Update SSH_OPTS to use control socket
SSH_OPTS="-S $SSH_SOCKET"

# Test connection
echo -e "\n[2/8] Testing connection..."
ssh $SSH_OPTS $SSH_HOST "echo 'SSH connection successful'"

# Install system dependencies
echo -e "\n[3/8] Installing system dependencies..."
ssh $SSH_OPTS $SSH_HOST << 'EOF'
apt-get update
apt-get install -y git python3 python3-venv python3-pip
EOF

# Clone repositories
echo -e "\n[4/8] Cloning repositories..."
ssh $SSH_OPTS $SSH_HOST << 'EOF'
cd ~

# Clone Boolformer
git clone https://github.com/arthurenard/Boolformer.git
echo "✓ Cloned Boolformer"

# Clone mctx inside Boolformer (matches local structure)
cd ~/Boolformer
git clone -b boolformer-example https://github.com/Majdoddin/mctx.git
echo "✓ Cloned mctx (boolformer-example branch)"

# Clone flax inside Boolformer (matches local structure)
git clone -b rope-rmsnorm https://github.com/Majdoddin/flax.git
echo "✓ Cloned flax (rope-rmsnorm branch)"
EOF

# Create virtual environment and install dependencies
echo -e "\n[5/8] Creating virtual environment..."
ssh $SSH_OPTS $SSH_HOST << 'EOF'
cd ~/Boolformer

python3 -m venv .venv
echo "✓ Created .venv"

source .venv/bin/activate

# Upgrade pip
pip install --no-cache-dir --upgrade pip

echo "✓ Virtual environment ready"
EOF

# Install JAX with CUDA support
echo -e "\n[6/8] Installing JAX with CUDA support..."
ssh $SSH_OPTS $SSH_HOST << 'EOF'
cd ~/Boolformer
source .venv/bin/activate

# Install JAX with CUDA 12 support (compatible with CUDA 13.0)
pip install --no-cache-dir -U "jax[cuda12]"

echo "✓ JAX with CUDA installed"
EOF

# Install Python dependencies
echo -e "\n[7/8] Installing Python dependencies..."
ssh $SSH_OPTS $SSH_HOST << 'EOF'
cd ~/Boolformer
source .venv/bin/activate

# Install custom flax
cd flax
pip install --no-cache-dir -e .
cd ..

# Install mctx
cd mctx
pip install --no-cache-dir -e .
cd ..

# Install Boolformer dependencies
pip install --no-cache-dir -r requirements.txt

echo "✓ All Python packages installed"
EOF

# Verify installation
echo -e "\n[8/8] Verifying installation..."
ssh $SSH_OPTS $SSH_HOST << 'EOF'
cd ~/Boolformer
source .venv/bin/activate

echo "Checking JAX GPU support..."
python3 -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'Devices: {jax.devices()}'); print(f'Default backend: {jax.default_backend()}')"

echo ""
echo "Checking installed packages..."
pip list | grep -E "(jax|flax|mctx|numpy|optax)"

echo ""
echo "✓ Installation verification complete"
EOF

# Disable auto-tmux for future logins
echo -e "\nDisabling auto-tmux..."
ssh $SSH_OPTS $SSH_HOST "touch ~/.no_auto_tmux"
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
echo "1. SSH to the server: ssh -p $SSH_PORT $SSH_HOST"
echo "2. Activate venv: source ~/Boolformer/.venv/bin/activate"
echo "3. Navigate to: cd ~/Boolformer/mctx/examples/boolformer"
echo "4. Run training: python train.py"
echo ""
echo "To update code on cloud (after pushing local changes):"
echo "ssh -p $SSH_PORT $SSH_HOST 'cd ~/Boolformer/mctx && git pull'"
echo "ssh -p $SSH_PORT $SSH_HOST 'cd ~/Boolformer/flax && git pull'"
echo ""
