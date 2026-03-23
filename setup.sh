#!/bin/bash
# Setup script for NAIC AIS Graph Classification Demonstrator
# Run this after cloning: ./setup.sh

set -e

echo "=== NAIC AIS Graph Classification Demonstrator Setup ==="
echo ""

# Check if module system is available (NAIC VMs / Sigma2 clusters)
if command -v module &> /dev/null; then
    echo "Module system detected (NAIC VM / Sigma2)"
    echo "Loading Python module..."
    module load Python/3.11.5-GCCcore-13.2.0 2>/dev/null || module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null || echo "Using system Python"
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
echo "Python version: $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo "ERROR: Python 3.10+ required. Found: $PYTHON_VERSION"
    exit 1
fi
echo "Python version OK"

# Check for GPU
echo ""
echo "Checking GPU availability..."
GPU_AVAILABLE=false
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "GPU detected but nvidia-smi query failed"
    GPU_AVAILABLE=true

    # Setup CUDA library symlinks if needed
    if [ ! -d "$HOME/cuda_link" ]; then
        echo "Setting up CUDA library symlinks..."
        mkdir -p $HOME/cuda_link
        ln -sf /lib/x86_64-linux-gnu/libcuda.so.1 $HOME/cuda_link/libcuda.so.1 2>/dev/null || true
        ln -sf /lib/x86_64-linux-gnu/libcuda.so.1 $HOME/cuda_link/libcuda.so 2>/dev/null || true
    fi
    export LD_LIBRARY_PATH=$HOME/cuda_link:$LD_LIBRARY_PATH
else
    echo "No NVIDIA GPU detected (CPU-only mode)"
fi

# Create virtual environment
echo ""
if [ -d "venv" ]; then
    echo "Virtual environment already exists."
    echo "Reusing existing venv. To recreate: rm -rf venv && ./setup.sh"
else
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install PyTorch (CPU or GPU)
echo ""
echo "Installing PyTorch..."
if [ "$GPU_AVAILABLE" = true ]; then
    pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118 --quiet
else
    pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu --quiet
fi

# Install DGL
echo "Installing DGL..."
if [ "$GPU_AVAILABLE" = true ]; then
    pip install dgl==2.4.0+cu118 -f https://data.dgl.ai/wheels/torch-2.4/cu118/repo.html --quiet
else
    pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/repo.html --quiet
fi

# Install remaining dependencies
echo "Installing project dependencies..."
pip install -r requirements.txt --quiet

# Install the project package
echo "Installing ais_dgl package..."
pip install -e . --quiet

# Install ipykernel for Jupyter
pip install ipykernel --quiet
python -m ipykernel install --user --name=ais_dgl --display-name "AIS-DGL"

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "
import torch
import dgl
print(f'PyTorch version: {torch.__version__}')
print(f'DGL version: {dgl.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
from graph_classification.models import GCN, GAT, GraphSAGE
print('Models imported successfully: GCN, GAT, GraphSAGE')
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the demonstrator:"
echo "  Option A: jupyter lab (select 'AIS-DGL' kernel)"
echo "  Option B: export PYTHONPATH=\$PYTHONPATH:\$(pwd)/src"
echo "            python src/graph_classification/train_graph_classification_ais.py --data_folder data/"
echo ""
