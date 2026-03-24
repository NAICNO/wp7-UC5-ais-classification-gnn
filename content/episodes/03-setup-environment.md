# Setting Up the Environment

```{objectives}
- Clone the repository and run the automated setup script
- Understand what `setup.sh` does at each step
- Perform manual installation for CPU-only or custom CUDA setups
- Verify that PyTorch, DGL, and the `ais_dgl` package work together
- Register a Jupyter kernel for notebook usage
```

## Quick Start

The fastest way to get started is the automated setup script:

```bash
git clone https://github.com/NAICNO/wp7-UC5-ais-classification-gnn.git
cd graph-based-classification-of-ais-time-series-data
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

## What `setup.sh` Does

The setup script performs these steps in order:

1. **Detect GPU availability**: Checks for `nvidia-smi` and queries the CUDA version to determine whether to install GPU or CPU packages
2. **Create a Python virtual environment**: Creates `venv/` using `python3 -m venv`
3. **Upgrade pip**: Ensures the latest pip version is available
4. **Install PyTorch**: Installs PyTorch with the matching CUDA version (or CPU-only if no GPU is detected)
5. **Install DGL**: Installs the Deep Graph Library with the matching CUDA backend
6. **Install the `ais_dgl` package**: Installs the project package in development mode (`pip install -e .`)
7. **Register Jupyter kernel**: Creates an `ais_dgl` kernel for use in Jupyter notebooks

```{admonition} CUDA Version Matching
:class: tip

PyTorch and DGL must be installed with matching CUDA versions. The `setup.sh` script handles this automatically. If you see errors about CUDA version mismatches, delete the `venv/` directory and run `setup.sh` again. The script reads the CUDA version from `nvidia-smi` output and selects the appropriate package index.
```

## Manual Installation

If you need more control over the installation (e.g., a specific PyTorch version or CPU-only setup), follow these steps.

### CPU-Only Installation

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Install PyTorch (CPU only)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install DGL (CPU only)
pip install dgl -f https://data.dgl.ai/wheels/repo.html

# Install project package
pip install -e .
```

### GPU Installation (CUDA 12.x)

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Install PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install DGL with CUDA 12.1
pip install dgl -f https://data.dgl.ai/wheels/cu121/repo.html

# Install project package
pip install -e .
```

### GPU Installation (CUDA 11.x)

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Install PyTorch with CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install DGL with CUDA 11.8
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html

# Install project package
pip install -e .
```

## Verify Installation

Run these checks to confirm everything is installed correctly:

```bash
# Check PyTorch and DGL versions
python -c "import torch; import dgl; print(f'PyTorch {torch.__version__}, DGL {dgl.__version__}')"

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check that DGL can create a graph
python -c "import dgl; g = dgl.graph(([0,1],[1,2])); print(f'DGL graph: {g.num_nodes()} nodes, {g.num_edges()} edges')"

# Run the test suite
pytest tests/ -v
```

### Verifying DGL and PyTorch Work Together

A quick smoke test to confirm the full pipeline works:

```python
import torch
import dgl

# Create a simple graph
g = dgl.graph(([0, 1, 2], [1, 2, 0]))
g = dgl.add_self_loop(g)

# Add node features
g.ndata['feat'] = torch.randn(3, 4)

# Test a GCN layer
from dgl.nn import GraphConv
conv = GraphConv(4, 2)
out = conv(g, g.ndata['feat'])
print(f'Input shape: {g.ndata["feat"].shape}')
print(f'Output shape: {out.shape}')
print('DGL + PyTorch integration OK')
```

If you have a GPU, also verify GPU tensor operations:

```python
import torch
if torch.cuda.is_available():
    x = torch.randn(3, 4, device='cuda')
    print(f'GPU tensor device: {x.device}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
else:
    print('No GPU available -- CPU mode will be used')
```

## Jupyter Kernel Setup

To use the environment in Jupyter notebooks, register the kernel:

```bash
source venv/bin/activate
python -m ipykernel install --user --name=ais_dgl --display-name "AIS DGL (Python 3)"
```

Verify the kernel is registered:

```bash
jupyter kernelspec list
```

You should see `ais_dgl` in the list. When opening the notebook, select **"AIS DGL (Python 3)"** as the kernel.

If Jupyter Lab is not installed, add it to the environment:

```bash
pip install jupyterlab
jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

## Data Requirements

Place the AIS dataset files in a `data/` directory:

| File | Shape | Description |
|------|-------|-------------|
| `X_ts12.npy` | `(N, 3, 12)` | Feature array: velocity, distance to shore, curvature for 12 time steps |
| `y_ts12.npy` | `(N,)` | Label array: 0 (non-fishing) or 1 (fishing) |
| `bidx_ts12.npy` | `(50, N)` | Bootstrap split indices for 50 different train/val/test splits |

The total dataset contains ~23,500 samples split into training (14,100), validation (4,700), and test (4,700) sets.

```{keypoints}
- Run `setup.sh` for automated installation with GPU auto-detection
- PyTorch and DGL must have matching CUDA versions -- the setup script handles this
- For CPU-only installations, use the `--index-url` flag with the CPU wheel URL
- Verify the installation with `import torch; import dgl` and a simple graph creation test
- Register the Jupyter kernel with `python -m ipykernel install --user --name=ais_dgl`
- The dataset consists of three `.npy` files placed in the `data/` directory
- Delete `venv/` and re-run `setup.sh` if you encounter version mismatch errors
```
