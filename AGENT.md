# AI Agent Setup Instructions

## Quick Start

### Step 1: SSH into your NAIC VM

Replace the IP address with the one shown in the NAIC Orchestrator portal.
Do NOT type the angle brackets -- use the actual IP and key path.

```bash
ssh -i ~/.ssh/naic-vm.pem ubuntu@10.212.136.52
```

### Step 2: Initialize VM (first time only)

```bash
curl -O https://raw.githubusercontent.com/NAICNO/wp7-UC5-ais-classification-gnn/main/vm-init.sh
chmod +x vm-init.sh
./vm-init.sh
```

### Step 3: Clone and setup

```bash
git clone https://github.com/NAICNO/wp7-UC5-ais-classification-gnn.git
cd graph-based-classification-of-ais-time-series-data
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### Step 4: Run training

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python src/graph_classification/train_graph_classification_ais.py --data_folder data/ --epochs 100 --patience 20
```

### Step 5: Evaluate models

```bash
python src/graph_classification/eval_graph_classification_ais.py --data_folder data/ --model_path results/
```

## Jupyter Notebook Access

Start Jupyter on the VM, then create an SSH tunnel from your laptop.

**On the VM:**
```bash
cd graph-based-classification-of-ais-time-series-data
source venv/bin/activate
jupyter lab --no-browser --ip=0.0.0.0 --port=8888
```

**On your laptop** (new terminal):
```bash
ssh -v -N -L 8888:localhost:8888 -i ~/.ssh/naic-vm.pem ubuntu@10.212.136.52
# Then open: http://localhost:8888
```

## Verification Steps

1. Check Python: `python3 --version` (need 3.10+)
2. Check GPU: `python -c "import torch; print(torch.cuda.is_available())"`
3. Check DGL: `python -c "import dgl; print(dgl.__version__)"`
4. Check data: `ls data/*.npy` (should show X_ts12.npy, y_ts12.npy, bidx_ts12.npy)
5. Run tests: `pytest tests/`

## Project Structure

```
graph-based-classification-of-ais-time-series-data/
├── AGENT.md                    # This file
├── AGENT.yaml                  # Machine-readable version
├── README.md                   # Project overview
├── setup.sh                    # Environment setup
├── vm-init.sh                  # VM initialization
├── requirements.txt            # ML dependencies
├── requirements-docs.txt       # Documentation dependencies
├── pyproject.toml              # Package configuration
├── widgets.py                  # Jupyter widgets
├── utils.py                    # Cluster utilities
├── notebooks/
│   └── DGL_Demonstrator.ipynb  # Interactive notebook
├── src/graph_classification/   # Source code
│   ├── models.py               # GCN, GAT, GraphSAGE
│   ├── heads.py                # Classification heads
│   ├── ais_timeseries_dataset.py  # DGL dataset
│   ├── utils.py                # Training utilities
│   ├── train_graph_classification_ais.py  # Training script
│   └── eval_graph_classification_ais.py   # Evaluation script
├── tests/                      # pytest test suite
├── results/                    # Output directory
└── content/                    # Sphinx documentation
```

## CLI Reference

```bash
# Training
python src/graph_classification/train_graph_classification_ais.py [OPTIONS]

Options:
  --data_folder PATH    Dataset directory (default: ../data/)
  --model_path PATH     Output directory for models (default: ../results)
  --models LIST         Comma-separated models: GCN, GSG, GAT
  --lrs LIST            Comma-separated learning rates (default: 5e-2, 3e-2, 1e-2)
  --epochs N            Training epochs (default: 1000)
  --hidden N            Hidden layer size (default: 32)
  --batch_size N        Batch size (default: 600)
  --patience N          Early stopping patience (default: 200)
  --gpu N               GPU device index (default: 0)

# Evaluation
python src/graph_classification/eval_graph_classification_ais.py [OPTIONS]

Options:
  --data_folder PATH    Dataset directory (default: ../data)
  --model_path PATH     Directory with saved .pt models (default: ../results)
  --batch_size N        Batch size for evaluation (default: 4000)
```

## Models

| Model | Type | Description |
|-------|------|-------------|
| GCN | Graph Convolutional Network | Standard spectral convolutions |
| GSG (GraphSAGE) | Sample and Aggregate | Inductive learning via neighborhood sampling |
| GAT | Graph Attention Network | Attention-weighted neighbor aggregation |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| SSH "Permission denied" | `chmod 600 ~/.ssh/your-key.pem` |
| ModuleNotFoundError | `source venv/bin/activate` |
| DGL import error | Run `./setup.sh` to install compatible versions |
| Data not found | Place X_ts12.npy, y_ts12.npy, bidx_ts12.npy in data/ |
| CUDA out of memory | Reduce batch size: `--batch_size 100` |
| No GPU detected | Check `nvidia-smi`; run vm-init.sh |
