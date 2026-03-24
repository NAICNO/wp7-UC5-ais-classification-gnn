# FAQ and Troubleshooting

```{objectives}
- Find answers to common questions about the GNN demonstrator
- Diagnose and resolve common installation and runtime errors
- Understand model behavior: early stopping, region force model, attention performance
- Know how to extend the demonstrator with new features or custom data
```

## Frequently Asked Questions

### What Python version do I need?

Python 3.10 or higher is required. The DGL library requires specific PyTorch/Python version combinations. You can check your version with:

```bash
python3 --version
```

### Can I train without a GPU?

Yes. The training script automatically detects GPU availability. CPU training is slower but fully functional. Set `--gpu -1` to force CPU mode even if a GPU is available:

```bash
python src/graph_classification/train_graph_classification_ais.py \
    --data_folder data/ --gpu -1 --epochs 50
```

### How long does training take?

Training time depends heavily on hardware:

| Hardware | Time per Epoch | 100 Epochs |
|----------|---------------|------------|
| CPU (4 cores) | ~2 minutes | ~3 hours |
| GPU (NVIDIA T4) | ~1 second | ~2 minutes |
| GPU (NVIDIA A100) | <0.5 second | <1 minute |

Early stopping typically triggers well before the maximum epoch count, so actual training time is often shorter.

### How does early stopping work?

Early stopping monitors the validation accuracy after each epoch. If the validation accuracy does not improve for `patience` consecutive epochs, training stops and the best model checkpoint is retained.

For example, with `--patience 20`:
- Epoch 30: validation accuracy reaches 94.2% (new best)
- Epochs 31-50: no improvement
- Epoch 50: training stops; the model from epoch 30 is used

The best model is saved based on the highest validation accuracy, not the final epoch. This prevents overfitting and ensures the saved model generalizes well.

### What is the region force model?

The region force model is a key architectural component that goes beyond standard GNN classification. Instead of directly using GNN output for classification, the pipeline uses an iterative refinement process:

1. An **initial GCN** produces a first classification estimate $u_0$
2. The **region force model** (GCN, GraphSAGE, or GAT) produces a correction signal $f$
3. These are combined iteratively: $u_{t+1} = \tanh(u_t + f \cdot \Delta t)$ for $T=3$ steps

This is inspired by neural ODE methods where the classification is refined over multiple "time steps" using a learned force field. The tanh activation keeps the output bounded while allowing the region force to push the classification toward the correct class over multiple iterations.

### Why does GAT underperform compared to GCN and GraphSAGE?

GAT achieves 93.1% accuracy at lr=0.01, compared to 94.4% for GCN and GraphSAGE. Several factors contribute:

1. **Chain graph regularity**: In a chain graph, every non-endpoint node has exactly 2 neighbors (plus self-loop). The attention mechanism is most beneficial when neighborhood sizes and structures vary, which is not the case here.

2. **Dropout**: GAT uses 0.5 dropout on both features and attention coefficients. On a small graph with only 12 nodes, this aggressive regularization can discard useful information.

3. **Learning rate sensitivity**: At lr=0.025, GAT drops to 86.3% while GCN and GraphSAGE remain above 94%. The attention weight computation introduces additional parameters that are harder to optimize at higher learning rates.

4. **Hidden dimension**: GAT uses 32 hidden dimensions (vs. 64 for GCN/GraphSAGE) because multi-head attention multiplies the effective dimension by the number of heads.

### What is the bootstrap index (bidx)?

The dataset includes 50 different train/val/test splits stored in `bidx_ts12.npy` with shape `(50, N)`. Each row defines a different partition:
- Value 1 = training sample
- Value 2 = validation sample
- Value 3 = test sample

Use `--bootstrap_index N` (0-49) to select a specific split. When not specified, a combined split is used. Running across multiple indices provides confidence intervals for reported accuracy.

### Can I add new features?

Yes, but it requires modifying the data pipeline. The current input format is `(N, 3, 12)` -- 3 features per time step. To add features:

1. **Prepare the data**: Create a new numpy array with shape `(N, K, 12)` where K is the new feature count
2. **Update the data files**: Save as `X_ts12.npy` (same filename, new shape)
3. **No model changes needed**: The model automatically reads `dim_nfeats` from the dataset, so it adapts to any feature dimension

Candidate additional features:
- **Acceleration**: Rate of speed change, useful for detecting speed-up/slow-down patterns
- **Heading change rate**: Angular velocity, complementary to curvature
- **Time of day**: Encoded as sin/cos components, captures diurnal fishing patterns
- **Depth at position**: Bathymetric data, since fishing often occurs at specific depth ranges

### How can I use this with my own AIS data?

To use the demonstrator with your own AIS data:

1. **Pre-process your AIS data** into fixed-length segments of 12 time steps
2. **Extract the three features** (velocity, distance to shore, curvature) for each time step
3. **Create numpy arrays**:
   - `X_ts12.npy`: shape `(N, 3, 12)`, float32
   - `y_ts12.npy`: shape `(N,)`, float32, values 0 or 1
   - `bidx_ts12.npy`: shape `(K, N)`, split indices (1=train, 2=val, 3=test)
4. **Place the files** in the `data/` directory
5. **Run training** as usual

For the bootstrap indices, you can create a simple random split:

```python
import numpy as np

N = len(your_labels)
bidx = np.zeros((1, N), dtype=int)
indices = np.random.permutation(N)
n_train = int(0.6 * N)
n_val = int(0.2 * N)

bidx[0, indices[:n_train]] = 1          # Training
bidx[0, indices[n_train:n_train+n_val]] = 2  # Validation
bidx[0, indices[n_train+n_val:]] = 3    # Test

np.save('data/bidx_ts12.npy', bidx)
```

## Common Issues

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'dgl'` | Run `./setup.sh` or activate the virtual environment: `source venv/bin/activate` |
| `ModuleNotFoundError: No module named 'graph_classification'` | Set PYTHONPATH: `export PYTHONPATH=$PYTHONPATH:$(pwd)/src` |
| `FileNotFoundError: data_folder does not exist` | Create `data/` directory and add the `.npy` files |
| `CUDA out of memory` | Reduce batch size: `--batch_size 100` |
| DGL version mismatch | Ensure PyTorch and DGL CUDA versions match; delete `venv/` and re-run `./setup.sh` |
| Jupyter kernel not found | Run `python -m ipykernel install --user --name=ais_dgl` |
| `RuntimeError: 0-in-degree nodes` | Self-loops should be added automatically; check that `dgl.add_self_loop()` is called |
| `ValueError: Raw data files are not set` | Ensure `.npy` files exist in the data folder and filenames match expected pattern |
| Training loss not decreasing | Try a lower learning rate (`--lrs "1e-3"`); check that data labels are correct |
| GPU detected but not used | Set `--gpu 0` explicitly; verify CUDA with `python -c "import torch; print(torch.cuda.is_available())"` |

## Performance Tips

### Speed Up Training

- **Use GPU**: Training is 100x+ faster on GPU. Even a T4 GPU reduces 3-hour CPU runs to 2 minutes.
- **Increase batch size**: Larger batches (1000-4000) reduce the number of gradient updates per epoch and improve GPU utilization. Adjust upward until GPU memory is fully utilized.
- **Reduce patience**: Setting `--patience 20` instead of 200 stops training sooner when the model has converged.
- **Pin memory**: Keep `--pin_memory True` (default) for faster CPU-to-GPU data transfer.

### Improve Accuracy

- **Try multiple learning rates**: The default script iterates over `5e-2, 3e-2, 1e-2`. Adding smaller rates like `5e-3, 1e-3` may help.
- **Increase hidden dimensions**: Use `--hidden 64` or `--hidden 128` for more model capacity.
- **Run multiple bootstrap splits**: Average results across 5-10 splits for more reliable comparisons.
- **Increase depth**: The models use 3 GNN layers by default. More layers capture longer-range dependencies but risk over-smoothing.

### Reduce Memory Usage

- **Lower batch size**: `--batch_size 100` uses less GPU memory at the cost of slower training.
- **Use CPU**: Set `--gpu -1` to avoid GPU memory constraints entirely.
- **Reduce hidden size**: `--hidden 16` uses less memory but may reduce accuracy.

## Getting Help

- NAIC project: [naic.no](https://naic.no)
- DGL documentation: [docs.dgl.ai](https://docs.dgl.ai)
- PyTorch documentation: [pytorch.org/docs](https://pytorch.org/docs)
- NORCE: [norceresearch.no](https://norceresearch.no)
- Repository issues: [GitHub Issues](https://github.com/NAICNO/wp7-UC5-ais-classification-gnn/issues)

```{keypoints}
- Python 3.10+ is required; GPU is optional but recommended (100x speedup)
- Early stopping saves the best model checkpoint based on validation accuracy
- The region force model iteratively refines classification using a neural ODE-inspired approach
- GAT underperforms due to chain graph regularity, aggressive dropout, and learning rate sensitivity
- New features can be added by changing the input array shape -- no model code changes needed
- Custom AIS data requires pre-processing into fixed-length segments with matching numpy formats
- Most runtime errors are resolved by activating the virtual environment and setting PYTHONPATH
- Batch size is the primary lever for trading memory usage against training speed
```
