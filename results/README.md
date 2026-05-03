# Experiment Results

This directory contains the raw NumPy result files from the 9-method comparative evaluation.

## File Naming Convention

Each `.npy` file follows the pattern: `{attack_type}__{aggregation_method}.npy`

### Attack Types

| Attack | Description |
|--------|-------------|
| `gaussian` | Byzantine clients submit Gaussian noise updates (primary threat model) |
| `label_flip` | Byzantine clients flip training labels before local training |
| `sleeper` | Byzantine clients behave honestly for early rounds, then attack |

### Aggregation Methods (7 baselines + 2 Active-Ledger variants)

| Method | Source |
|--------|--------|
| `FedAvg` | McMahan et al., AISTATS 2017 — Defenseless baseline |
| `Krum` | Blanchard et al., NeurIPS 2017 — Single closest client |
| `MultiKrum` | Blanchard et al., NeurIPS 2017 — Top-k closest, then FedAvg |
| `Median` | Yin et al., ICML 2018 — Coordinate-wise median |
| `TrimmedMean` | Yin et al., ICML 2018 — Coordinate-wise trimmed mean (β=0.2) |
| `Bulyan` | El Mhamdi et al., ICML 2018 — Iterative Krum → TrimmedMean |
| `PoC` | **Ours** — Proof-of-Contribution scoring, no diffusion |

### Diffusion-Augmented Methods

| File | Description |
|------|-------------|
| `diffusion__MultiKrum_BlindLDM.npy` | Multi-Krum + Blind 1D Latent Diffusion |
| `diffusion__PoC_TrustGated_LDM.npy` | **Active-Ledger Full System** — PoC + Trust-Gated LDM |

## Loading Results

```python
import numpy as np

results = np.load("results/experiment_tracker_npys/gaussian__MultiKrum.npy", allow_pickle=True).item()

# Available keys depend on the experiment, typically:
# - 'round_f1': per-round mean F1 scores
# - 'per_class_f1': per-round per-class F1 matrix
# - 'final_f1': final round F1 scores
```
