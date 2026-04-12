# Phase 3.1: Deep Compute Execution Guide

> **Objective:** Run the flagship 40-round Ledger-Guided Latent Diffusion clinical evaluation on a GPU-equipped machine to produce publication-quality F1-score benchmarks.

---

## 1. Prerequisites & Hardware

### Minimum Requirements
| Component | Minimum | Recommended |
| :--- | :--- | :--- |
| **GPU** | RTX 3060 (12 GB VRAM) | NVIDIA A100 / L4 / H100 |
| **RAM** | 16 GB | 32 GB+ |
| **Storage** | 10 GB free | 50 GB (SSD preferred) |
| **OS** | Ubuntu 22.04 / Windows 11 | Ubuntu 22.04 |
| **Python** | 3.10+ | 3.11 |
| **CUDA** | 11.8+ | 12.x |

> [!CAUTION]
> Running `--mode clinical-compute` on **CPU only** will take an estimated **48–96 hours** due to 50 diffusion inference steps × 500 synthetic samples × multiple classes × 10 clients × 40 rounds. **A GPU is mandatory for practical execution.**

### Software dependencies
```bash
pip install -r requirements.txt
# Verify CUDA is available:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 2. Step-by-Step Execution Pipeline

Perform all commands from the **repository root**.

### Step 1 — Start Ganache (Local Blockchain)
```bash
npx ganache -p 8545
```
Keep this terminal running in the background for the entire experiment.

### Step 2 — Deploy Smart Contract
```bash
python core/deploy_contract.py
```
This writes the deployed contract address to `contracts/deployed_contract.json`. Run once per fresh Ganache instance.

### Step 3 — Data Preparation (Run once)
```bash
python core/download_data.py       # Downloads 15 MIT-BIH records (~120 MB)
python core/preprocess_data.py     # Segments signals, extracts features
python core/partition_data.py      # Dirichlet-partitions data across 10 clients
```
> [!NOTE]
> Partitioned data is saved to `data/partitioned/`. Once generated, these three steps do **not** need to be repeated.

### Step 4 — Launch the Clinical Evaluation
```bash
python main.py --mode clinical-compute
```
The script now performs the following automatically:
1. **Diffusion pre-training** — Pools all client ECG data and trains the 1D-UNet for 30 epochs using DDPM noise-prediction. Saves weights to `checkpoints/diffusion_pretrained.pth`. Skips if weights already exist.
2. **40-round FL training** — Each client uses class-weighted loss and blockchain-governed diffusion augmentation.
3. **Final evaluation** — Per-class F1 scores, training curve, and PDF chart.

### Expected Terminal Output (first few rounds)
```
✅ CUDA GPU detected: NVIDIA RTX 3060
   VRAM: 12.0 GB

[..] Pre-training diffusion UNet (30 epochs)...
     Pooled 22335 ECG signals for diffusion training
  [DIFFUSION] Epoch 1/30   Loss: 0.982145
  [DIFFUSION] Epoch 5/30   Loss: 0.431207
  [DIFFUSION] Epoch 30/30  Loss: 0.089234
[OK] Diffusion model pre-trained and saved to checkpoints/diffusion_pretrained.pth

--- Round   1/50 ---
  Mean F1: 0.4832  |  Latency: 48.3s  |  Per-class F1: [0.812 0.412 0.391 0.350 0.412]
...
[CHECKPOINT] Saved → checkpoints/global_model_round_10.pth
```

---

## 3. Hyperparameter Tuning Guide

All constants are defined at the top of `benchmarks/run_final_clinical_evaluation.py`:

| Parameter | Default | Effect |
| :--- | :---: | :--- |
| `NUM_ROUNDS` | `50` | More rounds → higher accuracy, longer runtime (linear) |
| `DIFFUSION_STEPS` | `50` | More steps → higher fidelity synthetic ECGs, slower generation (linear) |
| `SYNTHETIC_QUANTITY` | `500` | More samples → better class balance, more GPU memory required |
| `TOP_K` | `7` | Higher → less Byzantine resistance; lower → more conservative aggregation |
| `CHECKPOINT_EVERY` | `10` | Save frequency; lower value = more recovery points |
| `DIFFUSION_PRETRAIN_EPOCHS` | `30` | Epochs to pre-train UNet on real ECG data before FL begins |

### Time Estimates (RTX 3060, full MIT-BIH dataset)
| Rounds | Diffusion Steps | Qty | Estimated Time |
| :---: | :---: | :---: | :--- |
| 50 | 50 | 500 | ~6–10 hours (+ ~15 min diffusion pre-training) |
| 20 | 20 | 200 | ~1.5–2.5 hours |
| 5  | 5  | 50  | ~15–25 minutes |

---

## 4. Checkpointing & Recovery

Model checkpoints are saved to `checkpoints/` every `CHECKPOINT_EVERY` rounds (default: every 10 rounds), plus a mandatory final save at Round 50.

### Checkpoint files produced
```
checkpoints/
  diffusion_pretrained.pth    ← pre-trained UNet (generated before FL)
  global_model_round_10.pth
  global_model_round_20.pth
  global_model_round_30.pth
  global_model_round_40.pth
  global_model_round_50.pth   ← final model
  final_f1_scores.npy         ← per-class F1 array [5]
  round_mean_f1.npy           ← training curve array [50]
```

### Resuming from a checkpoint
```python
import torch
from core.model import create_model
from core.utils import load_config

config = load_config()
model  = create_model(config)
model.load_state_dict(torch.load("checkpoints/global_model_round_30.pth"))
model.eval()
```

### Output Artefacts
| File | Description |
| :--- | :--- |
| `final_clinical_f1_scores.pdf` | Per-class F1 bar chart + training curve (publication-ready) |
| `checkpoints/final_f1_scores.npy` | Raw NumPy array of final per-class F1 scores |
| `checkpoints/round_mean_f1.npy` | Mean F1 training curve across 40 rounds |

---

## 5. Key Design Decisions

### Class-Weighted Loss
The `FLClient.fit()` method computes inverse-frequency class weights from each client's local data distribution. This ensures minority classes (LBBB, APB, PVC) contribute proportionally more to training gradients, preventing the model from defaulting to the majority class.

### Diffusion Pre-Training
Before FL begins, the script pools all client ECG signals and trains the 1D-UNet using standard DDPM (add noise → predict noise → MSE loss). This ensures the diffusion model generates realistic ECG patterns rather than random noise. Weights are auto-loaded by `ECGDiffusionGenerator.__init__()`.

---

## 6. Troubleshooting

| Issue | Fix |
| :--- | :--- |
| `ModuleNotFoundError: core` | Run from the repo root, not from `benchmarks/` |
| `ConnectionError: Cannot connect to Ganache` | Verify `npx ganache -p 8545` is running |
| `FileNotFoundError: deployed_contract.json` | Re-run `python core/deploy_contract.py` |
| `CUDA out of memory` | Reduce `SYNTHETIC_QUANTITY` to 100 or `DIFFUSION_STEPS` to 20 |
| `CUDA: no kernel image` | Use T4 GPU instead of P100 (cc≥7.0 required) |
| Daemon flooding logs | Normal for malicious clients; they are correctly rejected by PoC |
| Diffusion loss not decreasing | Increase `DIFFUSION_PRETRAIN_EPOCHS` to 50 |
