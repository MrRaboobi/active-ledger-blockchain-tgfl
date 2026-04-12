"""
Phase 3.2 — Run 2B: Blind Diffusion (Ablation — Diffusion Without PoC)
======================================================================
All 10 clients (8 normal + 2 Byzantine) are aggregated every round.
Synthetic data generation via LDM is ENABLED, but the approval daemon
runs in 'blind' mode — every generation request is auto-approved
without PoC reputation gating.

This isolates the impact of the PoC gatekeeper by comparing:
    Run C  (Ledger-Guided)  =  PoC + Diffusion   ← full system
    Run 2B (Blind Diffusion) =  NO PoC + Diffusion ← this script
    Run 1A (Standard FedAvg) =  NO PoC + NO Diffusion

Parameters (mirrored from Run C):
    Rounds          : 40
    Clients         : 10 (8 Normal, 2 Byzantine)
    Learning Rate   : 0.001
    Local Epochs    : 3
    Weighting       : Clamped Inverse-Frequency (Floor=0.3, Cap=10.0)
    Diffusion Steps : 30
    Synthetic Qty   : 500

Usage (via main.py):
    python main.py --mode blind-ablation

Or directly:
    python benchmarks/run_2B_blind_diffusion.py
"""

import sys
import time
import threading
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.train_utils import load_client_data, create_data_loaders, train_epoch, evaluate
from core.model import create_model
from core.utils import load_config
from core.blockchain import BlockchainManager, fetch_client_history
from core.server import calculate_score, start_approval_daemon
from core.client import FLClient

# ── Run 2B constants (mirrored from Run C) ────────────────────────────────────
NUM_ROUNDS          = 40
NUM_NORMAL          = 8
NUM_MALICIOUS       = 2
TOTAL_CLIENTS       = 10
DIFFUSION_STEPS     = 30
SYNTHETIC_QUANTITY  = 500
GANACHE_URL         = "http://127.0.0.1:8545"
CHECKPOINT_DIR      = Path("checkpoints")
CHECKPOINT_EVERY    = 10
DIFFUSION_PRETRAIN_EPOCHS = 30

# ── 2B-specific overrides ─────────────────────────────────────────────────────
ENABLE_SYNTHETIC    = True    # Diffusion augmentation ON
USE_POC             = False   # Aggregate ALL 10 clients (no PoC filtering)
DAEMON_MODE         = "blind" # Auto-approve all synthetic requests (no PoC check)
RUN_TAG             = "2B_Blind_Diffusion"
RESULTS_FILE        = "checkpoints/results_2B_blind.npy"

# ── Utilities (identical to Run C) ────────────────────────────────────────────

def _get_weights(model):
    return [val.cpu().detach().numpy() for _, val in model.state_dict().items()]


def _set_weights(model, weights):
    state_dict = dict(zip(model.state_dict().keys(), [torch.tensor(w) for w in weights]))
    model.load_state_dict(state_dict, strict=True)


def _fedavg_aggregate(global_model, client_weights_list, sizes):
    total     = sum(sizes)
    global_sd = global_model.state_dict()
    param_keys = [k for k in global_sd if "num_batches_tracked" not in k]
    agg = {k: torch.zeros_like(v, dtype=torch.float32)
           for k, v in global_sd.items() if k in param_keys}

    for weights, size in zip(client_weights_list, sizes):
        factor = size / total
        tmp    = deepcopy(global_model)
        _set_weights(tmp, weights)
        tmp_sd = tmp.state_dict()
        for k in param_keys:
            agg[k] += tmp_sd[k].float() * factor

    new_sd = dict(global_sd)
    for k in param_keys:
        new_sd[k] = agg[k].to(global_sd[k].dtype)
    global_model.load_state_dict(new_sd)
    return global_model


def evaluate_f1_scores(global_model, val_loaders, device):
    global_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for loader in val_loaders:
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                outputs = global_model(X)
                preds   = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average=None,
                  labels=[0, 1, 2, 3, 4], zero_division=0)
    report = classification_report(all_labels, all_preds,
                                   target_names=["Normal", "LBBB", "RBBB", "APB", "PVC"],
                                   zero_division=0)
    return f1, report


def _save_checkpoint(global_model, rnd):
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    path = CHECKPOINT_DIR / f"global_model_2B_round_{rnd}.pth"
    torch.save(global_model.state_dict(), path)
    print(f"[CHECKPOINT] Saved → {path}")


def _load_all_client_data(config, device):
    """Load full (non-truncated) partitioned data for clinical evaluation."""
    partitioned_dir = Path(config["data"]["partitioned_dir"])
    batch_size      = config["training"]["batch_size"]
    loaders, val_loaders, sizes = [], [], []

    for cid in range(1, TOTAL_CLIENTS + 1):
        data = load_client_data(cid, str(partitioned_dir))
        tl, vl = create_data_loaders(
            data["X_train"], data["y_train"],
            data["X_val"],   data["y_val"],
            batch_size
        )
        loaders.append(tl)
        val_loaders.append(vl)
        sizes.append(len(data["y_train"]))
        print(f"  Client {cid:2d}: {sizes[-1]} train samples")

    return loaders, val_loaders, sizes


# ── Run 2B Main ───────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True

    # ── Device selection ─────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("=" * 70)
        print("⚠️  WARNING: No CUDA GPU detected. Falling back to CPU.")
        print("   With DIFFUSION_STEPS=30 and SYNTHETIC_QUANTITY=500,")
        print("   this run will likely take 48-96 hours on CPU hardware.")
        print("   Strongly recommended: RTX 3060+ / A100 / L4 GPU.")
        print("=" * 70)
    else:
        print(f"✅ CUDA GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    config = load_config()

    print("\n" + "=" * 70)
    print(f"PHASE 3.2 — RUN 2B: BLIND DIFFUSION (ABLATION)")
    print(f"  Rounds            : {NUM_ROUNDS}")
    print(f"  Clients           : {NUM_NORMAL} normal + {NUM_MALICIOUS} Byzantine = {TOTAL_CLIENTS}")
    print(f"  PoC Selection     : DISABLED (aggregating all {TOTAL_CLIENTS} clients)")
    print(f"  Synthetic Data    : ENABLED (blind approval — no PoC gating)")
    print(f"  Diffusion Steps   : {DIFFUSION_STEPS}")
    print(f"  Synthetic Qty     : {SYNTHETIC_QUANTITY}")
    print(f"  Learning Rate     : {config['model']['learning_rate']}")
    print(f"  Local Epochs      : {config['federated']['local_epochs']}")
    print(f"  Device            : {device}")
    print("=" * 70)

    # ── Blockchain setup ─────────────────────────────────────────────────────
    print("\n[..] Connecting to blockchain...")
    try:
        blockchain        = BlockchainManager(GANACHE_URL)
        blockchain_daemon = BlockchainManager(GANACHE_URL)
        eth_accounts      = list(blockchain.w3.eth.accounts)
        while len(eth_accounts) < TOTAL_CLIENTS:
            eth_accounts.append(blockchain.deployer)
        blockchain_ok = True
    except Exception as e:
        print(f"[WARN] Blockchain connection failed: {e}")
        print("[WARN] Continuing without on-chain logging (results still valid).")
        blockchain        = None
        blockchain_daemon = None
        eth_accounts      = [f"0x{'0'*40}"] * TOTAL_CLIENTS
        blockchain_ok     = False

    # ── Synthetic quota setup ────────────────────────────────────────────────
    if blockchain_ok:
        print("[..] Setting synthetic quotas for all clients...")
        for cid in range(1, TOTAL_CLIENTS + 1):
            blockchain.set_synthetic_quota(cid, 1_000_000)

    # ── Start BLIND approval daemon ──────────────────────────────────────────
    stop_daemon   = threading.Event()
    daemon_thread = None
    if blockchain_ok:
        daemon_thread = threading.Thread(
            target=start_approval_daemon,
            args=(blockchain_daemon, eth_accounts, stop_daemon, 2.0),
            kwargs={"mode": DAEMON_MODE},  # 'blind' — auto-approve everything
            daemon=True,
        )
        daemon_thread.start()

    # ── Data loading ─────────────────────────────────────────────────────────
    print("\n[..] Loading full clinical dataset (no truncation)...")
    loaders, val_loaders, sizes = _load_all_client_data(config, device)
    print(f"[OK] Loaded {TOTAL_CLIENTS} clients. Total train samples: {sum(sizes)}")

    # ── Pre-train diffusion model on real ECG data ────────────────────────────
    from core.diffusion import ECGDiffusionGenerator, PRETRAINED_PATH

    if not PRETRAINED_PATH.exists():
        print(f"\n[..] Pre-training diffusion UNet ({DIFFUSION_PRETRAIN_EPOCHS} epochs)...")
        all_ecg = []
        all_labels = []
        for loader in loaders:
            for X_batch, y_batch in loader:
                all_ecg.append(X_batch.numpy())
                all_labels.append(y_batch.numpy())
        all_ecg = np.concatenate(all_ecg, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        print(f"     Pooled {all_ecg.shape[0]} ECG signals for conditional diffusion training")

        diff_device = "cpu"
        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability(0)
            if cc >= (7, 0):
                diff_device = "cuda"

        gen = ECGDiffusionGenerator()
        gen.train_on_data(all_ecg, all_labels, epochs=DIFFUSION_PRETRAIN_EPOCHS, device=diff_device)
        gen.save_weights()
        print(f"[OK] Diffusion model pre-trained and saved to {PRETRAINED_PATH}")
    else:
        print(f"\n[OK] Pre-trained diffusion weights found: {PRETRAINED_PATH}")

    # ── Training loop ─────────────────────────────────────────────────────────
    malicious_ids = set(range(NUM_NORMAL, TOTAL_CLIENTS))
    global_model  = create_model(config).to(device)
    lr            = config["model"]["learning_rate"]
    criterion     = nn.CrossEntropyLoss()

    round_f1s    = []
    round_lats   = []
    per_class_f1 = []
    t_total = time.time()

    for rnd in range(1, NUM_ROUNDS + 1):
        t_rnd = time.time()
        print(f"\n--- Round {rnd:3d}/{NUM_ROUNDS} [{RUN_TAG}] ---")

        results = []
        global_weights = _get_weights(global_model)

        for cid in range(TOTAL_CLIENTS):
            is_malicious = cid in malicious_ids
            client = FLClient(
                client_id       = cid + 1,
                model           = global_model,
                train_loader    = loaders[cid],
                val_loader      = val_loaders[cid],
                config          = config,
                is_malicious    = is_malicious,
                blockchain_manager = blockchain,
                enable_synthetic   = ENABLE_SYNTHETIC,   # True
                diffusion_steps    = DIFFUSION_STEPS,
                synthetic_quantity = SYNTHETIC_QUANTITY,
            )
            w, n, metrics = client.fit(global_weights, config)
            results.append((float(metrics["accuracy"]), w, n, cid,
                            eth_accounts[cid] if blockchain_ok else None))

        # ── On-chain logging (best-effort) ───────────────────────────────────
        if blockchain_ok:
            for acc, w, n, cid, addr in results:
                try:
                    acc_int    = int(acc * 10000)
                    dummy_hash = bytes([cid % 256] * 32)
                    tx = blockchain.contract.functions.logUpdate(
                        rnd, cid + 1, dummy_hash, n, acc_int
                    ).transact({"from": addr})
                    blockchain.w3.eth.wait_for_transaction_receipt(tx)
                except Exception as e:
                    print(f"  [warn] on-chain log failed cid={cid+1}: {e}")

        # ── NO PoC selection → aggregate ALL clients ─────────────────────────
        all_weights = [w for _, w, _, _, _ in results]
        all_sizes   = [n for _, _, n, _, _ in results]
        global_model = _fedavg_aggregate(global_model, all_weights, all_sizes)

        # ── Evaluation & logging (same format as Run C) ──────────────────────
        f1_now, _ = evaluate_f1_scores(global_model, val_loaders, device)
        mean_f1   = float(np.mean(f1_now))
        t_elapsed = time.time() - t_rnd
        round_f1s.append(mean_f1)
        round_lats.append(t_elapsed)
        per_class_f1.append(f1_now.tolist())
        print(f"  Mean F1: {mean_f1:.4f}  |  Latency: {t_elapsed:.1f}s  |  Per-class F1: {np.round(f1_now, 3)}")

        # ── Incremental checkpointing (Kaggle crash protection) ──────────────
        CHECKPOINT_DIR.mkdir(exist_ok=True)
        np.save(RESULTS_FILE, {
            "round_mean_f1":  np.array(round_f1s),
            "round_latency":  np.array(round_lats),
            "per_class_f1":   np.array(per_class_f1),
            "final_f1":       f1_now,
        })

        if rnd % CHECKPOINT_EVERY == 0:
            _save_checkpoint(global_model, rnd)

    # ── Final mandatory checkpoint ────────────────────────────────────────────
    _save_checkpoint(global_model, NUM_ROUNDS)

    # ── Stop daemon ───────────────────────────────────────────────────────────
    stop_daemon.set()
    if daemon_thread is not None:
        daemon_thread.join()

    # ── Final evaluation ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"FINAL EVALUATION — {RUN_TAG}")
    f1_final, report = evaluate_f1_scores(global_model, val_loaders, device)
    print(report)
    print(f"Total runtime: {(time.time() - t_total) / 3600:.2f} hours")

    # ── Save final metrics ────────────────────────────────────────────────────
    np.save(RESULTS_FILE, {
        "round_mean_f1":  np.array(round_f1s),
        "round_latency":  np.array(round_lats),
        "per_class_f1":   np.array(per_class_f1),
        "final_f1":       f1_final,
    })

    # ── Results plot ──────────────────────────────────────────────────────────
    class_names = ["Normal", "LBBB", "RBBB", "APB", "PVC"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Per-class bar chart
    colors = ["#2C7BB6", "#D7191C", "#1A9641", "#FDAE61", "#762A83"]
    bars = ax1.bar(class_names, f1_final, color=colors, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, f1_final):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel("F1 Score", fontsize=11)
    ax1.set_title(f"Run 2B: Per-Class F1 Scores\n({NUM_ROUNDS} Rounds, Blind Diffusion — No PoC Gating)", fontsize=11)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # Training curve
    ax2.plot(range(1, NUM_ROUNDS + 1), round_f1s,
             color="#D7191C", linewidth=2, marker="o", markersize=3)
    ax2.set_xlabel("Round", fontsize=11)
    ax2.set_ylabel("Mean F1 Score", fontsize=11)
    ax2.set_title("Run 2B: Training Curve — Mean F1 vs Round", fontsize=11)
    ax2.grid(linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("results_2B_blind_diffusion.pdf", format="pdf", bbox_inches="tight", dpi=300)
    print("[OK] Saved: results_2B_blind_diffusion.pdf")
    print(f"[OK] Saved: {RESULTS_FILE}")
    print(f"[SUCCESS] Run 2B ({RUN_TAG}) Complete.")


if __name__ == "__main__":
    main()
