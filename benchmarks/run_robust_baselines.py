"""
benchmarks/run_robust_baselines.py
====================================
Phase 3.2 — Session 1: Byzantine-Robust Aggregation Baseline Comparison
========================================================================

Runs 7 methods sequentially, each for 40 FL rounds:

    A  FedAvg           Standard FedAvg (defenseless baseline)
    B  Krum             Single closest client by Euclidean distance (NeurIPS 2017)
    C  Multi-Krum       Top-7 closest clients, then FedAvg (NeurIPS 2017)
    D  Median           Coordinate-wise median across all clients (ICML 2018)
    E  TrimmedMean      Coordinate-wise trimmed mean, beta=0.2 (ICML 2018)
    F  Bulyan           Iterative Krum → TrimmedMean, f=1 (ICML 2018)
    G  PoC-Only         Active-Ledger PoC Top-7, NO diffusion (our method, ablation)

Design decisions:
-   Methods A–F use NO blockchain and NO diffusion.
    They differ ONLY in the aggregation function. This isolates aggregation.
-   Method G uses Ganache (started mid-run) for PoC scoring.
    Blockchain is only active from method G onwards.
-   All methods: same CNN-LSTM, same data splits, same seeds (42), same
    clamped class weights (floor=0.3, cap=10.0), same Byzantine attack
    (clients 9-10 return Gaussian noise).
-   Results saved incrementally so a Kaggle crash does not lose all data.

Usage (via main.py):
    python main.py --mode robust-baselines

Or directly:
    python benchmarks/run_robust_baselines.py

Change log:
    2026-04-18  Created for Phase 3.2 (ACISP reviewer response).
"""

import os
import sys
import time
import threading
from copy import deepcopy
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, classification_report,
    precision_recall_fscore_support,
)


# ─────────────────────────────────────────────────────────────────────────────
# Tee Logger — writes every print() to both stdout AND a .log file
# ─────────────────────────────────────────────────────────────────────────────

class TeeLogger:
    """
    Duplicates all stdout writes to a log file.
    Usage:
        sys.stdout = TeeLogger('session1.log')
    """
    def __init__(self, filepath):
        self._terminal = sys.__stdout__
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self._log = open(filepath, 'a', buffering=1, encoding='utf-8')

    def write(self, message):
        self._terminal.write(message)
        self._log.write(message)

    def flush(self):
        self._terminal.flush()
        self._log.flush()

    def close(self):
        self._log.close()
        sys.stdout = self._terminal

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.train_utils import load_client_data, create_data_loaders, train_epoch, evaluate
from core.model import create_model
from core.utils import load_config
from core.client import FLClient
from core.robust_aggregation import (
    fedavg_aggregate,
    krum_aggregate,
    multi_krum_aggregate,
    median_aggregate,
    trimmed_mean_aggregate,
    bulyan_aggregate,
)

# ── Constants (mirrored from Run C for a fair comparison) ─────────────────────
NUM_ROUNDS      = 40
NUM_NORMAL      = 8
NUM_MALICIOUS   = 2
TOTAL_CLIENTS   = 10
GANACHE_URL     = "http://127.0.0.1:8545"
CHECKPOINT_DIR  = Path("checkpoints")
CHECKPOINT_EVERY = 10
RUN_TAG         = "Session1_RobustBaselines"

# ── Aggregation method registry ───────────────────────────────────────────────
# Each entry: (display_name, function, kwargs)
# Methods A–F require no blockchain; method G requires Ganache.
METHODS = OrderedDict([
    ("A_FedAvg",      (fedavg_aggregate,       {})),
    ("B_Krum",        (krum_aggregate,         {"f": 2})),
    ("C_MultiKrum",   (multi_krum_aggregate,   {"f": 2, "k": 7})),
    ("D_Median",      (median_aggregate,       {})),
    ("E_TrimmedMean", (trimmed_mean_aggregate, {"beta": 0.2})),
    ("F_Bulyan",      (bulyan_aggregate,       {"f": 1})),
    # G_PoC_Only is handled separately (needs blockchain)
])

POC_METHOD_KEY = "G_PoC_Only"

# ── Weight helpers ────────────────────────────────────────────────────────────

def _get_weights(model):
    return [val.cpu().detach().numpy() for _, val in model.state_dict().items()]


def _set_weights(model, weights):
    state_dict = dict(
        zip(model.state_dict().keys(), [torch.tensor(w) for w in weights])
    )
    model.load_state_dict(state_dict, strict=True)


# ── Evaluation ────────────────────────────────────────────────────────────────

CLASS_NAMES = ["Normal", "LBBB", "RBBB", "APB", "PVC"]


def evaluate_full(global_model, val_loaders, device):
    """
    Full evaluation returning F1, precision, recall, support, and
    the sklearn classification_report string.

    Returns:
        f1        : np.ndarray shape (5,)  — per-class F1
        precision : np.ndarray shape (5,)  — per-class precision
        recall    : np.ndarray shape (5,)  — per-class recall
        support   : np.ndarray shape (5,)  — per-class sample count
        report    : str — full sklearn classification_report
    """
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

    f1 = f1_score(
        all_labels, all_preds, average=None,
        labels=[0, 1, 2, 3, 4], zero_division=0,
    )
    precision, recall, _, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=[0, 1, 2, 3, 4],
        zero_division=0,
    )
    report = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    return f1, precision, recall, support, report


def evaluate_f1_scores(global_model, val_loaders, device):
    """Thin wrapper kept for backwards compatibility."""
    f1, _, _, _, report = evaluate_full(global_model, val_loaders, device)
    return f1, report


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_client_data(config, device):
    """Load partitioned data for all 10 clients."""
    partitioned_dir = Path(config["data"]["partitioned_dir"])
    batch_size      = config["training"]["batch_size"]
    loaders, val_loaders, sizes = [], [], []

    for cid in range(1, TOTAL_CLIENTS + 1):
        data = load_client_data(cid, str(partitioned_dir))
        tl, vl = create_data_loaders(
            data["X_train"], data["y_train"],
            data["X_val"],   data["y_val"],
            batch_size,
        )
        loaders.append(tl)
        val_loaders.append(vl)
        sizes.append(len(data["y_train"]))
        print(f"  Client {cid:2d}: {sizes[-1]:6d} train samples")

    return loaders, val_loaders, sizes


# ── Single-method training loop ───────────────────────────────────────────────

def run_one_method(
    method_name, agg_fn, agg_kwargs,
    global_model_init, loaders, val_loaders, sizes,
    device, config, malicious_ids,
    blockchain=None,
):
    """
    Run 40 FL rounds for one aggregation method.

    Args:
        method_name      : string label (e.g. "A_FedAvg")
        agg_fn           : aggregation function from robust_aggregation.py
        agg_kwargs       : extra keyword arguments for agg_fn
        global_model_init: freshly initialized model (same weights every time)
        loaders          : training DataLoaders per client
        val_loaders      : validation DataLoaders per client
        sizes            : training sample counts per client
        device           : torch.device
        config           : config dict
        malicious_ids    : set of 0-based client indices that are Byzantine
        blockchain       : BlockchainManager or None (only needed for PoC-Only)

    Returns:
        dict with keys: round_f1, round_latency, per_class_f1, final_f1
    """
    print("\n" + "=" * 70)
    print(f"METHOD: {method_name}")
    print("=" * 70)

    # Fresh model with same weights as initial (reproducible)
    global_model = deepcopy(global_model_init)
    global_model.to(device)

    round_f1s        = []
    round_lats       = []
    per_class_f1     = []   # shape: (n_rounds, 5)
    per_class_prec   = []   # shape: (n_rounds, 5)
    per_class_recall = []   # shape: (n_rounds, 5)
    client_accs      = []   # shape: (n_rounds, n_clients)
    t_method         = time.time()

    for rnd in range(1, NUM_ROUNDS + 1):
        t_rnd = time.time()
        print(f"\n  --- Round {rnd:3d}/{NUM_ROUNDS} [{method_name}] ---")

        global_weights  = _get_weights(global_model)
        results         = []
        round_client_accs = []

        # ── Each client trains locally (NO synthetic data for A–F) ───────────
        for cid in range(TOTAL_CLIENTS):
            is_malicious = cid in malicious_ids

            client = FLClient(
                client_id          = cid + 1,
                model              = global_model,
                train_loader       = loaders[cid],
                val_loader         = val_loaders[cid],
                config             = config,
                is_malicious       = is_malicious,
                blockchain_manager = blockchain,       # None for A–F
                enable_synthetic   = False,            # No diffusion in Session 1
                diffusion_steps    = 0,
                synthetic_quantity = 0,
            )
            w, n, metrics = client.fit(global_weights, config)
            acc = float(metrics["accuracy"])
            results.append((acc, w, n, cid))
            round_client_accs.append(acc)

            flag = " [BYZANTINE]" if is_malicious else ""
            print(f"    Client {cid+1:2d}{flag}: local_acc={acc:.4f}  "
                  f"n_samples={n}")

        client_accs.append(round_client_accs)

        # ── PoC-Only (method G): on-chain log + Top-7 selection ──────────────
        if blockchain is not None:
            from core.blockchain import fetch_client_history
            from core.server import calculate_score

            eth_accounts = list(blockchain.w3.eth.accounts)
            while len(eth_accounts) < TOTAL_CLIENTS:
                eth_accounts.append(blockchain.deployer)

            # Log all updates on-chain (best effort)
            for acc, w, n, cid in results:
                try:
                    acc_int    = int(acc * 10000)
                    dummy_hash = bytes([cid % 256] * 32)
                    tx = blockchain.contract.functions.logUpdate(
                        rnd, cid + 1, dummy_hash, n, acc_int
                    ).transact({"from": eth_accounts[cid]})
                    blockchain.w3.eth.wait_for_transaction_receipt(tx)
                except Exception as e:
                    print(f"    [warn] on-chain log failed cid={cid+1}: {e}")

            # Score clients by PoC and take Top-7
            scored = []
            poc_scores_log = []
            for acc, w, n, cid in results:
                try:
                    history = fetch_client_history(
                        eth_accounts[cid],
                        blockchain.contract,
                        blockchain.w3,
                    )
                    score = calculate_score(history)
                except Exception as e:
                    print(f"    [warn] PoC fetch failed cid={cid+1}: {e}. "
                          f"Using neutral score 0.5.")
                    score = 0.5
                scored.append((score, w, n))
                poc_scores_log.append((cid + 1, score))

            scored.sort(key=lambda x: x[0], reverse=True)
            top7 = scored[:7]

            print(f"  [PoC-Only] All PoC scores: "
                  f"{[(cid, round(s, 3)) for cid, s in poc_scores_log]}")
            print(f"  [PoC-Only] Top-7 selected | "
                  f"Scores: {[round(s, 3) for s, _, _ in top7]}")

            sel_weights = [w for _, w, _ in top7]
            sel_sizes   = [n for _, _, n in top7]
            global_model = fedavg_aggregate(global_model, sel_weights, sel_sizes)

        else:
            # ── Methods A–F: pure aggregation, all 10 clients ────────────────
            all_weights = [w for _, w, _, _ in results]
            all_sizes   = [n for _, _, n, _ in results]
            global_model = agg_fn(global_model, all_weights, all_sizes,
                                  **agg_kwargs)

        # ── Full evaluation (F1 + precision + recall + report) ───────────────
        f1_now, prec_now, rec_now, sup_now, report_now = \
            evaluate_full(global_model, val_loaders, device)
        mean_f1   = float(np.mean(f1_now))
        t_elapsed = time.time() - t_rnd

        round_f1s.append(mean_f1)
        round_lats.append(t_elapsed)
        per_class_f1.append(f1_now.tolist())
        per_class_prec.append(prec_now.tolist())
        per_class_recall.append(rec_now.tolist())

        print(f"  Mean F1: {mean_f1:.4f}  |  Latency: {t_elapsed:.1f}s")
        print(f"  Per-class F1       : {np.round(f1_now, 4)}")
        print(f"  Per-class Precision: {np.round(prec_now, 4)}")
        print(f"  Per-class Recall   : {np.round(rec_now, 4)}")
        print(f"  Support            : {sup_now}")

        # ── Incremental checkpoint (Kaggle crash protection) ──────────────────
        CHECKPOINT_DIR.mkdir(exist_ok=True)
        np.save(
            CHECKPOINT_DIR / f"partial_{method_name}.npy",
            {
                "round_f1":        np.array(round_f1s),
                "round_lat":       np.array(round_lats),
                "per_class_f1":    np.array(per_class_f1),
                "per_class_prec":  np.array(per_class_prec),
                "per_class_recall":np.array(per_class_recall),
                "client_accs":     np.array(client_accs),
                "rounds_done":     rnd,
            },
        )

        if rnd % CHECKPOINT_EVERY == 0:
            ckpt = CHECKPOINT_DIR / f"model_{method_name}_round_{rnd}.pth"
            torch.save(global_model.state_dict(), ckpt)
            print(f"  [CHECKPOINT] Model saved → {ckpt}")

    # ── Final full evaluation with classification report ──────────────────────
    f1_final, prec_final, rec_final, sup_final, report_final = \
        evaluate_full(global_model, val_loaders, device)

    # Save final model
    final_ckpt = CHECKPOINT_DIR / f"model_{method_name}_final.pth"
    torch.save(global_model.state_dict(), final_ckpt)

    # Save classification report to text file
    report_path = CHECKPOINT_DIR / f"report_{method_name}.txt"
    with open(report_path, "w") as rf:
        rf.write(f"Method: {method_name}\n")
        rf.write(f"Rounds: {NUM_ROUNDS}\n")
        rf.write(f"Final Mean F1: {float(np.mean(f1_final)):.4f}\n")
        rf.write("\n" + "=" * 50 + "\n")
        rf.write("FINAL CLASSIFICATION REPORT\n")
        rf.write("=" * 50 + "\n")
        rf.write(report_final)
        rf.write("\n" + "=" * 50 + "\n")
        rf.write("PER-ROUND MEAN F1:\n")
        for i, v in enumerate(round_f1s, 1):
            rf.write(f"  Round {i:3d}: {v:.4f}\n")
    print(f"  [REPORT] Saved → {report_path}")

    elapsed_h = (time.time() - t_method) / 3600
    print(f"\n  [{method_name}] Finished. Total time: {elapsed_h:.2f}h")
    print(f"  [{method_name}] Final Mean F1: {float(np.mean(f1_final)):.4f}")
    print(f"\n  FINAL CLASSIFICATION REPORT [{method_name}]:")
    print(report_final)

    return {
        # Round-level metrics
        "round_f1":         np.array(round_f1s),
        "round_lat":        np.array(round_lats),
        "per_class_f1":     np.array(per_class_f1),     # (40, 5)
        "per_class_prec":   np.array(per_class_prec),   # (40, 5)
        "per_class_recall": np.array(per_class_recall), # (40, 5)
        "client_accs":      np.array(client_accs),      # (40, 10)
        # Final-round metrics
        "final_f1":         f1_final,                   # (5,)
        "final_precision":  prec_final,                  # (5,)
        "final_recall":     rec_final,                   # (5,)
        "final_support":    sup_final,                   # (5,)
        "final_report":     report_final,                # str
        "elapsed_h":        elapsed_h,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def generate_plots(all_results):
    """
    Generate two-panel comparison figure:
        Left  : Grouped bar chart of per-class F1 (final round) for all methods
        Right : Training curve (Mean F1 vs Round) for all methods

    Also generates a LaTeX-ready comparison table.
    """
    class_names  = ["Normal", "LBBB", "RBBB", "APB", "PVC"]
    method_names = list(all_results.keys())
    n_methods    = len(method_names)

    # ── Colour palette (colourblind-safe) ─────────────────────────────────────
    colors = [
        "#4E79A7",   # A: FedAvg (steel blue)
        "#F28E2B",   # B: Krum (orange)
        "#E15759",   # C: Multi-Krum (red)
        "#76B7B2",   # D: Median (teal)
        "#59A14F",   # E: TrimmedMean (green)
        "#EDC948",   # F: Bulyan (yellow)
        "#B07AA1",   # G: PoC-Only (purple)
    ][:n_methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "Phase 3.2 — Session 1: Byzantine-Robust Aggregation Comparison\n"
        "(40 Rounds, 10 Clients, 2 Byzantine, No Diffusion)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    # ── Left panel: grouped bar chart ─────────────────────────────────────────
    x          = np.arange(len(class_names))
    bar_width  = 0.9 / n_methods
    offsets    = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods)

    for idx, (name, color) in enumerate(zip(method_names, colors)):
        f1_vals = all_results[name]["final_f1"]
        bars    = ax1.bar(
            x + offsets[idx] * bar_width, f1_vals,
            width=bar_width * 0.9, label=name, color=color,
            edgecolor="white", linewidth=0.5,
        )
        # Value labels on bars
        for bar, val in zip(bars, f1_vals):
            if val > 0.05:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center", va="bottom", fontsize=6, fontweight="bold",
                )

    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, fontsize=11)
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("F1 Score", fontsize=11)
    ax1.set_title("Per-Class F1 Scores (Final Round)", fontsize=11)
    ax1.axhline(y=0.75, color="red", linestyle="--", alpha=0.5,
                label="Clinical Threshold (0.75)")
    ax1.legend(fontsize=7, loc="upper right")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # ── Right panel: training curves ──────────────────────────────────────────
    for name, color in zip(method_names, colors):
        rnd_f1 = all_results[name]["round_f1"]
        ax2.plot(
            range(1, len(rnd_f1) + 1), rnd_f1,
            label=name, color=color, linewidth=2, marker="o", markersize=3,
        )

    ax2.set_xlabel("Round", fontsize=11)
    ax2.set_ylabel("Mean F1 Score", fontsize=11)
    ax2.set_title("Training Curve — Mean F1 vs Round", fontsize=11)
    ax2.axhline(y=0.75, color="red", linestyle="--", alpha=0.5,
                label="Clinical Threshold (0.75)")
    ax2.axhline(y=0.89, color="black", linestyle=":", alpha=0.7,
                label="Active-Ledger Full (0.89)")
    ax2.legend(fontsize=8)
    ax2.grid(linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("session1_robust_comparison.pdf",
                format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig("session1_robust_comparison.png",
                format="png", bbox_inches="tight", dpi=150)
    print("[OK] Saved: session1_robust_comparison.pdf / .png")


def generate_latex_table(all_results):
    """Print a LaTeX-ready comparison table to stdout and save to file."""
    class_names   = ["Normal", "LBBB", "RBBB", "APB", "PVC"]
    method_labels = {
        "A_FedAvg":      "FedAvg (no defense)",
        "B_Krum":        "Krum (f=2)",
        "C_MultiKrum":   "Multi-Krum (f=2, k=7)",
        "D_Median":      "Coordinate-wise Median",
        "E_TrimmedMean": "Trimmed Mean ($\\beta=0.2$)",
        "F_Bulyan":      "Bulyan (f=1)",
        "G_PoC_Only":    "\\textbf{Active-Ledger PoC-Only}",
    }

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Phase 3.2 Session 1 --- Byzantine-Robust Aggregation Comparison}",
        r"\label{tab:robust_baselines}",
        r"\small",
        r"\begin{tabular}{@{}lcccccc@{}}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Normal} & \textbf{LBBB} & "
        r"\textbf{RBBB} & \textbf{APB} & \textbf{PVC} & \textbf{Mean F1} \\",
        r"\midrule",
    ]

    for key, label in method_labels.items():
        if key not in all_results:
            continue
        f1 = all_results[key]["final_f1"]
        mean_f1 = float(np.mean(f1))
        row = f"{label} & " + " & ".join(f"{v:.3f}" for v in f1) \
              + f" & {mean_f1:.3f} \\\\"
        lines.append(row)

    # Add Active-Ledger full system row from saved Run C data
    lines.append(r"\midrule")
    lines.append(
        r"\textbf{Active-Ledger (PoC + Diffusion)} & "
        r"0.970 & 0.968 & 0.915 & 0.654 & 0.933 & \textbf{0.889} \\"
    )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    table_str = "\n".join(lines)
    print("\n" + "=" * 70)
    print("LaTeX TABLE:")
    print("=" * 70)
    print(table_str)

    with open("session1_latex_table.tex", "w") as f:
        f.write(table_str)
    print("\n[OK] Saved: session1_latex_table.tex")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True

    # ── Start logging to file ─────────────────────────────────────────────────
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    log_path = CHECKPOINT_DIR / "session1_full.log"
    tee = TeeLogger(str(log_path))
    sys.stdout = tee
    print(f"[LOG] All output is being saved to: {log_path}")
    print(f"[LOG] Session started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  No GPU detected. Baselines A–F will be slow but can run on CPU.")
        print("   Method G (PoC-Only) may take ~4–6h on CPU. Consider GPU.")

    config          = load_config()
    malicious_ids   = set(range(NUM_NORMAL, TOTAL_CLIENTS))   # clients 8, 9 (0-based)
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("PHASE 3.2 — SESSION 1: ROBUST BASELINE COMPARISON")
    print(f"  Rounds            : {NUM_ROUNDS}")
    print(f"  Clients           : {NUM_NORMAL} honest + {NUM_MALICIOUS} Byzantine")
    print(f"  Methods           : {len(METHODS) + 1} (A–G)")
    print(f"  Synthetic Data    : DISABLED for all methods in Session 1")
    print(f"  Device            : {device}")
    print("=" * 70)

    # ── Load data once (shared across all A–F methods) ───────────────────────
    print("\n[..] Loading client data...")
    loaders, val_loaders, sizes = load_all_client_data(config, device)
    total_samples = sum(sizes)
    print(f"[OK] Loaded {TOTAL_CLIENTS} clients | Total train samples: {total_samples}")

    # ── Initialize model once with a fixed seed (same starting point) ─────────
    torch.manual_seed(42)
    np.random.seed(42)
    global_model_init = create_model(config)
    global_model_init.to(device)
    init_weights = _get_weights(global_model_init)   # save for reuse
    print("[OK] Global model initialized (seed=42). Will be reset for each method.")

    # ── Run methods A–F (no blockchain) ──────────────────────────────────────
    all_results = {}
    t_total = time.time()

    for method_key, (agg_fn, agg_kwargs) in METHODS.items():
        # Reset model to identical initial weights for each method
        torch.manual_seed(42)
        np.random.seed(42)
        fresh_model = create_model(config)
        fresh_model.to(device)
        _set_weights(fresh_model, init_weights)

        result = run_one_method(
            method_name      = method_key,
            agg_fn           = agg_fn,
            agg_kwargs       = agg_kwargs,
            global_model_init = fresh_model,
            loaders          = loaders,
            val_loaders      = val_loaders,
            sizes            = sizes,
            device           = device,
            config           = config,
            malicious_ids    = malicious_ids,
            blockchain       = None,
        )
        all_results[method_key] = result

        # Save results after every method (Kaggle crash protection)
        np.save(
            CHECKPOINT_DIR / "robust_comparison.npy",
            all_results,
        )
        print(f"  [SAVED] robust_comparison.npy updated after {method_key}")

    # ── Method G: PoC-Only (requires Ganache) ────────────────────────────────
    print("\n" + "=" * 70)
    print("METHOD G: Active-Ledger PoC-Only (requires blockchain)")
    print("=" * 70)

    blockchain = None
    try:
        from core.blockchain import BlockchainManager
        blockchain = BlockchainManager(GANACHE_URL)
        print(f"[OK] Connected to Ganache at {GANACHE_URL}")
    except Exception as e:
        print(f"[WARN] Could not connect to blockchain: {e}")
        print("[WARN] Method G will run WITHOUT PoC scoring (falls back to FedAvg-all).")
        print("[WARN] Start Ganache before running if you want proper PoC-Only results.")

    torch.manual_seed(42)
    np.random.seed(42)
    fresh_model_g = create_model(config)
    fresh_model_g.to(device)
    _set_weights(fresh_model_g, init_weights)

    result_g = run_one_method(
        method_name       = POC_METHOD_KEY,
        agg_fn            = fedavg_aggregate,          # PoC path bypasses this
        agg_kwargs        = {},
        global_model_init = fresh_model_g,
        loaders           = loaders,
        val_loaders       = val_loaders,
        sizes             = sizes,
        device            = device,
        config            = config,
        malicious_ids     = malicious_ids,
        blockchain        = blockchain,                 # enables PoC branch
    )
    all_results[POC_METHOD_KEY] = result_g

    # Final save
    np.save(CHECKPOINT_DIR / "robust_comparison.npy", all_results)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = (time.time() - t_total) / 3600
    print("\n" + "=" * 70)
    print(f"SESSION 1 COMPLETE — Total runtime: {total_elapsed:.2f} hours")
    print("=" * 70)
    print(f"\n{'Method':<20} {'Normal':>7} {'LBBB':>7} {'RBBB':>7} "
          f"{'APB':>7} {'PVC':>7} {'Mean F1':>8}")
    print("-" * 70)
    for name, res in all_results.items():
        f1 = res["final_f1"]
        mean_f1 = float(np.mean(f1))
        print(f"{name:<20} "
              + " ".join(f"{v:7.3f}" for v in f1)
              + f" {mean_f1:8.3f}")

    # Reference row
    print("-" * 70)
    print(f"{'(Run C: Full System)':<20} "
          "  0.970   0.968   0.915   0.654   0.933    0.889  [already done]")

    # ── Generate plots and LaTeX table ────────────────────────────────────────
    generate_plots(all_results)
    generate_latex_table(all_results)

    print("\n✅ Session 1 complete. Files saved:")
    print("   checkpoints/robust_comparison.npy  — all raw results")
    print("   session1_robust_comparison.pdf      — comparison figure")
    print("   session1_robust_comparison.png      — PNG version")
    print("   session1_latex_table.tex            — LaTeX table for paper")
    print("\nNext step: tell me which method (B–F) performed best.")
    print("We will add diffusion to that winner in Session 2.")


if __name__ == "__main__":
    main()
