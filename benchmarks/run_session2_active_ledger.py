import os
import sys
import time
import threading
import torch
import torch.nn as nn
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.utils import load_config
from core.model import create_model
from core.client import FLClient
from core.blockchain import BlockchainManager, fetch_client_history
from core.server import start_approval_daemon, calculate_score
from benchmarks.run_robust_baselines import load_all_client_data
from core.robust_aggregation import fedavg_aggregate
from benchmarks.run_robust_baselines import _get_weights, _set_weights, evaluate_full, TeeLogger

NUM_ROUNDS         = 40
TOTAL_CLIENTS      = 10
NUM_MALICIOUS      = 2
DIFFUSION_STEPS    = 20
SYNTHETIC_QUANTITY = 200
CHECKPOINT_EVERY   = 10
CHECKPOINT_DIR     = Path("checkpoints/session2_active_ledger")
GANACHE_URL        = "http://127.0.0.1:8545"

def main():
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = CHECKPOINT_DIR / "session2_active_ledger.log"
    tee = TeeLogger(str(log_path))
    sys.stdout = tee
    
    print("=" * 70)
    print("SESSION 2: ACTIVE-LEDGER (POC + TRUST-GATED DIFFUSION)")
    print(f"Rounds: {NUM_ROUNDS} | Clients: {TOTAL_CLIENTS} ({NUM_MALICIOUS} Malicious)")
    print(f"Diffusion: {DIFFUSION_STEPS} steps | {SYNTHETIC_QUANTITY} samples/class")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device: {device}")
    
    config = load_config()
    loaders, val_loaders, sizes = load_all_client_data(config, device)
    
    # ── Blockchain Setup ─────────────────────────────────────────────────────
    print("\n[..] Connecting to Ganache and deploying daemon...")
    blockchain        = BlockchainManager(GANACHE_URL)
    blockchain_daemon = BlockchainManager(GANACHE_URL)
    eth_accounts      = list(blockchain.w3.eth.accounts)
    while len(eth_accounts) < TOTAL_CLIENTS:
        eth_accounts.append(blockchain.deployer)

    for cid in range(1, TOTAL_CLIENTS + 1):
        blockchain.set_synthetic_quota(cid, 10000)

    stop_daemon   = threading.Event()
    daemon_thread = threading.Thread(
        target=start_approval_daemon,
        args=(blockchain_daemon, eth_accounts, stop_daemon, 2.0, False), # blind_mode=False
        daemon=True
    )
    daemon_thread.start()

    # ── Pre-train diffusion model if needed ──────────────────────────────────
    from core.diffusion import ECGDiffusionGenerator, PRETRAINED_PATH
    if not PRETRAINED_PATH.exists():
        print(f"\n[..] Pre-training diffusion UNet (30 epochs)...")
        all_ecg, all_labels = [], []
        for loader in loaders:
            for X_batch, y_batch in loader:
                all_ecg.append(X_batch.numpy())
                all_labels.append(y_batch.numpy())
        all_ecg = np.concatenate(all_ecg, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        diff_device = "cuda" if torch.cuda.is_available() else "cpu"
        gen = ECGDiffusionGenerator()
        gen.train_on_data(all_ecg, all_labels, epochs=30, device=diff_device)
        gen.save_weights()
    else:
        print(f"\n[OK] Pre-trained diffusion weights found.")

    # ── Training ─────────────────────────────────────────────────────────────
    malicious_ids = set(range(TOTAL_CLIENTS - NUM_MALICIOUS, TOTAL_CLIENTS))
    global_model = create_model(config).to(device)
    
    round_f1s, round_lats = [], []
    per_class_f1, per_class_prec, per_class_recall, client_accs = [], [], [], []
    
    t_start = time.time()
    for rnd in range(1, NUM_ROUNDS + 1):
        t_rnd = time.time()
        print(f"\n--- Round {rnd:3d}/{NUM_ROUNDS} [Active-Ledger] ---")
        
        global_weights = _get_weights(global_model)
        results = []
        round_client_accs = []
        
        for cid in range(TOTAL_CLIENTS):
            is_mali = cid in malicious_ids
            client = FLClient(
                client_id=cid + 1,
                model=global_model,
                train_loader=loaders[cid],
                val_loader=val_loaders[cid],
                config=config,
                is_malicious=is_mali,
                blockchain_manager=blockchain, # PoC active
                enable_synthetic=True,
                diffusion_steps=DIFFUSION_STEPS,
                synthetic_quantity=SYNTHETIC_QUANTITY
            )
            w, n, metrics = client.fit(global_weights, config)
            acc = float(metrics["accuracy"])
            results.append((acc, w, n, cid))
            round_client_accs.append(acc)
            
            flag = " [BYZANTINE]" if is_mali else ""
            print(f"    Client {cid+1:2d}{flag}: local_acc={acc:.4f} samples={n}")
            
        client_accs.append(round_client_accs)
        
        # Log to chain
        for acc, w, n, cid in results:
            try:
                acc_int = int(acc * 10000)
                dummy_hash = bytes([cid % 256] * 32)
                tx = blockchain.contract.functions.logUpdate(
                    rnd, cid + 1, dummy_hash, n, acc_int
                ).transact({"from": eth_accounts[cid]})
                blockchain.w3.eth.wait_for_transaction_receipt(tx)
            except Exception as e:
                print(f"    [warn] on-chain log failed cid={cid+1}")

        # PoC Score + Top-7 selection
        scored = []
        for acc, w, n, cid in results:
            try:
                h = fetch_client_history(eth_accounts[cid], blockchain.contract, blockchain.w3)
                score = calculate_score(h)
            except Exception:
                score = 0.5
            scored.append((score, w, n))
            
        scored.sort(key=lambda x: x[0], reverse=True)
        top7 = scored[:7]
        print(f"  [Active-Ledger] Top-7 Scores: {[round(s,3) for s,_,_ in top7]}")
        
        all_weights = [w for _, w, _ in top7]
        all_sizes   = [n for _, _, n in top7]
        
        # Top-7 FedAvg Aggregation
        global_model = fedavg_aggregate(global_model, all_weights, all_sizes)
        
        # Eval
        f1_now, prec_now, rec_now, sup_now, report_now = evaluate_full(global_model, val_loaders, device)
        mean_f1 = float(np.mean(f1_now))
        t_elapsed = time.time() - t_rnd
        
        round_f1s.append(mean_f1)
        round_lats.append(t_elapsed)
        per_class_f1.append(f1_now.tolist())
        per_class_prec.append(prec_now.tolist())
        per_class_recall.append(rec_now.tolist())
        
        print(f"  Mean F1: {mean_f1:.4f} | Latency: {t_elapsed:.1f}s")
        print(f"  Per-class F1: {np.round(f1_now, 4)}")
        
        np.save(CHECKPOINT_DIR / "partial_results.npy", {
            "round_f1": round_f1s, "per_class_f1": per_class_f1, 
            "per_class_prec": per_class_prec, "per_class_recall": per_class_recall
        })
        
        if rnd % CHECKPOINT_EVERY == 0:
            torch.save(global_model.state_dict(), CHECKPOINT_DIR / f"model_round_{rnd}.pth")

    stop_daemon.set()

    # Final wrap up
    f1_final, prec_final, rec_final, sup_final, report_final = evaluate_full(global_model, val_loaders, device)
    torch.save(global_model.state_dict(), CHECKPOINT_DIR / "model_final.pth")
    with open(CHECKPOINT_DIR / "final_report.txt", "w") as f:
        f.write(report_final)
        
    print(f"\nFinal Mean F1: {float(np.mean(f1_final)):.4f}")
    print(f"Total Time: {(time.time()-t_start)/3600:.2f}h")
    print(report_final)

if __name__ == "__main__":
    main()
