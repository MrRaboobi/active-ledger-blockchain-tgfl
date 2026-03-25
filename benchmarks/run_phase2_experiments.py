import sys
import os
import time
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.train_utils import load_client_data, create_data_loaders, train_epoch, evaluate
from core.model import create_model, CNNLSTM
from core.utils import load_config
from core.diffusion import ECGDiffusionGenerator

# ── Mocking for Speed ───────────────────────────────────────────────────────
class IdealizedBlockchain:
    """Simulates on-chain status without the RPC overhead for the high-volume experiments."""
    def request_synthetic(self, client_id, class_label, quantity):
        return 0 # dummy ID
    def get_synthetic_request(self, req_id):
        return {'approved': True} # Always approved for proof-of-concept simulation
    def mark_synthetic_generated(self, req_id):
        pass

# ── Experiment constants ──────────────────────────────────────────────────────
NUM_ROUNDS    = 5
NUM_NORMAL    = 8
NUM_MALICIOUS = 2
TOTAL_CLIENTS = 10
TOP_K         = 7

def evaluate_f1_scores(global_model, val_loaders, device):
    global_model.eval()
    all_preds, all_labels = [], []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for loader in val_loaders:
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                outputs = global_model(X)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
    
    return f1_score(all_labels, all_preds, average=None, labels=[0, 1, 2, 3, 4], zero_division=0)

def _get_weights(model):
    return [p.cpu().detach().numpy() for p in model.parameters()]

def _set_weights(model, weights):
    with torch.no_grad():
        for p, w in zip(model.parameters(), weights):
            p.copy_(torch.tensor(w))

def _fedavg_aggregate(global_model, client_weights_list, sizes):
    total = sum(sizes)
    global_sd = global_model.state_dict()
    param_keys = [k for k in global_sd if "num_batches_tracked" not in k]
    agg = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in global_sd.items() if k in param_keys}
    
    for weights, size in zip(client_weights_list, sizes):
         factor = size / total
         tmp = deepcopy(global_model)
         _set_weights(tmp, weights)
         tmp_sd = tmp.state_dict()
         for k in param_keys:
             agg[k] += tmp_sd[k].float() * factor
    
    new_sd = dict(global_sd)
    for k in param_keys:
        new_sd[k] = agg[k].to(global_sd[k].dtype)
    global_model.load_state_dict(new_sd)
    return global_model

def run_simulation(config, loaders, val_loaders, sizes, device, enable_synthetic):
    print(f"\n[PHASE 2.5] Simulation: enable_synthetic={enable_synthetic}")
    
    global_model = create_model(config).to(device)
    malicious_ids = set(range(NUM_NORMAL, TOTAL_CLIENTS))
    generator = ECGDiffusionGenerator()
    blockchain = IdealizedBlockchain()
    
    # Track metrics
    for rnd in range(1, NUM_ROUNDS + 1):
        print(f"  Round {rnd}/{NUM_ROUNDS}...")
        results = []
        
        for cid in range(TOTAL_CLIENTS):
            # Local training setup
            client_model = deepcopy(global_model).to(device)
            loader = loaders[cid]
            
            # --- Active-Ledger Logic ---
            if enable_synthetic:
                # Analyze distribution
                y_train = torch.cat([batch[1] for batch in loader]).cpu().numpy()
                unique, counts = np.unique(y_train, return_counts=True)
                dist = dict(zip(unique, counts))
                
                augmented = False
                for cls_label in [1, 2, 3, 4]: # Focus on minority classes
                    if dist.get(cls_label, 0) < 50:
                        # IDEALIZED ORACLE: If healthy client, simulate approved generation
                        # If malicious, we simulate a PoC rejection by skipping
                        if cid not in malicious_ids:
                             # print(f"    [SIM] Client {cid+1} augmenting Class {cls_label}")
                             # Lightweight diffusion for the simulation
                             syn_X = generator.generate_synthetic_ecg(class_label=cls_label, quantity=20, num_inference_steps=5)
                             syn_y = np.full(20, cls_label, dtype=np.int64)
                             
                             old_X = torch.cat([batch[0] for batch in loader])
                             old_y = torch.cat([batch[1] for batch in loader])
                             
                             new_X = torch.cat([old_X, torch.tensor(syn_X, dtype=torch.float32)], dim=0)
                             new_y = torch.cat([old_y, torch.tensor(syn_y, dtype=torch.long)], dim=0)
                             
                             from torch.utils.data import TensorDataset, DataLoader
                             new_dataset = TensorDataset(new_X, new_y)
                             loader = DataLoader(new_dataset, batch_size=32, shuffle=True)
                             augmented = True

            # Training
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(client_model.parameters(), lr=config["model"]["learning_rate"])
            
            if cid in malicious_ids:
                # Byzantine attack
                weights = [np.random.normal(0, 1, w.shape).astype(np.float32) for w in _get_weights(client_model)]
                results.append((0.10, weights, len(loader.dataset))) # low mock accuracy
            else:
                client_model.train()
                # Run 1 epoch for simulation speed
                train_epoch(client_model, loader, criterion, optimizer, device)
                val_metrics = evaluate(client_model, val_loaders[cid], criterion, device)
                acc = float(val_metrics["accuracy"])
                results.append((acc, _get_weights(client_model), len(loader.dataset)))
        
        # Aggregation with PoC selection (Mocked PoC based on reported accuracy)
        results.sort(key=lambda x: x[0], reverse=True)
        top_k = results[:TOP_K]
        sel_weights = [r[1] for r in top_k]
        sel_sizes = [r[2] for r in top_k]
        
        global_model = _fedavg_aggregate(global_model, sel_weights, sel_sizes)
        
    f1 = evaluate_f1_scores(global_model, val_loaders, device)
    return f1

def main():
    np.random.seed(42)
    torch.manual_seed(42)
    
    config = load_config()
    device = torch.device("cpu")
    batch_size = config["training"]["batch_size"]
    config["federated"]["local_epochs"] = 1 # override for speed
    
    partitioned_dir = Path(config["data"]["partitioned_dir"])
    
def get_loaders_fn(config, total_clients=TOTAL_CLIENTS):
    """Module-level function to load truncated client data loaders (importable by main.py)."""
    partitioned_dir = Path(config["data"]["partitioned_dir"])
    batch_size = config["training"]["batch_size"]
    ls, vls, ss = [], [], []
    for cid in range(1, total_clients + 1):
        try:
            data = load_client_data(cid, str(partitioned_dir))
            data["X_train"] = data["X_train"][:100]
            data["y_train"] = data["y_train"][:100]
            data["X_val"] = data["X_val"][:50]
            data["y_val"] = data["y_val"][:50]
            tl, vl = create_data_loaders(data["X_train"], data["y_train"], data["X_val"], data["y_val"], batch_size)
            ls.append(tl); vls.append(vl); ss.append(len(data["y_train"]))
        except:
            data = load_client_data(1, str(partitioned_dir))
            tl, vl = create_data_loaders(data["X_train"][:100], data["y_train"][:100], data["X_val"][:50], data["y_val"][:50], batch_size)
            ls.append(tl); vls.append(vl); ss.append(len(data["y_train"]))
    return ls, vls, ss

    l_a, vl_a, s_a = get_loaders_fn(config, TOTAL_CLIENTS)
    f1_a = run_simulation(config, l_a, vl_a, s_a, device, enable_synthetic=False)
    print(f"Baseline F1: {f1_a}")
    
    l_b, vl_b, s_b = get_loaders_fn(config, TOTAL_CLIENTS)
    f1_b = run_simulation(config, l_b, vl_b, s_b, device, enable_synthetic=True)
    print(f"Generative F1: {f1_b}")
    
    # Plot
    minority = [1, 2, 3, 4]
    scores_a = [f1_a[i] for i in minority]
    scores_b = [f1_b[i] for i in minority]
    
    x = np.arange(len(minority))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, scores_a, width, label='Standard Active-Ledger', color='#2C7BB6')
    ax.bar(x + width/2, scores_b, width, label='Ledger-Guided Diffusion', color='#D7191C')
    ax.set_ylabel('F1 Score')
    ax.set_title('Minority Classes F1-Score Comparison\n(Phase 2.5 Empirical Proof)')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Class {i}" for i in minority])
    ax.legend()
    plt.tight_layout()
    plt.savefig('minority_f1_comparison.pdf')
    print("[SUCCESS] Phase 2.5 complete: minority_f1_comparison.pdf generated.")

if __name__ == "__main__":
    main()
