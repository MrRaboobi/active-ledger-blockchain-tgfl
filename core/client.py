"""
Flower Client for FL with optional malicious behaviour simulation.
Phase 1.4 — Malicious Client Simulation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import math
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from typing import Dict, List, Tuple

import flwr as fl
from flwr.common import (
    NDArrays,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from core.model import create_model, CNNLSTM
from core.train_utils import train_epoch, evaluate


class FLClient(fl.client.NumPyClient):
    """
    Flower NumPy client wrapping the CNN-LSTM model.

    Args:
        client_id  : integer identifier (1-based)
        model      : instantiated CNNLSTM model
        train_loader: PyTorch DataLoader for training set
        val_loader : PyTorch DataLoader for validation set
        config     : project config dict
        is_malicious: when True the client returns Gaussian noise instead of
                      real weight updates (Byzantine attack simulation)
    """

    def __init__(
        self,
        client_id: int,
        model: CNNLSTM,
        train_loader,
        val_loader,
        config: Dict,
        is_malicious: bool = False,
        blockchain_manager=None,
        enable_synthetic: bool = False,
        diffusion_steps: int = 2,
        synthetic_quantity: int = 2,
    ):
        self.client_id = client_id
        self.model = deepcopy(model)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.is_malicious = is_malicious
        self.blockchain = blockchain_manager
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.enable_synthetic = enable_synthetic
        self.diffusion_steps = diffusion_steps
        self.synthetic_quantity = synthetic_quantity
        from core.diffusion import ECGDiffusionGenerator
        self.generator = ECGDiffusionGenerator()

    # ------------------------------------------------------------------
    def get_parameters(self, config: Dict) -> NDArrays:
        return [val.cpu().detach().numpy() for _, val in self.model.state_dict().items()]

    # ------------------------------------------------------------------
    def set_parameters(self, parameters: NDArrays) -> None:
        state_dict = dict(zip(self.model.state_dict().keys(), [torch.tensor(p) for p in parameters]))
        self.model.load_state_dict(state_dict, strict=True)

    # ------------------------------------------------------------------
    def analyze_local_distribution(self):
        class_counts = {i: 0 for i in range(5)}
        for data, label in self.train_loader.dataset:
            label_val = int(label.item() if hasattr(label, 'item') else label)
            if label_val in class_counts:
                class_counts[label_val] += 1
        return class_counts

    # ------------------------------------------------------------------
    def _compute_class_weights(self):
        """Compute clamped inverse-frequency class weights.

        Uses raw inverse-frequency weights (proven to learn minority classes
        by round 15-20) with a floor of 0.3 for majority class and a ceiling
        of 10.0 for minority classes to prevent extreme gradient imbalance.
        """
        counts = self.analyze_local_distribution()
        total  = sum(counts.values())
        if total == 0:
            return None
        weights = []
        for c in range(5):
            if counts[c] > 0:
                w = total / (5 * counts[c])
                w = max(0.3, min(w, 10.0))   # floor=0.3, cap=10
                weights.append(w)
            else:
                weights.append(1.0)
        return torch.FloatTensor(weights).to(self.device)

    # ------------------------------------------------------------------
    def fit(
        self, parameters: NDArrays, config: Dict
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Local training step."""
        distribution = self.analyze_local_distribution()
        generated_req_ids = []
        if self.enable_synthetic:
            total_local = sum(distribution.values())
            for missing_class, count in distribution.items():
                threshold = max(50, 0.15 * total_local)
                if count < threshold and self.blockchain is not None:
                    try:
                        req_id = self.blockchain.request_synthetic(client_id=int(self.client_id), class_label=int(missing_class), quantity=100)
                        print(f"Client {self.client_id} requesting synthetic data for class {missing_class}")
                        attempts = 0
                        while True:
                            status = self.blockchain.get_synthetic_request(req_id)
                            if status.get('approved') == True:
                                print(f"Generation authorized for class {missing_class}")
                                
                                synthetic_X = self.generator.generate_synthetic_ecg(
                                    class_label=int(missing_class),
                                    quantity=self.synthetic_quantity,
                                    num_inference_steps=self.diffusion_steps
                                )
                                synthetic_y = np.full(self.synthetic_quantity, int(missing_class), dtype=np.int64)
                                
                                old_X = torch.cat([batch[0] for batch in self.train_loader])
                                old_y = torch.cat([batch[1] for batch in self.train_loader])
                                
                                new_X = torch.cat([old_X, torch.tensor(synthetic_X, dtype=torch.float32)], dim=0)
                                new_y = torch.cat([old_y, torch.tensor(synthetic_y, dtype=torch.long)], dim=0)
                                
                                from torch.utils.data import TensorDataset, DataLoader
                                new_dataset = TensorDataset(new_X, new_y)
                                import os
                                os_workers = 0 if os.name == 'nt' else 2
                                bs = self.config['training']['batch_size']
                                self.train_loader = DataLoader(new_dataset, batch_size=bs, shuffle=True, pin_memory=True, num_workers=os_workers)
                                
                                print(f"[CLIENT] Augmented local dataset with {self.synthetic_quantity} synthetic samples for class {missing_class}.")
                                generated_req_ids.append(req_id)
                                break
                            
                            attempts += 1
                            if attempts >= 3:
                                print(f"Client {self.client_id} request timed out/rejected. Bypassing.")
                                break
                            time.sleep(2)
                    except Exception as e:
                        print(f"  [warn] Blockchain error for Client {self.client_id} class {missing_class}: {e}. Continuing without synthetic data.")

        self.set_parameters(parameters)

        local_epochs = int(config.get("local_epochs", self.config["federated"]["local_epochs"]))
        lr = self.config["model"]["learning_rate"]

        criterion = nn.CrossEntropyLoss(weight=self._compute_class_weights())
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for _ in range(local_epochs):
            train_epoch(self.model, self.train_loader, criterion, optimizer, self.device)

        for req_id in generated_req_ids:
            try:
                self.blockchain.mark_synthetic_generated(req_id)
                print("[CLIENT] Synthetic data generation cryptographically verified and marked on-chain.")
            except Exception as e:
                print(f"  [warn] mark_synthetic_generated failed for req {req_id}: {e}. Training continues.")

        # Evaluate to get honest accuracy before (potentially) poisoning
        val_metrics = evaluate(self.model, self.val_loader, criterion, self.device)
        honest_accuracy = float(val_metrics["accuracy"])

        updated_weights = self.get_parameters({})
        num_samples = len(self.train_loader.dataset)

        if self.is_malicious:
            # Byzantine attack: replace weights with scaled Gaussian noise
            poisoned_weights = [
                np.random.normal(0, 1, w.shape).astype(np.float32)
                for w in updated_weights
            ]
            return poisoned_weights, num_samples, {"accuracy": 0.10, "is_malicious": 1}

        return updated_weights, num_samples, {"accuracy": honest_accuracy, "is_malicious": 0}

    # ------------------------------------------------------------------
    def evaluate(
        self, parameters: NDArrays, config: Dict
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate current global model on local validation set."""
        self.set_parameters(parameters)

        criterion = nn.CrossEntropyLoss()
        metrics = evaluate(self.model, self.val_loader, criterion, self.device)

        return (
            float(metrics["loss"]),
            len(self.val_loader.dataset),
            {
                "accuracy": float(metrics["accuracy"]),
                "f1": float(metrics["f1"]),
            },
        )
