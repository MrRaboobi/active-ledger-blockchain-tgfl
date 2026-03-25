"""
Flower Client for FL with optional malicious behaviour simulation.
Phase 1.4 — Malicious Client Simulation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import time
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

from model import create_model, CNNLSTM
from train_utils import train_epoch, evaluate


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
    ):
        self.client_id = client_id
        self.model = deepcopy(model)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.is_malicious = is_malicious
        self.blockchain = blockchain_manager
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.enable_synthetic = enable_synthetic
        from diffusion import ECGDiffusionGenerator
        self.generator = ECGDiffusionGenerator()

    # ------------------------------------------------------------------
    def get_parameters(self, config: Dict) -> NDArrays:
        return [p.cpu().detach().numpy() for p in self.model.parameters()]

    # ------------------------------------------------------------------
    def set_parameters(self, parameters: NDArrays) -> None:
        params_dict = zip(self.model.parameters(), parameters)
        for p, new_p in params_dict:
            with torch.no_grad():
                p.copy_(torch.tensor(new_p))

    # ------------------------------------------------------------------
    def analyze_local_distribution(self):
        class_counts = {i: 0 for i in range(5)}
        for data, label in self.train_loader.dataset:
            label_val = int(label.item() if hasattr(label, 'item') else label)
            if label_val in class_counts:
                class_counts[label_val] += 1
        return class_counts

    # ------------------------------------------------------------------
    def fit(
        self, parameters: NDArrays, config: Dict
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Local training step."""
        distribution = self.analyze_local_distribution()
        generated_req_ids = []
        if self.enable_synthetic:
            for missing_class, count in distribution.items():
                if count < 50 and self.blockchain is not None:
                    req_id = self.blockchain.request_synthetic(client_id=int(self.client_id), class_label=int(missing_class), quantity=100)
                    print(f"Client {self.client_id} requesting synthetic data for class {missing_class}")
                    attempts = 0
                    while True:
                        status = self.blockchain.get_synthetic_request(req_id)
                        if status.get('approved') == True:
                            print(f"Generation authorized for class {missing_class}")
                            
                            synthetic_X = self.generator.generate_synthetic_ecg(class_label=int(missing_class), quantity=2, num_inference_steps=2)
                            synthetic_y = np.full(2, int(missing_class), dtype=np.int64)
                            
                            old_X = torch.cat([batch[0] for batch in self.train_loader])
                            old_y = torch.cat([batch[1] for batch in self.train_loader])
                            
                            new_X = torch.cat([old_X, torch.tensor(synthetic_X, dtype=torch.float32)], dim=0)
                            new_y = torch.cat([old_y, torch.tensor(synthetic_y, dtype=torch.long)], dim=0)
                            
                            from torch.utils.data import TensorDataset, DataLoader
                            new_dataset = TensorDataset(new_X, new_y)
                            self.train_loader = DataLoader(new_dataset, batch_size=32, shuffle=True)
                            
                            print(f"[CLIENT] Augmented local dataset with 2 synthetic samples for class {missing_class}.")
                            generated_req_ids.append(req_id)
                            break
                        
                        attempts += 1
                        if attempts >= 10:
                            print(f"Client {self.client_id} request timed out/rejected. Bypassing.")
                            break
                        time.sleep(2)

        self.set_parameters(parameters)

        local_epochs = int(config.get("local_epochs", self.config["federated"]["local_epochs"]))
        lr = self.config["model"]["learning_rate"]

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for _ in range(local_epochs):
            train_epoch(self.model, self.train_loader, criterion, optimizer, self.device)

        for req_id in generated_req_ids:
            self.blockchain.mark_synthetic_generated(req_id)
            print("[CLIENT] Synthetic data generation cryptographically verified and marked on-chain.")

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
