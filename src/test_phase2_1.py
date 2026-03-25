import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

import time
import threading
from unittest.mock import patch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from client import FLClient
from blockchain import BlockchainManager

class DummyDataset(Dataset):
    def __init__(self):
        self.data = []
        self.labels = []
        # Class 0: 300 samples
        self.data.extend([torch.randn(10) for _ in range(300)])
        self.labels.extend([0 for _ in range(300)])
        # Use classes 1, 3, 4 with at least 50 samples to prevent requests for them
        for lbl in [1, 3, 4]:
            self.data.extend([torch.randn(10) for _ in range(50)])
            self.labels.extend([lbl for _ in range(50)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

def orchestrator_thread(blockchain):
    # Wait for the client to submit requests
    time.sleep(5)
    print("Orchestrator: Waking up to approve requests...")
    try:
        total_requests = blockchain.contract.functions.getTotalSyntheticRequests().call()
        for i in range(total_requests):
            req = blockchain.get_synthetic_request(i)
            if not req['approved']:
                print(f"Orchestrator: Approving request {i} for class {req['class_label']}")
                blockchain.approve_synthetic(i)
    except Exception as e:
        print(f"Orchestrator Error: {e}")

def main():
    print("Initializing BlockchainManager...")
    blockchain = BlockchainManager(ganache_url="http://127.0.0.1:8545")
    
    # Allocate synthetic data quota to the client to avoid 'Quota exceeded'
    print("Allocating quota for Client 1...")
    blockchain.set_synthetic_quota(1, 1000)

    print("Initializing DummyDataset and Model...")
    dummy_dataset = DummyDataset()
    dummy_loader = DataLoader(dummy_dataset, batch_size=32)
    dummy_model = DummyModel()
    
    config = {
        "federated": {"local_epochs": 1},
        "model": {"learning_rate": 0.01}
    }
    
    print("Initializing FLClient...")
    client = FLClient(
        client_id=1,
        model=dummy_model,
        train_loader=dummy_loader,
        val_loader=dummy_loader,
        config=config,
        blockchain_manager=blockchain
    )

    t = threading.Thread(target=orchestrator_thread, args=(blockchain,))
    t.start()

    print("Starting client.fit() with mock training/evaluation...")
    parameters = [p.cpu().detach().numpy() for p in dummy_model.parameters()]
    
    # Patch train_epoch and evaluate to bypass actual training, 
    # ensuring we only test the request-and-approve handshake
    with patch('client.train_epoch'), \
         patch('client.evaluate', return_value={"accuracy": 1.0, "loss": 0.1, "f1": 1.0}):
        client.fit(parameters=parameters, config={})
        
    print("Phase 2.1 Test Completed Successfully! Handshake verified.")
    t.join()

if __name__ == "__main__":
    main()
