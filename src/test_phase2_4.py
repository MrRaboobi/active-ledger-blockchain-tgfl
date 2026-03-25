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
from server import start_approval_daemon
import numpy as np

class DummyDataset(Dataset):
    def __init__(self):
        self.data = []
        self.labels = []
        self.data.extend([torch.randn(360) for _ in range(300)])
        self.labels.extend([0 for _ in range(300)])
        for lbl in [1, 3, 4]:
            self.data.extend([torch.randn(360) for _ in range(50)])
            self.labels.extend([lbl for _ in range(50)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(360, 5)

    def forward(self, x):
        return self.linear(x)

def main():
    print("Initializing BlockchainManager...")
    blockchain = BlockchainManager(ganache_url="http://127.0.0.1:8545")
    
    print("Allocating quota for Client 1...")
    blockchain.set_synthetic_quota(1, 1000)

    eth_accounts = [blockchain.deployer] * 10
    
    print("Injecting fake history for Client 1 to boost PoC...")
    dummy_model = DummyModel()
    for r in range(1, 4):
        blockchain.log_update(r, 1, dummy_model.state_dict(), 300, 0.9)

    print("Initializing DummyDataset and Model for Client 1...")
    dummy_dataset = DummyDataset()
    dummy_loader = DataLoader(dummy_dataset, batch_size=32)
    
    config = {
        "federated": {"local_epochs": 1},
        "model": {"learning_rate": 0.01}
    }
    
    client = FLClient(
        client_id=1,
        model=dummy_model,
        train_loader=dummy_loader,
        val_loader=dummy_loader,
        config=config,
        blockchain_manager=blockchain
    )

    stop_daemon = threading.Event()
    daemon_thread = threading.Thread(
        target=start_approval_daemon, 
        args=(blockchain, eth_accounts, stop_daemon, 2.0)
    )
    daemon_thread.start()

    print("Starting client.fit() on main thread...")
    parameters = [p.cpu().detach().numpy() for p in dummy_model.parameters()]
    
    with patch('client.train_epoch'), \
         patch('client.evaluate', return_value={"accuracy": 1.0, "loss": 0.1, "f1": 1.0}):
        client.fit(parameters=parameters, config={})
        
    print("Stopping Server Daemon...")
    stop_daemon.set()
    daemon_thread.join()
    
    total_reqs = blockchain.contract.functions.getTotalSyntheticRequests().call()
    req = blockchain.get_synthetic_request(total_reqs - 1)
    assert req['generated'] == True, f"Failed to mark synthetic data as generated in smart contract! Req ID {total_reqs-1} status: {req}"
    
    print("[SUCCESS] Phase 2.4 complete: Full generative loop verified.")

if __name__ == "__main__":
    main()
