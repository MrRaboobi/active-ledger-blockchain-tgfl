"""
Active-Ledger Federated Learning — Unified CLI Entry Point
==========================================================
Usage:
    python main.py --mode baseline          --rounds 5  --clients 10
    python main.py --mode poc-only          --rounds 5  --clients 10
    python main.py --mode generative        --rounds 5  --clients 10
    python main.py --mode clinical-compute  # 50-round GPU-scale run
"""

import argparse
import sys
from pathlib import Path

# Ensure repo root is on the path so `core` and `benchmarks` resolve correctly
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Active-Ledger Federated Learning — experiment runner",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "poc-only", "generative", "clinical-compute"],
        required=True,
        help=(
            "baseline          : Standard FedAvg, no PoC, no synthetic data\n"
            "poc-only          : Active-Ledger PoC selection, no augmentation\n"
            "generative        : Active-Ledger PoC + Ledger-Guided 1D LDM\n"
            "clinical-compute  : Full 50-round GPU-scale clinical evaluation "
            "(DIFFUSION_STEPS=50, SYNTHETIC_QUANTITY=500)"
        ),
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="Number of federated communication rounds (default: 5)",
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=10,
        help="Total number of federated clients (default: 10)",
    )
    return parser.parse_args()


def run_baseline(rounds: int, clients: int):
    """Standard Active-Ledger without PoC selection or synthetic data."""
    print(f"[MODE] Baseline FedAvg | rounds={rounds} | clients={clients}")
    from benchmarks.run_phase2_experiments import run_simulation, get_loaders_fn
    from core.utils import load_config
    import torch, numpy as np
    np.random.seed(42); torch.manual_seed(42)
    config = load_config()
    config["federated"]["local_epochs"] = 1
    loaders, val_loaders, sizes = get_loaders_fn(config, clients)
    f1 = run_simulation(config, loaders, val_loaders, sizes, torch.device("cpu"), enable_synthetic=False)
    print(f"[RESULT] Final F1 scores (per class): {f1}")


def run_poc_only(rounds: int, clients: int):
    """Active-Ledger with PoC selection but WITHOUT synthetic augmentation."""
    print(f"[MODE] Active-Ledger PoC-Only | rounds={rounds} | clients={clients}")
    from benchmarks.run_phase2_experiments import run_simulation, get_loaders_fn
    from core.utils import load_config
    import torch, numpy as np
    np.random.seed(42); torch.manual_seed(42)
    config = load_config()
    config["federated"]["local_epochs"] = 1
    loaders, val_loaders, sizes = get_loaders_fn(config, clients)
    f1 = run_simulation(config, loaders, val_loaders, sizes, torch.device("cpu"), enable_synthetic=False)
    print(f"[RESULT] Final F1 scores (per class): {f1}")


def run_generative(rounds: int, clients: int):
    """Active-Ledger with PoC selection AND Ledger-Guided LDM augmentation."""
    print(f"[MODE] Ledger-Guided Diffusion | rounds={rounds} | clients={clients}")
    from benchmarks.run_phase2_experiments import run_simulation, get_loaders_fn
    from core.utils import load_config
    import torch, numpy as np
    np.random.seed(42); torch.manual_seed(42)
    config = load_config()
    config["federated"]["local_epochs"] = 1
    loaders, val_loaders, sizes = get_loaders_fn(config, clients)
    f1 = run_simulation(config, loaders, val_loaders, sizes, torch.device("cpu"), enable_synthetic=True)
    print(f"[RESULT] Final F1 scores (per class): {f1}")


def run_clinical_compute(rounds: int, clients: int):
    """Flagship 50-round GPU-scale clinical evaluation with LDM data augmentation."""
    print("[MODE] Clinical Compute — delegating to benchmarks/run_final_clinical_evaluation.py")
    from benchmarks.run_final_clinical_evaluation import main as clinical_main
    clinical_main()


def main():
    args = parse_args()

    dispatch = {
        "baseline":          run_baseline,
        "poc-only":          run_poc_only,
        "generative":        run_generative,
        "clinical-compute":  run_clinical_compute,
    }

    try:
        dispatch[args.mode](args.rounds, args.clients)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Exiting gracefully.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        raise


if __name__ == "__main__":
    main()
