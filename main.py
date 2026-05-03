"""
main.py — Active-Ledger: Experiment Entry Point
=================================================

Usage:
    python main.py --mode robust-baselines         # Session 1: 7-method Byzantine-robust comparison
    python main.py --mode session2-multikrum        # Session 2: Multi-Krum + Blind Diffusion
    python main.py --mode session2-active-ledger    # Session 2: Active-Ledger Full System
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
        description="Active-Ledger Federated Learning — Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode robust-baselines
  python main.py --mode session2-multikrum
  python main.py --mode session2-active-ledger
        """,
    )
    parser.add_argument(
        "--mode",
        choices=[
            "robust-baselines",
            "session2-multikrum",
            "session2-active-ledger",
        ],
        required=True,
        help=(
            "robust-baselines       : Session 1 — 7-method Byzantine-robust aggregation comparison\n"
            "session2-multikrum     : Session 2 — Multi-Krum + Blind Diffusion augmentation\n"
            "session2-active-ledger : Session 2 — Active-Ledger Full System (PoC + Trust-Gated LDM)"
        ),
    )
    return parser.parse_args()


def run_robust_baselines():
    """Session 1: 7-method Byzantine-robust aggregation comparison."""
    print("[MODE] Session 1 — Robust Baselines (7 methods × 40 rounds)")
    from benchmarks.run_robust_baselines import main as run_main
    run_main()


def run_session2_multikrum():
    """Session 2: Multi-Krum + Blind Diffusion augmentation."""
    print("[MODE] Session 2 — Multi-Krum + Blind Diffusion")
    from benchmarks.run_session2_multikrum import main as run_main
    run_main()


def run_session2_active_ledger():
    """Session 2: Active-Ledger Full System (PoC + Trust-Gated LDM)."""
    print("[MODE] Session 2 — Active-Ledger Full System")
    from benchmarks.run_session2_active_ledger import main as run_main
    run_main()


def main():
    args = parse_args()

    dispatch = {
        "robust-baselines":      run_robust_baselines,
        "session2-multikrum":    run_session2_multikrum,
        "session2-active-ledger": run_session2_active_ledger,
    }

    try:
        dispatch[args.mode]()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user. Exiting gracefully.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        raise


if __name__ == "__main__":
    main()
