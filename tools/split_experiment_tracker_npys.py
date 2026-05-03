"""
Split Phase-3 result archives into per-experiment .npy files.

Goal (for experiment tracker / research showcase):
  - 7 defences × 3 attacks = 21 npys
  - + 2 diffusion-augmented cells (Session 4) = 23 npys total

Inputs (existing):
  friends experimentation/robust_comparison (1).npy      (Gaussian poisoning)
  friends experimentation/semantic_results_recovered.npy (Static label-flip)
  friends experimentation/sota_sleeper_results.npy       (Sleeper attack)
  friends experimentation/session4_results.npy           (Diffusion comparison)

Outputs (new):
  friends experimentation/experiment_tracker_npys/*.npy

Each output file is a dict saved via np.save(...) containing:
  - metadata: attack_id, attack_name, method_id, method_name, source_file
  - payload: the method's original result dict (round_f1, per_class_f1, etc.)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass(frozen=True)
class AttackSpec:
    attack_id: str
    attack_name: str
    source_npy: str


ATTACKS = [
    AttackSpec("gaussian", "Gaussian poisoning", "robust_comparison (1).npy"),
    AttackSpec("label_flip", "Static label-flip", "semantic_results_recovered.npy"),
    AttackSpec("sleeper", "Sleeper attack", "sota_sleeper_results.npy"),
]


METHOD_PRETTY = {
    "A_FedAvg": "FedAvg",
    "B_Krum": "Krum",
    "C_MultiKrum": "MultiKrum",
    "D_Median": "Median",
    "E_TrimmedMean": "TrimmedMean",
    "F_Bulyan": "Bulyan",
    "G_PoC_Only": "PoC",
    # diffusion session (explicit names; kept stable for tracker)
    "A_PoC_TrustGated_LDM": "PoC_TrustGated_LDM",
    "C_MultiKrum_BlindLDM": "MultiKrum_BlindLDM",
}


def _load_item_dict(path: str) -> Dict[str, Dict[str, Any]]:
    return np.load(path, allow_pickle=True).item()


def _safe_name(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ("_", "-")) else "_" for ch in s).strip("_")


def _write_one(out_path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, obj, allow_pickle=True)


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    base = os.path.join(repo_root, "friends experimentation")
    out_dir = os.path.join(base, "experiment_tracker_npys")
    os.makedirs(out_dir, exist_ok=True)

    written = 0

    # 21 aggregation experiments
    for atk in ATTACKS:
        src_path = os.path.join(base, atk.source_npy)
        src = _load_item_dict(src_path)

        for method_id, payload in src.items():
            method_name = METHOD_PRETTY.get(method_id, method_id)
            out_name = f"{atk.attack_id}__{_safe_name(method_name)}.npy"
            out_path = os.path.join(out_dir, out_name)

            wrapped = {
                "metadata": {
                    "attack_id": atk.attack_id,
                    "attack_name": atk.attack_name,
                    "method_id": method_id,
                    "method_name": method_name,
                    "source_file": atk.source_npy,
                },
                "result": payload,
            }
            _write_one(out_path, wrapped)
            written += 1

    # 2 diffusion-augmented experiments (session4)
    s4_path = os.path.join(base, "session4_results.npy")
    s4 = _load_item_dict(s4_path)
    for method_id, payload in s4.items():
        method_name = METHOD_PRETTY.get(method_id, method_id)
        out_name = f"diffusion__{_safe_name(method_name)}.npy"
        out_path = os.path.join(out_dir, out_name)
        wrapped = {
            "metadata": {
                "attack_id": "sleeper",
                "attack_name": "Sleeper attack (diffusion-augmented comparison)",
                "method_id": method_id,
                "method_name": method_name,
                "source_file": "session4_results.npy",
            },
            "result": payload,
        }
        _write_one(out_path, wrapped)
        written += 1

    # quick integrity check: count files written
    produced = [p for p in os.listdir(out_dir) if p.lower().endswith(".npy")]
    produced.sort()
    print(f"Wrote {written} npy files into: {out_dir}")
    print(f"Folder now contains {len(produced)} npy files.")
    if len(produced) != written:
        raise RuntimeError("Mismatch: produced file count != written count")


if __name__ == "__main__":
    main()

