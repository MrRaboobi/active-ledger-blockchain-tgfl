import os
import numpy as np


def main() -> None:
    base = r"C:\Users\T14s\Desktop\FYP-Blockchain-FL\friends experimentation"
    files = [
        "robust_comparison (1).npy",
        "semantic_results_recovered.npy",
        "sota_sleeper_results.npy",
        "session4_results.npy",
    ]

    for fn in files:
        path = os.path.join(base, fn)
        d = np.load(path, allow_pickle=True).item()
        print(f"== {fn} ==")
        # show schema flags for first method entry
        for m, v in d.items():
            flags = {
                "round_f1": "round_f1" in v,
                "round_lat": "round_lat" in v,
                "latency_history": "latency_history" in v,
                "per_class_f1": "per_class_f1" in v,
                "class_f1_history": "class_f1_history" in v,
                "per_class_prec": "per_class_prec" in v,
                "per_class_recall": "per_class_recall" in v,
                "final_f1": "final_f1" in v,
                "final_precision": "final_precision" in v,
                "final_recall": "final_recall" in v,
                "final_support": "final_support" in v,
                "final_report": "final_report" in v,
                "round_apb_f1": "round_apb_f1" in v,
                "round_bsr": "round_bsr" in v,
            }
            print(" method:", m)
            print(" flags:", flags)
            break

    # confirm class-index meaning via support vector (known from AAMI mapping)
    d = np.load(os.path.join(base, "robust_comparison (1).npy"), allow_pickle=True).item()
    v = d["G_PoC_Only"]
    print("== support check (robust_comparison (1).npy / G_PoC_Only) ==")
    print("support:", v["final_support"])
    print("support[3] (APB):", int(v["final_support"][3]))


if __name__ == "__main__":
    main()

