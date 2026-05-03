# Phase 3 Figure Captions, Alt Text, Provenance and Paper Placement

This file is the **single source of truth** for the figure pack that will be
submitted with the ESORICS paper. It covers:

- **Final 7-image figure pack** (2 existing + 5 generated).
- **Captions** in LNCS style.
- **Alt Text** (Springer accessibility requirement).
- **Data provenance** (exact `.npy` field for every quantitative number).
- **Paper placement** (which section the figure supports).

> Rule: any numeric claim in the paper text must cite the corresponding
> figure/table and match the value shown in that figure. See
> `docs/LNCS_ESORICS_LATEX_INTEGRATION_CHECKLIST.md`.

---

## Final figure pack (6 images) + architecture placeholder

| # | Name | Source | Intended section |
|---|---|---|---|
| 1 | APB-F1 robustness matrix | `docs/figures_phase3/fig_apb_heatmap.pdf` | **Results — headline table/figure** |
| 2 | Macro-F1 robustness matrix | `docs/figures_phase3/fig_macro_heatmap.pdf` | **Results — supporting / discussion** |
| 3 | Sleeper-attack fingerprint | `docs/figures_phase3/fig_sleeper_fingerprint.pdf` | **Results — Sleeper attack** |
| 4 | APB worst-case + diffusion comparison | `docs/figures_phase3/fig_apb_with_ldm.pdf` | **Results — synthesis / diffusion framing** |
| 5 | APB/Macro vs round (existing image #1) | provided screenshot | **Results — diffusion time-series** |
| 6 | BSR vs round (existing image #2) | provided screenshot | **Results — Security metric (BSR)** |

### Architecture diagram placeholder (not included in this pack)
- **Status**: Placeholder only. The architecture diagram will be created later by the author.
- **Paper location**: End of System Overview / start of Threat Model.
- **What it must show**: Orchestration cycle (clients → server → ledger receipts) and trust-gated diffusion pipeline with the augmentation stealth channel blocked by the gate.

---

## 1) Fig. 1 — APB-F1 robustness matrix (primary clinical metric)
- **Files**: `docs/figures_phase3/fig_apb_heatmap.pdf` (and `.png`)
- **Caption (LNCS)**: *APB-F1 (Class 3) across three adversarial threat models for seven defenses.* Each cell reports the final-round APB-F1. PoC sustains APB-F1 ≥ 0.78 in every regime; FedAvg collapses to 0.000 under Gaussian; Krum collapses to 0.306 under the sleeper attack.
- **Alt Text**: A color-coded matrix with methods as rows (FedAvg, Krum, Multi-Krum, Median, TrimmedMean, Bulyan, PoC) and attacks as columns (Gaussian, Label-Flip, Sleeper). Each cell shows the APB-F1 value; red cells indicate failure (e.g., 0.000 for FedAvg/Gaussian; 0.306 for Krum/Sleeper); green cells indicate strong performance (e.g., 0.877 for PoC/Gaussian).
- **Data provenance**:
  - Gaussian: `robust_comparison (1).npy`, method `X`, `final_f1[3]`
  - Label-flip: `semantic_results_recovered.npy`, method `X`, `final_f1[3]`
  - Sleeper: `sota_sleeper_results.npy`, method `X`, `final_f1[3]`
- **Where in the paper**: Primary results figure. Must be the **first Results figure** (anchors the “APB-first” framing).

---

## 2) Fig. 2 — Macro-F1 robustness matrix (secondary metric)
- **Files**: `docs/figures_phase3/fig_macro_heatmap.pdf` (and `.png`)
- **Caption (LNCS)**: *Macro-F1 (secondary metric).* Macro-F1 under the three attack models for the same seven defenses. The disparity between APB-F1 and macro-F1 rankings (Fig. 2 vs Fig. 3) motivates the paper’s argument that macro-F1 can be misleading under clinical class imbalance.
- **Alt Text**: A color-coded matrix of macro-F1 values. FedAvg is very low under Gaussian (0.171) but high under label-flip (0.961) and sleeper (0.925), while PoC is consistently strong across all three attacks.
- **Data provenance**: Same artifacts as Fig. 2, using `round_f1[-1]` for each method/session.
- **Where in the paper**: Immediately after Fig. 2, used in Discussion to explain the macro-F1 “dilution effect” and why APB-F1 must be the primary metric.

---

## 3) Fig. 3 — Sleeper-attack fingerprint
- **Files**: `docs/figures_phase3/fig_sleeper_fingerprint.pdf` (and `.png`)
- **Caption (LNCS)**: *Sleeper-attack fingerprint.* Macro-F1 (top) and APB-F1 (bottom) versus federated round for PoC (ours), FedAvg, Krum, and Multi-Krum. Attack activates at round 15. After activation, Krum's APB-F1 degrades and ends near 0.31, while PoC remains at or above the clinical APB-F1 threshold of 0.75.
- **Alt Text**: Two stacked line plots. Macro-F1 (top) converges near 0.9 for most methods. APB-F1 (bottom) shows divergence after round 15: Krum drops to around 0.3 by round 40, while PoC remains above 0.75.
- **Data provenance**: `sota_sleeper_results.npy`
  - Macro-F1: `round_f1[0:40]` per method
  - APB-F1: `class_f1_history[0:40, 3]` per method
- **Where in the paper**: Sleeper-attack results subsection; cite as the visual fingerprint that explains the APB-F1 matrix column for "Sleeper attack".

---

## 4) Fig. 4 — Rare-class integrity across attacks, with and without trust-gated LDM
- **Files**: `docs/figures_phase3/fig_apb_with_ldm.pdf` (and `.png`)
- **Caption (LNCS)**: *APB-F1 synthesis.* Left group: worst-case APB-F1 across the three attack models per defense (FedAvg = 0.000; Krum = 0.306; PoC = 0.786). Right group: final-round APB-F1 with diffusion augmentation: PoC + Trust-Gated LDM (ours) = 0.878; Multi-Krum + Blind LDM = 0.857. Dashed line: clinical APB-F1 threshold (0.75).
- **Alt Text**: Bar chart. Left block shows worst-case APB-F1 for seven defenses with PoC highlighted. Right block shows two bars (hatched): PoC with Trust-Gated LDM (highest), and Multi-Krum with Blind LDM. Both are above the 0.75 clinical threshold, PoC slightly higher.
- **Data provenance**:
  - Left bars: worst-case over the three attacks of `final_f1[3]` per method
  - Right bars: `session4_results.npy`, `round_apb_f1[-1]` for each LDM method
- **Where in the paper**: End of Results, as the summary figure that ties adversarial robustness to the diffusion augmentation story.

---

## 5) Fig. 5 (existing, keep) — APB-F1 and Macro-F1 vs round (diffusion comparison)
- **Files**: Your provided screenshot (combined APB / Macro line plot).
- **Caption (LNCS)**: *Training dynamics under the sleeper attack with diffusion.* APB-F1 (left) and macro-F1 (right) versus federated round for PoC + Trust-Gated LDM (ours) and Multi-Krum + Blind LDM. Attack activates at round 15; clinical APB-F1 threshold is indicated at 0.75.
- **Alt Text**: Two line plots. Left: APB-F1 rises from below 0.2 in early rounds to above 0.85 by round 20 for both methods, with the trust-gated method higher by a small margin and stable above 0.75. Right: macro-F1 converges above 0.95 for both methods; attack onset at round 15 does not cause visible collapse.
- **Data provenance**: Numeric counterpart is `session4_results.npy` (`round_apb_f1`, `round_f1`).
- **Where in the paper**: Diffusion results subsection; paired with Fig. 7 (BSR).

---

## 6) Fig. 6 (existing, keep) — Backdoor Success Rate (BSR) vs round
- **Files**: Your provided screenshot (BSR line plot).
- **Caption (LNCS)**: *Backdoor success rate (BSR) versus round (sleeper attack).* BSR is zero before round 15 for both methods; after activation it fluctuates in the 0.05–0.20 range, with final-round BSR higher for Multi-Krum + Blind LDM than for PoC + Trust-Gated LDM. A lower BSR is better.
- **Alt Text**: Line plot of backdoor success rate over 40 rounds. Curves are flat at zero for rounds 1–14; after round 15 both oscillate, and the blind-LDM curve trends higher than the trust-gated curve near the final rounds.
- **Data provenance**: Numeric counterpart is `session4_results.npy`, `round_bsr`.
- **Where in the paper**: Security metric in the diffusion results; immediately after Fig. 6.
