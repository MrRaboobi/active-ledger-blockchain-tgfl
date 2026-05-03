"""
Phase 3 figure generator (ESORICS / LNCS style).

Design rules (kept consistent across every figure):
  * Professional palette (blue / amber / muted red / teal / slate);
    primary "ours" is the brand blue `#1F4E79`.
  * Heatmaps use `RdYlGn` (intuitive in clinical papers) with adaptive text
    colour so every annotation is readable at any cell value.
  * Distinct linestyles/markers on line plots for grayscale fallback.
  * Vector PDF + 600dpi PNG for every figure.
  * NO latency figure (dropped by decision).
"""

import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


# ---------------------------------------------------------------------------
# Global styling
# ---------------------------------------------------------------------------
PALETTE = {
    "ours": "#1F4E79",        # deep blue (PoC / ours)
    "baseline": "#C0504D",    # muted brick red (weak defenses / failure modes)
    "accent1": "#E8A33D",     # warm amber (FedAvg baseline)
    "accent2": "#2E7D32",     # deep green (strong non-ledger baseline)
    "neutral": "#5F6A6A",     # slate grey (other robust aggregators)
    "rule": "#2F2F2F",        # reference lines (threshold etc.)
}

LINE_STYLES = {
    "G_PoC_Only":   dict(color=PALETTE["ours"],     ls="-",  lw=2.6, marker="o", ms=4),
    "A_FedAvg":     dict(color=PALETTE["accent1"],  ls="-",  lw=2.0, marker="s", ms=4),
    "B_Krum":       dict(color=PALETTE["baseline"], ls="--", lw=2.0, marker="^", ms=4),
    "C_MultiKrum":  dict(color=PALETTE["accent2"],  ls="-.", lw=2.0, marker="D", ms=4),
}

METHOD_ORDER: List[str] = [
    "A_FedAvg",
    "B_Krum",
    "C_MultiKrum",
    "D_Median",
    "E_TrimmedMean",
    "F_Bulyan",
    "G_PoC_Only",
]

METHOD_PRETTY: Dict[str, str] = {
    "A_FedAvg": "FedAvg",
    "B_Krum": "Krum",
    "C_MultiKrum": "Multi-Krum",
    "D_Median": "Median",
    "E_TrimmedMean": "TrimmedMean",
    "F_Bulyan": "Bulyan",
    "G_PoC_Only": "PoC (ours)",
    "A_PoC_TrustGated_LDM": "PoC + Trust-Gated LDM (ours)",
    "C_MultiKrum_BlindLDM": "Multi-Krum + Blind LDM",
}


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SessionArtifact:
    name: str
    path: str


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "0.88",
        "grid.linewidth": 0.6,
    })
    return plt


def load_artifact(path: str) -> Dict[str, dict]:
    return np.load(path, allow_pickle=True).item()


def get_round_f1(values: dict) -> np.ndarray:
    return np.asarray(values["round_f1"], dtype=float)


def get_class_f1_hist(values: dict) -> np.ndarray:
    if "per_class_f1" in values:
        arr = np.asarray(values["per_class_f1"], dtype=float)
        if arr.ndim != 2:
            return np.zeros((0, 5), dtype=float)
        return arr
    if "class_f1_history" in values:
        return np.asarray(values["class_f1_history"], dtype=float)
    raise KeyError("No per-class F1 history found")


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_fig(plt, out_base: str) -> None:
    plt.savefig(out_base + ".pdf", bbox_inches="tight")
    plt.savefig(out_base + ".png", dpi=600, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Heatmap (performance metric)
# ---------------------------------------------------------------------------
def plot_metric_heatmap(
    matrix: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    out_base: str,
    cbar_label: str = "F1 score",
) -> None:
    plt = _mpl()
    from matplotlib.colors import Normalize

    fig, ax = plt.subplots(figsize=(9.6, 3.8))
    cmap = plt.get_cmap("RdYlGn")
    norm = Normalize(vmin=0.0, vmax=1.0)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    # adaptive text colour using rec.709 luminance
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=10, color="black")
                continue
            rgba = cmap(norm(value))
            lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            txt_color = "black" if lum > 0.60 else "white"
            ax.text(
                j, i,
                f"{value:.3f}",
                ha="center", va="center",
                fontsize=10.5, fontweight="bold",
                color=txt_color,
            )

    # cell separators
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", length=0)
    ax.grid(False)

    ax.set_title(title, pad=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    save_fig(plt, out_base)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sleeper fingerprint (coloured line plot)
# ---------------------------------------------------------------------------
def plot_sleeper_fingerprint(
    sleeper: Dict[str, dict],
    methods: List[str],
    out_base: str,
    attack_round: int = 15,
) -> None:
    plt = _mpl()

    fig, axes = plt.subplots(2, 1, figsize=(10.6, 6.6), sharex=True)
    ax1, ax2 = axes

    for m in methods:
        if m not in sleeper:
            continue
        v = sleeper[m]
        r = get_round_f1(v)[:40]
        ch = get_class_f1_hist(v)[:40]
        apb = ch[:, 3] if ch.shape[0] > 0 else np.zeros_like(r)
        style = LINE_STYLES.get(m, {"color": PALETTE["neutral"], "ls": "-", "lw": 1.8})
        label = METHOD_PRETTY.get(m, m)
        x = np.arange(1, len(r) + 1)
        ax1.plot(x, r, label=label, **style)
        ax2.plot(x, apb, label=label, **style)

    # attack activation + clinical threshold
    for ax in (ax1, ax2):
        ax.axvline(attack_round, color=PALETTE["rule"], linestyle=":", linewidth=1.6, alpha=0.85)
    ax2.axhline(0.75, color=PALETTE["rule"], linestyle="--", linewidth=1.3,
                label="Clinical threshold (0.75)")

    ax1.text(attack_round + 0.4, 0.12, "Attack activates (R15)",
             fontsize=8.5, color=PALETTE["rule"])

    ax1.set_ylabel("Macro-F1")
    ax2.set_ylabel("APB-F1 (Class 3)")
    ax2.set_xlabel("Federated round")
    ax1.set_title("Sleeper-attack fingerprint: macro-F1 (top) vs APB-F1 (bottom)",
                  pad=8)
    ax1.set_ylim(0.0, 1.02)
    ax2.set_ylim(0.0, 1.02)

    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
               frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_fig(plt, out_base)
    plt.close(fig)


# ---------------------------------------------------------------------------
# APB worst-case + Session-4 LDM comparison (single clean panel)
# ---------------------------------------------------------------------------
def plot_apb_worstcase_plus_ldm(
    s1: Dict[str, dict], s2: Dict[str, dict], s3: Dict[str, dict], s4: Dict[str, dict],
    out_base: str,
) -> None:
    plt = _mpl()

    worst = {}
    for m in METHOD_ORDER:
        vals = []
        for d in (s1, s2, s3):
            if m in d and "final_f1" in d[m]:
                vals.append(float(np.asarray(d[m]["final_f1"])[3]))
        if len(vals) == 3:
            worst[m] = min(vals)

    s4_labels, s4_values = [], []
    if "A_PoC_TrustGated_LDM" in s4:
        s4_labels.append("PoC +\nTrust-Gated LDM\n(ours)")
        s4_values.append(float(np.asarray(s4["A_PoC_TrustGated_LDM"]["round_apb_f1"])[-1]))
    if "C_MultiKrum_BlindLDM" in s4:
        s4_labels.append("Multi-Krum +\nBlind LDM")
        s4_values.append(float(np.asarray(s4["C_MultiKrum_BlindLDM"]["round_apb_f1"])[-1]))

    ordered = [m for m in METHOD_ORDER if m in worst]
    ys = [worst[m] for m in ordered]
    labels = [METHOD_PRETTY.get(m, m) for m in ordered]

    base_colors = []
    for m in ordered:
        if m == "G_PoC_Only":
            base_colors.append(PALETTE["ours"])
        elif m in ("A_FedAvg", "B_Krum"):
            base_colors.append(PALETTE["baseline"])
        else:
            base_colors.append(PALETTE["neutral"])

    s4_colors = [PALETTE["ours"], PALETTE["accent2"]]

    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    x1 = np.arange(len(ordered))
    gap = 1.2
    x2 = np.arange(len(s4_values)) + len(ordered) + gap

    ax.bar(x1, ys, color=base_colors, edgecolor="black", linewidth=0.9, width=0.72)
    ax.bar(x2, s4_values, color=s4_colors, edgecolor="black", linewidth=0.9, width=0.72,
           hatch="//")

    for xi, yi in zip(x1, ys):
        ax.text(xi, yi + 0.02, f"{yi:.3f}", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold")
    for xi, yi in zip(x2, s4_values):
        ax.text(xi, yi + 0.02, f"{yi:.3f}", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold")

    ax.axhline(0.75, color=PALETTE["rule"], linestyle="--", linewidth=1.4,
               label="Clinical threshold (0.75)")

    all_x = list(x1) + list(x2)
    all_labels = labels + s4_labels
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels, rotation=0, ha="center")

    # group titles
    ax.annotate("Worst-case APB-F1 across attacks",
                xy=(float(np.mean(x1)), 1.035), xycoords=("data", "axes fraction"),
                ha="center", fontsize=10, fontweight="bold")
    ax.annotate("Trust-gated diffusion comparison",
                xy=(float(np.mean(x2)), 1.035), xycoords=("data", "axes fraction"),
                ha="center", fontsize=10, fontweight="bold")

    # visual separator
    sep_x = (max(x1) + min(x2)) / 2.0
    ax.axvline(sep_x, color="0.75", linestyle=":", linewidth=1.0)

    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("APB-F1 (Class 3)")
    ax.set_title("Rare-class integrity under attack: worst-case robustness and diffusion comparison",
                 pad=28)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    save_fig(plt, out_base)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Architecture diagram (two-panel; no overlapping arrows)
# ---------------------------------------------------------------------------
def plot_architecture_diagram(out_base: str) -> None:
    plt = _mpl()
    from matplotlib.patches import FancyBboxPatch  # noqa: F401

    # Compact two-panel diagram suitable for 1-column paper width
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.8))
    ax_top, ax_bot = axes
    for ax in (ax_top, ax_bot):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 4.4)
        ax.axis("off")
        ax.grid(False)

    def box(ax, x, y, w, h, label, face="#EEF2F7", edge="#34495E", bold=False):
        p = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            linewidth=1.3, edgecolor=edge, facecolor=face,
        )
        ax.add_patch(p)
        ax.text(
            x + w / 2, y + h / 2, label,
            ha="center", va="center",
            fontsize=9.2, fontweight=("bold" if bold else "normal"),
            color="#222222",
        )

    def arrow(ax, x1, y1, x2, y2, text=None, color="#2F2F2F", above=True, offset=0.18):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                lw=1.3,
                shrinkA=1,
                shrinkB=1,
                mutation_scale=14,
            ),
        )
        if text:
            tx = (x1 + x2) / 2
            ty = (y1 + y2) / 2 + (offset if above else -offset)
            ax.text(tx, ty, text, ha="center",
                    va=("bottom" if above else "top"),
                    fontsize=8.6, color=color)

    # ---------------- TOP PANEL: aggregation flow ----------------
    ax_top.text(0.02, 4.18, "(A)  FL aggregation with immutable on-chain receipts",
                fontsize=11, fontweight="bold", color="#1B2631")

    # top row
    box(ax_top, 0.20, 2.75, 2.00, 0.95, "Clients", face="#DDE6F0")
    box(ax_top, 2.60, 2.75, 2.00, 0.95, "Local training\n(CNN–LSTM)", face="#DDE6F0")
    box(ax_top, 5.00, 2.75, 2.25, 0.95, "Server\nPoC-ranked aggregator", face="#E6F0DC")
    box(ax_top, 7.65, 2.75, 2.10, 0.95, "Global model", face="#E6F0DC")

    arrow(ax_top, 2.20, 3.22, 2.60, 3.22, "data")
    arrow(ax_top, 4.60, 3.22, 5.00, 3.22, "updates")
    arrow(ax_top, 7.25, 3.22, 7.65, 3.22, "broadcast")

    # ledger band (bottom)
    box(ax_top, 0.20, 0.60, 9.55, 1.20,
        "Ledger: immutable receipts\n"
        "PoC score = EMA$_{\\gamma}$(accuracy history) × participation fraction",
        face="#FDF3E1", bold=True)

    arrow(ax_top, 3.60, 2.75, 3.60, 1.82, "log_update", above=False, offset=0.25)
    arrow(ax_top, 6.10, 2.75, 6.10, 1.82, "RoundCompleted", above=False, offset=0.25)

    # ---------------- BOTTOM PANEL: stealth channel blocked by PoC gate ----------------
    ax_bot.text(0.02, 4.18,
                "(B)  Trust-gated diffusion closes the augmentation stealth channel",
                fontsize=11, fontweight="bold", color="#1B2631")

    # honest request path (top row)
    box(ax_bot, 0.20, 2.75, 1.80, 0.95, "Client\n(rare-class deficient)", face="#DDE6F0")
    box(ax_bot, 2.20, 2.75, 2.10, 0.95, "SyntheticRequested\n(on-chain)", face="#FDF3E1")
    box(ax_bot, 4.50, 2.75, 2.10, 0.95, "Approval daemon\nPoC ≥ 0.4 ?", face="#E6F0DC")
    box(ax_bot, 6.80, 2.75, 1.70, 0.95, "Diffusion\nGenerator (LDM)", face="#E6F0DC")
    box(ax_bot, 8.70, 2.75, 1.10, 0.95, "Augmented\ntrain set", face="#E6F0DC")

    arrow(ax_bot, 2.00, 3.22, 2.20, 3.22, "request")
    arrow(ax_bot, 4.30, 3.22, 4.50, 3.22, "poll")
    arrow(ax_bot, 6.60, 3.22, 6.80, 3.22, "approved")
    arrow(ax_bot, 8.50, 3.22, 8.70, 3.22, "samples")

    # attacker path (middle row) -> BLOCKED at gate
    box(ax_bot, 0.20, 1.50, 1.80, 0.90, "Byzantine client\n(after activation)",
        face="#F5D5D2", edge="#8B0000")

    arrow(ax_bot, 2.00, 1.95, 4.40, 1.95, "stealth request", color=PALETTE["baseline"])
    ax_bot.plot([4.55, 5.10], [1.70, 2.20], color=PALETTE["baseline"], lw=3.0)
    ax_bot.plot([4.55, 5.10], [2.20, 1.70], color=PALETTE["baseline"], lw=3.0)
    ax_bot.text(5.30, 1.95, "BLOCKED  (PoC < 0.4)",
                ha="left", va="center", fontsize=9.5, fontweight="bold",
                color=PALETTE["baseline"])

    # full-width immutable receipts band at the bottom
    box(ax_bot, 0.20, 0.25, 9.55, 0.95,
        "SyntheticGenerated receipt — immutable audit trail",
        face="#FDF3E1", bold=True)
    # link generator down to audit band
    arrow(ax_bot, 7.65, 2.75, 7.65, 1.22, None)

    fig.suptitle("Active-Ledger architecture: selection + trust-gated augmentation",
                 fontsize=11.5, fontweight="bold", y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save_fig(plt, out_base)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    root = r"C:\Users\T14s\Desktop\FYP-Blockchain-FL"
    base = os.path.join(root, "friends experimentation")
    outdir = os.path.join(root, "docs", "figures_phase3")
    ensure_outdir(outdir)

    s1 = load_artifact(os.path.join(base, "robust_comparison (1).npy"))
    s2 = load_artifact(os.path.join(base, "semantic_results_recovered.npy"))
    s3 = load_artifact(os.path.join(base, "sota_sleeper_results.npy"))
    s4 = load_artifact(os.path.join(base, "session4_results.npy"))

    attacks = ["Gaussian poisoning", "Label-flip", "Sleeper attack"]
    methods = METHOD_ORDER
    row_labels = [METHOD_PRETTY.get(m, m) for m in methods]

    apb = np.zeros((len(methods), 3), dtype=float)
    macro = np.zeros((len(methods), 3), dtype=float)
    for i, m in enumerate(methods):
        apb[i, 0] = float(np.asarray(s1[m]["final_f1"])[3]) if m in s1 else np.nan
        apb[i, 1] = float(np.asarray(s2[m]["final_f1"])[3]) if m in s2 else np.nan
        apb[i, 2] = float(np.asarray(s3[m]["final_f1"])[3]) if m in s3 else np.nan
        macro[i, 0] = float(np.asarray(s1[m]["round_f1"])[-1]) if m in s1 else np.nan
        macro[i, 1] = float(np.asarray(s2[m]["round_f1"])[-1]) if m in s2 else np.nan
        macro[i, 2] = float(np.asarray(s3[m]["round_f1"])[-1]) if m in s3 else np.nan

    plot_metric_heatmap(
        matrix=apb,
        row_labels=row_labels,
        col_labels=attacks,
        title="APB-F1 robustness matrix (Class 3 — primary clinical metric)",
        cbar_label="APB-F1 (higher is better)",
        out_base=os.path.join(outdir, "fig_apb_heatmap"),
    )
    plot_metric_heatmap(
        matrix=macro,
        row_labels=row_labels,
        col_labels=attacks,
        title="Macro-F1 robustness matrix (secondary metric — can be misleading)",
        cbar_label="Macro-F1 (higher is better)",
        out_base=os.path.join(outdir, "fig_macro_heatmap"),
    )

    plot_sleeper_fingerprint(
        sleeper=s3,
        methods=["G_PoC_Only", "A_FedAvg", "B_Krum", "C_MultiKrum"],
        out_base=os.path.join(outdir, "fig_sleeper_fingerprint"),
    )

    plot_apb_worstcase_plus_ldm(
        s1=s1, s2=s2, s3=s3, s4=s4,
        out_base=os.path.join(outdir, "fig_apb_with_ldm"),
    )

    plot_architecture_diagram(
        out_base=os.path.join(outdir, "fig_architecture_diagram"),
    )

    print("Wrote figures to:", outdir)


if __name__ == "__main__":
    main()
