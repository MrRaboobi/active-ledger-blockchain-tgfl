"""
core/robust_aggregation.py
==========================
Industry-standard Byzantine-robust aggregation algorithms for Phase 3.2.

All functions share an identical interface:
    aggregate(global_model, client_weights_list, sizes, **kwargs) -> model

Where:
    global_model        : the current global PyTorch model (modified in-place)
    client_weights_list : list of weight lists, one per client
                          (each weight list = [np.ndarray, ...] matching model params)
    sizes               : list of ints, training samples per client (same order)

References:
    Krum / Multi-Krum  : Blanchard et al., NeurIPS 2017
    Median / TrimMean  : Yin et al., ICML 2018
    Bulyan             : El Mhamdi et al., ICML 2018

Change log:
    2026-04-18  Created for Phase 3.2 robust baseline experiments.
"""

from copy import deepcopy

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _set_weights(model, weights):
    """Load a list of numpy arrays into model state_dict."""
    state_dict = dict(
        zip(model.state_dict().keys(), [torch.tensor(w) for w in weights])
    )
    model.load_state_dict(state_dict, strict=True)


def _get_weights(model):
    """Extract model parameters as a list of numpy arrays."""
    return [val.cpu().detach().numpy() for _, val in model.state_dict().items()]


def _flatten(weights_list):
    """Flatten a list of numpy arrays into a single 1-D numpy vector."""
    return np.concatenate([w.flatten() for w in weights_list])


def _fedavg_on_subset(global_model, subset_weights, subset_sizes):
    """Weighted average of a subset of clients. Used inside Multi-Krum & Bulyan."""
    total     = sum(subset_sizes)
    global_sd = global_model.state_dict()
    param_keys = [k for k in global_sd if "num_batches_tracked" not in k]

    agg = {k: torch.zeros_like(v, dtype=torch.float32)
           for k, v in global_sd.items() if k in param_keys}

    for weights, size in zip(subset_weights, subset_sizes):
        factor = size / total
        tmp = deepcopy(global_model)
        _set_weights(tmp, weights)
        tmp_sd = tmp.state_dict()
        for k in param_keys:
            agg[k] += tmp_sd[k].float() * factor

    new_sd = dict(global_sd)
    for k in param_keys:
        new_sd[k] = agg[k].to(global_sd[k].dtype)
    global_model.load_state_dict(new_sd)
    return global_model


# ─────────────────────────────────────────────────────────────────────────────
# A: FedAvg  (baseline — defenseless)
# ─────────────────────────────────────────────────────────────────────────────

def fedavg_aggregate(global_model, client_weights_list, sizes):
    """
    Standard FedAvg: weighted average of ALL client updates.
    No Byzantine defense whatsoever — included as the defenseless baseline.

    Reference: McMahan et al., "Communication-Efficient Learning of Deep
    Networks from Decentralized Data", AISTATS 2017.
    """
    return _fedavg_on_subset(global_model, client_weights_list, sizes)


# ─────────────────────────────────────────────────────────────────────────────
# B: Krum
# ─────────────────────────────────────────────────────────────────────────────

def _krum_scores(flat_vecs, f):
    """
    Compute Krum scores for each client.

    Score(i) = sum of squared Euclidean distances to the (n - f - 2)
    nearest neighbours of client i.

    Lower score = more central = more likely to be honest.

    Args:
        flat_vecs : np.ndarray, shape (n_clients, param_dim)
        f         : number of assumed Byzantine clients

    Returns:
        scores    : np.ndarray, shape (n_clients,)
    """
    n = len(flat_vecs)
    neighbours = n - f - 2   # At least 1 required

    if neighbours < 1:
        raise ValueError(
            f"Krum requires n ≥ f+3. Got n={n}, f={f}. "
            f"Reduce f or add more clients."
        )

    # Pairwise squared Euclidean distances
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.sum((flat_vecs[i] - flat_vecs[j]) ** 2))
            dists[i, j] = d
            dists[j, i] = d

    # For each client, sort distances to others and sum the closest `neighbours`
    scores = np.zeros(n)
    for i in range(n):
        sorted_dists = np.sort(dists[i])     # ascending; index 0 is dist to self (0)
        scores[i] = np.sum(sorted_dists[1:neighbours + 1])  # skip self

    return scores


def krum_aggregate(global_model, client_weights_list, sizes, f=2):
    """
    Krum: Select the SINGLE client update closest to its neighbours.

    The selected client's weights become the new global model.
    No averaging — pure selection.

    Args:
        f : number of assumed Byzantine clients (default=2, matching our setup)

    Reference: Blanchard et al., NeurIPS 2017.
    """
    flat_vecs = np.array([_flatten(w) for w in client_weights_list])
    scores    = _krum_scores(flat_vecs, f)
    winner    = int(np.argmin(scores))

    print(f"  [Krum] Winner: client {winner + 1}  |  "
          f"Scores: {np.round(scores, 2)}")

    _set_weights(global_model, client_weights_list[winner])
    return global_model


# ─────────────────────────────────────────────────────────────────────────────
# C: Multi-Krum
# ─────────────────────────────────────────────────────────────────────────────

def multi_krum_aggregate(global_model, client_weights_list, sizes, f=2, k=7):
    """
    Multi-Krum: Select the TOP-K clients by Krum score, then FedAvg them.

    When k=1  → equivalent to standard Krum.
    When k=n  → equivalent to standard FedAvg (no defense).
    We use k=7 to mirror Active-Ledger's Top-7 selection (fair comparison).

    Args:
        f : number of assumed Byzantine clients (default=2)
        k : number of clients to keep after scoring (default=7)

    Reference: Blanchard et al., NeurIPS 2017.
    """
    n         = len(client_weights_list)
    flat_vecs = np.array([_flatten(w) for w in client_weights_list])
    scores    = _krum_scores(flat_vecs, f)

    # Select k clients with the lowest (best) Krum scores
    top_k_idx = np.argsort(scores)[:k]

    print(f"  [Multi-Krum k={k}] Selected clients: {[i+1 for i in top_k_idx]}  |  "
          f"Rejected: {[i+1 for i in range(n) if i not in top_k_idx]}")

    sel_weights = [client_weights_list[i] for i in top_k_idx]
    sel_sizes   = [sizes[i] for i in top_k_idx]

    return _fedavg_on_subset(global_model, sel_weights, sel_sizes)


# ─────────────────────────────────────────────────────────────────────────────
# D: Coordinate-wise Median
# ─────────────────────────────────────────────────────────────────────────────

def median_aggregate(global_model, client_weights_list, sizes):
    """
    Coordinate-wise Median: for every parameter element, take the
    median value across all clients. Ignores sample counts.

    Resistant to arbitrary Byzantine updates because the median of
    n values is unaffected by up to floor((n-1)/2) outliers.

    Reference: Yin et al., "Byzantine-Robust Distributed Learning:
    Towards Optimal Statistical Rates", ICML 2018.
    """
    global_sd  = global_model.state_dict()
    param_keys = [k for k in global_sd if "num_batches_tracked" not in k]

    # Build one model per client to access state_dicts by key
    client_models = []
    for weights in client_weights_list:
        tmp = deepcopy(global_model)
        _set_weights(tmp, weights)
        client_models.append(tmp.state_dict())

    new_sd = dict(global_sd)
    for k in param_keys:
        # Stack: (n_clients, *param_shape)
        stacked = torch.stack(
            [cm[k].float() for cm in client_models], dim=0
        )
        # Per-element median along the client dimension
        median_val = torch.median(stacked, dim=0).values
        new_sd[k]  = median_val.to(global_sd[k].dtype)

    global_model.load_state_dict(new_sd)
    print(f"  [Median] Aggregated {len(client_weights_list)} clients "
          f"(coordinate-wise median)")
    return global_model


# ─────────────────────────────────────────────────────────────────────────────
# E: Trimmed Mean
# ─────────────────────────────────────────────────────────────────────────────

def trimmed_mean_aggregate(global_model, client_weights_list, sizes, beta=0.2):
    """
    Coordinate-wise Trimmed Mean: for every parameter element, sort the
    values across clients, trim the top and bottom β fraction, then average.

    With beta=0.2 and n=10: trim floor(0.2*10)=2 from each end → mean of 6.
    This exactly removes our 2 Byzantine clients if they are indeed outliers.

    Reference: Yin et al., ICML 2018.
    """
    n          = len(client_weights_list)
    trim_count = int(np.floor(beta * n))   # = 2 for beta=0.2, n=10
    keep       = n - 2 * trim_count        # = 6

    if keep < 1:
        raise ValueError(f"Trimming too aggressive: beta={beta}, n={n}, "
                         f"trim_count={trim_count} leaves 0 clients.")

    global_sd  = global_model.state_dict()
    param_keys = [k for k in global_sd if "num_batches_tracked" not in k]

    client_models = []
    for weights in client_weights_list:
        tmp = deepcopy(global_model)
        _set_weights(tmp, weights)
        client_models.append(tmp.state_dict())

    new_sd = dict(global_sd)
    for k in param_keys:
        stacked = torch.stack(
            [cm[k].float() for cm in client_models], dim=0
        )  # (n, *shape)
        sorted_vals, _ = torch.sort(stacked, dim=0)
        trimmed        = sorted_vals[trim_count: trim_count + keep]
        new_sd[k]      = trimmed.mean(dim=0).to(global_sd[k].dtype)

    global_model.load_state_dict(new_sd)
    print(f"  [TrimmedMean beta={beta}] Trimmed {trim_count} from each end, "
          f"averaged {keep} clients")
    return global_model


# ─────────────────────────────────────────────────────────────────────────────
# F: Bulyan
# ─────────────────────────────────────────────────────────────────────────────

def bulyan_aggregate(global_model, client_weights_list, sizes, f=1):
    """
    Bulyan: Two-stage Byzantine-robust aggregation.

    Stage 1 (Selection): Iteratively apply Krum to select (n - 2f) clients.
    Stage 2 (Refinement): Apply Coordinate-wise Trimmed Mean on survivors.

    Theoretical guarantee: Bulyan requires n ≥ 4f + 3.
    With n=10 and f=2 this requires n≥11 (one short) so we use f=1.
    Paper explicitly notes this as a known Bulyan limitation (small federation).

    With f=1:
        n=10 ≥ 4(1)+3 = 7  ✓  (constraint satisfied)
        Stage 1 selects n-2f = 8 clients
        Stage 2 trims 1 from each end, averages 6

    Reference: El Mhamdi et al., "The Hidden Vulnerability of Distributed
    Learning in Byzantium", ICML 2018.
    """
    n        = len(client_weights_list)
    n_select = n - 2 * f       # = 8 for f=1, n=10

    if n < 4 * f + 3:
        raise ValueError(
            f"Bulyan requires n ≥ 4f+3. Got n={n}, f={f}. "
            f"Use f≤{(n-3)//4} for this federation size."
        )

    # ── Stage 1: Iterative Krum selection ────────────────────────────────────
    remaining_idx   = list(range(n))
    selected_idx    = []

    flat_all = np.array([_flatten(w) for w in client_weights_list])

    while len(selected_idx) < n_select:
        # Compute Krum scores on the remaining clients
        remaining_vecs  = flat_all[remaining_idx]
        f_eff           = max(0, f - (n - len(remaining_idx)))  # reduce f as pool shrinks
        neighbours      = len(remaining_idx) - f_eff - 2

        if neighbours < 1:
            # Pool too small for proper Krum; just take the rest
            selected_idx.extend(remaining_idx)
            break

        scores = _krum_scores(remaining_vecs, f_eff)
        local_winner = int(np.argmin(scores))
        global_winner = remaining_idx[local_winner]
        selected_idx.append(global_winner)
        remaining_idx.remove(global_winner)

    selected_idx = selected_idx[:n_select]
    print(f"  [Bulyan f={f}] Stage-1 selected clients: "
          f"{[i+1 for i in selected_idx]}  |  "
          f"Rejected: {[i+1 for i in range(n) if i not in selected_idx]}")

    # ── Stage 2: Coordinate-wise Trimmed Mean on survivors ───────────────────
    sel_weights = [client_weights_list[i] for i in selected_idx]
    sel_sizes   = [sizes[i] for i in selected_idx]
    # Trim 1 from each end of the 8 survivors
    trim_beta   = f / n_select          # = 1/8 = 0.125

    print(f"  [Bulyan f={f}] Stage-2 TrimmedMean on {len(sel_weights)} "
          f"clients (trim={f} each end)")

    return trimmed_mean_aggregate(global_model, sel_weights, sel_sizes,
                                  beta=trim_beta)
