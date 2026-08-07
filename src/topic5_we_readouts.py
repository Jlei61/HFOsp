"""Per-unit functional portraits and module lesions, taken while the model lives.

Both of these need the trained weights, so they run at the end of a training
unit rather than as a later pass over saved graphs.  Nothing here is ever a
training signal: modules are found after the fact, the lesion is applied without
retraining, and the mode labels only ever reach the evaluation side.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np
import torch

from src.topic5_we_graph_analysis import contiguous_random_lesion, module_of_each_node

N_RANK_BINS = 6


@torch.no_grad()
def unit_tuning(model, tensors: Dict[str, torch.Tensor], event_index: np.ndarray,
                mode: np.ndarray, device, batch_size: int = 256) -> np.ndarray:
    """Mean activation per tissue unit at each stage of an event, and per mode.

    Columns are six equal slices of normalised event progress followed by the
    mean over mode-0 events and the mean over mode-1 events.  Progress is binned
    rather than indexed by raw step so patients with different event lengths line
    up on the same axis.
    """
    if model.arm == "STATIC_CONTACT" or event_index.size == 0:
        return np.zeros((0, N_RANK_BINS + 2), np.float32)
    model.eval()
    n_units = model.n_nodes
    total = np.zeros((n_units, N_RANK_BINS + 2), np.float64)
    weight = np.zeros(N_RANK_BINS + 2)

    for start in range(0, event_index.size, batch_size):
        chunk = event_index[start:start + batch_size]
        idx = torch.as_tensor(chunk)
        x = tensors["x"][idx].to(device)
        valid = tensors["valid"][idx].to(device)
        steps = x.shape[1]
        h = torch.zeros(x.shape[0], n_units * model.state_dim, device=device)
        lengths = valid.float().sum(1).clamp_min(1.0)
        chunk_mode = mode[chunk]
        for t in range(steps):
            h = model._step(h, x[:, t])
            unit = h.reshape(h.shape[0], n_units, model.state_dim).mean(-1)
            live = valid[:, t]
            if not bool(live.any()):
                continue
            progress = torch.full((x.shape[0],), float(t), device=device) / lengths
            bins = torch.clamp((progress * N_RANK_BINS).long(), 0, N_RANK_BINS - 1)
            for b in range(N_RANK_BINS):
                pick = live & (bins == b)
                if bool(pick.any()):
                    total[:, b] += unit[pick].sum(0).double().cpu().numpy()
                    weight[b] += float(pick.sum())
            for m in (0, 1):
                pick = live & torch.as_tensor(chunk_mode == m, device=device)
                if bool(pick.any()):
                    total[:, N_RANK_BINS + m] += unit[pick].sum(0).double().cpu().numpy()
                    weight[N_RANK_BINS + m] += float(pick.sum())
    return (total / np.maximum(weight, 1.0)).astype(np.float32)


def _cut_cost(mask: np.ndarray, strength: np.ndarray, distance: np.ndarray,
              members: np.ndarray) -> tuple[float, float, float]:
    touched = np.zeros_like(mask, bool)
    touched[members, :] = True
    touched[:, members] = True
    cut = mask & touched
    if not cut.any():
        return 0.0, 0.0, 0.0
    return float(cut.sum()), float(strength[cut].sum()), float(distance[cut].mean())


def matched_contiguous_patch(mask: np.ndarray, strength: np.ndarray, distance: np.ndarray,
                             nodes_xy: np.ndarray, members: np.ndarray,
                             seed: int = 0, n_candidates: int = 40) -> np.ndarray:
    """A patch of the same size whose cut resembles the module's cut.

    A Louvain module on a plane is a patch, so a scattered control compares a
    patch against confetti and the patch wins for free.  Among same-size patches
    the one whose edge count, total strength and mean cut length are closest to
    the module's is chosen, so what remains is the module structure itself.
    """
    target = np.array(_cut_cost(mask, strength, distance, members))
    scale = np.maximum(np.abs(target), 1e-9)
    best, best_score = None, np.inf
    exclude = set(members.tolist())
    for k in range(n_candidates):
        patch = contiguous_random_lesion(nodes_xy, len(members), seed=seed * 1000 + k)
        if len(set(patch.tolist()) & exclude) > 0.5 * len(members):
            continue
        score = float(np.abs(np.array(_cut_cost(mask, strength, distance, patch)) - target) @ (1.0 / scale))
        if score < best_score:
            best, best_score = patch, score
    if best is None:
        best = contiguous_random_lesion(nodes_xy, len(members), seed=seed)
    return best


@torch.no_grad()
def module_lesion(model, nodes_xy: np.ndarray, evaluate: Callable[[], Dict[str, float]],
                  evaluate_mode: Callable[[int], Dict[str, float]],
                  seed: int = 0) -> Dict[str, Any]:
    """Cut a whole module's recurrent traffic, without retraining, and re-score.

    Recurrent connections are cut rather than units deleted: deleting units would
    also change what the fixed read-out sees, and then a drop in prediction would
    not distinguish "this module computes something" from "these contacts lost
    their observation support".
    """
    if model.arm == "STATIC_CONTACT":
        return {}
    mask = model.node_mask.detach().cpu().numpy() > 0
    strength = model.edge_strength().cpu().numpy()
    distance = model.D_mm.detach().cpu().numpy()
    membership, communities = module_of_each_node(mask, seed=seed)
    if len(communities) < 2:
        return {"n_modules": len(communities), "skipped": "fewer than two modules"}

    base = evaluate()
    base_mode = {m: evaluate_mode(m) for m in (0, 1)}
    original = model.node_mask.detach().clone()
    largest = max(communities, key=len)
    members = np.array(sorted(largest), int)

    def score_with(cut_members: np.ndarray) -> Dict[str, Any]:
        cut = original.clone()
        cut[torch.as_tensor(cut_members), :] = 0.0
        cut[:, torch.as_tensor(cut_members)] = 0.0
        model.node_mask.copy_(cut)
        out = {"next_bce": evaluate()["next_bce"]}
        for m in (0, 1):
            value = evaluate_mode(m)["next_bce"]
            out[f"mode{m}"] = value
        model.node_mask.copy_(original)
        return out

    module_scores = score_with(members)
    patch = matched_contiguous_patch(mask, strength, distance, nodes_xy, members, seed=seed)
    patch_scores = score_with(patch)

    def delta(after: float, before: float) -> float:
        return float(after - before) if np.isfinite(after) and np.isfinite(before) else float("nan")

    result: Dict[str, Any] = {
        "n_modules": len(communities),
        "module_size": int(len(members)),
        "module_members": members.tolist(),
        "matched_patch": patch.tolist(),
        "base_next_bce": float(base["next_bce"]),
        "module_delta_next_bce": delta(module_scores["next_bce"], base["next_bce"]),
        "matched_patch_delta_next_bce": delta(patch_scores["next_bce"], base["next_bce"]),
        "module_cut": _cut_cost(mask, strength, distance, members),
        "matched_patch_cut": _cut_cost(mask, strength, distance, patch),
    }
    for m in (0, 1):
        result[f"module_delta_mode{m}"] = delta(module_scores[f"mode{m}"],
                                                base_mode[m]["next_bce"])
        result[f"matched_patch_delta_mode{m}"] = delta(patch_scores[f"mode{m}"],
                                                       base_mode[m]["next_bce"])
    a, b = result["module_delta_mode0"], result["module_delta_mode1"]
    result["mode_selectivity"] = (float(abs(a - b) / max(abs(a) + abs(b), 1e-9))
                                  if np.isfinite(a) and np.isfinite(b) else float("nan"))
    return result
