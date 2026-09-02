#!/usr/bin/env python3
"""Target-free same-mode, cross-mode and matched-random flow attenuation.

This analysis asks whether the same fitted L3 network uses different effective
added-edge bundles for its two train-only event modes.  Exact edge identity is
not interpreted as anatomy; the inferential object is patient-level selective
damage to held-out suffix likelihood.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyse_topic5_multiscale_mechanism_v0_5 import (  # noqa: E402
    mechanism_metrics_paths, mode_mapping,
)
from run_topic5_multiscale_attenuation_v0_5 import instantiate  # noqa: E402
from scripts.train_topic5_lbss_unit_v0_2 import decision_rows, evaluate  # noqa: E402
from src.topic5_lbss_analysis_v0_2 import attenuate_mask, mask_sha256  # noqa: E402
from src.topic5_wiring_economy_rnn import build_event_tensors  # noqa: E402


DEFAULT_OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
DEFAULT_OLD = ROOT / "results/topic5_lbss_full_tissue_rnn_v0_3"
TEMPLATES = ("A", "B")


def paired_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    nonzero = values[np.abs(values) > 1e-9]
    p = 1.0 if len(nonzero) == 0 else float(
        wilcoxon(nonzero, alternative="greater", method="auto").pvalue
    )
    return {
        "n": int(len(values)), "median": float(np.median(values)),
        "n_positive": int(np.sum(values > 1e-9)),
        "n_negative": int(np.sum(values < -1e-9)),
        "n_tied": int(np.sum(np.abs(values) <= 1e-9)),
        "wilcoxon_p_greater": p,
    }


def _descriptor(mask: np.ndarray, absolute_flow: np.ndarray, strength: np.ndarray,
                distance: np.ndarray, H: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, bool)
    source_degree = mask.sum(0).astype(float)
    target_degree = mask.sum(1).astype(float)
    source_contact = H @ source_degree
    target_contact = H @ target_degree
    source_contact /= max(source_contact.sum(), 1e-12)
    target_contact /= max(target_contact.sum(), 1e-12)
    return np.r_[
        float(np.nansum(np.where(mask, absolute_flow, 0.0))),
        float(np.nansum(np.where(mask, strength, 0.0))),
        float(np.mean(distance[mask])),
        np.sort(source_degree), np.sort(target_degree),
        source_contact, target_contact,
    ]


def _inside_match_calipers(candidate: np.ndarray, target: np.ndarray,
                           n_nodes: int, n_contacts: int) -> bool:
    def relative(left: float, right: float) -> float:
        return abs(left - right) / max(abs(right), 1e-8)
    source = slice(3, 3 + n_nodes)
    sink = slice(3 + n_nodes, 3 + 2 * n_nodes)
    source_contact = slice(3 + 2 * n_nodes, 3 + 2 * n_nodes + n_contacts)
    target_contact = slice(3 + 2 * n_nodes + n_contacts,
                           3 + 2 * n_nodes + 2 * n_contacts)
    return bool(
        relative(candidate[0], target[0]) <= 0.25
        and relative(candidate[1], target[1]) <= 0.25
        and relative(candidate[2], target[2]) <= 0.15
        and np.abs(candidate[source] - target[source]).sum() /
            max(1.0, target[source].sum()) <= 0.30
        and np.abs(candidate[sink] - target[sink]).sum() /
            max(1.0, target[sink].sum()) <= 0.30
        and np.abs(candidate[source_contact] - target[source_contact]).sum() <= 0.75
        and np.abs(candidate[target_contact] - target[target_contact]).sum() <= 0.75
    )


def matched_random_masks(added: np.ndarray, target: np.ndarray, absolute_flow: np.ndarray,
                         strength: np.ndarray, distance: np.ndarray, H: np.ndarray,
                         seed: int, draws: int = 2000, keep: int = 8) -> tuple[list[np.ndarray], dict]:
    edges = np.argwhere(added)
    k = int(target.sum())
    if k < 1 or len(edges) < k:
        return [], {"eligible": False, "reason": "INSUFFICIENT_ACTIVE_ADDED_EDGES"}
    target_desc = _descriptor(target, absolute_flow, strength, distance, H)
    scale = np.maximum(np.abs(target_desc), np.nanmedian(np.abs(target_desc)) + 1e-6)
    rng = np.random.default_rng(seed)
    candidates, accepted = [], []
    seen = {mask_sha256(target)}
    no_swap_streak = 0
    for _ in range(int(draws)):
        # Randomize the target bundle by directed double-edge swaps restricted
        # to edges that actually exist in this trained L3 graph.  This keeps
        # source out-degree, target in-degree and contact endpoint density
        # exactly fixed while breaking source-target pairing.
        mask = target.copy()
        successful = 0
        for _attempt in range(max(200, 30 * k)):
            active = np.argwhere(mask)
            if len(active) < 2:
                break
            left, right = rng.choice(len(active), size=2, replace=False)
            t1, s1 = map(int, active[left]); t2, s2 = map(int, active[right])
            if t1 == t2 or s1 == s2:
                continue
            if not (added[t2, s1] and added[t1, s2]):
                continue
            if mask[t2, s1] or mask[t1, s2]:
                continue
            mask[t1, s1] = False; mask[t2, s2] = False
            mask[t2, s1] = True; mask[t1, s2] = True
            successful += 1
            if successful >= max(1, int(np.ceil(0.25 * k))):
                break
        if successful == 0 or np.mean(mask != target) < 0.25 * (2.0 * k / mask.size):
            no_swap_streak += 1
            if no_swap_streak >= 200 and not candidates:
                break
            continue
        no_swap_streak = 0
        digest = mask_sha256(mask)
        if digest in seen:
            continue
        seen.add(digest)
        desc = _descriptor(mask, absolute_flow, strength, distance, H)
        score = float(np.sqrt(np.nanmean(np.square((desc - target_desc) / scale))))
        candidates.append((score, digest, mask))
        if _inside_match_calipers(desc, target_desc, added.shape[0], H.shape[0]):
            accepted.append((score, digest, mask))
    candidates.sort(key=lambda item: (item[0], item[1]))
    accepted.sort(key=lambda item: (item[0], item[1]))
    if not candidates:
        # Some very sparse fitted graphs admit no active degree-preserving
        # swap.  Keep a nearest arbitrary active-edge subset for visualization
        # only; eligibility remains false and it is excluded from inference.
        for _ in range(min(int(draws), 2000)):
            choice = rng.choice(len(edges), k, replace=False)
            mask = np.zeros_like(added, dtype=bool)
            mask[tuple(edges[choice].T)] = True
            digest = mask_sha256(mask)
            if digest in seen:
                continue
            seen.add(digest)
            desc = _descriptor(mask, absolute_flow, strength, distance, H)
            score = float(np.sqrt(np.nanmean(np.square((desc - target_desc) / scale))))
            candidates.append((score, digest, mask))
        candidates.sort(key=lambda item: (item[0], item[1]))
    inferential = len(accepted) >= int(keep)
    pool = accepted if inferential else candidates
    selected = [item[2] for item in pool[: int(keep)]]
    return selected, {
        "eligible": inferential, "draws": int(draws),
        "unique_candidates": len(candidates), "valid_caliper_candidates": len(accepted),
        "selected": len(selected), "selection_pool": "CALIPER_VALID" if inferential else "NEAREST_DESCRIPTIVE",
        "best_score": float(pool[0][0]) if pool else float("nan"),
        "worst_selected_score": float(pool[min(len(pool), keep) - 1][0]) if pool else float("nan"),
        "calipers": {
            "total_train_flow_relative": 0.25, "total_abs_weight_relative": 0.25,
            "mean_length_relative": 0.15, "source_degree_l1_relative": 0.30,
            "target_degree_l1_relative": 0.30,
            "source_endpoint_density_l1": 0.75, "target_endpoint_density_l1": 0.75,
        },
        "randomization": "directed_degree_preserving_double_edge_swap_within_active_added_graph",
    }


def evaluate_mode(model, events: dict, provenance: dict, plane: dict,
                  template: str, device: torch.device) -> dict:
    keep = events["split"] >= 0
    ranks = events["ranks"][keep]
    split = events["split"][keep]
    modes = events["mode"][keep]
    mapping = mode_mapping(Path(provenance["cache_path"]), provenance)
    test = np.asarray([
        index for index in np.flatnonzero(split == 2)
        if mapping[int(modes[index])] == template
    ], dtype=int)
    tensors = build_event_tensors(ranks)
    overall = evaluate(model, tensors, test, device)
    rows = decision_rows(model, tensors, ranks, test, plane["contacts_xy_mm"], device)
    # Formal v0.5 inference uses the frozen local-backbone radius. Quantile
    # bins inherited from rollout diagnostics are descriptive only.
    r_local_mm = float(provenance["r_local_mm"])
    distal = [row["contact_nll"] for row in rows
              if np.isfinite(row["frontier_distance_mm"])
              and row["frontier_distance_mm"] > r_local_mm]
    local = [row["contact_nll"] for row in rows
             if np.isfinite(row["frontier_distance_mm"])
             and row["frontier_distance_mm"] <= r_local_mm]
    return {
        "contact_nll": float(overall["contact_nll"]),
        "local_nll": float(np.mean(local)) if local else float("nan"),
        "distal_nll": float(np.mean(distal)) if distal else float("nan"),
        "n_events": int(len(test)), "n_local": int(len(local)), "n_distal": int(len(distal)),
    }


def summarize_unit(out: Path, metrics_path: Path, device: torch.device) -> list[dict]:
    model, _, metrics, plane, events, provenance, graph = instantiate(out, metrics_path, device)
    cache = out / "cache" / metrics["fit_id"]
    provenance = dict(provenance)
    provenance["cache_path"] = str(cache)
    mechanism = np.load(
        out / "mechanism/per_fit_seed" / metrics["fit_id"] /
        metrics["arm"] / f"seed{metrics['seed']}.npz", allow_pickle=False
    )
    base = model.recurrent.detach().clone()
    rows = []
    for template in TEMPLATES:
        intact = evaluate_mode(model, events, provenance, plane, template, device)
        same = mechanism[f"{template}_bundle_mask"].astype(bool)
        cross = mechanism[f"{'B' if template == 'A' else 'A'}_bundle_mask"].astype(bool)
        random_masks, match = matched_random_masks(
            graph["added_mask"].astype(bool), same,
            mechanism[f"absolute_flow_{template}"], graph["strength"], plane["D_mm"],
            plane["H"], seed=int.from_bytes(hashlib.sha256(
                f"{metrics['fit_id']}|{metrics['seed']}|{template}|mode-random".encode()
            ).digest()[:4], "little"),
        )
        conditions = [("SAME_MODE", same), ("CROSS_MODE", cross)] + [
            ("MATCHED_RANDOM", mask) for mask in random_masks
        ]
        for draw, (condition, mask) in enumerate(conditions):
            with torch.no_grad():
                model.recurrent.copy_(base)
            attenuate_mask(model, mask, 1.0)
            result = evaluate_mode(model, events, provenance, plane, template, device)
            rows.append({
                "subject": metrics["subject"], "fit_id": metrics["fit_id"],
                "scope": metrics["scope"], "seed": int(metrics["seed"]),
                "template": template, "condition": condition,
                "draw": draw if condition == "MATCHED_RANDOM" else 0,
                "mask_sha256": mask_sha256(mask), "bundle_edges": int(mask.sum()),
                "intact_contact_nll": intact["contact_nll"],
                "intact_local_nll": intact["local_nll"],
                "intact_distal_nll": intact["distal_nll"],
                **{key: value for key, value in result.items()},
                "contact_damage": result["contact_nll"] - intact["contact_nll"],
                "local_damage": result["local_nll"] - intact["local_nll"],
                "distal_damage": result["distal_nll"] - intact["distal_nll"],
                "distal_selectivity": (
                    result["distal_nll"] - intact["distal_nll"]
                    - result["local_nll"] + intact["local_nll"]
                ),
                "random_match_eligible": bool(match["eligible"]),
                "random_match_valid_candidates": int(match.get("valid_caliper_candidates", 0)),
                "random_match_best_score": match.get("best_score", float("nan")),
                "target_values_read": False,
            })
    with torch.no_grad():
        model.recurrent.copy_(base)
    return rows


def worker(payload: tuple[str, str, str]):
    out, path, device = payload
    torch.set_num_threads(2)
    return summarize_unit(Path(out), Path(path), torch.device(device))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out, old = args.out_root.resolve(), args.old_root.resolve()
    if not (out / "MECHANISM_ANALYSIS_COMPLETE.json").exists():
        raise RuntimeError("mode-flow attenuation requires frozen mechanism bundles")
    # Same-mode versus cross-mode attenuation is identifiable only when A and
    # B are generated by the same fitted recurrent network.  Shared fits meet
    # that contract.  Non-collinear own_a/own_b patients use two separately
    # trained fits and therefore have no within-network cross-mode bundle.
    paths = []
    for path in mechanism_metrics_paths(out, old):
        if "L3_LOCAL_PLUS_LEARNED_LR" not in str(path):
            continue
        metrics = json.loads(path.read_text())
        if str(metrics["scope"]) == "shared":
            paths.append(path)
    if len(paths) != 42:
        raise RuntimeError(f"expected 42 shared-fit L3 units, found {len(paths)}")
    rows = []
    with ProcessPoolExecutor(max_workers=min(max(1, args.workers), 8)) as executor:
        futures = [executor.submit(worker, (str(out), str(path), args.device)) for path in paths]
        for index, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result())
            if index % 10 == 0:
                print(json.dumps({"completed": index, "total": len(paths)}), flush=True)
    draw = pd.DataFrame(rows)
    root = out / "mechanism"
    draw.to_csv(root / "MODE_FLOW_ATTENUATION_PER_DRAW.csv", index=False)
    metrics = ["contact_damage", "local_damage", "distal_damage", "distal_selectivity"]
    condition = draw.groupby(
        ["subject", "fit_id", "scope", "seed", "template", "condition"], as_index=False
    )[metrics].median()
    seed = condition.groupby(
        ["subject", "fit_id", "scope", "template", "condition"], as_index=False
    )[metrics].median()
    patient = seed.groupby(["subject", "template", "condition"], as_index=False)[metrics].mean()
    eligibility = draw.groupby(["subject", "template"], as_index=False).agg(
        random_match_eligible=("random_match_eligible", "all"),
        random_match_valid_candidates=("random_match_valid_candidates", "min"),
    )
    patient = patient.merge(eligibility, on=["subject", "template"], validate="many_to_one")
    patient.to_csv(root / "MODE_FLOW_ATTENUATION_PER_PATIENT.csv", index=False)
    pivot = patient.pivot(index=["subject", "template"], columns="condition", values=metrics)
    same_cross = pivot["distal_selectivity"]["SAME_MODE"] - pivot["distal_selectivity"]["CROSS_MODE"]
    same_random = pivot["distal_selectivity"]["SAME_MODE"] - pivot["distal_selectivity"]["MATCHED_RANDOM"]
    eligible = eligibility.set_index(["subject", "template"]).random_match_eligible
    same_random = same_random.loc[eligible.reindex(same_random.index).fillna(False).astype(bool)]
    patient_level = pd.DataFrame({
        "same_minus_cross": same_cross, "same_minus_random": same_random,
    }).groupby(level="subject").mean()
    summary = {
        "contract": "topic5_mode_flow_attenuation_v0_5",
        "status": "PASS_TARGET_FREE", "created_utc": datetime.now(timezone.utc).isoformat(),
        "units": len(paths), "patients": int(patient.subject.nunique()),
        "eligibility": "SHARED_FIT_ONLY_SAME_RNN_CONTAINS_BOTH_TRAIN_ONLY_MODES",
        "noncollinear_patients": "NOT_IDENTIFIABLE_FOR_WITHIN_NETWORK_CROSS_MODE_ATTENUATION",
        "same_minus_cross_distal_selectivity": paired_summary(patient_level.same_minus_cross),
        "same_minus_matched_random_distal_selectivity": paired_summary(
            same_random.groupby(level="subject").mean().to_numpy()
        ),
        "matched_random_eligible_patients": int(
            eligibility.groupby("subject").random_match_eligible.all().sum()
        ),
        "exact_edge_identity_secondary": True, "target_values_read": False,
    }
    (root / "MODE_FLOW_ATTENUATION_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "MODE_FLOW_ATTENUATION_COMPLETE.json").write_text(json.dumps({
        "status": "PASS_TARGET_FREE", "units": len(paths), "target_values_read": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
