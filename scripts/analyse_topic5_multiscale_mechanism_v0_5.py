#!/usr/bin/env python3
"""Target-free effective-flow, endpoint and finite-horizon gain analysis."""
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
from scipy.stats import spearmanr
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyse_topic5_rnn_motif_influence_v0_4 import hidden_before, prefix_inventory  # noqa: E402
from build_topic5_multiscale_fields_v0_5 import train_mode_to_ab  # noqa: E402
from run_topic5_multiscale_attenuation_v0_5 import instantiate, sha256_file  # noqa: E402
from src.topic5_lbss_analysis_v0_2 import endpoint_density, mask_sha256  # noqa: E402
from src.topic5_wiring_economy_rnn import build_event_tensors  # noqa: E402


DEFAULT_OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
OLD_ROOT = ROOT / "results/topic5_lbss_full_tissue_rnn_v0_3"
FIELD_ROOT = Path(
    "/home/honglab/leijiaxin/HFOsp/results/"
    "interictal_propagation_masked/template_gradient_fields/per_subject"
)
ARMS = {
    "L2M_MACRO_MATCHED_RANDOM_LR", "L3_LOCAL_PLUS_LEARNED_LR", "C_L3_ORDER_SHUFFLED"
}


def mechanism_metrics_paths(out: Path, old: Path) -> list[Path]:
    """Return the complete L2m/L3/suffix matrix, including exact v0.3 L3 reuse.

    The attenuation path enumerator cannot be reused here: by construction it
    contains only perturbation arms and therefore never includes C-suffix.
    """
    census = pd.read_csv(out / "FULL_PARENT_FIT_CENSUS.csv")
    reuse = pd.read_csv(out / "V0_3_CHECKPOINT_REUSE_AUDIT.csv")
    reused = set(reuse.loc[
        reuse.checkpoint_reuse_eligible.astype(bool), "fit_id"
    ].astype(str))
    paths = []
    for fit_id in census.fit_id.astype(str):
        for arm in sorted(ARMS):
            for seed in range(3):
                if arm == "L3_LOCAL_PLUS_LEARNED_LR" and fit_id in reused:
                    path = old / "per_fit" / fit_id / arm / f"seed{seed}" / "metrics.json"
                else:
                    path = out / "formal_units" / fit_id / arm / f"seed{seed}" / "metrics.json"
                paths.append(path)
    expected = len(census) * len(ARMS) * 3
    if expected != 378 or len(paths) != expected:
        raise RuntimeError(f"expected 378 mechanism units, found {len(paths)}")
    return paths


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    use = np.isfinite(a) & np.isfinite(b)
    if int(use.sum()) < 3 or np.std(a[use]) == 0 or np.std(b[use]) == 0:
        return float("nan")
    value = spearmanr(a[use], b[use]).statistic
    return float(value) if np.isfinite(value) else float("nan")


def prefix_inventory_for_split(tensors: dict[str, torch.Tensor], split: np.ndarray,
                               selected_split: int, max_prefixes: int) -> list[tuple[int, int]]:
    candidates = []
    for event in np.flatnonzero(split == int(selected_split)):
        valid = tensors["valid"][event].numpy()
        is_last = tensors["is_last"][event].numpy()
        for step in np.flatnonzero(valid & ~is_last):
            if step >= 1 and int(tensors["available"][event, step].sum()) >= 3:
                candidates.append((int(event), int(step)))
    if len(candidates) <= int(max_prefixes):
        return candidates
    take = np.linspace(0, len(candidates) - 1, int(max_prefixes)).round().astype(int)
    return [candidates[index] for index in np.unique(take)]


def state_step_quantities(model, h0: torch.Tensor, current: torch.Tensor):
    w = model.masked_recurrent()[0]
    u = model._inject(current[None])[:, 0].reshape(1, -1)[0]
    pre = u + h0 @ w.T + model.bias[0]
    kappa = torch.sigmoid(model.kappa_logit)[0]
    slope = kappa * (1.0 - torch.tanh(pre).square())
    h1 = (1.0 - kappa) * h0 + kappa * torch.tanh(pre)
    signed_flow = slope[:, None] * w * h0[None, :]
    jacobian = torch.eye(len(h0), device=h0.device) * (1.0 - kappa) + slope[:, None] * w
    return h1, signed_flow, jacobian


@torch.no_grad()
def finite_horizon_gain(model, x: torch.Tensor, h_before: torch.Tensor, step: int, horizon: int = 3) -> tuple[float, float]:
    h = h_before[0]
    product = torch.eye(len(h), device=h.device)
    gains = []
    for offset in range(horizon):
        index = step + offset
        if index >= len(x):
            break
        h, _, jacobian = state_step_quantities(model, h, x[index])
        product = jacobian @ product
        gains.append(float(torch.linalg.matrix_norm(product, ord=2).item()))
    return (float(max(gains)) if gains else float("nan"),
            float(gains[0]) if gains else float("nan"))


@torch.no_grad()
def empirical_output_amplification(model, x: torch.Tensor, h_before: torch.Tensor,
                                   step: int, horizon: int = 3,
                                   epsilon: float = 1e-3) -> float:
    """Held-out contact-probability response to a standardized state pulse."""
    baseline = h_before.clone()
    direction = baseline.clone()
    if float(torch.linalg.vector_norm(direction)) < 1e-8:
        direction = model._inject(x[step:step + 1])[:, 0].reshape_as(baseline)
    if float(torch.linalg.vector_norm(direction)) < 1e-8:
        direction = torch.ones_like(baseline)
    direction /= torch.linalg.vector_norm(direction).clamp_min(1e-12)
    perturbed = baseline + float(epsilon) * direction
    gains = []
    for offset in range(horizon):
        index = step + offset
        if index >= len(x):
            break
        current = x[index:index + 1]
        baseline = model._step(baseline, current)
        perturbed = model._step(perturbed, current)
        probability_base = torch.softmax(model._readout(baseline), dim=-1)
        probability_pulse = torch.softmax(model._readout(perturbed), dim=-1)
        gains.append(float(torch.linalg.vector_norm(
            probability_pulse - probability_base
        ).item() / float(epsilon)))
    return float(max(gains)) if gains else float("nan")


def mode_mapping(cache: Path, provenance: dict) -> dict[int, str]:
    # Follow the exact fit-to-field contract used by the frozen field builder.
    # A shared fit contains both train-only modes and aligns them to canonical
    # A/B.  A non-collinear own_a/own_b fit has already been filtered to its
    # designated event family, so every retained prefix contributes to that
    # fit's canonical candidate regardless of the cache's local integer label.
    scope = str(provenance["scope"])
    if scope == "own_a":
        return {0: "A", 1: "A"}
    if scope == "own_b":
        return {0: "B", 1: "B"}
    if scope != "shared":
        raise ValueError(f"unknown fit scope: {scope}")
    return train_mode_to_ab(cache, provenance["subject"],
                            np.asarray(provenance["joint_contacts"]), FIELD_ROOT)


def precedence(ranks: np.ndarray, event_indices: np.ndarray, minimum: int | None = None,
               prior: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = ranks.shape[1]
    wins = np.zeros((n, n), float)
    count = np.zeros((n, n), int)
    for index in event_indices:
        row = ranks[index]
        present = np.flatnonzero(row >= 0)
        if len(present) < 2:
            continue
        rr = row[present]
        block = np.ix_(present, present)
        count[block] += 1
        wins[block] += (rr[:, None] < rr[None, :]).astype(float)
    score = np.full((n, n), np.nan)
    threshold = max(5, int(np.ceil(0.01 * len(event_indices)))) if minimum is None else int(minimum)
    use = count >= threshold
    # source x target convention: positive means row/source precedes column/target
    score[use] = (wins[use] - wins.T[use]) / (count[use] + 2.0 * float(prior))
    weighted = np.where(np.isfinite(score), score * count, 0.0)
    denominator = np.where(np.isfinite(score), count, 0).sum(1)
    q = np.divide(weighted.sum(1), denominator, out=np.full(n, np.nan), where=denominator > 0)
    return score, count, q


def summarize_unit(out: Path, metrics_path: Path, device: torch.device,
                   max_prefixes: int) -> dict:
    model, _, metrics, plane, events, provenance, graph = instantiate(out, metrics_path, device)
    keep = events["split"] >= 0
    ranks, split, mode = events["ranks"][keep], events["split"][keep], events["mode"][keep]
    tensors = build_event_tensors(ranks)
    gain_prefixes = prefix_inventory(tensors, split, max_prefixes)
    flow_prefixes = prefix_inventory_for_split(tensors, split, 0, max_prefixes)
    if not gain_prefixes or not flow_prefixes:
        raise RuntimeError(f"missing train or held-out mechanism prefixes: {metrics['fit_id']}")
    mapping = mode_mapping(out / "cache" / metrics["fit_id"], provenance)
    flows = {"A": [], "B": []}
    gains, lag1, output_amplification = [], [], []
    # Dynamic route selection is train-only.  These frozen bundles are later
    # attenuated on held-out events by a separate script.
    for event, step in flow_prefixes:
        x = tensors["x"][event].to(device)
        h_before = hidden_before(model, x, step)
        _, signed, _ = state_step_quantities(model, h_before[0], x[step])
        flows[mapping[int(mode[event])]].append(signed.detach().cpu().numpy())
    # Gain remains a held-out trajectory audit and never selects a bundle.
    for event, step in gain_prefixes:
        x = tensors["x"][event].to(device)
        h_before = hidden_before(model, x, step)
        g3, g1 = finite_horizon_gain(model, x, h_before, step)
        gains.append(g3); lag1.append(g1)
        output_amplification.append(
            empirical_output_amplification(model, x, h_before, step)
        )
    added = graph["added_mask"].astype(bool)
    local = graph["local_mask"].astype(bool)
    arrays = {
        "contacts": np.asarray(provenance["joint_contacts"], dtype="U64"),
        "nodes_xy_mm": plane["nodes_xy_mm"], "contacts_xy_mm": plane["contacts_xy_mm"],
        "added_mask": added.astype(np.uint8), "local_mask": local.astype(np.uint8),
        "proposal_exposure": graph["exposure_count"].astype(np.float32),
    }
    row = {
        "subject": metrics["subject"], "fit_id": metrics["fit_id"], "scope": metrics["scope"],
        "arm": metrics["arm"], "seed": metrics["seed"],
        "n_prefixes": len(gain_prefixes), "n_gain_test_prefixes": len(gain_prefixes),
        "n_flow_train_prefixes": len(flow_prefixes),
        "median_G3": float(np.nanmedian(gains)), "p95_G3": float(np.nanpercentile(gains, 95)),
        "median_lag1_gain": float(np.nanmedian(lag1)), "added_edges": int(added.sum()),
        "median_empirical_output_amplification": float(np.nanmedian(output_amplification)),
        "p95_empirical_output_amplification": float(np.nanpercentile(output_amplification, 95)),
        "target_values_read": False,
    }
    for template in ("A", "B"):
        if flows[template]:
            signed = np.mean(np.stack(flows[template]), axis=0) * (local | added)
            absolute = np.mean(np.abs(np.stack(flows[template])), axis=0) * (local | added)
        else:
            signed = np.full_like(graph["strength"], np.nan, dtype=float)
            absolute = np.full_like(graph["strength"], np.nan, dtype=float)
        arrays[f"signed_flow_{template}"] = signed.astype(np.float32)
        arrays[f"absolute_flow_{template}"] = absolute.astype(np.float32)
        if np.isfinite(absolute).any():
            endpoint = endpoint_density(np.nan_to_num(absolute), added, plane["H"])
            for name, value in endpoint.items():
                arrays[f"{template}_endpoint_{name}"] = value.astype(np.float32)
            active_scores = np.where(added, np.nan_to_num(absolute), -np.inf)
            k = max(1, int(round(0.25 * added.sum())))
            flat = np.argpartition(active_scores.ravel(), -k)[-k:]
            bundle = np.zeros_like(added)
            bundle.flat[flat] = True
            bundle &= added
            arrays[f"{template}_bundle_mask"] = bundle.astype(np.uint8)
            row[f"{template}_bundle_edges"] = int(bundle.sum())
            row[f"{template}_bundle_hash"] = mask_sha256(bundle)
        else:
            row[f"{template}_bundle_edges"] = 0
            row[f"{template}_bundle_hash"] = "NOT_AVAILABLE_IN_THIS_FIT"
    # Train-only precedence is descriptive alignment, never a causal label.
    for template in ("A", "B"):
        events_for_template = np.asarray([
            index for index in np.flatnonzero(split == 0) if mapping[int(mode[index])] == template
        ], dtype=int)
        p, support, q = precedence(ranks, events_for_template)
        arrays[f"precedence_{template}"] = p.astype(np.float32)
        arrays[f"precedence_support_{template}"] = support.astype(np.int32)
        arrays[f"precedence_q_{template}"] = q.astype(np.float32)
        flow = arrays[f"signed_flow_{template}"]
        if np.isfinite(flow).any():
            contact_flow = plane["H"] @ np.nan_to_num(flow) @ plane["H"].T
            use = np.isfinite(p) & ~np.eye(len(p), dtype=bool)
            row[f"{template}_precedence_alignment"] = safe_corr(contact_flow.T[use], p[use])
        else:
            row[f"{template}_precedence_alignment"] = float("nan")
    destination = out / "mechanism/per_fit_seed" / metrics["fit_id"] / metrics["arm"] / f"seed{metrics['seed']}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    row["path"] = str(destination); row["sha256"] = sha256_file(destination)
    return row


def worker(payload: tuple[str, str, str, int]):
    out, path, device, max_prefixes = payload
    torch.set_num_threads(2)
    return summarize_unit(Path(out), Path(path), torch.device(device), max_prefixes)


def aggregate(rows: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = ["n_prefixes", "n_gain_test_prefixes", "n_flow_train_prefixes",
               "median_G3", "p95_G3", "median_lag1_gain", "added_edges",
               "median_empirical_output_amplification", "p95_empirical_output_amplification",
               "A_bundle_edges", "B_bundle_edges", "A_precedence_alignment", "B_precedence_alignment"]
    fit = rows.groupby(["subject", "fit_id", "scope", "arm"], as_index=False)[numeric].median()
    patient = fit.groupby(["subject", "arm"], as_index=False)[numeric].mean()
    pattern_rows = []
    for (subject, arm), group in rows.groupby(["subject", "arm"], sort=False):
        candidates = {}
        for template in ("A", "B"):
            matches = []
            for fit_id, fit_group in group.groupby("fit_id"):
                arrays = [np.load(path, allow_pickle=False) for path in fit_group.path]
                available = [item for item in arrays if f"{template}_endpoint_source_contact" in item.files]
                if not available:
                    continue
                pattern = np.nanmedian(np.stack([
                    np.r_[item[f"{template}_endpoint_source_contact"],
                          item[f"{template}_endpoint_target_contact"]] for item in available
                ]), axis=0)
                matches.append((fit_id, pattern))
            if matches:
                # Shared-axis patients contribute one fit; non-collinear
                # patients contribute the mean of their two geometry views.
                candidates[template] = np.nanmean(
                    np.stack([pattern for _, pattern in matches]), axis=0
                )
        if set(candidates) != {"A", "B"}:
            raise RuntimeError(f"patient mechanism A/B assembly failed: {subject} {arm}")
        destination = out / "mechanism/per_patient" / subject / f"{arm}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, A_pattern=candidates["A"], B_pattern=candidates["B"])
        pattern_rows.append({
            "subject": subject, "arm": arm, "path": str(destination),
            "n_geometry_fits": int(group.fit_id.nunique()),
        })
    patterns = pd.DataFrame(pattern_rows)
    comparisons = []
    for subject, group in patterns.groupby("subject"):
        by_arm = {row.arm: np.load(row.path, allow_pickle=False) for row in group.itertuples()}
        true = by_arm["L3_LOCAL_PLUS_LEARNED_LR"]
        shuffle = by_arm["C_L3_ORDER_SHUFFLED"]
        matched = by_arm["L2M_MACRO_MATCHED_RANDOM_LR"]
        comparisons.append({
            "subject": subject,
            "A_true_vs_suffix_r": safe_corr(true["A_pattern"], shuffle["A_pattern"]),
            "B_true_vs_suffix_r": safe_corr(true["B_pattern"], shuffle["B_pattern"]),
            "A_true_vs_l2m_r": safe_corr(true["A_pattern"], matched["A_pattern"]),
            "B_true_vs_l2m_r": safe_corr(true["B_pattern"], matched["B_pattern"]),
        })
    comparison = pd.DataFrame(comparisons)
    return patient.merge(comparison, on="subject", how="left"), patterns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--old-root", type=Path, default=OLD_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-prefixes", type=int, default=32)
    args = parser.parse_args()
    out, old = args.out_root.resolve(), args.old_root.resolve()
    if not (out / "MODEL_FIELDS_FROZEN.json").exists():
        raise RuntimeError("intact fields must be frozen before mechanism analysis")
    jobs = []
    seen = set()
    for path in mechanism_metrics_paths(out, old):
        metrics = json.loads(path.read_text())
        key = (metrics["fit_id"], metrics["arm"], int(metrics["seed"]))
        if metrics["arm"] in ARMS and key not in seen:
            jobs.append((str(out), str(path), args.device, args.max_prefixes)); seen.add(key)
    if len(jobs) != 378:
        raise RuntimeError(f"expected 378 mechanism units, found {len(jobs)}")
    rows = []
    with ProcessPoolExecutor(max_workers=min(max(1, args.workers), 8)) as executor:
        futures = [executor.submit(worker, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if index % 20 == 0:
                print(json.dumps({"completed": index, "total": len(jobs)}), flush=True)
    frame = pd.DataFrame(rows)
    root = out / "mechanism"; root.mkdir(exist_ok=True)
    frame.to_csv(root / "MECHANISM_PER_FIT_SEED.csv", index=False)
    patient, patterns = aggregate(frame, out)
    patient.to_csv(root / "MECHANISM_PER_PATIENT.csv", index=False)
    patterns.to_csv(root / "MECHANISM_PATTERN_MANIFEST.csv", index=False)
    bundle_rows = []
    for row in frame.loc[frame.arm == "L3_LOCAL_PLUS_LEARNED_LR"].itertuples():
        arrays = np.load(row.path, allow_pickle=False)
        for template in ("A", "B"):
            key = f"{template}_bundle_mask"
            if key in arrays.files:
                bundle_rows.append({
                    "subject": row.subject, "fit_id": row.fit_id, "seed": row.seed,
                    "template": template, "path": row.path,
                    "mask_sha256": mask_sha256(arrays[key]), "n_edges": int(arrays[key].sum()),
                    "target_values_read": False,
                })
    bundle = pd.DataFrame(bundle_rows)
    bundle.to_csv(root / "MODE_FLOW_BUNDLE_MANIFEST.csv", index=False)
    marker = {
        "status": "PASS_TARGET_FREE", "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_values_read": False, "units": len(frame), "patients": int(patient.subject.nunique()),
        "primary_objects": ["contact_space_effective_endpoint_pattern", "finite_horizon_G3",
                            "empirical_output_amplification", "mode_flow_bundle"],
        "bundle_selection_split": "TRAIN_ONLY",
        "attenuation_evaluation_split": "HELDOUT_TEST_IN_SEPARATE_SCRIPT",
        "exact_edge_identity_secondary": True,
        "bundle_manifest_sha256": sha256_file(root / "MODE_FLOW_BUNDLE_MANIFEST.csv"),
    }
    (out / "MECHANISM_ANALYSIS_COMPLETE.json").write_text(json.dumps(marker, indent=2) + "\n")


if __name__ == "__main__":
    main()
