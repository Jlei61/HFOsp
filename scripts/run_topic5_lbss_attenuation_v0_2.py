#!/usr/bin/env python3
"""Target-free arm-specific attenuation and frozen-field construction for LBSS v0.2."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_topic5_rnn_motif_fields_v0_4 import (  # noqa: E402
    aggregate_records,
    derive_common_contrast,
    empirical_score,
    safe_corr,
    template_for_mode,
)
from scripts.train_topic5_lbss_unit_v0_2 import (  # noqa: E402
    decision_rows,
    evaluate,
    sequence_agreement,
)
from src.topic5_lbss_analysis_v0_2 import (  # noqa: E402
    attenuate_mask,
    instantiate_lbss,
    mask_sha256,
    match_local_control_subsets,
)
from src.topic5_rnn_motif_v0_4 import rollout_with_size_head  # noqa: E402
from src.topic5_wiring_economy_rnn import build_event_tensors  # noqa: E402


ALPHAS = (0.25, 0.50, 0.75, 1.00)
TARGETS = {
    "L1_ADDED": "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
    "L2_ADDED": "L2_LOCAL_PLUS_RANDOM_LR",
    "L3_ADDED": "L3_LOCAL_PLUS_LEARNED_LR",
    "L3_MATCHED_LOCAL": "L3_LOCAL_PLUS_LEARNED_LR",
}
DEFAULT_FIELD_ROOT = Path(
    "/home/honglab/leijiaxin/HFOsp/results/"
    "interictal_propagation_masked/template_gradient_fields/per_subject"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def attenuation_unit_cache_path(
    out: Path, metrics_path: Path, target_name: str
) -> Path:
    """Return the unique, restart-safe cache path for one attenuation job."""
    metrics = json.loads(metrics_path.read_text())
    return (
        out / "attenuation" / "unit_cache" / str(metrics["fit_id"])
        / target_name / f"seed{int(metrics['seed'])}.json.gz"
    )


def write_attenuation_unit_cache(
    destination: Path,
    metrics_path: Path,
    target_name: str,
    draw_rows: list[dict],
    field_rows: list[dict],
) -> None:
    """Atomically persist a completed job before returning it to the parent.

    ProcessPoolExecutor otherwise keeps every completed result only in parent
    memory until all 372 jobs finish.  A network/session interruption near the
    end would therefore force the entire target-free attenuation experiment to
    be recomputed.  This sidecar changes scheduling durability only; scientific
    inputs, masks, doses, rollouts and aggregation remain identical.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    payload = {
        "contract": "topic5_lbss_attenuation_unit_cache_v0_3",
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "target": target_name,
        "draw_rows": draw_rows,
        "field_rows": field_rows,
        "target_values_read": False,
    }
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream, separators=(",", ":"), allow_nan=True)
    os.replace(temporary, destination)


def load_attenuation_unit_cache(
    destination: Path, metrics_path: Path, target_name: str
) -> tuple[list[dict], list[dict]] | None:
    """Load a complete unit only when its producer inputs still match."""
    if not destination.exists():
        return None
    try:
        with gzip.open(destination, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("contract") != "topic5_lbss_attenuation_unit_cache_v0_3":
            return None
        if payload.get("target_values_read") is not False:
            return None
        if payload.get("target") != target_name:
            return None
        if payload.get("metrics_sha256") != sha256_file(metrics_path):
            return None
        draw_rows = payload["draw_rows"]
        field_rows = payload["field_rows"]
        if not isinstance(draw_rows, list) or not isinstance(field_rows, list):
            return None
        for row in field_rows:
            field_path = Path(row["path"])
            rollout_path = Path(row["rollout_path"])
            if not field_path.exists() or sha256_file(field_path) != row["field_sha256"]:
                return None
            if not rollout_path.exists() or sha256_file(rollout_path) != row["rollout_sha256"]:
                return None
        return draw_rows, field_rows
    except (OSError, EOFError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def rollout_reach(sequence: list[list[int]], seed: np.ndarray, xy: np.ndarray) -> float:
    recruited = [int(contact) for rank_set in sequence[1:] for contact in rank_set]
    if not recruited or len(seed) == 0:
        return 0.0
    distance = np.linalg.norm(xy[np.asarray(recruited), None] - xy[np.asarray(seed)][None], axis=-1)
    return float(np.max(np.min(distance, axis=1)))


def evaluate_variant(
    model,
    decoder,
    events: dict,
    provenance: dict,
    plane: dict,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    keep = events["split"] >= 0
    ranks = events["ranks"][keep]
    split = events["split"][keep]
    mode = events["mode"][keep]
    tensors = build_event_tensors(ranks)
    test_idx = np.flatnonzero(split == 2)
    overall = evaluate(model, tensors, test_idx, device)
    rows = decision_rows(model, tensors, ranks, test_idx, plane["contacts_xy_mm"], device)
    q50, q80 = provenance["distance_thresholds_mm"]
    for row in rows:
        value = row["frontier_distance_mm"]
        row["distance_bin"] = (
            "invalid" if not np.isfinite(value) else
            "local" if value <= q50 else
            "intermediate" if value <= q80 else "distal"
        )
    bin_metrics = {}
    for label in ("local", "intermediate", "distal"):
        selected = [row for row in rows if row["distance_bin"] == label]
        bin_metrics[label] = {
            "n": len(selected),
            "contact_nll": float(np.mean([row["contact_nll"] for row in selected])) if selected else np.nan,
            "top1": float(np.mean([row["top1"] for row in selected])) if selected else np.nan,
        }
    starts = [np.flatnonzero(ranks[index] == 0) for index in test_idx]
    generated = rollout_with_size_head(model, decoder, starts, device)
    agreement = [sequence_agreement(ranks[index], sequence) for index, sequence in zip(test_idx, generated)]
    reach = [rollout_reach(sequence, start, plane["contacts_xy_mm"])
             for sequence, start in zip(generated, starts)]
    source_index = events["event_source_index"][keep]
    event_time = events["event_abs_time"][keep]
    records = [{
        "kept_event_index": int(index),
        "event_source_index": int(source_index[index]),
        "event_abs_time": float(event_time[index]),
        "mode": int(mode[index]),
        "seed_contacts": starts[offset].astype(int).tolist(),
        "generated_rank_sets": sequence,
    } for offset, (index, sequence) in enumerate(zip(test_idx, generated))]
    return {
        "contact_nll": overall["contact_nll"],
        "top1": overall["top1"],
        "local_nll": bin_metrics["local"]["contact_nll"],
        "intermediate_nll": bin_metrics["intermediate"]["contact_nll"],
        "distal_nll": bin_metrics["distal"]["contact_nll"],
        "local_n": bin_metrics["local"]["n"],
        "intermediate_n": bin_metrics["intermediate"]["n"],
        "distal_n": bin_metrics["distal"]["n"],
        "rollout_spearman": float(np.nanmedian(agreement)),
        "rollout_reach_mm": float(np.nanmedian(reach)),
    }, records


def median_dict(rows: list[dict]) -> dict:
    keys = rows[0].keys()
    return {key: float(np.nanmedian([row[key] for row in rows])) for key in keys}


def aggregate_draw_fields(
    records_by_draw: list[list[dict]], provenance: dict, n_contacts: int
) -> dict[str, dict[str, np.ndarray]]:
    output: dict[str, dict[str, np.ndarray]] = {}
    for template in ("A", "B"):
        draw_fields = []
        for records in records_by_draw:
            selected = [row for row in records if template_for_mode(provenance, row["mode"]) == template]
            if selected:
                draw_fields.append(aggregate_records(selected, n_contacts))
        if not draw_fields:
            continue
        output[template] = {}
        for endpoint in ("canonical_full", "seed_removed", "participation"):
            output[template][endpoint] = np.nanmedian(
                np.stack([item[endpoint] for item in draw_fields]), axis=0
            )
        output[template]["seed_removed_denominator"] = np.nanmedian(
            np.stack([item["seed_removed_denominator"] for item in draw_fields]), axis=0
        ).astype(np.int32)
    return output


def prepare_local_controls(
    out: Path, metrics_path: Path, model, plane: dict, metrics: dict
) -> dict:
    graph = np.load(metrics_path.parent / "graph.npz", allow_pickle=False)
    seed = int.from_bytes(hashlib.sha256(
        f"{metrics['fit_id']}|{metrics['seed']}|local-control".encode()
    ).digest()[:4], "little")
    matched = match_local_control_subsets(
        graph["local_mask"], graph["added_mask"], graph["strength"],
        plane["nodes_xy_mm"], plane["H"], seed=seed,
    )
    root = out / "attenuation" / "matched_local" / metrics["fit_id"] / f"seed{metrics['seed']}"
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        root / "selected_masks.npz",
        masks=matched["selected_masks"], scores=matched["selected_scores"],
    )
    serializable = {key: value for key, value in matched.items()
                    if key not in {"selected_masks", "selected_scores"}}
    (root / "match.json").write_text(json.dumps(serializable, indent=2))
    return matched


def unit_target(
    out: Path,
    metrics_path: Path,
    target_name: str,
    device: torch.device,
) -> tuple[list[dict], list[dict]]:
    model, decoder, metrics, plane, events, provenance = instantiate_lbss(out, metrics_path, device)
    metrics = dict(metrics)
    q = metrics["distance_thresholds_mm"]
    provenance = dict(provenance)
    provenance["distance_thresholds_mm"] = (float(q["q50"]), float(q["q80"]))
    base_recurrent = model.recurrent.detach().clone()
    intact_decisions = json.loads((metrics_path.parent / "distance_decisions.json").read_text())
    intact = {
        "intact_contact_nll": float(metrics["test"]["contact_nll"]),
        "intact_local_nll": float(np.mean([
            row["contact_nll"] for row in intact_decisions if row["distance_bin"] == "local"
        ])),
        "intact_intermediate_nll": float(np.mean([
            row["contact_nll"] for row in intact_decisions if row["distance_bin"] == "intermediate"
        ])),
        "intact_distal_nll": float(np.mean([
            row["contact_nll"] for row in intact_decisions if row["distance_bin"] == "distal"
        ])),
        "intact_rollout_spearman": float(metrics["rollout"]["seed_removed_spearman_median"]),
    }
    if target_name == "L3_MATCHED_LOCAL":
        matched = prepare_local_controls(out, metrics_path, model, plane, metrics)
        target_masks = [mask.astype(bool) for mask in matched["selected_masks"]]
        match_count = int(matched["n_valid_matched_draws"])
        inferential = bool(matched["inferential_eligible"])
        target_hashes = matched["selected_hashes"]
    else:
        target_masks = [model.added_mask.detach().cpu().numpy().astype(bool)]
        match_count = 1
        inferential = True
        target_hashes = [mask_sha256(target_masks[0])]
    if not target_masks:
        # A failed matched-control search is a prespecified descriptive-only
        # outcome, not a pipeline failure.  Preserve one explicit row per dose
        # so eligibility propagates through the patient-first aggregation, but
        # do not invent a counterfactual field when no active local subset met
        # the frozen calipers.
        empty = {
            "contact_nll": np.nan, "top1": np.nan,
            "local_nll": np.nan, "intermediate_nll": np.nan, "distal_nll": np.nan,
            "local_n": 0, "intermediate_n": 0, "distal_n": 0,
            "rollout_spearman": np.nan, "rollout_reach_mm": np.nan,
        }
        rows = [{
            "subject": metrics["subject"], "fit_id": metrics["fit_id"],
            "scope": metrics["scope"], "arm": metrics["arm"], "seed": metrics["seed"],
            "target": target_name, "alpha": alpha, "draw": -1,
            **empty, **intact, "target_mask_sha256": "NO_VALID_MATCH",
            "n_valid_matched_draws": match_count, "inferential_eligible": False,
            "target_values_read": False,
        } for alpha in ALPHAS]
        return rows, []

    metric_rows, field_rows = [], []
    for alpha in ALPHAS:
        draw_metrics, draw_records = [], []
        for draw, mask in enumerate(target_masks):
            with torch.no_grad():
                model.recurrent.copy_(base_recurrent)
            attenuate_mask(model, mask, alpha)
            result, records = evaluate_variant(model, decoder, events, provenance, plane, device)
            draw_metrics.append(result)
            draw_records.append(records)
            for record in records:
                record["control_draw"] = draw
            metric_rows.append({
                "subject": metrics["subject"], "fit_id": metrics["fit_id"], "scope": metrics["scope"],
                "arm": metrics["arm"], "seed": metrics["seed"], "target": target_name,
                "alpha": alpha, "draw": draw, **result,
                **intact,
                "target_mask_sha256": target_hashes[draw], "n_valid_matched_draws": match_count,
                "inferential_eligible": inferential, "target_values_read": False,
            })
        summary = median_dict(draw_metrics)
        fields = aggregate_draw_fields(draw_records, provenance, int(metrics["n_contacts"]))
        for template, payload in fields.items():
            destination = (out / "attenuation" / "fields" / "per_fit_seed" / metrics["fit_id"] /
                           target_name / f"seed{metrics['seed']}_alpha{alpha:.2f}_{template}.npz")
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination, contacts=np.asarray(provenance["contacts"], dtype="U64"), **payload)
            field_rows.append({
                "subject": metrics["subject"], "fit_id": metrics["fit_id"], "scope": metrics["scope"],
                "arm": metrics["arm"], "seed": metrics["seed"], "target": target_name,
                "alpha": alpha, "template": template, "path": str(destination),
                "field_sha256": sha256_file(destination), "n_control_draws": len(target_masks),
                "n_valid_matched_draws": match_count, "inferential_eligible": inferential,
                **summary, "target_values_read": False,
            })
        destination = out / "attenuation" / "rollouts" / metrics["fit_id"] / target_name
        destination.mkdir(parents=True, exist_ok=True)
        rollout_path = destination / f"seed{metrics['seed']}_alpha{alpha:.2f}.json.gz"
        with gzip.open(rollout_path, "wt", encoding="utf-8") as stream:
            json.dump({
                "target_hashes": target_hashes,
                "alpha": alpha,
                "draw_records": draw_records,
                "target_values_read": False,
            }, stream, separators=(",", ":"))
        for row in field_rows[-len(fields):]:
            row["rollout_path"] = str(rollout_path)
            row["rollout_sha256"] = sha256_file(rollout_path)
    with torch.no_grad():
        model.recurrent.copy_(base_recurrent)
    return metric_rows, field_rows


def unit_target_worker(payload: tuple[str, str, str, str]) -> tuple[list[dict], list[dict]]:
    """Spawn-safe wrapper; every target/unit pair writes to a unique path."""
    out, metrics_path, target_name, device = payload
    out_path = Path(out)
    metrics = Path(metrics_path)
    cache_path = attenuation_unit_cache_path(out_path, metrics, target_name)
    cached = load_attenuation_unit_cache(cache_path, metrics, target_name)
    if cached is not None:
        return cached
    torch.set_num_threads(2)
    draw_rows, field_rows = unit_target(
        out_path, metrics, target_name, torch.device(device)
    )
    write_attenuation_unit_cache(
        cache_path, metrics, target_name, draw_rows, field_rows
    )
    return draw_rows, field_rows


def aggregate_patient_fields(out: Path, field_frame: pd.DataFrame, field_root: Path) -> pd.DataFrame:
    fit_fields = {}
    for key, group in field_frame.groupby(
        ["subject", "fit_id", "scope", "arm", "target", "alpha", "template"], sort=False
    ):
        arrays = [np.load(path, allow_pickle=False) for path in group.path]
        contacts = arrays[0]["contacts"]
        payload = {"contacts": contacts}
        for endpoint in ("canonical_full", "seed_removed", "participation"):
            payload[endpoint] = np.nanmedian(np.stack([item[endpoint] for item in arrays]), axis=0)
        payload["seed_removed_denominator"] = np.sum(
            np.stack([item["seed_removed_denominator"] for item in arrays]), axis=0
        )
        fit_fields[key] = payload

    rows, manifest_rows = [], []
    subjects = sorted({key[0] for key in fit_fields})
    for subject in subjects:
        for target, arm in TARGETS.items():
            for alpha in ALPHAS:
                candidates, producers = {}, {}
                for template in ("A", "B"):
                    matches = [(key, value) for key, value in fit_fields.items()
                               if key[0] == subject and key[3] == arm and key[4] == target
                               and np.isclose(key[5], alpha) and key[6] == template]
                    if len(matches) == 1:
                        candidates[template] = matches[0][1]
                        producers[template] = matches[0][0][1]
                if set(candidates) != {"A", "B"}:
                    continue
                if not np.array_equal(candidates["A"]["contacts"], candidates["B"]["contacts"]):
                    raise RuntimeError(f"attenuated A/B support mismatch: {subject} {target} {alpha}")
                payload = {"contacts": candidates["A"]["contacts"]}
                for template in ("A", "B"):
                    for endpoint in ("canonical_full", "seed_removed", "participation", "seed_removed_denominator"):
                        payload[f"{template}_{endpoint}"] = candidates[template][endpoint]
                for endpoint in ("canonical_full", "seed_removed", "participation"):
                    common, contrast = derive_common_contrast(
                        candidates["A"][endpoint], candidates["B"][endpoint]
                    )
                    payload[f"{endpoint}_common"] = common
                    payload[f"{endpoint}_contrast"] = contrast
                destination = out / "attenuation" / "fields" / "per_patient" / subject / target / f"alpha{alpha:.2f}.npz"
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(destination, **payload)
                empirical = json.loads((field_root / f"{subject}.json").read_text())["interictal_field"]
                order = [str(value) for value in empirical["contact_order"]]
                take = np.asarray([order.index(str(value)) for value in payload["contacts"]], int)
                empirical_a = empirical_score(np.asarray(empirical["rank_a"], float)[take])
                empirical_b = empirical_score(np.asarray(empirical["rank_b"], float)[take])
                rows.append({
                    "subject": subject, "arm": arm, "target": target, "alpha": alpha,
                    "aggregation": "shared_single_fit" if producers["A"] == producers["B"] else "own_a_own_b_separate",
                    "canonical_empirical_r": float(np.nanmean([
                        safe_corr(payload["A_canonical_full"], empirical_a),
                        safe_corr(payload["B_canonical_full"], empirical_b),
                    ])),
                    "seed_removed_empirical_r": float(np.nanmean([
                        safe_corr(payload["A_seed_removed"], empirical_a),
                        safe_corr(payload["B_seed_removed"], empirical_b),
                    ])),
                    "path": str(destination), "field_sha256": sha256_file(destination),
                })
                for endpoint in (
                    "A_canonical_full", "B_canonical_full", "canonical_full_common", "canonical_full_contrast",
                    "A_seed_removed", "B_seed_removed", "seed_removed_common", "seed_removed_contrast",
                ):
                    vector = np.asarray(payload[endpoint])
                    manifest_rows.append({
                        "subject": subject, "arm": arm, "target": target, "alpha": alpha,
                        "endpoint": endpoint, "path": str(destination),
                        "file_sha256": sha256_file(destination),
                        "vector_sha256": hashlib.sha256(np.ascontiguousarray(vector).view(np.uint8)).hexdigest(),
                        "n_contacts": len(vector), "target_values_read": False,
                    })
    table = pd.DataFrame(rows)
    table.to_csv(out / "attenuation" / "attenuated_field_patient_metrics.csv", index=False)
    manifest = out / "ATTENUATED_FIELD_MANIFEST.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest, index=False)
    return table


def aggregate_metrics(out: Path, draw_frame: pd.DataFrame, field_frame: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "contact_nll", "top1", "local_nll", "intermediate_nll", "distal_nll",
        "rollout_spearman", "rollout_reach_mm", "intact_contact_nll", "intact_local_nll",
        "intact_intermediate_nll", "intact_distal_nll", "intact_rollout_spearman",
    ]
    unit_keys = ["subject", "fit_id", "scope", "arm", "seed", "target", "alpha"]
    unit_group = draw_frame.groupby(unit_keys, sort=False)
    unit = unit_group[metric_columns].median().reset_index()
    unit = unit.merge(
        unit_group.agg(
            inferential_eligible=("inferential_eligible", "min"),
            n_valid_matched_draws=("n_valid_matched_draws", "min"),
        ).reset_index(),
        on=unit_keys, how="left", validate="one_to_one",
    )
    fit_keys = ["subject", "fit_id", "arm", "target", "alpha"]
    fit_group = unit.groupby(fit_keys, sort=False)
    fit = fit_group[metric_columns].median().reset_index()
    fit = fit.merge(
        fit_group.agg(
            inferential_eligible=("inferential_eligible", "min"),
            n_valid_matched_draws=("n_valid_matched_draws", "min"),
        ).reset_index(),
        on=fit_keys, how="left", validate="one_to_one",
    )
    patient_keys = ["subject", "arm", "target", "alpha"]
    patient_group = fit.groupby(patient_keys, sort=False)
    patient = patient_group[metric_columns].mean().reset_index()
    patient = patient.merge(
        patient_group.agg(
            inferential_eligible=("inferential_eligible", "min"),
            n_valid_matched_draws=("n_valid_matched_draws", "min"),
        ).reset_index(),
        on=patient_keys, how="left", validate="one_to_one",
    )
    for endpoint in ("contact_nll", "local_nll", "intermediate_nll", "distal_nll"):
        patient[f"delta_{endpoint}"] = patient[endpoint] - patient[f"intact_{endpoint}"]
    patient["distal_selectivity"] = patient["delta_distal_nll"] - patient["delta_local_nll"]
    patient.to_csv(out / "attenuation" / "attenuation_per_patient_alpha.csv", index=False)

    auc_rows = []
    for key, group in patient.groupby(["subject", "arm", "target"], sort=False):
        subject, arm, target = key
        group = group.sort_values("alpha")
        x = np.r_[0.0, group.alpha.to_numpy(float)]
        row = {
            "subject": subject, "arm": arm, "target": target,
            "inferential_eligible": bool(group.inferential_eligible.min() >= 1.0),
            "n_valid_matched_draws_min": int(group.n_valid_matched_draws.min()),
        }
        for endpoint in ("delta_contact_nll", "delta_local_nll", "delta_distal_nll", "distal_selectivity"):
            y = np.r_[0.0, group[endpoint].to_numpy(float)]
            row[f"auc_{endpoint}"] = float(np.trapz(y, x))
            row[f"slope_{endpoint}"] = float(np.polyfit(x, y, 1)[0])
        auc_rows.append(row)
    auc = pd.DataFrame(auc_rows)
    wide = auc.pivot(index="subject", columns="target", values="auc_distal_selectivity")
    if {"L3_ADDED", "L3_MATCHED_LOCAL"}.issubset(wide):
        dd = (wide["L3_ADDED"] - wide["L3_MATCHED_LOCAL"]).rename("double_dissociation_auc")
        auc = auc.merge(dd, left_on="subject", right_index=True, how="left")
    auc.to_csv(out / "attenuation" / "attenuation_patient_auc.csv", index=False)
    return patient


def plot_stage(patient: pd.DataFrame, out: Path) -> None:
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    colors = {
        "L1_ADDED": "#8395a7", "L2_ADDED": "#a970b5",
        "L3_ADDED": "#c83e32", "L3_MATCHED_LOCAL": "#2f6fa3",
    }
    labels = {"L1_ADDED": "Extra local", "L2_ADDED": "Random nonlocal",
              "L3_ADDED": "Selected nonlocal", "L3_MATCHED_LOCAL": "Matched local"}
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0))
    for target in TARGETS:
        group = patient[patient.target == target].groupby("alpha", sort=True)
        x = np.asarray(sorted(group.groups), float)
        local = np.asarray([np.nanmedian(group.get_group(a).delta_local_nll) for a in x])
        distal = np.asarray([np.nanmedian(group.get_group(a).delta_distal_nll) for a in x])
        axes[0].plot(x, local, marker="o", color=colors[target], lw=1.4, label=labels[target])
        axes[1].plot(x, distal, marker="o", color=colors[target], lw=1.4)
    axes[0].set(xlabel="Attenuation", ylabel="Local NLL increase")
    axes[1].set(xlabel="Attenuation", ylabel="Distal NLL increase")
    l3 = patient[patient.target.isin(["L3_ADDED", "L3_MATCHED_LOCAL"])]
    for subject, group in l3.groupby("subject"):
        values = group.groupby("target").distal_selectivity.mean()
        if len(values) == 2:
            axes[2].plot([0, 1], [values["L3_MATCHED_LOCAL"], values["L3_ADDED"]],
                         color="#aeb6bd", lw=0.7, alpha=0.7)
    med = l3.groupby("target").distal_selectivity.mean()
    axes[2].scatter([0, 1], [med.get("L3_MATCHED_LOCAL", np.nan), med.get("L3_ADDED", np.nan)],
                    s=45, color=[colors["L3_MATCHED_LOCAL"], colors["L3_ADDED"]], zorder=3)
    axes[2].set_xticks([0, 1], ["Matched\nlocal", "Selected\nnonlocal"])
    axes[2].set_ylabel("Distal-selective damage")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    for label, ax in zip("ABC", axes):
        ax.axhline(0, color="#777777", lw=0.6, ls="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.18, 1.05, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
    fig.tight_layout(w_pad=2.0)
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"stage_f_attenuation_double_dissociation_interictal.{suffix}",
                    dpi=600, bbox_inches="tight")
    plt.close(fig)
    readme = figures / "README.md"
    with readme.open("a") as stream:
        stream.write(
            "\n### stage_f_attenuation_double_dissociation_interictal.png\n\n"
            "A、B 分别显示削弱各 arm 自己拥有的新增边后，局部和远端 held-out transition 的损失变化。"
            "C 在同一 L3 网络内比较匹配局部边与 task-selected nonlocal edges 的远端选择性损害。\n\n"
            "**关注点**：只有 nonlocal attenuation 对远端传播产生额外选择性损害，才支持功能双重解离。\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=Path("results/topic5_lbss_rnn_v0_2"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--field-root", type=Path, default=DEFAULT_FIELD_ROOT)
    args = parser.parse_args()
    out = args.out_root.resolve()
    if not (out / "PATHWAY_ANALYSIS_COMPLETE.json").exists():
        raise RuntimeError("pathway analysis and intact fields must be frozen before attenuation")
    jobs = []
    for target, arm in TARGETS.items():
        paths = sorted((out / "per_fit").glob(f"*/{arm}/seed*/metrics.json"))
        if len(paths) != 31 * 3:
            raise RuntimeError(f"expected 93 units for {target}, observed {len(paths)}")
        for metrics_path in paths:
            jobs.append((str(out), str(metrics_path), target, args.device))
    if args.workers <= 1:
        results = [unit_target_worker(job) for job in jobs]
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=mp.get_context("spawn")
        ) as executor:
            results = list(executor.map(unit_target_worker, jobs, chunksize=1))
    draw_rows = [row for rows, _ in results for row in rows]
    field_rows = [row for _, fields in results for row in fields]
    attenuation = out / "attenuation"
    attenuation.mkdir(exist_ok=True)
    draw_frame, field_frame = pd.DataFrame(draw_rows), pd.DataFrame(field_rows)
    draw_frame.to_csv(attenuation / "attenuation_per_fit_seed_draw.csv", index=False)
    field_frame.to_csv(attenuation / "attenuated_field_per_fit_seed.csv", index=False)
    field_counts = field_frame.groupby("target").size().to_dict()
    expected_simple = 42 * 3 * 4
    for target in ("L1_ADDED", "L2_ADDED", "L3_ADDED"):
        if int(field_counts.get(target, 0)) != expected_simple:
            raise RuntimeError(
                f"unexpected {target} fit-seed-template field count: "
                f"{field_counts.get(target, 0)} != {expected_simple}"
            )
    if int(field_counts.get("L3_MATCHED_LOCAL", 0)) < 1:
        raise RuntimeError("no matched-local fields were evaluable in the cohort")
    field_root = args.field_root.resolve()
    if len(list(field_root.glob("*.json"))) < 21:
        raise RuntimeError(f"canonical empirical field root is incomplete: {field_root}")
    patient_fields = aggregate_patient_fields(out, field_frame, field_root)
    patient = aggregate_metrics(out, draw_frame, field_frame)
    plot_stage(patient, out)
    manifest = out / "ATTENUATED_FIELD_MANIFEST.csv"
    skipped_units = draw_frame.loc[draw_frame.draw < 0, ["fit_id", "seed", "target"]].drop_duplicates()
    ineligible_units = draw_frame.loc[
        ~draw_frame.inferential_eligible.astype(bool), ["fit_id", "seed", "target"]
    ].drop_duplicates()
    (out / "ATTENUATED_FIELD_MANIFEST.json").write_text(json.dumps({
        "status": "FROZEN_WITH_DESCRIPTIVE_ONLY_CONTROLS",
        "n_scheduled_fit_seed_target_alpha": 31 * 3 * 4 * 4,
        "n_field_rows": len(field_frame),
        "field_rows_by_target": {str(key): int(value) for key, value in field_counts.items()},
        "n_patient_target_alpha": len(patient_fields),
        "n_skipped_no_valid_control_units": len(skipped_units),
        "n_inferentially_ineligible_units": len(ineligible_units),
        "manifest": str(manifest), "manifest_sha256": sha256_file(manifest),
        "local_control_candidate_draws": 20_000,
        "local_control_valid_target": 500,
        "local_control_evaluated_draws": 16,
        "target_access_count": 0, "target_values_read": False,
    }, indent=2) + "\n")
    (out / "ATTENUATION_COMPLETE.json").write_text(json.dumps({
        "status": "PASS_WITH_DESCRIPTIVE_ONLY_CONTROLS", "target_values_read": False,
        "n_metric_draw_rows": len(draw_frame), "n_field_rows": len(field_frame),
        "n_restart_safe_unit_caches": len(list(
            (out / "attenuation" / "unit_cache").glob("*/*/seed*.json.gz")
        )),
        "n_skipped_no_valid_control_units": len(skipped_units),
        "n_inferentially_ineligible_units": len(ineligible_units),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
