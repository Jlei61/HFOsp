#!/usr/bin/env python3
"""Validation-only finite-horizon gain matching for L3 versus L2m.

The higher-gain arm is rescaled on validation trajectories only.  The frozen
scale is then evaluated on held-out interictal decisions and converted to the
same patient-level model fields before any early-ictal value is accessible.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyse_topic5_multiscale_mechanism_v0_5 import (  # noqa: E402
    finite_horizon_gain, hidden_before, mechanism_metrics_paths,
)
from build_topic5_rnn_motif_fields_v0_4 import derive_common_contrast  # noqa: E402
from build_topic5_multiscale_fields_v0_5 import remap_record_modes, sha256_file, train_mode_to_ab  # noqa: E402
from run_topic5_multiscale_attenuation_v0_5 import aggregate_draw_fields, instantiate  # noqa: E402
from run_topic5_lbss_attenuation_v0_2 import evaluate_variant  # noqa: E402
from src.topic5_wiring_economy_rnn import build_event_tensors  # noqa: E402


DEFAULT_OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
DEFAULT_OLD = ROOT / "results/topic5_lbss_full_tissue_rnn_v0_3"
ARMS = ("L2M_MACRO_MATCHED_RANDOM_LR", "L3_LOCAL_PLUS_LEARNED_LR")
FIELD_ROOT = Path(
    "/home/honglab/leijiaxin/HFOsp/results/"
    "interictal_propagation_masked/template_gradient_fields/per_subject"
)


def sampled_prefixes(tensors: dict, split: np.ndarray, selected: int,
                     maximum: int = 32) -> list[tuple[int, int]]:
    candidates = []
    for event in np.flatnonzero(split == int(selected)):
        valid = tensors["valid"][event].numpy()
        is_last = tensors["is_last"][event].numpy()
        for step in np.flatnonzero(valid & ~is_last):
            if step >= 1 and int(tensors["available"][event, step].sum()) >= 3:
                candidates.append((int(event), int(step)))
    if len(candidates) <= maximum:
        return candidates
    take = np.linspace(0, len(candidates) - 1, maximum).round().astype(int)
    return [candidates[index] for index in np.unique(take)]


@torch.no_grad()
def median_g3(model, events: dict, device: torch.device, selected_split: int = 1) -> float:
    keep = events["split"] >= 0
    ranks, split = events["ranks"][keep], events["split"][keep]
    tensors = build_event_tensors(ranks)
    prefixes = sampled_prefixes(tensors, split, selected_split)
    gains = []
    for event, step in prefixes:
        x = tensors["x"][event].to(device)
        h = hidden_before(model, x, step)
        value, _ = finite_horizon_gain(model, x, h, step)
        gains.append(value)
    return float(np.nanmedian(gains)) if gains else float("nan")


def find_scale(model, events: dict, target: float, device: torch.device) -> tuple[float, float]:
    base = model.recurrent.detach().clone()
    intact = median_g3(model, events, device)
    if not np.isfinite(target) or not np.isfinite(intact) or intact <= target * 1.001:
        return 1.0, intact
    lower, upper = 0.0, 1.0
    best_scale, best_gain = 1.0, intact
    for _ in range(18):
        scale = 0.5 * (lower + upper)
        with torch.no_grad():
            model.recurrent.copy_(base * scale)
        gain = median_g3(model, events, device)
        if abs(gain - target) < abs(best_gain - target):
            best_scale, best_gain = scale, gain
        if gain > target:
            upper = scale
        else:
            lower = scale
    with torch.no_grad():
        model.recurrent.copy_(base * best_scale)
    return float(best_scale), float(best_gain)


def unit_pair(out: Path, paths: dict[str, Path], device: torch.device) -> tuple[list[dict], list[dict]]:
    loaded = {arm: instantiate(out, paths[arm], device) for arm in ARMS}
    models = {arm: loaded[arm][0] for arm in ARMS}
    events = {arm: loaded[arm][4] for arm in ARMS}
    validation_gain = {arm: median_g3(models[arm], events[arm], device) for arm in ARMS}
    target = float(np.nanmin(list(validation_gain.values())))
    higher = max(ARMS, key=lambda arm: validation_gain[arm])
    scale, matched_gain = find_scale(models[higher], events[higher], target, device)
    scales = {arm: (scale if arm == higher else 1.0) for arm in ARMS}
    matched = {arm: (matched_gain if arm == higher else validation_gain[arm]) for arm in ARMS}
    metric_rows, field_rows = [], []
    for arm in ARMS:
        model, decoder, metrics, plane, arm_events, provenance, _ = loaded[arm]
        result, records = evaluate_variant(model, decoder, arm_events, provenance, plane, device)
        metric_rows.append({
            "subject": metrics["subject"], "fit_id": metrics["fit_id"],
            "scope": metrics["scope"], "seed": int(metrics["seed"]), "arm": arm,
            "validation_G3_intact": validation_gain[arm],
            "validation_G3_matched": matched[arm], "recurrent_scale": scales[arm],
            **result, "target_values_read": False,
        })
        cache = out / "cache" / metrics["fit_id"]
        records = remap_record_modes(records, cache)
        if provenance["scope"] == "shared":
            mapping = train_mode_to_ab(cache, provenance["subject"],
                                       np.asarray(provenance["joint_contacts"]), FIELD_ROOT)
        else:
            label = "A" if provenance["scope"] == "own_a" else "B"
            mapping = {0: label, 1: label}
        for row in records:
            row["mode"] = 0 if mapping[int(row["mode"])] == "A" else 1
        proxy = dict(provenance)
        proxy["scope"] = "shared"
        proxy["subject"] = metrics["subject"]
        proxy["joint_contacts"] = provenance["joint_contacts"]
        # aggregate_draw_fields remaps modes itself, so use a direct one-draw
        # construction with a temporary identity cache-independent mapping.
        grouped = defaultdict(list)
        for row in records:
            grouped["A" if int(row["mode"]) == 0 else "B"].append(row)
        from build_topic5_rnn_motif_fields_v0_4 import aggregate_records
        for template, selected in grouped.items():
            payload = aggregate_records(selected, int(provenance["n_joint_contacts"]))
            destination = (out / "gain_adjusted_fields/per_fit_seed" / metrics["fit_id"] /
                           arm / f"seed{metrics['seed']}_{template}.npz")
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination,
                                contacts=np.asarray(provenance["joint_contacts"], dtype="U64"), **payload)
            field_rows.append({
                "subject": metrics["subject"], "fit_id": metrics["fit_id"],
                "scope": metrics["scope"], "seed": int(metrics["seed"]), "arm": arm,
                "template": template, "path": str(destination),
                "sha256": sha256_file(destination), "target_values_read": False,
            })
    return metric_rows, field_rows


def worker(payload: tuple[str, dict[str, str], str]):
    out, paths, device = payload
    torch.set_num_threads(2)
    return unit_pair(Path(out), {arm: Path(path) for arm, path in paths.items()}, torch.device(device))


def aggregate_fields(out: Path, rows: pd.DataFrame) -> pd.DataFrame:
    fit_fields = {}
    for key, group in rows.groupby(["subject", "fit_id", "scope", "arm", "template"], sort=False):
        arrays = [np.load(path, allow_pickle=False) for path in group.path]
        payload = {"contacts": arrays[0]["contacts"]}
        for endpoint in ("canonical_full", "seed_removed", "participation"):
            payload[endpoint] = np.nanmedian(np.stack([item[endpoint] for item in arrays]), axis=0)
        payload["seed_removed_denominator"] = np.sum(
            np.stack([item["seed_removed_denominator"] for item in arrays]), axis=0
        )
        fit_fields[key] = payload
    manifest = []
    for subject in sorted(rows.subject.unique()):
        for arm in ARMS:
            candidates = {}
            for template in ("A", "B"):
                matches = [(key, value) for key, value in fit_fields.items()
                           if key[0] == subject and key[3] == arm and key[4] == template]
                if len(matches) != 1:
                    raise RuntimeError(f"gain field A/B assembly failed: {subject} {arm} {template}")
                candidates[template] = matches[0][1]
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
            destination = out / "gain_adjusted_fields/per_patient" / subject / f"{arm}.npz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination, **payload)
            manifest.append({
                "subject": subject, "arm": arm, "path": str(destination),
                "sha256": sha256_file(destination), "n_contacts": len(payload["contacts"]),
                "target_values_read": False,
            })
    frame = pd.DataFrame(manifest)
    frame.to_csv(out / "GAIN_ADJUSTED_FIELD_MANIFEST.csv", index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out, old = args.out_root.resolve(), args.old_root.resolve()
    if not (out / "MODEL_FIELDS_FROZEN.json").exists():
        raise RuntimeError("intact fields must be frozen before gain sensitivity")
    grouped = defaultdict(dict)
    for path in mechanism_metrics_paths(out, old):
        metrics = json.loads(path.read_text())
        if metrics["arm"] in ARMS:
            grouped[(metrics["fit_id"], int(metrics["seed"]))][metrics["arm"]] = path
    if len(grouped) != 126 or any(set(paths) != set(ARMS) for paths in grouped.values()):
        raise RuntimeError("gain sensitivity requires 126 complete L2m/L3 pairs")
    metric_rows, field_rows = [], []
    with ProcessPoolExecutor(max_workers=min(max(1, args.workers), 8)) as executor:
        futures = [executor.submit(worker, (
            str(out), {arm: str(path) for arm, path in paths.items()}, args.device
        )) for paths in grouped.values()]
        for index, future in enumerate(as_completed(futures), start=1):
            metrics, fields = future.result(); metric_rows.extend(metrics); field_rows.extend(fields)
            if index % 10 == 0:
                print(json.dumps({"completed": index, "total": len(grouped)}), flush=True)
    metrics = pd.DataFrame(metric_rows)
    fields = pd.DataFrame(field_rows)
    metrics.to_csv(out / "GAIN_ADJUSTED_PER_FIT_SEED.csv", index=False)
    fields.to_csv(out / "GAIN_ADJUSTED_FIT_FIELD_INDEX.csv", index=False)
    fit = metrics.groupby(["subject", "fit_id", "scope", "arm"], as_index=False).median(numeric_only=True)
    patient = fit.groupby(["subject", "arm"], as_index=False).mean(numeric_only=True)
    patient.to_csv(out / "GAIN_ADJUSTED_PER_PATIENT.csv", index=False)
    manifest = aggregate_fields(out, fields)
    (out / "GAIN_ADJUSTED_SENSITIVITY_COMPLETE.json").write_text(json.dumps({
        "status": "PASS_TARGET_FREE", "created_utc": datetime.now(timezone.utc).isoformat(),
        "pairs": len(grouped), "patients": int(patient.subject.nunique()),
        "field_rows": len(manifest), "target_values_read": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
