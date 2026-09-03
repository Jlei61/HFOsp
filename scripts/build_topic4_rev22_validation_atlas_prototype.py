"""rev22-DCI Task 1b: validation-atlas grammar prototype on the historical rev20 artifacts.

Descriptive only. Runs after the Task 1 objective-qualification JSON is immutable. Reads
the held-out patient endpoint (allowed here: this is the selection-blind validation layer,
never a training input), the rev20 worker artifacts and the rev20 validation aggregates,
and writes only under ``<output_root>/validation_atlas_prototype/``. It never modifies the
objective-qualification or geometry-domain freeze files.

Endpoints per candidate (pooled over topology units, spec section 5.3):
  held-out D_support, held-out D_order, held-out D_time_ms (= D_lag view), fixed-budget
  recall (n_cov / r_cov), natural KMeans balanced alignment and OOD (from the rev20
  validation aggregates), composite held-out distance (continuity), yield, C2ST AUC.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_rev22_interictal_objective import (  # noqa: E402
    COMPONENTS, STATUS_OK, PatientBlockViews, block_split_floors, embedding_features,
    pooled_candidate, standardized_excess, transform_patient_embedding,
)
from src.topic4_rev22_validation import (  # noqa: E402
    RECALL_OK, REFERENCE_SUPPORT_BUDGET_OK, calibrate_coverage_radius,
    classifier_two_sample_auc, fixed_budget_recall, freeze_reference_support_budget,
    paired_unit_bootstrap,
)
from src.topic4_shaft_aware import contract_groups, contract_pairs  # noqa: E402

PHASES = ("screen", "confirmation")
EXPECTED_UNITS = {"screen": 4, "confirmation": 12}
CONDITIONAL = ("D_order", "D_lag")
UNCONDITIONAL = ("D_support", "D_cover")
ENDPOINTS = (
    ("heldout_D_support", "Support view (held-out)", "lower"),
    ("heldout_D_order", "Order | support (held-out, bits)", "lower"),
    ("heldout_D_time_ms", "Physical lag | joint (held-out, ms)", "lower"),
    ("recall_fixed_budget", "Fixed-budget recall", "higher"),
    ("kmeans_balanced_alignment", "Two-template concordance", "higher"),
    ("ood_all_returned", "Out-of-distribution fraction", "lower"),
)
SECONDARY = (
    ("heldout_composite", "Composite held-out distance", "lower"),
    ("returned_families_per_unit", "Returned families per network", "higher"),
    ("c2st_auc", "Classifier two-sample AUC", "lower"),
)
FAMILIES = (
    ("g_EE", "learned E→E dose"), ("g_EtoI", "learned E→I dose"),
    ("ellipse_angle_deg", "Long-axis angle"), ("ellipse_aspect_ratio", "Long/short ratio"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _resolve(root: Path, relative: str) -> Path:
    return ROOT / relative if relative.startswith(("config/", "docs/")) else root / relative


def _verify(root: Path, entry: dict, name: str) -> Path:
    path = _resolve(root, entry["path"])
    if "sha256" in entry and _sha256(path) != entry["sha256"]:
        raise RuntimeError(f"frozen input changed: {name} ({path})")
    return path


def _git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def _embedding_from_contract(npz) -> dict:
    return {"center": np.asarray(npz["feature_center"], float), "scale": np.asarray(npz["feature_scale"], float),
            "components": np.asarray(npz["pca_components"], float),
            "directions": np.asarray(npz["sw_directions"], float),
            "reference_z": np.asarray(npz["reference_z"], float)}


def _jackknife_sd(values) -> float | None:
    v = np.asarray([x for x in values if x is not None], float)
    if len(v) < 2:
        return None
    m = len(v)
    return float(np.sqrt((m - 1) / m * np.sum((v - v.mean()) ** 2)))


def _ratio(candidate_values: dict, candidate_noise: dict) -> dict:
    vals = np.asarray([v for v in candidate_values.values() if v is not None], float)
    noise = np.asarray([v for v in candidate_noise.values() if v is not None], float)
    if len(vals) < 2 or len(noise) < 1:
        return {"ratio": None, "n_candidates": int(len(vals))}
    between = float(np.quantile(vals, 0.9) - np.quantile(vals, 0.1))
    within = float(np.median(noise))
    return {"ratio": float(between / within) if within > 0 else float("inf"), "between_range": between,
            "within_sd": within, "n_candidates": int(len(vals))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=ROOT / "config/topic4_rev22_dci_dual_core_interictal_identifiability.json")
    parser.add_argument("--artifact-root", type=Path, default=Path("/home/honglab/leijiaxin/HFOsp"))
    parser.add_argument("--floor-draws", type=int, default=128)
    parser.add_argument("--radius-draws", type=int, default=1000)
    parser.add_argument("--recall-subsamples", type=int, default=200)
    parser.add_argument("--c2st-resamples", type=int, default=20)
    args = parser.parse_args()
    started = time.time()
    root = args.artifact_root.resolve()
    config = json.loads(args.config.read_text())
    cfg_obj = config["objective"]
    out_root = root / config["output_root"] / "validation_atlas_prototype"
    fig_root = out_root / "figures"
    fig_root.mkdir(parents=True, exist_ok=True)

    qual_dir = root / config["output_root"] / "objective_qualification"
    qual_path = qual_dir / "objective_qualification.json"
    if not qual_path.is_file():
        raise RuntimeError("Task 1 objective qualification must be frozen before the prototype runs")
    qualification = json.loads(qual_path.read_text())
    contract_npz = np.load(qual_dir / "patient_training_contract_v1.npz", allow_pickle=True)
    if _sha256(qual_dir / "patient_training_contract_v1.npz") != qualification["patient_training_contract_sha256"]:
        raise RuntimeError("patient training contract hash drifted")
    embedding = _embedding_from_contract(contract_npz)

    inputs = config["inputs"]
    training_path = _verify(root, inputs["patient_training_target"], "patient_training_target")
    contract_path = _verify(root, inputs["contact_contract"], "contact_contract")
    heldout_path = _verify(root, inputs["patient_heldout_npz"], "patient_heldout_npz")
    manifest_path = _verify(root, inputs["rev20_candidate_manifest"], "rev20_candidate_manifest")
    training = np.load(training_path, allow_pickle=True)
    heldout = np.load(heldout_path, allow_pickle=True)
    contract = json.loads(contract_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if list(heldout["contact_names"]) != list(training["contact_names"]):
        raise RuntimeError("held-out and training contact orders differ")
    groups, pairs = contract_groups(contract), contract_pairs(contract)
    lag_cap = float(cfg_obj["lag_cap_ms"])
    n_pair_min = int(cfg_obj["n_pair_min"])
    q = float(cfg_obj["cover_quantile"])
    train_ms = np.asarray(training["patient_train_onsets"], float) * 1000.0
    held_ms = np.asarray(heldout["heldout_onsets"], float) * 1000.0
    if np.nanmax(held_ms) - np.nanmin(held_ms) > 5e5:
        raise RuntimeError("held-out onsets do not look like seconds")
    held_blocks = np.asarray(heldout["heldout_block_ids"])
    held_views = PatientBlockViews(held_ms, held_blocks, groups, pairs, embedding, lag_cap_ms=lag_cap)
    held_reference = held_views.full_reference()
    train_z = transform_patient_embedding(embedding_features(train_ms, groups, lag_cap_ms=lag_cap), embedding)
    held_z = held_views.z
    print(f"[{time.time()-started:5.0f}s] held-out {len(held_ms)} events / {len(held_views.blocks)} blocks", flush=True)

    # ---- rev20 units: onsets + historical validation fields ----
    rev20_root = _resolve(root, inputs["rev20_output_root"]["path"])
    meta = {c["candidate_id"]: c for c in manifest["candidates"]}
    candidates = {}
    for phase in PHASES:
        agg = json.loads((rev20_root / phase / "aggregate.json").read_text())
        hist = {(r["candidate_id"], int(r["seed"])): r for r in agg["per_network"]}
        for json_path in sorted((rev20_root / phase / "workers").glob("*.json")):
            payload = json.loads(json_path.read_text())
            cid, seed = payload["candidate_id"], int(payload["seed"])
            entry = candidates.setdefault((phase, cid), {
                "phase": phase, "candidate_id": cid, "family": meta[cid]["family"],
                "level": meta[cid]["level"], "is_reference": bool(meta[cid].get("is_reference")), "units": []})
            row = hist.get((cid, seed), {})
            validation = row.get("validation", {})
            runaway = payload["simulation"].get("runaway_early_stop_ms") is not None
            npz_path = json_path.with_suffix(".npz")
            if runaway or not npz_path.is_file():
                entry["units"].append({
                    "seed": seed, "valid": False,
                    "invalid_reason": "RUNAWAY" if runaway else "MISSING_NPZ",
                    "runaway_early_stop_ms": payload["simulation"].get("runaway_early_stop_ms"),
                    "onsets_ms": np.zeros((0, len(training["contact_names"])), float),
                    "kmeans_balanced_alignment": None, "ood_all_returned": None,
                    "heldout_composite": None, "unreadable_fraction": None,
                    "returned_families": 0,
                })
                continue
            arrays = np.load(npz_path)
            returned = np.asarray(arrays["event_returned"], bool)
            entry["units"].append({
                "seed": seed, "valid": True, "invalid_reason": None,
                "runaway_early_stop_ms": None,
                "onsets_ms": np.asarray(arrays["onsets"], float)[returned],
                "kmeans_balanced_alignment": validation.get("direction_balanced_alignment"),
                "ood_all_returned": validation.get("ood_all_returned"),
                "heldout_composite": validation.get("complete_distribution_distance_reference"),
                "unreadable_fraction": validation.get("unreadable_fraction"),
                "returned_families": int(validation.get("n_returned_families", returned.sum())),
            })
    print(f"[{time.time()-started:5.0f}s] {len(candidates)} candidate groups", flush=True)

    # ---- recall contract from the screen reference (per-unit n) ----
    ref_screen = candidates[("screen", "dualcore_s39_reference")]
    ref_counts = [u["returned_families"] for u in ref_screen["units"] if u["valid"]]
    budget = freeze_reference_support_budget(
        ref_counts, expected_units=EXPECTED_UNITS["screen"],
        min_events=int(cfg_obj["fit_estimability"]["min_returned_families"]),
    )
    if budget["status"] != REFERENCE_SUPPORT_BUDGET_OK:
        raise RuntimeError(f"reference support budget is not estimable: {budget}")
    n_cov = int(budget["n_cov"])
    radius = calibrate_coverage_radius(train_z, n_cov, draws=int(args.radius_draws), seed=20260906)
    r_cov = radius["r_cov"]
    print(f"[{time.time()-started:5.0f}s] n_cov={n_cov} r_cov={r_cov:.3f}", flush=True)

    # ---- pooled held-out views, recall, C2ST ----
    for (phase, cid), entry in candidates.items():
        units = entry["units"]
        valid_units = [u for u in units if u["valid"]]
        entry["formal_estimable"] = (len(units) == EXPECTED_UNITS[phase]
                                      and len(valid_units) == EXPECTED_UNITS[phase])
        entry["invalid_units"] = len(units) - len(valid_units)
        entry["runaway_fraction"] = float(np.mean([not u["valid"] for u in units])) if units else 1.0
        if not valid_units:
            entry.update(pooled=None, jackknife_sd={k: None for k in COMPONENTS},
                         n_pooled_events=0, recruitment_profile=None,
                         c2st={"status": "PRIMARY_ENDPOINT_NOT_ESTIMABLE", "auc": None})
            continue
        pooled = pooled_candidate([u["onsets_ms"] for u in valid_units], held_reference, groups, pairs, embedding,
                                  n_pair_min=n_pair_min, lag_cap_ms=lag_cap, cover_quantile=q)
        entry["pooled"] = pooled["pooled"]
        entry["jackknife_sd"] = pooled["jackknife_sd"]
        entry["n_pooled_events"] = pooled["n_pooled_events"]
        entry["recruitment_profile"] = pooled["recruitment_profile"]
        model_z_units = []
        for i, u in enumerate(valid_units):
            u["per_unit_views"] = {k: pooled["per_unit"][i][k]["value"] for k in COMPONENTS}
            z = (transform_patient_embedding(embedding_features(u["onsets_ms"], groups, lag_cap_ms=lag_cap), embedding)
                 if len(u["onsets_ms"]) else np.zeros((0, held_z.shape[1])))
            model_z_units.append(z)
            u["recall"] = fixed_budget_recall(z, held_z, n_cov=n_cov, r_cov=r_cov,
                                              subsamples=int(args.recall_subsamples), seed=u["seed"])
        model_z = np.vstack(model_z_units)
        model_groups = np.concatenate([[u["seed"]] * len(z) for u, z in zip(valid_units, model_z_units)])
        entry["c2st"] = (classifier_two_sample_auc(
            held_z, held_blocks, model_z, model_groups, seed=20260907,
            resamples=int(args.c2st_resamples), permutations=20)
            if entry["formal_estimable"] else
            {"status": "PRIMARY_ENDPOINT_NOT_ESTIMABLE", "auc": None})
        for u in units:
            del u["onsets_ms"]
    print(f"[{time.time()-started:5.0f}s] pooled views / recall / C2ST", flush=True)

    # ---- held-out floors (block split on held-out blocks) ----
    requests, seen = [], set()
    for (phase, cid), entry in candidates.items():
        if not entry["formal_estimable"]:
            continue
        n = entry["n_pooled_events"]
        if n < 2:
            continue
        if f"n{n}" not in seen:
            seen.add(f"n{n}")
            requests.append({"key": f"n{n}", "n": n, "components": UNCONDITIONAL, "thin_profile": None})
        requests.append({"key": f"{phase}::{cid}", "n": n, "components": CONDITIONAL,
                         "thin_profile": entry["recruitment_profile"]})
    floors = block_split_floors(held_views, requests, draws=int(args.floor_draws), seed=20260908,
                                n_pair_min=n_pair_min, cover_quantile=q)
    print(f"[{time.time()-started:5.0f}s] held-out floors", flush=True)

    # ---- assemble endpoint table ----
    rows = []
    for (phase, cid), entry in candidates.items():
        n = entry["n_pooled_events"]
        units = entry["units"]
        floor = {k: floors.get(f"{phase}::{cid}" if k in CONDITIONAL else f"n{n}", {}).get(k) for k in COMPONENTS}

        def unit_mean(key):
            v = [u[key] for u in units if u.get(key) is not None]
            return (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None) if v else (None, None)

        recalls = [u["recall"]["recall"] for u in units
                   if u["valid"] and u.get("recall", {}).get("status") == RECALL_OK]
        recall_estimable = entry["formal_estimable"] and len(recalls) == EXPECTED_UNITS[phase]
        pooled_values = entry["pooled"] if entry["formal_estimable"] else None
        row = {
            "phase": phase, "candidate_id": cid, "family": entry["family"], "level": entry["level"],
            "is_reference": entry["is_reference"], "n_units": len(units), "n_pooled_events": n,
            "formal_estimable": entry["formal_estimable"], "invalid_units": entry["invalid_units"],
            "runaway_fraction": entry["runaway_fraction"],
            "heldout_D_support": None if pooled_values is None else pooled_values["D_support"]["value"],
            "heldout_D_order": None if pooled_values is None else pooled_values["D_order"]["value"],
            "heldout_D_time_ms": None if pooled_values is None else pooled_values["D_lag"]["value"],
            "heldout_D_cover": None if pooled_values is None else pooled_values["D_cover"]["value"],
            "status_order": ("PRIMARY_ENDPOINT_NOT_ESTIMABLE" if pooled_values is None
                             else pooled_values["D_order"]["status"]),
            "status_lag": ("PRIMARY_ENDPOINT_NOT_ESTIMABLE" if pooled_values is None
                           else pooled_values["D_lag"]["status"]),
            "jackknife": {"heldout_D_support": entry["jackknife_sd"]["D_support"],
                          "heldout_D_order": entry["jackknife_sd"]["D_order"],
                          "heldout_D_time_ms": entry["jackknife_sd"]["D_lag"]},
            "standardized": {"heldout_D_support": standardized_excess(None if pooled_values is None else pooled_values["D_support"]["value"], floor["D_support"]),
                             "heldout_D_order": standardized_excess(None if pooled_values is None else pooled_values["D_order"]["value"], floor["D_order"]),
                             "heldout_D_time_ms": standardized_excess(None if pooled_values is None else pooled_values["D_lag"]["value"], floor["D_lag"])},
            "floor": floor,
            "recall_fixed_budget": float(np.mean(recalls)) if recall_estimable else None,
            "recall_se": (float(np.std(recalls, ddof=1) / np.sqrt(len(recalls)))
                          if recall_estimable and len(recalls) > 1 else None),
            "recall_units_estimable": len(recalls),
            "c2st_auc": entry["c2st"].get("auc"), "c2st_status": entry["c2st"]["status"],
            "per_unit": units,
        }
        for key in ("kmeans_balanced_alignment", "ood_all_returned", "heldout_composite", "unreadable_fraction"):
            if entry["formal_estimable"]:
                row[key], row[f"{key}_se"] = unit_mean(key)
            else:
                row[key], row[f"{key}_se"] = None, None
        row["returned_families_per_unit"] = float(np.mean([u["returned_families"] for u in units]))
        rows.append(row)

    # ---- display identifiability per endpoint (screen phase, pooled/mean values) ----
    display = {}
    for key, _, _ in ENDPOINTS + SECONDARY:
        vals, noise = {}, {}
        for row in rows:
            if row["phase"] != "screen" or row.get(key) is None:
                continue
            if key in row["standardized"]:
                vals[row["candidate_id"]] = row["standardized"][key]
                f = row["floor"][{"heldout_D_support": "D_support", "heldout_D_order": "D_order",
                                  "heldout_D_time_ms": "D_lag"}[key]]
                jk = row["jackknife"][key]
                noise[row["candidate_id"]] = (None if (jk is None or f is None or f.get("q95") is None)
                                              else jk / (f["q95"] - f["q50"] + 1e-9))
            else:
                vals[row["candidate_id"]] = row[key]
                noise[row["candidate_id"]] = row.get(f"{key}_se") if key != "recall_fixed_budget" else row.get("recall_se")
        display[key] = _ratio(vals, noise)
        display[key]["main_atlas"] = bool(display[key]["ratio"] is not None and display[key]["ratio"] >= 1.0)

    # ---- paired differences vs reference on confirmation (topology-unit bootstrap) ----
    ref = next(r for r in rows if r["phase"] == "confirmation" and r["is_reference"])
    ref_units = {u["seed"]: u for u in ref["per_unit"]}
    matrix = []
    for row in rows:
        if row["phase"] != "confirmation" or row["is_reference"]:
            continue
        cand_units = {u["seed"]: u for u in row["per_unit"]}
        seeds = sorted(set(ref_units) & set(cand_units))
        cells = {}
        for key, label, better in ENDPOINTS + SECONDARY:
            if not row["formal_estimable"] or not ref["formal_estimable"]:
                cells[key] = {"status": "PRIMARY_ENDPOINT_NOT_ESTIMABLE", "delta": None,
                              "lo": None, "hi": None, "n": len(seeds), "positive_is_better": True}
                continue
            if key in ("heldout_D_support", "heldout_D_order", "heldout_D_time_ms"):
                view = {"heldout_D_support": "D_support", "heldout_D_order": "D_order", "heldout_D_time_ms": "D_lag"}[key]
                a = [cand_units[s].get("per_unit_views", {}).get(view) for s in seeds]
                b = [ref_units[s].get("per_unit_views", {}).get(view) for s in seeds]
            elif key == "recall_fixed_budget":
                a = [cand_units[s].get("recall", {}).get("recall") for s in seeds]
                b = [ref_units[s].get("recall", {}).get("recall") for s in seeds]
            elif key == "returned_families_per_unit":
                a = [cand_units[s]["returned_families"] for s in seeds]
                b = [ref_units[s]["returned_families"] for s in seeds]
            elif key == "c2st_auc":
                cells[key] = {"status": "POOLED_SECONDARY_NO_PAIRED_INTERVAL",
                              "delta": None if row["c2st_auc"] is None or ref["c2st_auc"] is None
                              else float(ref["c2st_auc"] - row["c2st_auc"]),
                              "lo": None, "hi": None, "n": 1, "positive_is_better": True}
                continue
            else:
                a = [cand_units[s][key] for s in seeds]
                b = [ref_units[s][key] for s in seeds]
            if any(x is None for x in a + b):
                cells[key] = {"status": "PRIMARY_ENDPOINT_NOT_ESTIMABLE", "delta": None,
                              "lo": None, "hi": None, "n": len(seeds), "positive_is_better": True}
                continue
            cells[key] = paired_unit_bootstrap(
                a, b, draws=4096, seed=20260909 + len(matrix) * 100 + len(cells),
                higher_is_better=(better == "higher"),
            )
        matrix.append({"candidate_id": row["candidate_id"], "family": row["family"], "level": row["level"],
                       "cells": cells})

    # ---- figures ----
    _plot_layer1(rows, display, fig_root)
    _plot_layer2(matrix, fig_root)
    readme = fig_root / "README.md"
    readme.write_text(_readme_text(n_cov, r_cov, display))

    payload = {
        "schema_id": "topic4_rev22_dci_validation_atlas_prototype_v1",
        "status": "PROTOTYPE_DESCRIPTIVE_ONLY",
        "git_commit": _git_commit(),
        "objective_qualification_sha256": _sha256(qual_path),
        "objective_status": qualification["status"],
        "recall_contract_prototype": {"n_cov_rev20": n_cov, "reference_counts": budget["counts"],
                                      **radius, "note": "prototype values; formal freeze in Task 8"},
        "display_identifiability": display,
        "endpoints": [k for k, _, _ in ENDPOINTS], "secondary": [k for k, _, _ in SECONDARY],
        "rows": [{k: v for k, v in r.items() if k != "per_unit"} | {"per_unit": r["per_unit"]} for r in rows],
        "family_matrix_confirmation_vs_reference": matrix,
        "held_out_floors": floors,
        "claim_boundary": "layout and assay-sensitivity prototype on historical rev20 artifacts; not candidate selection",
        "elapsed_seconds": float(time.time() - started),
    }
    (out_root / "validation_atlas_prototype.json").write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n")
    with (out_root / "validation_atlas_prototype.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        keys = [k for k, _, _ in ENDPOINTS + SECONDARY]
        writer.writerow(["phase", "candidate_id", "family", "level", "n_units", "n_pooled_events", *keys,
                         "status_order", "status_lag", "c2st_status"])
        for r in rows:
            writer.writerow([r["phase"], r["candidate_id"], r["family"], r["level"], r["n_units"], r["n_pooled_events"],
                             *[r.get(k) for k in keys], r["status_order"], r["status_lag"], r["c2st_status"]])
    print(json.dumps({"status": payload["status"], "n_cov": n_cov, "r_cov": r_cov,
                      "main_atlas_rows": [k for k, v in display.items() if v.get("main_atlas")],
                      "output": str(out_root), "elapsed_s": round(time.time() - started)}, indent=2))


def _plot_layer1(rows, display, fig_root: Path) -> None:
    main_rows = [e for e in ENDPOINTS if display[e[0]].get("main_atlas")]
    side_rows = [e for e in ENDPOINTS if not display[e[0]].get("main_atlas")]
    for name, endpoints in (("validation_atlas_prototype", main_rows), ("validation_atlas_prototype_sidecar", side_rows)):
        if not endpoints:
            continue
        fig, axes = plt.subplots(len(endpoints), len(FAMILIES), figsize=(2.1 * len(FAMILIES), 1.9 * len(endpoints)),
                                 squeeze=False, sharey="row")
        for j, (family, flabel) in enumerate(FAMILIES):
            fam_rows = [r for r in rows if r["family"] == family or r["is_reference"]]
            for i, (key, label, _) in enumerate(endpoints):
                ax = axes[i, j]
                for phase, marker, fill in (("screen", "o", "none"), ("confirmation", "D", "full")):
                    pts = [r for r in fam_rows if r["phase"] == phase and r.get(key) is not None]
                    if not pts:
                        continue
                    x = [float(r["level"]) if not r["is_reference"] else _reference_level(family) for r in pts]
                    y = [r[key] for r in pts]
                    err = [_noise(r, key) for r in pts]
                    order = np.argsort(x)
                    x, y = np.asarray(x)[order], np.asarray(y)[order]
                    err = np.asarray([0.0 if e is None else e for e in err])[order]
                    ax.errorbar(x, y, yerr=1.645 * err, fmt=marker, ms=4.5, color="#222222" if phase == "confirmation" else "#7A7A7A",
                                markerfacecolor="#222222" if fill == "full" else "white", ecolor="#9A9A9A",
                                elinewidth=0.8, capsize=0, lw=0.8, ls="-" if phase == "screen" else "none")
                floor = _floor_band(fam_rows, key)
                if floor is not None:
                    ax.axhspan(floor[0], floor[1], color="#DDDDDD", alpha=0.6, lw=0, zorder=0)
                ax.axvline(_reference_level(family), color="#BBBBBB", lw=0.6, ls=":")
                if i == 0:
                    ax.set_title(flabel, fontsize=8)
                if j == 0:
                    ax.set_ylabel(label, fontsize=7)
                ax.tick_params(labelsize=6)
                for spine in ("top", "right"):
                    ax.spines[spine].set_visible(False)
        fig.tight_layout(w_pad=0.4, h_pad=0.4)
        for ext in ("png", "pdf"):
            fig.savefig(fig_root / f"{name}.{ext}", dpi=200)
        plt.close(fig)


def _plot_layer2(matrix, fig_root: Path) -> None:
    keys = [k for k, _, _ in ENDPOINTS] + ["heldout_composite", "returned_families_per_unit"]
    labels = [l for _, l, _ in ENDPOINTS] + ["Composite distance", "Returned families"]
    fig, axes = plt.subplots(1, len(keys), figsize=(1.6 * len(keys), 0.45 * len(matrix) + 1.2), sharey=True)
    names = [f"{m['family']} = {m['level']}" for m in matrix]
    ypos = np.arange(len(matrix))
    for ax, key, label in zip(axes, keys, labels):
        for y, m in zip(ypos, matrix):
            c = m["cells"].get(key, {})
            if c.get("delta") is None:
                continue
            color = "#2A7F62" if (c.get("lo") is not None and c["lo"] > 0) else ("#B0413E" if (c.get("hi") is not None and c["hi"] < 0) else "#666666")
            if c.get("lo") is not None:
                ax.plot([c["lo"], c["hi"]], [y, y], color=color, lw=1.2)
            ax.plot(c["delta"], y, "o", color=color, ms=4)
        ax.axvline(0, color="#999999", lw=0.7)
        ax.set_title(label, fontsize=7)
        ax.tick_params(labelsize=6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(names, fontsize=6)
    axes[0].invert_yaxis()
    fig.suptitle("rev20 confirmed candidates vs reference (paired over 12 networks; positive = better)", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(fig_root / f"validation_family_matrix_prototype.{ext}", dpi=200)
    plt.close(fig)


def _reference_level(family: str) -> float:
    return {"node_gain": 1.0, "core_budget_scale": 1.0, "signed_depth_shrinkage": 1.0, "g_EE": 0.5,
            "g_EtoI": 1.0, "both_scale": 1.0, "ellipse_angle_deg": 45.0, "ellipse_aspect_ratio": 2.0}[family]


def _noise(row, key):
    if key in row.get("jackknife", {}):
        return row["jackknife"][key]
    if key == "recall_fixed_budget":
        return row.get("recall_se")
    return row.get(f"{key}_se")


def _floor_band(rows, key):
    view = {"heldout_D_support": "D_support", "heldout_D_order": "D_order", "heldout_D_time_ms": "D_lag"}.get(key)
    if view is None:
        return None
    ref = [r for r in rows if r["is_reference"] and r["phase"] == "screen"]
    if not ref or ref[0]["floor"].get(view) is None:
        return None
    f = ref[0]["floor"][view]
    return (f["q05"], f["q95"]) if f.get("q05") is not None else None


def _readme_text(n_cov, r_cov, display) -> str:
    main_rows = [k for k, v in display.items() if v.get("main_atlas")]
    side_rows = [k for k, v in display.items() if k in [e[0] for e in ENDPOINTS] and not v.get("main_atlas")]
    return f"""### validation_atlas_prototype.png

rev22 验证图谱第一层的版式原型，数据来自 rev20 的 232 条历史轨迹（每个候选按拓扑单元合并打分）。
列是 rev22 将继续研究的四个连接参数，行是六个选择盲验证端点中在 rev20 数据上"候选间差异
超过种子噪声"的那些（本次进入主图：{', '.join(main_rows)}）。空心圆是 4 网络 screen，实心菱形
是 12 网络确认，误差线是留一拓扑 jackknife 或单元均值的 90% 区间，灰带是参考点事件数下患者
自身对比的地板。固定预算 recall 使用 n_cov={n_cov}、r_cov={r_cov:.3f}（原型值，正式冻结在 Task 8）。

**关注点**：先看哪一列的曲线在多行同时离开灰带；单行改善、其他行变差的族属于 trade-off。

### validation_atlas_prototype_sidecar.png

未通过展示可辨识规则（候选间 q90−q10 除以噪声中位数 < 1）的端点行：{', '.join(side_rows) or '无'}。
这些行完整保留，只是不进主图。

**关注点**：这些端点在 rev20 参数范围内分辨不开候选，不代表端点无意义。

### validation_family_matrix_prototype.png

第二层的版式原型：rev20 的确认候选相对参考点的配对差（按 12 个网络种子做配对 bootstrap，正值 = 更好），
每列一个端点，含次要的复合距离和事件产量列。绿色区间整体大于 0，红色整体小于 0，灰色跨 0。

**关注点**：一个候选要在多列同为绿色才算"有用"；只在一列绿、另一列红是 trade-off。
"""


if __name__ == "__main__":
    main()
