#!/usr/bin/env python3
"""Single-process target unseal and R3 early-ictal benchmark for v0.4 fields."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_gradient_grid_field import (  # noqa: E402
    build_event_scorer,
    score_event_detail_single,
    score_event_maxab_batch,
)
from src.topic5_template_axis_field import scorers_from_interictal_record  # noqa: E402


CORE_MODELS = (
    "M0_NO_REC", "M1_DENSE", "M2_UNIFORM_SET", "M3_FIXED_LOCAL",
    "M4_SPATIAL_GROWTH", "M6_SPATIAL_MID", "M8_UNIFORM_COST_MID",
)
FACTORIAL = ("M2_UNIFORM_SET", "M4_SPATIAL_GROWTH", "M6_SPATIAL_MID", "M8_UNIFORM_COST_MID")
ENDPOINTS = ("canonical_full", "seed_removed")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def atomic_json(path: Path, payload: Any) -> None:
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        if isinstance(value, (np.integer, np.bool_)):
            return value.item()
        if isinstance(value, np.floating):
            value = float(value)
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(clean(payload), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty table: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def aligned_field(npz: Any, template: str, endpoint: str,
                  order: list[str]) -> np.ndarray:
    names = np.asarray(npz[f"{template}_contacts"]).astype(str).tolist()
    values = np.asarray(npz[f"{template}_{endpoint}"], float)
    lookup = {name: float(value) for name, value in zip(names, values)}
    return np.asarray([lookup.get(name, np.nan) for name in order], float)


def build_scorer(record: dict[str, Any], a: np.ndarray, b: np.ndarray,
                 finite: np.ndarray) -> dict[str, Any]:
    models = scorers_from_interictal_record(record)
    ma, mb = models["own_a"], models["own_b"]
    return build_event_scorer(
        pts_a=ma["points"], support_a=ma["support"], earliness_a=a,
        pts_b=mb["points"], support_b=mb["support"], earliness_b=b,
        sigma_a=ma["sigma"], sigma_b=mb["sigma"], finite=finite,
        model_a="A", model_b="B",
    )


def permutation_indices(n: int, eligible: np.ndarray, shafts: list[str],
                        draws: int, seed: int, within_shaft: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.arange(n, dtype=np.int16)
    out = np.tile(base, (draws, 1))
    if within_shaft:
        groups = [np.asarray([index for index in eligible if shafts[index] == shaft], int)
                  for shaft in sorted({shafts[index] for index in eligible})]
    else:
        groups = [eligible]
    for draw in range(draws):
        for group in groups:
            if len(group) > 1:
                out[draw, group] = rng.permutation(group)
    return out


def score_one(scorer: dict[str, Any], target: np.ndarray, permutations: np.ndarray) -> dict[str, Any]:
    detail = score_event_detail_single(scorer, target)
    null = score_event_maxab_batch(scorer, target[permutations])
    observed = float(detail["maxab"])
    finite_null = null[np.isfinite(null)]
    if not np.isfinite(observed) or finite_null.size == 0:
        return {"observed": np.nan, "null": null, "null_median": np.nan,
                "margin": np.nan, "empirical_p": np.nan, **detail}
    null_median = float(np.median(finite_null))
    return {
        "observed": observed, "null": null, "null_median": null_median,
        "margin": observed - null_median,
        "empirical_p": float((1 + np.sum(finite_null >= observed - 1e-15)) / (1 + len(finite_null))),
        **detail,
    }


def paired_summary(values: Iterable[float], draws: int = 10000, seed: int = 1) -> dict[str, Any]:
    values = np.asarray(list(values), float)
    values = values[np.isfinite(values)]
    tol = 1e-9
    nonzero = values[np.abs(values) > tol]
    p = float(wilcoxon(nonzero, method="auto").pvalue) if nonzero.size else 1.0
    rng = np.random.default_rng(seed)
    bootstrap = np.median(rng.choice(values, (draws, len(values)), replace=True), axis=1)
    signs = rng.choice((-1.0, 1.0), (draws, len(values)))
    permuted = np.median(signs * values[None, :], axis=1)
    observed = float(np.median(values))
    return {
        "n": int(len(values)), "median": observed,
        "bootstrap_95ci": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "positive": int(np.sum(values > tol)), "negative": int(np.sum(values < -tol)),
        "tied": int(np.sum(np.abs(values) <= tol)), "wilcoxon_p": p,
        "sign_permutation_p": float((1 + np.sum(np.abs(permuted) >= abs(observed) - 1e-15)) / (draws + 1)),
    }


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    items = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    n = len(items)
    for rank, (name, value) in enumerate(items):
        running = max(running, min(1.0, (n - rank) * float(value)))
        adjusted[name] = running
    return adjusted


def aggregate_patients(seizure_rows: list[dict[str, Any]], null_store: dict[str, np.ndarray],
                       supportive: str) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    keys = sorted({(row["subject"], row["model"], row["cell"], row["endpoint"])
                   for row in seizure_rows})
    patients = []
    patient_nulls = {}
    for subject, model, cell, endpoint in keys:
        selected = [row for row in seizure_rows if (row["subject"], row["model"], row["cell"], row["endpoint"])
                    == (subject, model, cell, endpoint)]
        all_null = np.stack([null_store[row["null_key_all"]] for row in selected])
        shaft_null = np.stack([null_store[row["null_key_shaft"]] for row in selected])
        pnull_all = np.nanmedian(all_null, axis=0)
        pnull_shaft = np.nanmedian(shaft_null, axis=0)
        observed = float(np.nanmedian([row["observed"] for row in selected]))
        row = {
            "subject": subject, "primary": subject != supportive, "supportive": subject == supportive,
            "model": model, "cell": cell, "endpoint": endpoint, "n_seizures": len(selected),
            "n_contacts_min": min(int(item["n_contacts"]) for item in selected),
            "observed": observed,
            "all_contact_null_median": float(np.nanmedian(pnull_all)),
            "all_contact_margin": observed - float(np.nanmedian(pnull_all)),
            "all_contact_p": float((1 + np.sum(pnull_all >= observed - 1e-15)) / (1 + np.isfinite(pnull_all).sum())),
            "within_shaft_null_median": float(np.nanmedian(pnull_shaft)),
            "within_shaft_margin": observed - float(np.nanmedian(pnull_shaft)),
            "within_shaft_p": float((1 + np.sum(pnull_shaft >= observed - 1e-15)) / (1 + np.isfinite(pnull_shaft).sum())),
        }
        patients.append(row)
        key = f"{subject}|{model}|{cell}|{endpoint}"
        patient_nulls[key] = pnull_all
    return patients, patient_nulls


def conditional_effects(patient_rows: list[dict[str, Any]], fidelity_rows: list[dict[str, str]],
                        endpoint: str = "canonical_full") -> dict[str, Any]:
    fidelity = {(row["subject"], row["model"], row["cell"]): float(row["matched_empirical_r"])
                for row in fidelity_rows}
    selected = [row for row in patient_rows if row["primary"] and row["endpoint"] == endpoint
                and row["cell"] == "rnn" and row["model"] in CORE_MODELS]
    by_patient: dict[str, list[dict[str, Any]]] = {}
    for row in selected:
        row = dict(row)
        row["fidelity"] = fidelity[(row["subject"], row["model"], row["cell"])]
        by_patient.setdefault(row["subject"], []).append(row)
    x, y = [], []
    for rows in by_patient.values():
        x.extend([row["fidelity"] - np.mean([item["fidelity"] for item in rows]) for row in rows])
        y.extend([row["all_contact_margin"] - np.mean([item["all_contact_margin"] for item in rows]) for row in rows])
    beta = float(np.dot(x, y) / max(np.dot(x, x), 1e-12))
    contrasts = {}
    pairs = (("M6_SPATIAL_MID", "M2_UNIFORM_SET"),
             ("M6_SPATIAL_MID", "M1_DENSE"),
             ("M6_SPATIAL_MID", "M0_NO_REC"))
    for model, baseline in pairs:
        values = []
        for rows in by_patient.values():
            lookup = {row["model"]: row for row in rows}
            if model in lookup and baseline in lookup:
                values.append((lookup[model]["all_contact_margin"] - lookup[baseline]["all_contact_margin"])
                              - beta * (lookup[model]["fidelity"] - lookup[baseline]["fidelity"]))
        contrasts[f"{model}_vs_{baseline}"] = paired_summary(values, seed=stable_seed(model, baseline, "conditional"))
    return {"endpoint": endpoint, "patient_intercept_removed": True,
            "within_patient_fidelity_slope": beta, "contrasts": contrasts}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--target-cache-root", type=Path, required=True)
    parser.add_argument("--n-perm", type=int, default=5000)
    args = parser.parse_args()
    out_root = args.out_root.resolve()
    target_root = args.target_cache_root.resolve()
    manifest_path = out_root / "MODEL_FIELD_MANIFEST.json"
    authorization_path = out_root / "TARGET_UNSEAL_AUTHORIZATION.json"
    metadata = json.loads((out_root / "EARLY_ICTAL_METADATA_INVENTORY.json").read_text())
    manifest = json.loads(manifest_path.read_text())
    authorization = json.loads(authorization_path.read_text())
    if not authorization.get("authorized"):
        raise RuntimeError("target unseal is not authorized")
    if authorization["model_field_manifest_sha256"] != sha256(manifest_path):
        raise RuntimeError("model field manifest changed after authorization")
    if manifest.get("target_values_read") is not False or metadata.get("target_values_read") is not False:
        raise RuntimeError("target seal is already broken")

    unlock = {
        "contract": "topic5_rnn_motif_early_ictal_unlock_v0_4",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_values_read_before_this_record": False,
        "target_values_unlocked_after_field_freeze": True,
        "model_field_manifest_sha256": sha256(manifest_path),
        "authorization_sha256": sha256(authorization_path),
        "metadata_inventory_sha256": sha256(out_root / "EARLY_ICTAL_METADATA_INVENTORY.json"),
        "scorer_sha256": sha256(Path(__file__).resolve()),
        "target_key": "target_1_150", "n_permutations": int(args.n_perm),
    }
    unlock_path = out_root / "early_ictal" / "TARGET_UNLOCK_RECORD.json"
    unlock_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(unlock_path, unlock)

    fidelity_rows = read_csv(out_root / "model_field_patient_metrics.csv")
    model_keys = sorted({(row["model"], row["cell"]) for row in fidelity_rows})
    primary = list(metadata["actual_primary_join"])
    supportive = str(metadata["supportive_subject"])
    subjects = primary + ([supportive] if metadata["supportive_available"] else [])
    seizure_rows: list[dict[str, Any]] = []
    null_store: dict[str, np.ndarray] = {}
    null_labels = []
    null_all, null_shaft = [], []

    for subject in subjects:
        record_path = Path(manifest["patient_geometry"][subject]["empirical_record"])
        record = json.loads(record_path.read_text())
        field = record["interictal_field"]
        order = [str(value) for value in field["contact_order"]]
        shafts = [str(value) for value in field["shafts"]]
        loaded: dict[tuple[str, str], dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        for model, cell in model_keys:
            path = out_root / "model_fields" / "per_patient" / subject / f"{model}__{cell}.npz"
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as data:
                loaded[(model, cell)] = {
                    endpoint: (aligned_field(data, "A", endpoint, order),
                               aligned_field(data, "B", endpoint, order))
                    for endpoint in ENDPOINTS
                }
        if not loaded:
            raise RuntimeError(f"no model fields for {subject}")

        target_files = sorted((target_root / f"outer_{subject}").glob(f"{subject}__*.npz"))
        for target_path in target_files:
            # This is the only target-value access point in the v0.4 pipeline.
            with np.load(target_path, allow_pickle=False) as data:
                target_names = np.asarray(data["contact_names"]).astype(str).tolist()
                target_values = np.asarray(data["target_1_150"], float)
            target_lookup = {name: float(value) for name, value in zip(target_names, target_values)}
            target = np.asarray([target_lookup.get(name, np.nan) for name in order], float)
            seizure_id = target_path.stem.split("__", 1)[-1]
            for endpoint in ENDPOINTS:
                common_model_finite = np.ones(len(order), bool)
                for values in loaded.values():
                    common_model_finite &= np.isfinite(values[endpoint][0]) & np.isfinite(values[endpoint][1])
                finite = common_model_finite & np.isfinite(target)
                eligible = np.flatnonzero(finite)
                if len(eligible) < 6:
                    if endpoint == "canonical_full":
                        raise RuntimeError(f"{subject} {seizure_id}: canonical support < 6")
                    continue
                perm_all = permutation_indices(
                    len(order), eligible, shafts, args.n_perm,
                    stable_seed(subject, seizure_id, endpoint, "all_contact"), False,
                )
                perm_shaft = permutation_indices(
                    len(order), eligible, shafts, args.n_perm,
                    stable_seed(subject, seizure_id, endpoint, "within_shaft"), True,
                )
                # Frozen empirical reference uses the identical support and permutations.
                candidates = {
                    ("EMPIRICAL_REFERENCE", "reference"): (
                        np.asarray(field["earliness_a"], float),
                        np.asarray(field["earliness_b"], float),
                    ),
                    **{key: value[endpoint] for key, value in loaded.items()},
                }
                for (model, cell), (a, b) in candidates.items():
                    scorer = build_scorer(record, a, b, finite)
                    all_score = score_one(scorer, target, perm_all)
                    shaft_score = score_one(scorer, target, perm_shaft)
                    key = f"{subject}|{seizure_id}|{model}|{cell}|{endpoint}"
                    key_all, key_shaft = key + "|all", key + "|shaft"
                    null_store[key_all] = all_score["null"]
                    null_store[key_shaft] = shaft_score["null"]
                    null_labels.append(key)
                    null_all.append(all_score["null"].astype(np.float32))
                    null_shaft.append(shaft_score["null"].astype(np.float32))
                    seizure_rows.append({
                        "subject": subject, "primary": subject in primary, "supportive": subject == supportive,
                        "seizure_id": seizure_id, "model": model, "cell": cell, "endpoint": endpoint,
                        "n_contacts": int(len(eligible)), "observed": all_score["observed"],
                        "abs_a": all_score["abs_a"], "abs_b": all_score["abs_b"],
                        "best_template": all_score["best_template"],
                        "all_contact_null_median": all_score["null_median"],
                        "all_contact_margin": all_score["margin"], "all_contact_p": all_score["empirical_p"],
                        "within_shaft_null_median": shaft_score["null_median"],
                        "within_shaft_margin": shaft_score["margin"], "within_shaft_p": shaft_score["empirical_p"],
                        "null_key_all": key_all, "null_key_shaft": key_shaft,
                    })

    write_csv(out_root / "early_ictal_per_seizure.csv", seizure_rows)
    patient_rows, patient_nulls = aggregate_patients(seizure_rows, null_store, supportive)
    write_csv(out_root / "early_ictal_per_patient_model.csv", patient_rows)
    np.savez_compressed(
        out_root / "early_ictal_null_matrices.npz",
        labels=np.asarray(null_labels, dtype="U256"), all_contact=np.stack(null_all),
        within_shaft=np.stack(null_shaft),
        patient_labels=np.asarray(list(patient_nulls), dtype="U256"),
        patient_all_contact=np.stack(list(patient_nulls.values())),
    )

    lookup = {(row["subject"], row["model"], row["cell"], row["endpoint"]): row
              for row in patient_rows if row["primary"]}
    contrasts = {}
    contrast_pairs = []
    for model, cell in model_keys:
        contrast_pairs.append(((model, cell), ("M0_NO_REC", cell)))
    contrast_pairs.extend([
        (("M6_SPATIAL_MID", "rnn"), ("M2_UNIFORM_SET", "rnn")),
        (("M6_SPATIAL_MID", "rnn"), ("M1_DENSE", "rnn")),
        (("M6_SPATIAL_MID", "rnn"), ("M6_SPATIAL_MID", "gru")),
    ])
    pvalues = {}
    for endpoint in ENDPOINTS:
        for (model, cell), (baseline, baseline_cell) in contrast_pairs:
            if model == baseline and cell == baseline_cell:
                continue
            values = []
            for subject in primary:
                a = lookup.get((subject, model, cell, endpoint))
                b = lookup.get((subject, baseline, baseline_cell, endpoint))
                if a and b:
                    values.append(float(a["all_contact_margin"]) - float(b["all_contact_margin"]))
            if not values:
                continue
            name = f"{endpoint}|{model}__{cell}_vs_{baseline}__{baseline_cell}"
            contrasts[name] = paired_summary(values, seed=stable_seed(name))
            if endpoint == "canonical_full" and model in CORE_MODELS and baseline in CORE_MODELS:
                pvalues[name] = contrasts[name]["wilcoxon_p"]
    adjusted = holm(pvalues)
    for name, q in adjusted.items():
        contrasts[name]["holm_q_core_family"] = q
    for endpoint in ENDPOINTS:
        for model, cell in model_keys:
            values = [float(lookup[(subject, model, cell, endpoint)]["all_contact_margin"])
                      for subject in primary if (subject, model, cell, endpoint) in lookup]
            if values:
                contrasts[f"{endpoint}|{model}__{cell}_margin_gt_zero"] = paired_summary(
                    values, seed=stable_seed(endpoint, model, cell, "zero")
                )
    atomic_json(out_root / "early_ictal_model_contrasts.json", contrasts)

    factorial_effects = {}
    for endpoint in ENDPOINTS:
        definitions = {
            "growth_at_zero": ("M4_SPATIAL_GROWTH", "M2_UNIFORM_SET"),
            "growth_at_mid": ("M6_SPATIAL_MID", "M8_UNIFORM_COST_MID"),
            "cost_uniform": ("M8_UNIFORM_COST_MID", "M2_UNIFORM_SET"),
            "cost_spatial": ("M6_SPATIAL_MID", "M4_SPATIAL_GROWTH"),
        }
        effects = {}
        raw = {}
        for name, (a_model, b_model) in definitions.items():
            raw[name] = np.asarray([
                lookup[(subject, a_model, "rnn", endpoint)]["all_contact_margin"]
                - lookup[(subject, b_model, "rnn", endpoint)]["all_contact_margin"]
                for subject in primary
            ], float)
            effects[name] = paired_summary(raw[name], seed=stable_seed(endpoint, name))
        interaction = raw["cost_spatial"] - raw["cost_uniform"]
        effects["interaction"] = paired_summary(interaction, seed=stable_seed(endpoint, "interaction"))
        factorial_effects[endpoint] = effects
    atomic_json(out_root / "factorial_effects_early_ictal.json", factorial_effects)
    conditional = conditional_effects(patient_rows, fidelity_rows)
    atomic_json(out_root / "early_ictal_conditional_on_interictal_fidelity.json", conditional)
    atomic_json(out_root / "target_access_audit.json", {
        **unlock, "target_values_read": True,
        "actual_primary_subjects": primary, "n_primary_subjects": len(primary),
        "supportive_subject": supportive, "n_seizures": len({(row['subject'], row['seizure_id']) for row in seizure_rows}),
        "null_permutations_synchronized_across_models": True,
        "all_contact_and_within_shaft_rebuild_inside_each_draw": True,
        "training_or_model_selection_after_unseal": False,
    })
    print(json.dumps({"status": "COMPLETE", "n_primary_subjects": len(primary),
                      "n_seizure_model_endpoint_rows": len(seizure_rows),
                      "target_values_read": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
