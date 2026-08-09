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
DOSE_MODELS = ("M5_SPATIAL_LOW", "M6_SPATIAL_MID", "M7_SPATIAL_HIGH")
DOSE_ETA = np.asarray((0.01, 0.03, 0.10), float)
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


def locked_target_artifacts(
    out_root: Path, target_root: Path, metadata: dict[str, Any],
) -> dict[str, list[Path]]:
    """Resolve only the target artifacts frozen by the metadata audit."""
    inventory_path = out_root / "early_ictal_metadata_inventory.csv"
    if sha256(inventory_path) != metadata.get("inventory_csv_sha256"):
        raise RuntimeError("early-ictal metadata inventory changed after freeze")
    if Path(metadata["target_cache_root"]).resolve() != target_root.resolve():
        raise RuntimeError("target cache root differs from the frozen metadata inventory")
    rows = read_csv(inventory_path)
    by_subject: dict[str, list[Path]] = {}
    for row in rows:
        path = Path(row["artifact_path"]).resolve()
        if not path.is_relative_to(target_root.resolve()):
            raise RuntimeError(f"target artifact escaped frozen cache root: {path}")
        if not path.is_file():
            raise RuntimeError(f"frozen target artifact is missing: {path}")
        if sha256(path) != row["artifact_sha256"]:
            raise RuntimeError(f"frozen target artifact hash changed: {path}")
        by_subject.setdefault(row["subject"], []).append(path)
    observed_counts = {subject: len(paths) for subject, paths in by_subject.items()}
    expected_counts = {str(subject): int(count) for subject, count in
                       metadata["seizure_file_counts_filename_only"].items()}
    if observed_counts != expected_counts:
        raise RuntimeError(
            f"frozen target artifact count mismatch: {observed_counts} != {expected_counts}"
        )
    return {subject: sorted(paths) for subject, paths in by_subject.items()}


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


def permutation_support(eligible: np.ndarray, shafts: list[str]) -> dict[str, int]:
    """Describe how much support can actually move in the shaft-preserving null."""
    sizes = [sum(shafts[index] == shaft for index in eligible)
             for shaft in sorted({shafts[index] for index in eligible})]
    return {
        "n_eligible_contacts": int(len(eligible)),
        "n_shafts": int(len(sizes)),
        "n_within_shaft_permutable_contacts": int(sum(size for size in sizes if size > 1)),
        "n_within_shaft_permutable_groups": int(sum(size > 1 for size in sizes)),
    }


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
    if values.size == 0:
        return {
            "n": 0, "median": np.nan, "bootstrap_95ci": [np.nan, np.nan],
            "positive": 0, "negative": 0, "tied": 0,
            "wilcoxon_p": np.nan, "sign_permutation_p": np.nan,
        }
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
        common_all_null = np.stack([null_store[row["null_key_common_all"]] for row in selected])
        common_shaft_null = np.stack([null_store[row["null_key_common_shaft"]] for row in selected])
        pnull_all = np.nanmedian(all_null, axis=0)
        pnull_shaft = np.nanmedian(shaft_null, axis=0)
        pnull_common_all = np.nanmedian(common_all_null, axis=0)
        pnull_common_shaft = np.nanmedian(common_shaft_null, axis=0)
        observed = float(np.nanmedian([row["observed"] for row in selected]))
        common_observed = float(np.nanmedian([row["common_observed"] for row in selected]))
        row = {
            "subject": subject, "primary": subject != supportive, "supportive": subject == supportive,
            "model": model, "cell": cell, "endpoint": endpoint, "n_seizures": len(selected),
            "n_contacts_min": min(int(item["n_contacts"]) for item in selected),
            "within_shaft_permutable_contacts_min": min(
                int(item["n_within_shaft_permutable_contacts"]) for item in selected
            ),
            "within_shaft_permutable_groups_min": min(
                int(item["n_within_shaft_permutable_groups"]) for item in selected
            ),
            "observed": observed,
            "all_contact_null_median": float(np.nanmedian(pnull_all)),
            "all_contact_margin": observed - float(np.nanmedian(pnull_all)),
            "all_contact_p": float((1 + np.sum(pnull_all >= observed - 1e-15)) / (1 + np.isfinite(pnull_all).sum())),
            "within_shaft_null_median": float(np.nanmedian(pnull_shaft)),
            "within_shaft_margin": observed - float(np.nanmedian(pnull_shaft)),
            "within_shaft_p": float((1 + np.sum(pnull_shaft >= observed - 1e-15)) / (1 + np.isfinite(pnull_shaft).sum())),
            "common_observed": common_observed,
            "common_all_contact_null_median": float(np.nanmedian(pnull_common_all)),
            "common_all_contact_margin": common_observed - float(np.nanmedian(pnull_common_all)),
            "common_all_contact_p": float(
                (1 + np.sum(pnull_common_all >= common_observed - 1e-15))
                / (1 + np.isfinite(pnull_common_all).sum())
            ),
            "common_within_shaft_null_median": float(np.nanmedian(pnull_common_shaft)),
            "common_within_shaft_margin": common_observed - float(np.nanmedian(pnull_common_shaft)),
            "common_within_shaft_p": float(
                (1 + np.sum(pnull_common_shaft >= common_observed - 1e-15))
                / (1 + np.isfinite(pnull_common_shaft).sum())
            ),
        }
        patients.append(row)
        key = f"{subject}|{model}|{cell}|{endpoint}"
        patient_nulls[key + "|maxab"] = pnull_all
        patient_nulls[key + "|common"] = pnull_common_all
    return patients, patient_nulls


def _fit_patient_fixed_effects(
    rows: list[dict[str, Any]], models: tuple[str, ...], outcome_key: str = "outcome",
) -> dict[str, Any]:
    """Fit outcome ~ patient block + fidelity + model with no asymptotic SEs."""
    blocks = sorted({str(row["block"]) for row in rows})
    reference = "M0_NO_REC"
    nonreference = [model for model in models if model != reference]
    block_index = {value: index for index, value in enumerate(blocks)}
    model_index = {value: index for index, value in enumerate(nonreference)}
    x = np.zeros((len(rows), len(blocks) + 1 + len(nonreference)), float)
    y = np.asarray([float(row[outcome_key]) for row in rows], float)
    for index, row in enumerate(rows):
        x[index, block_index[str(row["block"])]] = 1.0
        x[index, len(blocks)] = float(row["fidelity"])
        model = str(row["model"])
        if model != reference:
            x[index, len(blocks) + 1 + model_index[model]] = 1.0
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    gamma = {reference: 0.0}
    for model, index in model_index.items():
        gamma[model] = float(coefficients[len(blocks) + 1 + index])
    return {
        "beta": float(coefficients[len(blocks)]),
        "gamma": gamma,
        "fitted": x @ coefficients,
        "residual": y - x @ coefficients,
        "rank": int(np.linalg.matrix_rank(x)),
        "n_parameters": int(x.shape[1]),
    }


def _fit_reduced_patient_fidelity(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Reduced patient + fidelity fit used by the Freedman--Lane null."""
    blocks = sorted({str(row["block"]) for row in rows})
    block_index = {value: index for index, value in enumerate(blocks)}
    x = np.zeros((len(rows), len(blocks) + 1), float)
    y = np.asarray([float(row["outcome"]) for row in rows], float)
    for index, row in enumerate(rows):
        x[index, block_index[str(row["block"])]] = 1.0
        x[index, len(blocks)] = float(row["fidelity"])
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    fitted = x @ coefficients
    return fitted, y - fitted


def conditional_effects(patient_rows: list[dict[str, Any]], fidelity_rows: list[dict[str, str]],
                        endpoint: str = "canonical_full", draws: int = 10000,
                        seed: int = 20260809) -> dict[str, Any]:
    """Patient-blocked model effects with cluster bootstrap and label null.

    Every bootstrap resamples whole patients and refits both the fidelity slope
    and model coefficients.  The permutation is a within-patient
    Freedman--Lane residual-label permutation under the reduced
    patient+fidelity model, so it preserves the two nuisance terms while
    removing model identity.
    """
    fidelity = {(row["subject"], row["model"], row["cell"]): float(row["matched_empirical_r"])
                for row in fidelity_rows}
    selected = [row for row in patient_rows if row["primary"] and row["endpoint"] == endpoint
                and row["cell"] == "rnn" and row["model"] in CORE_MODELS]
    by_patient: dict[str, list[dict[str, Any]]] = {}
    for row in selected:
        row = dict(row)
        key = (row["subject"], row["model"], row["cell"])
        if key not in fidelity or not np.isfinite(fidelity[key]):
            continue
        row["fidelity"] = fidelity[key]
        row["outcome"] = float(row["all_contact_margin"])
        row["block"] = str(row["subject"])
        by_patient.setdefault(row["subject"], []).append(row)
    models = tuple(model for model in CORE_MODELS if model in {
        row["model"] for rows in by_patient.values() for row in rows
    })
    required = set(models)
    complete = sorted(subject for subject, rows in by_patient.items()
                      if {row["model"] for row in rows} == required)
    analysis_rows = [row for subject in complete for row in by_patient[subject]]
    if len(complete) < 2 or len(models) < 2:
        raise RuntimeError("conditional early-ictal model needs at least two complete patients and models")
    observed = _fit_patient_fixed_effects(analysis_rows, models)
    pair_definitions = [
        ("M6_SPATIAL_MID", "M2_UNIFORM_SET"),
        ("M6_SPATIAL_MID", "M1_DENSE"),
        ("M6_SPATIAL_MID", "M0_NO_REC"),
    ]
    pair_definitions = [pair for pair in pair_definitions if set(pair) <= set(models)]

    rng = np.random.default_rng(seed)
    bootstrap_beta = np.empty(draws, float)
    bootstrap_gamma = {model: np.empty(draws, float) for model in models}
    bootstrap_contrast = {f"{a}_vs_{b}": np.empty(draws, float)
                          for a, b in pair_definitions}
    for draw in range(draws):
        sampled = rng.choice(complete, size=len(complete), replace=True)
        rows = []
        for replicate, subject in enumerate(sampled):
            for source in by_patient[str(subject)]:
                row = dict(source)
                row["block"] = f"{subject}__bootstrap_{replicate}"
                rows.append(row)
        fitted = _fit_patient_fixed_effects(rows, models)
        bootstrap_beta[draw] = fitted["beta"]
        for model in models:
            bootstrap_gamma[model][draw] = fitted["gamma"][model]
        for a, b in pair_definitions:
            bootstrap_contrast[f"{a}_vs_{b}"][draw] = fitted["gamma"][a] - fitted["gamma"][b]

    reduced_fitted, reduced_residual = _fit_reduced_patient_fidelity(analysis_rows)
    block_indices = {
        subject: np.asarray([index for index, row in enumerate(analysis_rows)
                             if row["block"] == subject], int)
        for subject in complete
    }
    permutation_gamma = {model: np.empty(draws, float) for model in models}
    permutation_contrast = {f"{a}_vs_{b}": np.empty(draws, float)
                            for a, b in pair_definitions}
    for draw in range(draws):
        permuted = reduced_residual.copy()
        for indices in block_indices.values():
            permuted[indices] = rng.permutation(permuted[indices])
        rows = [dict(row, permuted_outcome=float(value))
                for row, value in zip(analysis_rows, reduced_fitted + permuted)]
        fitted = _fit_patient_fixed_effects(rows, models, outcome_key="permuted_outcome")
        for model in models:
            permutation_gamma[model][draw] = fitted["gamma"][model]
        for a, b in pair_definitions:
            permutation_contrast[f"{a}_vs_{b}"][draw] = fitted["gamma"][a] - fitted["gamma"][b]

    def inference(value: float, bootstrap: np.ndarray, permutation: np.ndarray) -> dict[str, Any]:
        return {
            "estimate": float(value),
            "patient_cluster_bootstrap_95ci": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "patient_label_permutation_p": float(
                (1 + np.sum(np.abs(permutation) >= abs(value) - 1e-15)) / (draws + 1)
            ),
        }

    model_effects = {
        model: inference(observed["gamma"][model], bootstrap_gamma[model], permutation_gamma[model])
        for model in models if model != "M0_NO_REC"
    }
    contrasts = {}
    for a, b in pair_definitions:
        name = f"{a}_vs_{b}"
        value = observed["gamma"][a] - observed["gamma"][b]
        contrasts[name] = inference(value, bootstrap_contrast[name], permutation_contrast[name])
    return {
        "endpoint": endpoint,
        "model": "early_margin ~ patient_fixed_effect + interictal_fidelity + model",
        "reference_model": "M0_NO_REC",
        "complete_patients": complete,
        "excluded_incomplete_patients": sorted(set(by_patient) - set(complete)),
        "n_complete_patients": len(complete),
        "n_rows": len(analysis_rows),
        "bootstrap_draws": draws,
        "permutation_draws": draws,
        "permutation_contract": "within-patient Freedman-Lane residual-label permutation",
        "within_patient_fidelity_slope": {
            "estimate": observed["beta"],
            "patient_cluster_bootstrap_95ci": np.quantile(
                bootstrap_beta, [0.025, 0.975]
            ).tolist(),
            "permutation_p": None,
            "reason_no_permutation_p": "fidelity is a nuisance covariate, not a tested model label",
        },
        "model_effects": model_effects,
        "contrasts": contrasts,
        "design_rank": observed["rank"],
        "n_parameters": observed["n_parameters"],
    }


def compute_factorial_effects(
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    primary: list[str], endpoint: str,
) -> dict[str, Any]:
    """Compute every 2x2 contrast on one identical complete-patient denominator."""
    required = set(FACTORIAL)
    complete = [subject for subject in primary if all(
        (subject, model, "rnn", endpoint) in lookup for model in required
    )]
    definitions = {
        "growth_at_zero": ("M4_SPATIAL_GROWTH", "M2_UNIFORM_SET"),
        "growth_at_mid": ("M6_SPATIAL_MID", "M8_UNIFORM_COST_MID"),
        "cost_uniform": ("M8_UNIFORM_COST_MID", "M2_UNIFORM_SET"),
        "cost_spatial": ("M6_SPATIAL_MID", "M4_SPATIAL_GROWTH"),
    }
    effects: dict[str, Any] = {
        "n_complete_patients": len(complete),
        "complete_patients": complete,
        "excluded_incomplete_patients": sorted(set(primary) - set(complete)),
    }
    raw: dict[str, np.ndarray] = {}
    for name, (a_model, b_model) in definitions.items():
        raw[name] = np.asarray([
            lookup[(subject, a_model, "rnn", endpoint)]["all_contact_margin"]
            - lookup[(subject, b_model, "rnn", endpoint)]["all_contact_margin"]
            for subject in complete
        ], float)
        effects[name] = paired_summary(raw[name], seed=stable_seed(endpoint, name))
    effects["interaction"] = paired_summary(
        raw["cost_spatial"] - raw["cost_uniform"],
        seed=stable_seed(endpoint, "interaction"),
    )
    family = tuple(definitions) + ("interaction",)
    adjusted = holm({name: float(effects[name]["wilcoxon_p"]) for name in family})
    for name in family:
        effects[name]["holm_q_factorial_family"] = adjusted[name]
    effects["holm_family"] = list(family)
    return effects


def compute_dose_trend(
    lookup: dict[tuple[str, str, str, str], dict[str, Any]],
    primary: list[str], endpoint: str,
) -> dict[str, Any]:
    """Patient-level slope across the three preassigned positive eta values."""
    complete = [subject for subject in primary if all(
        (subject, model, "rnn", endpoint) in lookup for model in DOSE_MODELS
    )]
    x = np.log10(DOSE_ETA)
    x = x - x.mean()
    denominator = float(np.dot(x, x))
    slopes = []
    for subject in complete:
        y = np.asarray([
            lookup[(subject, model, "rnn", endpoint)]["all_contact_margin"]
            for model in DOSE_MODELS
        ], float)
        slopes.append(float(np.dot(x, y - y.mean()) / denominator))
    result = paired_summary(slopes, seed=stable_seed(endpoint, "spatial_eta_dose_trend"))
    result.update({
        "models": list(DOSE_MODELS),
        "eta": DOSE_ETA.tolist(),
        "x_scale": "log10_eta",
        "complete_patients": complete,
        "excluded_incomplete_patients": sorted(set(primary) - set(complete)),
        "interpretation": "positive slope means larger wiring cost has larger early-ictal null-relative margin",
    })
    return result


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
    target_files_by_subject = locked_target_artifacts(out_root, target_root, metadata)

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

        target_files = target_files_by_subject[subject]
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
                perm_support = permutation_support(eligible, shafts)
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
                    common = 0.5 * (a + b)
                    common_scorer = build_scorer(record, common, common, finite)
                    common_all_score = score_one(common_scorer, target, perm_all)
                    common_shaft_score = score_one(common_scorer, target, perm_shaft)
                    key = f"{subject}|{seizure_id}|{model}|{cell}|{endpoint}"
                    key_all, key_shaft = key + "|all", key + "|shaft"
                    key_common_all, key_common_shaft = key + "|common_all", key + "|common_shaft"
                    null_store[key_all] = all_score["null"]
                    null_store[key_shaft] = shaft_score["null"]
                    null_store[key_common_all] = common_all_score["null"]
                    null_store[key_common_shaft] = common_shaft_score["null"]
                    for label, score_all, score_shaft in (
                        (key + "|maxab", all_score, shaft_score),
                        (key + "|common", common_all_score, common_shaft_score),
                    ):
                        null_labels.append(label)
                        null_all.append(score_all["null"].astype(np.float32))
                        null_shaft.append(score_shaft["null"].astype(np.float32))
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
                        "common_observed": common_all_score["observed"],
                        "common_all_contact_null_median": common_all_score["null_median"],
                        "common_all_contact_margin": common_all_score["margin"],
                        "common_all_contact_p": common_all_score["empirical_p"],
                        "common_within_shaft_null_median": common_shaft_score["null_median"],
                        "common_within_shaft_margin": common_shaft_score["margin"],
                        "common_within_shaft_p": common_shaft_score["empirical_p"],
                        **perm_support,
                        "null_key_all": key_all, "null_key_shaft": key_shaft,
                        "null_key_common_all": key_common_all,
                        "null_key_common_shaft": key_common_shaft,
                    })

    write_csv(out_root / "early_ictal_per_seizure.csv", seizure_rows)
    patient_rows, patient_nulls = aggregate_patients(seizure_rows, null_store, supportive)
    fidelity_lookup = {
        (row["subject"], row["model"], row["cell"]): row for row in fidelity_rows
    }
    for row in patient_rows:
        if row["model"] == "EMPIRICAL_REFERENCE":
            row["interictal_common_fidelity"] = 1.0
            row["interictal_contrast_fidelity"] = 1.0
            continue
        fidelity = fidelity_lookup.get((row["subject"], row["model"], row["cell"]), {})
        prefix = "canonical" if row["endpoint"] == "canonical_full" else "seed_removed"
        row["interictal_common_fidelity"] = fidelity.get(f"{prefix}_common_fidelity", np.nan)
        row["interictal_contrast_fidelity"] = fidelity.get(f"{prefix}_contrast_fidelity", np.nan)
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
            common_values = [float(lookup[(subject, model, cell, endpoint)]["common_all_contact_margin"])
                             for subject in primary if (subject, model, cell, endpoint) in lookup]
            if common_values:
                contrasts[f"{endpoint}|{model}__{cell}_common_margin_gt_zero"] = paired_summary(
                    common_values, seed=stable_seed(endpoint, model, cell, "common_zero")
                )
    atomic_json(out_root / "early_ictal_model_contrasts.json", contrasts)

    factorial_effects = {}
    for endpoint in ENDPOINTS:
        factorial_effects[endpoint] = compute_factorial_effects(lookup, primary, endpoint)
        factorial_effects[endpoint]["spatial_eta_dose_trend"] = compute_dose_trend(
            lookup, primary, endpoint
        )
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
    atomic_json(out_root / "stage_f_scientific_drift_audit.json", {
        "status": "ALIGNED",
        "scientific_question": (
            "whether target-free frozen model-generated interictal fields reproduce the existing "
            "within-patient early-ictal broadband field correspondence"
        ),
        "target_role": "external frozen benchmark only",
        "target_values_read_after_field_freeze": True,
        "n_primary_subjects": len(primary),
        "cohort_mismatch_known_before_unseal": metadata.get("join_status"),
        "primary_endpoint": "canonical_full maxAB versus synchronized all-contact null",
        "secondary_endpoints": ["seed_removed maxAB", "common-field concordance",
                                "interictal A/B contrast fidelity", "within-shaft sensitivity"],
        "not_claimed": ["seizure prediction", "causal interictal-to-ictal transition",
                        "recovery of an anatomical connectome"],
    })
    print(json.dumps({"status": "COMPLETE", "n_primary_subjects": len(primary),
                      "n_seizure_model_endpoint_rows": len(seizure_rows),
                      "target_values_read": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
