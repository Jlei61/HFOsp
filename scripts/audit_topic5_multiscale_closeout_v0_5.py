#!/usr/bin/env python3
"""Requirement-by-requirement closeout audit for Topic 5.1 v0.5.

This verifier is intentionally independent of the launcher.  It treats every
completion claim as unproven until the current artifacts, payload hashes,
denominators, target-access ordering and rendered assets are all checked.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_rnn_motif_v0_4 import ROLLOUT_DECODER_CONTRACT  # noqa: E402


CANONICAL_ROOT = ROOT.parents[1] if (ROOT.parents[1] / "results").exists() else ROOT
DEFAULT_OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
DEFAULT_FIGURE = ROOT / "results/paper-ready-figure/fig6_multiscale_scaffold_v0_5/figures"
MASKED_RANK_DATASET = (
    CANONICAL_ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4/per_subject"
)
BROADBAND_TARGET_ROOT = (
    CANONICAL_ROOT
    / "results/topic5_ictal_recruitment/t0_feature_cache_bb150_1_150"
)
PARENT_V0_3_COMMIT = "bd9d86217eb6bed013661b0f6d8aa8f397f6c986"
PARENT_V0_3_TAG = "topic5-lbss-full-tissue-v0.3-closeout"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def iso_timestamp(payload: dict, key: str = "created_utc") -> datetime:
    value = str(payload[key]).replace("Z", "+00:00")
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def verify_manifest(out: Path, relative: str, hash_column: str) -> dict:
    path = out / relative
    frame = pd.read_csv(path)
    failures = []
    if "target_values_read" in frame and not frame.target_values_read.eq(False).all():
        failures.append("TARGET_FLAG_NOT_UNIFORMLY_FALSE")
    for row in frame[["path", hash_column]].drop_duplicates().itertuples(index=False):
        payload = Path(str(row[0]))
        if not payload.is_absolute():
            payload = out / payload
        if not payload.exists():
            failures.append(f"MISSING:{payload}")
        elif sha256_file(payload) != str(row[1]):
            failures.append(f"HASH_MISMATCH:{payload}")
    return {"rows": len(frame), "failures": failures, "pass": not failures}


def compare_numeric_tables(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    keys: list[str],
    atol: float = 1e-12,
) -> dict:
    """Compare independently reconstructed patient-first numeric aggregates."""
    common = [
        column for column in observed.select_dtypes(include="number").columns
        if column in expected.columns and column not in keys
    ]
    merged = observed.merge(
        expected, on=keys, suffixes=("_observed", "_expected"),
        how="outer", validate="one_to_one", indicator=True,
    )
    failures = []
    if not merged._merge.eq("both").all():
        failures.append("ROW_KEY_MISMATCH")
    maximum = 0.0
    for column in common:
        left = merged[f"{column}_observed"].to_numpy(float)
        right = merged[f"{column}_expected"].to_numpy(float)
        finite = np.isfinite(left) & np.isfinite(right)
        mismatch = np.isfinite(left) ^ np.isfinite(right)
        if mismatch.any():
            failures.append(f"FINITE_MASK_MISMATCH:{column}")
        if finite.any():
            delta = float(np.max(np.abs(left[finite] - right[finite])))
            maximum = max(maximum, delta)
            if delta > atol:
                failures.append(f"NUMERIC_MISMATCH:{column}:{delta:.6g}")
    return {
        "pass": not failures,
        "rows": len(merged),
        "numeric_columns": common,
        "maximum_absolute_error": maximum,
        "failures": failures,
    }


def verify_source_snapshot(snapshot: dict, mapping: dict[str, Path] | None = None) -> dict:
    failures = []
    for relative, expected in snapshot.get("source_hashes", {}).items():
        path = mapping.get(relative) if mapping else ROOT / relative
        if path is None or not path.exists():
            failures.append(f"MISSING:{relative}")
        elif sha256_file(path) != expected:
            failures.append(f"HASH_MISMATCH:{relative}")
    return {
        "pass": not failures,
        "files": len(snapshot.get("source_hashes", {})),
        "failures": failures,
    }


def assess_suffix_control_lineage(
    out: Path,
    input_manifest: dict,
    metrics: list[dict],
    expected_fits: int = 42,
) -> dict:
    """Prove that formal C-suffix units consumed the frozen cross-event null.

    v0.5 must not silently fall back to the older within-event rank derangement.
    Each model seed therefore points to its own precomputed, hash-frozen suffix
    reassignment.  The null may alter train/validation ranks but must preserve
    the real held-out test decisions exactly.
    """
    failures: list[str] = []
    records = {
        str(record.get("fit_id")): record.get("files", {})
        for record in input_manifest.get("cache_records", [])
    }
    suffix_metrics = [
        row for row in metrics if row.get("arm") == "C_L3_ORDER_SHUFFLED"
    ]
    expected_keys = {
        (fit_id, seed) for fit_id in records for seed in range(3)
    }
    observed_keys: list[tuple[str, int]] = []
    changed_train_validation = 0
    heldout_exact = 0
    stored_but_unconsumed_test_differences = 0
    for row in suffix_metrics:
        fit_id = str(row.get("fit_id"))
        seed = int(row.get("seed", -1))
        key = (fit_id, seed)
        observed_keys.append(key)
        expected_name = f"events_suffix_null_seed{seed}.npz"
        audit = row.get("shuffle_audit") or {}
        evidence = records.get(fit_id, {}).get(expected_name, {})
        path = out / "cache" / fit_id / expected_name
        real_path = out / "cache" / fit_id / "events.npz"
        label = f"{fit_id}:seed{seed}"
        if audit.get("scope") != "precomputed_suffix_pairing_train_and_validation_only":
            failures.append(f"SCOPE:{label}")
        if audit.get("events_file_name") != expected_name:
            failures.append(f"FILE_NAME:{label}")
        if audit.get("heldout_test_unchanged") is not True:
            failures.append(f"METRIC_HELDOUT_FLAG:{label}")
        if not path.exists() or not real_path.exists():
            failures.append(f"MISSING_INPUT:{label}")
            continue
        actual_hash = sha256_file(path)
        if evidence.get("sha256") != actual_hash:
            failures.append(f"MANIFEST_HASH:{label}")
        if audit.get("null_events_sha256") != actual_hash:
            failures.append(f"METRIC_HASH:{label}")
        with np.load(real_path, allow_pickle=False) as real, np.load(
            path, allow_pickle=False
        ) as null:
            if not np.array_equal(real["split"], null["split"]):
                failures.append(f"SPLIT_CHANGED:{label}")
                continue
            split = np.asarray(real["split"])
            # Reconstruct the exact application in train_unit: null rows are
            # copied only for train/validation, never for the held-out test.
            effective = np.asarray(real["ranks"]).copy()
            train_validation = (split == 0) | (split == 1)
            effective[train_validation] = null["ranks"][train_validation]
            if not np.array_equal(effective[split == 2], real["ranks"][split == 2]):
                failures.append(f"EFFECTIVE_HELDOUT_CHANGED:{label}")
            else:
                heldout_exact += 1
            stored_but_unconsumed_test_differences += int(
                not np.array_equal(
                    real["ranks"][split == 2], null["ranks"][split == 2]
                )
            )
            if np.array_equal(
                real["ranks"][(split == 0) | (split == 1)],
                null["ranks"][(split == 0) | (split == 1)],
            ):
                failures.append(f"TRAIN_VALIDATION_NOT_CHANGED:{label}")
            else:
                changed_train_validation += 1
    duplicate_keys = len(observed_keys) - len(set(observed_keys))
    if duplicate_keys:
        failures.append(f"DUPLICATE_UNITS:{duplicate_keys}")
    if set(observed_keys) != expected_keys:
        failures.append("FIT_SEED_KEY_MISMATCH")
    if len(records) != expected_fits:
        failures.append(f"FIT_COUNT:{len(records)}")
    return {
        "pass": not failures,
        "contract": "FROZEN_CROSS_EVENT_SUFFIX_REASSIGNMENT_PER_MODEL_SEED",
        "formal_suffix_units": len(suffix_metrics),
        "expected_suffix_units": expected_fits * 3,
        "fits": len(records),
        "heldout_test_exact_units": heldout_exact,
        "train_validation_changed_units": changed_train_validation,
        "stored_but_unconsumed_test_differences": (
            stored_but_unconsumed_test_differences
        ),
        "failures": failures,
    }


def assess_geometry_view_event_scope(out: Path, census: pd.DataFrame) -> dict:
    """Prove own_a/own_b are geometry views of the same all-event task.

    A previous prose incident described these fits as event-family-filtered.
    The production cache instead keeps all eligible events in both views.  This
    check makes that interpretation boundary executable rather than relying on
    the corrected prose alone.
    """
    failures: list[str] = []
    exact_subjects = 0
    noncollinear = census[census.scope.astype(str).isin(["own_a", "own_b"])]
    for subject, rows in noncollinear.groupby("subject", sort=False):
        failure_count_before = len(failures)
        by_scope = {str(row.scope): str(row.fit_id) for row in rows.itertuples()}
        if set(by_scope) != {"own_a", "own_b"}:
            failures.append(f"SCOPE_PAIR:{subject}:{sorted(by_scope)}")
            continue
        left_path = out / "cache" / by_scope["own_a"] / "events.npz"
        right_path = out / "cache" / by_scope["own_b"] / "events.npz"
        if not left_path.exists() or not right_path.exists():
            failures.append(f"MISSING_EVENTS:{subject}")
            continue
        with np.load(left_path, allow_pickle=False) as left, np.load(
            right_path, allow_pickle=False
        ) as right:
            for key in ("ranks", "split", "mode"):
                if key not in left.files or key not in right.files:
                    failures.append(f"MISSING_KEY:{subject}:{key}")
                elif not np.array_equal(left[key], right[key]):
                    failures.append(f"EVENT_VIEW_MISMATCH:{subject}:{key}")
        if len(failures) == failure_count_before:
            exact_subjects += 1
    return {
        "pass": not failures and noncollinear.subject.nunique() == 14
        and exact_subjects == 14,
        "noncollinear_patients": int(noncollinear.subject.nunique()),
        "exact_all_event_geometry_view_pairs": exact_subjects,
        "contract": "OWN_A_OWN_B_SHARE_IDENTICAL_ALL_EVENT_TASK_AND_DIFFER_ONLY_IN_GEOMETRY_VIEW",
        "failures": failures,
    }


def assess_gain_matching(gain_seed: pd.DataFrame, tolerance: float = 0.01) -> dict:
    """Verify that each validation-only L2m/L3 gain match really succeeded."""
    gain_pairs = gain_seed.pivot(
        index=["subject", "fit_id", "scope", "seed"],
        columns="arm",
        values="validation_G3_matched",
    )
    if gain_pairs.shape[1] == 2:
        denominator = gain_pairs.abs().max(axis=1).clip(lower=1e-12)
        relative_error = (
            gain_pairs.iloc[:, 0] - gain_pairs.iloc[:, 1]
        ).abs() / denominator
    else:
        relative_error = pd.Series(dtype=float)
    scale_valid = bool(
        np.isfinite(gain_seed.validation_G3_intact.to_numpy(float)).all()
        and np.isfinite(gain_seed.validation_G3_matched.to_numpy(float)).all()
        and gain_seed.recurrent_scale.between(0.0, 1.0, inclusive="right").all()
    )
    passed = bool(
        len(gain_pairs) == 126
        and len(relative_error) == 126
        and np.isfinite(relative_error.to_numpy(float)).all()
        and float(relative_error.max()) <= float(tolerance)
        and scale_valid
    )
    return {
        "pass": passed,
        "pairs": len(gain_pairs),
        "maximum_relative_error": (
            float(relative_error.max()) if len(relative_error) else None
        ),
        "median_relative_error": (
            float(relative_error.median()) if len(relative_error) else None
        ),
        "maximum_allowed_relative_error": float(tolerance),
        "scale_range": (
            [float(gain_seed.recurrent_scale.min()),
             float(gain_seed.recurrent_scale.max())]
            if len(gain_seed) else None
        ),
    }


def densify_groups(groups: np.ndarray) -> np.ndarray:
    """Re-index present tie groups from zero while retaining absence as -1."""
    values = np.asarray(groups, dtype=np.int16)
    output = np.full_like(values, -1)
    for event_index, row in enumerate(values):
        present = np.unique(row[row >= 0])
        for new_rank, old_rank in enumerate(present):
            output[event_index, row == old_rank] = new_rank
    return output


def assess_masked_rank_lineage(dataset_root: Path, cache_root: Path) -> dict:
    """Prove that v0.5 consumes participation-masked tie groups, not phantom ranks."""
    failures: list[str] = []
    fit_count = 0
    subjects: set[str] = set()
    cached_events = 0
    source_mask_mismatches = 0
    cache_value_mismatches = 0
    nondense_cache_events = 0
    checked_source_paths: set[Path] = set()
    for provenance_path in sorted(cache_root.glob("*/provenance.json")):
        provenance = load_json(provenance_path)
        fit_id = str(provenance["fit_id"])
        subject = str(provenance["subject"])
        subjects.add(subject)
        fit_count += 1
        source_path = dataset_root / f"{subject}.npz"
        if not source_path.exists():
            failures.append(f"MISSING_SOURCE:{subject}")
            continue
        if sha256_file(source_path) != str(provenance["dataset_sha256"]):
            failures.append(f"SOURCE_HASH_MISMATCH:{subject}")
        with np.load(source_path, allow_pickle=False) as source:
            required = {"event_group_ids", "event_participation", "contact_names"}
            if not required.issubset(source.files):
                failures.append(f"SOURCE_FIELDS_MISSING:{subject}")
                continue
            source_groups = np.asarray(source["event_group_ids"], dtype=np.int16)
            source_participation = np.asarray(source["event_participation"], dtype=bool)
            names = [str(value) for value in source["contact_names"]]
        if source_path not in checked_source_paths:
            source_mask_mismatches += int(
                np.count_nonzero((source_groups >= 0) != source_participation)
            )
            checked_source_paths.add(source_path)
        joint = [str(value) for value in provenance["joint_contacts"]]
        try:
            columns = np.asarray([names.index(name) for name in joint], dtype=int)
        except ValueError:
            failures.append(f"JOINT_CONTACT_NOT_IN_SOURCE:{fit_id}")
            continue
        events_path = provenance_path.parent / "events_raw.npz"
        if not events_path.exists():
            failures.append(f"MISSING_CACHE_EVENTS:{fit_id}")
            continue
        with np.load(events_path, allow_pickle=False) as cache:
            cached = np.asarray(cache["ranks"], dtype=np.int16)
        expected = densify_groups(source_groups[:, columns])
        cached_events += int(len(cached))
        if cached.shape != expected.shape:
            failures.append(f"CACHE_SHAPE_MISMATCH:{fit_id}")
            continue
        cache_value_mismatches += int(np.count_nonzero(cached != expected))
        for row in cached:
            present = np.unique(row[row >= 0])
            if present.size and not np.array_equal(present, np.arange(present.size)):
                nondense_cache_events += 1
    passed = bool(
        fit_count == 42
        and len(subjects) == 28
        and source_mask_mismatches == 0
        and cache_value_mismatches == 0
        and nondense_cache_events == 0
        and not failures
    )
    return {
        "pass": passed,
        "fits": fit_count,
        "subjects": len(subjects),
        "unique_source_files": len(checked_source_paths),
        "cached_events": cached_events,
        "source_participation_mask_mismatches": source_mask_mismatches,
        "cache_value_mismatches": cache_value_mismatches,
        "nondense_cache_events": nondense_cache_events,
        "absence_sentinel": -1,
        "source_contract": "MASKED_DATASET_V0_4_EVENT_GROUP_IDS_FROM_PARTICIPATION_AND_RAW_LAG",
        "failures": failures,
    }


def assess_attenuation_draw_semantics(attenuation: pd.DataFrame) -> dict:
    """Independently recheck the arm-specific attenuation draw contract."""
    failures: list[str] = []
    valid = attenuation[attenuation.draw.astype(int) >= 0].copy()
    for column in (
        "contact_nll", "local_nll", "distal_nll",
        "local_damage", "distal_damage", "distal_selectivity",
    ):
        if not np.isfinite(valid[column].to_numpy(float)).all():
            failures.append(f"NONFINITE_VALID_ROW:{column}")
    undefined = ~np.isfinite(valid.rollout_spearman.to_numpy(float))
    if undefined.any() and not valid.loc[undefined, "rollout_spearman_n"].eq(0).all():
        failures.append("UNDEFINED_ROLLOUT_WITH_NONZERO_DENOMINATOR")
    for keys, group in attenuation.groupby(
        ["subject", "fit_id", "target", "seed"], sort=False
    ):
        label = "|".join(map(str, keys))
        if sorted(map(float, group.alpha.unique())) != [0.25, 0.5, 0.75, 1.0]:
            failures.append(f"DOSE_SET:{label}")
            continue
        counts = group.n_valid_matched_draws.astype(int).unique()
        if len(counts) != 1:
            failures.append(f"VALID_DRAW_COUNT_DRIFT:{label}")
            continue
        n_valid = int(counts[0])
        expected_eligible = n_valid >= 200 if keys[2] == "L3_MATCHED_LOCAL" else True
        if not group.inferential_eligible.astype(bool).eq(expected_eligible).all():
            failures.append(f"ELIGIBILITY_MISMATCH:{label}")
        if keys[2] != "L3_MATCHED_LOCAL":
            if len(group) != 4 or set(group.draw.astype(int)) != {0}:
                failures.append(f"ADDED_EDGE_DRAW_CONTRACT:{label}")
            if group.target_mask_sha256.nunique(dropna=True) != 1:
                failures.append(f"ADDED_EDGE_MASK_DRIFT:{label}")
            continue
        expected_draws = min(16, n_valid)
        if expected_draws == 0:
            if len(group) != 4 or set(group.draw.astype(int)) != {-1}:
                failures.append(f"EMPTY_MATCHED_LOCAL_PLACEHOLDER:{label}")
            continue
        if len(group) != 4 * expected_draws:
            failures.append(f"MATCHED_LOCAL_ROW_COUNT:{label}")
        expected_ids = set(range(expected_draws))
        for alpha, dose in group.groupby("alpha"):
            if set(dose.draw.astype(int)) != expected_ids:
                failures.append(f"MATCHED_LOCAL_DRAW_SET:{label}:{alpha}")
        mask_by_draw = group.groupby("draw").target_mask_sha256.nunique(dropna=True)
        if not mask_by_draw.eq(1).all() or group.target_mask_sha256.nunique(dropna=True) != expected_draws:
            failures.append(f"MATCHED_LOCAL_MASK_CONTRACT:{label}")
    return {
        "pass": not failures,
        "rows": int(len(attenuation)),
        "valid_rows": int(len(valid)),
        "rollout_undefined_rows": int(undefined.sum()),
        "rollout_undefined_fraction": (
            float(undefined.mean()) if len(undefined) else 0.0
        ),
        "failures": failures,
    }


def assess_broadband_target_lineage(
    out: Path,
    target_root: Path,
    expected_patients: int = 17,
    expected_seizures: int = 167,
) -> dict:
    """After unseal, prove every saved target equals the registered source cache."""
    failures: list[str] = []
    routing = pd.read_csv(out / "EARLY_ICTAL_ROUTING_METADATA.csv")
    manifest = pd.read_csv(out / "early_ictal/EARLY_ICTAL_TARGET_MANIFEST.csv")
    source_hashes: list[dict] = []
    compared_values = 0
    for subject, routes in routing.groupby("subject", sort=False):
        rows = manifest[manifest.subject.astype(str) == str(subject)]
        if len(rows) != 1:
            failures.append(f"TARGET_MANIFEST_ROW_COUNT:{subject}:{len(rows)}")
            continue
        derived_path = Path(str(rows.iloc[0].path))
        if not derived_path.is_absolute():
            derived_path = out / derived_path
        if not derived_path.exists() or sha256_file(derived_path) != str(rows.iloc[0].sha256):
            failures.append(f"DERIVED_TARGET_HASH:{subject}")
            continue
        metadata_path = target_root / f"{subject}.json"
        source_path = target_root / f"{subject}.npz"
        if not metadata_path.exists() or not source_path.exists():
            failures.append(f"SOURCE_TARGET_MISSING:{subject}")
            continue
        metadata = load_json(metadata_path)
        if list(map(float, metadata.get("band_broad_1_150", []))) != [1.0, 150.0]:
            failures.append(f"SOURCE_BAND_CONTRACT:{subject}")
        if list(map(float, metadata.get("t_window", []))) != [0.0, 10.0]:
            failures.append(f"SOURCE_TIME_CONTRACT:{subject}")
        if metadata.get("line_noise_masked_1_150") is not True:
            failures.append(f"SOURCE_LINE_MASK_CONTRACT:{subject}")
        if "mean baseline-robust-z 1-150Hz over [0,10]s" not in str(metadata.get("feature", "")):
            failures.append(f"SOURCE_FEATURE_CONTRACT:{subject}")
        source_hashes.append({
            "subject": str(subject),
            "metadata_sha256": sha256_file(metadata_path),
            "source_npz_sha256": sha256_file(source_path),
        })
        with np.load(derived_path, allow_pickle=False) as derived:
            contacts = [str(value) for value in derived["contacts"]]
            observed = np.asarray(derived["all_seizure_broadband_energy"], float)
            saved_median = np.asarray(derived["median_broadband_energy"], float)
            n_seizures = int(np.asarray(derived["n_seizures"]).item())
            time_window = np.asarray(derived["time_window_s"], float)
            band = np.asarray(derived["frequency_band_hz"], float)
        if n_seizures != len(routes) or observed.shape != (len(routes), len(contacts)):
            failures.append(f"DERIVED_TARGET_SHAPE:{subject}")
            continue
        if not np.array_equal(time_window, np.asarray([0.0, 10.0])):
            failures.append(f"DERIVED_TIME_CONTRACT:{subject}")
        if not np.array_equal(band, np.asarray([1.0, 150.0])):
            failures.append(f"DERIVED_BAND_CONTRACT:{subject}")
        names = [str(value) for value in metadata.get("channels", [])]
        if not set(contacts).issubset(names):
            failures.append(f"CONTACT_JOIN:{subject}")
            continue
        columns = np.asarray([names.index(name) for name in contacts], dtype=int)
        with np.load(source_path, allow_pickle=False) as source:
            expected_rows = []
            for event in routes.itertuples(index=False):
                key = f"bb150_auc__{int(event.seizure_idx)}"
                if key not in source.files:
                    failures.append(f"SOURCE_MEMBER_MISSING:{subject}:{key}")
                    continue
                expected_rows.append(np.asarray(source[key], float)[columns])
        if len(expected_rows) != len(routes):
            continue
        expected = np.stack(expected_rows)
        if not np.array_equal(observed, expected):
            delta = float(np.nanmax(np.abs(observed - expected)))
            failures.append(f"TARGET_VALUE_MISMATCH:{subject}:{delta:.6g}")
        if not np.isfinite(observed).all():
            failures.append(f"NONFINITE_DERIVED_TARGET:{subject}")
        compared_values += int(observed.size)
        if not np.array_equal(saved_median, np.nanmedian(expected, axis=0)):
            failures.append(f"TARGET_MEDIAN_MISMATCH:{subject}")
    return {
        "pass": bool(
            routing.subject.nunique() == expected_patients
            and len(routing) == expected_seizures
            and manifest.subject.nunique() == expected_patients
            and len(source_hashes) == expected_patients
            and not failures
        ),
        "patients": int(routing.subject.nunique()),
        "seizures": int(len(routing)),
        "compared_contact_values": compared_values,
        "target": "clinical onset 0-10 s mean baseline-robust-z 1-150 Hz broadband energy",
        "source_hashes": source_hashes,
        "failures": failures,
    }


def _signed_spearman_with_frozen_permutations(
    prediction: np.ndarray,
    target: np.ndarray,
    permutations: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Independent tied-rank implementation of the locked spatial null."""
    prediction = np.asarray(prediction, float)
    target = np.asarray(target, float)
    permutations = np.asarray(permutations, dtype=np.intp)
    if (
        prediction.ndim != 1
        or target.ndim != 1
        or len(prediction) != len(target)
        or permutations.ndim != 2
        or permutations.shape[1] != len(target)
        or not np.isfinite(prediction).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("primary null-fold audit received an invalid vector")
    x = rankdata(prediction, method="average").astype(float)
    y = rankdata(target, method="average").astype(float)
    x -= x.mean()
    y -= y.mean()
    denominator = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denominator <= 0:
        return float("nan"), np.full(len(permutations), np.nan, dtype=np.float32)
    observed = float(np.dot(x, y) / denominator)
    null = (y[permutations] @ x / denominator).astype(np.float32, copy=False)
    return observed, null


def assess_primary_early_ictal_null_folding(
    out: Path,
    routing: pd.DataFrame,
    seizure: pd.DataFrame,
    patient: pd.DataFrame,
) -> dict:
    """Rebuild primary maxAB and the synchronized null from frozen inputs."""
    condition = "INTACT|L3_LOCAL_PLUS_LEARNED_LR"
    endpoint = "canonical_full"
    failures: list[str] = []
    records: list[dict] = []
    scored = seizure[
        (seizure.condition == condition) & (seizure.endpoint == endpoint)
    ].copy()
    expected_mode = np.full(len(scored), "NOT_IDENTIFIABLE", dtype=object)
    choose_a = np.isfinite(scored.mode_a_r) & (
        ~np.isfinite(scored.mode_b_r) | (scored.mode_a_r >= scored.mode_b_r)
    )
    choose_b = ~choose_a & np.isfinite(scored.mode_b_r)
    expected_mode[choose_a] = "A"
    expected_mode[choose_b] = "B"
    mode_metadata_mismatches = int(
        np.sum(scored.selected_mode.astype(str).to_numpy() != expected_mode)
    )
    if mode_metadata_mismatches:
        failures.append(f"SELECTED_MODE_METADATA:{mode_metadata_mismatches}")

    for subject, routes in routing.groupby("subject", sort=False):
        field_path = (
            out / "model_fields/intact/per_patient" / str(subject)
            / "L3_LOCAL_PLUS_LEARNED_LR.npz"
        )
        target_path = out / "early_ictal/per_patient_targets" / f"{subject}.npz"
        if not field_path.exists() or not target_path.exists():
            failures.append(f"MISSING_PRIMARY_INPUT:{subject}")
            continue
        with np.load(field_path, allow_pickle=False) as fields:
            contacts = fields["contacts"].astype(str).tolist()
            mode_a = np.asarray(fields["A_canonical_full"], float)
            mode_b = np.asarray(fields["B_canonical_full"], float)
        with np.load(target_path, allow_pickle=False) as targets:
            target_contacts = targets["contacts"].astype(str).tolist()
            target_matrix = np.asarray(targets["all_seizure_broadband_energy"], float)
        if contacts != target_contacts or len(routes) != len(target_matrix):
            failures.append(f"PRIMARY_INPUT_ALIGNMENT:{subject}")
            continue

        observed_by_seizure: list[float] = []
        null_by_seizure: list[np.ndarray] = []
        for route, target in zip(routes.itertuples(index=False), target_matrix):
            null_path = out / "null_maps" / (
                f"{subject}__seizure{int(route.seizure_idx)}.npz"
            )
            with np.load(null_path, allow_pickle=False) as null_map:
                if null_map["contacts"].astype(str).tolist() != contacts:
                    failures.append(
                        f"PRIMARY_NULL_CONTACT_ALIGNMENT:{subject}:{route.seizure_idx}"
                    )
                    continue
                permutations = np.asarray(null_map["all_contact"], dtype=np.intp)
            ra, null_a = _signed_spearman_with_frozen_permutations(
                mode_a, target, permutations
            )
            rb, null_b = _signed_spearman_with_frozen_permutations(
                mode_b, target, permutations
            )
            observed_by_seizure.append(float(np.nanmax([ra, rb])))
            null_by_seizure.append(np.fmax(null_a, null_b))
        if len(observed_by_seizure) != len(routes):
            failures.append(f"PRIMARY_NULL_SEIZURE_COUNT:{subject}")
            continue
        folded_null = np.nanmedian(np.stack(null_by_seizure), axis=0)
        observed = float(np.nanmedian(observed_by_seizure))
        null_median = float(np.nanmedian(folded_null))
        finite_null = folded_null[np.isfinite(folded_null)]
        p_value = (
            float((1 + np.sum(finite_null >= observed - 1e-7)) / (1 + len(finite_null)))
            if np.isfinite(observed) and len(finite_null) else float("nan")
        )
        reported = patient[
            (patient.subject.astype(str) == str(subject))
            & (patient.condition == condition)
            & (patient.endpoint == endpoint)
        ]
        if len(reported) != 1:
            failures.append(f"PRIMARY_PATIENT_ROW:{subject}:{len(reported)}")
            continue
        row = reported.iloc[0]
        comparisons = {
            "observed": observed,
            "all_contact_null_median": null_median,
            "all_contact_margin": observed - null_median,
            "all_contact_p": p_value,
            "all_contact_null_finite_draws": int(len(finite_null)),
        }
        maximum_error = 0.0
        for key, expected in comparisons.items():
            error = abs(float(row[key]) - expected)
            maximum_error = max(maximum_error, error)
            tolerance = 2e-7 if key != "all_contact_p" else 1e-12
            if error > tolerance:
                failures.append(f"PRIMARY_NULL_VALUE:{subject}:{key}:{error:.6g}")
        records.append({
            "subject": str(subject),
            "n_seizures": int(len(routes)),
            "n_contacts": int(len(contacts)),
            "permutations": int(len(folded_null)),
            "maximum_absolute_error": maximum_error,
        })
    return {
        "pass": not failures and len(records) == 17,
        "contract": (
            "MAXAB_REPEATED_WITHIN_EACH_PERMUTATION_THEN_"
            "SEIZURE_MEDIAN_FOLDED_WITHIN_PATIENT"
        ),
        "patients": len(records),
        "seizures": int(sum(record["n_seizures"] for record in records)),
        "permutations_per_seizure": sorted(
            {record["permutations"] for record in records}
        ),
        "selected_mode_metadata_mismatches": mode_metadata_mismatches,
        "maximum_absolute_error": float(
            max((record["maximum_absolute_error"] for record in records), default=0.0)
        ),
        "failures": failures,
    }


def assess_train_prevalence_mixture_algebra(
    out: Path,
    census: pd.DataFrame,
) -> dict:
    """Recompute every intact non-oracle mixture from frozen A/B components."""
    failures: list[str] = []
    checked = 0
    repaired_component_rows = 0
    arms = (
        "L0_LOCAL_ONLY",
        "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
        "L2M_MACRO_MATCHED_RANDOM_LR",
        "L3_LOCAL_PLUS_LEARNED_LR",
        "C_L3_ORDER_SHUFFLED",
    )
    for subject in census.subject.astype(str).drop_duplicates():
        for arm in arms:
            path = out / "model_fields/intact/per_patient" / subject / f"{arm}.npz"
            component_path = (
                out / "model_fields/train_mode_mixture_components/per_patient"
                / subject / f"{arm}.npz"
            )
            with np.load(path, allow_pickle=False) as payload:
                p_a = float(np.asarray(payload["train_prevalence_A"]).item())
                p_b = float(np.asarray(payload["train_prevalence_B"]).item())
                if component_path.exists():
                    repaired_component_rows += 1
                    with np.load(component_path, allow_pickle=False) as components:
                        c_p_a = float(np.asarray(components["train_prevalence_A"]).item())
                        c_p_b = float(np.asarray(components["train_prevalence_B"]).item())
                        if not np.isclose([p_a, p_b], [c_p_a, c_p_b], atol=1e-7).all():
                            failures.append(f"COMPONENT_PREVALENCE:{subject}:{arm}")
                        source = {
                            endpoint: (
                                np.asarray(components[f"A_{endpoint}"], float),
                                np.asarray(components[f"B_{endpoint}"], float),
                            )
                            for endpoint in ("canonical_full", "seed_removed")
                        }
                else:
                    source = {
                        endpoint: (
                            np.asarray(payload[f"A_{endpoint}"], float),
                            np.asarray(payload[f"B_{endpoint}"], float),
                        )
                        for endpoint in ("canonical_full", "seed_removed")
                    }
                if not np.isclose(p_a + p_b, 1.0, rtol=0, atol=1e-6):
                    failures.append(f"PREVALENCE_SUM:{subject}:{arm}:{p_a+p_b:.8g}")
                for endpoint, (left, right) in source.items():
                    if endpoint == "seed_removed":
                        left = np.nan_to_num(left, nan=0.0)
                        right = np.nan_to_num(right, nan=0.0)
                    expected = p_a * left + p_b * right
                    observed = np.asarray(
                        payload[f"{endpoint}_train_prevalence_mixture"], float
                    )
                    if not np.allclose(observed, expected, rtol=0, atol=2e-7):
                        failures.append(f"MIXTURE_VALUE:{subject}:{arm}:{endpoint}")
                    checked += 1
    return {
        "pass": not failures and checked == 28 * 5 * 2,
        "patient_arm_endpoint_vectors": checked,
        "expected_vectors": 28 * 5 * 2,
        "noncollinear_repaired_patient_arm_components": repaired_component_rows,
        "expected_noncollinear_repaired_components": 14 * 5,
        "contract": "TRAIN_ONLY_PREVALENCE_TIMES_A_PLUS_B_WITHOUT_TARGET_ORACLE",
        "failures": failures,
    }


def assess_primary_early_interaction(
    out: Path,
    routing: pd.DataFrame,
    patient: pd.DataFrame,
) -> dict:
    """Independently rebuild both arms, observed interaction and spatial null."""
    arm_conditions = {
        "L3_LOCAL_PLUS_LEARNED_LR": "INTACT|L3_LOCAL_PLUS_LEARNED_LR",
        "L2M_MACRO_MATCHED_RANDOM_LR": "INTACT|L2M_MACRO_MATCHED_RANDOM_LR",
    }
    reconstructed_observed: dict[str, dict[str, float]] = {
        arm: {} for arm in arm_conditions
    }
    reconstructed_null: dict[str, dict[str, np.ndarray]] = {
        arm: {} for arm in arm_conditions
    }
    failures: list[str] = []
    for subject, routes in routing.groupby("subject", sort=False):
        subject = str(subject)
        target_path = out / "early_ictal/per_patient_targets" / f"{subject}.npz"
        with np.load(target_path, allow_pickle=False) as targets:
            target_contacts = targets["contacts"].astype(str).tolist()
            target_matrix = np.asarray(targets["all_seizure_broadband_energy"], float)
        for arm in arm_conditions:
            field_path = out / "model_fields/intact/per_patient" / subject / f"{arm}.npz"
            with np.load(field_path, allow_pickle=False) as fields:
                contacts = fields["contacts"].astype(str).tolist()
                mode_a = np.asarray(fields["A_canonical_full"], float)
                mode_b = np.asarray(fields["B_canonical_full"], float)
            if contacts != target_contacts or len(routes) != len(target_matrix):
                failures.append(f"INPUT_ALIGNMENT:{subject}:{arm}")
                continue
            observed_by_seizure: list[float] = []
            null_by_seizure: list[np.ndarray] = []
            for route, target in zip(routes.itertuples(index=False), target_matrix):
                null_path = out / "null_maps" / (
                    f"{subject}__seizure{int(route.seizure_idx)}.npz"
                )
                with np.load(null_path, allow_pickle=False) as null_map:
                    if null_map["contacts"].astype(str).tolist() != contacts:
                        failures.append(
                            f"NULL_ALIGNMENT:{subject}:{arm}:{route.seizure_idx}"
                        )
                        continue
                    permutations = np.asarray(null_map["all_contact"], dtype=np.intp)
                ra, null_a = _signed_spearman_with_frozen_permutations(
                    mode_a, target, permutations
                )
                rb, null_b = _signed_spearman_with_frozen_permutations(
                    mode_b, target, permutations
                )
                observed_by_seizure.append(float(np.nanmax([ra, rb])))
                null_by_seizure.append(np.fmax(null_a, null_b))
            if len(observed_by_seizure) != len(routes):
                failures.append(f"SEIZURE_COUNT:{subject}:{arm}")
                continue
            reconstructed_observed[arm][subject] = float(
                np.nanmedian(observed_by_seizure)
            )
            reconstructed_null[arm][subject] = np.nanmedian(
                np.stack(null_by_seizure), axis=0
            )

    l3_arm = "L3_LOCAL_PLUS_LEARNED_LR"
    l2m_arm = "L2M_MACRO_MATCHED_RANDOM_LR"
    l3 = arm_conditions[l3_arm]
    l2m = arm_conditions[l2m_arm]
    table = patient[patient.endpoint == "canonical_full"].pivot(
        index="subject", columns="condition", values="observed"
    )
    delta = table[l3] - table[l2m]
    j_table = pd.read_csv(out / "CROSSFIT_NONLOCALITY_PATIENT_SUMMARY.csv").set_index(
        "subject"
    )
    subjects = delta.index.intersection(j_table.index).astype(str)
    independent_delta = pd.Series({
        subject: reconstructed_observed[l3_arm][subject]
        - reconstructed_observed[l2m_arm][subject]
        for subject in subjects
    })
    delta_error = float(np.max(np.abs(
        independent_delta.loc[subjects].to_numpy(float)
        - delta.loc[subjects].to_numpy(float)
    )))
    observed_rho = float(spearmanr(
        j_table.loc[subjects, "J_lat_exceedance_burden"].to_numpy(float),
        independent_delta.loc[subjects].to_numpy(float),
    ).statistic)
    spatial_reported = pd.read_csv(
        out / "early_ictal/PRIMARY_SYNCHRONIZED_SPATIAL_NULL_INTERACTION.csv"
    )["rho_J_by_L3_minus_L2m_null"].to_numpy(float)
    independent_null_delta = np.stack([
        reconstructed_null[l3_arm][subject]
        - reconstructed_null[l2m_arm][subject]
        for subject in subjects
    ])
    spatial = np.asarray([
        spearmanr(
            j_table.loc[subjects, "J_lat_exceedance_burden"].to_numpy(float),
            independent_null_delta[:, draw],
        ).statistic
        for draw in range(independent_null_delta.shape[1])
    ])
    spatial_array_error = float(np.nanmax(np.abs(spatial - spatial_reported)))
    finite = spatial[np.isfinite(spatial)]
    spatial_p = float(
        (1 + np.sum(finite >= observed_rho - 1e-7)) / (1 + len(finite))
    )
    summary = load_json(out / "early_ictal/EARLY_ICTAL_V0_5_SUMMARY.json")[
        "primary_interaction"
    ]
    rng = np.random.default_rng(20260813)
    patient_label_null = np.asarray([
        spearmanr(
            j_table.loc[subjects, "J_lat_exceedance_burden"].to_numpy(float),
            rng.permutation(delta.loc[subjects].to_numpy(float)),
        ).statistic
        for _ in range(100_000)
    ])
    patient_label_p = float(
        (1 + np.sum(patient_label_null >= observed_rho)) / 100_001
    )
    joint = max(patient_label_p, spatial_p)
    errors = {
        "rho": abs(observed_rho - float(summary["spearman_rho"])),
        "patient_label_p": abs(
            patient_label_p - float(summary["permutation_p_greater"])
        ),
        "spatial_p": abs(
            spatial_p - float(summary["spatial_null"]["spatial_null_p_greater"])
        ),
        "joint_p": abs(joint - float(summary["joint_primary_p_greater"])),
    }
    return {
        "pass": bool(
            len(subjects) == 17
            and len(finite) == 5000
            and max(errors.values()) <= 2e-7
            and delta_error <= 2e-7
            and spatial_array_error <= 2e-7
            and not failures
            and summary.get("joint_primary_contract")
            == "BOTH_PATIENT_LABEL_AND_SYNCHRONIZED_SPATIAL_NULL_MUST_PASS"
        ),
        "patients": len(subjects),
        "observed_rho": observed_rho,
        "patient_label_p_greater": patient_label_p,
        "spatial_null_p_greater": spatial_p,
        "joint_primary_p_greater": joint,
        "finite_spatial_null_draws": len(finite),
        "maximum_absolute_error": max(errors.values()),
        "independent_patient_delta_maximum_absolute_error": delta_error,
        "independent_spatial_null_array_maximum_absolute_error": spatial_array_error,
        "failures": failures,
        "contract": "JOINT_MAX_OF_PATIENT_LABEL_AND_COHERENT_SPATIAL_NULL_P",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE)
    args = parser.parse_args()
    out, figure = args.out_root.resolve(), args.figure_dir.resolve()
    checks: dict[str, dict] = {}

    required_markers = (
        "FEASIBILITY_AUDIT_COMPLETE.json", "STAGE_A_COMPLETE.json",
        "STAGE_B_COMPLETE.json", "STAGE_C_GRAPH_CONTROL_COMPLETE.json",
        "STAGE_D_J_COMPLETE.json", "STAGE_E_TRAINING_COMPLETE.json",
        "STAGE_E_INTERICTAL_ANALYSIS_COMPLETE.json", "STAGE_F_TARGET_FREE_COMPLETE.json",
        "TRAIN_PREVALENCE_MIXTURE_REPAIR_COMPLETE.json",
        "PREUNSEAL_RESUME_GUARD_COMPLETE.json",
        "TARGET_UNSEAL_AUTHORIZATION.json", "EARLY_ICTAL_SCORING_COMPLETE.json",
        "PIPELINE_COMPLETE.json", "FIGURE6_FINAL_RENDER_COMPLETE.json",
        "FINAL_CLAIM_ADJUDICATION.json",
        "SCORER_CONTRACT_PREFREEZE_REPAIR.json",
        "CLOSEOUT_TOOLING_PREFREEZE_MANIFEST.json",
    )
    missing = [name for name in required_markers if not (out / name).exists()]
    active_failures = [name for name in (
        "STAGE_E_TRAINING_FAILED.json", "STAGE_F_TARGET_FREE_FAILED.json",
        "POSTTRAINING_PIPELINE_FAILED.json",
    ) if (out / name).exists()]
    checks["A_H_markers"] = {
        "pass": not missing and not active_failures,
        "missing": missing, "active_failure_markers": active_failures,
    }

    census = pd.read_csv(out / "FULL_PARENT_FIT_CENSUS.csv")
    routing = pd.read_csv(out / "EARLY_ICTAL_ROUTING_METADATA.csv")
    checks["cohort_contract"] = {
        "pass": bool(
            census.subject.nunique() == 28 and len(census) == 42
            and census.valid.astype(bool).all()
            and int(census.n_joint_contacts.min()) >= 6
            and census.strongly_connected.astype(bool).all()
            and float(census.contact_supported_pairwise_reachability.min()) == 1.0
            and routing.subject.nunique() == 17 and len(routing) == 167
        ),
        "spatial_patients": int(census.subject.nunique()), "fits": len(census),
        "contact_range": [int(census.n_joint_contacts.min()), int(census.n_joint_contacts.max())],
        "early_patients": int(routing.subject.nunique()), "seizures": len(routing),
    }
    checks["full_tissue_latent_domain_contract"] = {
        "pass": bool(
            census.n_nodes.gt(census.n_joint_contacts).all()
            and census.n_zero_h_nodes.gt(0).all()
            and census.zero_h_fraction.gt(0.0).all()
            and census.strongly_connected.astype(bool).all()
            and census.contact_supported_pairwise_reachability.eq(1.0).all()
        ),
        "fits": len(census),
        "node_range": [int(census.n_nodes.min()), int(census.n_nodes.max())],
        "zero_H_node_range": [
            int(census.n_zero_h_nodes.min()), int(census.n_zero_h_nodes.max())
        ],
        "zero_H_fraction_range": [
            float(census.zero_h_fraction.min()),
            float(census.zero_h_fraction.max()),
        ],
        "fits_with_more_nodes_than_contacts": int(
            census.n_nodes.gt(census.n_joint_contacts).sum()
        ),
        "claim_boundary": (
            "CONTACTS_ARE_FROZEN_H_INJECTION_READOUTS;_"
            "LATENT_COMPUTATION_COVERS_UNOBSERVED_TISSUE_NODES"
        ),
    }
    checks["geometry_view_event_scope"] = assess_geometry_view_event_scope(out, census)

    mixture_repair = load_json(out / "TRAIN_PREVALENCE_MIXTURE_REPAIR_COMPLETE.json")
    mixture_rows = pd.read_csv(out / "TRAIN_PREVALENCE_MIXTURE_REPAIR.csv")
    preunseal_guard_for_repair = load_json(out / "PREUNSEAL_RESUME_GUARD_COMPLETE.json")
    frozen_repair = preunseal_guard_for_repair.get("evidence", {}).get(
        "train_prevalence_mixture_repair", {}
    )
    mixture_failures = []
    for row in mixture_rows.itertuples():
        component = Path(row.component_path)
        patient_field = Path(row.patient_field_path)
        if not component.exists() or sha256_file(component) != row.component_sha256:
            mixture_failures.append(f"COMPONENT_HASH:{row.subject}:{row.arm}")
        if not patient_field.exists() or sha256_file(patient_field) != row.patient_field_sha256:
            mixture_failures.append(f"PATIENT_FIELD_HASH:{row.subject}:{row.arm}")
    checks["nonoracle_train_mixture_geometry_mode_separation"] = {
        "pass": bool(
            mixture_repair.get("status") == "PASS_TARGET_FREE"
            and mixture_repair.get("target_values_read") is False
            and mixture_repair.get("oracle_ab_vectors_changed") is False
            and mixture_repair.get("changed_subjects") == 14
            and mixture_repair.get("changed_patient_arm_fields") == 70
            and len(mixture_rows) == 70
            and mixture_rows.subject.nunique() == 14
            and mixture_rows.arm.nunique() == 5
            and mixture_rows.oracle_vectors_unchanged.astype(bool).all()
            and mixture_rows.target_values_read.eq(False).all()
            and frozen_repair.get("sha256") == sha256_file(
                out / "TRAIN_PREVALENCE_MIXTURE_REPAIR_COMPLETE.json"
            )
            and mixture_repair.get("repair_table_sha256") == sha256_file(
                out / "TRAIN_PREVALENCE_MIXTURE_REPAIR.csv"
            )
            and mixture_repair.get("model_field_manifest_sha256") == sha256_file(
                out / "MODEL_FIELD_MANIFEST.csv"
            )
            and Path(str(mixture_repair.get("producer_script", ""))).exists()
            and sha256_file(Path(str(mixture_repair.get("producer_script", ""))))
            == mixture_repair.get("producer_script_sha256")
            and not mixture_failures
        ),
        "subjects": int(mixture_rows.subject.nunique()),
        "patient_arm_fields": int(len(mixture_rows)),
        "oracle_ab_vectors_changed": mixture_repair.get("oracle_ab_vectors_changed"),
        "contract": (
            "ORACLE_AB_GEOMETRY_CANDIDATES_UNCHANGED;_NONORACLE_MIXTURE_USES_"
            "TRAIN_MODE_SPECIFIC_COMPONENTS"
        ),
        "failures": mixture_failures,
    }

    hotfill_complete_path = out / "ATTENUATION_HOTFILL_COMPLETE.json"
    if hotfill_complete_path.exists():
        hotfill = load_json(hotfill_complete_path)
        parity_path = out / "ATTENUATION_HOTFILL_EXACT_PARITY.json"
        parity = load_json(parity_path)
        producer = Path(str(hotfill.get("producer_script", "")))
        frozen_hotfill = preunseal_guard_for_repair.get("evidence", {}).get(
            "attenuation_hotfill", {}
        )
        hotfill_failures = []
        hotfilled_cache = []
        for path in (out / "attenuation/unit_cache").glob("**/*.json.gz"):
            try:
                import gzip
                with gzip.open(path, "rt", encoding="utf-8") as stream:
                    payload = json.load(stream)
            except (OSError, EOFError, json.JSONDecodeError) as error:
                hotfill_failures.append(f"UNREADABLE_CACHE:{path}:{error!r}")
                continue
            if "rollout_dedup_contract" in payload:
                hotfilled_cache.append(path)
                if (
                    payload.get("rollout_dedup_contract")
                    != "DETERMINISTIC_SAME_MODEL_SAME_FIRST_RANK_EXACT_EXPANSION"
                    or payload.get("rollout_dedup_producer_sha256")
                    != hotfill.get("producer_script_sha256")
                    or payload.get("target_values_read") is not False
                ):
                    hotfill_failures.append(f"INVALID_DEDUP_CACHE:{path}")
        checks["attenuation_deduplicated_rollout_hotfill"] = {
            "pass": bool(
                hotfill.get("status") == "PASS_TARGET_FREE"
                and hotfill.get("target_values_read") is False
                and parity.get("status") == "PASS_TARGET_FREE"
                and parity.get("target_values_read") is False
                and parity.get("events") == 1492
                and parity.get("mismatches") == 0
                and producer.exists()
                and sha256_file(producer) == hotfill.get("producer_script_sha256")
                and hotfill.get("exact_parity_sha256") == sha256_file(parity_path)
                and frozen_hotfill.get("status") == "PASS_TARGET_FREE"
                and frozen_hotfill.get("complete", {}).get("sha256")
                == sha256_file(hotfill_complete_path)
                and frozen_hotfill.get("exact_parity", {}).get("sha256")
                == sha256_file(parity_path)
                and frozen_hotfill.get("annotated_unit_targets") == len(hotfilled_cache)
                and len(hotfilled_cache) == int(hotfill.get("hotfilled", -1))
                and not hotfill_failures
            ),
            "hotfilled_unit_targets": len(hotfilled_cache),
            "parity_events": parity.get("events"),
            "parity_mismatches": parity.get("mismatches"),
            "scientific_contract": "EXACT_SAME_START_EXPANSION_NO_UNIT_OR_DRAW_REMOVAL",
            "failures": hotfill_failures,
        }
    else:
        checks["attenuation_deduplicated_rollout_hotfill"] = {
            "pass": not (out / "ATTENUATION_HOTFILL_ACTIVE.json").exists(),
            "status": "NOT_USED" if not (out / "ATTENUATION_HOTFILL_ACTIVE.json").exists() else "INCOMPLETE",
        }

    parent = load_json(out / "PARENT_V0_3_PROVENANCE.json")
    parent_audit = ROOT / str(parent.get("parent_closeout_audit", ""))
    local_tag_object = subprocess.run(
        ["git", "rev-parse", PARENT_V0_3_TAG], cwd=ROOT,
        capture_output=True, text=True, check=False,
    )
    local_tag_commit = subprocess.run(
        ["git", "rev-parse", f"{PARENT_V0_3_TAG}^{{}}"], cwd=ROOT,
        capture_output=True, text=True, check=False,
    )
    parent_audit_payload = load_json(parent_audit) if parent_audit.exists() else {}
    checks["immutable_parent_v0_3_closeout"] = {
        "pass": bool(
            parent.get("parent_commit") == PARENT_V0_3_COMMIT
            and parent.get("annotated_tag") == PARENT_V0_3_TAG
            and parent.get("remote_tag_verified") is True
            and parent.get("remote_tag_dereferenced_commit") == PARENT_V0_3_COMMIT
            and local_tag_object.returncode == 0
            and local_tag_object.stdout.strip() == parent.get("annotated_tag_object")
            and local_tag_commit.returncode == 0
            and local_tag_commit.stdout.strip() == PARENT_V0_3_COMMIT
            and parent_audit.exists()
            and sha256_file(parent_audit) == parent.get("parent_closeout_audit_sha256")
            and parent_audit_payload.get("status") == "PASS"
            and parent_audit_payload.get("errors") == []
            and parent.get("target_values_read") is False
        ),
        "parent_commit": parent.get("parent_commit"),
        "tag": parent.get("annotated_tag"),
        "remote_tag_verified": parent.get("remote_tag_verified"),
        "parent_closeout_status": parent_audit_payload.get("status"),
        "parent_closeout_sha256": (
            sha256_file(parent_audit) if parent_audit.exists() else None
        ),
    }

    schedule = pd.read_csv(out / "FORMAL_TRAINING_SCHEDULE.csv")
    expected_arms = {"L0": 93, "L1": 93, "L2m": 126, "L3": 93, "C-suffix": 126}
    observed_arms = schedule.arm.value_counts().to_dict()
    metric_paths = sorted((out / "formal_units").glob("*/*/seed*/metrics.json"))
    metrics = [load_json(path) for path in metric_paths]
    checks["formal_training"] = {
        "pass": bool(
            len(schedule) == 531 and observed_arms == expected_arms and len(metrics) == 531
            and all(row.get("converged") is True for row in metrics)
            and all(row.get("hit_ceiling") is False for row in metrics)
            and all(row.get("best_checkpoint_eligible") is True for row in metrics)
            and all(int(row["best_epoch"]) >= int(row["mask_freeze_epoch"]) for row in metrics)
            and all(row.get("target_values_read") is False for row in metrics)
        ),
        "scheduled_units": len(schedule), "metric_units": len(metrics),
        "arms": observed_arms, "converged": sum(row.get("converged") is True for row in metrics),
        "hit_ceiling": sum(row.get("hit_ceiling") is True for row in metrics),
    }
    l2m_metrics = [
        row for row in metrics if row.get("arm") == "L2M_MACRO_MATCHED_RANDOM_LR"
    ]
    checks["L2m_independent_refit_and_initial_weight_match"] = {
        "pass": bool(
            len(l2m_metrics) == 126
            and all((row.get("initial_weight_match") or {}).get("exact_multiset") is True
                    for row in l2m_metrics)
            and all((row.get("initial_weight_match") or {}).get("n_values", 0) > 0
                    for row in l2m_metrics)
            and all(row.get("best_checkpoint_eligible") is True for row in l2m_metrics)
            and all(int(row["best_epoch"]) >= int(row["mask_freeze_epoch"])
                    for row in l2m_metrics)
        ),
        "units": len(l2m_metrics),
        "exact_initial_weight_multiset_units": sum(
            (row.get("initial_weight_match") or {}).get("exact_multiset") is True
            for row in l2m_metrics
        ),
        "contract": (
            "MACRO_MATCHED_RANDOM_NONLOCAL_MASK;_"
            "EXACT_L3_INITIAL_ADDED_WEIGHT_MULTISET;_INDEPENDENT_REFIT"
        ),
    }
    expected_decoder = {
        "fit_support": "train teacher-forced continue decisions only",
        "selection_support": "interictal validation only",
        "stop_precedence": "stop_probability>=0.5 before contact selection",
        "cardinality": "argmax(size_head)+1; never observed future set size",
        "repeat_mask": True,
        "tie_break": "lowest contact index",
        "maximum_ranks": "n_contacts",
    }
    decoder_contract_matches = all(
        ROLLOUT_DECODER_CONTRACT.get(key) == value
        for key, value in expected_decoder.items()
    )
    checks["future_free_rollout_decoder"] = {
        "pass": bool(
            decoder_contract_matches
            and len(metrics) == 531
            and all((row.get("rollout_decoder") or {}).get("n_train_decisions", 0) > 0
                    for row in metrics)
            and all((row.get("rollout_decoder") or {}).get("n_validation_decisions", 0) > 0
                    for row in metrics)
        ),
        "units": len(metrics),
        "decoder_contract": ROLLOUT_DECODER_CONTRACT,
        "minimum_train_decisions": min(
            int(row["rollout_decoder"]["n_train_decisions"]) for row in metrics
        ),
        "minimum_validation_decisions": min(
            int(row["rollout_decoder"]["n_validation_decisions"]) for row in metrics
        ),
        "claim_boundary": "FREE_ROLLOUT_NEVER_READS_OBSERVED_FUTURE_SET_SIZE",
    }

    input_manifest_path = out / "INPUT_CACHE_MANIFEST.json"
    input_manifest = load_json(input_manifest_path)
    input_failures = []
    cache_records = input_manifest.get("cache_records", [])
    for record in cache_records:
        fit_id = str(record.get("fit_id"))
        for relative, evidence in record.get("files", {}).items():
            path = out / "cache" / fit_id / relative
            if not path.exists():
                input_failures.append(f"MISSING:{fit_id}:{relative}")
            elif sha256_file(path) != evidence.get("sha256"):
                input_failures.append(f"HASH_MISMATCH:{fit_id}:{relative}")
    metric_provenance_failures = []
    expected_input_hash = sha256_file(input_manifest_path)
    expected_run_hash = sha256_file(out / "RUN_CONTRACT.json")
    for row in metrics:
        producer = row.get("producer_hashes", {})
        if producer.get("input_manifest") != expected_input_hash:
            metric_provenance_failures.append(
                f"INPUT_MANIFEST:{row.get('fit_id')}:{row.get('arm')}:{row.get('seed')}"
            )
        if producer.get("run_contract") != expected_run_hash:
            metric_provenance_failures.append(
                f"RUN_CONTRACT:{row.get('fit_id')}:{row.get('arm')}:{row.get('seed')}"
            )
    checks["input_and_metric_provenance_freshness"] = {
        "pass": bool(
            input_manifest.get("target_values_read") is False
            and input_manifest.get("fits") == 42
            and len(cache_records) == 42
            and not input_failures
            and not metric_provenance_failures
        ),
        "fits": len(cache_records),
        "required_files_per_fit": input_manifest.get("required_files_per_fit"),
        "input_failures": input_failures,
        "metric_provenance_failures": metric_provenance_failures,
    }
    checks["formal_suffix_control_lineage"] = assess_suffix_control_lineage(
        out, input_manifest, metrics
    )
    checks["masked_rank_lineage_no_phantom_reentry"] = assess_masked_rank_lineage(
        MASKED_RANK_DATASET, out / "cache"
    )

    per_seed = pd.read_csv(out / "INTERICTAL_PER_FIT_SEED.csv")
    per_fit = pd.read_csv(out / "INTERICTAL_PER_FIT.csv")
    per_patient = pd.read_csv(out / "INTERICTAL_PER_PATIENT.csv")
    fit_keys = ["subject", "fit_id", "scope", "arm"]
    expected_fit = per_seed.groupby(fit_keys, as_index=False).median(numeric_only=True)
    fit_aggregation = compare_numeric_tables(per_fit, expected_fit, fit_keys)
    patient_keys = ["subject", "arm"]
    expected_patient = per_fit.groupby(patient_keys, as_index=False).mean(numeric_only=True)
    patient_aggregation = compare_numeric_tables(
        per_patient, expected_patient, patient_keys, atol=2e-12,
    )
    checks["interictal_patient_first_aggregation"] = {
        "pass": fit_aggregation["pass"] and patient_aggregation["pass"],
        "seed_to_fit_contract": "MEDIAN",
        "fit_to_patient_contract": "MEAN_ACROSS_OWN_A_OWN_B_FITS",
        "seed_to_fit": fit_aggregation,
        "fit_to_patient": patient_aggregation,
    }

    run_contract = load_json(out / "RUN_CONTRACT.json")
    stage_f_snapshot = load_json(out / "STAGE_F_RUN_SNAPSHOT.json")
    posttraining_snapshot = load_json(out / "POSTTRAINING_PIPELINE_SNAPSHOT.json")
    engineering_amendment = load_json(out / "TARGET_UNSEAL_ENGINEERING_AMENDMENT.json")
    amendment_base_valid = bool(
        engineering_amendment.get("status")
        == "POST_UNSEAL_TARGET_INDEPENDENT_INVENTORY_REPAIR"
        and engineering_amendment.get("original_authorization_sha256")
        == sha256_file(out / "TARGET_UNSEAL_AUTHORIZATION.json")
        and engineering_amendment.get("model_or_field_generation_after_unseal") is False
        and engineering_amendment.get("primary_estimand_changed") is False
    )
    posttraining_paths = {
        "driver": ROOT / "scripts/run_topic5_multiscale_posttraining_v0_5.py",
        "embargo": ROOT / "scripts/run_topic5_v0_5_target_free.py",
        "interictal": ROOT / "scripts/analyse_topic5_multiscale_interictal_v0_5.py",
        "stage_f": ROOT / "scripts/run_topic5_multiscale_stage_f_v0_5.py",
        "authorize": ROOT / "scripts/prepare_topic5_multiscale_target_unseal_v0_5.py",
        "score": ROOT / "scripts/score_topic5_multiscale_early_ictal_v0_5.py",
        "figure": ROOT / "scripts/paper_figures/plot_topic5_figure6_multiscale_scaffold_v0_5.py",
    }
    posttraining_check = verify_source_snapshot(posttraining_snapshot, posttraining_paths)
    if (
        amendment_base_valid
        and posttraining_check["failures"] == ["HASH_MISMATCH:score"]
        and engineering_amendment.get("old_scorer_sha256")
        == posttraining_snapshot.get("source_hashes", {}).get("score")
        and engineering_amendment.get("new_scorer_sha256")
        == sha256_file(posttraining_paths["score"])
    ):
        posttraining_check.update({
            "pass": True,
            "amended_failures": posttraining_check["failures"],
            "failures": [],
            "amendment": "TARGET_INDEPENDENT_CONDITION_INVENTORY_ONLY",
        })
    snapshot_checks = {
        "formal_execution": verify_source_snapshot(run_contract),
        "stage_f": verify_source_snapshot(stage_f_snapshot),
        "posttraining": posttraining_check,
    }
    checks["hash_verified_execution_snapshots"] = {
        "pass": all(value["pass"] for value in snapshot_checks.values()),
        "protection_level": "HASH_VERIFIED_NOT_FILESYSTEM_READ_ONLY",
        "details": snapshot_checks,
    }
    scorer_repair = load_json(out / "SCORER_CONTRACT_PREFREEZE_REPAIR.json")
    checks["scorer_contract_prefreeze_repair"] = {
        "pass": bool(
            scorer_repair.get("status") == "PASS_TARGET_FREE"
            and scorer_repair.get("target_values_read") is False
            and scorer_repair.get("target_authorization_absent") is True
            and scorer_repair.get("snapshot_sha256")
            == sha256_file(out / "POSTTRAINING_PIPELINE_SNAPSHOT.json")
            and scorer_repair.get("scorer_sha256")
            == engineering_amendment.get("old_scorer_sha256")
            and engineering_amendment.get("new_scorer_sha256")
            == sha256_file(ROOT / "scripts/score_topic5_multiscale_early_ictal_v0_5.py")
            and amendment_base_valid
            and scorer_repair.get("authorizer_sha256")
            == sha256_file(ROOT / "scripts/prepare_topic5_multiscale_target_unseal_v0_5.py")
        ),
        "reason": posttraining_snapshot.get("prefreeze_repair", {}).get("reason"),
        "target_values_read": scorer_repair.get("target_values_read"),
    }

    closeout_tooling_path = out / "CLOSEOUT_TOOLING_PREFREEZE_MANIFEST.json"
    closeout_tooling = load_json(closeout_tooling_path)
    closeout_authorization = load_json(out / "TARGET_UNSEAL_AUTHORIZATION.json")
    closeout_tooling_failures = []
    amended_closeout_sources = []
    for relative, expected in closeout_tooling.get("sources", {}).items():
        path = ROOT / relative
        if not path.is_file():
            closeout_tooling_failures.append(relative)
            continue
        current = sha256_file(path)
        if current == str(expected):
            continue
        if (
            relative == "scripts/audit_topic5_multiscale_closeout_v0_5.py"
            and amendment_base_valid
            and engineering_amendment.get("old_audit_sha256") == str(expected)
            and engineering_amendment.get("new_audit_sha256") == current
        ):
            amended_closeout_sources.append(relative)
            continue
        closeout_tooling_failures.append(relative)
    checks["closeout_tooling_prefreeze"] = {
        "pass": bool(
            closeout_tooling.get("status") == "PASS_TARGET_FREE"
            and closeout_tooling.get("target_values_read") is False
            and closeout_tooling.get("source_count") == 8
            and len(closeout_tooling.get("sources", {})) == 8
            and not closeout_tooling_failures
            and iso_timestamp(closeout_authorization) >= iso_timestamp(closeout_tooling)
        ),
        "sources": len(closeout_tooling.get("sources", {})),
        "hash_failures": closeout_tooling_failures,
        "postunseal_target_independent_amendments": amended_closeout_sources,
        "authorization_after_prefreeze": bool(
            iso_timestamp(closeout_authorization) >= iso_timestamp(closeout_tooling)
        ),
    }
    checks["postunseal_target_independent_inventory_repair"] = {
        "pass": bool(
            amendment_base_valid
            and engineering_amendment.get("old_scorer_sha256")
            == closeout_authorization.get("scorer_sha256")
            and engineering_amendment.get("new_scorer_sha256")
            == sha256_file(ROOT / "scripts/score_topic5_multiscale_early_ictal_v0_5.py")
            and engineering_amendment.get("new_audit_sha256")
            == sha256_file(Path(__file__).resolve())
            and engineering_amendment.get("recovery_driver_sha256")
            == sha256_file(
                ROOT / "scripts/recover_topic5_multiscale_postunseal_inventory_v0_5.py"
            )
            and (out / "POSTUNSEAL_INVENTORY_RECOVERY_SNAPSHOT.json").exists()
            and load_json(
                out / "POSTUNSEAL_INVENTORY_RECOVERY_SNAPSHOT.json"
            ).get("source_hashes", {}).get("driver")
            == engineering_amendment.get("recovery_driver_sha256")
        ),
        "target_values_had_been_read_before_repair": True,
        "scope": engineering_amendment.get("scope"),
        "primary_estimand_changed": engineering_amendment.get("primary_estimand_changed"),
        "model_or_field_generation_after_unseal": engineering_amendment.get(
            "model_or_field_generation_after_unseal"
        ),
    }

    authorization = load_json(out / "TARGET_UNSEAL_AUTHORIZATION.json")
    embargo = load_json(out / "TARGET_PHYSICAL_EMBARGO_ACTIVE.json")
    protected_roots = embargo.get("protected_roots", [])
    checks["physical_target_embargo_contract"] = {
        "pass": bool(
            embargo.get("target_values_read") is False
            and len(protected_roots) >= 8
            and all(row.get("status") == "BIND_HIDDEN_EMPTY" for row in protected_roots)
            and embargo.get("wrapper_sha256")
            == run_contract.get("source_hashes", {}).get(
                "scripts/run_topic5_v0_5_target_free.py"
            )
            and iso_timestamp(authorization) >= iso_timestamp(embargo)
        ),
        "protected_roots": len(protected_roots),
        "statuses": sorted({row.get("status") for row in protected_roots}),
        "authorization_after_embargo": bool(
            iso_timestamp(authorization) >= iso_timestamp(embargo)
        ),
        "target_values_read_during_embargo": embargo.get("target_values_read"),
    }

    resume_guard = load_json(out / "PREUNSEAL_RESUME_GUARD_COMPLETE.json")
    authorization = load_json(out / "TARGET_UNSEAL_AUTHORIZATION.json")
    checks["preunseal_resume_guard"] = {
        "pass": bool(
            resume_guard.get("status") == "PASS_TARGET_FREE"
            and resume_guard.get("target_values_read") is False
            and iso_timestamp(authorization) >= iso_timestamp(resume_guard)
            and len(resume_guard.get("evidence", {}).get("metric_files", [])) == 10
            and len(resume_guard.get("evidence", {}).get("empirical_files", [])) == 42
            and len(resume_guard.get("evidence", {}).get("posttraining_sources", [])) == 7
            and len(resume_guard.get("evidence", {}).get(
                "closeout_tooling_sources", []
            )) == 8
            and "train_prevalence_mixture_repair" in resume_guard.get("evidence", {})
            and resume_guard.get("evidence", {}).get(
                "attenuation_hotfill", {}
            ).get("status") == "PASS_TARGET_FREE"
        ),
        "authorization_after_guard": bool(
            iso_timestamp(authorization) >= iso_timestamp(resume_guard)
        ),
        "metric_files": len(resume_guard.get("evidence", {}).get("metric_files", [])),
        "empirical_rows": len(resume_guard.get("evidence", {}).get("empirical_files", [])),
        "posttraining_sources": len(
            resume_guard.get("evidence", {}).get("posttraining_sources", [])
        ),
        "closeout_tooling_sources": len(
            resume_guard.get("evidence", {}).get("closeout_tooling_sources", [])
        ),
    }

    execution = pd.concat([
        pd.read_csv(path) for path in sorted(out.glob("PHASE_*_EXECUTION.csv"))
    ], ignore_index=True)
    unresolved_oom = int(execution.get("unresolved_oom", pd.Series(False)).astype(bool).sum())
    unresolved_failed = int(
        (~execution.status.astype(str).eq("DONE") | execution.returncode.astype(int).ne(0)).sum()
    )
    checks["execution_failures_and_oom"] = {
        "pass": unresolved_oom == 0 and unresolved_failed == 0,
        "rows": len(execution), "unresolved_oom": unresolved_oom,
        "nonpass_rows": unresolved_failed,
    }
    history_path = out / "TRAINING_EXECUTION_HISTORY_AUDIT.json"
    history = load_json(history_path)
    history_table = Path(str(history.get("history_table", "")))
    history_producer = Path(str(history.get("producer_script", "")))
    phase_hashes = history.get("phase_table_hashes", {})
    checks["complete_training_attempt_history"] = {
        "pass": bool(
            history.get("status") == "PASS_WITH_RECORDED_PRELAUNCH_INCIDENT"
            and history.get("target_values_read") is False
            and history.get("formal_units") == 531
            and history.get("final_phase_done") == 531
            and history.get("final_phase_nonzero_returncode") == 0
            and history.get("final_phase_retry_attempt_gt_zero") == 0
            and history.get("unresolved_failed_units") == 0
            and history.get("cuda_oom_occurrences_all_launches") == 0
            and history.get("oom_occurrences_all_launches") == 0
            and history.get("units_with_prelaunch_n_contacts_keyerror") == 206
            and history.get("prelaunch_n_contacts_keyerrors") == 401
            and history.get("peak_vram_telemetry") == "NOT_RECORDED_PER_UNIT"
            and history_table.is_file()
            and sha256_file(history_table) == history.get("history_table_sha256")
            and history_producer.is_file()
            and sha256_file(history_producer) == history.get("producer_script_sha256")
            and all(
                (out / name).is_file()
                and sha256_file(out / name) == digest
                for name, digest in phase_hashes.items()
            )
        ),
        "final_frozen_execution": "531/531_DONE_ATTEMPT0_NO_OOM",
        "prelaunch_fail_closed_incident": {
            "affected_units": history.get("units_with_prelaunch_n_contacts_keyerror"),
            "tracebacks": history.get("prelaunch_n_contacts_keyerrors"),
            "cause": "LEGACY_N_CONTACTS_PROVENANCE_KEY",
        },
        "peak_vram_telemetry": history.get("peak_vram_telemetry"),
        "claim_boundary": (
            "DO_NOT_EQUATE_FINAL_ZERO_FAILURES_WITH_NO_EARLIER_FAIL_CLOSED_LAUNCH;_"
            "DO_NOT_CLAIM_PER_UNIT_PEAK_VRAM_TELEMETRY"
        ),
    }

    # The graph-control builder's immutable, row-level source of truth is the
    # manifest itself.  Do not depend on an unproduced summary alias here: the
    # manifest carries one row per fit/seed, the exact-match verdict, and the
    # hash of the graph payload that formal training consumed.
    graph = pd.read_csv(out / "L2M_GRAPH_CONTROL_MANIFEST.csv")
    checks["L2m_matching"] = {
        "pass": bool(len(graph) == 126 and graph.get("all_exact", pd.Series(False)).astype(bool).all()),
        "rows": len(graph), "all_exact": bool(graph.get("all_exact", pd.Series(False)).astype(bool).all()),
    }

    capacity = pd.read_csv(out / "CANDIDATE_EXPOSURE_POSTTRAIN_AUDIT.csv")
    strict_l3_l1_patients = capacity.groupby("subject").L3_minus_L1_mechanism_eligible.all()
    checks["candidate_capacity_interpretation_boundary"] = {
        "pass": bool(
            len(capacity) == 42
            and capacity.subject.nunique() == 28
            and int(capacity.opportunity_severe.astype(bool).sum()) == 15
            and int(strict_l3_l1_patients.sum()) == 16
        ),
        "fits": len(capacity),
        "severe_opportunity_imbalance_fits": int(
            capacity.opportunity_severe.astype(bool).sum()
        ),
        "strict_L3_minus_L1_mechanism_patients": int(strict_l3_l1_patients.sum()),
        "claim_boundary": (
            "L3_MINUS_L1_MECHANISM_ONLY_IN_PATIENTS_WITH_ALL_FITS_ELIGIBLE;_"
            "FULL_COHORT_CONTRAST_DESCRIPTIVE"
        ),
    }

    mechanism_scope = load_json(out / "MECHANISM_SCOPE_ADJUDICATION.json")
    scope_counts = census.scope.value_counts().to_dict()
    checks["mechanism_scope_interpretation"] = {
        "pass": bool(
            mechanism_scope.get("status") == "PASS_WITH_EXPLICIT_SCOPE_BOUNDARY"
            and mechanism_scope.get("target_values_read") is False
            and mechanism_scope.get("affected_primary_results") is False
            and mechanism_scope.get("shared_fit_patients") == 14
            and mechanism_scope.get("noncollinear_patients") == 14
            and scope_counts == {"shared": 14, "own_a": 14, "own_b": 14}
        ),
        "fit_scope_counts": scope_counts,
        "within_network_mode_flow_denominator": 14,
        "boundary": (
            "OWN_A_OWN_B_ARE_ALL_EVENT_GEOMETRY_VIEWS;_"
            "SAME_NETWORK_A_VS_B_ROUTE_SELECTIVITY_IS_SHARED_FIT_ONLY"
        ),
    }

    j = pd.read_csv(out / "CROSSFIT_NONLOCALITY_PATIENT_SUMMARY.csv")
    stage_d_j = load_json(out / "STAGE_D_J_COMPLETE.json")
    j_path = out / "CROSSFIT_NONLOCALITY_PATIENT_SUMMARY.csv"
    checks["crossfit_J"] = {
        "pass": bool(
            len(j) == 28
            and j.subject.nunique() == 28
            and j.all_fits_identifiable.astype(bool).all()
            and stage_d_j.get("target_values_read") is False
            and stage_d_j.get("patients") == 28
            and stage_d_j.get("fits") == 42
            and stage_d_j.get("patient_summary_sha256") == sha256_file(j_path)
        ),
        "patients": int(j.subject.nunique()),
        "identifiable_patients": int(j.all_fits_identifiable.astype(bool).sum()),
        "local_wave_unsupported_patients": int(
            j.any_local_wave_unsupported.astype(bool).sum()
        ),
        "target_free_evidence": "STAGE_D_J_COMPLETE.json_AND_FROZEN_PATIENT_TABLE_HASH",
    }

    block = load_json(out / "INTERICTAL_BLOCK_HELDOUT_SENSITIVITY.json")
    block_fit = pd.read_csv(out / "INTERICTAL_BLOCK_HELDOUT_PER_FIT.csv")
    checks["recording_block_heldout_sensitivity"] = {
        "pass": bool(
            block.get("target_values_read") is False
            and block.get("fit_audit", {}).get("fits_with_at_least_one_strict_test_event") == 40
            and block.get("primary_interaction", {}).get("n") == 27
            and block_fit.fit_id.nunique() == 42
        ),
        "fits": int(block_fit.fit_id.nunique()),
        "patients": int(block_fit.subject.nunique()),
        "fits_with_strict_events": block.get("fit_audit", {}).get(
            "fits_with_at_least_one_strict_test_event"
        ),
        "inference_patients": block.get("primary_interaction", {}).get("n"),
        "interpretation": block.get("interpretation"),
    }

    authorization = load_json(out / "TARGET_UNSEAL_AUTHORIZATION.json")
    sensitivity = load_json(out / "INTERICTAL_PRIMARY_SENSITIVITY_ADDENDUM.json")
    sensitivity_manifest = load_json(out / "INTERICTAL_PRIMARY_SENSITIVITY_PREFREEZE_MANIFEST.json")
    block_manifest = load_json(out / "INTERICTAL_BLOCK_HELDOUT_PREFREEZE_MANIFEST.json")
    checks["target_free_sensitivity_addenda"] = {
        "pass": bool(
            sensitivity.get("target_values_read") is False
            and block.get("target_values_read") is False
            and sensitivity_manifest.get("addendum_sha256")
            == sha256_file(out / "INTERICTAL_PRIMARY_SENSITIVITY_ADDENDUM.json")
            and block_manifest.get("summary_sha256")
            == sha256_file(out / "INTERICTAL_BLOCK_HELDOUT_SENSITIVITY.json")
            and iso_timestamp(authorization) >= iso_timestamp(sensitivity_manifest)
            and iso_timestamp(authorization) >= iso_timestamp(block_manifest)
        ),
        "primary_sensitivity_n": sensitivity.get("analyses", {}).get(
            "all_prespecified_primary_patients", {}
        ).get("n"),
        "block_sensitivity_contract": block.get("interpretation"),
    }

    metric_manifest = load_json(out / "PREUNSEAL_ANALYSIS_METRIC_MANIFEST.json")
    metric_freeze = load_json(out / "PREUNSEAL_ANALYSIS_METRIC_FREEZE_COMPLETE.json")
    metric_failures = []
    for relative, evidence in metric_manifest.get("files", {}).items():
        path = out / relative
        if not path.exists():
            metric_failures.append(f"MISSING:{relative}")
        elif sha256_file(path) != evidence.get("sha256"):
            metric_failures.append(f"HASH_MISMATCH:{relative}")
    checks["preunseal_analysis_metric_freeze"] = {
        "pass": bool(
            metric_manifest.get("status") == "PASS_TARGET_FREE"
            and metric_manifest.get("target_values_read") is False
            and metric_freeze.get("status") == "PASS_TARGET_FREE"
            and metric_freeze.get("target_values_read") is False
            and metric_freeze.get("manifest_sha256")
            == sha256_file(out / "PREUNSEAL_ANALYSIS_METRIC_MANIFEST.json")
            and iso_timestamp(authorization) >= iso_timestamp(metric_manifest)
            and iso_timestamp(authorization) >= iso_timestamp(metric_freeze)
            and not metric_failures
            and metric_manifest.get("files", {}).get(
                "GAIN_ADJUSTED_PER_FIT_SEED.csv", {}
            ).get("validation_gain_matching", {}).get("pass") is True
            and metric_manifest.get("files", {}).get(
                "GAIN_ADJUSTED_PER_FIT_SEED.csv", {}
            ).get("validation_gain_matching", {}).get("pairs") == 126
            and metric_manifest.get("files", {}).get(
                "ATTENUATION_PER_DRAW.csv", {}
            ).get("coverage", {}).get("pass") is True
            and metric_manifest.get("files", {}).get(
                "ATTENUATION_PER_DRAW.csv", {}
            ).get("coverage", {}).get("unit_targets") == 504
            and metric_manifest.get("files", {}).get(
                "ATTENUATION_PER_DRAW.csv", {}
            ).get("coverage", {}).get("unit_target_dose_groups") == 2016
            and metric_manifest.get("files", {}).get(
                "ATTENUATION_PER_DRAW.csv", {}
            ).get("draw_semantics", {}).get("pass") is True
        ),
        "files": len(metric_manifest.get("files", {})),
        "authorization_after_metric_freeze": bool(
            iso_timestamp(authorization) >= iso_timestamp(metric_freeze)
        ),
        "failures": metric_failures,
        "validation_gain_matching": metric_manifest.get("files", {}).get(
            "GAIN_ADJUSTED_PER_FIT_SEED.csv", {}
        ).get("validation_gain_matching"),
        "attenuation_coverage": metric_manifest.get("files", {}).get(
            "ATTENUATION_PER_DRAW.csv", {}
        ).get("coverage"),
        "attenuation_draw_semantics": metric_manifest.get("files", {}).get(
            "ATTENUATION_PER_DRAW.csv", {}
        ).get("draw_semantics"),
    }

    empirical_manifest = load_json(out / "EMPIRICAL_FIELD_INPUT_PREFREEZE_MANIFEST.json")
    empirical_failures = []
    for row in empirical_manifest.get("fields", []):
        path = Path(row["path"])
        if not path.exists():
            empirical_failures.append(f"MISSING:{path}")
        elif sha256_file(path) != row["expected_sha256"]:
            empirical_failures.append(f"HASH_MISMATCH:{path}")
    checks["empirical_field_input_prefreeze"] = {
        "pass": bool(
            empirical_manifest.get("status") == "PASS"
            and empirical_manifest.get("target_values_read") is False
            and empirical_manifest.get("fit_rows") == 42
            and empirical_manifest.get("spatial_patients") == 28
            and empirical_manifest.get("early_patients_covered") == 17
            and iso_timestamp(authorization) >= iso_timestamp(empirical_manifest)
            and not empirical_failures
        ),
        "fit_rows": empirical_manifest.get("fit_rows"),
        "spatial_patients": empirical_manifest.get("spatial_patients"),
        "early_patients_covered": empirical_manifest.get("early_patients_covered"),
        "failures": empirical_failures,
    }

    manifest_contracts = (
        ("MODEL_FIELD_MANIFEST.csv", "file_sha256"),
        ("TEMPLATE_FIELD_MANIFEST.csv", "file_sha256"),
        ("ATTENUATED_FIELD_MANIFEST.csv", "file_sha256"),
        ("GAIN_ADJUSTED_FIELD_MANIFEST.csv", "sha256"),
        ("NULL_INDEX_MAP_MANIFEST.csv", "sha256"),
    )
    manifests = {name: verify_manifest(out, name, column)
                 for name, column in manifest_contracts}
    checks["immutable_manifests"] = {
        "pass": all(value["pass"] for value in manifests.values()), "details": manifests,
    }

    attenuation_draw = pd.read_csv(out / "ATTENUATION_PER_DRAW.csv")
    attenuation_patient = pd.read_csv(out / "ATTENUATION_PER_PATIENT_DOSE.csv")
    attenuation_numeric = [
        "local_damage", "distal_damage", "distal_selectivity", "contact_nll",
        "rollout_spearman",
    ]
    draw_keys = ["subject", "fit_id", "scope", "target", "alpha", "seed"]
    draw_aggregate = attenuation_draw.groupby(draw_keys, as_index=False).agg(
        **{name: (name, "median") for name in attenuation_numeric},
        inferential_eligible=("inferential_eligible", "all"),
        n_valid_matched_draws=("n_valid_matched_draws", "min"),
    )
    fit_keys = ["subject", "fit_id", "scope", "target", "alpha"]
    seed_aggregate = draw_aggregate.groupby(fit_keys, as_index=False).agg(
        **{name: (name, "median") for name in attenuation_numeric},
        inferential_eligible=("inferential_eligible", "all"),
        n_valid_matched_draws=("n_valid_matched_draws", "min"),
    )
    expected_attenuation_patient = seed_aggregate.groupby(
        ["subject", "target", "alpha"], as_index=False
    ).agg(
        **{name: (name, "mean") for name in attenuation_numeric},
        inferential_eligible=("inferential_eligible", "all"),
        n_valid_matched_draws=("n_valid_matched_draws", "min"),
    )
    attenuation_expected_rows = 28 * 4 * 4
    attenuation_aggregation = compare_numeric_tables(
        attenuation_patient,
        expected_attenuation_patient,
        ["subject", "target", "alpha"],
        atol=2e-12,
    )
    checks["attenuation_patient_first_aggregation"] = {
        "pass": bool(
            attenuation_aggregation["pass"]
            and assess_attenuation_draw_semantics(attenuation_draw)["pass"]
            and attenuation_draw.target_values_read.eq(False).all()
            and len(attenuation_patient) == attenuation_expected_rows
            and attenuation_patient.subject.nunique() == 28
            and attenuation_patient.target.nunique() == 4
            and attenuation_patient.alpha.nunique() == 4
        ),
        "contract": "DRAW_MEDIAN_THEN_SEED_MEDIAN_THEN_FIT_MEAN_WITHIN_PATIENT",
        "expected_patient_dose_rows": attenuation_expected_rows,
        "observed_patient_dose_rows": len(attenuation_patient),
        "comparison": attenuation_aggregation,
        "draw_semantics": assess_attenuation_draw_semantics(attenuation_draw),
    }

    gain_seed = pd.read_csv(out / "GAIN_ADJUSTED_PER_FIT_SEED.csv")
    gain_patient = pd.read_csv(out / "GAIN_ADJUSTED_PER_PATIENT.csv")
    gain_fit = gain_seed.groupby(
        ["subject", "fit_id", "scope", "arm"], as_index=False
    ).median(numeric_only=True)
    expected_gain_patient = gain_fit.groupby(
        ["subject", "arm"], as_index=False
    ).mean(numeric_only=True)
    gain_aggregation = compare_numeric_tables(
        gain_patient, expected_gain_patient, ["subject", "arm"], atol=2e-12,
    )
    # This is a sensitivity analysis only if the validation-selected recurrent
    # rescaling actually places both arms on the same finite-horizon-gain
    # scale.  Merely producing a CSV is not evidence that matching succeeded.
    gain_match = assess_gain_matching(gain_seed, tolerance=0.01)
    checks["gain_adjusted_patient_first_aggregation"] = {
        "pass": bool(
            gain_aggregation["pass"]
            and len(gain_seed) == 252
            and len(gain_patient) == 56
            and gain_patient.subject.nunique() == 28
            and gain_match["pass"]
        ),
        "contract": "SEED_MEDIAN_THEN_FIT_MEAN_WITHIN_PATIENT",
        "fit_seed_rows": len(gain_seed),
        "patient_rows": len(gain_patient),
        "validation_gain_matching": gain_match,
        "comparison": gain_aggregation,
    }

    unlock = load_json(out / "early_ictal/TARGET_UNLOCK_RECORD.json")
    summary = load_json(out / "early_ictal/EARLY_ICTAL_V0_5_SUMMARY.json")
    frozen_failures = [relative for relative, digest in authorization["frozen_hashes"].items()
                       if sha256_file(out / relative) != digest]
    checks["target_seal_and_locked_scoring"] = {
        "pass": bool(
            authorization.get("authorized") is True
            and authorization.get("target_values_read") is False
            and unlock.get("target_values_read_by_this_v0_5_scorer_before_record") is False
            and iso_timestamp(unlock) >= iso_timestamp(authorization)
            and not frozen_failures
            and summary.get("patients") == 17 and summary.get("seizures") == 167
            and summary.get("target_values_read") is True
        ),
        "authorization_before_unlock": iso_timestamp(unlock) >= iso_timestamp(authorization),
        "frozen_hash_failures": frozen_failures,
        "status": summary.get("status"),
    }
    scorer_attempts_path = out / "early_ictal/TARGET_SCORER_ATTEMPT_LEDGER.jsonl"
    scorer_attempts = [
        json.loads(line) for line in scorer_attempts_path.read_text().splitlines()
        if line.strip()
    ] if scorer_attempts_path.exists() else []
    checks["immutable_first_unlock_and_attempt_ledger"] = {
        "pass": bool(
            scorer_attempts
            and all(
                attempt.get("authorization_sha256") == sha256_file(
                    out / "TARGET_UNSEAL_AUTHORIZATION.json"
                )
                and attempt.get("first_unlock_record_sha256")
                == sha256_file(out / "early_ictal/TARGET_UNLOCK_RECORD.json")
                for attempt in scorer_attempts
            )
            and iso_timestamp(unlock) <= datetime.fromisoformat(
                scorer_attempts[0]["started_utc"].replace("Z", "+00:00")
            ).astimezone(timezone.utc)
        ),
        "attempts": len(scorer_attempts),
        "first_unlock_sha256": sha256_file(
            out / "early_ictal/TARGET_UNLOCK_RECORD.json"
        ),
        "contract": "EXCLUSIVE_FIRST_UNLOCK_PLUS_APPEND_ONLY_ATTEMPT_LEDGER",
    }
    checks["broadband_target_source_lineage"] = assess_broadband_target_lineage(
        out, BROADBAND_TARGET_ROOT
    )

    null_manifest = pd.read_csv(out / "NULL_INDEX_MAP_MANIFEST.csv")
    expected_null_rows = routing[["subject", "seizure_idx"]].drop_duplicates()
    observed_null_rows = null_manifest[["subject", "seizure_idx"]].drop_duplicates()
    null_join = expected_null_rows.merge(
        observed_null_rows, on=["subject", "seizure_idx"], how="outer", indicator=True,
    )
    checks["synchronized_spatial_null_contract"] = {
        "pass": bool(
            len(null_manifest) == 167
            and null_manifest[["subject", "seizure_idx"]].drop_duplicates().shape[0] == 167
            and null_manifest.n_permutations.eq(5000).all()
            and null_manifest.target_values_read.eq(False).all()
            and null_join._merge.eq("both").all()
        ),
        "rows": len(null_manifest),
        "unique_seizures": int(
            null_manifest[["subject", "seizure_idx"]].drop_duplicates().shape[0]
        ),
        "permutations_per_seizure": sorted(
            map(int, null_manifest.n_permutations.unique().tolist())
        ),
        "primary": "SYNCHRONIZED_ALL_CONTACT_LABEL_PERMUTATION",
    }
    two_dimensional_by_subject = census.groupby("subject").geometry_class.apply(
        lambda values: bool(np.all(values == "TWO_DIMENSIONAL"))
    )
    null_is_2d = null_manifest.subject.map(two_dimensional_by_subject).astype(bool)
    expected_distance_eligible = null_manifest.n_contacts.ge(9) & null_is_2d
    expected_variogram_eligible = null_manifest.n_contacts.ge(12) & null_is_2d
    checks["small_montage_spatial_null_eligibility"] = {
        "pass": bool(
            null_manifest.n_contacts.min() >= 6
            and null_manifest.n_contacts.max() <= 52
            and null_manifest.within_shaft_contract.eq(
                "PURE_WITHIN_SHAFT_MIN_GROUP_4_NO_FALLBACK"
            ).all()
            and null_manifest.distance_bin_eligible.eq(
                expected_distance_eligible
            ).all()
            and null_manifest.variogram_eligible.eq(
                expected_variogram_eligible
            ).all()
            and (~null_manifest.spectral_eligible.astype(bool)
                 | (null_manifest.n_contacts.ge(8) & null_is_2d)).all()
        ),
        "contact_range": [
            int(null_manifest.n_contacts.min()), int(null_manifest.n_contacts.max())
        ],
        "within_shaft_eligible_seizures": int(
            null_manifest.within_shaft_eligible.astype(bool).sum()
        ),
        "distance_bin_eligible_seizures": int(
            null_manifest.distance_bin_eligible.astype(bool).sum()
        ),
        "spectral_eligible_seizures": int(
            null_manifest.spectral_eligible.astype(bool).sum()
        ),
        "variogram_eligible_seizures": int(
            null_manifest.variogram_eligible.astype(bool).sum()
        ),
        "boundary": "ROBUSTNESS_NULLS_RUN_ONLY_WHEN_GEOMETRY_IS_IDENTIFIABLE",
    }

    early_seizure = pd.read_csv(out / "early_ictal/EARLY_ICTAL_PER_SEIZURE.csv")
    early_patient = pd.read_csv(out / "early_ictal/EARLY_ICTAL_PER_PATIENT.csv")
    condition_inventory = load_json(out / "early_ictal/EARLY_ICTAL_CONDITION_INVENTORY.json")
    inventory_rows = early_seizure.groupby(
        ["subject", "seizure_idx", "endpoint"], sort=False
    ).size()
    patient_inventory_rows = early_patient.groupby(
        ["subject", "endpoint"], sort=False
    ).size()
    primary_null_rows = early_patient[
        early_patient.condition.isin((
            "INTACT|L3_LOCAL_PLUS_LEARNED_LR",
            "INTACT|L2M_MACRO_MATCHED_RANDOM_LR",
        ))
        & early_patient.endpoint.isin(("canonical_full", "seed_removed"))
    ]
    expected_inventory = {
        (str(row["subject"]), int(row["seizure_idx"]), str(row["endpoint"])):
        int(row["expected_conditions"])
        for row in condition_inventory.get("rows", [])
    }
    observed_inventory = {
        (str(subject), int(seizure), str(endpoint)): int(value)
        for (subject, seizure, endpoint), value in inventory_rows.items()
    }
    expected_patient_inventory = {}
    for (subject, _, endpoint), value in expected_inventory.items():
        key = (subject, endpoint)
        if key in expected_patient_inventory and expected_patient_inventory[key] != value:
            expected_patient_inventory[key] = -1
        else:
            expected_patient_inventory[key] = value
    observed_patient_inventory = {
        (str(subject), str(endpoint)): int(value)
        for (subject, endpoint), value in patient_inventory_rows.items()
    }
    checks["early_ictal_complete_condition_inventory"] = {
        "pass": bool(
            len(early_seizure) == int(condition_inventory["expected_per_seizure_rows"])
            and len(early_patient) == int(condition_inventory["expected_per_patient_rows"])
            and observed_inventory == expected_inventory
            and observed_patient_inventory == expected_patient_inventory
            and condition_inventory.get(
                "omitted_conditions_are_prefrozen_nonidentifiable_only"
            ) is True
            and len(primary_null_rows) == 17 * 2 * 2
            and primary_null_rows.all_contact_null_finite_draws.eq(5000).all()
        ),
        "per_seizure_rows": len(early_seizure),
        "per_patient_rows": len(early_patient),
        "canonical_condition_counts": sorted(set(
            inventory_rows.xs("canonical_full", level="endpoint").astype(int)
        )),
        "seed_removed_condition_counts": sorted(set(
            inventory_rows.xs("seed_removed", level="endpoint").astype(int)
        )),
        "inventory_contract": condition_inventory.get("contract"),
        "finite_primary_null_draws": sorted(
            map(int, primary_null_rows.all_contact_null_finite_draws.unique())
        ),
        "all_condition_nonidentifiable_rows": int(
            early_patient.all_contact_null_finite_draws.eq(0).sum()
        ),
    }
    checks["train_prevalence_mixture_algebra"] = (
        assess_train_prevalence_mixture_algebra(out, census)
    )
    early_expected_rows = []
    for keys, group in early_seizure.groupby(
        ["subject", "condition", "endpoint"], sort=False
    ):
        subject, condition, endpoint = keys
        first = group.iloc[0]
        early_expected_rows.append({
            "subject": subject, "condition": condition, "endpoint": endpoint,
            "family": first.family, "arm": first.arm, "target": first.target,
            "alpha": first.alpha, "n_seizures": len(group),
            "n_contacts": int(group.n_contacts.min()),
            "observed": float(np.nanmedian(group.observed)),
            "within_shaft_margin": float(np.nanmedian(group.within_shaft_margin)),
            "distance_bin_margin": float(np.nanmedian(group.distance_bin_margin)),
            "spectral_margin": float(np.nanmedian(group.spectral_margin)),
            "variogram_margin": float(np.nanmedian(group.variogram_margin)),
            "variogram_fitted_range_mm": float(np.nanmedian(group.variogram_fitted_range_mm)),
            "rank_weighted_concordance": float(np.nanmedian(group.rank_weighted_concordance)),
            "top20_jaccard": float(np.nanmedian(group.top20_jaccard)),
            "peak_contact_distance_mm": float(np.nanmedian(group.peak_contact_distance_mm)),
            "spatial_sinkhorn_normalized": float(
                np.nanmedian(group.spatial_sinkhorn_normalized)
            ),
        })
    early_expected = pd.DataFrame(early_expected_rows)
    early_aggregation = compare_numeric_tables(
        early_patient,
        early_expected,
        ["subject", "condition", "endpoint"],
        atol=2e-12,
    )
    checks["early_ictal_patient_first_aggregation"] = {
        "pass": bool(
            early_aggregation["pass"]
            and early_seizure.subject.nunique() == 17
            and early_seizure[["subject", "seizure_idx"]].drop_duplicates().shape[0] == 167
        ),
        "seizure_to_patient_contract": "MEDIAN_WITHIN_PATIENT",
        "patients": int(early_seizure.subject.nunique()),
        "seizures": int(early_seizure[["subject", "seizure_idx"]].drop_duplicates().shape[0]),
        "comparison": early_aggregation,
        "null_fold_note": "all-contact null is synchronized then median-folded within patient",
    }
    checks["primary_early_ictal_null_folding"] = assess_primary_early_ictal_null_folding(
        out, routing, early_seizure, early_patient
    )
    checks["primary_early_interaction"] = assess_primary_early_interaction(
        out, routing, early_patient
    )
    checks["early_endpoint_semantics"] = {
        "pass": bool(
            summary.get("target") == "clinical onset 0-10 s, 1-150 Hz broadband energy"
            and summary.get("primary_endpoint")
            == "signed best-mode Spearman oracle repertoire coverage"
            and summary.get("primary_null")
            == (
                "joint patient-label permutation and synchronized all-contact "
                "spatial-null interaction; both must pass"
            )
            and summary.get("primary_interaction", {}).get(
                "joint_primary_contract"
            ) == "BOTH_PATIENT_LABEL_AND_SYNCHRONIZED_SPATIAL_NULL_MUST_PASS"
            and np.isfinite(
                summary.get("primary_interaction", {}).get(
                    "joint_primary_p_greater", np.nan
                )
            )
            and summary.get("primary_delta_contract")
            == "raw_signed_oracle_L3_minus_L2m; null margin separate"
        ),
        "target": summary.get("target"),
        "endpoint": summary.get("primary_endpoint"),
        "delta": summary.get("primary_delta_contract"),
        "forbidden_interpretation": "BROADBAND_FIELD_IS_NOT_ARRIVAL_OR_RECRUITMENT_ORDER",
    }

    pytest_evidence_path = out / "PREFINAL_RELATED_PYTEST_EVIDENCE.json"
    pytest_evidence = load_json(pytest_evidence_path) if pytest_evidence_path.exists() else {}
    pytest_log = out / "PREFINAL_RELATED_PYTEST.log"
    pytest_summary = str(pytest_evidence.get("summary_tail", ""))
    pytest_matches = [int(value) for value in re.findall(r"(\d+) passed", pytest_summary)]
    pytest_passed = max(pytest_matches, default=0)
    checks["persisted_related_tests"] = {
        "pass": bool(
            pytest_evidence.get("returncode") == 0
            and pytest_passed >= 124
            and pytest_log.exists()
            and pytest_evidence.get("log_sha256") == sha256_file(pytest_log)
            and pytest_evidence.get("target_values_read") is False
        ),
        "summary": pytest_summary,
        "passed_tests": pytest_passed,
        "minimum_frozen_related_test_count": 124,
        "target_values_read": pytest_evidence.get("target_values_read"),
    }

    stem = figure / "topic5_figure6_multiscale_scaffold_v0_5"
    assets = [stem.with_suffix(suffix) for suffix in (".png", ".pdf", ".svg")]
    pdf_pages = -1
    if assets[1].exists():
        info = subprocess.run(["pdfinfo", str(assets[1])], capture_output=True, text=True, check=False)
        for line in info.stdout.splitlines():
            if line.startswith("Pages:"):
                pdf_pages = int(line.split(":", 1)[1])
    png_shape = None
    if assets[0].exists():
        with Image.open(assets[0]) as image:
            png_shape = [int(image.width), int(image.height)]
    source_manifest = (
        load_json(figure / "FIGURE6_SOURCE_DATA_MANIFEST.json")
        if (figure / "FIGURE6_SOURCE_DATA_MANIFEST.json").exists() else {}
    )
    panel_c_source = source_manifest.get("source_tables", {}).get("panel_c", {})
    panel_e_source = source_manifest.get("source_tables", {}).get("panel_e", {})
    panel_a_nodes = source_manifest.get("source_tables", {}).get("panel_a_nodes", {})
    panel_a_contacts = source_manifest.get("source_tables", {}).get("panel_a_contacts", {})
    panel_a_edges = source_manifest.get("source_tables", {}).get("panel_a_edges", {})
    panel_b_source = source_manifest.get("source_tables", {}).get("panel_b", {})
    panel_d_source = source_manifest.get("source_tables", {}).get("panel_d", {})
    source_readme = source_manifest.get("source_readme", {})
    source_producer = source_manifest.get("producer", {})
    source_readme_path = Path(str(source_readme.get("path", "")))
    source_producer_path = Path(str(source_producer.get("path", "")))
    source_hash_failures = []
    for label, evidence in source_manifest.get("source_tables", {}).items():
        path_text = str(evidence.get("path", ""))
        path = Path(path_text) if path_text else Path("/__missing_figure_source__")
        if not path.is_file():
            source_hash_failures.append(f"MISSING:{label}:{path}")
        elif sha256_file(path) != evidence.get("sha256"):
            source_hash_failures.append(f"HASH_MISMATCH:{label}:{path}")
    checks["figure_package"] = {
        "pass": bool(all(path.exists() and path.stat().st_size > 0 for path in assets)
                     and pdf_pages == 1 and png_shape is not None
                     and min(png_shape) >= 4000 and (figure / "README.md").exists()
                     and (figure / "FIGURE6_COMPLETE.json").exists()
                     and (figure / "FIGURE6_SOURCE_DATA_MANIFEST.json").exists()
                     and source_manifest.get("no_endpoint_recomputation") is True
                     and bool(source_readme.get("path"))
                     and source_readme_path.is_file()
                     and sha256_file(source_readme_path) == source_readme.get("sha256")
                     and source_producer_path.resolve()
                     == (ROOT / "scripts/export_topic5_figure6_source_data_v0_5.py").resolve()
                     and source_producer_path.is_file()
                     and sha256_file(source_producer_path) == source_producer.get("sha256")
                     and source_manifest.get("visual_source_export_contract")
                     == "PANELS_A_B_D_REUSE_THE_SAME_FROZEN_E1146_PLANE_ROLLOUTS_FIELDS_AND_TARGET_VECTOR"
                     and panel_a_nodes.get("rows") == 104
                     and panel_a_contacts.get("rows") == 15
                     and panel_a_edges.get("rows", 0) > 1000
                     and panel_b_source.get("rows") == 1800
                     and panel_d_source.get("rows") == 15
                     and panel_c_source.get("rows") == 28
                     and panel_e_source.get("rows") == 51
                     and not source_hash_failures
                     and load_json(figure / "FIGURE6_METADATA.json").get("panel_c", {}).get("contract")
                     == "v0.5_true_suffix_vs_split_matched_reassigned_suffix"
                     and load_json(figure / "FIGURE6_METADATA.json").get("panel_e", {}).get("contract")
                     == "oracle_plus_train_prevalence_mixture_plus_primary_J_interaction"
                     and load_json(figure / "FIGURE6_METADATA.json").get("panel_e", {}).get("significance_marks")
                     == "JOINT_PRIMARY_J_INTERACTION_ONLY"
                     and load_json(figure / "FIGURE6_METADATA.json").get("panels_f_i", {}).get("panel_i_contract")
                     == "patient_paired_heldout_finite_horizon_G3_L2m_vs_L3"),
        "assets": {path.name: (sha256_file(path) if path.exists() else None) for path in assets},
        "pdf_pages": pdf_pages, "png_shape": png_shape,
        "panel_c_source_rows": panel_c_source.get("rows"),
        "panel_e_source_rows": panel_e_source.get("rows"),
        "panel_a_node_rows": panel_a_nodes.get("rows"),
        "panel_a_contact_rows": panel_a_contacts.get("rows"),
        "panel_a_edge_rows": panel_a_edges.get("rows"),
        "panel_b_source_rows": panel_b_source.get("rows"),
        "panel_d_source_rows": panel_d_source.get("rows"),
        "source_hash_failures": source_hash_failures,
        "source_producer_sha256": source_producer.get("sha256"),
        "readme": (figure / "README.md").exists(),
        "source_data_manifest": (figure / "FIGURE6_SOURCE_DATA_MANIFEST.json").exists(),
    }

    finalizer_manifest = load_json(out / "FIGURE6_FINALIZER_R2_PREFREEZE_MANIFEST.json")
    finalizer_checks = {
        "panel_c_decision_sha256": out / "FIGURE6_PREUNSEAL_PANEL_C_DECISION.json",
        "panel_e_decision_sha256": out / "FIGURE6_PREUNSEAL_PANEL_E_DECISION.json",
        "panel_i_decision_sha256": out / "FIGURE6_PREUNSEAL_PANEL_I_DECISION.json",
        "finalizer_script_sha256": ROOT / "scripts/finalize_topic5_figure6_multiscale_scaffold_v0_5_r2.py",
    }
    finalizer_failures = [
        key for key, path in finalizer_checks.items()
        if finalizer_manifest.get(key) != sha256_file(path)
    ]
    checks["figure_prefreeze_contract"] = {
        "pass": bool(
            finalizer_manifest.get("target_values_read") is False
            and not finalizer_failures
            and iso_timestamp(authorization) >= iso_timestamp(finalizer_manifest)
        ),
        "hash_failures": finalizer_failures,
        "authorization_after_prefreeze": iso_timestamp(authorization) >= iso_timestamp(
            finalizer_manifest
        ),
    }

    adjudicator_manifest = load_json(out / "FINAL_CLAIM_ADJUDICATOR_PREFREEZE_MANIFEST.json")
    adjudicator = load_json(out / "FINAL_CLAIM_ADJUDICATION.json")
    adjudicator_script = ROOT / "scripts/adjudicate_topic5_multiscale_claims_v0_5.py"
    checks["claim_adjudication_prefreeze_contract"] = {
        "pass": bool(
            adjudicator_manifest.get("target_values_read") is False
            and adjudicator_manifest.get("script_sha256") == sha256_file(adjudicator_script)
            and iso_timestamp(authorization) >= iso_timestamp(adjudicator_manifest)
            and adjudicator.get("status") == "COMPLETE_LOCKED_INTERNAL_FOLLOWUP"
            and adjudicator.get("target_role")
            == "LOCKED_INTERNAL_MECHANISTIC_FOLLOWUP_NOT_INDEPENDENT_CONFIRMATION"
        ),
        "authorization_after_prefreeze": iso_timestamp(authorization) >= iso_timestamp(
            adjudicator_manifest
        ),
        "adjudication_status": adjudicator.get("status"),
    }

    report_path = ROOT / (
        "docs/archive/topic5/"
        "multiscale_effective_scaffold_v0_5_closeout_2026-08-14.md"
    )
    report_text = report_path.read_text() if report_path.exists() else ""
    checks["closeout_report_finalized"] = {
        "pass": bool(
            report_text.count("<!-- FINAL_RESULTS_BEGIN -->") == 1
            and report_text.count("<!-- FINAL_RESULTS_END -->") == 1
            and "待 `PIPELINE_COMPLETE.json`" not in report_text
            and "尚未完成的验收项" not in report_text
            and "Stage G 尚未运行" not in report_text
            and "当前没有 unseal authorization" not in report_text
            and "| 运行中 |" not in report_text
            and "| 待 F 完成后自动运行 |" not in report_text
            and "| 待 G 完成后自动运行 |" not in report_text
            and "FINAL_CLAIM_ADJUDICATION.json" in report_text
            and "SCORER_CONTRACT_PREFREEZE_REPAIR.json" in report_text
            and "CLOSEOUT_TOOLING_PREFREEZE_MANIFEST.json" in report_text
        ),
        "path": str(report_path),
    }

    diff_check = subprocess.run(
        ["git", "diff", "--check"], cwd=ROOT, capture_output=True, text=True, check=False,
    )
    checks["working_tree_patch_integrity"] = {
        "pass": diff_check.returncode == 0,
        "git_diff_check_returncode": diff_check.returncode,
        "stderr": diff_check.stderr[-2000:],
        "note": "Unrelated dirty paths are preserved; this checks patch syntax only.",
    }

    failures = [name for name, check in checks.items() if not check["pass"]]
    report = {
        "contract": "topic5_multiscale_closeout_audit_v0_5",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not failures else "FAIL",
        "failed_checks": failures, "checks": checks,
    }
    destination = out / "CLOSEOUT_AUDIT.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
