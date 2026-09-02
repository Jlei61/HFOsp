#!/usr/bin/env python3
"""Score frozen v0.5 fields on the 17-patient broadband early-ictal benchmark."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
CANONICAL_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
FIELD_ROOT = CANONICAL_ROOT / "results/interictal_propagation_masked/template_gradient_fields/per_subject"
BB_ROOT = CANONICAL_ROOT / "results/topic5_ictal_recruitment/t0_feature_cache_bb150_1_150"
ARMS = (
    "L0_LOCAL_ONLY", "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
    "L2M_MACRO_MATCHED_RANDOM_LR", "L3_LOCAL_PLUS_LEARNED_LR",
    "C_L3_ORDER_SHUFFLED",
)
ATTENUATION_TARGETS = ("L1_ADDED", "L2M_ADDED", "L3_ADDED", "L3_MATCHED_LOCAL")
ENDPOINTS = ("canonical_full", "seed_removed")


def expected_condition_inventory(
    endpoint: str,
    attenuation_conditions: set[str] | None = None,
) -> set[str]:
    conditions = {f"INTACT|{arm}" for arm in ARMS}
    conditions |= {f"INTACT_MIXTURE|{arm}" for arm in ARMS}
    conditions |= {
        f"GAIN_MATCHED|{arm}"
        for arm in ("L2M_MACRO_MATCHED_RANDOM_LR", "L3_LOCAL_PLUS_LEARNED_LR")
    }
    conditions |= (
        {
            f"ATTEN|{target}|{alpha:.2f}"
            for target in ATTENUATION_TARGETS
            for alpha in (0.25, 0.50, 0.75, 1.00)
        }
        if attenuation_conditions is None
        else set(attenuation_conditions)
    )
    conditions.add("EMPIRICAL_REFERENCE")
    if endpoint == "canonical_full":
        conditions |= {"TEMPLATE|ORACLE", "TEMPLATE|TRAIN_MIXTURE"}
    return conditions


def available_attenuation_conditions(out: Path, subject: str) -> set[str]:
    """Return only attenuation fields frozen before target unseal.

    Matched-local attenuation is undefined when no valid matched control draw
    exists.  Stage F intentionally omits those fields; the scorer must neither
    reject that prespecified non-identifiability nor synthesize a field after
    target access.
    """
    conditions: set[str] = set()
    for target in ATTENUATION_TARGETS:
        for alpha in (0.25, 0.50, 0.75, 1.00):
            path = (
                out / "attenuation/fields/per_patient" / subject / target /
                f"alpha{alpha:.2f}.npz"
            )
            if path.exists():
                conditions.add(f"ATTEN|{target}|{alpha:.2f}")
    return conditions


def scorer_authorization_status(out: Path, authorization: dict, scorer_path: Path) -> str:
    """Validate the frozen scorer or a recorded target-independent inventory repair."""
    current = sha256_file(scorer_path)
    if authorization.get("scorer_sha256") == current:
        return "ORIGINAL_PREFREEZE_SCORER"
    amendment_path = out / "TARGET_UNSEAL_ENGINEERING_AMENDMENT.json"
    if not amendment_path.exists():
        raise RuntimeError("this scorer is not the frozen authorized source")
    amendment = json.loads(amendment_path.read_text())
    if not (
        amendment.get("status") == "POST_UNSEAL_TARGET_INDEPENDENT_INVENTORY_REPAIR"
        and amendment.get("original_authorization_sha256")
        == sha256_file(out / "TARGET_UNSEAL_AUTHORIZATION.json")
        and amendment.get("old_scorer_sha256") == authorization.get("scorer_sha256")
        and amendment.get("new_scorer_sha256") == current
        and amendment.get("model_or_field_generation_after_unseal") is False
        and amendment.get("primary_estimand_changed") is False
    ):
        raise RuntimeError("post-unseal scorer amendment is invalid")
    return "RECORDED_POST_UNSEAL_INVENTORY_REPAIR"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_payload_manifests(out: Path) -> dict[str, int]:
    """Verify every scorer-consumed payload before reading one target value."""
    contracts = (
        ("MODEL_FIELD_MANIFEST.csv", "file_sha256"),
        ("TEMPLATE_FIELD_MANIFEST.csv", "file_sha256"),
        ("ATTENUATED_FIELD_MANIFEST.csv", "file_sha256"),
        ("GAIN_ADJUSTED_FIELD_MANIFEST.csv", "sha256"),
        ("NULL_INDEX_MAP_MANIFEST.csv", "sha256"),
    )
    checked: dict[str, int] = {}
    for relative, hash_column in contracts:
        manifest = pd.read_csv(out / relative)
        if "target_values_read" in manifest and not manifest.target_values_read.eq(False).all():
            raise RuntimeError(f"target marker is not uniformly false in {relative}")
        pairs = manifest[["path", hash_column]].drop_duplicates()
        for row in pairs.itertuples(index=False):
            path = Path(str(row[0]))
            if not path.is_absolute():
                path = out / path
            if not path.exists() or sha256_file(path) != str(row[1]):
                raise RuntimeError(f"frozen payload hash mismatch: {relative}: {path}")
        checked[relative] = len(pairs)
    return checked


def signed_spearman(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction, target = np.asarray(prediction, float), np.asarray(target, float)
    use = np.isfinite(prediction) & np.isfinite(target)
    if int(use.sum()) < 3 or np.std(prediction[use]) == 0 or np.std(target[use]) == 0:
        return float("nan")
    value = spearmanr(prediction[use], target[use]).statistic
    return float(value) if np.isfinite(value) else float("nan")


def signed_spearman_permutations(
    prediction: np.ndarray,
    target: np.ndarray,
    permutations: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Compute tied-rank Spearman for one field and many label permutations."""
    prediction = np.asarray(prediction, float)
    target = np.asarray(target, float)
    permutations = np.asarray(permutations, dtype=np.intp)
    if (
        prediction.ndim != 1 or target.ndim != 1
        or len(prediction) != len(target)
        or permutations.ndim != 2 or permutations.shape[1] != len(target)
    ):
        raise ValueError("invalid prediction/target/permutation shapes")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        observed = signed_spearman(prediction, target)
        null = np.asarray([
            signed_spearman(prediction, target[row]) for row in permutations
        ], dtype=np.float32)
        return observed, null
    if len(target) < 3:
        return float("nan"), np.full(len(permutations), np.nan, dtype=np.float32)
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


def signed_spearman_target_matrix(
    prediction: np.ndarray,
    surrogate_targets: np.ndarray,
) -> np.ndarray:
    """Signed Spearman against rows of frozen/generated target surrogates."""
    prediction = np.asarray(prediction, float)
    surrogate_targets = np.asarray(surrogate_targets, float)
    if (prediction.ndim != 1 or surrogate_targets.ndim != 2
            or surrogate_targets.shape[1] != len(prediction)):
        raise ValueError("invalid prediction/surrogate target shapes")
    if not np.isfinite(prediction).all() or not np.isfinite(surrogate_targets).all():
        return np.asarray([
            signed_spearman(prediction, row) for row in surrogate_targets
        ], dtype=np.float32)
    x = rankdata(prediction, method="average").astype(float)
    y = rankdata(surrogate_targets, method="average", axis=1).astype(float)
    x -= x.mean()
    y -= y.mean(axis=1, keepdims=True)
    x_norm = float(np.sqrt(np.sum(x * x)))
    y_norm = np.sqrt(np.sum(y * y, axis=1))
    denominator = x_norm * y_norm
    return np.divide(
        y @ x, denominator,
        out=np.full(len(y), np.nan, dtype=float), where=denominator > 0,
    ).astype(np.float32)


def spectral_surrogates(target: np.ndarray, eigenvectors: np.ndarray,
                        signs: np.ndarray) -> np.ndarray:
    """Preserve graph-Laplacian spectral power while randomizing phase signs."""
    target = np.asarray(target, float)
    eigenvectors = np.asarray(eigenvectors, float)
    signs = np.asarray(signs, float)
    if eigenvectors.shape != (len(target), len(target)) or signs.shape[1] != len(target):
        raise ValueError("invalid spectral null operator")
    mean = float(target.mean())
    coefficients = eigenvectors.T @ (target - mean)
    return (signs * coefficients[None, :]) @ eigenvectors.T + mean


def fit_variogram_range(target: np.ndarray, xy: np.ndarray) -> float:
    """Fit one isotropic exponential range to the target rank variogram."""
    target, xy = np.asarray(target, float), np.asarray(xy, float)
    highness = (rankdata(target, method="average") - 1.0) / max(len(target) - 1, 1)
    distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    upper = np.triu_indices(len(target), 1)
    d = distance[upper]
    gamma = 0.5 * np.square(highness[:, None] - highness[None, :])[upper]
    positive = d > 0
    d, gamma = d[positive], gamma[positive]
    if len(d) < 6:
        raise ValueError("variogram range requires at least six contact pairs")
    sill = max(float(np.var(highness)), 1e-6)
    grid = np.geomspace(max(float(np.min(d)) * 0.5, 1e-4),
                        max(float(np.max(d)) * 2.0, 2e-4), 48)
    error = [float(np.mean(np.square(gamma - sill * (1.0 - np.exp(-d / value)))))
             for value in grid]
    return float(grid[int(np.argmin(error))])


def variogram_surrogates(target: np.ndarray, xy: np.ndarray,
                         frozen_normals: np.ndarray) -> tuple[np.ndarray, float]:
    """Generate marginal-preserving fields from a fitted spatial covariance."""
    target, xy = np.asarray(target, float), np.asarray(xy, float)
    frozen_normals = np.asarray(frozen_normals, float)
    if frozen_normals.ndim != 2 or frozen_normals.shape[1] != len(target):
        raise ValueError("invalid variogram frozen innovations")
    fitted_range = fit_variogram_range(target, xy)
    distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    covariance = np.exp(-distance / fitted_range) + np.eye(len(target)) * 1e-6
    raw = frozen_normals @ np.linalg.cholesky(covariance).T
    order = np.argsort(raw, axis=1, kind="stable")
    sorted_target = np.sort(target, kind="stable")
    surrogates = np.empty_like(raw)
    np.put_along_axis(
        surrogates, order,
        np.broadcast_to(sorted_target, order.shape), axis=1,
    )
    return surrogates, fitted_range


def weighted_concordance(prediction: np.ndarray, target: np.ndarray) -> float:
    use = np.isfinite(prediction) & np.isfinite(target)
    if int(use.sum()) < 3:
        return float("nan")
    x = np.asarray(prediction[use], float)
    q = rankdata(-np.asarray(target[use], float), method="average")
    y = 1.0 - (q - 1.0) / max(len(q) - 1, 1)
    weights = np.exp(-(q - 1.0) / max(0.2 * len(q), 1e-12))
    weights /= weights.sum()
    x0, y0 = x - np.sum(weights * x), y - np.sum(weights * y)
    denominator = np.sqrt(np.sum(weights * x0 * x0) * np.sum(weights * y0 * y0))
    return float(np.sum(weights * x0 * y0) / denominator) if denominator > 0 else float("nan")


def sinkhorn_distance(prediction: np.ndarray, target: np.ndarray, xy: np.ndarray) -> float:
    prediction, target = np.asarray(prediction, float), np.asarray(target, float)
    if (len(prediction) != len(target) or len(xy) != len(target)
            or not np.isfinite(target).all() or not np.isfinite(prediction).any()):
        return float("nan")
    pred_floor = float(np.nanmin(prediction))
    target_floor = float(np.nanmin(target))
    a = rankdata(np.nan_to_num(prediction, nan=pred_floor))
    b = rankdata(np.nan_to_num(target, nan=target_floor))
    a, b = a / a.sum(), b / b.sum()
    distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    epsilon = max(float(np.median(distance[distance > 0])), 1e-3)
    kernel = np.exp(-distance / epsilon) + 1e-12
    u, v = np.ones_like(a), np.ones_like(b)
    for _ in range(100):
        u = a / np.maximum(kernel @ v, 1e-12)
        v = b / np.maximum(kernel.T @ u, 1e-12)
    plan = u[:, None] * kernel * v[None, :]
    diameter = float(np.max(distance))
    return float(np.sum(plan * distance) / diameter) if diameter > 0 else float("nan")


def tied_peak_distance(prediction: np.ndarray, target: np.ndarray, xy: np.ndarray) -> float:
    prediction, target, xy = np.asarray(prediction, float), np.asarray(target, float), np.asarray(xy, float)
    use = np.isfinite(prediction) & np.isfinite(target) & np.isfinite(xy).all(axis=1)
    if not use.any():
        return float("nan")
    p = np.flatnonzero(use & np.isclose(prediction, np.nanmax(prediction[use]), rtol=0, atol=1e-12))
    t = np.flatnonzero(use & np.isclose(target, np.nanmax(target[use]), rtol=0, atol=1e-12))
    if not len(p) or not len(t):
        return float("nan")
    return float(np.linalg.norm(xy[p].mean(axis=0) - xy[t].mean(axis=0)))


def align(names: np.ndarray, values: np.ndarray, order: list[str], endpoint: str) -> np.ndarray:
    lookup = dict(zip(map(str, names), map(float, values)))
    missing_names = [name for name in order if name not in lookup]
    if missing_names:
        raise RuntimeError(f"field is missing evaluation contacts: {missing_names}")
    result = np.asarray([lookup.get(name, np.nan) for name in order], float)
    if endpoint == "seed_removed":
        # A contact never generated after the supplied first rank is assigned
        # the lowest possible recurrence score, not treated as missing support.
        result = np.nan_to_num(result, nan=0.0)
    elif not np.isfinite(result).all():
        raise RuntimeError("canonical field contains non-finite evaluation values")
    return result


def empirical_score(rank: np.ndarray) -> np.ndarray:
    rank = np.asarray(rank, float)
    span = np.nanmax(rank) - np.nanmin(rank)
    return np.ones_like(rank) if span == 0 else 1.0 - (rank - np.nanmin(rank)) / span


def load_candidates(out: Path, subject: str, endpoint: str, contacts: list[str]) -> dict[str, dict]:
    result = {}
    for arm in ARMS:
        path = out / "model_fields/intact/per_patient" / subject / f"{arm}.npz"
        with np.load(path, allow_pickle=False) as data:
            result[f"INTACT|{arm}"] = {
                "family": "intact", "arm": arm, "target": "", "alpha": 0.0,
                "a": align(data["contacts"], data[f"A_{endpoint}"], contacts, endpoint),
                "b": align(data["contacts"], data[f"B_{endpoint}"], contacts, endpoint),
                "oracle": True,
            }
            mixture = align(
                data["contacts"], data[f"{endpoint}_train_prevalence_mixture"],
                contacts, endpoint,
            )
            result[f"INTACT_MIXTURE|{arm}"] = {
                "family": "intact_mixture", "arm": arm, "target": "",
                "alpha": 0.0, "a": mixture, "b": mixture, "oracle": False,
            }
    for arm in ("L2M_MACRO_MATCHED_RANDOM_LR", "L3_LOCAL_PLUS_LEARNED_LR"):
        path = out / "gain_adjusted_fields/per_patient" / subject / f"{arm}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as data:
            result[f"GAIN_MATCHED|{arm}"] = {
                "family": "gain_adjusted", "arm": arm, "target": "", "alpha": 0.0,
                "a": align(data["contacts"], data[f"A_{endpoint}"], contacts, endpoint),
                "b": align(data["contacts"], data[f"B_{endpoint}"], contacts, endpoint),
                "oracle": True,
            }
    if endpoint == "canonical_full":
        path = out / "model_fields/templates/per_patient" / subject / "TRAIN_ONLY_TEMPLATE_FIELDS.npz"
        with np.load(path, allow_pickle=False) as data:
            result["TEMPLATE|ORACLE"] = {
                "family": "template", "arm": "PREFIX_TEMPLATE", "target": "", "alpha": 0.0,
                "a": align(data["contacts"], data["A"], contacts, endpoint),
                "b": align(data["contacts"], data["B"], contacts, endpoint), "oracle": True,
            }
            mixture = align(data["contacts"], data["train_prevalence_mixture"], contacts, endpoint)
            result["TEMPLATE|TRAIN_MIXTURE"] = {
                "family": "template", "arm": "TRAIN_PREVALENCE_MIXTURE", "target": "", "alpha": 0.0,
                "a": mixture, "b": mixture, "oracle": False,
            }
    for target in ATTENUATION_TARGETS:
        for alpha in (0.25, 0.50, 0.75, 1.00):
            path = out / "attenuation/fields/per_patient" / subject / target / f"alpha{alpha:.2f}.npz"
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as data:
                result[f"ATTEN|{target}|{alpha:.2f}"] = {
                    "family": "attenuated", "arm": target.split("_ADDED")[0],
                    "target": target, "alpha": alpha,
                    "a": align(data["contacts"], data[f"A_{endpoint}"], contacts, endpoint),
                    "b": align(data["contacts"], data[f"B_{endpoint}"], contacts, endpoint),
                    "oracle": True,
                }
    empirical = json.loads((FIELD_ROOT / f"{subject}.json").read_text())["interictal_field"]
    empirical_order = [str(value) for value in empirical["contact_order"]]
    take = np.asarray([empirical_order.index(name) for name in contacts], dtype=int)
    result["EMPIRICAL_REFERENCE"] = {
        "family": "reference", "arm": "EMPIRICAL_REFERENCE", "target": "", "alpha": 0.0,
        "a": empirical_score(np.asarray(empirical["rank_a"], float)[take]),
        "b": empirical_score(np.asarray(empirical["rank_b"], float)[take]), "oracle": True,
    }
    return result


def score_candidate(candidate: dict, target: np.ndarray, permutations: np.ndarray) -> tuple[dict, np.ndarray]:
    ra, null_a = signed_spearman_permutations(candidate["a"], target, permutations)
    rb, null_b = signed_spearman_permutations(candidate["b"], target, permutations)
    if candidate["oracle"]:
        observed = float(np.nanmax([ra, rb])) if np.isfinite([ra, rb]).any() else float("nan")
        null = np.fmax(null_a, null_b)
    else:
        observed = ra
        null = null_a
    if np.isfinite(ra) and (not np.isfinite(rb) or ra >= rb):
        selected_mode = "A"
        chosen = candidate["a"]
    elif np.isfinite(rb):
        selected_mode = "B"
        chosen = candidate["b"]
    else:
        selected_mode = "NOT_IDENTIFIABLE"
        chosen = np.full_like(np.asarray(target, float), np.nan)
    k = max(1, int(np.ceil(0.20 * len(target))))
    if np.isfinite(chosen).all():
        predicted_top = set(np.argsort(chosen)[-k:])
        observed_top = set(np.argsort(target)[-k:])
        overlap = len(predicted_top & observed_top) / len(predicted_top | observed_top)
        rank_weighted = weighted_concordance(chosen, target)
    else:
        overlap = float("nan")
        rank_weighted = float("nan")
    return {
        "observed": observed, "mode_a_r": ra, "mode_b_r": rb,
        "selected_mode": selected_mode,
        "identifiable": selected_mode != "NOT_IDENTIFIABLE",
        "rank_weighted_concordance": rank_weighted,
        "top20_jaccard": float(overlap),
    }, null


def score_candidate_surrogates(candidate: dict, surrogate_targets: np.ndarray) -> np.ndarray:
    null_a = signed_spearman_target_matrix(candidate["a"], surrogate_targets)
    if not candidate["oracle"]:
        return null_a
    null_b = signed_spearman_target_matrix(candidate["b"], surrogate_targets)
    return np.fmax(null_a, null_b)


def load_broadband(subject: str, seizure_idx: int, contacts: list[str]) -> np.ndarray:
    meta = json.loads((BB_ROOT / f"{subject}.json").read_text())
    with np.load(BB_ROOT / f"{subject}.npz", allow_pickle=True) as cache:
        names = [str(value) for value in meta.get("channels", cache["channels"].tolist())]
        key = f"bb150_auc__{int(seizure_idx)}"
        if key not in cache.files:
            raise KeyError(f"all-seizure broadband target missing: {subject} {seizure_idx}")
        values = np.asarray(cache[key], float)
    lookup = dict(zip(names, values))
    return np.asarray([lookup.get(name, np.nan) for name in contacts], float)


def paired_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, float); values = values[np.isfinite(values)]
    nonzero = values[np.abs(values) > 1e-9]
    p = 1.0 if len(nonzero) == 0 else float(wilcoxon(nonzero, alternative="greater").pvalue)
    return {
        "n": len(values), "median": float(np.median(values)),
        "n_positive": int(np.sum(values > 1e-9)), "n_negative": int(np.sum(values < -1e-9)),
        "n_tied": int(np.sum(np.abs(values) <= 1e-9)), "wilcoxon_p_greater": p,
    }


def interaction(J: np.ndarray, gain: np.ndarray, seed: int = 20260813) -> dict:
    J, gain = np.asarray(J, float), np.asarray(gain, float)
    use = np.isfinite(J) & np.isfinite(gain)
    J, gain = J[use], gain[use]
    if len(J) < 5 or np.unique(J).size < 2 or np.unique(gain).size < 2:
        return {"n": int(len(J)), "status": "NOT_IDENTIFIABLE"}
    observed = float(spearmanr(J, gain).statistic)
    rng = np.random.default_rng(seed)
    null = np.asarray([spearmanr(J, rng.permutation(gain)).statistic for _ in range(100_000)])
    bootstrap = []
    for _ in range(10_000):
        take = rng.integers(0, len(J), len(J))
        value = spearmanr(J[take], gain[take]).statistic
        if np.isfinite(value): bootstrap.append(value)
    leaveout = [spearmanr(J[np.arange(len(J)) != i], gain[np.arange(len(J)) != i]).statistic
                for i in range(len(J))]
    return {
        "n": len(J), "spearman_rho": observed,
        "permutation_p_greater": float((1 + np.sum(null >= observed)) / 100_001),
        "bootstrap_95_ci": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
        "leave_one_patient_out_range": [float(np.nanmin(leaveout)), float(np.nanmax(leaveout))],
    }


def empirical_null_p(
    observed: float,
    null: np.ndarray,
    *,
    tolerance: float = 1e-7,
) -> tuple[float, int]:
    """Finite-denominator one-sided Monte Carlo P with an explicit tie band."""
    null = np.asarray(null, float)
    finite = null[np.isfinite(null)]
    if not np.isfinite(observed) or len(finite) == 0:
        return float("nan"), int(len(finite))
    p_value = (1 + np.sum(finite >= observed - tolerance)) / (1 + len(finite))
    return float(p_value), int(len(finite))


def record_target_unlock(
    early: Path,
    authorization_path: Path,
    verified_payloads: dict[str, int],
) -> Path:
    """Write the immutable first-unlock record and append a scorer attempt."""
    unlock_record = early / "TARGET_UNLOCK_RECORD.json"
    authorization_sha256 = sha256_file(authorization_path)
    unlock_payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_sha256": authorization_sha256,
        "target_values_read_by_this_v0_5_scorer_before_record": False,
        "project_history_target_previously_viewed": True,
        "patients": 17,
        "seizures": 167,
        "target": "clinical onset 0-10 s, 1-150 Hz broadband energy",
        "verified_frozen_payloads": verified_payloads,
    }
    try:
        with unlock_record.open("x") as stream:
            stream.write(json.dumps(unlock_payload, indent=2) + "\n")
    except FileExistsError:
        existing_unlock = json.loads(unlock_record.read_text())
        if existing_unlock.get("authorization_sha256") != authorization_sha256:
            raise RuntimeError("existing first-unlock record belongs to another authorization")
    attempts = early / "TARGET_SCORER_ATTEMPT_LEDGER.jsonl"
    with attempts.open("a") as stream:
        stream.write(json.dumps({
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "pid": int(os.getpid()),
            "authorization_sha256": authorization_sha256,
            "first_unlock_record_sha256": sha256_file(unlock_record),
        }) + "\n")
    return unlock_record


def spatial_null_interaction(
    J: pd.Series,
    observed_delta: pd.Series,
    l3_null_by_subject: dict[str, np.ndarray],
    l2m_null_by_subject: dict[str, np.ndarray],
) -> dict:
    """Coherent patient-level J interaction under synchronized spatial nulls."""
    subjects = [
        str(subject) for subject in observed_delta.index
        if str(subject) in J.index
        and str(subject) in l3_null_by_subject
        and str(subject) in l2m_null_by_subject
    ]
    if len(subjects) < 5:
        return {"status": "NOT_IDENTIFIABLE", "n": len(subjects)}
    j_values = J.loc[subjects].to_numpy(float)
    delta_values = observed_delta.loc[subjects].to_numpy(float)
    observed_rho = float(spearmanr(j_values, delta_values).statistic)
    null_matrix = np.stack([
        np.asarray(l3_null_by_subject[subject], float)
        - np.asarray(l2m_null_by_subject[subject], float)
        for subject in subjects
    ])
    rho_null = np.asarray([
        spearmanr(j_values, null_matrix[:, draw]).statistic
        for draw in range(null_matrix.shape[1])
    ], dtype=float)
    p_value, finite_draws = empirical_null_p(observed_rho, rho_null)
    return {
        "status": "IDENTIFIABLE" if np.isfinite(p_value) else "NOT_IDENTIFIABLE",
        "n": len(subjects),
        "spearman_rho": observed_rho,
        "spatial_null_p_greater": p_value,
        "finite_spatial_null_draws": finite_draws,
        "spatial_null_rho_median": (
            float(np.nanmedian(rho_null)) if finite_draws else float("nan")
        ),
        "contract": (
            "SYNCHRONIZED_MAXAB_WITHIN_SEIZURE_THEN_PATIENT_MEDIAN_"
            "THEN_J_BY_L3_MINUS_L2M"
        ),
    }


def primary_raw_delta(patient: pd.DataFrame, endpoint: str, l3: str, l2m: str) -> pd.Series:
    """Registered C_L3-C_L2m from raw signed oracle correspondence."""
    table = patient[patient.endpoint == endpoint].pivot(
        index="subject", columns="condition", values="observed"
    )
    return table[l3] - table[l2m]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out_root.resolve()
    authorization_path = out / "TARGET_UNSEAL_AUTHORIZATION.json"
    authorization = json.loads(authorization_path.read_text())
    if not authorization.get("authorized"):
        raise RuntimeError("target access is not authorized")
    scorer_status = scorer_authorization_status(out, authorization, Path(__file__).resolve())
    for relative, digest in authorization["frozen_hashes"].items():
        if sha256_file(out / relative) != digest:
            raise RuntimeError(f"frozen artifact changed after authorization: {relative}")
    null_manifest = out / "NULL_INDEX_MAP_MANIFEST.csv"
    if sha256_file(null_manifest) != authorization["null_manifest_sha256"]:
        raise RuntimeError("synchronized null manifest changed")
    verified_payloads = verify_frozen_payload_manifests(out)
    routing = pd.read_csv(out / "EARLY_ICTAL_ROUTING_METADATA.csv")
    if routing.subject.nunique() != 17 or len(routing) != 167:
        raise RuntimeError("locked target denominator changed")
    early = out / "early_ictal"; early.mkdir(exist_ok=True)
    record_target_unlock(early, authorization_path, verified_payloads)
    census = pd.read_csv(out / "FULL_PARENT_FIT_CENSUS.csv")
    plane_fit_by_subject = census.groupby("subject", sort=False).first().fit_id.to_dict()
    rows, nulls = [], {}
    target_vectors: dict[str, list[np.ndarray]] = {}
    target_contacts: dict[str, list[str]] = {}
    for event in routing.itertuples():
        intact = out / "model_fields/intact/per_patient" / event.subject / "L3_LOCAL_PLUS_LEARNED_LR.npz"
        with np.load(intact, allow_pickle=False) as data:
            contacts = data["contacts"].astype(str).tolist()
        target = load_broadband(event.subject, int(event.seizure_idx), contacts)
        if not np.isfinite(target).all() or len(target) < 6:
            raise RuntimeError(f"exact broadband join failed: {event.subject} {event.seizure_idx}")
        if event.subject in target_contacts and target_contacts[event.subject] != contacts:
            raise RuntimeError(f"target contact order changed inside patient: {event.subject}")
        target_contacts[event.subject] = contacts
        target_vectors.setdefault(event.subject, []).append(target.copy())
        null_path = out / "null_maps" / f"{event.subject}__seizure{int(event.seizure_idx)}.npz"
        with np.load(null_path, allow_pickle=False) as null_map:
            if null_map["contacts"].astype(str).tolist() != contacts:
                raise RuntimeError("null map contact order changed")
            permutations_all = null_map["all_contact"].copy()
            permutations_shaft = null_map["within_shaft"].copy()
            permutations_distance = null_map["distance_bin"].copy()
            spectral_eigenvectors = null_map["spectral_eigenvectors"].copy()
            spectral_signs = null_map["spectral_signs"].copy()
            variogram_normals = null_map["variogram_normals"].copy()
            null_xy = null_map["contact_xy_mm"].copy()
        plane_fit = plane_fit_by_subject[event.subject]
        provenance = json.loads((out / "cache" / plane_fit / "provenance.json").read_text())
        if list(map(str, provenance["joint_contacts"])) != contacts:
            raise RuntimeError(f"plane/contact order mismatch for {event.subject}")
        plane = np.load(out / "cache" / plane_fit / "plane.npz", allow_pickle=False)
        xy = plane["contacts_xy_mm"]
        if not np.allclose(null_xy, xy, rtol=0, atol=1e-5):
            raise RuntimeError(f"null geometry changed: {event.subject}")
        spectral_targets = (
            spectral_surrogates(target, spectral_eigenvectors, spectral_signs)
            if len(spectral_signs) else np.empty((0, len(target)))
        )
        if len(variogram_normals):
            variogram_targets, variogram_range_mm = variogram_surrogates(
                target, xy, variogram_normals
            )
        else:
            variogram_targets = np.empty((0, len(target)))
            variogram_range_mm = float("nan")
        for endpoint in ENDPOINTS:
            for condition, candidate in load_candidates(out, event.subject, endpoint, contacts).items():
                score, all_null = score_candidate(candidate, target, permutations_all)
                if len(permutations_shaft):
                    shaft_score, shaft_null = score_candidate(candidate, target, permutations_shaft)
                    shaft_null_median = float(np.nanmedian(shaft_null))
                    shaft_margin = shaft_score["observed"] - shaft_null_median
                else:
                    shaft_null_median = float("nan")
                    shaft_margin = float("nan")
                if len(permutations_distance):
                    _, distance_null = score_candidate(candidate, target, permutations_distance)
                    distance_null_median = float(np.nanmedian(distance_null))
                    distance_margin = score["observed"] - distance_null_median
                else:
                    distance_null_median = float("nan")
                    distance_margin = float("nan")
                if len(spectral_targets):
                    spectral_null = score_candidate_surrogates(candidate, spectral_targets)
                    spectral_null_median = float(np.nanmedian(spectral_null))
                    spectral_margin = score["observed"] - spectral_null_median
                else:
                    spectral_null_median = float("nan")
                    spectral_margin = float("nan")
                if len(variogram_targets):
                    variogram_null = score_candidate_surrogates(candidate, variogram_targets)
                    variogram_null_median = float(np.nanmedian(variogram_null))
                    variogram_margin = score["observed"] - variogram_null_median
                else:
                    variogram_null_median = float("nan")
                    variogram_margin = float("nan")
                chosen = (
                    candidate[score["selected_mode"].lower()]
                    if score["identifiable"]
                    else np.full_like(target, np.nan)
                )
                rows.append({
                    "dataset": event.dataset, "subject": event.subject,
                    "seizure_idx": int(event.seizure_idx), "condition": condition,
                    "family": candidate["family"], "arm": candidate["arm"],
                    "target": candidate["target"], "alpha": candidate["alpha"],
                    "endpoint": endpoint, "n_contacts": len(target), **score,
                    "all_contact_null_median": float(np.nanmedian(all_null)),
                    "all_contact_margin": score["observed"] - float(np.nanmedian(all_null)),
                    "within_shaft_null_median": shaft_null_median,
                    "within_shaft_margin": shaft_margin,
                    "distance_bin_null_median": distance_null_median,
                    "distance_bin_margin": distance_margin,
                    "spectral_null_median": spectral_null_median,
                    "spectral_margin": spectral_margin,
                    "variogram_null_median": variogram_null_median,
                    "variogram_margin": variogram_margin,
                    "variogram_fitted_range_mm": variogram_range_mm,
                    "peak_contact_distance_mm": (
                        tied_peak_distance(chosen, target, xy)
                        if score["identifiable"] else float("nan")
                    ),
                    "spatial_sinkhorn_normalized": (
                        sinkhorn_distance(chosen, target, xy)
                        if score["identifiable"] else float("nan")
                    ),
                    "null_key": f"{event.subject}|{event.seizure_idx}|{condition}|{endpoint}",
                })
                nulls[rows[-1]["null_key"]] = all_null
    seizure = pd.DataFrame(rows)
    inventory_failures = []
    inventory_rows = []
    expected_seizure_rows = 0
    for keys, group in seizure.groupby(
        ["subject", "seizure_idx", "endpoint"], sort=False
    ):
        available_attenuation = available_attenuation_conditions(out, str(keys[0]))
        expected = expected_condition_inventory(str(keys[2]), available_attenuation)
        expected_seizure_rows += len(expected)
        observed_conditions = set(group.condition.astype(str))
        omitted = sorted(
            {
                f"ATTEN|{target}|{alpha:.2f}"
                for target in ATTENUATION_TARGETS
                for alpha in (0.25, 0.50, 0.75, 1.00)
            } - available_attenuation
        )
        inventory_rows.append({
            "subject": str(keys[0]), "seizure_idx": int(keys[1]),
            "endpoint": str(keys[2]), "expected_conditions": len(expected),
            "observed_conditions": len(group),
            "omitted_unidentifiable_attenuation": omitted,
        })
        if observed_conditions != expected or len(group) != len(expected):
            inventory_failures.append({
                "subject": str(keys[0]), "seizure_idx": int(keys[1]),
                "endpoint": str(keys[2]),
                "missing": sorted(expected - observed_conditions),
                "unexpected": sorted(observed_conditions - expected),
                "rows": int(len(group)),
            })
    if inventory_failures or len(seizure) != expected_seizure_rows:
        raise RuntimeError(
            f"early-ictal condition inventory incomplete: rows={len(seizure)}, "
            f"failures={inventory_failures[:3]}"
        )
    inventory_subject_rows = {
        (str(row.subject), endpoint): len(expected_condition_inventory(
            endpoint, available_attenuation_conditions(out, str(row.subject))
        ))
        for row in routing[["subject"]].drop_duplicates().itertuples(index=False)
        for endpoint in ENDPOINTS
    }
    expected_patient_rows = int(sum(inventory_subject_rows.values()))
    (early / "EARLY_ICTAL_CONDITION_INVENTORY.json").write_text(json.dumps({
        "contract": "SUBJECT_SPECIFIC_PREFROZEN_FIELD_INVENTORY",
        "scorer_authorization_status": scorer_status,
        "expected_per_seizure_rows": expected_seizure_rows,
        "expected_per_patient_rows": expected_patient_rows,
        "omitted_conditions_are_prefrozen_nonidentifiable_only": True,
        "rows": inventory_rows,
    }, indent=2) + "\n")
    seizure.to_csv(early / "EARLY_ICTAL_PER_SEIZURE.csv", index=False)
    target_root = early / "per_patient_targets"; target_root.mkdir(exist_ok=True)
    target_manifest_rows = []
    for subject, vectors in target_vectors.items():
        destination = target_root / f"{subject}.npz"
        np.savez_compressed(
            destination, contacts=np.asarray(target_contacts[subject], dtype="U64"),
            median_broadband_energy=np.nanmedian(np.stack(vectors), axis=0),
            all_seizure_broadband_energy=np.stack(vectors),
            n_seizures=np.asarray(len(vectors), dtype=np.int32),
            time_window_s=np.asarray([0.0, 10.0], dtype=np.float32),
            frequency_band_hz=np.asarray([1.0, 150.0], dtype=np.float32),
        )
        target_manifest_rows.append({
            "subject": subject, "n_seizures": len(vectors),
            "n_contacts": len(target_contacts[subject]), "path": str(destination),
            "sha256": sha256_file(destination),
        })
    pd.DataFrame(target_manifest_rows).to_csv(early / "EARLY_ICTAL_TARGET_MANIFEST.csv", index=False)
    patient_rows = []
    patient_nulls: dict[tuple[str, str, str], np.ndarray] = {}
    for key, group in seizure.groupby(["subject", "condition", "endpoint"], sort=False):
        subject, condition, endpoint = key
        folded_null = np.nanmedian(np.stack([nulls[value] for value in group.null_key]), axis=0)
        observed = float(np.nanmedian(group.observed))
        first = group.iloc[0]
        patient_nulls[(str(subject), str(condition), str(endpoint))] = folded_null
        all_contact_p, finite_all_contact_null = empirical_null_p(observed, folded_null)
        patient_rows.append({
            "subject": subject, "condition": condition, "endpoint": endpoint,
            "family": first.family, "arm": first.arm, "target": first.target,
            "alpha": first.alpha, "n_seizures": len(group),
            "n_contacts": int(group.n_contacts.min()), "observed": observed,
            "all_contact_null_median": float(np.nanmedian(folded_null)),
            "all_contact_margin": observed - float(np.nanmedian(folded_null)),
            "all_contact_p": all_contact_p,
            "all_contact_null_finite_draws": finite_all_contact_null,
            "within_shaft_margin": float(np.nanmedian(group.within_shaft_margin)),
            "distance_bin_margin": float(np.nanmedian(group.distance_bin_margin)),
            "spectral_margin": float(np.nanmedian(group.spectral_margin)),
            "variogram_margin": float(np.nanmedian(group.variogram_margin)),
            "variogram_fitted_range_mm": float(np.nanmedian(group.variogram_fitted_range_mm)),
            "rank_weighted_concordance": float(np.nanmedian(group.rank_weighted_concordance)),
            "top20_jaccard": float(np.nanmedian(group.top20_jaccard)),
            "peak_contact_distance_mm": float(np.nanmedian(group.peak_contact_distance_mm)),
            "spatial_sinkhorn_normalized": float(np.nanmedian(group.spatial_sinkhorn_normalized)),
        })
    patient = pd.DataFrame(patient_rows)
    if len(patient) != expected_patient_rows or patient.subject.nunique() != 17:
        raise RuntimeError(
            f"patient-level condition inventory incomplete: rows={len(patient)}"
        )
    patient.to_csv(early / "EARLY_ICTAL_PER_PATIENT.csv", index=False)
    canonical_observed = patient[patient.endpoint == "canonical_full"].pivot(
        index="subject", columns="condition", values="observed"
    )
    canonical_margin = patient[patient.endpoint == "canonical_full"].pivot(
        index="subject", columns="condition", values="all_contact_margin"
    )
    seed_removed_observed = patient[patient.endpoint == "seed_removed"].pivot(
        index="subject", columns="condition", values="observed"
    )
    seed_removed_margin = patient[patient.endpoint == "seed_removed"].pivot(
        index="subject", columns="condition", values="all_contact_margin"
    )
    l3_arm = "L3_LOCAL_PLUS_LEARNED_LR"
    l2m_arm = "L2M_MACRO_MATCHED_RANDOM_LR"
    l3, l2m = f"INTACT|{l3_arm}", f"INTACT|{l2m_arm}"
    J = pd.read_csv(out / "CROSSFIT_NONLOCALITY_PATIENT_SUMMARY.csv").set_index("subject")
    subjects = canonical_observed.index.intersection(J.index)
    # C^EI is the raw signed best-mode Spearman correspondence.  The
    # synchronized channel-shuffle distribution is a separate spatial null;
    # subtracting arm-specific null medians here would change the registered
    # L3-L2m interaction estimand.
    delta = primary_raw_delta(patient, "canonical_full", l3, l2m).loc[subjects]
    interaction_summary = interaction(J.loc[subjects, "J_lat_exceedance_burden"], delta)
    l3_spatial_null = {
        str(subject): patient_nulls[(str(subject), l3, "canonical_full")]
        for subject in subjects
    }
    l2m_spatial_null = {
        str(subject): patient_nulls[(str(subject), l2m, "canonical_full")]
        for subject in subjects
    }
    spatial_interaction = spatial_null_interaction(
        J["J_lat_exceedance_burden"], delta, l3_spatial_null, l2m_spatial_null
    )
    interaction_summary["spatial_null"] = spatial_interaction
    interaction_summary["joint_primary_p_greater"] = (
        float(max(
            interaction_summary["permutation_p_greater"],
            spatial_interaction["spatial_null_p_greater"],
        ))
        if np.isfinite(spatial_interaction.get("spatial_null_p_greater", np.nan))
        else float("nan")
    )
    interaction_summary["joint_primary_contract"] = (
        "BOTH_PATIENT_LABEL_AND_SYNCHRONIZED_SPATIAL_NULL_MUST_PASS"
    )
    spatial_delta_rows = []
    for draw in range(len(next(iter(l3_spatial_null.values())))):
        null_delta = np.asarray([
            l3_spatial_null[str(subject)][draw]
            - l2m_spatial_null[str(subject)][draw]
            for subject in subjects
        ])
        spatial_delta_rows.append({
            "draw": draw,
            "rho_J_by_L3_minus_L2m_null": float(spearmanr(
                J.loc[subjects, "J_lat_exceedance_burden"], null_delta
            ).statistic),
        })
    pd.DataFrame(spatial_delta_rows).to_csv(
        early / "PRIMARY_SYNCHRONIZED_SPATIAL_NULL_INTERACTION.csv", index=False
    )
    contact_count = patient.groupby("subject").n_contacts.min()
    no_small = subjects[contact_count.loc[subjects].to_numpy() > 7]
    interaction_no_small = interaction(
        J.loc[no_small, "J_lat_exceedance_burden"], delta.loc[no_small], seed=20260814
    )
    maximum_j_subject = str(J.loc[subjects, "J_lat_exceedance_burden"].idxmax())
    no_maximum_j = subjects[subjects != maximum_j_subject]
    interaction_without_maximum_j = interaction(
        J.loc[no_maximum_j, "J_lat_exceedance_burden"], delta.loc[no_maximum_j], seed=20260815
    )
    geometry_2d = census.groupby("subject").geometry_class.apply(
        lambda values: bool(np.all(values == "TWO_DIMENSIONAL"))
    )
    two_d = subjects[geometry_2d.loc[subjects].to_numpy()]
    interaction_2d = interaction(
        J.loc[two_d, "J_lat_exceedance_burden"], delta.loc[two_d], seed=20260816
    )
    attenuation_auc_rows = []
    for endpoint, table in (
        ("canonical_full", canonical_observed),
        ("seed_removed", seed_removed_observed),
    ):
        for target in ATTENUATION_TARGETS:
            columns = [f"ATTEN|{target}|{alpha:.2f}" for alpha in (0.25, 0.50, 0.75, 1.00)]
            if not set(columns).issubset(table.columns):
                continue
            intact_condition = {
                "L1_ADDED": "INTACT|L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
                "L2M_ADDED": "INTACT|L2M_MACRO_MATCHED_RANDOM_LR",
                "L3_ADDED": "INTACT|L3_LOCAL_PLUS_LEARNED_LR",
                "L3_MATCHED_LOCAL": "INTACT|L3_LOCAL_PLUS_LEARNED_LR",
            }[target]
            for subject in table.index:
                x = np.asarray([0.0, 0.25, 0.50, 0.75, 1.00])
                intact_value = float(table.loc[subject, intact_condition])
                attenuated_values = np.asarray(
                    [float(table.loc[subject, column]) for column in columns], float
                )
                if not np.isfinite(intact_value) or not np.isfinite(attenuated_values).all():
                    continue
                damage = np.asarray([0.0] + [
                    intact_value - value for value in attenuated_values
                ])
                attenuation_auc_rows.append({
                    "subject": subject, "endpoint": endpoint, "target": target,
                    "concordance_damage_auc": float(np.trapz(damage, x=x)),
                    "alpha1_damage": float(damage[-1]),
                })
    attenuation_auc = pd.DataFrame(attenuation_auc_rows)
    attenuation_auc.to_csv(early / "EARLY_ICTAL_ATTENUATION_AUC.csv", index=False)
    d2_auc = attenuation_auc.loc[
        (attenuation_auc.endpoint == "seed_removed") &
        (attenuation_auc.target == "L3_ADDED"), "concordance_damage_auc"
    ].to_numpy()
    summary = {
        "contract": "topic5_multiscale_early_ictal_v0_5",
        "status": "LOCKED_INTERNAL_MECHANISTIC_FOLLOWUP_NOT_INDEPENDENT_CONFIRMATION",
        "target": "clinical onset 0-10 s, 1-150 Hz broadband energy",
        "patients": int(patient.subject.nunique()), "seizures": len(routing),
        "condition_inventory": {
            "per_seizure_rows": int(len(seizure)),
            "per_patient_rows": int(len(patient)),
            "subject_specific_prefrozen_availability": True,
        },
        "scorer_authorization_status": scorer_status,
        "primary_endpoint": "signed best-mode Spearman oracle repertoire coverage",
        "primary_null": (
            "joint patient-label permutation and synchronized all-contact "
            "spatial-null interaction; both must pass"
        ),
        "primary_delta_contract": "raw_signed_oracle_L3_minus_L2m; null margin separate",
        "primary_interaction": interaction_summary,
        "primary_interaction_sensitivities": {
            "exclude_6_7_contact_patients": interaction_no_small,
            "excluded_6_7_contact_patients": sorted(set(subjects) - set(no_small)),
            "leave_out_highest_J_patient": interaction_without_maximum_j,
            "highest_J_patient": maximum_j_subject,
            "two_dimensional_geometry_only": interaction_2d,
        },
        "D1_L3_full_margin_gt_zero": paired_summary(canonical_margin[l3]),
        "D2_L3_minus_L2m_seed_removed_signed_oracle": paired_summary(
            seed_removed_observed[l3] - seed_removed_observed[l2m]
        ),
        "D2_L3_minus_L2m_seed_removed_null_relative_sensitivity": paired_summary(
            seed_removed_margin[l3] - seed_removed_margin[l2m]
        ),
        "D2_L3_added_attenuation_auc_seed_removed_gt_zero": paired_summary(d2_auc),
        "nonoracle_L3_mixture_margin_gt_zero": paired_summary(
            canonical_margin[f"INTACT_MIXTURE|{l3_arm}"]
        ),
        "nonoracle_L3_minus_L2m_mixture_signed": paired_summary(
            canonical_observed[f"INTACT_MIXTURE|{l3_arm}"]
            - canonical_observed[f"INTACT_MIXTURE|{l2m_arm}"]
        ),
        "nonoracle_L3_minus_template_mixture_signed": paired_summary(
            canonical_observed[f"INTACT_MIXTURE|{l3_arm}"]
            - canonical_observed["TEMPLATE|TRAIN_MIXTURE"]
        ),
        "gain_adjusted_L3_minus_L2m_full_signed_oracle": paired_summary(
            canonical_observed[f"GAIN_MATCHED|{l3_arm}"]
            - canonical_observed[f"GAIN_MATCHED|{l2m_arm}"]
        ),
        "L3_minus_suffix_full_signed_oracle": paired_summary(
            canonical_observed[l3] - canonical_observed["INTACT|C_L3_ORDER_SHUFFLED"]
        ),
        "L3_minus_template_oracle_full_signed": paired_summary(
            canonical_observed[l3] - canonical_observed["TEMPLATE|ORACLE"]
        ),
        "template_oracle_margin_gt_zero": paired_summary(canonical_margin["TEMPLATE|ORACLE"]),
        "template_mixture_margin_gt_zero": paired_summary(
            canonical_margin["TEMPLATE|TRAIN_MIXTURE"]
        ),
        "target_values_read": True,
    }
    (early / "EARLY_ICTAL_V0_5_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "TARGET_ACCESS_AUDIT.json").write_text(json.dumps({
        "target_values_read": True, "training_or_model_selection_after_unseal": False,
        "model_or_field_generation_after_unseal": False,
        "scorer_authorization_status": scorer_status,
        "patients": 17, "seizures": 167,
        "target": "clinical onset 0-10 s, 1-150 Hz broadband energy",
        "primary_null": (
            "joint patient-label permutation and synchronized all-contact "
            "spatial-null interaction; both must pass"
        ),
    }, indent=2) + "\n")
    (out / "EARLY_ICTAL_SCORING_COMPLETE.json").write_text(json.dumps({
        "status": "PASS", "patients": 17, "seizures": 167, "target_values_read": True,
        "scorer_authorization_status": scorer_status,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
