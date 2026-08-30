"""Risk-set construction and low-capacity seizure probes for H2b.

The module deliberately consumes tables rather than a particular state extractor.
Every row is one candidate anchor in one seizure-specific risk set.  The state,
memoryless code, wrong-time state, explicit observations, and history covariates are
columns with frozen prefixes; seizure labels are used only by the ridge probe.

The estimator is an intercept-free conditional logistic model.  Within every risk
set its scores are normalised with a softmax, so a seizure (one case) rather than a
control window is the statistical unit.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.metrics import average_precision_score, roc_auc_score

from .contract import PRIMARY_LEAD_MINUTES, PROBE_ARMS, support_tier


FEATURE_PREFIXES: Mapping[str, tuple[str, ...]] = {
    "B_history": ("history__",),
    "B_observation": ("history__", "observation__"),
    "B_state": ("history__", "observation__", "state__"),
    "memoryless": ("history__", "observation__", "memoryless__"),
    "wrong_time": ("history__", "observation__", "wrong_time__"),
}

RISK_REQUIRED_COLUMNS = (
    "patient_id",
    "seed",
    "seizure_id",
    "risk_set_id",
    "lead_minutes",
    "split",
    "evaluation_tier",
    "anchor_id",
    "anchor_time",
    "seizure_onset",
    "segment_id",
    "segment_start",
    "segment_end",
    "is_case",
    "horizon_seizure_free",
    "in_ictal_or_postictal",
    "observation_available",
    "observation_signature",
    "wrong_time_donor_valid",
    "wrong_time_same_segment",
    "wrong_time_exclusion_clear",
)


@dataclass(frozen=True)
class ConditionalRidgeModel:
    """A standardised, intercept-free conditional logistic model."""

    coefficient: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    l2: float
    converged: bool
    n_iterations: int

    def score(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=float)
        return ((x - self.center) / self.scale) @ self.coefficient


@dataclass(frozen=True)
class ProbeRunResult:
    per_seed: pd.DataFrame
    patient_medians: pd.DataFrame
    audit: dict[str, Any]


def _require_columns(frame: pd.DataFrame, names: Iterable[str], label: str) -> None:
    missing = sorted(set(names).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def feature_columns(frame: pd.DataFrame, arm: str) -> list[str]:
    if arm not in FEATURE_PREFIXES:
        raise ValueError(f"unknown probe arm {arm!r}")
    columns: list[str] = []
    for prefix in FEATURE_PREFIXES[arm]:
        matched = sorted(name for name in frame.columns if name.startswith(prefix))
        if not matched:
            raise ValueError(f"arm {arm} has no columns with required prefix {prefix!r}")
        columns.extend(matched)
    if len(columns) != len(set(columns)):
        raise ValueError(f"arm {arm} resolved duplicate feature columns")
    return columns


def risk_set_hash(frame: pd.DataFrame) -> str:
    """Hash the feature-independent rows shared by every comparison arm."""
    columns = [
        "patient_id", "seed", "seizure_id", "risk_set_id", "lead_minutes",
        "split", "anchor_id", "anchor_time", "is_case",
    ]
    _require_columns(frame, columns, "risk-set table")
    rows = (
        frame[columns]
        .sort_values(
            ["patient_id", "seed", "risk_set_id", "anchor_time", "anchor_id"],
            kind="mergesort",
        )
        .to_dict(orient="records")
    )
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def chronological_split_map(seizures: pd.DataFrame) -> tuple[dict[str, str], str]:
    """Assign one patient's eligible seizures without looking at probe features."""
    _require_columns(seizures, ("seizure_id", "onset_time"), "seizure table")
    ordered = (
        seizures[["seizure_id", "onset_time"]]
        .drop_duplicates("seizure_id")
        .sort_values(["onset_time", "seizure_id"], kind="mergesort")
    )
    ids = ordered["seizure_id"].astype(str).tolist()
    tier = support_tier(len(ids))
    if tier == "primary_chronological":
        n_train = int(np.floor(0.60 * len(ids)))
        n_select = int(np.floor(0.20 * len(ids)))
        n_train = max(1, n_train)
        n_select = max(1, n_select)
        if n_train + n_select >= len(ids):
            n_select = 1
            n_train = len(ids) - 2
        mapping = {
            seizure_id: (
                "TRAIN" if index < n_train else
                "SELECT" if index < n_train + n_select else
                "TEST"
            )
            for index, seizure_id in enumerate(ids)
        }
    elif tier == "sensitivity_loso":
        mapping = {seizure_id: "LOSO" for seizure_id in ids}
    elif tier == "descriptive_case_series":
        mapping = {seizure_id: "DESCRIPTIVE" for seizure_id in ids}
    else:
        mapping = {seizure_id: "NOT_ESTIMABLE" for seizure_id in ids}
    return mapping, tier


def _nearest_case_anchor(
    anchors: pd.DataFrame,
    target_time: float,
    *,
    tolerance_seconds: float,
) -> pd.Series | None:
    if anchors.empty:
        return None
    delta = np.abs(anchors["anchor_time"].to_numpy(dtype=float) - float(target_time))
    index = int(np.argmin(delta))
    if not np.isfinite(delta[index]) or delta[index] > float(tolerance_seconds):
        return None
    return anchors.iloc[index]


def build_risk_sets(
    anchors: pd.DataFrame,
    seizures: pd.DataFrame,
    *,
    lead_minutes: Sequence[int] = (5, 15, 30, 60, 120),
    primary_lead_minutes: int = PRIMARY_LEAD_MINUTES,
    controls_per_case: int = 5,
    case_anchor_tolerance_seconds: float = 1.0,
    random_seed: int = 1729,
    arms: Sequence[str] = PROBE_ARMS,
    require_wrong_time: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build same-patient, same-segment seizure risk sets.

    ``anchors`` is extractor-independent.  Besides the columns named below it must
    carry numeric feature columns using the prefixes in :data:`FEATURE_PREFIXES`.
    ``horizon_seizure_free`` is recomputed here; callers cannot assert it by fiat.

    Patient support and the split contract are frozen from seizures having an exact
    primary-lead case anchor.  Those same seizure IDs are then used for every lead,
    preventing a favourable sensitivity lead from changing the evidence tier.
    """
    anchor_required = (
        "patient_id", "seed", "anchor_id", "anchor_time", "segment_id",
        "segment_start", "segment_end", "observation_available",
        "observation_signature", "in_ictal_or_postictal",
        "wrong_time_donor_valid", "wrong_time_same_segment",
        "wrong_time_exclusion_clear",
    )
    seizure_required = ("patient_id", "seizure_id", "onset_time", "segment_id")
    _require_columns(anchors, anchor_required, "anchor table")
    _require_columns(seizures, seizure_required, "seizure table")
    if int(controls_per_case) < 1:
        raise ValueError("controls_per_case must be positive")
    if anchors["anchor_time"].dtype == np.float32 or seizures["onset_time"].dtype == np.float32:
        raise ValueError("absolute timestamps must not be stored as float32")
    for column in (
        "observation_available", "in_ictal_or_postictal",
        "wrong_time_donor_valid", "wrong_time_same_segment",
        "wrong_time_exclusion_clear",
    ):
        if not pd.api.types.is_bool_dtype(anchors[column]):
            raise ValueError(f"anchor column {column!r} must have boolean dtype")
    if anchors.duplicated(["patient_id", "seed", "anchor_id"]).any():
        raise ValueError("anchor_id must be unique within patient and seed")
    for arm in arms:
        feature_columns(anchors, arm)
    if require_wrong_time != ("wrong_time" in arms):
        raise ValueError("require_wrong_time must match whether the wrong_time arm is run")

    leads = tuple(int(value) for value in lead_minutes)
    if int(primary_lead_minutes) not in leads:
        raise ValueError("primary lead must be present in lead_minutes")
    rows: list[dict[str, Any]] = []
    exclusions: dict[str, int] = {}
    split_audit: dict[str, Any] = {}

    for patient_id, patient_seizures in seizures.groupby("patient_id", sort=True):
        patient_seizures = patient_seizures.sort_values("onset_time", kind="mergesort")
        patient_anchors = anchors[anchors["patient_id"] == patient_id]
        supported_primary: list[dict[str, Any]] = []
        supported_any: list[dict[str, Any]] = []
        for seizure in patient_seizures.itertuples(index=False):
            candidates = patient_anchors[
                (patient_anchors["segment_id"].astype(str) == str(seizure.segment_id))
                & patient_anchors["observation_available"].astype(bool)
                & ~patient_anchors["in_ictal_or_postictal"].astype(bool)
            ]
            supported_leads = [
                lead for lead in leads if any(
                    _nearest_case_anchor(
                        group, float(seizure.onset_time) - 60.0 * lead,
                        tolerance_seconds=case_anchor_tolerance_seconds,
                    ) is not None
                    for _, group in candidates.groupby("seed", sort=False)
                )
            ]
            if supported_leads:
                supported_any.append({
                    "seizure_id": str(seizure.seizure_id),
                    "onset_time": float(seizure.onset_time),
                })
            declared_primary = bool(getattr(seizure, "primary_30min_supported", True))
            if int(primary_lead_minutes) in supported_leads and declared_primary:
                supported_primary.append({
                    "seizure_id": str(seizure.seizure_id),
                    "onset_time": float(seizure.onset_time),
                })
            elif declared_primary:
                exclusions["missing_primary_case_anchor"] = (
                    exclusions.get("missing_primary_case_anchor", 0) + 1
                )
        primary_frame = pd.DataFrame(
            supported_primary, columns=["seizure_id", "onset_time"]
        )
        support_frame = pd.DataFrame(
            supported_any, columns=["seizure_id", "onset_time"]
        )
        tier = support_tier(len(primary_frame))
        # The 30-min primary support set fixes the patient denominator and the
        # seizure identities for every lead.  Sensitivity leads may become
        # non-estimable, but may not recruit extra seizures that lack the
        # primary anchor and thereby change the scientific population.
        if tier == "primary_chronological":
            split_map, _ = chronological_split_map(primary_frame)
        else:
            split_label = {
                "sensitivity_loso": "LOSO",
                "descriptive_case_series": "DESCRIPTIVE",
                "not_estimable": "NOT_ESTIMABLE",
            }[tier]
            split_map = {
                str(row.seizure_id): split_label
                for row in primary_frame.itertuples(index=False)
            }
        split_audit[str(patient_id)] = {
            "evaluation_tier": tier,
            "eligible_primary_seizure_ids": primary_frame.get(
                "seizure_id", pd.Series(dtype=str)
            ).astype(str).tolist(),
            "eligible_any_lead_seizure_ids": support_frame.get(
                "seizure_id", pd.Series(dtype=str)
            ).astype(str).tolist(),
            "split_seizure_ids": {
                split: sorted([sid for sid, value in split_map.items() if value == split])
                for split in sorted(set(split_map.values()))
            },
        }
        if not split_map:
            continue

        eligible_seizures = patient_seizures[
            patient_seizures["seizure_id"].astype(str).isin(split_map)
        ]
        all_onsets = patient_seizures["onset_time"].to_numpy(dtype=float)
        for seed, seed_anchors in patient_anchors.groupby("seed", sort=True):
            case_by_key: dict[tuple[str, int], pd.Series] = {}
            for seizure in eligible_seizures.itertuples(index=False):
                same_segment = seed_anchors[
                    (seed_anchors["segment_id"].astype(str) == str(seizure.segment_id))
                    & seed_anchors["observation_available"].astype(bool)
                    & ~seed_anchors["in_ictal_or_postictal"].astype(bool)
                ]
                for lead in leads:
                    target = float(seizure.onset_time) - 60.0 * lead
                    case = _nearest_case_anchor(
                        same_segment, target,
                        tolerance_seconds=case_anchor_tolerance_seconds,
                    )
                    if case is None or float(case["segment_end"]) < float(seizure.onset_time):
                        exclusions["missing_or_gap_crossing_case_anchor"] = (
                            exclusions.get("missing_or_gap_crossing_case_anchor", 0) + 1
                        )
                        continue
                    case_by_key[(str(seizure.seizure_id), lead)] = case

            all_case_ids = {str(row["anchor_id"]) for row in case_by_key.values()}
            anchor_split_owner: dict[str, str] = {}
            # Protect TEST, then SELECT, from reuse by earlier fitting partitions.
            split_priority = {"TEST": 0, "SELECT": 1, "TRAIN": 2,
                              "LOSO": 0, "DESCRIPTIVE": 0, "NOT_ESTIMABLE": 0}
            jobs = []
            for seizure in eligible_seizures.itertuples(index=False):
                seizure_id = str(seizure.seizure_id)
                for lead in leads:
                    case = case_by_key.get((seizure_id, lead))
                    if case is not None:
                        jobs.append((split_priority[split_map[seizure_id]], float(seizure.onset_time),
                                     seizure, lead, case))
            for _, _, seizure, lead, case in sorted(jobs, key=lambda value: value[:2]):
                seizure_id = str(seizure.seizure_id)
                split = split_map[seizure_id]
                case_id = str(case["anchor_id"])
                owner = anchor_split_owner.get(case_id)
                if owner is not None and owner != split:
                    exclusions["case_anchor_cross_split_conflict"] = (
                        exclusions.get("case_anchor_cross_split_conflict", 0) + 1
                    )
                    continue
                anchor_split_owner[case_id] = split
                horizon_seconds = 60.0 * lead
                candidates = seed_anchors[
                    (seed_anchors["segment_id"].astype(str) == str(seizure.segment_id))
                    & seed_anchors["observation_available"].astype(bool)
                    & ~seed_anchors["in_ictal_or_postictal"].astype(bool)
                    & (seed_anchors["observation_signature"].astype(str)
                       == str(case["observation_signature"]))
                    & (seed_anchors["anchor_time"].astype(float) + horizon_seconds
                       <= seed_anchors["segment_end"].astype(float))
                    & ~seed_anchors["anchor_id"].astype(str).isin(all_case_ids)
                ].copy()
                if not candidates.empty:
                    candidate_times = candidates["anchor_time"].to_numpy(dtype=float)
                    has_future_seizure = np.array([
                        np.any((all_onsets > anchor_time) &
                               (all_onsets <= anchor_time + horizon_seconds))
                        for anchor_time in candidate_times
                    ])
                    candidates = candidates.loc[~has_future_seizure]
                candidates = candidates[
                    candidates["anchor_id"].astype(str).map(
                        lambda anchor_id: anchor_split_owner.get(anchor_id, split) == split
                    )
                ]
                if len(candidates) < int(controls_per_case):
                    exclusions["insufficient_same_segment_controls"] = (
                        exclusions.get("insufficient_same_segment_controls", 0) + 1
                    )
                    continue
                local_seed = (
                    int(random_seed)
                    # Optimizer seeds are repeated readouts of the same patient,
                    # so they must receive the same sampled control anchor IDs.
                    + sum(ord(char) for char in f"{patient_id}|{seizure_id}|{lead}")
                ) % (2**32)
                rng = np.random.default_rng(local_seed)
                selected = candidates.iloc[
                    np.sort(rng.choice(len(candidates), size=int(controls_per_case), replace=False))
                ]
                for control_id in selected["anchor_id"].astype(str):
                    anchor_split_owner.setdefault(control_id, split)
                risk_set_id = f"{patient_id}__seed{seed}__{seizure_id}__lead{lead}m"

                def emit(anchor: pd.Series, is_case: bool) -> dict[str, Any]:
                    output = anchor.to_dict()
                    output.update({
                        "patient_id": str(patient_id),
                        "seed": int(seed),
                        "seizure_id": seizure_id,
                        "risk_set_id": risk_set_id,
                        "lead_minutes": int(lead),
                        "split": split,
                        "evaluation_tier": tier,
                        "anchor_id": str(anchor["anchor_id"]),
                        "anchor_time": float(anchor["anchor_time"]),
                        "seizure_onset": float(seizure.onset_time),
                        "segment_id": str(anchor["segment_id"]),
                        "segment_start": float(anchor["segment_start"]),
                        "segment_end": float(anchor["segment_end"]),
                        "is_case": bool(is_case),
                        "horizon_seizure_free": bool(not is_case),
                        "in_ictal_or_postictal": bool(anchor["in_ictal_or_postictal"]),
                        "observation_available": bool(anchor["observation_available"]),
                        "observation_signature": str(anchor["observation_signature"]),
                        "wrong_time_donor_valid": bool(anchor["wrong_time_donor_valid"]),
                        "wrong_time_same_segment": bool(anchor["wrong_time_same_segment"]),
                        "wrong_time_exclusion_clear": bool(
                            anchor["wrong_time_exclusion_clear"]
                        ),
                    })
                    return output

                rows.append(emit(case, True))
                rows.extend(emit(control, False) for _, control in selected.iterrows())

    frame = pd.DataFrame(rows)
    if not frame.empty:
        validate_risk_table(
            frame,
            case_anchor_tolerance_seconds=case_anchor_tolerance_seconds,
            arms=arms,
            require_wrong_time=require_wrong_time,
        )
    audit = {
        "status": "COMPLETE",
        "identical_risk_sets_across_arms": True,
        "regularization_selected_only_on_train_select": True,
        "seed_is_patient_replicate": False,
        "seed_aggregation": "median_within_patient",
        "control_sampling_shared_across_optimizer_seeds": True,
        "arms": list(arms),
        "wrong_time_required_for_entry": bool(require_wrong_time),
        "primary_support_defines_tier_for_all_leads": True,
        "split_by_patient": split_audit,
        "exclusions": exclusions,
        "n_rows": int(len(frame)),
        "n_risk_sets": int(frame["risk_set_id"].nunique()) if not frame.empty else 0,
        "risk_set_hash": risk_set_hash(frame) if not frame.empty else None,
    }
    return frame, audit


def validate_risk_table(
    frame: pd.DataFrame,
    *,
    case_anchor_tolerance_seconds: float = 1.0,
    arms: Sequence[str] = PROBE_ARMS,
    require_wrong_time: bool = True,
) -> dict[str, Any]:
    """Fail closed on leakage, gap, availability, and table-shape drift."""
    _require_columns(frame, RISK_REQUIRED_COLUMNS, "risk-set table")
    if frame.empty:
        raise ValueError("risk-set table is empty")
    if frame["anchor_time"].dtype == np.float32 or frame["seizure_onset"].dtype == np.float32:
        raise ValueError("absolute timestamps must not be stored as float32")
    for column in (
        "is_case", "horizon_seizure_free", "in_ictal_or_postictal",
        "observation_available", "wrong_time_donor_valid",
        "wrong_time_same_segment", "wrong_time_exclusion_clear",
    ):
        if not pd.api.types.is_bool_dtype(frame[column]):
            raise ValueError(f"risk-set column {column!r} must have boolean dtype")
    if frame[list(RISK_REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("required risk-set fields may not be missing")
    if not frame["observation_available"].astype(bool).all():
        raise ValueError("every case and control must have observation availability")
    if frame["in_ictal_or_postictal"].astype(bool).any():
        raise ValueError("ictal/postictal anchors are forbidden")
    if require_wrong_time != ("wrong_time" in arms):
        raise ValueError("require_wrong_time must match whether the wrong_time arm is run")
    if require_wrong_time:
        for column in (
            "wrong_time_donor_valid", "wrong_time_same_segment",
            "wrong_time_exclusion_clear",
        ):
            if not frame[column].astype(bool).all():
                raise ValueError(f"wrong-time donor contract failed: {column}")
    horizon_end = frame["anchor_time"].to_numpy(dtype=float) + (
        60.0 * frame["lead_minutes"].to_numpy(dtype=float)
    )
    if np.any(horizon_end > frame["segment_end"].to_numpy(dtype=float) + 1e-9):
        raise ValueError("case/control horizon crosses a recorded segment boundary")
    if np.any(frame["anchor_time"].to_numpy(dtype=float) <
              frame["segment_start"].to_numpy(dtype=float) - 1e-9):
        raise ValueError("anchor precedes its recorded segment")
    controls = frame[~frame["is_case"].astype(bool)]
    if not controls["horizon_seizure_free"].astype(bool).all():
        raise ValueError("every control horizon must be seizure-free")
    cases = frame[frame["is_case"].astype(bool)]
    case_error = np.abs(
        cases["anchor_time"].to_numpy(dtype=float)
        - (cases["seizure_onset"].to_numpy(dtype=float)
           - 60.0 * cases["lead_minutes"].to_numpy(dtype=float))
    )
    if np.any(case_error > float(case_anchor_tolerance_seconds)):
        raise ValueError("case anchor is not located at seizure onset minus lead")

    for arm in arms:
        columns = feature_columns(frame, arm)
        values = frame[columns].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"arm {arm} contains non-finite features")

    for risk_set_id, group in frame.groupby("risk_set_id", sort=False):
        if int(group["is_case"].astype(bool).sum()) != 1 or len(group) < 2:
            raise ValueError(f"risk set {risk_set_id!r} must have one case and >=1 control")
        for column in (
            "patient_id", "seed", "seizure_id", "lead_minutes", "split",
            "evaluation_tier", "segment_id", "observation_signature",
        ):
            if group[column].nunique(dropna=False) != 1:
                raise ValueError(f"risk set {risk_set_id!r} mixes {column}")

    # No exact anchor may be a TRAIN/SELECT/TEST member in more than one partition.
    fixed = frame[frame["split"].isin(["TRAIN", "SELECT", "TEST"])]
    leakage = (
        fixed.groupby(["patient_id", "seed", "anchor_id"])["split"].nunique() > 1
    )
    if leakage.any():
        raise ValueError("the same anchor appears across TRAIN/SELECT/TEST")
    time_leakage = (
        fixed.groupby(
            ["patient_id", "seed", "segment_id", "anchor_time"]
        )["split"].nunique() > 1
    )
    if time_leakage.any():
        raise ValueError("the same time point appears across TRAIN/SELECT/TEST")

    seizure_split = (
        frame.groupby(["patient_id", "seizure_id"])["split"].nunique() > 1
    )
    if seizure_split.any():
        raise ValueError("one seizure has different splits across lead times or seeds")
    risk_set_split = frame.groupby("risk_set_id")["split"].nunique() > 1
    if risk_set_split.any():
        raise ValueError("a risk set crosses splits")

    split_ids: dict[str, dict[str, list[str]]] = {}
    for patient_id, patient in frame.groupby("patient_id", sort=True):
        split_ids[str(patient_id)] = {}
        unique = patient[["seizure_id", "split"]].drop_duplicates()
        for split, group in unique.groupby("split", sort=True):
            split_ids[str(patient_id)][str(split)] = sorted(
                group["seizure_id"].astype(str).tolist()
            )
    return {
        "status": "PASS",
        "n_rows": int(len(frame)),
        "n_risk_sets": int(frame["risk_set_id"].nunique()),
        "identical_risk_sets_across_arms": True,
        "train_select_test_anchor_disjoint": True,
        "lead_to_split_consistency": True,
        "wrong_time_donors_same_patient_segment_and_exclusion_clear": (
            True if require_wrong_time else None
        ),
        "wrong_time_required_for_entry": bool(require_wrong_time),
        "arms": list(arms),
        "split_seizure_ids": split_ids,
        "risk_set_hash": risk_set_hash(frame),
    }


def _group_indices(groups: np.ndarray) -> list[np.ndarray]:
    _, inverse = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    return [np.flatnonzero(inverse == value) for value in range(int(inverse.max()) + 1)]


def fit_conditional_ridge(
    values: np.ndarray,
    labels: np.ndarray,
    risk_sets: np.ndarray,
    *,
    l2: float,
    max_iterations: int = 1000,
) -> ConditionalRidgeModel:
    """Fit ridge conditional logistic regression with equal risk-set weight."""
    x = np.asarray(values, dtype=float)
    y = np.asarray(labels, dtype=float)
    groups = np.asarray(risk_sets).astype(str)
    if x.ndim != 2 or len(x) != len(y) or len(y) != len(groups):
        raise ValueError("features, labels, and risk-set IDs must align")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("conditional probe inputs must be finite")
    indices = _group_indices(groups)
    if any(int(y[index].sum()) != 1 or len(index) < 2 for index in indices):
        raise ValueError("each conditional risk set needs exactly one case and one or more controls")
    center = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    z = (x - center) / scale
    penalty = float(l2)
    if penalty < 0:
        raise ValueError("l2 must be non-negative")

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        loss = 0.0
        gradient = np.zeros_like(beta)
        for index in indices:
            score = z[index] @ beta
            probability = np.exp(score - logsumexp(score))
            loss += float(logsumexp(score) - score[np.flatnonzero(y[index] == 1)[0]])
            gradient += z[index].T @ (probability - y[index])
        count = float(len(indices))
        loss = loss / count + 0.5 * penalty * float(beta @ beta)
        gradient = gradient / count + penalty * beta
        return loss, gradient

    fitted = minimize(
        objective,
        np.zeros(z.shape[1], dtype=float),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": int(max_iterations), "ftol": 1e-12, "gtol": 1e-8},
    )
    return ConditionalRidgeModel(
        coefficient=np.asarray(fitted.x, dtype=float),
        center=center,
        scale=scale,
        l2=penalty,
        converged=bool(fitted.success),
        n_iterations=int(fitted.nit),
    )


def conditional_probabilities(scores: np.ndarray, risk_sets: np.ndarray) -> np.ndarray:
    score = np.asarray(scores, dtype=float)
    groups = np.asarray(risk_sets).astype(str)
    probability = np.empty(len(score), dtype=float)
    for index in _group_indices(groups):
        probability[index] = np.exp(score[index] - logsumexp(score[index]))
    return probability


def conditional_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    risk_sets: np.ndarray,
) -> dict[str, float | int]:
    y = np.asarray(labels, dtype=int)
    score = np.asarray(scores, dtype=float)
    groups = np.asarray(risk_sets).astype(str)
    indices = _group_indices(groups)
    probability = conditional_probabilities(score, groups)
    losses, ranks, percentiles = [], [], []
    weights = np.zeros(len(y), dtype=float)
    for index in indices:
        case_local = int(np.flatnonzero(y[index] == 1)[0])
        case_index = int(index[case_local])
        losses.append(float(logsumexp(score[index]) - score[case_index]))
        greater = float(np.sum(score[index] > score[case_index]))
        equal = float(np.sum(score[index] == score[case_index]))
        rank = 1.0 + greater + 0.5 * (equal - 1.0)
        ranks.append(rank)
        percentiles.append((rank - 1.0) / max(1.0, len(index) - 1.0))
        weights[index] = 1.0 / len(index)
    weights /= float(np.sum(weights))
    brier = float(np.sum(weights * (probability - y) ** 2))
    ece = 0.0
    for left, right in zip(np.linspace(0.0, 1.0, 6)[:-1], np.linspace(0.0, 1.0, 6)[1:]):
        selected = (probability >= left) & (
            (probability <= right) if right == 1.0 else (probability < right)
        )
        weight = float(np.sum(weights[selected]))
        if weight:
            ece += weight * abs(
                float(np.average(probability[selected], weights=weights[selected]))
                - float(np.average(y[selected], weights=weights[selected]))
            )
    return {
        "conditional_log_loss": float(np.mean(losses)),
        "risk_set_rank": float(np.mean(ranks)),
        "risk_set_rank_percentile": float(np.mean(percentiles)),
        "auroc": float(roc_auc_score(y, score, sample_weight=weights)),
        "auprc": float(average_precision_score(y, score, sample_weight=weights)),
        "calibration_brier": brier,
        "calibration_ece_5bin": float(ece),
        "n_risk_sets": int(len(indices)),
        "n_rows": int(len(y)),
    }


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: Sequence[str],
    *,
    l2: float,
) -> tuple[ConditionalRidgeModel, np.ndarray]:
    model = fit_conditional_ridge(
        train[list(columns)].to_numpy(dtype=float),
        train["is_case"].to_numpy(dtype=int),
        train["risk_set_id"].to_numpy(),
        l2=float(l2),
    )
    return model, model.score(test[list(columns)].to_numpy(dtype=float))


def _purge_anchor_overlap(train: pd.DataFrame, heldout: pd.DataFrame) -> pd.DataFrame:
    held_anchors = set(heldout["anchor_id"].astype(str))
    held_times = set(zip(
        heldout["segment_id"].astype(str), heldout["anchor_time"].astype(float),
    ))
    bad_ids = set()
    for risk_set_id, group in train.groupby("risk_set_id", sort=False):
        anchor_overlap = bool(held_anchors.intersection(group["anchor_id"].astype(str)))
        time_overlap = bool(held_times.intersection(zip(
            group["segment_id"].astype(str), group["anchor_time"].astype(float),
        )))
        if anchor_overlap or time_overlap:
            bad_ids.add(str(risk_set_id))
    return train[~train["risk_set_id"].astype(str).isin(bad_ids)].copy()


def _select_l2(
    train: pd.DataFrame,
    select: pd.DataFrame,
    columns: Sequence[str],
    ridge_grid: Sequence[float],
) -> float:
    scores = []
    for l2 in ridge_grid:
        _, predicted = _fit_predict(train, select, columns, l2=float(l2))
        metric = conditional_metrics(
            select["is_case"].to_numpy(dtype=int), predicted,
            select["risk_set_id"].to_numpy(),
        )["conditional_log_loss"]
        scores.append((float(metric), -float(l2), float(l2)))
    return min(scores)[2]


def _select_l2_loso(
    train: pd.DataFrame,
    columns: Sequence[str],
    ridge_grid: Sequence[float],
) -> float:
    candidate_ids = train["risk_set_id"].drop_duplicates().astype(str).tolist()
    scores: list[tuple[float, float, float]] = []
    for l2 in ridge_grid:
        fold_losses = []
        for held_id in candidate_ids:
            select = train[train["risk_set_id"].astype(str) == held_id]
            inner = train[train["risk_set_id"].astype(str) != held_id]
            inner = _purge_anchor_overlap(inner, select)
            if inner["risk_set_id"].nunique() < 1:
                continue
            _, predicted = _fit_predict(inner, select, columns, l2=float(l2))
            fold_losses.append(float(conditional_metrics(
                select["is_case"].to_numpy(dtype=int), predicted,
                select["risk_set_id"].to_numpy(),
            )["conditional_log_loss"]))
        if fold_losses:
            scores.append((float(np.mean(fold_losses)), -float(l2), float(l2)))
    if not scores:
        return float(max(ridge_grid))
    return min(scores)[2]


def _evaluate_chronological(
    frame: pd.DataFrame,
    columns: Sequence[str],
    ridge_grid: Sequence[float],
) -> tuple[dict[str, Any], np.ndarray, pd.DataFrame]:
    train = frame[frame["split"] == "TRAIN"]
    select = frame[frame["split"] == "SELECT"]
    test = frame[frame["split"] == "TEST"]
    if min(part["risk_set_id"].nunique() for part in (train, select, test)) < 1:
        raise ValueError("chronological evaluation needs TRAIN, SELECT, and TEST risk sets")
    chosen = _select_l2(train, select, columns, ridge_grid)
    refit = pd.concat([train, select], ignore_index=True)
    model, predicted = _fit_predict(refit, test, columns, l2=chosen)
    return {
        "chosen_l2": chosen,
        "all_fits_converged": bool(model.converged),
        "selection_scope": "TRAIN_fit_SELECT_choose_then_TRAIN_plus_SELECT_refit",
    }, predicted, test


def _evaluate_leave_one_out(
    frame: pd.DataFrame,
    columns: Sequence[str],
    ridge_grid: Sequence[float],
    *,
    tune: bool,
) -> tuple[dict[str, Any], np.ndarray, pd.DataFrame]:
    predictions, held_rows, chosen_values, convergence = [], [], [], []
    risk_ids = frame["risk_set_id"].drop_duplicates().astype(str).tolist()
    for held_id in risk_ids:
        held = frame[frame["risk_set_id"].astype(str) == held_id].copy()
        train = frame[frame["risk_set_id"].astype(str) != held_id].copy()
        train = _purge_anchor_overlap(train, held)
        if train["risk_set_id"].nunique() < 1:
            continue
        chosen = (
            _select_l2_loso(train, columns, ridge_grid)
            if tune else float(max(ridge_grid))
        )
        model, predicted = _fit_predict(train, held, columns, l2=chosen)
        predictions.append(predicted)
        held_rows.append(held)
        chosen_values.append(chosen)
        convergence.append(model.converged)
    if not held_rows:
        raise ValueError("leave-one-seizure-out produced no estimable fold")
    return {
        "chosen_l2": float(np.median(chosen_values)),
        "chosen_l2_by_fold": [float(value) for value in chosen_values],
        "all_fits_converged": bool(all(convergence)),
        "selection_scope": (
            "nested_LOSO_within_probe_training_seizures"
            if tune else "prespecified_strongest_ridge_descriptive_only"
        ),
    }, np.concatenate(predictions), pd.concat(held_rows, ignore_index=True)


def _evaluate_arm(
    frame: pd.DataFrame,
    arm: str,
    ridge_grid: Sequence[float],
) -> dict[str, Any]:
    columns = feature_columns(frame, arm)
    tier = str(frame["evaluation_tier"].iloc[0])
    if tier == "primary_chronological":
        available = set(frame["split"].astype(str))
        if not {"TRAIN", "SELECT", "TEST"}.issubset(available):
            return {
                "status": "NOT_ESTIMABLE_AT_LEAD",
                "reason": "lead lacks one or more frozen chronological partitions",
            }
        fit, prediction, test = _evaluate_chronological(frame, columns, ridge_grid)
    elif tier == "sensitivity_loso":
        if frame["seizure_id"].nunique() < 2:
            return {
                "status": "NOT_ESTIMABLE_AT_LEAD",
                "reason": "fewer than two lead-specific LOSO seizures",
            }
        fit, prediction, test = _evaluate_leave_one_out(
            frame, columns, ridge_grid, tune=True,
        )
    elif tier == "descriptive_case_series":
        if frame["seizure_id"].nunique() < 2:
            return {
                "status": "NOT_ESTIMABLE_AT_LEAD",
                "reason": "fewer than two lead-specific descriptive seizures",
            }
        fit, prediction, test = _evaluate_leave_one_out(
            frame, columns, ridge_grid, tune=False,
        )
    else:
        return {"status": "NOT_ESTIMABLE", "reason": "fewer than two eligible seizures"}
    metrics = conditional_metrics(
        test["is_case"].to_numpy(dtype=int), prediction,
        test["risk_set_id"].to_numpy(),
    )
    return {
        "status": "ok",
        "n_features": int(len(columns)),
        **fit,
        **metrics,
    }


def run_probe_table(
    frame: pd.DataFrame,
    *,
    ridge_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    arms: Sequence[str] = PROBE_ARMS,
    validate: bool = True,
) -> ProbeRunResult:
    """Fit all requested arms and collapse optimizer seeds within patient."""
    if validate:
        table_audit = validate_risk_table(
            frame, arms=arms, require_wrong_time="wrong_time" in arms,
        )
    else:
        table_audit = {"status": "BYPASSED_AFTER_ORIGINAL_TABLE_VALIDATED"}
    ridge = tuple(float(value) for value in ridge_grid)
    if not ridge or any(value < 0 for value in ridge):
        raise ValueError("ridge_grid must contain non-negative values")
    unknown = sorted(set(arms).difference(PROBE_ARMS))
    if unknown:
        raise ValueError(f"unknown probe arms: {unknown}")

    rows: list[dict[str, Any]] = []
    group_columns = ["patient_id", "seed", "lead_minutes", "evaluation_tier"]
    for keys, group in frame.groupby(group_columns, sort=True):
        patient_id, seed, lead, tier = keys
        row: dict[str, Any] = {
            "patient_id": str(patient_id),
            "seed": int(seed),
            "lead_minutes": int(lead),
            "evaluation_tier": str(tier),
            "n_seizures": int(group["seizure_id"].nunique()),
            "n_risk_sets": int(group["risk_set_id"].nunique()),
            "identical_risk_sets_across_arms": True,
            "regularization_selected_only_on_train_select": True,
        }
        arm_results = {}
        for arm in arms:
            result = _evaluate_arm(group, arm, ridge)
            arm_results[arm] = result
            for key, value in result.items():
                if isinstance(value, (str, bool, int, float, np.integer, np.floating)):
                    row[f"{arm}__{key}"] = value
        if {"B_observation", "B_state"}.issubset(arms) and all(
            arm_results[name].get("status") == "ok"
            for name in ("B_observation", "B_state")
        ):
            row["state_minus_observation_conditional_log_loss"] = (
                arm_results["B_state"]["conditional_log_loss"]
                - arm_results["B_observation"]["conditional_log_loss"]
            )
        if all(arm_results.get(name, {}).get("status") == "ok"
               for name in ("B_state", "memoryless") if name in arms) and {
                   "B_state", "memoryless"
               }.issubset(arms):
            row["persistent_minus_memoryless_conditional_log_loss"] = (
                arm_results["B_state"]["conditional_log_loss"]
                - arm_results["memoryless"]["conditional_log_loss"]
            )
        if all(arm_results.get(name, {}).get("status") == "ok"
               for name in ("B_state", "wrong_time") if name in arms) and {
                   "B_state", "wrong_time"
               }.issubset(arms):
            row["correct_minus_wrong_time_conditional_log_loss"] = (
                arm_results["B_state"]["conditional_log_loss"]
                - arm_results["wrong_time"]["conditional_log_loss"]
            )
        rows.append(row)
    per_seed = pd.DataFrame(rows)
    patient_medians = patient_seed_medians(per_seed)
    split_ids = table_audit.get("split_seizure_ids", {})
    audit = {
        "status": "COMPLETE",
        "risk_table": table_audit,
        "identical_risk_sets_across_arms": True,
        "identical_risk_set_hash_across_arms": table_audit.get("risk_set_hash"),
        "train_select_test_seizure_ids": split_ids,
        "lead_to_split_consistency": True,
        "regularization_selected_only_on_train_select": True,
        "wrong_time_donors_same_patient_segment_and_exclusion_clear": (
            True if "wrong_time" in arms else None
        ),
        "wrong_time_confounders_adjusted_in_probe": (
            True if "wrong_time" in arms else None
        ),
        "seed_is_patient_replicate": False,
        "seed_aggregation": "median_within_patient_before_cohort_inference",
        "probe_model": "ridge_conditional_risk_set_softmax",
        "ridge_grid": list(ridge),
        "arms": list(arms),
        "primary_metric": (
            "held-out 30-min conditional log loss B_state-B_observation; negative favours state"
        ),
    }
    return ProbeRunResult(per_seed=per_seed, patient_medians=patient_medians, audit=audit)


def patient_seed_medians(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Take optimizer-seed medians inside patient; never turn seeds into patients."""
    if per_seed.empty:
        return per_seed.copy()
    keys = ["patient_id", "lead_minutes", "evaluation_tier"]
    numeric = [
        column for column in per_seed.columns
        if column not in keys + ["seed"]
        and pd.api.types.is_numeric_dtype(per_seed[column])
        and not pd.api.types.is_bool_dtype(per_seed[column])
    ]
    medians = per_seed.groupby(keys, as_index=False)[numeric].median(numeric_only=True)
    counts = per_seed.groupby(keys, as_index=False)["seed"].nunique().rename(
        columns={"seed": "n_optimizer_seeds"}
    )
    output = medians.merge(counts, on=keys, how="left")
    output["seed_aggregation"] = "median_within_patient"
    output["seed_is_patient_replicate"] = False
    return output


def time_label_permutation_audit(
    frame: pd.DataFrame,
    *,
    n_permutations: int = 100,
    ridge_grid: Sequence[float] = (0.01, 0.1, 1.0, 10.0),
    random_seed: int = 9917,
    primary_lead_minutes: int = PRIMARY_LEAD_MINUTES,
) -> dict[str, Any]:
    """Shuffle the case time inside each patient risk set and refit the probe."""
    validate_risk_table(
        frame, arms=("B_observation", "B_state"), require_wrong_time=False,
    )
    observed_run = run_probe_table(
        frame, ridge_grid=ridge_grid, arms=("B_observation", "B_state"), validate=False,
    )
    observed_rows = observed_run.patient_medians[
        observed_run.patient_medians["lead_minutes"] == int(primary_lead_minutes)
    ]
    observed = float(np.nanmedian(
        observed_rows["state_minus_observation_conditional_log_loss"].to_numpy(dtype=float)
    ))
    rng = np.random.default_rng(int(random_seed))
    null = []
    for _ in range(int(n_permutations)):
        shuffled = frame.reset_index(drop=True).copy()
        shuffled["is_case"] = False
        for _, index in frame.groupby("risk_set_id", sort=False).groups.items():
            chosen = int(rng.choice(np.asarray(list(index), dtype=int)))
            shuffled.loc[chosen, "is_case"] = True
        # The original table has already passed the seizure-free control audit.  In
        # this deliberate label permutation, the original case may become a null
        # control, so the scientific table validator is not reapplied.
        result = run_probe_table(
            shuffled, ridge_grid=ridge_grid,
            arms=("B_observation", "B_state"), validate=False,
        )
        selected = result.patient_medians[
            result.patient_medians["lead_minutes"] == int(primary_lead_minutes)
        ]
        null.append(float(np.nanmedian(
            selected["state_minus_observation_conditional_log_loss"].to_numpy(dtype=float)
        )))
    values = np.asarray(null, dtype=float)
    return {
        "status": "COMPLETE",
        "permutation_unit": "case label within patient-specific risk set",
        "n_permutations": int(n_permutations),
        "observed_state_minus_observation": observed,
        "null_median": float(np.nanmedian(values)),
        "null_mean": float(np.nanmean(values)),
        "null_q025": float(np.nanquantile(values, 0.025)),
        "null_q975": float(np.nanquantile(values, 0.975)),
        "null_values": [float(value) for value in values],
    }


def make_positive_synthetic_risk_table(
    *,
    n_seizures: int = 60,
    n_controls: int = 4,
    n_seeds: int = 1,
    state_strength: float = 4.0,
    random_seed: int = 31415,
) -> pd.DataFrame:
    """Synthetic where a frozen-like persistent state determines seizure choice."""
    if n_seizures < 10:
        raise ValueError("positive synthetic uses the primary chronological tier (>=10 seizures)")
    rng = np.random.default_rng(int(random_seed))
    seizure_ids = [f"sz{index:03d}" for index in range(int(n_seizures))]
    split_map, tier = chronological_split_map(pd.DataFrame({
        "seizure_id": seizure_ids,
        "onset_time": np.arange(n_seizures, dtype=float),
    }))
    rows = []
    for seed in range(int(n_seeds)):
        for index, seizure_id in enumerate(seizure_ids):
            count = int(n_controls) + 1
            state = rng.normal(size=count)
            probability = np.exp(float(state_strength) * state - logsumexp(float(state_strength) * state))
            case_local = int(rng.choice(count, p=probability))
            onset = 1_000_000.0 + index * 20_000.0
            risk_set_id = f"synthetic__seed{seed}__{seizure_id}__lead30m"
            for local in range(count):
                is_case = local == case_local
                anchor_time = onset - 1800.0 if is_case else onset - 10_000.0 - 10.0 * local
                rows.append({
                    "patient_id": "synthetic_patient",
                    "seed": seed,
                    "seizure_id": seizure_id,
                    "risk_set_id": risk_set_id,
                    "lead_minutes": 30,
                    "split": split_map[seizure_id],
                    "evaluation_tier": tier,
                    "anchor_id": f"seed{seed}_{index}_{local}",
                    "anchor_time": np.float64(anchor_time),
                    "seizure_onset": np.float64(onset),
                    "segment_id": "continuous_synthetic_segment",
                    "segment_start": 0.0,
                    "segment_end": 1_000_000.0 + n_seizures * 20_000.0 + 10_000.0,
                    "is_case": is_case,
                    "horizon_seizure_free": not is_case,
                    "in_ictal_or_postictal": False,
                    "observation_available": True,
                    "observation_signature": "complete",
                    "wrong_time_donor_valid": True,
                    "wrong_time_same_segment": True,
                    "wrong_time_exclusion_clear": True,
                    "history__recent_count": rng.normal(),
                    "observation__spectral": rng.normal(),
                    "state__persistent_0": state[local],
                    "memoryless__code_0": rng.normal(),
                    "wrong_time__state_0": rng.normal(),
                })
    frame = pd.DataFrame(rows)
    validate_risk_table(frame)
    return frame
