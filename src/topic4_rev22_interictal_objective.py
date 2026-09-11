"""rev22-DCI vector training endpoint: support / order|support / lag|joint / cover.

Pure functions only. No SNN, no filesystem I/O, no model labels, no KMeans, no OOD
classifier and no held-out input. Every function takes contact onsets in
milliseconds (events x contacts, NaN = not recruited) and the frozen contact
contract groups/pairs.

The decomposition follows spec section 5.1: recruitment information lives in the
support view only; the order view is conditioned on joint recruitment; the lag
view is conditioned on joint recruitment. "Not jointly recruited" therefore never
enters the order or lag distances.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance

from src.topic4_shaft_aware import (
    PAIR_CLASS_ORDER, SHAFT_ORDER, fit_patient_embedding,
    sliced_event_cloud_distance, transform_patient_embedding,
)

N_PAIR_MIN = 5
LAG_CAP_MS = 180.0
TIE_TOLERANCE_MS = 1e-9
ORDER_STATES = ("i_before_j", "j_before_i", "tie")
COMPONENTS = ("D_support", "D_order", "D_lag", "D_cover")
STATUS_OK = "OK"
STATUS_LOW_JOINT = "NOT_ESTIMABLE_LOW_JOINT_SUPPORT"
STATUS_LOW_EVENTS = "NOT_ESTIMABLE_LOW_EVENTS"
MIN_ELIGIBLE_FRACTION = 0.5


# --------------------------------------------------------------------------- #
# representation
# --------------------------------------------------------------------------- #
def _validate_onsets(onsets) -> np.ndarray:
    values = np.asarray(onsets, dtype=float)
    if values.ndim != 2:
        raise ValueError("onsets must be an events x contacts table in ms")
    return values


def relative_onsets(onsets_ms) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mask, physical relative onset l, normalized order u)."""
    values = _validate_onsets(onsets_ms)
    mask = np.isfinite(values)
    any_row = mask.any(axis=1)
    first = np.where(any_row, np.min(np.where(mask, values, np.inf), axis=1), 0.0)
    last = np.where(any_row, np.max(np.where(mask, values, -np.inf), axis=1), 0.0)
    relative = np.where(mask, values - first[:, None], 0.0)
    span = last - first
    normalized = np.where(
        mask & (span[:, None] > 1e-12), relative / np.where(span > 1e-12, span, 1.0)[:, None],
        0.0,
    )
    return mask, relative, normalized


def embedding_features(onsets_ms, groups: Mapping, *, lag_cap_ms: float = LAG_CAP_MS) -> np.ndarray:
    """Spec section 5 embedding vector with the ``1 +`` recruited offset."""
    values = _validate_onsets(onsets_ms)
    mask, relative, normalized = relative_onsets(values)
    order_block = np.where(mask, 1.0 + normalized, 0.0)
    physical_block = np.where(mask, 1.0 + np.clip(relative / float(lag_cap_ms), 0.0, 1.0), 0.0)
    fractions = np.column_stack([
        mask[:, np.asarray(groups[shaft], dtype=int)].mean(axis=1) for shaft in SHAFT_ORDER
    ])
    first = {}
    present = {}
    for shaft in SHAFT_ORDER:
        idx = np.asarray(groups[shaft], dtype=int)
        shaft_values = np.where(mask[:, idx], values[:, idx], np.inf)
        present[shaft] = mask[:, idx].any(axis=1)
        first[shaft] = shaft_values.min(axis=1)
    valid = present["ICL"] & present["SCL"]
    delta = np.zeros(len(values), dtype=float)
    delta[valid] = (first["SCL"][valid] - first["ICL"][valid]) / float(lag_cap_ms)
    return np.column_stack([
        mask.astype(float), order_block, physical_block, fractions,
        delta, valid.astype(float),
    ])


def clipping_fractions(onsets_ms, *, lag_cap_ms: float = LAG_CAP_MS) -> dict:
    values = _validate_onsets(onsets_ms)
    mask, relative, _ = relative_onsets(values)
    clipped = mask & (relative > float(lag_cap_ms))
    n_finite = int(mask.sum())
    return {
        "event_contact_fraction": float(clipped.sum() / n_finite) if n_finite else 0.0,
        "any_event_fraction": float(clipped.any(axis=1).mean()) if len(values) else 0.0,
    }


def fit_training_embedding(features: np.ndarray, *, seed: int, variance_fraction: float = 0.95,
                           max_components: int = 24, reference_n: int = 8192,
                           n_directions: int = 64) -> dict:
    return fit_patient_embedding(
        features, variance_fraction=variance_fraction, max_components=max_components,
        reference_n=reference_n, n_directions=n_directions, seed=seed,
    )


# --------------------------------------------------------------------------- #
# views
# --------------------------------------------------------------------------- #
def support_view(onsets_ms, groups: Mapping, pairs_by_class: Mapping) -> dict:
    values = _validate_onsets(onsets_ms)
    mask = np.isfinite(values)
    n_contacts = values.shape[1]
    counts = mask.sum(axis=1)
    return {
        "n_events": int(len(values)),
        "n_contacts": int(n_contacts),
        "recruitment": {
            shaft: mask[:, np.asarray(groups[shaft], dtype=int)].mean(axis=0)
            if len(values) else np.full(len(groups[shaft]), np.nan)
            for shaft in SHAFT_ORDER
        },
        "count_histogram": (
            np.bincount(counts, minlength=n_contacts + 1) / len(values)
            if len(values) else np.full(n_contacts + 1, np.nan)
        ),
        "joint": {
            pair_class: (
                (mask[:, pairs[:, 0]] & mask[:, pairs[:, 1]]).mean(axis=0)
                if len(values) else np.full(len(pairs), np.nan)
            )
            for pair_class, pairs in (
                (c, np.asarray(pairs_by_class[c], dtype=int).reshape((-1, 2)))
                for c in PAIR_CLASS_ORDER
            )
        },
    }


def order_view(onsets_ms, pairs_by_class: Mapping, *, tie_tolerance: float = TIE_TOLERANCE_MS) -> dict:
    """Three-state counts per pair over jointly recruited events only."""
    values = _validate_onsets(onsets_ms)
    output = {"counts": {}, "joint_count": {}}
    for pair_class in PAIR_CLASS_ORDER:
        pairs = np.asarray(pairs_by_class[pair_class], dtype=int).reshape((-1, 2))
        left = values[:, pairs[:, 0]]
        right = values[:, pairs[:, 1]]
        joint = np.isfinite(left) & np.isfinite(right)
        difference = np.where(joint, left - right, 0.0)
        before = joint & (difference < -tie_tolerance)
        after = joint & (difference > tie_tolerance)
        tie = joint & ~before & ~after
        output["counts"][pair_class] = np.column_stack([
            before.sum(axis=0), after.sum(axis=0), tie.sum(axis=0),
        ]).astype(float)
        output["joint_count"][pair_class] = joint.sum(axis=0).astype(float)
    return output


def lag_view(onsets_ms, pairs_by_class: Mapping, *, lag_cap_ms: float = LAG_CAP_MS) -> dict:
    """Per-pair |lag| samples in ms over jointly recruited events, capped."""
    values = _validate_onsets(onsets_ms)
    output = {}
    for pair_class in PAIR_CLASS_ORDER:
        pairs = np.asarray(pairs_by_class[pair_class], dtype=int).reshape((-1, 2))
        rows = []
        for i, j in pairs:
            joint = np.isfinite(values[:, i]) & np.isfinite(values[:, j])
            lag = np.abs(values[joint, i] - values[joint, j])
            rows.append(np.minimum(lag, float(lag_cap_ms)))
        output[pair_class] = rows
    return output


def merge_order_views(views: list[dict]) -> dict:
    output = {"counts": {}, "joint_count": {}}
    for pair_class in PAIR_CLASS_ORDER:
        output["counts"][pair_class] = np.sum([v["counts"][pair_class] for v in views], axis=0)
        output["joint_count"][pair_class] = np.sum(
            [v["joint_count"][pair_class] for v in views], axis=0,
        )
    return output


def merge_lag_views(views: list[dict]) -> dict:
    return {
        pair_class: [
            np.concatenate([v[pair_class][k] for v in views])
            for k in range(len(views[0][pair_class]))
        ]
        for pair_class in PAIR_CLASS_ORDER
    }


def merge_support_views(views: list[dict]) -> dict:
    weights = np.asarray([v["n_events"] for v in views], dtype=float)
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("cannot merge empty support views")

    def _avg(key_fn):
        return np.sum([w * key_fn(v) for w, v in zip(weights, views)], axis=0) / total

    return {
        "n_events": int(total),
        "n_contacts": int(views[0]["n_contacts"]),
        "recruitment": {s: _avg(lambda v, s=s: v["recruitment"][s]) for s in SHAFT_ORDER},
        "count_histogram": _avg(lambda v: v["count_histogram"]),
        "joint": {c: _avg(lambda v, c=c: v["joint"][c]) for c in PAIR_CLASS_ORDER},
    }


# --------------------------------------------------------------------------- #
# distances
# --------------------------------------------------------------------------- #
def _js_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    midpoint = 0.5 * (left + right)
    with np.errstate(divide="ignore", invalid="ignore"):
        lt = np.where(left > 0.0, left * np.log2(left / midpoint), 0.0)
        rt = np.where(right > 0.0, right * np.log2(right / midpoint), 0.0)
    return 0.5 * (lt.sum(axis=1) + rt.sum(axis=1))


def support_distance(model: Mapping, patient: Mapping) -> dict:
    if model["n_events"] < 1:
        return {"value": None, "status": STATUS_LOW_EVENTS,
                "recruitment": None, "count": None, "joint": None}
    n_contacts = int(model["n_contacts"])
    recruitment = float(np.mean([
        np.mean(np.abs(np.asarray(model["recruitment"][s]) - np.asarray(patient["recruitment"][s])))
        for s in SHAFT_ORDER
    ]))
    support = np.arange(n_contacts + 1, dtype=float)
    count = float(wasserstein_distance(
        support, support, np.asarray(model["count_histogram"]),
        np.asarray(patient["count_histogram"]),
    )) / n_contacts
    joint = float(np.mean([
        np.mean(np.abs(np.asarray(model["joint"][c]) - np.asarray(patient["joint"][c])))
        for c in PAIR_CLASS_ORDER
    ]))
    return {
        "value": float((recruitment + count + joint) / 3.0),
        "status": STATUS_OK,
        "recruitment": recruitment, "count": count, "joint": joint,
    }


def _conditional_probabilities(view: Mapping, pair_class: str) -> np.ndarray:
    counts = np.asarray(view["counts"][pair_class], dtype=float)
    joint = np.asarray(view["joint_count"][pair_class], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(joint[:, None] > 0, counts / np.where(joint > 0, joint, 1.0)[:, None], np.nan)


def order_distance(model: Mapping, patient: Mapping, *, n_pair_min: int = N_PAIR_MIN) -> dict:
    per_class = {}
    eligible_fraction = {}
    for pair_class in PAIR_CLASS_ORDER:
        model_joint = np.asarray(model["joint_count"][pair_class], dtype=float)
        patient_joint = np.asarray(patient["joint_count"][pair_class], dtype=float)
        eligible = (model_joint >= n_pair_min) & (patient_joint >= 1)
        eligible_fraction[pair_class] = float(eligible.mean()) if len(eligible) else 0.0
        if not eligible.any():
            per_class[pair_class] = None
            continue
        pm = _conditional_probabilities(model, pair_class)[eligible]
        pp = _conditional_probabilities(patient, pair_class)[eligible]
        per_class[pair_class] = float(np.mean(_js_rows(pm, pp)))
    status = STATUS_OK if all(
        eligible_fraction[c] >= MIN_ELIGIBLE_FRACTION for c in PAIR_CLASS_ORDER
    ) else STATUS_LOW_JOINT
    valid = [v for v in per_class.values() if v is not None]
    return {
        "value": float(np.mean(valid)) if len(valid) == len(PAIR_CLASS_ORDER) else None,
        "status": status,
        "per_class": per_class,
        "eligible_fraction": eligible_fraction,
    }


def lag_distance(model: Mapping, patient: Mapping, *, n_pair_min: int = N_PAIR_MIN) -> dict:
    per_class = {}
    eligible_fraction = {}
    for pair_class in PAIR_CLASS_ORDER:
        model_rows = model[pair_class]
        patient_rows = patient[pair_class]
        distances = []
        n_eligible = 0
        for m_row, p_row in zip(model_rows, patient_rows):
            if len(m_row) >= n_pair_min and len(p_row) >= 1:
                n_eligible += 1
                distances.append(float(wasserstein_distance(m_row, p_row)))
        eligible_fraction[pair_class] = float(n_eligible / len(model_rows)) if len(model_rows) else 0.0
        per_class[pair_class] = float(np.mean(distances)) if distances else None
    status = STATUS_OK if all(
        eligible_fraction[c] >= MIN_ELIGIBLE_FRACTION for c in PAIR_CLASS_ORDER
    ) else STATUS_LOW_JOINT
    valid = [v for v in per_class.values() if v is not None]
    return {
        "value": float(np.mean(valid)) if len(valid) == len(PAIR_CLASS_ORDER) else None,
        "status": status,
        "per_class": per_class,
        "eligible_fraction": eligible_fraction,
    }


def coverage_distance(model_z: np.ndarray, query_z: np.ndarray, *, quantile: float = 0.90) -> dict:
    model_z = np.asarray(model_z, dtype=float)
    query_z = np.asarray(query_z, dtype=float)
    if model_z.ndim != 2 or len(model_z) < 1:
        return {"value": None, "status": STATUS_LOW_EVENTS, "quantile": float(quantile)}
    nearest, _ = cKDTree(model_z).query(query_z, k=1)
    return {
        "value": float(np.quantile(nearest, float(quantile))),
        "status": STATUS_OK,
        "quantile": float(quantile),
    }


# --------------------------------------------------------------------------- #
# patient reference and full component vector
# --------------------------------------------------------------------------- #
def patient_reference(onsets_ms, groups: Mapping, pairs_by_class: Mapping, embedding: Mapping, *,
                      lag_cap_ms: float = LAG_CAP_MS) -> dict:
    values = _validate_onsets(onsets_ms)
    return {
        "support": support_view(values, groups, pairs_by_class),
        "order": order_view(values, pairs_by_class),
        "lag": lag_view(values, pairs_by_class, lag_cap_ms=lag_cap_ms),
        "z": transform_patient_embedding(embedding_features(values, groups, lag_cap_ms=lag_cap_ms), embedding),
    }


def component_vector(model_onsets_ms, reference: Mapping, groups: Mapping, pairs_by_class: Mapping,
                     embedding: Mapping, *, n_pair_min: int = N_PAIR_MIN,
                     lag_cap_ms: float = LAG_CAP_MS, cover_quantile: float = 0.90,
                     composite: bool = True, components=COMPONENTS) -> dict:
    """Training components for one model event table (returned families only).

    ``components`` selects which of the four are computed; the others are absent.
    """
    values = _validate_onsets(model_onsets_ms)
    out = {"n_events": int(len(values))}
    if "D_support" in components:
        out["D_support"] = support_distance(support_view(values, groups, pairs_by_class), reference["support"])
    if "D_order" in components:
        out["D_order"] = order_distance(order_view(values, pairs_by_class), reference["order"],
                                        n_pair_min=n_pair_min)
    if "D_lag" in components:
        out["D_lag"] = lag_distance(lag_view(values, pairs_by_class, lag_cap_ms=lag_cap_ms), reference["lag"],
                                    n_pair_min=n_pair_min)
    need_embedding = "D_cover" in components or composite
    if need_embedding:
        features = embedding_features(values, groups, lag_cap_ms=lag_cap_ms) if len(values) else np.zeros((0, 1))
        if len(values):
            model_z = transform_patient_embedding(features, embedding)
            if "D_cover" in components:
                out["D_cover"] = coverage_distance(model_z, reference["z"], quantile=cover_quantile)
            if composite:
                composite_value = sliced_event_cloud_distance(
                    features, embedding, reference_z=reference["z"],
                ) if len(values) >= 2 else float("nan")
                out["D_cloud_composite"] = None if not np.isfinite(composite_value) else float(composite_value)
        else:
            if "D_cover" in components:
                out["D_cover"] = {"value": None, "status": STATUS_LOW_EVENTS, "quantile": float(cover_quantile)}
            if composite:
                out["D_cloud_composite"] = None
    out["clipping"] = clipping_fractions(values, lag_cap_ms=lag_cap_ms)
    return out


def component_values(vector: Mapping) -> dict:
    return {k: vector[k]["value"] for k in COMPONENTS}


# --------------------------------------------------------------------------- #
# floors, excess, identifiability
# --------------------------------------------------------------------------- #
def split_blocks(block_ids: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random half split of recording blocks; returns (pseudo_model_mask, reference_mask)."""
    blocks = np.unique(np.asarray(block_ids))
    if len(blocks) < 2:
        raise ValueError("block split needs at least two recording blocks")
    permuted = rng.permutation(blocks)
    half = len(permuted) // 2
    model_blocks = set(permuted[:half].tolist())
    model_mask = np.asarray([b in model_blocks for b in np.asarray(block_ids)], dtype=bool)
    return model_mask, ~model_mask


class PatientBlockViews:
    """Per-recording-block sufficient statistics so any block subset merges cheaply."""

    def __init__(self, onsets_ms, block_ids, groups: Mapping, pairs_by_class: Mapping,
                 embedding: Mapping, *, lag_cap_ms: float = LAG_CAP_MS):
        values = _validate_onsets(onsets_ms)
        block_ids = np.asarray(block_ids)
        self.values = values
        self.block_ids = block_ids
        self.blocks = np.unique(block_ids)
        self.groups, self.pairs, self.embedding, self.lag_cap_ms = groups, pairs_by_class, embedding, float(lag_cap_ms)
        self.z = transform_patient_embedding(embedding_features(values, groups, lag_cap_ms=lag_cap_ms), embedding)
        self._support, self._order, self._lag, self._rows = {}, {}, {}, {}
        for block in self.blocks:
            rows = np.flatnonzero(block_ids == block)
            self._rows[block] = rows
            sub = values[rows]
            self._support[block] = support_view(sub, groups, pairs_by_class)
            self._order[block] = order_view(sub, pairs_by_class)
            self._lag[block] = lag_view(sub, pairs_by_class, lag_cap_ms=lag_cap_ms)

    def rows(self, blocks) -> np.ndarray:
        return np.concatenate([self._rows[b] for b in blocks]) if len(blocks) else np.zeros(0, int)

    def reference(self, blocks) -> dict:
        blocks = list(blocks)
        return {
            "support": merge_support_views([self._support[b] for b in blocks]),
            "order": merge_order_views([self._order[b] for b in blocks]),
            "lag": merge_lag_views([self._lag[b] for b in blocks]),
            "z": self.z[self.rows(blocks)],
        }

    def full_reference(self) -> dict:
        return self.reference(self.blocks)


def recruitment_profile(onsets_ms) -> np.ndarray:
    values = _validate_onsets(onsets_ms)
    return np.isfinite(values).mean(axis=0) if len(values) else np.full(values.shape[1], np.nan)


def thin_recruitment(onsets_ms, target_profile, source_profile, rng: np.random.Generator) -> np.ndarray:
    """Drop recruited entries so the sample's per-contact recruitment matches ``target_profile``.

    Keep probability per contact is min(1, target / source); contacts the sample never
    recruits are untouched. The recruitment mask changes; kept onsets are unchanged.
    """
    values = _validate_onsets(onsets_ms).copy()
    target = np.asarray(target_profile, float)
    source = np.asarray(source_profile, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        keep = np.where(source > 0, np.minimum(1.0, target / np.where(source > 0, source, 1.0)), 1.0)
    keep = np.where(np.isfinite(keep), keep, 1.0)
    drop = np.isfinite(values) & (rng.random(values.shape) >= keep[None, :])
    values[drop] = np.nan
    return values


def block_split_floors(block_views: PatientBlockViews, requests, *, draws: int, seed: int,
                       n_pair_min: int = N_PAIR_MIN, cover_quantile: float = 0.90) -> dict:
    """Count-matched block-split floors.

    ``requests`` is a list of dicts ``{"key", "n", "components", "thin_profile"}``. One
    block split per draw is shared by all requests. The pseudo-model sample is drawn
    uniformly from the pseudo-model half; when ``thin_profile`` is given (order/lag floors)
    the sample is recruitment-thinned toward that per-contact profile before scoring.
    """
    rng = np.random.default_rng(int(seed))
    records = {r["key"]: {k: [] for k in r["components"]} for r in requests}
    values = block_views.values
    for _ in range(int(draws)):
        model_mask, reference_mask = split_blocks(block_views.block_ids, rng)
        reference_blocks = np.unique(block_views.block_ids[reference_mask])
        reference = block_views.reference(reference_blocks)
        pool = np.flatnonzero(model_mask)
        pool_profile = recruitment_profile(values[pool])
        for request in requests:
            take = min(int(request["n"]), len(pool))
            sample = values[rng.choice(pool, size=take, replace=False)]
            if request.get("thin_profile") is not None:
                sample = thin_recruitment(sample, request["thin_profile"], pool_profile, rng)
            vector = component_vector(sample, reference, block_views.groups, block_views.pairs,
                                      block_views.embedding, n_pair_min=n_pair_min,
                                      lag_cap_ms=block_views.lag_cap_ms, cover_quantile=cover_quantile,
                                      composite=False, components=tuple(request["components"]))
            for k in request["components"]:
                records[request["key"]][k].append(vector[k]["value"])
    output = {}
    for key, comps in records.items():
        output[key] = {}
        for k, vals in comps.items():
            finite = np.asarray([v for v in vals if v is not None], dtype=float)
            output[key][k] = {
                "draws": int(len(finite)),
                "q05": float(np.quantile(finite, 0.05)) if len(finite) else None,
                "q50": float(np.quantile(finite, 0.50)) if len(finite) else None,
                "q95": float(np.quantile(finite, 0.95)) if len(finite) else None,
            }
    return output


def pooled_candidate(unit_onsets: list, reference: Mapping, groups: Mapping, pairs_by_class: Mapping,
                     embedding: Mapping, *, n_pair_min: int = N_PAIR_MIN, lag_cap_ms: float = LAG_CAP_MS,
                     cover_quantile: float = 0.90) -> dict:
    """Pooled candidate-level components with leave-one-unit-out jackknife SD."""
    units = [_validate_onsets(u) for u in unit_onsets]
    pooled = np.concatenate(units, axis=0) if units else np.zeros((0, int(reference["support"]["n_contacts"])))
    vector = component_vector(pooled, reference, groups, pairs_by_class, embedding, n_pair_min=n_pair_min,
                              lag_cap_ms=lag_cap_ms, cover_quantile=cover_quantile, composite=True)
    per_unit = [component_vector(u, reference, groups, pairs_by_class, embedding, n_pair_min=n_pair_min,
                                 lag_cap_ms=lag_cap_ms, cover_quantile=cover_quantile, composite=True)
                for u in units]
    jackknife = {k: None for k in COMPONENTS}
    if len(units) >= 2:
        leave_out = []
        for i in range(len(units)):
            rest = np.concatenate([u for j, u in enumerate(units) if j != i], axis=0)
            leave_out.append(component_vector(rest, reference, groups, pairs_by_class, embedding,
                                              n_pair_min=n_pair_min, lag_cap_ms=lag_cap_ms,
                                              cover_quantile=cover_quantile, composite=False))
        m = len(units)
        for k in COMPONENTS:
            vals = np.asarray([lo[k]["value"] if lo[k]["value"] is not None else np.nan for lo in leave_out], float)
            if np.isfinite(vals).all():
                jackknife[k] = float(np.sqrt((m - 1) / m * np.sum((vals - vals.mean()) ** 2)))
    return {
        "pooled": vector,
        "n_units": len(units),
        "n_pooled_events": int(len(pooled)),
        "recruitment_profile": recruitment_profile(pooled) if len(pooled) else None,
        "per_unit": per_unit,
        "jackknife_sd": jackknife,
    }


def standardized_excess(value, floor: Mapping, *, epsilon: float = 1e-9):
    """Unclipped (D - floor_q50) / (floor_q95 - floor_q50); used for identifiability."""
    if value is None or floor is None or floor.get("q50") is None or floor.get("q95") is None:
        return None
    return float((float(value) - float(floor["q50"]))
                 / (float(floor["q95"]) - float(floor["q50"]) + float(epsilon)))


def normalized_excess(value, floor: Mapping, *, epsilon: float = 1e-9):
    standardized = standardized_excess(value, floor, epsilon=epsilon)
    return None if standardized is None else float(max(0.0, standardized))


def identifiability_ratio(candidate_values: Mapping[str, float | None],
                          candidate_noise: Mapping[str, float | None]) -> dict:
    """Between-candidate q90-q10 of candidate-level values over the median candidate noise SD."""
    values = np.asarray([v for v in candidate_values.values() if v is not None], float)
    noise = np.asarray([v for v in candidate_noise.values() if v is not None], float)
    if len(values) < 2 or len(noise) < 1:
        return {"ratio": None, "between_range": None, "within_sd": None,
                "n_candidates": int(len(values)), "n_noise": int(len(noise))}
    between = float(np.quantile(values, 0.90) - np.quantile(values, 0.10))
    within = float(np.median(noise))
    return {
        "ratio": float(between / within) if within > 0 else float("inf"),
        "between_range": between, "within_sd": within,
        "n_candidates": int(len(values)), "n_noise": int(len(noise)),
    }


def minimax_proposal_scalar(excesses: Mapping[str, float | None], identifiable: list[str]):
    """J_fit = max over identifiable components; None if any identifiable one is missing."""
    values = [excesses.get(k) for k in identifiable]
    if not identifiable or any(v is None for v in values):
        return None
    return float(max(values))


# --------------------------------------------------------------------------- #
# synthetic manipulations (controls)
# --------------------------------------------------------------------------- #
def censor_shaft(onsets_ms, groups: Mapping, shaft: str) -> np.ndarray:
    values = _validate_onsets(onsets_ms).copy()
    values[:, np.asarray(groups[shaft], dtype=int)] = np.nan
    return values


def stretch_onsets(onsets_ms, factor: float) -> np.ndarray:
    values = _validate_onsets(onsets_ms)
    mask, relative, _ = relative_onsets(values)
    first = np.where(mask.any(axis=1), np.min(np.where(mask, values, np.inf), axis=1), 0.0)
    return np.where(mask, first[:, None] + float(factor) * relative, np.nan)


def permute_within_shaft(onsets_ms, groups: Mapping, rng: np.random.Generator) -> np.ndarray:
    """Permute onset times among recruited contacts within each shaft; mask unchanged."""
    values = _validate_onsets(onsets_ms).copy()
    for shaft in SHAFT_ORDER:
        idx = np.asarray(groups[shaft], dtype=int)
        for row in range(len(values)):
            recruited = idx[np.isfinite(values[row, idx])]
            if len(recruited) >= 2:
                values[row, recruited] = values[row, rng.permutation(recruited)]
    return values
