"""Metrics for the rev12-ND Node-only dual-mode refit.

The module contains no patient or simulator I/O.  It keeps contact-distribution
fit and sheet-level source-topology reproducibility separate so that a good rank
prototype cannot hide unstable multi-site nucleation.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.ndimage import label as connected_component_labels


def merge_detected_event_fragments(events: list[dict], *,
                                   maximum_gap_ms: float,
                                   sample_dt_ms: float) -> list[dict]:
    """Merge detector fragments into settling-consistent biological episodes."""
    maximum_gap_ms = float(maximum_gap_ms)
    sample_dt_ms = float(sample_dt_ms)
    if maximum_gap_ms < 0.0 or sample_dt_ms <= 0.0:
        raise ValueError("merge gap must be nonnegative and sample_dt_ms positive")
    merged: list[dict] = []
    for detector_index, event in enumerate(events):
        row = dict(event)
        row["detector_fragment_indices"] = [int(detector_index)]
        row["fragment_count"] = 1
        if merged:
            gap = float(row["t_on"]) - float(merged[-1]["t_off"])
            if gap <= maximum_gap_ms:
                previous = merged[-1]
                previous["t_off"] = float(max(previous["t_off"], row["t_off"]))
                previous["dur_ms"] = float(
                    previous["t_off"] - previous["t_on"] + sample_dt_ms
                )
                previous["peak_ext"] = float(max(
                    previous["peak_ext"], row["peak_ext"],
                ))
                previous["returned"] = bool(
                    previous["returned"] and row["returned"]
                )
                previous["detector_fragment_indices"].append(int(detector_index))
                previous["fragment_count"] += 1
                continue
        merged.append(row)
    return merged


def _stable_low_start(values: np.ndarray, *, start: int, stop: int,
                      threshold: float, required_steps: int) -> int | None:
    """Return the first low-state dwell start inside ``[start, stop)``."""
    values = np.asarray(values, float)
    start = max(0, int(start))
    stop = min(len(values), int(stop))
    required_steps = int(required_steps)
    if required_steps <= 0:
        raise ValueError("required_steps must be positive")
    if stop - start < required_steps:
        return None
    low = values[start:stop] <= float(threshold)
    padded = np.concatenate(([False], low, [False])).astype(np.int8)
    changes = np.flatnonzero(np.diff(padded))
    for left, right in zip(changes[::2], changes[1::2]):
        if right - left >= required_steps:
            return int(start + left)
    return None


def population_excursion_episodes(events: list[dict], active_fraction: np.ndarray,
                                  *, sample_dt_ms: float,
                                  event_on_threshold: float,
                                  low_threshold_fraction: float,
                                  reset_ms: float,
                                  pre_roll_ms: float) -> list[dict]:
    """Group threshold fragments until the fast population state truly resets.

    Boundaries depend only on the latent population activity, never on virtual
    contact placement or recruitment.  A new episode starts only after activity
    has stayed below ``low_threshold_fraction * event_on_threshold`` for a full
    fast-state reset interval.
    """
    values = np.asarray(active_fraction, float)
    sample_dt_ms = float(sample_dt_ms)
    event_on_threshold = float(event_on_threshold)
    low_threshold_fraction = float(low_threshold_fraction)
    reset_ms = float(reset_ms)
    pre_roll_ms = float(pre_roll_ms)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("active_fraction must be a finite one-dimensional trace")
    if sample_dt_ms <= 0.0 or event_on_threshold <= 0.0:
        raise ValueError("sample_dt_ms and event_on_threshold must be positive")
    if not 0.0 < low_threshold_fraction < 1.0:
        raise ValueError("low_threshold_fraction must lie in (0, 1)")
    if reset_ms <= 0.0 or pre_roll_ms < 0.0:
        raise ValueError("reset_ms must be positive and pre_roll_ms nonnegative")
    if not events:
        return []
    required_steps = max(1, int(np.ceil(reset_ms / sample_dt_ms)))
    low_threshold = low_threshold_fraction * event_on_threshold
    ordered = sorted(
        enumerate(events), key=lambda item: (float(item[1]["t_on"]), item[0]),
    )
    groups: list[tuple[list[tuple[int, dict]], int | None]] = []
    current = [ordered[0]]
    for detector_index, event in ordered[1:]:
        previous = current[-1][1]
        gap_start = int(np.floor(float(previous["t_off"]) / sample_dt_ms)) + 1
        gap_stop = int(np.ceil(float(event["t_on"]) / sample_dt_ms))
        reset_start = _stable_low_start(
            values, start=gap_start, stop=gap_stop,
            threshold=low_threshold, required_steps=required_steps,
        )
        if reset_start is None:
            current.append((detector_index, event))
        else:
            groups.append((current, reset_start))
            current = [(detector_index, event)]
    last_event = current[-1][1]
    final_start = int(np.floor(float(last_event["t_off"]) / sample_dt_ms)) + 1
    final_reset = _stable_low_start(
        values, start=final_start, stop=len(values),
        threshold=low_threshold, required_steps=required_steps,
    )
    groups.append((current, final_reset))

    output = []
    total_ms = len(values) * sample_dt_ms
    for group, reset_start in groups:
        fragment_indices = [int(index) for index, _ in group]
        trigger_on = float(group[0][1]["t_on"])
        trigger_off = float(group[-1][1]["t_off"])
        analysis_on = max(0.0, trigger_on - pre_roll_ms)
        analysis_off = (
            float(reset_start * sample_dt_ms)
            if reset_start is not None else total_ms
        )
        start_step = max(0, int(np.floor(analysis_on / sample_dt_ms)))
        stop_step = min(len(values), int(np.ceil(analysis_off / sample_dt_ms)) + 1)
        peak = float(np.max(values[start_step:stop_step]))
        output.append({
            "t_on": analysis_on,
            "t_off": analysis_off,
            "dur_ms": float(max(sample_dt_ms, analysis_off - analysis_on)),
            "peak_ext": peak,
            "returned": reset_start is not None,
            "trigger_t_on": trigger_on,
            "trigger_t_off": trigger_off,
            "reset_start_ms": (
                None if reset_start is None else float(reset_start * sample_dt_ms)
            ),
            "detector_fragment_indices": fragment_indices,
            "fragment_count": int(len(fragment_indices)),
            "event_unit": "population_excursion",
            "low_threshold": float(low_threshold),
            "reset_ms": reset_ms,
            "pre_roll_ms": pre_roll_ms,
        })
    return output


def event_features(normalized_ranks: np.ndarray) -> np.ndarray:
    """Build fixed-contact [recruitment, masked normalized rank] features."""
    ranks = np.asarray(normalized_ranks, float)
    if ranks.ndim != 2:
        raise ValueError("normalized_ranks must have shape (event, contact)")
    mask = np.isfinite(ranks).astype(float)
    return np.concatenate([mask, np.nan_to_num(ranks, nan=0.0)], axis=1)


def normalize_event_ranks(ranks: np.ndarray) -> np.ndarray:
    """Normalize every finite event row to [0, 1] without filling missing contacts."""
    values = np.asarray(ranks, float)
    if values.ndim != 2:
        raise ValueError("ranks must have shape (event, contact)")
    output = np.full_like(values, np.nan, dtype=float)
    for index, row in enumerate(values):
        finite = np.isfinite(row)
        if not np.any(finite):
            continue
        low = float(np.min(row[finite]))
        span = float(np.max(row[finite]) - low)
        output[index, finite] = (row[finite] - low) / span if span > 0 else 0.0
    return output


def _shaft_names(contact_names: np.ndarray) -> np.ndarray:
    return np.asarray([
        "".join(character for character in str(name) if not character.isdigit())
        for name in np.asarray(contact_names).astype(str)
    ])


def shaft_balanced_feature_weights(contact_names: np.ndarray) -> np.ndarray:
    """Give recruitment/rank and ICL/SCL equal total weight."""
    shafts = _shaft_names(contact_names)
    weights = np.zeros(2 * len(shafts), float)
    for offset in (0, len(shafts)):
        for shaft in ("ICL", "SCL"):
            selected = np.flatnonzero(shafts == shaft)
            if not len(selected):
                raise ValueError(f"contact contract misses shaft {shaft}")
            weights[offset + selected] = 0.25 / len(selected)
    return weights


def weighted_r2(events: np.ndarray, labels: np.ndarray, prototypes: np.ndarray,
                global_mean: np.ndarray, weights: np.ndarray) -> float:
    """Variance explained relative to a fixed global-mean reference."""
    events = np.asarray(events, float)
    labels = np.asarray(labels, int)
    prototypes = np.asarray(prototypes, float)
    global_mean = np.asarray(global_mean, float)
    weights = np.asarray(weights, float)
    if events.ndim != 2 or labels.shape != (len(events),):
        raise ValueError("events and labels do not align")
    if np.any((labels < 0) | (labels >= len(prototypes))):
        raise ValueError("labels do not index prototypes")
    sst = float(np.sum((events - global_mean) ** 2 * weights))
    sse = float(np.sum((events - prototypes[labels]) ** 2 * weights))
    return float("nan") if sst <= 0.0 else float(1.0 - sse / sst)


def contrast_r2(model_prototypes: np.ndarray,
                patient_train_prototypes: np.ndarray,
                patient_heldout_prototypes: np.ndarray,
                weights: np.ndarray) -> dict:
    """Evaluate a TA-TB contrast with one nonnegative scale fitted on train."""
    model = np.asarray(model_prototypes, float)
    train = np.asarray(patient_train_prototypes, float)
    heldout = np.asarray(patient_heldout_prototypes, float)
    weights = np.asarray(weights, float)
    model_delta = model[0] - model[1]
    train_delta = train[0] - train[1]
    heldout_delta = heldout[0] - heldout[1]
    energy = float(np.sum(weights * model_delta ** 2))
    scale = (
        0.0 if energy <= 0.0 else max(
            0.0, float(np.sum(weights * train_delta * model_delta) / energy),
        )
    )
    denominator = float(np.sum(weights * heldout_delta ** 2))
    raw_sse = float(np.sum(weights * (heldout_delta - model_delta) ** 2))
    scaled_sse = float(np.sum(weights * (heldout_delta - scale * model_delta) ** 2))
    cosine_denominator = float(np.sqrt(
        np.sum(weights * heldout_delta ** 2) * np.sum(weights * model_delta ** 2)
    ))
    return {
        "train_fitted_nonnegative_scale": scale,
        "heldout_raw_r2": (
            float("nan") if denominator <= 0.0 else float(1.0 - raw_sse / denominator)
        ),
        "heldout_train_scaled_r2": (
            float("nan") if denominator <= 0.0 else float(1.0 - scaled_sse / denominator)
        ),
        "heldout_weighted_cosine": (
            float("nan") if cosine_denominator <= 0.0
            else float(np.sum(weights * heldout_delta * model_delta) / cosine_denominator)
        ),
    }


def lse_max(values: np.ndarray, tau: float = 0.25) -> float:
    """Smooth maximum with zero offset for equal inputs."""
    values = np.asarray(values, float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("LSE inputs must be a finite non-empty vector")
    tau = float(tau)
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    maximum = float(np.max(values))
    return float(maximum + tau * np.log(np.mean(np.exp((values - maximum) / tau))))


def _weighted_quantile_grid(values: np.ndarray, n_quantiles: int) -> np.ndarray:
    values = np.asarray(values, float)
    if not len(values):
        raise ValueError("cannot form quantiles from an empty sample")
    quantiles = np.linspace(0.0, 1.0, int(n_quantiles))
    # An empirical inverse CDF is invariant to exact replication of a sample.
    # Linear order-statistic interpolation is not: repeating every event changes
    # interpolation knots and would let event count leak into the objective.
    return np.quantile(values, quantiles, method="inverted_cdf")


def fixed_projection_matrix(n_features: int, *, n_directions: int = 64,
                            seed: int = 20260821) -> np.ndarray:
    """Generate deterministic unit directions for sliced-Wasserstein scoring."""
    rng = np.random.default_rng(int(seed))
    matrix = rng.normal(size=(int(n_directions), int(n_features)))
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norm == 0.0):
        raise RuntimeError("projection generator produced a zero direction")
    return matrix / norm


def sliced_wasserstein(x: np.ndarray, y: np.ndarray, *, weights: np.ndarray,
                       projections: np.ndarray, n_quantiles: int = 101) -> float:
    """Matched-quantile sliced-Wasserstein distance with fixed feature weights."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    weights = np.asarray(weights, float)
    projections = np.asarray(projections, float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("event samples must share a feature dimension")
    if not len(x) or not len(y):
        raise ValueError("event samples must be non-empty")
    if weights.shape != (x.shape[1],) or projections.shape[1] != x.shape[1]:
        raise ValueError("weights or projections do not match event features")
    scale = np.sqrt(weights / np.sum(weights))
    x_projection = (x * scale) @ projections.T
    y_projection = (y * scale) @ projections.T
    distances = []
    for direction in range(projections.shape[0]):
        xq = _weighted_quantile_grid(x_projection[:, direction], n_quantiles)
        yq = _weighted_quantile_grid(y_projection[:, direction], n_quantiles)
        distances.append(float(np.sqrt(np.mean((xq - yq) ** 2))))
    return float(np.mean(distances))


def js_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-12) -> float:
    """Jensen-Shannon divergence in nats."""
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    if p.shape != q.shape or p.ndim != 1:
        raise ValueError("probability vectors must align")
    if np.any(p < 0.0) or np.any(q < 0.0) or p.sum() <= 0.0 or q.sum() <= 0.0:
        raise ValueError("probability vectors must be nonnegative and non-empty")
    p = p / p.sum()
    q = q / q.sum()
    midpoint = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log((p + epsilon) / (midpoint + epsilon)))
                 + 0.5 * np.sum(q * np.log((q + epsilon) / (midpoint + epsilon))))


def recruitment_loss(model_ranks: np.ndarray, patient_ranks: np.ndarray,
                     contact_names: np.ndarray) -> float:
    """Shaft-balanced recruitment-probability MAE."""
    model = np.isfinite(np.asarray(model_ranks, float)).mean(axis=0)
    patient = np.isfinite(np.asarray(patient_ranks, float)).mean(axis=0)
    shafts = _shaft_names(contact_names)
    errors = []
    for shaft in ("ICL", "SCL"):
        selected = shafts == shaft
        errors.append(float(np.mean(np.abs(model[selected] - patient[selected]))))
    return float(np.mean(errors))


def _precedence_distribution(ranks: np.ndarray, i: int, j: int) -> np.ndarray:
    values = np.asarray(ranks, float)
    both = np.isfinite(values[:, i]) & np.isfinite(values[:, j])
    i_before = both & (values[:, i] < values[:, j])
    j_before = both & (values[:, j] < values[:, i])
    tied = both & ~(i_before | j_before)
    # Ties contribute equally to the two directional states; missing remains explicit.
    return np.asarray([
        np.sum(i_before) + 0.5 * np.sum(tied),
        np.sum(j_before) + 0.5 * np.sum(tied),
        np.sum(~both),
    ], float) / max(1, len(values))


def precedence_loss(model_ranks: np.ndarray, patient_ranks: np.ndarray,
                    contact_names: np.ndarray) -> dict:
    """Pair-class balanced JS loss including not-jointly-recruited events."""
    shafts = _shaft_names(contact_names)
    classes = {"ICL-ICL": [], "SCL-SCL": [], "ICL-SCL": []}
    for i, j in combinations(range(len(shafts)), 2):
        pair = f"{shafts[i]}-{shafts[j]}"
        if pair == "SCL-ICL":
            pair = "ICL-SCL"
        model = _precedence_distribution(model_ranks, i, j)
        patient = _precedence_distribution(patient_ranks, i, j)
        classes[pair].append(js_divergence(model, patient))
    output = {
        name: float(np.mean(values)) for name, values in classes.items()
    }
    output["balanced_mean"] = float(np.mean(list(output.values())))
    return output


def profile_loss(model_ranks: np.ndarray, patient_ranks: np.ndarray,
                 contact_names: np.ndarray) -> float:
    """Weighted error between mode mean recruitment/rank profiles."""
    model = np.mean(event_features(normalize_event_ranks(model_ranks)), axis=0)
    patient = np.mean(event_features(normalize_event_ranks(patient_ranks)), axis=0)
    weights = shaft_balanced_feature_weights(contact_names)
    return float(np.sqrt(np.sum(weights * (model - patient) ** 2)))


def mode_distribution_components(model_ranks: np.ndarray, patient_ranks: np.ndarray,
                                 contact_names: np.ndarray,
                                 projections: np.ndarray) -> dict:
    """Four complementary distances for one patient-defined mode."""
    model_normalized = normalize_event_ranks(model_ranks)
    patient_normalized = normalize_event_ranks(patient_ranks)
    weights = shaft_balanced_feature_weights(contact_names)
    model_features = event_features(model_normalized)
    patient_features = event_features(patient_normalized)
    precedence = precedence_loss(model_normalized, patient_normalized, contact_names)
    return {
        "recruitment": recruitment_loss(model_normalized, patient_normalized, contact_names),
        "precedence": precedence["balanced_mean"],
        "precedence_classes": precedence,
        "profile": profile_loss(model_normalized, patient_normalized, contact_names),
        "cloud": sliced_wasserstein(
            model_features, patient_features, weights=weights,
            projections=projections,
        ),
    }


def calibrate_component_scales(patient_ranks: np.ndarray,
                               patient_labels: np.ndarray,
                               patient_blocks: np.ndarray,
                               contact_names: np.ndarray,
                               projections: np.ndarray, *,
                               sample_size: int = 6,
                               draws: int = 256,
                               seed: int = 20260821) -> dict:
    """Put all four distances in patient recording-block excess-noise units."""
    ranks = np.asarray(patient_ranks, float)
    labels = np.asarray(patient_labels, int)
    blocks = np.asarray(patient_blocks)
    if (ranks.ndim != 2 or labels.shape != (len(ranks),)
            or blocks.shape != (len(ranks),)):
        raise ValueError("patient calibration arrays do not align")
    rng = np.random.default_rng(int(seed))
    keys = ("recruitment", "precedence", "profile", "cloud")
    output = {}
    for mode in (0, 1):
        mode_index = np.flatnonzero(labels == mode)
        eligible_blocks = np.asarray([
            block for block in np.unique(blocks)
            if np.sum((blocks == block) & (labels == mode)) >= int(sample_size)
        ])
        if len(eligible_blocks) < 2:
            raise ValueError("patient mode lacks two eligible recording blocks")
        floor = {key: [] for key in keys}
        null = {key: [] for key in keys}
        for _ in range(int(draws)):
            target_block, self_block = rng.choice(
                eligible_blocks, size=2, replace=False,
            )
            target_index = np.flatnonzero(
                (blocks == target_block) & (labels == mode)
            )
            self_index = np.flatnonzero(
                (blocks == self_block) & (labels == mode)
            )
            target = ranks[rng.choice(
                target_index, size=int(sample_size), replace=False,
            )]
            self_sample = ranks[rng.choice(
                self_index, size=int(sample_size), replace=False,
            )]
            pooled = ranks[rng.choice(len(ranks), size=int(sample_size), replace=False)]
            self_components = mode_distribution_components(
                self_sample, target, contact_names, projections,
            )
            null_components = mode_distribution_components(
                pooled, target, contact_names, projections,
            )
            for key in keys:
                floor[key].append(float(self_components[key]))
                null[key].append(float(null_components[key]))
        output[str(mode)] = {}
        for key in keys:
            floor_median = float(np.median(floor[key]))
            floor_q95 = float(np.quantile(floor[key], 0.95))
            null_median = float(np.median(null[key]))
            null_q95 = float(np.quantile(null[key], 0.95))
            scale = floor_q95 - floor_median
            reference = "patient_block_floor_q95"
            if scale <= 1e-8:
                scale = null_q95 - floor_median
                reference = "pooled_global_null_q95_fallback"
            if scale <= 1e-8:
                raise RuntimeError(
                    f"patient calibration has no {key} dynamic range for mode {mode}"
                )
            output[str(mode)][key] = {
                "floor_median": floor_median,
                "floor_q95": floor_q95,
                "global_null_median": null_median,
                "global_null_q95": null_q95,
                "scale": scale,
                "scale_reference": reference,
            }
    return {
        "modes": output,
        "sample_size_per_side": int(sample_size),
        "draws": int(draws),
        "seed": int(seed),
        "floor_resampling": "different recording blocks within the same mode",
        "global_null_resampling": "pooled patient training events across modes",
        "unit": (
            "0=patient block floor median; 1=patient block floor q95; "
            "pooled-null q95 is used only if the block floor is degenerate"
        ),
    }


def normalize_components(components: dict, calibration: dict, mode: int) -> dict:
    normalized = {}
    for key in ("recruitment", "precedence", "profile", "cloud"):
        contract = calibration["modes"][str(int(mode))][key]
        normalized[key] = max(
            0.0,
            (float(components[key]) - float(contract["floor_median"]))
            / float(contract["scale"]),
        )
    return normalized


def dual_mode_objective(model_ranks: np.ndarray, model_labels: np.ndarray,
                        patient_ranks: np.ndarray, patient_labels: np.ndarray,
                        contact_names: np.ndarray, *,
                        missing_mode_penalty: float,
                        projections: np.ndarray,
                        calibration: dict | None = None,
                        tau: float = 0.25) -> dict:
    """Worst-mode patient loss with explicit occupancy mismatch."""
    model_labels = np.asarray(model_labels, int)
    patient_labels = np.asarray(patient_labels, int)
    per_mode, losses = {}, []
    for mode in (0, 1):
        model_selected = np.asarray(model_ranks)[model_labels == mode]
        patient_selected = np.asarray(patient_ranks)[patient_labels == mode]
        if not len(model_selected):
            if calibration is None:
                raw = None
                components = {
                    key: float(missing_mode_penalty)
                    for key in ("recruitment", "precedence", "profile", "cloud")
                }
            else:
                raw = mode_distribution_components(
                    np.full((1, np.asarray(patient_ranks).shape[1]), np.nan),
                    patient_selected, contact_names, projections,
                )
                normalized = normalize_components(raw, calibration, mode)
                components = {
                    key: max(float(missing_mode_penalty), normalized[key])
                    for key in ("recruitment", "precedence", "profile", "cloud")
                }
            components["raw"] = raw
            components["missing"] = True
        else:
            raw = mode_distribution_components(
                model_selected, patient_selected, contact_names, projections,
            )
            components = (
                dict(raw) if calibration is None
                else normalize_components(raw, calibration, mode)
            )
            components["raw"] = dict(raw)
            components["missing"] = False
        components["mean"] = float(np.mean([
            components[key] for key in ("recruitment", "precedence", "profile", "cloud")
        ]))
        per_mode[str(mode)] = components
        losses.append(components["mean"])
    model_count = np.bincount(model_labels, minlength=2).astype(float)
    patient_count = np.bincount(patient_labels, minlength=2).astype(float)
    # Zero returned events are a valid negative simulation result, not missing
    # data.  They carry the maximum binary JS occupancy penalty while both mode
    # distances above retain the explicit missing-mode penalty.
    occupancy = (
        float(np.log(2.0)) if model_count.sum() == 0.0
        else js_divergence(model_count, patient_count)
    )
    weakest = lse_max(np.asarray(losses), tau=tau)
    return {
        "modes": per_mode,
        "weakest_mode_lse": weakest,
        "occupancy_js": occupancy,
        "objective": float(weakest + 0.25 * occupancy),
    }


def persistent_recruitment_onsets(event_counts: np.ndarray,
                                  baseline_counts: np.ndarray,
                                  relative_times_ms: np.ndarray, *,
                                  minimum_active_neurons: int = 2,
                                  persistence_frames: int = 2) -> np.ndarray:
    """First binwise pre-event-q99 exceedance sustained across frames."""
    event_counts = np.asarray(event_counts, float)
    baseline_counts = np.asarray(baseline_counts, float)
    times = np.asarray(relative_times_ms, float)
    if event_counts.ndim != 3 or baseline_counts.ndim != 3:
        raise ValueError("source counts must have shape (time, y, x)")
    if event_counts.shape[1:] != baseline_counts.shape[1:] or len(times) != len(event_counts):
        raise ValueError("source count arrays do not align")
    threshold = np.maximum(
        np.quantile(baseline_counts, 0.99, axis=0),
        float(minimum_active_neurons - 1),
    )
    above = event_counts > threshold
    sustained = np.zeros_like(above)
    for frame in range(len(above) - int(persistence_frames) + 1):
        sustained[frame] = np.all(above[frame:frame + int(persistence_frames)], axis=0)
    onset = np.full(above.shape[1:], np.nan)
    for frame, time_ms in enumerate(times):
        new = sustained[frame] & ~np.isfinite(onset)
        onset[new] = time_ms
    return onset


def sheet_bin_indices(positions: np.ndarray, *, bin_mm: float,
                      sheet_mm: float) -> tuple[np.ndarray, int]:
    """Map E-neuron positions to a fixed square sheet grid."""
    positions = np.asarray(positions, float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (neuron, 2)")
    size = int(round(float(sheet_mm) / float(bin_mm)))
    if size <= 0 or not np.isclose(size * float(bin_mm), float(sheet_mm)):
        raise ValueError("bin size must tile the square sheet")
    xy = np.floor(positions / float(bin_mm)).astype(int)
    xy = np.clip(xy, 0, size - 1)
    return xy[:, 1] * size + xy[:, 0], size


def _distinct_activity_frames(spikes: np.ndarray, *, dt_ms: float,
                              centers_ms: np.ndarray, width_ms: float,
                              neuron_bins: np.ndarray, size: int) -> np.ndarray:
    spikes = np.asarray(spikes, bool)
    centers_ms = np.asarray(centers_ms, float)
    if spikes.ndim != 2 or neuron_bins.shape != (spikes.shape[1],):
        raise ValueError("spikes and neuron bins must align")
    half_steps = int(round(0.5 * float(width_ms) / float(dt_ms)))
    output = np.zeros((len(centers_ms), size, size), float)
    for frame, center_ms in enumerate(centers_ms):
        center = int(round(float(center_ms) / float(dt_ms)))
        low, high = center - half_steps, center + half_steps + 1
        if low < 0 or high > len(spikes):
            raise ValueError("source-topology frame falls outside the simulation")
        active = np.any(spikes[low:high], axis=0)
        output[frame] = np.bincount(
            neuron_bins[active], minlength=size * size,
        ).reshape(size, size)
    return output


def sheet_activity_movie(spikes: np.ndarray, positions: np.ndarray, *,
                         dt_ms: float, frame_ms: float,
                         bin_mm: float, sheet_mm: float) -> dict:
    """Store a whole-run neuron-level sheet movie for event-unit re-audits."""
    spikes = np.asarray(spikes, bool)
    dt_ms = float(dt_ms)
    frame_ms = float(frame_ms)
    if spikes.ndim != 2 or dt_ms <= 0.0 or frame_ms <= 0.0:
        raise ValueError("spikes must be 2-D and movie time steps positive")
    steps = int(round(frame_ms / dt_ms))
    if steps <= 0 or not np.isclose(steps * dt_ms, frame_ms):
        raise ValueError("frame_ms must be an integer multiple of dt_ms")
    neuron_bins, size = sheet_bin_indices(
        positions, bin_mm=bin_mm, sheet_mm=sheet_mm,
    )
    if neuron_bins.shape != (spikes.shape[1],):
        raise ValueError("spikes and positions do not align")
    n_frames = int(np.ceil(len(spikes) / steps))
    output = np.zeros((n_frames, size, size), dtype=np.uint16)
    for frame in range(n_frames):
        active = np.any(
            spikes[frame * steps:min(len(spikes), (frame + 1) * steps)], axis=0,
        )
        counts = np.bincount(neuron_bins[active], minlength=size * size)
        if np.max(counts, initial=0) > np.iinfo(np.uint16).max:
            raise RuntimeError("sheet movie frame exceeds uint16 storage")
        output[frame] = counts.reshape(size, size).astype(np.uint16)
    return {
        "activity_counts": output,
        "frame_ms": frame_ms,
        "bin_mm": float(bin_mm),
        "sheet_mm": float(sheet_mm),
    }


def spatiotemporal_cascade_labels(activity_counts: np.ndarray, *,
                                  minimum_active_neurons: int) -> dict:
    """Find contact-independent cascades in a binned neuron-activity movie.

    Nodes are active sheet bins in one movie frame.  Nodes in the same or
    adjacent sheet bins connect within a frame and to the immediately adjacent
    frames.  At the frozen 2 ms / 1 mm resolution this is the conservative
    propagation cone for the 0.3 mm/ms axonal velocity.  This is an operational
    cascade definition, not proof of synaptic causation.
    """
    counts = np.asarray(activity_counts)
    minimum_active_neurons = int(minimum_active_neurons)
    if counts.ndim != 3 or not np.issubdtype(counts.dtype, np.number):
        raise ValueError("activity_counts must have shape (time, y, x)")
    if minimum_active_neurons <= 0 or np.any(counts < 0):
        raise ValueError("activity counts and threshold must be nonnegative")
    structure = np.ones((3, 3, 3), dtype=bool)
    labels, n_components = connected_component_labels(
        counts >= minimum_active_neurons, structure=structure,
    )
    flat_labels = labels.ravel()
    node_count = np.bincount(flat_labels, minlength=n_components + 1)[1:]
    activity_mass = np.bincount(
        flat_labels, weights=counts.ravel(), minlength=n_components + 1,
    )[1:]
    components = []
    for component in range(1, n_components + 1):
        frames = np.flatnonzero(np.any(labels == component, axis=(1, 2)))
        components.append({
            "cascade_id": component,
            "start_frame": int(frames[0]),
            "stop_frame": int(frames[-1]),
            "duration_frames": int(frames[-1] - frames[0] + 1),
            "active_bin_frames": int(node_count[component - 1]),
            "activity_mass": float(activity_mass[component - 1]),
        })
    return {
        "labels": labels.astype(np.int32, copy=False),
        "components": components,
        "minimum_active_neurons": minimum_active_neurons,
        "neighborhood": "3x3 sheet bins across adjacent movie frames",
    }


def directed_spatiotemporal_lineages(activity_counts: np.ndarray, *,
                                     minimum_active_neurons: int) -> dict:
    """Trace forward-time activity lineages without merging colliding roots.

    A contiguous active patch inherits roots only from its one-frame-back spatial
    neighborhood.  Patches reached by multiple roots are marked as collisions;
    their parents remain separate.  This is a directed observational lineage at
    the frozen movie resolution, not proof of a particular synaptic path.
    """
    counts = np.asarray(activity_counts)
    minimum_active_neurons = int(minimum_active_neurons)
    if counts.ndim != 3 or not np.issubdtype(counts.dtype, np.number):
        raise ValueError("activity_counts must have shape (time, y, x)")
    if minimum_active_neurons <= 0 or np.any(counts < 0):
        raise ValueError("activity counts and threshold must be nonnegative")

    active = counts >= minimum_active_neurons
    patch_structure = np.ones((3, 3), dtype=bool)
    lineage_labels = np.zeros(counts.shape, dtype=np.int32)
    collision_mask = np.zeros(counts.shape, dtype=bool)
    roots: dict[int, dict] = {}
    next_root = 1
    previous_lineages = np.zeros(active.shape[1:], dtype=np.int32)

    for frame in range(len(active)):
        patches, n_patches = connected_component_labels(
            active[frame], structure=patch_structure,
        )
        for patch_id in range(1, n_patches + 1):
            patch = patches == patch_id
            neighborhood = binary_dilation(patch, structure=patch_structure)
            inherited = np.unique(previous_lineages[neighborhood])
            inherited = inherited[inherited > 0].astype(int)
            if not len(inherited):
                inherited = np.asarray([next_root], int)
                roots[next_root] = {
                    "cascade_id": next_root,
                    "lineage_id": next_root,
                    "start_frame": int(frame),
                    "stop_frame": int(frame),
                    "last_any_frame": int(frame),
                    "active_bin_frames": 0,
                    "activity_mass": 0.0,
                    "collision_bin_frames": 0,
                    "collision_activity_mass": 0.0,
                }
                next_root += 1
            if len(inherited) == 1:
                lineage_labels[frame][patch] = int(inherited[0])
            else:
                coordinates = np.argwhere(patch)
                distance_rows = []
                for root_id in inherited:
                    source = np.argwhere(previous_lineages == root_id)
                    delta = coordinates[:, None, :] - source[None, :, :]
                    distance_rows.append(np.min(np.sum(delta * delta, axis=2), axis=1))
                distances = np.asarray(distance_rows, float)
                minimum = np.min(distances, axis=0)
                winners = np.isclose(distances, minimum[None, :])
                unique = np.sum(winners, axis=0) == 1
                selected = np.argmax(winners, axis=0)
                for local_index, coordinate in enumerate(coordinates):
                    location = (int(coordinate[0]), int(coordinate[1]))
                    if unique[local_index]:
                        lineage_labels[frame][location] = int(inherited[selected[local_index]])
                    else:
                        lineage_labels[frame][location] = -1
                        collision_mask[frame][location] = True

            for root_id in inherited:
                selected = (lineage_labels[frame] == root_id) & patch
                if np.any(selected):
                    row = roots[int(root_id)]
                    row["stop_frame"] = int(frame)
                    row["last_any_frame"] = int(frame)
                    row["active_bin_frames"] += int(np.sum(selected))
                    row["activity_mass"] += float(np.sum(counts[frame][selected]))
            collision = collision_mask[frame] & patch
            if np.any(collision):
                collision_bins = int(np.sum(collision))
                collision_mass = float(np.sum(counts[frame][collision]))
                for root_id in inherited:
                    row = roots[int(root_id)]
                    row["last_any_frame"] = int(frame)
                    row["collision_bin_frames"] += collision_bins
                    row["collision_activity_mass"] += collision_mass
        previous_lineages = lineage_labels[frame]

    components = []
    for root_id in sorted(roots):
        row = dict(roots[root_id])
        row["duration_frames"] = int(row["stop_frame"] - row["start_frame"] + 1)
        row["root_collision_observed"] = bool(row["collision_bin_frames"] > 0)
        components.append(row)
    return {
        "labels": lineage_labels,
        "collision_mask": collision_mask,
        "components": components,
        "minimum_active_neurons": minimum_active_neurons,
        "causal_rule": (
            "forward seeded watershed from one-frame-back 3x3 spatial neighborhood; "
            "multiple inherited roots remain separate and only equidistant boundaries "
            "are marked as collisions"
        ),
    }


def assign_detector_fragments_to_directed_lineages(
        activity_counts: np.ndarray, lineage_labels: np.ndarray,
        fragments: list[dict], *, frame_ms: float,
        minimum_dominance: float) -> list[dict]:
    """Assign fragments by root mass while counting collision mass as ambiguity."""
    counts = np.asarray(activity_counts, float)
    labels = np.asarray(lineage_labels, int)
    frame_ms = float(frame_ms)
    minimum_dominance = float(minimum_dominance)
    if counts.shape != labels.shape or counts.ndim != 3:
        raise ValueError("activity counts and directed lineage labels must align")
    if frame_ms <= 0.0 or not 0.5 < minimum_dominance <= 1.0:
        raise ValueError("invalid frame duration or lineage dominance threshold")
    output = []
    for index, fragment in enumerate(fragments):
        start = max(0, int(np.floor(float(fragment["t_on"]) / frame_ms)))
        stop = min(len(counts), int(np.floor(float(fragment["t_off"]) / frame_ms)) + 1)
        local_labels = labels[start:stop].ravel()
        local_counts = counts[start:stop].ravel()
        positive = local_labels > 0
        collision = local_labels < 0
        masses = np.bincount(
            local_labels[positive], weights=local_counts[positive],
        ) if np.any(positive) else np.zeros(1, float)
        positive_mass = float(np.sum(masses[1:])) if len(masses) > 1 else 0.0
        collision_mass = float(np.sum(local_counts[collision]))
        total = positive_mass + collision_mass
        if positive_mass <= 0.0 or total <= 0.0:
            dominant_id, dominance, second = None, 0.0, 0.0
        else:
            order = np.argsort(masses[1:])[::-1] + 1
            dominant_id = int(order[0])
            dominance = float(masses[dominant_id] / total)
            second = float(masses[order[1]] / total) if len(order) > 1 else 0.0
        output.append({
            "detector_fragment_index": int(index),
            "dominant_cascade_id": dominant_id,
            "dominant_lineage_id": dominant_id,
            "dominant_activity_fraction": dominance,
            "second_activity_fraction": second,
            "collision_activity_fraction": (
                float(collision_mass / total) if total > 0.0 else 0.0
            ),
            "compound": bool(dominant_id is None or dominance < minimum_dominance),
        })
    return output


def directed_lineage_onset_maps(lineage_labels: np.ndarray,
                                events: list[dict], *,
                                frame_ms: float) -> dict:
    """Return each directed root's first-arrival map at movie resolution."""
    labels = np.asarray(lineage_labels, int)
    frame_ms = float(frame_ms)
    if labels.ndim != 3 or frame_ms <= 0.0:
        raise ValueError("lineage labels must be 3-D and frame_ms positive")
    maps = np.full((len(events), *labels.shape[1:]), np.nan, np.float32)
    evaluable = np.zeros(len(events), bool)
    for event_index, event in enumerate(events):
        lineage_id = int(event["cascade_id"])
        support = labels == lineage_id
        if not np.any(support):
            continue
        frames, ys, xs = np.nonzero(support)
        first = np.full(labels.shape[1:], np.inf, float)
        np.minimum.at(first, (ys, xs), frames.astype(float) * frame_ms)
        finite = np.isfinite(first)
        first[finite] -= float(np.min(first[finite]))
        maps[event_index, finite] = first[finite].astype(np.float32)
        evaluable[event_index] = True
    return {"onset_maps_ms": maps, "evaluable": evaluable}


def sheet_contact_sampling_weights(contact_xy: np.ndarray,
                                   bin_population: np.ndarray, *,
                                   bin_mm: float,
                                   kernel_width_mm: float) -> np.ndarray:
    """Approximate the per-neuron Gaussian readout on a binned sheet.

    The denominator uses the number of E neurons in each sheet bin.  This is the
    binned equivalent of normalizing the original per-neuron Gaussian weights,
    rather than incorrectly giving every occupied sheet bin equal weight.
    """
    contacts = np.asarray(contact_xy, float)
    population = np.asarray(bin_population, float)
    bin_mm = float(bin_mm)
    kernel_width_mm = float(kernel_width_mm)
    if (contacts.ndim != 2 or contacts.shape[1] != 2
            or population.ndim != 2 or np.any(population < 0)
            or bin_mm <= 0.0 or kernel_width_mm <= 0.0):
        raise ValueError("invalid contact geometry or sheet population")
    y_index, x_index = np.indices(population.shape)
    centers = np.column_stack((
        (x_index.ravel() + 0.5) * bin_mm,
        (y_index.ravel() + 0.5) * bin_mm,
    ))
    weights = []
    for contact in contacts:
        distance2 = np.sum((centers - contact[None, :]) ** 2, axis=1)
        row = np.exp(-distance2 / (2.0 * kernel_width_mm ** 2))
        denominator = float(np.sum(row * population.ravel()))
        if denominator <= 0.0:
            raise RuntimeError("contact Gaussian footprint contains no E neurons")
        weights.append(row / denominator)
    return np.asarray(weights, float).reshape(len(contacts), *population.shape)


def binned_contact_envelope(activity_counts: np.ndarray,
                            contact_weights: np.ndarray, *,
                            frame_ms: float,
                            smooth_ms: float) -> np.ndarray:
    """Project a binned E-activity movie through the frozen contact sampler."""
    counts = np.asarray(activity_counts, float)
    weights = np.asarray(contact_weights, float)
    frame_ms = float(frame_ms)
    smooth_ms = float(smooth_ms)
    if (counts.ndim != 3 or weights.ndim != 3
            or counts.shape[1:] != weights.shape[1:]
            or frame_ms <= 0.0 or smooth_ms <= 0.0):
        raise ValueError("activity movie and contact weights do not align")
    raw = counts.reshape(len(counts), -1) @ weights.reshape(len(weights), -1).T
    sigma = max(1e-6, smooth_ms / frame_ms)
    half = int(np.ceil(3.0 * sigma))
    axis = np.arange(-half, half + 1)
    kernel = np.exp(-(axis ** 2) / (2.0 * sigma ** 2))
    kernel /= np.sum(kernel)
    output = []
    center = (len(kernel) - 1) // 2
    for contact in range(raw.shape[1]):
        full = np.convolve(raw[:, contact], kernel, mode="full")
        output.append(full[center:center + len(raw)])
    return np.stack(output)


def _dense_onset_ranks(onsets: np.ndarray) -> np.ndarray:
    ranks = np.full(len(onsets), np.nan)
    finite = np.flatnonzero(np.isfinite(onsets))
    if not len(finite):
        return ranks
    order = finite[np.argsort(onsets[finite], kind="mergesort")]
    rank = 0.0
    previous = float(onsets[order[0]])
    ranks[order[0]] = rank
    for index in order[1:]:
        value = float(onsets[index])
        if value > previous:
            rank += 1.0
            previous = value
        ranks[index] = rank
    return ranks


def lineage_restricted_contact_readout(activity_counts: np.ndarray,
                                       lineage_labels: np.ndarray,
                                       events: list[dict],
                                       contact_weights: np.ndarray, *,
                                       frame_ms: float,
                                       smooth_ms: float,
                                       participation_margin_fraction: float,
                                       timing_fraction: float) -> dict:
    """Extract contact ranks using only activity assigned to each event root."""
    counts = np.asarray(activity_counts, float)
    labels = np.asarray(lineage_labels, int)
    weights = np.asarray(contact_weights, float)
    frame_ms = float(frame_ms)
    smooth_ms = float(smooth_ms)
    margin = float(participation_margin_fraction)
    timing = float(timing_fraction)
    if counts.shape != labels.shape or counts.ndim != 3:
        raise ValueError("activity counts and lineage labels must align")
    if (weights.ndim != 3 or weights.shape[1:] != counts.shape[1:]
            or frame_ms <= 0.0 or smooth_ms <= 0.0
            or not 0.0 < margin < 1.0 or not 0.0 < timing <= 1.0):
        raise ValueError("invalid lineage contact-readout contract")
    n_contacts = len(weights)
    onset_rows = np.full((len(events), n_contacts), np.nan)
    rank_rows = np.full((len(events), n_contacts), np.nan)
    sigma = smooth_ms / frame_ms
    pad = int(np.ceil(3.0 * sigma))
    for event_index, event in enumerate(events):
        lineage_id = int(event["cascade_id"])
        start = max(0, int(round(float(event["t_on"]) / frame_ms)))
        stop = min(len(counts), int(round(float(event["t_off"]) / frame_ms)))
        if stop <= start:
            continue
        local_start, local_stop = max(0, start - pad), min(len(counts), stop + pad)
        selected = counts[local_start:local_stop] * (
            labels[local_start:local_stop] == lineage_id
        )
        envelope = binned_contact_envelope(
            selected, weights, frame_ms=frame_ms, smooth_ms=smooth_ms,
        )
        segment = envelope[:, start - local_start:stop - local_start]
        if not segment.size:
            continue
        floor = float(np.min(segment))
        peak = np.max(segment, axis=1)
        participation_bar = floor + margin * (float(np.max(segment)) - floor)
        participating = peak > participation_bar
        onsets = np.full(n_contacts, np.nan)
        for contact in np.flatnonzero(participating):
            crossing = np.flatnonzero(segment[contact] >= timing * peak[contact])
            if len(crossing):
                onsets[contact] = (start + int(crossing[0])) * frame_ms
        onset_rows[event_index] = onsets
        rank_rows[event_index] = _dense_onset_ranks(onsets)
    return {"onsets": onset_rows, "ranks": rank_rows}


def assign_detector_fragments_to_cascades(activity_counts: np.ndarray,
                                          cascade_labels: np.ndarray,
                                          fragments: list[dict], *,
                                          frame_ms: float,
                                          minimum_dominance: float) -> list[dict]:
    """Assign detector fragments to a dominant cascade or mark them compound."""
    counts = np.asarray(activity_counts, float)
    labels = np.asarray(cascade_labels, int)
    frame_ms = float(frame_ms)
    minimum_dominance = float(minimum_dominance)
    if counts.shape != labels.shape or counts.ndim != 3:
        raise ValueError("activity counts and cascade labels must align")
    if frame_ms <= 0.0 or not 0.5 < minimum_dominance <= 1.0:
        raise ValueError("invalid frame duration or cascade dominance threshold")
    output = []
    for index, fragment in enumerate(fragments):
        start = max(0, int(np.floor(float(fragment["t_on"]) / frame_ms)))
        stop = min(len(counts), int(np.floor(float(fragment["t_off"]) / frame_ms)) + 1)
        local_labels = labels[start:stop].ravel()
        local_counts = counts[start:stop].ravel()
        valid = local_labels > 0
        masses = np.bincount(
            local_labels[valid], weights=local_counts[valid],
        ) if np.any(valid) else np.zeros(1, float)
        if len(masses) <= 1 or float(np.sum(masses[1:])) <= 0.0:
            dominant_id, dominance, second = None, 0.0, 0.0
        else:
            order = np.argsort(masses[1:])[::-1] + 1
            total = float(np.sum(masses[1:]))
            dominant_id = int(order[0])
            dominance = float(masses[dominant_id] / total)
            second = float(masses[order[1]] / total) if len(order) > 1 else 0.0
        output.append({
            "detector_fragment_index": int(index),
            "dominant_cascade_id": dominant_id,
            "dominant_activity_fraction": dominance,
            "second_activity_fraction": second,
            "compound": bool(dominant_id is None or dominance < minimum_dominance),
        })
    return output


def cascade_event_windows(components: list[dict], assignments: list[dict],
                          fragments: list[dict], *, frame_ms: float,
                          total_ms: float) -> tuple[list[dict], list[dict]]:
    """Build causal-consistent event windows and retain compounds separately."""
    frame_ms = float(frame_ms)
    total_ms = float(total_ms)
    if frame_ms <= 0.0 or total_ms <= 0.0 or len(assignments) != len(fragments):
        raise ValueError("cascade event inputs are inconsistent")
    component_by_id = {int(row["cascade_id"]): row for row in components}
    groups: dict[int, list[int]] = {}
    compounds = []
    for assignment, fragment in zip(assignments, fragments):
        fragment_index = int(assignment["detector_fragment_index"])
        if assignment["compound"] or assignment["dominant_cascade_id"] is None:
            compounds.append({
                "detector_fragment_indices": [fragment_index],
                "t_on": float(fragment["t_on"]),
                "t_off": float(fragment["t_off"]),
                "compound": True,
                "dominant_activity_fraction": float(
                    assignment["dominant_activity_fraction"]
                ),
            })
            continue
        cascade_id = int(assignment["dominant_cascade_id"])
        if cascade_id not in component_by_id:
            raise RuntimeError("fragment references an absent cascade")
        groups.setdefault(cascade_id, []).append(fragment_index)
    events = []
    for cascade_id, fragment_indices in groups.items():
        component = component_by_id[cascade_id]
        selected = [fragments[index] for index in fragment_indices]
        component_on = float(component["start_frame"]) * frame_ms
        component_off = float(component["stop_frame"] + 1) * frame_ms
        t_on = max(0.0, min(component_on, min(
            float(fragment["t_on"]) for fragment in selected
        )))
        t_off = min(total_ms, max(component_off, max(
            float(fragment["t_off"]) for fragment in selected
        )))
        events.append({
            "cascade_id": cascade_id,
            "detector_fragment_indices": sorted(fragment_indices),
            "trigger_t_on": float(min(
                fragment["t_on"] for fragment in selected
            )),
            "trigger_t_off": float(max(
                fragment["t_off"] for fragment in selected
            )),
            "t_on": t_on,
            "t_off": t_off,
            "dur_ms": float(max(frame_ms, t_off - t_on)),
            "returned": bool(
                t_off < total_ms
                and all(bool(fragment.get("returned", True)) for fragment in selected)
            ),
            "compound": False,
            "activity_mass": float(component["activity_mass"]),
            "active_bin_frames": int(component["active_bin_frames"]),
        })
    events.sort(key=lambda row: (row["t_on"], row["cascade_id"]))
    compounds.sort(key=lambda row: row["t_on"])
    return events, compounds


def normalize_components_floor_ratio(components: dict, calibration: dict,
                                     mode: int) -> dict:
    """Continuously scale distances by the patient block-floor q95."""
    output = {}
    for key in ("recruitment", "precedence", "profile", "cloud"):
        q95 = float(calibration["modes"][str(int(mode))][key]["floor_q95"])
        if not np.isfinite(q95) or q95 <= 0.0:
            raise ValueError("patient floor q95 must be finite and positive")
        output[key] = float(components[key]) / q95
    return output


def matched_sample_dual_mode_objective(model_ranks: np.ndarray,
                                       model_labels: np.ndarray,
                                       patient_ranks: np.ndarray,
                                       patient_labels: np.ndarray,
                                       patient_blocks: np.ndarray,
                                       contact_names: np.ndarray, *,
                                       projections: np.ndarray,
                                       calibration: dict,
                                       sample_size: int = 6,
                                       draws: int = 64,
                                       seed: int = 20260824,
                                       missing_mode_penalty: float = 2.0,
                                       tau: float = 0.25) -> dict:
    """Matched 6-vs-6 continuous loss for the cascade-event objective."""
    model_ranks = np.asarray(model_ranks, float)
    model_labels = np.asarray(model_labels, int)
    patient_ranks = np.asarray(patient_ranks, float)
    patient_labels = np.asarray(patient_labels, int)
    patient_blocks = np.asarray(patient_blocks)
    sample_size, draws = int(sample_size), int(draws)
    if (model_ranks.ndim != 2 or patient_ranks.ndim != 2
            or model_ranks.shape[1] != patient_ranks.shape[1]
            or model_labels.shape != (len(model_ranks),)
            or patient_labels.shape != (len(patient_ranks),)
            or patient_blocks.shape != (len(patient_ranks),)
            or sample_size <= 0 or draws <= 0):
        raise ValueError("matched-sample objective arrays do not align")
    rng = np.random.default_rng(int(seed))
    per_mode, losses = {}, []
    for mode in (0, 1):
        model_index = np.flatnonzero(model_labels == mode)
        eligible_blocks = [
            block for block in np.unique(patient_blocks)
            if np.sum((patient_blocks == block) & (patient_labels == mode))
            >= sample_size
        ]
        if not eligible_blocks:
            raise ValueError(f"patient mode {mode} has no matched-sample block")
        if len(model_index) < sample_size:
            normalized = {
                key: float(missing_mode_penalty)
                for key in ("recruitment", "precedence", "profile", "cloud")
            }
            raw_mean = None
            missing = True
        else:
            normalized_draws = []
            raw_draws = []
            for _ in range(draws):
                block = eligible_blocks[int(rng.integers(len(eligible_blocks)))]
                patient_index = np.flatnonzero(
                    (patient_blocks == block) & (patient_labels == mode)
                )
                model_sample = model_ranks[rng.choice(
                    model_index, size=sample_size, replace=False,
                )]
                patient_sample = patient_ranks[rng.choice(
                    patient_index, size=sample_size, replace=False,
                )]
                raw = mode_distribution_components(
                    model_sample, patient_sample, contact_names, projections,
                )
                raw_draws.append(raw)
                normalized_draws.append(
                    normalize_components_floor_ratio(raw, calibration, mode)
                )
            normalized = {
                key: float(np.mean([row[key] for row in normalized_draws]))
                for key in ("recruitment", "precedence", "profile", "cloud")
            }
            raw_mean = {
                key: float(np.mean([row[key] for row in raw_draws]))
                for key in ("recruitment", "precedence", "profile", "cloud")
            }
            missing = False
        mean = float(np.mean(list(normalized.values())))
        per_mode[str(mode)] = {
            **normalized, "raw": raw_mean, "missing": missing, "mean": mean,
            "n_model_events": int(len(model_index)),
        }
        losses.append(mean)
    model_count = np.bincount(model_labels, minlength=2).astype(float)
    patient_count = np.bincount(patient_labels, minlength=2).astype(float)
    occupancy = (
        float(np.log(2.0)) if model_count.sum() == 0.0
        else js_divergence(model_count, patient_count)
    )
    weakest = lse_max(np.asarray(losses), tau=tau)
    return {
        "modes": per_mode,
        "weakest_mode_lse": weakest,
        "occupancy_js": occupancy,
        "objective": float(weakest + 0.25 * occupancy),
        "sample_size_per_side": sample_size,
        "draws": draws,
        "normalization": "raw_distance_divided_by_patient_block_floor_q95",
    }


def event_source_onset_maps(spikes: np.ndarray, positions: np.ndarray,
                            event_onsets_ms: np.ndarray,
                            event_selected: np.ndarray, *, dt_ms: float,
                            sheet_mm: float, bin_mm: float = 1.0,
                            event_relative_ms: np.ndarray | None = None,
                            baseline_relative_ms: np.ndarray | None = None,
                            frame_width_ms: float = 3.0) -> dict:
    """Reduce full spikes to one 1-mm local recruitment-onset map per event."""
    spikes = np.asarray(spikes, bool)
    event_onsets_ms = np.asarray(event_onsets_ms, float)
    selected = np.asarray(event_selected, bool)
    if selected.shape != event_onsets_ms.shape:
        raise ValueError("event selection and onset times must align")
    relative = (
        np.arange(-20.0, 80.0 + 1e-9, 2.0) if event_relative_ms is None
        else np.asarray(event_relative_ms, float)
    )
    baseline_relative = (
        np.arange(-120.0, -20.0 + 1e-9, 2.0) if baseline_relative_ms is None
        else np.asarray(baseline_relative_ms, float)
    )
    neuron_bins, size = sheet_bin_indices(
        positions, bin_mm=bin_mm, sheet_mm=sheet_mm,
    )
    maps = np.full((len(event_onsets_ms), size, size), np.nan, dtype=np.float32)
    activity = np.zeros(
        (len(event_onsets_ms), len(relative), size, size), dtype=np.uint16,
    )
    evaluable = np.zeros(len(event_onsets_ms), bool)
    for event_index in np.flatnonzero(selected):
        center = float(event_onsets_ms[event_index])
        all_centers = center + np.concatenate([baseline_relative, relative])
        half_width = 0.5 * float(frame_width_ms)
        if (np.min(all_centers) - half_width < 0.0
                or np.max(all_centers) + half_width >= len(spikes) * float(dt_ms)):
            continue
        baseline = _distinct_activity_frames(
            spikes, dt_ms=dt_ms, centers_ms=center + baseline_relative,
            width_ms=frame_width_ms, neuron_bins=neuron_bins, size=size,
        )
        event = _distinct_activity_frames(
            spikes, dt_ms=dt_ms, centers_ms=center + relative,
            width_ms=frame_width_ms, neuron_bins=neuron_bins, size=size,
        )
        maps[event_index] = persistent_recruitment_onsets(
            event, baseline, relative,
        ).astype(np.float32)
        if np.max(event) > np.iinfo(np.uint16).max:
            raise RuntimeError("event source activity exceeds uint16 storage")
        activity[event_index] = event.astype(np.uint16)
        evaluable[event_index] = True
    return {
        "onset_maps_ms": maps,
        "activity_counts": activity,
        "evaluable": evaluable,
        "relative_times_ms": relative,
        "baseline_relative_times_ms": baseline_relative,
        "bin_mm": float(bin_mm),
        "sheet_mm": float(sheet_mm),
    }


def source_topology_features(onset_maps: np.ndarray) -> np.ndarray:
    """Encode early source support and normalized local onset for each event."""
    values = np.asarray(onset_maps, float)
    if values.ndim != 3:
        raise ValueError("onset maps must have shape (event, y, x)")
    output = []
    for onset in values:
        finite = np.isfinite(onset)
        early = np.zeros_like(onset, dtype=float)
        normalized = np.zeros_like(onset, dtype=float)
        if np.any(finite):
            count = int(np.sum(finite))
            early_count = min(count, max(1, int(np.ceil(0.10 * count))))
            threshold = np.partition(onset[finite], early_count - 1)[early_count - 1]
            early[finite & (onset <= threshold)] = 1.0
            low = float(np.min(onset[finite]))
            span = float(np.max(onset[finite]) - low)
            normalized[finite] = (onset[finite] - low) / span if span > 0.0 else 0.0
        output.append(np.concatenate([early.ravel(), normalized.ravel()]))
    return np.asarray(output, float)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float("nan") if denominator <= 0.0 else float(np.dot(a, b) / denominator)


def topology_reproducibility(onset_maps: np.ndarray, labels: np.ndarray) -> dict:
    """Odd-even within-mode reliability and between-mode template distance."""
    features = source_topology_features(onset_maps)
    labels = np.asarray(labels, int)
    if labels.shape != (len(features),):
        raise ValueError("topology labels do not align")
    modes, reliabilities, templates = {}, [], []
    for mode in (0, 1):
        selected = features[labels == mode]
        if len(selected) < 2:
            reliability = float("nan")
            template = np.mean(selected, axis=0) if len(selected) else np.zeros(features.shape[1])
        else:
            first = np.mean(selected[::2], axis=0)
            second = np.mean(selected[1::2], axis=0)
            reliability = cosine_similarity(first, second)
            template = np.mean(selected, axis=0)
        modes[str(mode)] = {"n_events": int(len(selected)), "split_half_cosine": reliability}
        reliabilities.append(reliability)
        templates.append(template)
    separation = 1.0 - cosine_similarity(templates[0], templates[1])
    finite = np.asarray(reliabilities, float)
    mean_reliability = float(np.mean(finite[np.isfinite(finite)])) if np.any(np.isfinite(finite)) else float("nan")
    loss = (
        float("nan") if not np.isfinite(mean_reliability) or not np.isfinite(separation)
        else float(0.5 * (1.0 - mean_reliability) + 0.5 * (1.0 - separation))
    )
    return {
        "modes": modes,
        "mean_within_mode_reliability": mean_reliability,
        "between_mode_distance": separation,
        "topology_loss": loss,
    }


def topology_network_reproducibility(onset_maps_by_network: list[np.ndarray],
                                     labels_by_network: list[np.ndarray]) -> dict:
    """Equal-network source-template reliability and mode separation."""
    if len(onset_maps_by_network) != len(labels_by_network):
        raise ValueError("network source maps and labels do not align")
    mode_templates = {0: [], 1: []}
    within_network = []
    for maps, labels in zip(onset_maps_by_network, labels_by_network):
        maps = np.asarray(maps, float)
        labels = np.asarray(labels, int)
        if maps.ndim != 3 or labels.shape != (len(maps),):
            raise ValueError("one network source map bundle does not align")
        if len(maps):
            diagnostic = topology_reproducibility(maps, labels)
            if np.isfinite(diagnostic["mean_within_mode_reliability"]):
                within_network.append(diagnostic["mean_within_mode_reliability"])
            features = source_topology_features(maps)
            for mode in (0, 1):
                if np.any(labels == mode):
                    mode_templates[mode].append(np.mean(features[labels == mode], axis=0))
    modes, equal_network_templates = {}, []
    for mode in (0, 1):
        templates = mode_templates[mode]
        similarities = [
            cosine_similarity(templates[left], templates[right])
            for left, right in combinations(range(len(templates)), 2)
        ]
        similarities = np.asarray(similarities, float)
        similarities = similarities[np.isfinite(similarities)]
        modes[str(mode)] = {
            "n_networks": int(len(templates)),
            "pairwise_network_cosine_mean": (
                float(np.mean(similarities)) if len(similarities) else float("nan")
            ),
        }
        equal_network_templates.append(
            np.mean(templates, axis=0) if templates else None
        )
    if any(template is None for template in equal_network_templates):
        separation = float("nan")
    else:
        separation = 1.0 - cosine_similarity(
            equal_network_templates[0], equal_network_templates[1],
        )
    across = np.asarray([
        modes[str(mode)]["pairwise_network_cosine_mean"] for mode in (0, 1)
    ], float)
    across = across[np.isfinite(across)]
    return {
        "modes": modes,
        "mean_within_network_split_half_cosine": (
            float(np.mean(within_network)) if within_network else float("nan")
        ),
        "mean_across_network_template_cosine": (
            float(np.mean(across)) if len(across) else float("nan")
        ),
        "equal_network_between_mode_distance": separation,
    }
