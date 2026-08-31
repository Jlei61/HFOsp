"""Patient-internal heterogeneous seizure-entry estimators for H2b v0.4.

The state model is never trained here.  At most two low-capacity entry-route
prototypes are fitted from strictly earlier seizures inside each outer fold.
This permits different seizures from one patient to approach different state
regions without allowing a held-out seizure to define its own route.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from .v03_geometry import (
    _effect,
    _reference_nn_distance,
    _trajectory_rows,
    fit_decoder_projection,
    fit_two_basins,
    matched_control_trajectories,
    trajectory_features,
)
from .v03_hazard import (
    HazardDesign,
    _causal_context,
    _labels_known_by,
    horizon_outcome,
)
# The older full-grid estimator used the v0.3 binary-hazard grid.  v0.4 is a
# seizure-specific question, so its primary score is conditional within one
# case-plus-controls risk set.  This smaller grid matches the accepted H2b
# conditional probe and is selected only inside the outer training seizures.
CONDITIONAL_ALPHA_GRID = (1.0, 10.0, 100.0)
CONTROLS_PER_RISK_SET = 5
MINIMUM_ROUTE_SEPARATION_BANDWIDTH = 1.0


@dataclass(frozen=True)
class _ConditionalRidgeModel:
    coefficient: np.ndarray
    centre: np.ndarray
    scale: np.ndarray
    converged: bool

    def score(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        return ((matrix - self.centre) / self.scale) @ self.coefficient


def _group_indices(groups: np.ndarray) -> list[np.ndarray]:
    _, inverse = np.unique(np.asarray(groups, dtype=np.int64), return_inverse=True)
    return [np.flatnonzero(inverse == value) for value in range(int(inverse.max()) + 1)]


def _fit_conditional_ridge(
    values: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    l2: float,
) -> _ConditionalRidgeModel:
    """Minimal intercept-free conditional ridge with equal seizure weight."""
    matrix = np.asarray(values, dtype=np.float64)
    target = np.asarray(labels, dtype=np.float64)
    risk_group = np.asarray(groups, dtype=np.int64)
    if matrix.ndim != 2 or len(matrix) != len(target) or len(target) != len(risk_group):
        raise ValueError("conditional probe arrays disagree")
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError("conditional probe contains non-finite values")
    indices = _group_indices(risk_group)
    if any(int(np.sum(target[index])) != 1 or len(index) < 2 for index in indices):
        raise ValueError("each risk set needs exactly one case and at least one control")
    centre = np.mean(matrix, axis=0, dtype=np.float64)
    scale = np.std(matrix, axis=0, dtype=np.float64)
    scale = np.where(scale > 1e-12, scale, 1.0)
    z = (matrix - centre) / scale

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        loss = 0.0
        gradient = np.zeros_like(beta)
        for index in indices:
            score = z[index] @ beta
            probability = np.exp(score - logsumexp(score))
            case = int(np.flatnonzero(target[index] == 1)[0])
            loss += float(logsumexp(score) - score[case])
            gradient += z[index].T @ (probability - target[index])
        count = float(len(indices))
        penalty = float(l2)
        return (
            loss / count + 0.5 * penalty * float(beta @ beta),
            gradient / count + penalty * beta,
        )

    fitted = minimize(
        objective, np.zeros(z.shape[1], dtype=np.float64), jac=True,
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
    )
    return _ConditionalRidgeModel(
        coefficient=np.asarray(fitted.x, dtype=np.float64),
        centre=centre, scale=scale, converged=bool(fitted.success),
    )


def _conditional_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float]:
    target = np.asarray(labels, dtype=np.int8)
    score = np.asarray(scores, dtype=np.float64)
    losses: list[float] = []
    percentiles: list[float] = []
    for index in _group_indices(np.asarray(groups, dtype=np.int64)):
        case_local = int(np.flatnonzero(target[index] == 1)[0])
        case_score = float(score[index[case_local]])
        losses.append(float(logsumexp(score[index]) - case_score))
        greater = float(np.sum(score[index] > case_score))
        equal = float(np.sum(score[index] == case_score))
        rank = 1.0 + greater + 0.5 * (equal - 1.0)
        percentiles.append((rank - 1.0) / max(1.0, len(index) - 1.0))
    return {
        "conditional_log_loss": float(np.mean(losses)),
        "risk_set_rank_percentile": float(np.mean(percentiles)),
    }


@dataclass(frozen=True)
class RouteFeatureMap:
    centre: np.ndarray
    scale: np.ndarray
    active: np.ndarray
    loadings: np.ndarray
    route_centres: np.ndarray
    route_sizes: np.ndarray
    bandwidth: float

    @property
    def n_routes(self) -> int:
        return int(len(self.route_centres))

    def scores(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if not np.any(self.active):
            return np.zeros((len(matrix), 1), dtype=np.float64)
        standard = (matrix[:, self.active] - self.centre[self.active]) / self.scale[
            self.active
        ]
        return standard @ self.loadings.T

    def transform(self, values: np.ndarray) -> np.ndarray:
        score = self.scores(values)
        distance = np.sum(
            (score[:, None, :] - self.route_centres[None, :, :]) ** 2,
            axis=2,
        )
        similarity = -distance / max(self.bandwidth ** 2, 1e-8)
        if self.n_routes == 1:
            # Keep a fixed width without duplicating the sole distance.  A
            # duplicated column changes the effective ridge penalty and would
            # make the one-route comparator artificially different.
            padded = np.column_stack([
                similarity[:, 0], np.zeros(len(score)), np.zeros(len(score)),
            ])
        else:
            padded = np.column_stack([similarity[:, :2], np.max(similarity, axis=1)])
        return padded

    def assign(self, values: np.ndarray) -> np.ndarray:
        score = self.scores(values)
        distance = np.sum(
            (score[:, None, :] - self.route_centres[None, :, :]) ** 2,
            axis=2,
        )
        return np.argmin(distance, axis=1).astype(np.int64)

    def transform_single_axis(self, values: np.ndarray) -> np.ndarray:
        """Project onto one TRAIN-defined directed seizure-entry axis.

        Unlike distance to one centroid, this signed coordinate cannot label
        both opposite ends of an axis as high risk merely by flipping one
        regression coefficient.  It is therefore the proper scalar-axis
        comparator for a two-route union.
        """
        score = self.scores(values)
        direction = np.asarray(self.route_centres[0], dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-10:
            direction = np.zeros(score.shape[1], dtype=np.float64)
            direction[0] = 1.0
        else:
            direction = direction / norm
        projection = score @ direction
        return np.column_stack([
            projection, np.zeros(len(score)), np.zeros(len(score)),
        ])


def _two_means(values: np.ndarray, iterations: int = 50) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=np.float64)
    first = int(np.argmin(matrix[:, 0]))
    distance = np.sum((matrix - matrix[first]) ** 2, axis=1)
    second = int(np.argmax(distance))
    centres = np.vstack([matrix[first], matrix[second]])
    label = np.zeros(len(matrix), dtype=np.int64)
    for _ in range(int(iterations)):
        distance = np.sum(
            (matrix[:, None, :] - centres[None, :, :]) ** 2, axis=2,
        )
        label = np.argmin(distance, axis=1).astype(np.int64)
        if any(not np.any(label == group) for group in (0, 1)):
            break
        updated = np.vstack([np.mean(matrix[label == group], axis=0) for group in (0, 1)])
        if np.allclose(updated, centres, atol=1e-8, rtol=0.0):
            centres = updated
            break
        centres = updated
    return centres, label


def fit_route_feature_map(
    values: np.ndarray,
    *,
    train_rows: np.ndarray,
    prior_event_anchor_rows: np.ndarray,
    maximum_components: int = 4,
    maximum_routes: int = 2,
) -> RouteFeatureMap:
    """Fit PCA and one/two route prototypes on outer-TRAIN rows only."""
    matrix = np.asarray(values, dtype=np.float64)
    train = np.asarray(train_rows, dtype=np.int64)
    anchors = np.asarray(prior_event_anchor_rows, dtype=np.int64)
    if not len(train) or not len(anchors):
        raise ValueError("route map requires training rows and prior seizure anchors")
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("route map values must be a finite matrix")
    if np.any(train < 0) or np.any(train >= len(matrix)):
        raise ValueError("route map training rows are out of bounds")
    if np.any(anchors < 0) or np.any(anchors >= len(matrix)):
        raise ValueError("route map seizure anchors are out of bounds")
    # A causal seizure anchor can be outside the probe's eligible risk rows
    # (for example just beyond a complete-negative horizon at a segment edge).
    # Outer-fold timing is asserted by the caller; requiring membership in the
    # supervised TRAIN risk rows would incorrectly drop valid earlier events.
    if int(maximum_routes) not in (1, 2):
        raise ValueError("maximum_routes must be one or two")
    centre = np.mean(matrix[train], axis=0)
    scale = np.std(matrix[train], axis=0)
    active = np.isfinite(scale) & (scale > 1e-8)
    scale = np.where(active, scale, 1.0)
    if not np.any(active):
        return RouteFeatureMap(
            centre=centre, scale=scale, active=active,
            loadings=np.ones((1, 1), dtype=np.float64),
            route_centres=np.zeros((1, 1), dtype=np.float64),
            route_sizes=np.asarray([len(anchors)], dtype=np.int64), bandwidth=1.0,
        )
    standard = (matrix[train][:, active] - centre[active]) / scale[active]
    _, singular, vt = np.linalg.svd(standard, full_matrices=False)
    variance = singular ** 2
    cumulative = np.cumsum(variance) / max(float(np.sum(variance)), 1e-12)
    count = int(np.searchsorted(cumulative, 0.90) + 1)
    count = max(1, min(count, int(maximum_components), vt.shape[0]))
    loadings = np.array(vt[:count], copy=True)
    for row in range(len(loadings)):
        pivot = int(np.argmax(np.abs(loadings[row])))
        if loadings[row, pivot] < 0:
            loadings[row] *= -1.0
    all_score = ((matrix[:, active] - centre[active]) / scale[active]) @ loadings.T
    event_score = all_score[anchors]
    if int(maximum_routes) == 2 and len(anchors) >= 4 and np.max(np.linalg.norm(
        event_score - event_score[0], axis=1,
    )) > 1e-8:
        route_centres, label = _two_means(event_score)
        sizes = np.asarray([np.sum(label == group) for group in (0, 1)], dtype=np.int64)
        if int(np.min(sizes)) < 2:
            route_centres = np.mean(event_score, axis=0, keepdims=True)
            sizes = np.asarray([len(event_score)], dtype=np.int64)
    else:
        route_centres = np.mean(event_score, axis=0, keepdims=True)
        sizes = np.asarray([len(event_score)], dtype=np.int64)
    def route_bandwidth(centres: np.ndarray) -> float:
        nearest = np.sqrt(np.min(np.sum(
            (all_score[train, None, :] - centres[None, :, :]) ** 2,
            axis=2,
        ), axis=1))
        positive = nearest[nearest > 1e-8]
        return float(np.median(positive)) if len(positive) else 1.0

    bandwidth = route_bandwidth(route_centres)
    if len(route_centres) == 2:
        separation = float(np.linalg.norm(route_centres[0] - route_centres[1]))
        if separation / max(bandwidth, 0.25) < MINIMUM_ROUTE_SEPARATION_BANDWIDTH:
            route_centres = np.mean(event_score, axis=0, keepdims=True)
            sizes = np.asarray([len(event_score)], dtype=np.int64)
            bandwidth = route_bandwidth(route_centres)
    return RouteFeatureMap(
        centre=centre, scale=scale, active=active, loadings=loadings,
        route_centres=route_centres, route_sizes=sizes,
        bandwidth=max(bandwidth, 0.25),
    )


def _anchor_for_onset(
    design: HazardDesign, onset: float, segment: int, horizon_minutes: float,
) -> int | None:
    target = float(onset) - float(horizon_minutes) * 60.0
    rows = np.flatnonzero(
        (design.segment == int(segment)) & (design.time_epoch <= target + 1e-9)
    )
    if not len(rows):
        return None
    row = int(rows[np.argmax(design.time_epoch[rows])])
    return row if target - float(design.time_epoch[row]) <= 7.5 * 60.0 else None


def _supported_onsets(
    design: HazardDesign, eligible: np.ndarray, horizon_minutes: float,
) -> list[tuple[float, int, int]]:
    order = np.argsort(design.onset_time, kind="stable")
    supported: list[tuple[float, int, int]] = []
    for onset, segment in zip(design.onset_time[order], design.onset_segment[order]):
        anchor = _anchor_for_onset(design, float(onset), int(segment), horizon_minutes)
        if anchor is None:
            continue
        if np.any(
            eligible & (design.segment == int(segment))
            & (design.time_epoch < float(onset))
            & (design.time_epoch >= float(onset) - float(horizon_minutes) * 60.0)
        ):
            supported.append((float(onset), int(segment), int(anchor)))
    return supported


def _risk_set_controls(
    design: HazardDesign,
    *,
    outcome: np.ndarray,
    eligible: np.ndarray,
    case_row: int,
    all_case_rows: np.ndarray,
    label_known: np.ndarray,
    maximum_time: float,
    minimum_time_exclusive: float | None,
    controls_per_risk_set: int,
) -> np.ndarray:
    """Choose deterministic outcome-blind controls from admissible coverage.

    The sampler never ranks candidates by history, observation, memoryless code,
    persistent state, or an outcome magnitude.  The probe explicitly adjusts
    history and causal context, avoiding accidental hard matching of a slow state.
    """
    time = np.asarray(design.time_epoch, dtype=np.float64)
    candidate = (
        np.asarray(eligible, dtype=bool)
        & np.asarray(label_known, dtype=bool)
        & (np.asarray(outcome, dtype=np.int8) == 0)
        & (time <= float(maximum_time) + 1e-9)
    )
    if minimum_time_exclusive is not None:
        candidate &= time > float(minimum_time_exclusive) + 1e-9
    candidate[int(case_row)] = False
    candidate[np.asarray(all_case_rows, dtype=np.int64)] = False
    rows = np.flatnonzero(candidate)
    if len(rows) < int(controls_per_risk_set):
        return np.asarray([], dtype=np.int64)

    # Prefer the same coverage segment, but do not make it a hard gate: the
    # original H2b risk-set contract requires same-patient valid recording,
    # whereas same-segment is mandatory only for the wrong-time state donor.
    # Context adjustment retains segment/session position in every probe arm.
    cross_segment = (
        design.segment[rows] != int(design.segment[int(case_row)])
    ).astype(np.int8)
    # A stable integer mixer gives outcome-blind sampling while retaining the
    # exact same anchor identities across optimizer seeds and comparison arms.
    source = np.asarray(design.source_index[rows], dtype=np.int64)
    case_source = int(design.source_index[int(case_row)])
    case_key = np.uint64(
        (case_source * 1442695040888963407) & ((1 << 64) - 1)
    )
    key = (
        source.astype(np.uint64) * np.uint64(6364136223846793005)
        + case_key
    )
    order = np.lexsort((rows, source, key, cross_segment))
    return rows[order[:int(controls_per_risk_set)]].astype(np.int64, copy=False)


def _conditional_alpha(
    values: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    alpha_grid: Sequence[float],
) -> float:
    """Choose ridge on prior seizures only by leave-last-risk-set validation."""
    unique = np.unique(np.asarray(groups, dtype=np.int64))
    if len(unique) < 2:
        return float(max(alpha_grid))
    select_group = int(unique[-1])
    fit_rows = np.asarray(groups, dtype=np.int64) != select_group
    select_rows = ~fit_rows
    if len(np.unique(np.asarray(groups)[fit_rows])) < 1:
        return float(max(alpha_grid))
    scores: list[tuple[float, float, float]] = []
    for alpha in alpha_grid:
        model = _fit_conditional_ridge(
            np.asarray(values)[fit_rows], np.asarray(labels)[fit_rows],
            np.asarray(groups)[fit_rows], l2=float(alpha),
        )
        predicted = model.score(np.asarray(values)[select_rows])
        loss = _conditional_metrics(
            np.asarray(labels)[select_rows], predicted,
            np.asarray(groups)[select_rows],
        )["conditional_log_loss"]
        # Prefer stronger regularisation on an exact validation tie.
        scores.append((float(loss), -float(alpha), float(alpha)))
    return min(scores)[2]


def _fit_conditional_variants(
    train_x: np.ndarray,
    test_variants: dict[str, np.ndarray],
    train_y: np.ndarray,
    train_groups: np.ndarray,
    test_y: np.ndarray,
    test_groups: np.ndarray,
    alpha_grid: Sequence[float],
) -> tuple[dict[str, float], dict[str, float], float, bool]:
    alpha = _conditional_alpha(
        train_x, train_y, train_groups, alpha_grid,
    )
    model = _fit_conditional_ridge(
        train_x, train_y, train_groups, l2=float(alpha),
    )
    losses: dict[str, float] = {}
    ranks: dict[str, float] = {}
    for name, matrix in test_variants.items():
        metrics = _conditional_metrics(
            test_y, model.score(matrix), test_groups,
        )
        losses[name] = float(metrics["conditional_log_loss"])
        ranks[name] = float(metrics["risk_set_rank_percentile"])
    return losses, ranks, float(alpha), bool(model.converged)


def prequential_heterogeneous_hazard(
    design: HazardDesign,
    *,
    initial_k: int | None = None,
    horizon_minutes: float = 30.0,
    alpha_grid: Sequence[float] = CONDITIONAL_ALPHA_GRID,
    persistent_override: np.ndarray | None = None,
    wrong_time_state: np.ndarray | None = None,
    wrong_time_valid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Equal-seizure-weight OOF risk with TRAIN-only entry-route prototypes."""
    outcome, eligible = horizon_outcome(design, float(horizon_minutes))
    persistent = np.asarray(
        design.persistent_state if persistent_override is None else persistent_override,
        dtype=np.float64,
    )
    memoryless = np.asarray(design.memoryless_state, dtype=np.float64)
    wrong = np.asarray(
        memoryless if wrong_time_state is None else wrong_time_state,
        dtype=np.float64,
    )
    if persistent.shape != memoryless.shape or wrong.shape != memoryless.shape:
        raise ValueError("state representations disagree")
    wrong_valid = np.asarray(
        np.ones(len(design.time_epoch), dtype=bool)
        if wrong_time_valid is None else wrong_time_valid,
        dtype=bool,
    )
    if len(wrong_valid) != len(design.time_epoch):
        raise ValueError("wrong-time validity rows disagree")
    history = np.asarray(
        design.history[:, :min(11, design.history.shape[1])], dtype=np.float64,
    )
    observation = np.asarray(design.current_observation, dtype=np.float64)
    history_base = np.column_stack([history, _causal_context(design)])
    base = np.column_stack([history_base, observation])
    supported = _supported_onsets(design, eligible, float(horizon_minutes))
    if initial_k is None:
        if len(supported) >= 10:
            effective_initial_k = max(2, int(np.floor(0.60 * len(supported))))
            evaluation_tier = "primary_chronological_60_percent_train"
        elif len(supported) >= 5:
            effective_initial_k = 2
            evaluation_tier = "rolling_sensitivity_5_to_9"
        elif len(supported) >= 3:
            effective_initial_k = 2
            evaluation_tier = "descriptive_3_to_4"
        else:
            effective_initial_k = 2
            evaluation_tier = "not_estimable_lt_3"
    else:
        effective_initial_k = int(initial_k)
        evaluation_tier = "explicit_initial_k"
    all_case_rows = np.asarray([row[2] for row in supported], dtype=np.int64)
    horizon_seconds = float(horizon_minutes) * 60.0
    folds: list[dict[str, Any]] = []
    for position in range(effective_initial_k, len(supported)):
        cutoff = float(supported[position - 1][0])
        heldout_time, heldout_segment, heldout_anchor = supported[position]
        train_known = _labels_known_by(
            design, outcome, horizon_seconds=horizon_seconds, cutoff=cutoff,
        )
        test_known = _labels_known_by(
            design, outcome, horizon_seconds=horizon_seconds, cutoff=heldout_time,
        )
        train_population = np.flatnonzero(
            eligible & train_known & (design.time_epoch <= cutoff + 1e-9)
        )
        if (
            len(train_population) < 30
            or float(design.time_epoch[int(heldout_anchor)]) <= cutoff + 1e-9
        ):
            continue
        anchors = np.asarray([row[2] for row in supported[:position]], dtype=np.int64)
        if not np.all(design.time_epoch[anchors] <= cutoff + 1e-9):
            raise ValueError("a held-out seizure anchor entered the route fit")
        # A positive training row must be attributable to a seizure already
        # observed at the outer cutoff.  This is intentionally repeated here
        # rather than inferred from a helper receipt.
        for row in train_population[outcome[train_population] == 1]:
            known = (
                (design.onset_segment == design.segment[row])
                & (design.onset_time > design.time_epoch[row])
                & (design.onset_time <= design.time_epoch[row] + horizon_seconds + 1e-9)
                & (design.onset_time <= cutoff + 1e-9)
            )
            if not bool(np.any(known)):
                raise ValueError("prequential training label uses a future seizure")

        train_risk_rows: list[int] = []
        train_labels: list[int] = []
        train_groups: list[int] = []
        train_control_rows: list[int] = []
        for risk_group, (_, _, case_row) in enumerate(supported[:position]):
            controls = _risk_set_controls(
                design, outcome=outcome, eligible=eligible,
                case_row=int(case_row), all_case_rows=all_case_rows,
                label_known=train_known, maximum_time=cutoff,
                minimum_time_exclusive=None,
                controls_per_risk_set=CONTROLS_PER_RISK_SET,
            )
            if len(controls) != CONTROLS_PER_RISK_SET:
                continue
            rows = np.r_[int(case_row), controls]
            train_risk_rows.extend(int(row) for row in rows)
            train_labels.extend([1] + [0] * len(controls))
            train_groups.extend([int(risk_group)] * len(rows))
            train_control_rows.extend(int(row) for row in controls)
        if len(set(train_groups)) < 2:
            continue
        test_controls = _risk_set_controls(
            design, outcome=outcome, eligible=eligible,
            case_row=int(heldout_anchor), all_case_rows=all_case_rows,
            label_known=test_known, maximum_time=heldout_time,
            minimum_time_exclusive=cutoff,
            controls_per_risk_set=CONTROLS_PER_RISK_SET,
        )
        if len(test_controls) != CONTROLS_PER_RISK_SET:
            continue
        train = np.asarray(train_risk_rows, dtype=np.int64)
        train_y = np.asarray(train_labels, dtype=np.int8)
        train_group = np.asarray(train_groups, dtype=np.int64)
        test = np.r_[int(heldout_anchor), test_controls].astype(np.int64, copy=False)
        test_y = np.asarray([1] + [0] * len(test_controls), dtype=np.int8)
        test_group = np.zeros(len(test), dtype=np.int64)
        if set(train.tolist()).intersection(test.tolist()):
            raise ValueError("a time row entered both outer TRAIN and TEST risk sets")
        observation_map = fit_route_feature_map(
            observation, train_rows=train_population, prior_event_anchor_rows=anchors,
        )
        persistent_map = fit_route_feature_map(
            persistent, train_rows=train_population, prior_event_anchor_rows=anchors,
        )
        single_persistent_map = fit_route_feature_map(
            persistent, train_rows=train_population, prior_event_anchor_rows=anchors,
            maximum_routes=1,
        )
        memoryless_map = fit_route_feature_map(
            memoryless, train_rows=train_population, prior_event_anchor_rows=anchors,
        )
        obs_train = observation_map.transform(observation[train])
        obs_test = observation_map.transform(observation[test])
        state_train = persistent_map.transform(persistent[train])
        state_test = persistent_map.transform(persistent[test])
        single_state_train = single_persistent_map.transform_single_axis(persistent[train])
        single_state_test = single_persistent_map.transform_single_axis(persistent[test])
        wrong_test = persistent_map.transform(wrong[test])
        mem_train = memoryless_map.transform(memoryless[train])
        mem_test = memoryless_map.transform(memoryless[test])
        matrices = {
            "history": (history_base[train], history_base[test]),
            "observation": (base[train], base[test]),
            "linear_state": (
                np.column_stack([base[train], persistent[train]]),
                np.column_stack([base[test], persistent[test]]),
            ),
            "route_observation": (
                np.column_stack([base[train], obs_train]),
                np.column_stack([base[test], obs_test]),
            ),
            "route_memoryless": (
                np.column_stack([base[train], obs_train, mem_train]),
                np.column_stack([base[test], obs_test, mem_test]),
            ),
            "single_axis_state": (
                np.column_stack([base[train], obs_train, single_state_train]),
                np.column_stack([base[test], obs_test, single_state_test]),
            ),
        }
        losses: dict[str, float] = {}
        ranks: dict[str, float] = {}
        alphas: dict[str, float] = {}
        converged: dict[str, bool] = {}
        for name, (fit_x, score_x) in matrices.items():
            arm_loss, arm_rank, alpha, arm_converged = _fit_conditional_variants(
                fit_x, {name: score_x}, train_y, train_group,
                test_y, test_group, alpha_grid,
            )
            losses[name] = arm_loss[name]
            ranks[name] = arm_rank[name]
            alphas[name] = float(alpha)
            converged[name] = bool(arm_converged)
        route_train = np.column_stack([base[train], obs_train, state_train])
        route_test = np.column_stack([base[test], obs_test, state_test])
        wrong_route_test = np.column_stack([base[test], obs_test, wrong_test])
        route_variants = {"route_state": route_test}
        fold_wrong_time_valid = bool(np.all(wrong_valid[test]))
        if fold_wrong_time_valid:
            route_variants["route_state_wrong_time"] = wrong_route_test
        route_losses, route_ranks, route_alpha, route_converged = _fit_conditional_variants(
            route_train,
            route_variants,
            train_y, train_group, test_y, test_group, alpha_grid,
        )
        losses.update(route_losses)
        ranks.update(route_ranks)
        alphas["route_state"] = route_alpha
        converged["route_state"] = route_converged
        heldout_route = int(persistent_map.assign(persistent[[heldout_anchor]])[0])
        state_proximity = (
            state_test[:, 0] if persistent_map.n_routes == 1 else state_test[:, 2]
        )
        observation_proximity = (
            obs_test[:, 0] if observation_map.n_routes == 1 else obs_test[:, 2]
        )
        memoryless_proximity = (
            mem_test[:, 0] if memoryless_map.n_routes == 1 else mem_test[:, 2]
        )
        separation = 0.0
        if persistent_map.n_routes == 2:
            separation = float(np.linalg.norm(
                persistent_map.route_centres[0] - persistent_map.route_centres[1]
            ) / max(persistent_map.bandwidth, 1e-8))
        folds.append({
            "heldout_seizure_rank": int(position + 1),
            "heldout_onset_epoch": heldout_time,
            "heldout_segment": heldout_segment,
            "heldout_anchor_row": heldout_anchor,
            "train_cutoff_epoch": cutoff,
            "n_train_rows": int(len(train)), "n_test_rows": int(len(test)),
            "n_train_risk_sets": int(len(set(train_groups))),
            "n_train_controls": int(len(train_control_rows)),
            "n_test_risk_sets": 1,
            "n_test_controls": int(len(test_controls)),
            "n_test_controls_same_segment": int(np.sum(
                design.segment[test_controls] == int(heldout_segment)
            )),
            "test_control_source_indices": [
                int(value) for value in design.source_index[test_controls]
            ],
            "control_selection_uses_history_observation_or_state": False,
            "train_test_rows_disjoint": True,
            "identical_risk_set_rows_across_arms": True,
            "n_prior_seizures": int(len(anchors)),
            "persistent_n_routes": persistent_map.n_routes,
            "persistent_route_sizes": persistent_map.route_sizes.tolist(),
            "persistent_route_separation_bandwidth": separation,
            "heldout_route": heldout_route,
            "state_route_proximity_case_minus_control_median": float(
                state_proximity[0] - np.median(state_proximity[1:])
            ),
            "observation_route_proximity_case_minus_control_median": float(
                observation_proximity[0] - np.median(observation_proximity[1:])
            ),
            "memoryless_route_proximity_case_minus_control_median": float(
                memoryless_proximity[0] - np.median(memoryless_proximity[1:])
            ),
            "route_fit_outer_training_only": True,
            "heldout_seizure_did_not_define_route": True,
            "training_labels_known_by_cutoff": True,
            "test_labels_known_by_heldout_onset": True,
            "wrong_time_all_test_rows_valid": fold_wrong_time_valid,
            "fold_is_statistical_unit": True,
            "loss_metric": "conditional_risk_set_log_loss",
            **{f"logloss_{name}": value for name, value in losses.items()},
            **{f"risk_set_rank_percentile_{name}": value for name, value in ranks.items()},
            **{f"alpha_{name}": value for name, value in alphas.items()},
            **{f"converged_{name}": value for name, value in converged.items()},
        })
    if not folds:
        return {
            "status": "NOT_ESTIMABLE", "horizon_minutes": float(horizon_minutes),
            "initial_k": effective_initial_k,
            "initial_training_rule": evaluation_tier,
            "n_supported_seizures": len(supported),
            "n_oof_seizures": 0, "folds": [],
        }
    names = (
        "history", "observation", "linear_state", "route_observation",
        "route_state", "route_memoryless", "single_axis_state",
    )
    aggregate = {
        name: float(np.mean([row[f"logloss_{name}"] for row in folds]))
        for name in names
    }
    fold_effects = []
    for row in folds:
        fold_effects.append({
            "heldout_seizure_rank": row["heldout_seizure_rank"],
            "observation_minus_history": (
                row["logloss_observation"] - row["logloss_history"]
            ),
            "route_state_minus_history": (
                row["logloss_route_state"] - row["logloss_history"]
            ),
            "route_state_minus_observation": (
                row["logloss_route_state"] - row["logloss_route_observation"]
            ),
            "route_state_minus_memoryless": (
                row["logloss_route_state"] - row["logloss_route_memoryless"]
            ),
            "route_state_minus_linear_state": (
                row["logloss_route_state"] - row["logloss_linear_state"]
            ),
            "two_route_minus_single_axis_state": (
                row["logloss_route_state"] - row["logloss_single_axis_state"]
            ),
            "correct_minus_wrong_time": (
                row["logloss_route_state"] - row["logloss_route_state_wrong_time"]
                if "logloss_route_state_wrong_time" in row else None
            ),
        })
    effect_names = tuple(
        key for key in fold_effects[0]
        if key not in {"heldout_seizure_rank", "two_route_minus_single_axis_state"}
    )
    effects = {}
    median_effects = {}
    for name in effect_names:
        values = [row[name] for row in fold_effects if row[name] is not None]
        effects[name] = float(np.mean(values)) if values else None
        median_effects[name] = float(np.median(values)) if values else None
    two_route_effects = [
        effect["two_route_minus_single_axis_state"]
        for effect, fold in zip(fold_effects, folds)
        if fold["persistent_n_routes"] == 2
    ]
    effects["two_route_minus_single_axis_state"] = (
        float(np.mean(two_route_effects)) if two_route_effects else None
    )
    median_effects["two_route_minus_single_axis_state"] = (
        float(np.median(two_route_effects)) if two_route_effects else None
    )
    heldout_routes = [int(row["heldout_route"]) for row in folds]
    return {
        "status": "COMPLETE_DEVELOPMENT",
        "horizon_minutes": float(horizon_minutes), "initial_k": effective_initial_k,
        "initial_training_rule": evaluation_tier,
        "n_supported_seizures": len(supported), "n_oof_seizures": len(folds),
        "n_two_route_folds": int(sum(row["persistent_n_routes"] == 2 for row in folds)),
        "n_two_route_heterogeneity_folds": len(two_route_effects),
        "n_wrong_time_estimable_folds": int(sum(
            row["wrong_time_all_test_rows_valid"] for row in folds
        )),
        "n_heldout_routes_observed": len(set(heldout_routes)),
        "equal_seizure_weight_logloss": aggregate,
        "equal_seizure_weight_effects": effects,
        "median_seizure_effects": median_effects,
        "folds": folds, "fold_effects": fold_effects,
        "patient_is_inference_unit": True,
        "heldout_seizure_is_within_patient_unit": True,
        "seeds_are_not_patient_replicates": True,
        "route_fit_outer_training_only": True,
        "primary_metric": "conditional_risk_set_log_loss",
        "controls_per_risk_set": CONTROLS_PER_RISK_SET,
        "identical_risk_set_rows_across_arms": True,
        "control_selection_uses_history_observation_or_state": False,
        "minimum_route_separation_bandwidth": MINIMUM_ROUTE_SEPARATION_BANDWIDTH,
    }


def circular_shift_state_within_segment(
    design: HazardDesign, values: np.ndarray, fraction: float,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    result = np.array(matrix, copy=True)
    for label in np.unique(design.segment):
        rows = np.flatnonzero(design.segment == label)
        if len(rows) < 2:
            continue
        shift = max(1, int(round(float(fraction) * len(rows)))) % len(rows)
        result[rows] = matrix[np.roll(rows, shift)]
    return result


def _fit_entry_routes(
    endpoints: np.ndarray, directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    values = np.asarray(endpoints, dtype=np.float64)
    vectors = np.asarray(directions, dtype=np.float64)
    if len(values) >= 4:
        # Two seizures may terminate in the same region but approach it along
        # different directions.  Give endpoint position and unit direction
        # equal aggregate weight in a deterministic joint clustering space.
        centre = np.mean(values, axis=0)
        scale = np.std(values, axis=0)
        active = scale > 1e-8
        endpoint_score = np.zeros_like(values)
        if np.any(active):
            endpoint_score[:, active] = (
                values[:, active] - centre[active]
            ) / scale[active]
        endpoint_score /= np.sqrt(max(endpoint_score.shape[1], 1))
        direction_score = vectors / np.sqrt(max(vectors.shape[1], 1))
        joint = np.column_stack([endpoint_score, direction_score])
        joint_centres, labels = _two_means(joint)
        sizes = np.asarray([np.sum(labels == group) for group in (0, 1)], dtype=np.int64)
        if int(np.min(sizes)) < 2:
            labels = np.zeros(len(values), dtype=np.int64)
            sizes = np.asarray([len(values)], dtype=np.int64)
            joint_centres = np.mean(joint, axis=0, keepdims=True)
    else:
        labels = np.zeros(len(values), dtype=np.int64)
        sizes = np.asarray([len(values)], dtype=np.int64)
        joint = np.column_stack([values, vectors])
        joint_centres = np.mean(joint, axis=0, keepdims=True)
    centres = np.vstack([
        np.mean(values[labels == group], axis=0) for group in range(len(sizes))
    ])
    route_direction = []
    for group in range(len(centres)):
        direction = np.mean(vectors[labels == group], axis=0)
        norm = float(np.linalg.norm(direction))
        route_direction.append(direction / norm if norm > 1e-10 else np.zeros_like(direction))
    separation = 0.0
    if len(joint_centres) == 2:
        separation = float(np.linalg.norm(joint_centres[0] - joint_centres[1]))
    return centres, np.asarray(route_direction), labels, sizes, separation


def evaluate_oos_route_geometry_fold(
    *,
    grid_time: np.ndarray,
    grid_segment: np.ndarray,
    grid_decoder: np.ndarray,
    onset_time: np.ndarray,
    onset_segment: np.ndarray,
    heldout_position: int,
    lookback_minutes: float = 30.0,
    maximum_controls: int = 20,
    grid_spacing_seconds: float = 300.0,
    clean_interictal_exclusion_minutes: float = 120.0,
) -> dict[str, Any]:
    """OOS geometry where prior seizures may define two distinct entry routes."""
    time = np.asarray(grid_time, dtype=np.float64)
    segment = np.asarray(grid_segment, dtype=np.int64)
    decoder = np.asarray(grid_decoder, dtype=np.float64)
    onset = np.asarray(onset_time, dtype=np.float64)
    onset_group = np.asarray(onset_segment, dtype=np.int64)
    order = np.argsort(onset, kind="stable")
    onset, onset_group = onset[order], onset_group[order]
    position = int(heldout_position)
    if position < 2 or position >= len(onset):
        return {"status": "NOT_ESTIMABLE", "reason": "insufficient_prior_seizures"}
    cutoff, heldout = float(onset[position - 1]), float(onset[position])
    heldout_segment = int(onset_group[position])
    clean = time <= cutoff
    exclusion = float(clean_interictal_exclusion_minutes) * 60.0
    for event in onset[:position]:
        clean &= np.abs(time - float(event)) > exclusion
    train = np.flatnonzero(clean)
    if len(train) < 40:
        return {"status": "NOT_ESTIMABLE", "reason": "insufficient_past_full_grid_rows"}
    try:
        projection = fit_decoder_projection(decoder[train])
    except ValueError as error:
        if "no active dimensions" in str(error) or "rank zero" in str(error):
            return {"status": "NOT_ESTIMABLE", "reason": "collapsed_decoder_geometry"}
        raise
    score = projection.transform(decoder)
    train_score = score[train]
    basins = fit_two_basins(train_score)
    maximum_gap = max(1.2 * float(grid_spacing_seconds), 360.0)
    endpoints, directions = [], []
    for event, label in zip(onset[:position], onset_group[:position]):
        rows = _trajectory_rows(
            time, segment, endpoint=float(event), label=int(label),
            lookback_minutes=float(lookback_minutes), maximum_gap_seconds=maximum_gap,
            minimum_coverage_fraction=0.55,
        )
        if not len(rows):
            continue
        displacement = score[rows[-1]] - score[rows[0]]
        norm = float(np.linalg.norm(displacement))
        if norm > 1e-10:
            endpoints.append(score[rows[-1]])
            directions.append(displacement / norm)
    if len(endpoints) < 2:
        return {"status": "NOT_ESTIMABLE", "reason": "insufficient_prior_entry_trajectories"}
    route_centres, route_directions, route_labels, route_sizes, route_separation = _fit_entry_routes(
        np.asarray(endpoints), np.asarray(directions),
    )
    case_rows = _trajectory_rows(
        time, segment, endpoint=heldout, label=heldout_segment,
        lookback_minutes=float(lookback_minutes), maximum_gap_seconds=maximum_gap,
        minimum_coverage_fraction=0.55,
    )
    if not len(case_rows):
        return {"status": "NOT_ESTIMABLE", "reason": "heldout_trajectory_incomplete"}
    controls = matched_control_trajectories(
        time, segment, case_onset=heldout, lookback_minutes=float(lookback_minutes),
        maximum_controls=int(maximum_controls), maximum_endpoint=cutoff,
        forbidden_onsets=onset[:position + 1], maximum_gap_seconds=maximum_gap,
        minimum_coverage_fraction=0.55,
    )
    if len(controls) < 5:
        return {"status": "NOT_ESTIMABLE", "reason": "fewer_than_five_matched_controls"}
    reference = train_score
    if len(reference) > 1500:
        take = np.linspace(0, len(reference) - 1, 1500).round().astype(np.int64)
        reference = reference[take]
    reference_nn = _reference_nn_distance(reference)

    def features(rows: np.ndarray) -> tuple[int, dict[str, float]]:
        endpoint = score[rows[-1]]
        route = int(np.argmin(np.sum((route_centres - endpoint) ** 2, axis=1)))
        basin = int(np.argmin(np.sum((basins - route_centres[route]) ** 2, axis=1)))
        return route, trajectory_features(
            time[rows], score[rows], centres=basins, entry_basin=basin,
            entry_centroid=route_centres[route], entry_direction=route_directions[route],
            reference_scores=reference, reference_nn=reference_nn,
        )

    case_route, case = features(case_rows)
    control_rows = [features(rows) for rows in controls]
    control_features = [row[1] for row in control_rows]
    effects = {
        key: _effect(case[key], [row[key] for row in control_features])
        for key in case
    }
    family_scores = {
        "route_basin_gating": float(np.mean([
            effects["entry_basin_occupancy"]["signed_percentile"],
            effects["entry_basin_longest_dwell_minutes"]["signed_percentile"],
        ])),
        "route_directed_approach": float(np.mean([
            effects["approach_rate_per_minute"]["signed_percentile"],
            effects["flow_alignment"]["signed_percentile"],
        ])),
        "abrupt_transition": effects["max_off_manifold_z"]["signed_percentile"],
    }
    pooled_direction = np.mean(np.asarray(directions), axis=0)
    return {
        "status": "COMPLETE_DEVELOPMENT", "heldout_position": position,
        "heldout_onset_epoch": heldout, "heldout_segment": heldout_segment,
        "train_cutoff_epoch": cutoff, "n_past_full_grid_rows": int(len(train)),
        "n_prior_entry_trajectories": len(endpoints), "n_controls": len(controls),
        "n_routes": int(len(route_centres)), "route_sizes": route_sizes.tolist(),
        "route_joint_endpoint_direction_separation": route_separation,
        "heldout_route": case_route,
        "n_control_routes_observed": len(set(int(row[0]) for row in control_rows)),
        "single_route_mean_direction_norm": float(np.linalg.norm(pooled_direction)),
        "route_fit_outer_training_only": True,
        "heldout_seizure_did_not_define_route": True,
        "projection_fit_clean_interictal_only": True,
        "case_features": case, "effects": effects, "family_scores": family_scores,
        "family_score_scale": "matched_control_signed_percentile_in_minus1_plus1",
    }
