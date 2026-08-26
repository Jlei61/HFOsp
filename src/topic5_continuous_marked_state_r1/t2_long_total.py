"""Long-window total-effect instrument for cumulative IED exposure.

The model starts every arm from the same observation-informed T1 pre-event
state.  The frozen T1 generator supplies natural flow.  A 16-parameter linear
edge maps event occurrence and TRAIN-residualised load to state jumps.  The
edge is fitted in the frozen T1 decoder space, not by latent Euclidean distance.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import expm


LONG_TOTAL_REVISION = (
    "t2_long_total_effect_decoder_space_v4_scaled_ridge_estimability_guarded"
)


@dataclass(frozen=True)
class LongWindowDesign:
    start_index: np.ndarray
    end_index: np.ndarray
    split: np.ndarray
    duration_hours: np.ndarray
    n_events: np.ndarray
    start_state: np.ndarray
    target_state: np.ndarray
    natural_state: np.ndarray
    real_operator: np.ndarray
    delayed_operator: np.ndarray
    exposure_scale_events: float
    delay_events: int
    window_kind: str
    exposure_memory: str

    def validate(self) -> None:
        n = len(self.start_index)
        one_dimensional = (
            self.end_index, self.split, self.duration_hours, self.n_events,
        )
        if any(len(value) != n for value in one_dimensional):
            raise ValueError("long-window scalar arrays disagree")
        if any(len(value) != n for value in (
            self.start_state, self.target_state, self.natural_state,
            self.real_operator, self.delayed_operator,
        )):
            raise ValueError("long-window tensor arrays disagree")
        if self.start_state.ndim != 2:
            raise ValueError("long-window states must be two dimensional")
        if self.target_state.shape != self.start_state.shape:
            raise ValueError("long-window target-state shape mismatch")
        if self.natural_state.shape != self.start_state.shape:
            raise ValueError("long-window natural-state shape mismatch")
        dim = self.start_state.shape[1]
        if (self.real_operator.ndim != 3
                or self.real_operator.shape[:2] != (n, dim)
                or self.real_operator.shape[2] < 2 * dim
                or self.real_operator.shape[2] % dim):
            raise ValueError("long-window real operator shape mismatch")
        if self.delayed_operator.shape != self.real_operator.shape:
            raise ValueError("long-window delayed operator shape mismatch")
        if np.any(self.end_index <= self.start_index):
            raise ValueError("long window is empty or reversed")
        if np.any(self.duration_hours <= 0) or np.any(self.n_events <= 0):
            raise ValueError("long window has non-positive support")
        if set(np.unique(self.split).tolist()) - {0, 1}:
            raise ValueError("long window contains an unknown split")
        for value in (
            self.start_state, self.target_state, self.natural_state,
            self.real_operator, self.delayed_operator,
        ):
            if not np.isfinite(value).all():
                raise ValueError("long-window design is non-finite")


def _coverage_positions(event_segment: np.ndarray) -> dict[int, np.ndarray]:
    segment = np.asarray(event_segment, dtype=np.int64)
    return {
        int(label): np.flatnonzero(segment == label)
        for label in np.unique(segment)
    }


def _prefix_operators(event_time: np.ndarray, event_segment: np.ndarray,
                      innovation: np.ndarray, matrix: np.ndarray, *,
                      delay_events: int) -> tuple[np.ndarray, np.ndarray,
                                                  dict[int, np.ndarray]]:
    """Contribution before each event, accumulated in one linear pass."""
    time = np.asarray(event_time, dtype=np.float64)
    segment = np.asarray(event_segment, dtype=np.int64)
    innovation = np.asarray(innovation, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    if innovation.ndim == 1:
        innovation = innovation[:, None]
    if (time.ndim != 1 or segment.shape != time.shape
            or innovation.ndim != 2 or innovation.shape[0] != len(time)):
        raise ValueError("prefix-operator event arrays disagree")
    dim = matrix.shape[0]
    if matrix.shape != (dim, dim):
        raise ValueError("generator matrix is not square")
    width = (1 + innovation.shape[1]) * dim
    real = np.zeros((len(time), dim, width), dtype=np.float64)
    delayed = np.zeros_like(real)
    identity = np.eye(dim, dtype=np.float64)
    positions = _coverage_positions(segment)
    for index in positions.values():
        if len(index) < 2:
            continue
        if not np.all(np.diff(time[index]) > 0):
            raise ValueError("events are not strictly ordered within coverage")
        for local in range(len(index) - 1):
            current = int(index[local])
            following = int(index[local + 1])
            delta = (time[following] - time[current]) / 60.0
            transition = expm(matrix * delta)
            real_jump = np.concatenate([
                identity,
                *(identity * value for value in innovation[current]),
            ], axis=1)
            delayed_value = (
                innovation[int(index[local - int(delay_events)])]
                if local >= int(delay_events)
                else np.zeros(innovation.shape[1], dtype=np.float64)
            )
            delayed_jump = np.concatenate([
                identity,
                *(identity * value for value in delayed_value),
            ], axis=1)
            real[following] = transition @ (real[current] + real_jump)
            delayed[following] = transition @ (delayed[current] + delayed_jump)
    return real, delayed, positions


def _boxcar_prefix_operators(event_segment: np.ndarray,
                             innovation: np.ndarray, *, dim: int,
                             delay_events: int
                             ) -> tuple[np.ndarray, np.ndarray,
                                        dict[int, np.ndarray]]:
    """Unweighted whole-window counts and load sums before every event.

    Unlike ``_prefix_operators``, this accumulator is not decayed by the T1
    generator.  It therefore tests the requested event-count memory directly.
    The delayed arm uses the same-size exposure window shifted causally by
    ``delay_events``; the occurrence column remains matched exactly.
    """
    innovation = np.asarray(innovation, dtype=np.float64)
    segment = np.asarray(event_segment, dtype=np.int64)
    if innovation.ndim == 1:
        innovation = innovation[:, None]
    if (innovation.ndim != 2 or segment.ndim != 1
            or innovation.shape[0] != len(segment)):
        raise ValueError("boxcar event arrays disagree")
    width = (1 + innovation.shape[1]) * int(dim)
    real = np.zeros((len(innovation), int(dim), width), dtype=np.float64)
    delayed = np.zeros_like(real)
    identity = np.eye(int(dim), dtype=np.float64)
    positions = _coverage_positions(segment)
    for index in positions.values():
        local_value = innovation[index]
        prefix = np.vstack([
            np.zeros((1, innovation.shape[1]), dtype=np.float64),
            np.cumsum(local_value, axis=0, dtype=np.float64),
        ])
        for local, global_index in enumerate(index):
            occurrence = float(local)
            real[int(global_index)] = np.concatenate([
                identity * occurrence,
                *(identity * value for value in prefix[local]),
            ], axis=1)
            delayed_stop = max(local - int(delay_events), 0)
            delayed[int(global_index)] = np.concatenate([
                identity * occurrence,
                *(identity * value for value in prefix[delayed_stop]),
            ], axis=1)
    return real, delayed, positions


def _window_pairs(event_time: np.ndarray, event_split: np.ndarray,
                  positions: dict[int, np.ndarray], *, window_kind: str,
                  scale_events: int, duration_hours: float,
                  delay_events: int, coverage_start: np.ndarray | None
                  ) -> tuple[np.ndarray, np.ndarray]:
    starts: list[int] = []
    ends: list[int] = []
    for label, index in positions.items():
        local_time = event_time[index]
        for end_local in range(len(index)):
            if window_kind == "event_count" or window_kind.startswith("event_count_"):
                start_local = end_local - int(scale_events)
                if start_local < int(delay_events):
                    continue
            elif window_kind == "physical_6h":
                requested = float(local_time[end_local]) - float(duration_hours) * 3600.0
                if coverage_start is None or requested < float(coverage_start[label]) - 1e-6:
                    continue
                start_local = int(np.searchsorted(local_time, requested, side="left"))
                if start_local >= end_local or start_local < int(delay_events):
                    continue
            else:
                raise ValueError(f"unknown long-window kind {window_kind!r}")
            start = int(index[start_local])
            end = int(index[end_local])
            # Only the endpoint assigns the row to TRAIN/validation.  Its entire
            # exposure history is causal and may precede the split boundary.
            if event_split[end] not in (0, 1):
                continue
            starts.append(start)
            ends.append(end)
    if not starts:
        raise ValueError(f"no observable windows for {window_kind}")
    return np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64)


def build_long_window_design(event_time: np.ndarray, event_split: np.ndarray,
                             event_segment: np.ndarray,
                             pre_event_state: np.ndarray,
                             innovation: np.ndarray,
                             generator_matrix: np.ndarray,
                             generator_mu: np.ndarray, *,
                             window_kind: str,
                             scale_events: int = 10000,
                             duration_hours: float = 6.0,
                             delay_events: int = 1000,
                             coverage_start: np.ndarray | None = None,
                             exposure_memory: str = "generator_weighted",
                             ) -> LongWindowDesign:
    """Build real and causal-delayed long exposure operators on identical rows."""
    time = np.asarray(event_time, dtype=np.float64)
    split = np.asarray(event_split, dtype=np.int8)
    state = np.asarray(pre_event_state, dtype=np.float64)
    matrix = np.asarray(generator_matrix, dtype=np.float64)
    mu = np.asarray(generator_mu, dtype=np.float64)
    if state.shape[0] != len(time) or state.ndim != 2:
        raise ValueError("pre-event state denominator disagrees")
    if matrix.shape != (state.shape[1], state.shape[1]) or mu.shape != (state.shape[1],):
        raise ValueError("frozen generator shape disagrees with state")
    if exposure_memory == "generator_weighted":
        prefix_real, prefix_delayed, positions = _prefix_operators(
            time, event_segment, innovation, matrix,
            delay_events=int(delay_events),
        )
    elif exposure_memory == "boxcar":
        prefix_real, prefix_delayed, positions = _boxcar_prefix_operators(
            event_segment, innovation, dim=state.shape[1],
            delay_events=int(delay_events),
        )
    else:
        raise ValueError(f"unknown exposure memory {exposure_memory!r}")
    start, end = _window_pairs(
        time, split, positions, window_kind=window_kind,
        scale_events=int(scale_events), duration_hours=float(duration_hours),
        delay_events=int(delay_events), coverage_start=coverage_start,
    )
    duration_minutes = (time[end] - time[start]) / 60.0
    transitions = np.stack([expm(matrix * value) for value in duration_minutes])
    natural = mu + np.einsum("nij,nj->ni", transitions, state[start] - mu)
    if exposure_memory == "generator_weighted":
        real = prefix_real[end] - np.einsum(
            "nij,njk->nik", transitions, prefix_real[start]
        )
        delayed = prefix_delayed[end] - np.einsum(
            "nij,njk->nik", transitions, prefix_delayed[start]
        )
    else:
        real = prefix_real[end] - prefix_real[start]
        delayed = prefix_delayed[end] - prefix_delayed[start]
    n_events = end - start
    train_count = n_events[split[end] == 0]
    if not len(train_count):
        raise ValueError("long-window design has no TRAIN endpoint")
    exposure_scale = (
        float(scale_events)
        if window_kind == "event_count" or window_kind.startswith("event_count_")
        else float(np.median(train_count))
    )
    exposure_scale = max(exposure_scale, 1.0)
    real /= math.sqrt(exposure_scale)
    delayed /= math.sqrt(exposure_scale)
    result = LongWindowDesign(
        start_index=start, end_index=end, split=split[end],
        duration_hours=(duration_minutes / 60.0).astype(np.float64),
        n_events=n_events.astype(np.int64),
        start_state=state[start], target_state=state[end], natural_state=natural,
        real_operator=real, delayed_operator=delayed,
        exposure_scale_events=float(exposure_scale),
        delay_events=int(delay_events), window_kind=str(window_kind),
        exposure_memory=str(exposure_memory),
    )
    result.validate()
    return result


SCALE_FLOOR = 1e-4


def intercept_operator(design: LongWindowDesign) -> np.ndarray:
    """A window-independent state offset with no exposure content.

    The occurrence columns of ``real_operator``/``delayed_operator`` are the
    exponentially weighted event count ``sum_j exp(K (t_e - t_j))``.  Once the
    window is several generator time constants long that sum saturates, so both
    exposure arms carry a nearly constant block: they own a free state-space
    intercept that ``no_edge_natural_flow`` does not have.  Any systematic
    offset between the frozen natural flow and the observed target therefore
    makes ``real`` beat ``no_edge`` without a single bit of exposure
    information.  This arm restores that intercept to the reference so the
    comparison measures exposure-driven *variation*.
    """
    dim = int(design.start_state.shape[1])
    n = int(len(design.start_index))
    return np.broadcast_to(np.eye(dim, dtype=np.float64), (n, dim, dim)).copy()


def occurrence_block_variation(operator: np.ndarray, rows: np.ndarray | None = None
                               ) -> dict:
    """How much of the exposure operator is constant across windows."""
    operator = np.asarray(operator, dtype=np.float64)
    dim = operator.shape[1]
    block = operator[:, :, :dim]
    if rows is not None:
        block = block[np.asarray(rows, dtype=np.int64)]
    norm = np.linalg.norm(block.reshape(len(block), -1), axis=1)
    mean = float(np.mean(norm))
    return {
        "occurrence_block_norm_mean": mean,
        "occurrence_block_norm_sd": float(np.std(norm)),
        "coefficient_of_variation": (
            float(np.std(norm) / mean) if mean > 1e-12 else None
        ),
        "constant_fraction_warning": (
            "occurrence columns are near-constant; real-minus-no-edge is then a "
            "free intercept, not exposure evidence"
        ),
    }


def effective_memory_audit(event_time: np.ndarray, start_index: np.ndarray,
                           end_index: np.ndarray, generator_matrix: np.ndarray,
                           *, max_windows: int = 64) -> dict:
    """Report the exposure kernel the frozen generator actually imposes.

    A window may nominally span 10,000 events or six hours, but every event's
    contribution decays as ``exp(K dt)``.  When the slowest generator mode has a
    time constant far below the window length the operator is saturated: it
    measures recent exposure, not the whole window.  Reporting the nominal
    window as the tested time scale would overstate the experiment.
    """
    time = np.asarray(event_time, dtype=np.float64)
    start = np.asarray(start_index, dtype=np.int64)
    end = np.asarray(end_index, dtype=np.int64)
    matrix = np.asarray(generator_matrix, dtype=np.float64)
    eigenvalues = np.linalg.eigvals(matrix).real
    if np.any(eigenvalues >= 0.0):
        raise ValueError("generator is not strictly stable")
    slowest = float(-1.0 / eigenvalues.max())
    fastest = float(-1.0 / eigenvalues.min())
    take = np.unique(np.linspace(0, len(start) - 1, min(int(max_windows), len(start))).astype(int))
    effective, half, ninety, beyond_hour = [], [], [], []
    for row in take:
        lo, hi = int(start[row]), int(end[row])
        age = (time[hi] - time[lo:hi]) / 60.0
        weight = np.exp(-age / slowest)
        total = float(weight.sum())
        effective.append(total)
        order = np.argsort(age)
        cumulative = np.cumsum(weight[order]) / total
        half.append(float(age[order][int(np.searchsorted(cumulative, 0.5))] / 60.0))
        ninety.append(float(age[order][int(np.searchsorted(cumulative, 0.9))] / 60.0))
        beyond_hour.append(float(weight[age > 60.0].sum() / total))
    return {
        "slowest_mode_time_constant_minutes": slowest,
        "fastest_mode_time_constant_minutes": fastest,
        "sampled_windows": int(len(take)),
        "median_effective_weighted_events": float(np.median(effective)),
        "median_nominal_events": float(np.median((end - start)[take])),
        "median_hours_holding_half_the_weight": float(np.median(half)),
        "median_hours_holding_ninety_percent_weight": float(np.median(ninety)),
        "median_weight_fraction_older_than_one_hour": float(np.median(beyond_hour)),
        "interpretation": (
            "the tested exposure time scale is the generator time constant, not "
            "the nominal window length"
        ),
    }


def boxcar_memory_audit(event_time: np.ndarray, start_index: np.ndarray,
                        end_index: np.ndarray, *, max_windows: int = 64) -> dict:
    """Report the actual age support of an equal-weight whole-window exposure."""
    time = np.asarray(event_time, dtype=np.float64)
    start = np.asarray(start_index, dtype=np.int64)
    end = np.asarray(end_index, dtype=np.int64)
    take = np.unique(np.linspace(
        0, len(start) - 1, min(int(max_windows), len(start))
    ).astype(int))
    half, ninety, beyond_hour, count = [], [], [], []
    for row in take:
        age = (time[int(end[row])] - time[int(start[row]):int(end[row])]) / 60.0
        if not len(age):
            continue
        ordered = np.sort(age)
        half.append(float(ordered[min(int(np.ceil(0.5 * len(ordered))) - 1,
                                      len(ordered) - 1)] / 60.0))
        ninety.append(float(ordered[min(int(np.ceil(0.9 * len(ordered))) - 1,
                                        len(ordered) - 1)] / 60.0))
        beyond_hour.append(float(np.mean(age > 60.0)))
        count.append(int(len(age)))
    return {
        "memory_kernel": "equal_weight_boxcar",
        "sampled_windows": int(len(count)),
        "median_effective_weighted_events": float(np.median(count)),
        "median_nominal_events": float(np.median((end - start)[take])),
        "median_hours_holding_half_the_weight": float(np.median(half)),
        "median_hours_holding_ninety_percent_weight": float(np.median(ninety)),
        "median_weight_fraction_older_than_one_hour": float(np.median(beyond_hour)),
        "slowest_mode_time_constant_minutes": None,
        "fastest_mode_time_constant_minutes": None,
        "interpretation": (
            "all events in the requested window carry equal exposure weight; "
            "the frozen T1 generator only defines the no-edge natural-flow target"
        ),
    }


def endpoint_support_audit(event_time: np.ndarray, end_index: np.ndarray,
                           split: np.ndarray, generator_matrix: np.ndarray, *,
                           exposure_memory: str = "generator_weighted",
                           start_index: np.ndarray | None = None) -> dict:
    """Independent-window budget, which overlapping windows badly overstate.

    The decorrelation length is the exposure kernel, not always the generator.
    A boxcar arm weights the whole window equally, so two endpoints stay
    dependent until they are a *window* apart; dividing the endpoint span by the
    54-minute generator time constant would overstate the budget by the ratio of
    the two, which for an eight-hour boxcar is nearly tenfold.
    """
    time = np.asarray(event_time, dtype=np.float64)
    end = np.asarray(end_index, dtype=np.int64)
    split = np.asarray(split, dtype=np.int8)
    matrix = np.asarray(generator_matrix, dtype=np.float64)
    slowest = float(-1.0 / np.linalg.eigvals(matrix).real.max())
    if exposure_memory == "boxcar":
        if start_index is None:
            raise ValueError("boxcar endpoint support needs the window starts")
        start = np.asarray(start_index, dtype=np.int64)
        decorrelation = float(np.median(time[end] - time[start]) / 60.0)
        kernel = "equal_weight_boxcar_window_length"
    else:
        decorrelation = slowest
        kernel = "generator_slowest_mode"
    decorrelation = max(decorrelation, 1e-9)
    result = {
        "decorrelation_minutes": decorrelation,
        "decorrelation_source": kernel,
        "generator_slowest_mode_minutes": slowest,
    }
    for name, code in (("train", 0), ("validation", 1)):
        rows = np.flatnonzero(split == code)
        if not len(rows):
            continue
        endpoint = time[end[rows]]
        span_hours = float((endpoint.max() - endpoint.min()) / 3600.0)
        result[name] = {
            "windows": int(len(rows)),
            "endpoint_span_hours": span_hours,
            "effective_independent_windows": float(
                span_hours * 60.0 / decorrelation
            ),
            "warning": (
                "windows overlap almost completely; the window count is not a "
                "sample size"
            ),
        }
    return result


def delayed_control_overlap(design: LongWindowDesign) -> dict:
    """How much exposure the causal-delayed arm shares with the real arm.

    The delayed arm reads innovations from ``[start-delay, end-delay)``.  When
    the delay is smaller than the window, the two arms sum over overlapping
    events, so ``real - delayed`` compares two nearly identical regressors and a
    null there carries very little information.  A fixed 1000-event delay in a
    10,000-event window leaves 90% of the exposure shared.
    """
    n_events = np.asarray(design.n_events, dtype=np.float64)
    delay = float(design.delay_events)
    shared = np.clip(n_events - delay, 0.0, None) / np.maximum(n_events, 1.0)
    return {
        "delay_events": int(design.delay_events),
        "median_window_events": float(np.median(n_events)),
        "median_shared_exposure_fraction": float(np.median(shared)),
        "max_shared_exposure_fraction": float(np.max(shared)),
        "warning": (
            "the delayed control shares this fraction of its exposure events "
            "with the real arm; a null real-minus-delayed at high overlap is "
            "not evidence that load timing is uninformative"
        ),
    }


def estimability_guard(arm: dict, reference: dict, *,
                       ratio_limit: float = 4.0) -> dict:
    """Did the fitted arm produce an estimate, or extrapolate off the data?

    Every fitted arm nests the reference's constant offset, so on validation it
    should land near the reference or below it.  Landing far above means the
    fit is being evaluated outside the range it was estimated on; the resulting
    contrast measures extrapolation, not exposure.  Such a contrast has to be
    reported as non-estimable, the same way a structural zero is.
    """
    numerator = float(arm["decoder_total_equal_block_mse"])
    denominator = float(reference["decoder_total_equal_block_mse"])
    ratio = numerator / denominator if denominator > 0 else float("inf")
    return {
        "arm_over_reference_ratio": ratio,
        "ratio_limit": float(ratio_limit),
        "estimable": bool(np.isfinite(ratio) and ratio <= float(ratio_limit)),
        "interpretation": (
            "if not estimable the arm scores far worse than the constant it "
            "nests, so the contrast is an extrapolation artefact and must not "
            "be read as an exposure null"
        ),
    }


def target_shift_audit(target_delta: np.ndarray, split: np.ndarray,
                       readout: "DecoderReadout") -> dict:
    """TRAIN-to-validation drift of the quantity every arm has to predict."""
    delta = np.asarray(target_delta, dtype=np.float64)
    split = np.asarray(split, dtype=np.int8)
    projected = delta @ readout.scaled_matrix.T
    train = projected[split == 0]
    validation = projected[split == 1]
    train_sd = float(np.sqrt(np.mean(np.var(train, axis=0)))) if len(train) else 0.0
    train_sd = max(train_sd, 1e-12)
    shift = float(np.linalg.norm(
        validation.mean(axis=0) - train.mean(axis=0)
    ) / (train_sd * math.sqrt(max(projected.shape[1], 1))))
    spread = (
        float(np.sqrt(np.mean(np.var(validation, axis=0))) / train_sd)
        if len(validation) else 0.0
    )
    return {
        "validation_minus_train_mean_shift_in_train_sd": shift,
        "validation_over_train_sd_ratio": spread,
        "warning": (
            "a large shift means the frozen-T1 target at validation endpoints "
            "lies outside the TRAIN range the edge was fitted on; arm contrasts "
            "then reflect extrapolation"
        ),
    }


def nonoverlapping_window_audit(event_time: np.ndarray,
                                start_index: np.ndarray,
                                end_index: np.ndarray,
                                split: np.ndarray) -> dict:
    """Count disjoint full windows by greedy earliest-end interval scheduling."""
    time = np.asarray(event_time, dtype=np.float64)
    start = np.asarray(start_index, dtype=np.int64)
    end = np.asarray(end_index, dtype=np.int64)
    split = np.asarray(split, dtype=np.int8)
    result = {}
    for name, code in (("train", 0), ("validation", 1)):
        rows = np.flatnonzero(split == code)
        rows = rows[np.argsort(time[end[rows]], kind="stable")]
        selected = []
        last_end = -np.inf
        for row in rows:
            if time[start[row]] >= last_end:
                selected.append(int(row))
                last_end = float(time[end[row]])
        result[name] = {
            "windows": int(len(rows)),
            "nonoverlapping_full_windows": int(len(selected)),
            "selected_rows": selected,
        }
    return result


def delayed_union_start_index(start_index: np.ndarray,
                              event_segment: np.ndarray,
                              delay_events: int) -> np.ndarray:
    """Earliest event used by either the real or causal-delayed exposure arm.

    A real N-event window uses ``[start, end)``.  Its parameter-matched delayed
    arm uses innovations from ``[start-delay, end-delay)``.  Independence of
    the *contrast* therefore has to be counted on their union
    ``[start-delay, end)`` rather than on the nominal real window alone.
    """
    start = np.asarray(start_index, dtype=np.int64)
    segment = np.asarray(event_segment, dtype=np.int64)
    delay = int(delay_events)
    if start.ndim != 1 or segment.ndim != 1 or delay < 0:
        raise ValueError("invalid delayed-union support input")
    local_position = np.empty(len(segment), dtype=np.int64)
    positions = _coverage_positions(segment)
    for index in positions.values():
        local_position[index] = np.arange(len(index), dtype=np.int64)
    result = np.empty_like(start)
    for row, value in enumerate(start):
        index = positions[int(segment[value])]
        local = int(local_position[value])
        if local < delay:
            raise ValueError("window lacks the requested causal-delay history")
        result[row] = int(index[local - delay])
    return result


def count_windows_crossing_segment(start_index: np.ndarray, end_index: np.ndarray,
                                   event_segment: np.ndarray) -> int:
    """Verify the no-unrecorded-gap property instead of asserting it."""
    segment = np.asarray(event_segment, dtype=np.int64)
    return int(np.sum(
        segment[np.asarray(start_index, dtype=np.int64)]
        != segment[np.asarray(end_index, dtype=np.int64)]
    ))


@dataclass(frozen=True)
class DecoderReadout:
    blocks: dict[str, np.ndarray]
    scales: dict[str, float]
    scaled_matrix: np.ndarray
    rank: int
    raw_scales: dict[str, float]
    blocks_at_scale_floor: tuple[str, ...]
    degenerate: bool


def decoder_readout(model, target_minus_natural: np.ndarray,
                    train_mask: np.ndarray) -> DecoderReadout:
    """Freeze four equally weighted state-to-event decoder blocks.

    ``rank`` is taken on the *raw* stacked block matrix with numpy's relative
    tolerance.  Ranking the rescaled matrix against an absolute ``tol`` would
    let a numerically dead readout (weights near 1e-10, divided by the 1e-4
    scale floor) report ``rank > 0`` and flip the admissibility gate.  A block
    whose TRAIN target variation lands on the floor contributes no variation to
    the primary metric at all, so ``degenerate`` records that separately: the
    equal-block contract is only honoured when every block scale is real.
    """
    blocks = {
        "timing": model.state_timing.weight.detach().cpu().numpy().astype(np.float64),
        "stop": model.state_size.weight[:1].detach().cpu().numpy().astype(np.float64),
        "selecting_size": model.state_size.weight[1:].detach().cpu().numpy().astype(np.float64),
        "contact_subset": model.state_contact.weight.detach().cpu().numpy().astype(np.float64),
    }
    delta = np.asarray(target_minus_natural, dtype=np.float64)
    train = np.asarray(train_mask, dtype=bool)
    scaled = []
    scales: dict[str, float] = {}
    raw_scales: dict[str, float] = {}
    floored: list[str] = []
    for name, matrix in blocks.items():
        value = delta[train] @ matrix.T
        raw = float(np.sqrt(np.mean(np.var(value, axis=0)))) if value.size else 0.0
        raw_scales[name] = raw
        if raw < SCALE_FLOOR:
            floored.append(name)
        scale = max(raw, SCALE_FLOOR)
        scales[name] = scale
        scaled.append(matrix / (scale * math.sqrt(max(len(matrix), 1))))
    scaled_matrix = np.concatenate(scaled, axis=0)
    raw_matrix = np.concatenate(list(blocks.values()), axis=0)
    rank = int(np.linalg.matrix_rank(raw_matrix))
    # Degenerate means the decoder produces no variation at all, so every arm
    # is forced to the same score.  A *partial* floor does not do that: it
    # breaks the equal-block weighting for those blocks and is reported as a
    # caveat, but the remaining blocks still separate the arms.  Blocking the
    # whole run on a partial floor would be its own failure mode.
    return DecoderReadout(
        blocks=blocks, scales=scales, scaled_matrix=scaled_matrix, rank=rank,
        raw_scales=raw_scales, blocks_at_scale_floor=tuple(floored),
        degenerate=bool(rank == 0 or len(floored) == len(blocks)),
    )


def _solve_ridge(operator: np.ndarray, target_delta: np.ndarray,
                 readout: DecoderReadout, rows: np.ndarray,
                 ridge: float) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    d = readout.scaled_matrix
    x = np.einsum("rd,ndp->nrp", d, operator[rows]).reshape(-1, operator.shape[-1])
    y = (target_delta[rows] @ d.T).reshape(-1)
    gram = x.T @ x
    # Scale-free penalty.  An absolute `ridge * I` is meaningless here: the Gram
    # matrix grows with the number of windows times the number of readout rows,
    # so the whole fixed grid collapsed to "no regularisation" and the search
    # pinned itself at the grid maximum on 47 of 76 archived fits.
    unit = float(np.trace(gram) / max(gram.shape[0], 1))
    unit = unit if unit > 0.0 else 1.0
    penalty = np.eye(x.shape[1], dtype=np.float64) * float(ridge) * unit
    return np.linalg.solve(gram + penalty, x.T @ y)


def fit_decoder_space_edge(operator: np.ndarray, target_delta: np.ndarray,
                           split: np.ndarray, readout: DecoderReadout, *,
                           ridge_grid: tuple[float, ...] = (
                               1e-8, 1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4,
                           ),
                           ) -> tuple[np.ndarray, dict]:
    """Chronological inner-TRAIN ridge selection, then all-TRAIN refit."""
    operator = np.asarray(operator, dtype=np.float64)
    target_delta = np.asarray(target_delta, dtype=np.float64)
    split = np.asarray(split, dtype=np.int8)
    train = np.flatnonzero(split == 0)
    if len(train) < 10:
        raise ValueError("long edge needs at least 10 TRAIN windows")
    cut = int(np.clip(math.floor(0.8 * len(train)), 1, len(train) - 1))
    inner_train, inner_validation = train[:cut], train[cut:]
    scores = {}
    for ridge in ridge_grid:
        theta = _solve_ridge(
            operator, target_delta, readout, inner_train, float(ridge)
        )
        prediction = np.einsum("ndp,p->nd", operator[inner_validation], theta)
        error = (prediction - target_delta[inner_validation]) @ readout.scaled_matrix.T
        scores[str(float(ridge))] = float(np.mean(np.sum(error ** 2, axis=1)))
    best = min(ridge_grid, key=lambda value: (scores[str(float(value))], float(value)))
    theta = _solve_ridge(operator, target_delta, readout, train, float(best))
    return theta, {
        "ridge_grid": [float(value) for value in ridge_grid],
        "inner_train_windows": int(len(inner_train)),
        "inner_validation_windows": int(len(inner_validation)),
        "inner_validation_score": scores,
        "selected_ridge": float(best),
        "selected_ridge_at_grid_boundary": bool(
            float(best) in (float(min(ridge_grid)), float(max(ridge_grid)))
        ),
        "penalty_scaling": "ridge x mean diagonal of the TRAIN Gram matrix",
        "refit_train_windows": int(len(train)),
        "development_validation_used_for_selection": False,
    }


def predict_state(design: LongWindowDesign, operator: np.ndarray,
                  theta: np.ndarray) -> np.ndarray:
    return design.natural_state + np.einsum(
        "ndp,p->nd", np.asarray(operator, dtype=np.float64),
        np.asarray(theta, dtype=np.float64),
    )


def state_prediction_metrics(predicted: np.ndarray, target: np.ndarray,
                             rows: np.ndarray, readout: DecoderReadout) -> dict:
    rows = np.asarray(rows, dtype=np.int64)
    error = np.asarray(predicted, dtype=np.float64)[rows] - np.asarray(
        target, dtype=np.float64
    )[rows]
    block = {}
    for name, matrix in readout.blocks.items():
        value = error @ matrix.T
        block[name] = float(np.mean(value ** 2) / (readout.scales[name] ** 2))
    scaled = error @ readout.scaled_matrix.T
    return {
        "decoder_total_equal_block_mse": float(
            np.mean(np.sum(scaled ** 2, axis=1))
        ),
        "decoder_block_standardised_mse": block,
        "latent_mse_sensitivity": float(np.mean(error ** 2)),
        "n_windows": int(len(rows)),
    }


def metric_contrast(left: dict, right: dict) -> dict:
    """Left minus right; negative decoder loss favours the left arm."""
    result = {
        "decoder_total_equal_block_mse": float(
            left["decoder_total_equal_block_mse"]
            - right["decoder_total_equal_block_mse"]
        ),
        "latent_mse_sensitivity": float(
            left["latent_mse_sensitivity"] - right["latent_mse_sensitivity"]
        ),
        "decoder_block_standardised_mse": {},
    }
    for name in left["decoder_block_standardised_mse"]:
        result["decoder_block_standardised_mse"][name] = float(
            left["decoder_block_standardised_mse"][name]
            - right["decoder_block_standardised_mse"][name]
        )
    return result
