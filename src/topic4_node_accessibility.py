"""Bounded, field-gated Node threshold recovery for Topic 4 rev13."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json

import numpy as np


def build_spatial_shift_permutation(
    positions_xy,
    *,
    support_g=None,
    block_shape: tuple[int, int] = (8, 8),
    shift_blocks: tuple[int, int] | None = None,
) -> np.ndarray:
    """Build a deterministic, support-preserving coarse spatial shift.

    Neurons are ordered by frozen coarse spatial blocks and their within-block
    coordinates.  Source blocks are shifted on a torus and then rank-matched
    to target blocks, yielding a complete one-to-one permutation without an
    assignment-library dependency.  Positive-support and zero-support neurons
    are mapped separately so field-gated activity cannot be moved entirely
    outside the formal Node support.

    The returned permutation follows the controller convention
    ``mapped_state[target] = state[permutation[target]]``.
    """
    positions = np.asarray(positions_xy, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2 or positions.shape[0] < 2:
        raise ValueError("positions_xy must have shape (n_neurons, 2), n >= 2")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions_xy must be finite")

    try:
        n_x, n_y = (int(block_shape[0]), int(block_shape[1]))
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError("block_shape must contain two positive integers") from exc
    if n_x < 1 or n_y < 1 or (n_x == 1 and n_y == 1):
        raise ValueError("block_shape must define at least two spatial blocks")
    if tuple(block_shape) != (n_x, n_y):
        raise ValueError("block_shape must contain integer values")

    if shift_blocks is None:
        shift_x, shift_y = (n_x // 2 if n_x > 1 else 0, 0)
        if shift_x == 0:
            shift_y = n_y // 2
    else:
        try:
            shift_x, shift_y = (int(shift_blocks[0]), int(shift_blocks[1]))
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError("shift_blocks must contain two integers") from exc
        if tuple(shift_blocks) != (shift_x, shift_y):
            raise ValueError("shift_blocks must contain integer values")
    shift_x %= n_x
    shift_y %= n_y
    if shift_x == 0 and shift_y == 0:
        raise ValueError("shift_blocks must specify a non-trivial toroidal shift")

    n_neurons = int(positions.shape[0])
    if support_g is None:
        support_mask = np.ones(n_neurons, dtype=bool)
    else:
        support = np.asarray(support_g, dtype=np.float64)
        if support.shape != (n_neurons,) or not np.all(np.isfinite(support)):
            raise ValueError("support_g must be a finite vector aligned to positions")
        if np.any(support < 0.0):
            raise ValueError("support_g must be non-negative")
        support_mask = support > 0.0

    lower = np.min(positions, axis=0)
    span = np.max(positions, axis=0) - lower
    if (shift_x and span[0] <= 0.0) or (shift_y and span[1] <= 0.0):
        raise ValueError("shifted spatial axes must have non-zero extent")
    safe_span = np.where(span > 0.0, np.nextafter(span, np.inf), 1.0)
    unit = (positions - lower) / safe_span
    block_x = np.minimum((unit[:, 0] * n_x).astype(np.int64), n_x - 1)
    block_y = np.minimum((unit[:, 1] * n_y).astype(np.int64), n_y - 1)
    local_x = unit[:, 0] * n_x - block_x
    local_y = unit[:, 1] * n_y - block_y
    shifted_x = (block_x + shift_x) % n_x
    shifted_y = (block_y + shift_y) % n_y

    permutation = np.empty(n_neurons, dtype=np.int64)
    original_index = np.arange(n_neurons, dtype=np.int64)
    for group_mask in (support_mask, ~support_mask):
        indices = original_index[group_mask]
        if indices.size == 0:
            continue
        target_order = indices[
            np.lexsort(
                (
                    indices,
                    local_x[indices],
                    local_y[indices],
                    block_x[indices],
                    block_y[indices],
                )
            )
        ]
        shifted_source_order = indices[
            np.lexsort(
                (
                    indices,
                    local_x[indices],
                    local_y[indices],
                    shifted_x[indices],
                    shifted_y[indices],
                )
            )
        ]
        permutation[target_order] = shifted_source_order

    if not np.array_equal(np.sort(permutation), original_index):
        raise RuntimeError("spatial shift construction did not produce a permutation")
    if np.array_equal(permutation, original_index):
        raise ValueError("spatial shift is identity on these frozen positions")
    return permutation


class FieldGatedZeroSumNodeRecovery:
    """E-only spike trace that redistributes threshold inside a frozen field.

    ``spatial_shift`` is the formal matched control.  It maps the trace through
    a frozen coarse spatial shift, applies the same zero-sum transform, and
    matches the support-weighted delta SD to the corresponding unshifted
    ``zero_sum`` delta at every evaluation.  ``stratified_shuffle`` remains
    available only for backward compatibility and is not a formal control.
    """

    KIND = "field_gated_zero_sum_node_recovery"
    CHECKPOINT_VERSION = 1
    MODES = frozenset({"zero_sum", "raise_only", "spatial_shift", "stratified_shuffle"})

    def __init__(
        self,
        support_g,
        *,
        dt_ms: float,
        tau_ms: float,
        a_ref_mV: float,
        r_ref_hz: float = 50.0,
        mode: str = "zero_sum",
        shuffle_permutation=None,
        spatial_shift_permutation=None,
        trace_dt_ms: float = 10.0,
    ):
        support = np.asarray(support_g, dtype=np.float64)
        if support.ndim != 1 or support.size == 0:
            raise ValueError("support_g must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(support)):
            raise ValueError("support_g must be finite")
        if np.any(support < 0.0) or np.any(support > 1.0):
            raise ValueError("support_g must lie in [0, 1]")
        if not np.any(support > 0.0):
            raise ValueError("support_g must contain positive field support")
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {sorted(self.MODES)}")

        for name, value in (
            ("dt_ms", dt_ms),
            ("tau_ms", tau_ms),
            ("a_ref_mV", a_ref_mV),
            ("r_ref_hz", r_ref_hz),
            ("trace_dt_ms", trace_dt_ms),
        ):
            if not np.isfinite(value) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if float(trace_dt_ms) < float(dt_ms):
            raise ValueError("trace_dt_ms must be at least one engine step")

        self.n_e = int(support.size)
        self.support_g = np.array(support, copy=True)
        self.support_g.setflags(write=False)
        self.support_sum = float(np.sum(self.support_g))
        self.dt_ms = float(dt_ms)
        self.tau_ms = float(tau_ms)
        self.a_ref_mV = float(a_ref_mV)
        self.a_max_mV = 2.0 * self.a_ref_mV
        self.r_ref_hz = float(r_ref_hz)
        self.mode = str(mode)
        self.trace_dt_ms = float(trace_dt_ms)
        self.gamma = float(np.exp(-self.dt_ms / self.tau_ms))
        expected_spikes_per_step = self.r_ref_hz * self.dt_ms / 1000.0
        self.q_mV_per_spike = float(
            self.a_ref_mV * (1.0 - self.gamma) / expected_spikes_per_step
        )

        if self.mode in {"stratified_shuffle", "spatial_shift"}:
            supplied_permutation = (
                shuffle_permutation
                if self.mode == "stratified_shuffle"
                else spatial_shift_permutation
            )
            permutation_name = (
                "shuffle_permutation"
                if self.mode == "stratified_shuffle"
                else "spatial_shift_permutation"
            )
            if supplied_permutation is None:
                raise ValueError(f"{self.mode} requires a frozen {permutation_name}")
            permutation = np.asarray(supplied_permutation)
            if permutation.ndim != 1 or permutation.shape != (self.n_e,):
                raise ValueError(f"{permutation_name} must align to support_g")
            if not np.issubdtype(permutation.dtype, np.integer):
                raise ValueError(f"{permutation_name} must contain integer indices")
            permutation = np.asarray(permutation, dtype=np.int64)
            if not np.array_equal(np.sort(permutation), np.arange(self.n_e)):
                raise ValueError(f"{permutation_name} must be a complete permutation")
            if self.mode == "spatial_shift":
                positive_support = self.support_g > 0.0
                if not np.array_equal(positive_support[permutation], positive_support):
                    raise ValueError(
                        "spatial_shift_permutation must preserve positive support"
                    )
            mapping_permutation = np.array(permutation, copy=True)
            mapping_permutation.setflags(write=False)
        else:
            if shuffle_permutation is not None or spatial_shift_permutation is not None:
                raise ValueError(
                    "mapping permutations are only valid for mapped control modes"
                )
            mapping_permutation = None

        if self.mode == "stratified_shuffle" and spatial_shift_permutation is not None:
            raise ValueError(
                "spatial_shift_permutation is only valid for spatial_shift"
            )
        if self.mode == "spatial_shift" and shuffle_permutation is not None:
            raise ValueError("shuffle_permutation is only valid for stratified_shuffle")
        self.shuffle_permutation = (
            mapping_permutation if self.mode == "stratified_shuffle" else None
        )
        self.spatial_shift_permutation = (
            mapping_permutation if self.mode == "spatial_shift" else None
        )
        self._mapping_permutation = mapping_permutation

        self.config_sha256 = self._config_sha256()
        self.state_mV = np.zeros(self.n_e, dtype=np.float64)
        self.step_index = 0
        self._trace_every = max(1, int(round(self.trace_dt_ms / self.dt_ms)))
        self._trace_time_ms: list[float] = []
        self._trace_state_mean_mV: list[float] = []
        self._trace_state_max_mV: list[float] = []
        self._trace_saturation_fraction: list[float] = []
        self._trace_delta_sum_mV: list[float] = []
        self._trace_delta_min_mV: list[float] = []
        self._trace_delta_max_mV: list[float] = []
        self._trace_lowered_support_mass_fraction: list[float] = []
        self._last_threshold_diagnostics: dict[str, float] | None = None

    def _config_sha256(self) -> str:
        payload = {
            "kind": self.KIND,
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "mode": self.mode,
            "dt_ms": self.dt_ms,
            "tau_ms": self.tau_ms,
            "a_ref_mV": self.a_ref_mV,
            "r_ref_hz": self.r_ref_hz,
            "trace_dt_ms": self.trace_dt_ms,
            "n_e": self.n_e,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
        )
        digest.update(np.ascontiguousarray(self.support_g).tobytes())
        if self._mapping_permutation is not None:
            digest.update(np.ascontiguousarray(self._mapping_permutation).tobytes())
        return digest.hexdigest()

    def _mapped_state(self) -> np.ndarray:
        if self._mapping_permutation is None:
            return self.state_mV
        return self.state_mV[self._mapping_permutation]

    def _zero_sum_delta(self, state_mV: np.ndarray) -> np.ndarray:
        mean_g = float(np.dot(self.support_g, state_mV) / self.support_sum)
        delta = self.support_g * (state_mV - mean_g)
        residual = float(np.sum(delta, dtype=np.float64))
        if residual != 0.0:
            delta -= (residual / self.support_sum) * self.support_g
        return delta

    def _support_weighted_sd(self, values: np.ndarray) -> float:
        mean_g = float(np.dot(self.support_g, values) / self.support_sum)
        variance = float(
            np.dot(self.support_g, np.square(values - mean_g)) / self.support_sum
        )
        return float(np.sqrt(max(0.0, variance)))

    def _delta_theta(self) -> np.ndarray:
        mapped_state = self._mapped_state()
        if self.mode == "raise_only":
            return self.support_g * mapped_state
        delta = self._zero_sum_delta(mapped_state)
        if self.mode != "spatial_shift":
            return delta

        reference_delta = self._zero_sum_delta(self.state_mV)
        reference_sd = self._support_weighted_sd(reference_delta)
        shifted_sd = self._support_weighted_sd(delta)
        if reference_sd == 0.0:
            return np.zeros_like(delta)
        if shifted_sd == 0.0:
            raise FloatingPointError(
                "support-preserving spatial shift lost non-zero dynamic amplitude"
            )
        delta *= reference_sd / shifted_sd
        residual = float(np.sum(delta, dtype=np.float64))
        if residual != 0.0:
            delta -= (residual / self.support_sum) * self.support_g
        return delta

    def delta_theta(self) -> np.ndarray:
        """Return an independent copy of the current E-threshold displacement."""
        return np.array(self._delta_theta(), copy=True)

    def threshold(self, base):
        """Return a new threshold array; the E population is the leading block."""
        base_array = np.asarray(base)
        if base_array.ndim != 1 or base_array.size < self.n_e:
            raise ValueError("base threshold must contain the complete E population")
        if not np.all(np.isfinite(base_array)):
            raise ValueError("base threshold must be finite")

        delta = self._delta_theta()
        output = np.array(base_array, dtype=np.float64, copy=True)
        output[: self.n_e] += delta
        if not np.all(np.isfinite(output)):
            raise FloatingPointError("effective threshold contains non-finite values")

        lowered_mass = float(np.sum(self.support_g[delta < 0.0]) / self.support_sum)
        self._last_threshold_diagnostics = {
            "delta_sum_mV": float(np.sum(delta, dtype=np.float64)),
            "delta_min_mV": float(np.min(delta)),
            "delta_max_mV": float(np.max(delta)),
            "lowered_support_mass_fraction": lowered_mass,
            "effective_threshold_min_mV": float(np.min(output)),
            "effective_threshold_max_mV": float(np.max(output)),
        }
        return output

    def step(self, spikes_e, dt_ms: float) -> None:
        """Advance the bounded trace after the current step's E spikes."""
        if not np.isclose(float(dt_ms), self.dt_ms, rtol=0.0, atol=1e-12):
            raise ValueError("controller and engine dt differ")
        spikes = np.asarray(spikes_e)
        if spikes.shape != (self.n_e,):
            raise ValueError("spikes_e must align to the E population")
        if spikes.dtype != np.bool_:
            if not np.all((spikes == 0) | (spikes == 1)):
                raise ValueError("spikes_e must be boolean or binary")
            spikes = spikes.astype(bool, copy=False)

        self.state_mV *= self.gamma
        self.state_mV[spikes] += self.q_mV_per_spike
        np.minimum(self.state_mV, self.a_max_mV, out=self.state_mV)
        self.step_index += 1
        if self.step_index % self._trace_every == 0:
            self._record_trace()

    def _record_trace(self) -> None:
        delta = self._delta_theta()
        saturation_fraction = float(np.mean(self.state_mV >= self.a_max_mV - 1e-12))
        lowered_mass = float(np.sum(self.support_g[delta < 0.0]) / self.support_sum)
        self._trace_time_ms.append(self.step_index * self.dt_ms)
        self._trace_state_mean_mV.append(
            float(np.dot(self.support_g, self.state_mV) / self.support_sum)
        )
        self._trace_state_max_mV.append(float(np.max(self.state_mV)))
        self._trace_saturation_fraction.append(saturation_fraction)
        self._trace_delta_sum_mV.append(float(np.sum(delta, dtype=np.float64)))
        self._trace_delta_min_mV.append(float(np.min(delta)))
        self._trace_delta_max_mV.append(float(np.max(delta)))
        self._trace_lowered_support_mass_fraction.append(lowered_mass)

    def diagnostics(self) -> dict:
        """Return JSON-safe state and latest threshold diagnostics."""
        delta = self._delta_theta()
        result = {
            "kind": self.KIND,
            "mode": self.mode,
            "config_sha256": self.config_sha256,
            "n_e": self.n_e,
            "step_index": self.step_index,
            "time_ms": self.step_index * self.dt_ms,
            "tau_ms": self.tau_ms,
            "a_ref_mV": self.a_ref_mV,
            "a_max_mV": self.a_max_mV,
            "r_ref_hz": self.r_ref_hz,
            "q_mV_per_spike": self.q_mV_per_spike,
            "state_mean_mV": float(
                np.dot(self.support_g, self.state_mV) / self.support_sum
            ),
            "state_max_mV": float(np.max(self.state_mV)),
            "saturation_fraction": float(
                np.mean(self.state_mV >= self.a_max_mV - 1e-12)
            ),
            "delta_sum_mV": float(np.sum(delta, dtype=np.float64)),
            "delta_min_mV": float(np.min(delta)),
            "delta_max_mV": float(np.max(delta)),
            "delta_support_weighted_sd_mV": self._support_weighted_sd(delta),
            "lowered_support_mass_fraction": float(
                np.sum(self.support_g[delta < 0.0]) / self.support_sum
            ),
        }
        if self._last_threshold_diagnostics is not None:
            result["last_threshold"] = dict(self._last_threshold_diagnostics)
        return result

    def trace_arrays(self) -> dict[str, np.ndarray]:
        """Return independent arrays suitable for an NPZ sidecar."""
        return {
            "time_ms": np.asarray(self._trace_time_ms, dtype=np.float64),
            "state_mean_mV": np.asarray(self._trace_state_mean_mV, dtype=np.float64),
            "state_max_mV": np.asarray(self._trace_state_max_mV, dtype=np.float64),
            "saturation_fraction": np.asarray(
                self._trace_saturation_fraction, dtype=np.float64
            ),
            "delta_sum_mV": np.asarray(self._trace_delta_sum_mV, dtype=np.float64),
            "delta_min_mV": np.asarray(self._trace_delta_min_mV, dtype=np.float64),
            "delta_max_mV": np.asarray(self._trace_delta_max_mV, dtype=np.float64),
            "lowered_support_mass_fraction": np.asarray(
                self._trace_lowered_support_mass_fraction, dtype=np.float64
            ),
        }

    def checkpoint_state(self) -> dict:
        """Return a non-aliasing checkpoint payload for this controller."""
        payload = {
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "protocol_kind": self.KIND,
            "config_sha256": self.config_sha256,
            "step_index": self.step_index,
            "state_mV": np.array(self.state_mV, copy=True),
            "last_threshold_diagnostics": (
                None
                if self._last_threshold_diagnostics is None
                else dict(self._last_threshold_diagnostics)
            ),
        }
        payload.update(
            {
                f"trace__{key}": np.array(value, copy=True)
                for key, value in self.trace_arrays().items()
            }
        )
        return payload

    def restore_checkpoint_state(self, payload: Mapping) -> None:
        """Restore state only when the frozen controller configuration matches."""
        if not isinstance(payload, Mapping):
            raise TypeError("controller checkpoint must be a mapping")
        if payload.get("checkpoint_version") != self.CHECKPOINT_VERSION:
            raise ValueError("unsupported controller checkpoint version")
        if payload.get("protocol_kind") != self.KIND:
            raise ValueError("controller checkpoint kind mismatch")
        if payload.get("config_sha256") != self.config_sha256:
            raise ValueError("controller checkpoint configuration mismatch")

        state = np.asarray(payload.get("state_mV"), dtype=np.float64)
        if state.shape != (self.n_e,) or not np.all(np.isfinite(state)):
            raise ValueError("invalid controller checkpoint state")
        if np.any(state < 0.0) or np.any(state > self.a_max_mV + 1e-12):
            raise ValueError("controller checkpoint state exceeds its bounds")
        step_index = payload.get("step_index")
        if not isinstance(step_index, (int, np.integer)) or int(step_index) < 0:
            raise ValueError("invalid controller checkpoint step_index")

        expected_trace_keys = set(self.trace_arrays())
        trace = {key: payload.get(f"trace__{key}") for key in expected_trace_keys}
        if any(value is None for value in trace.values()):
            raise ValueError("invalid controller checkpoint trace")
        restored_trace = {
            key: np.asarray(trace[key], dtype=np.float64) for key in expected_trace_keys
        }
        lengths = {value.size for value in restored_trace.values()}
        if (
            any(value.ndim != 1 for value in restored_trace.values())
            or len(lengths) != 1
        ):
            raise ValueError("controller checkpoint traces must be aligned vectors")
        if not all(np.all(np.isfinite(value)) for value in restored_trace.values()):
            raise ValueError("controller checkpoint trace contains non-finite values")

        last_threshold = payload.get("last_threshold_diagnostics")
        if last_threshold is not None:
            if not isinstance(last_threshold, Mapping):
                raise ValueError("invalid threshold diagnostics in checkpoint")
            last_threshold = {
                str(key): float(value) for key, value in last_threshold.items()
            }
            if not all(np.isfinite(value) for value in last_threshold.values()):
                raise ValueError("threshold diagnostics contain non-finite values")

        self.state_mV = np.array(state, copy=True)
        self.step_index = int(step_index)
        self._trace_time_ms = restored_trace["time_ms"].tolist()
        self._trace_state_mean_mV = restored_trace["state_mean_mV"].tolist()
        self._trace_state_max_mV = restored_trace["state_max_mV"].tolist()
        self._trace_saturation_fraction = restored_trace["saturation_fraction"].tolist()
        self._trace_delta_sum_mV = restored_trace["delta_sum_mV"].tolist()
        self._trace_delta_min_mV = restored_trace["delta_min_mV"].tolist()
        self._trace_delta_max_mV = restored_trace["delta_max_mV"].tolist()
        self._trace_lowered_support_mass_fraction = restored_trace[
            "lowered_support_mass_fraction"
        ].tolist()
        self._last_threshold_diagnostics = last_threshold
