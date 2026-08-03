"""Pure contracts for the FCXR-LC3 frozen D-X geometry stage.

The functions in this module deliberately do not run the 40k network.  They lock
the complete 102-row matrix and classify already produced tail observables.  This
keeps outcome-dependent choices out of the simulation runner and lets the exact
state/field stage continue under its existing source lock.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pickle
import sys
from collections import Counter

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENGINE = os.path.join(_ROOT, "src", "snn_engine")
if _ENGINE not in sys.path:
    sys.path.insert(0, _ENGINE)

from src.topic4_mz_fcxr_dynamics import classify_run_workpoint, workpoint_metrics
from src.topic4_fcxr_lc3 import clone_loop_state, state_hash


SCHEMA_VERSION = "fcxr-lc3-geometry-contract-1.0"
H1_POINT_ID = "H1_ts1.25_r025"
H6_POINT_ID = "H6_ts1.25_r025"
PRIMARY_D_LABELS = ("D_healthy", "D10", "D30", "D50", "D70", "Dmax")
PRIMARY_X_LEVELS = (1.0, 0.9, 0.8, 0.65, 0.5, 0.3, 0.1)
SENTINEL_D_LABELS = ("D_healthy", "D50", "Dmax")
SENTINEL_X_LEVELS = (1.0, 0.5, 0.1)
STATE_KINDS = ("low", "high")
SCREEN_MS = 1500.0
SCREEN_TAIL_MS = 500.0
EXTENDED_MS = 5000.0
EXTENDED_TAIL_MS = 2000.0
REFRACTORY_CEILING_FRACTION_MAX = 0.05


def _hash_array(value) -> str:
    a = np.asarray(value)
    h = hashlib.sha256()
    h.update(a.dtype.str.encode("ascii"))
    h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _canonical_config(value):
    if isinstance(value, np.ndarray):
        return dict(kind="ndarray", dtype=value.dtype.str, shape=list(value.shape),
                    sha256=_hash_array(value))
    if isinstance(value, np.generic):
        return value.item()
    if dataclasses.is_dataclass(value):
        return {field.name: _canonical_config(getattr(value, field.name))
                for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(key): _canonical_config(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical_config(item) for item in value]
    return value


def configured_state_hash(state) -> str:
    """Bind the exact dynamic state to the static slow-variable configuration.

    ``state_hash`` intentionally hashes dynamic continuation state.  Geometry also
    changes H point and frozen fields, so its prepared-state identity must include
    the configuration that interprets those arrays.
    """

    payload = json.dumps(_canonical_config(state.slow.cfg), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    h = hashlib.sha256()
    h.update(state_hash(state).encode("ascii"))
    h.update(payload)
    return h.hexdigest()


def compact_checkpoint_diagnostics(state):
    """Clone a checkpoint and remove read-only histories without changing dynamics."""

    child = clone_loop_state(state)
    before = configured_state_hash(child)
    slow = child.slow
    for name, value in list(vars(slow).items()):
        if name.startswith("trace_") and isinstance(value, list):
            setattr(slow, name, [])
        elif name.startswith("calib_") and isinstance(value, list):
            setattr(slow, name, [])
    slow.snapshots = {}
    slow._snap_steps = None
    slow.h_lc2_observer = None
    slow.seeg_observer = None
    after = configured_state_hash(child)
    if after != before:
        raise RuntimeError("diagnostic compaction changed configured state identity")
    return child


def save_prepared_checkpoint(path: str, state, *, metadata: dict) -> dict:
    """Atomically pickle a compact exact checkpoint and return its bound hashes."""

    compact = compact_checkpoint_diagnostics(state)
    payload = dict(schema=SCHEMA_VERSION, metadata=dict(metadata), state=compact,
                   dynamic_state_hash=state_hash(compact),
                   configured_state_hash=configured_state_hash(compact))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=5)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    file_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            file_hash.update(block)
    return dict(path=path, file_sha256=file_hash.hexdigest(),
                dynamic_state_hash=payload["dynamic_state_hash"],
                configured_state_hash=payload["configured_state_hash"])


def load_prepared_checkpoint(path: str, *, expected_file_sha256: str | None = None):
    """Load a prepared checkpoint and loudly reject file or state drift."""

    if expected_file_sha256 is not None:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        if h.hexdigest() != expected_file_sha256:
            raise RuntimeError("prepared checkpoint file hash mismatch")
    with open(path, "rb") as f:
        payload = pickle.load(f)
    state = payload["state"]
    if state_hash(state) != payload.get("dynamic_state_hash"):
        raise RuntimeError("prepared checkpoint dynamic-state hash mismatch")
    if configured_state_hash(state) != payload.get("configured_state_hash"):
        raise RuntimeError("prepared checkpoint configured-state hash mismatch")
    return payload


def _x_tag(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def _validate_field_record(label: str, record: dict) -> None:
    required = ("field_sha256", "source_path", "source_sha256")
    missing = [key for key in required if not record.get(key)]
    if missing:
        raise ValueError(f"{label}: field record is missing {missing}")


def build_geometry_manifest_rows(
    *,
    fields: dict,
    prepared_state_hashes: dict,
    output_root: str,
    noise_seed: int = 401,
) -> list[dict]:
    """Return the pre-outcome 84 H1 + 18 H6 geometry rows.

    ``fields`` is keyed by the six primary D labels.  ``prepared_state_hashes``
    is keyed by ``(point_id, state_kind)``.  A caller must materialise this list
    before launching any row; there is no outcome-dependent row generation.
    """

    if set(fields) != set(PRIMARY_D_LABELS):
        raise ValueError("fields must contain exactly the six primary D labels")
    for label, record in fields.items():
        _validate_field_record(label, record)
    expected_states = {(point, state) for point in (H1_POINT_ID, H6_POINT_ID)
                       for state in STATE_KINDS}
    if set(prepared_state_hashes) != expected_states:
        raise ValueError("prepared_state_hashes must contain H1/H6 low/high exactly")
    if not all(prepared_state_hashes[key] for key in expected_states):
        raise ValueError("prepared state hashes must be non-empty")

    rows: list[dict] = []

    def add(point_id, d_labels, x_levels, *, sentinel):
        for d_label in d_labels:
            field = fields[d_label]
            for a_x in x_levels:
                for state_kind in STATE_KINDS:
                    row_id = f"{point_id}_{d_label}_aX{_x_tag(a_x)}_{state_kind}"
                    rows.append(dict(
                        index=len(rows), schema=SCHEMA_VERSION, row_id=row_id,
                        point_id=point_id, d_label=d_label,
                        d_field_sha256=field["field_sha256"],
                        d_source_path=field["source_path"],
                        d_source_sha256=field["source_sha256"],
                        a_x=float(a_x), state_kind=state_kind,
                        prepared_state_hash=prepared_state_hashes[(point_id, state_kind)],
                        connection_seed=1, noise_seed=int(noise_seed), no_kick=True,
                        sentinel=bool(sentinel), screen_ms=SCREEN_MS,
                        screen_tail_ms=SCREEN_TAIL_MS, extended_ms=EXTENDED_MS,
                        extended_tail_ms=EXTENDED_TAIL_MS,
                        output_path=os.path.join(output_root, "geometry_cells", f"{row_id}.json"),
                        done_path=os.path.join(output_root, "geometry_cells", f"{row_id}.DONE.json"),
                    ))

    add(H1_POINT_ID, PRIMARY_D_LABELS, PRIMARY_X_LEVELS, sentinel=False)
    add(H6_POINT_ID, SENTINEL_D_LABELS, SENTINEL_X_LEVELS, sentinel=True)
    validate_geometry_manifest(rows)
    return rows


def validate_geometry_manifest(rows: list[dict]) -> dict:
    """Fail closed unless the exact registered matrix is present once each."""

    if len(rows) != 102:
        raise ValueError(f"geometry manifest must contain 102 rows, got {len(rows)}")
    ids = [row.get("row_id") for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("geometry row_id values must be unique")
    if [row.get("index") for row in rows] != list(range(102)):
        raise ValueError("geometry row indices must be contiguous and ordered")
    counts = Counter((r.get("point_id"), bool(r.get("sentinel"))) for r in rows)
    if counts != Counter({(H1_POINT_ID, False): 84, (H6_POINT_ID, True): 18}):
        raise ValueError(f"geometry point/sentinel counts are wrong: {dict(counts)}")
    for row in rows:
        if row.get("state_kind") not in STATE_KINDS:
            raise ValueError("invalid state_kind")
        if not (0.0 <= float(row.get("a_x", np.nan)) <= 1.0):
            raise ValueError("a_x must be finite in [0,1]")
        if not row.get("d_field_sha256") or not row.get("prepared_state_hash"):
            raise ValueError("every row must bind a D field and prepared state hash")
        if float(row.get("screen_ms", np.nan)) != SCREEN_MS:
            raise ValueError("screen duration drift")
        if float(row.get("screen_tail_ms", np.nan)) != SCREEN_TAIL_MS:
            raise ValueError("screen tail drift")
    return dict(status="PASS", n_rows=102, n_h1=84, n_h6=18,
                n_low=51, n_high=51, schema=SCHEMA_VERSION)


def paired_field_shape_metrics(a, b, *, support_epsilon: float = 1e-12) -> dict:
    """Compare two fields only when they live on the same registered cells."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or a.size == 0:
        raise ValueError("paired fields must be non-empty matching 1-D arrays")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("paired fields must be finite")
    if not (np.isfinite(support_epsilon) and support_epsilon >= 0.0):
        raise ValueError("support_epsilon must be finite and non-negative")
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    cosine = float(np.dot(a, b) / (na * nb)) if na > 0.0 and nb > 0.0 else None
    sa, sb = a > support_epsilon, b > support_epsilon
    union = int(np.count_nonzero(sa | sb))
    intersection = int(np.count_nonzero(sa & sb))
    corr = (float(np.corrcoef(a, b)[0, 1])
            if float(np.std(a)) > 0.0 and float(np.std(b)) > 0.0 else None)
    return dict(
        pearson_cellwise=corr, cosine_cellwise=cosine,
        relative_l2_difference=float(np.linalg.norm(a - b) / max(na, nb, 1e-12)),
        mean_absolute_difference=float(np.mean(np.abs(a - b))),
        support_epsilon=float(support_epsilon),
        support_fraction_a=float(np.mean(sa)), support_fraction_b=float(np.mean(sb)),
        support_jaccard=(float(intersection / union) if union else 1.0),
        same_cell_identity_required=True,
    )


def classify_geometry_tail(
    *,
    rate_hz,
    dt_ms: float,
    baseline_roll_hi_hz: float,
    analysis_start_ms: float,
    per_cell_tail_spike_counts,
    tail_duration_ms: float,
    tau_ref_e_ms: float,
    h_mean_trace,
    theta_h: float,
    finite: bool = True,
    clip_frac_max: float = 0.0,
    tau_eff_min_ms: float = np.inf,
) -> dict:
    """Classify one tail while keeping tonic saturation separate from high state.

    The low/high labels reuse the accepted workpoint-relative Stage-D classifier.
    The additional per-cell ceiling diagnostic prevents a refractory plateau from
    being relabelled as an ictal carrier.  It is diagnostic, not a lifecycle claim.
    """

    rate = np.asarray(rate_hz, dtype=float)
    h = np.asarray(h_mean_trace, dtype=float)
    counts = np.asarray(per_cell_tail_spike_counts, dtype=float)
    if rate.ndim != 1 or h.ndim != 1 or counts.ndim != 1 or counts.size == 0:
        raise ValueError("rate, H trace and per-cell counts must be non-empty 1-D arrays")
    if not (np.isfinite(dt_ms) and dt_ms > 0 and np.isfinite(tail_duration_ms)
            and tail_duration_ms > 0 and np.isfinite(tau_ref_e_ms) and tau_ref_e_ms > 0):
        raise ValueError("time constants and durations must be finite and positive")
    all_finite = bool(finite and np.all(np.isfinite(rate)) and np.all(np.isfinite(h))
                      and np.all(np.isfinite(counts)))
    numerical_unsafe = bool((not all_finite) or float(clip_frac_max) > 0.0
                            or float(tau_eff_min_ms) < 2.0 * float(dt_ms))
    wp = workpoint_metrics(rate, float(dt_ms), float(baseline_roll_hi_hz),
                           float(analysis_start_ms))
    wp["numerical_unsafe"] = numerical_unsafe
    wp_label = classify_run_workpoint(wp)

    per_cell_hz = counts / (float(tail_duration_ms) * 1e-3)
    ceiling_hz = 0.8 * (1000.0 / float(tau_ref_e_ms))
    ceiling_fraction = float(np.mean(per_cell_hz >= ceiling_hz))
    h_slope_per_s = float("nan")
    h_slope_floor = -0.05 * max(float(np.mean(h)), float(theta_h))
    if h.size >= 2:
        x = np.arange(h.size, dtype=float) * float(dt_ms)
        h_slope_per_s = 1000.0 * float(np.polyfit(x, h, 1)[0])

    if numerical_unsafe:
        label = "NUMERICAL_UNSAFE"
    elif ceiling_fraction >= REFRACTORY_CEILING_FRACTION_MAX:
        label = "SATURATED_TONIC_BRANCH"
    elif (wp_label in ("FINITE_HIGH_FIXED", "FINITE_HIGH_ORBIT")
          and h_slope_per_s < h_slope_floor):
        label = "HIGH_RATE_H_DECAY_UNRESOLVED"
    else:
        label = wp_label
    return dict(
        label=label, workpoint_label=wp_label, workpoint_metrics=wp,
        numerical_unsafe=numerical_unsafe, ceiling_hz=ceiling_hz,
        refractory_ceiling_fraction=ceiling_fraction,
        refractory_ceiling_fraction_max=REFRACTORY_CEILING_FRACTION_MAX,
        h_mean=float(np.mean(h)), h_slope_per_s=h_slope_per_s,
        h_slope_floor=h_slope_floor,
    )


def extension_required(*, state_kind: str, label: str) -> bool:
    """Lock the 1.5 s -> 5 s routing before opening outcomes."""

    if state_kind not in STATE_KINDS:
        raise ValueError("state_kind must be low or high")
    if label == "NUMERICAL_UNSAFE":
        return False  # engineering hard stop, never an extension
    if label == "SATURATED_TONIC_BRANCH":
        return False  # scientifically bad but already resolved
    same_basin = (
        label == "INTERICTAL_WORKPOINT" if state_kind == "low"
        else label in ("FINITE_HIGH_FIXED", "FINITE_HIGH_ORBIT")
    )
    return bool(not same_basin)
