"""Pure lifecycle-landmark logic for FCXR-LC3 no-kick reconnaissance."""
from __future__ import annotations

import numpy as np


SCHEMA_VERSION = "fcxr-lc3-recon-1.0"


def checkpoint_step_for_snapshot(snapshot: dict) -> int:
    """Map a post-update slow snapshot to the matching continuation step."""

    if not isinstance(snapshot, dict):
        raise TypeError("snapshot must be a dict")
    if snapshot.get("captured_after_update") is not True:
        raise ValueError("snapshot must be captured after its registered update")
    step = snapshot.get("step")
    if not isinstance(step, (int, np.integer)) or int(step) < 0:
        raise ValueError("snapshot step must be a non-negative integer")
    # The slow snapshot is captured after update k; the complete loop state at
    # that same physical instant has already advanced to continuation step k+1.
    return int(step) + 1


def select_landmark_times(lifecycle: dict, *, win_ms: float, total_ms: float) -> dict:
    """Deterministically map a window-level lifecycle result to snapshot times."""

    if not (np.isfinite(win_ms) and win_ms > 0 and np.isfinite(total_ms) and total_ms > 0):
        raise ValueError("win_ms and total_ms must be finite and positive")
    bout = lifecycle.get("bout")
    if bout is None:
        return {"no_onset_final": float(total_ms)}
    b0, b1 = (int(bout[0]), int(bout[1]))
    onset = b0 * float(win_ms)
    offset = (b1 + 1) * float(win_ms)
    rows = {
        "pre_onset": max(0.0, onset - 1000.0),
        "onset": onset,
        "early_high": min(total_ms, onset + 500.0),
        "late_high_pre_offset": min(total_ms, max(onset, offset - 250.0)),
    }
    if offset < total_ms:
        rows["post_offset"] = min(total_ms, offset + 500.0)
    if lifecycle.get("label") == "RECOVERED_INTERICTAL":
        rows["recovered"] = min(total_ms, offset + 8000.0)
    return rows


def nearest_snapshot_labels(snapshot_times: dict, targets: dict) -> dict:
    """Choose the nearest available full-field snapshot with stable tie-breaking."""

    if not snapshot_times:
        raise ValueError("at least one snapshot is required")
    ordered = sorted((float(t), str(label)) for label, t in snapshot_times.items())
    out = {}
    for name, target in targets.items():
        target = float(target)
        _distance, t, label = min((abs(t - target), t, label) for t, label in ordered)
        out[name] = dict(snapshot_label=label, snapshot_time_ms=t,
                         target_time_ms=target, abs_error_ms=abs(t - target))
    return out


def reconnaissance_verdict(*, lifecycle: dict, numerical_unsafe: bool,
                            refractory_ceiling_fraction: float,
                            x_activates_after_onset: bool | None) -> str:
    """Classify completed reconnaissance without promoting it to parameter acceptance."""

    if numerical_unsafe:
        return "RECON_NUMERICAL_BLOCKED"
    label = lifecycle.get("label")
    if label in ("INTERICTAL_BASELINE", "DENSE_EVENT_TRAIN", "UNRESOLVED") \
            and lifecycle.get("bout") is None:
        return "RECON_NO_ICTAL_ONSET"
    if refractory_ceiling_fraction >= 0.05:
        return "RECON_SATURATED_TONIC_BAD_DATA"
    if label == "ICTAL_LIKE_BOUNDED":
        return "RECON_HIGH_WITHOUT_OFFSET"
    if label in ("TERMINATED_REFRACTORY", "PERMANENT_SILENCE", "RAPID_RELAPSE"):
        return "RECON_OFFSET_WITHOUT_STATISTICAL_RECOVERY"
    if label == "RECOVERED_INTERICTAL":
        return ("RECON_RECOVERED_PATTERN" if x_activates_after_onset
                else "RECON_RECOVERY_X_ORDER_UNRESOLVED")
    return "RECON_OTHER_COMPLETED"
