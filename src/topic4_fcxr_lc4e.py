"""Pure provenance and adjudication contracts for the LC4e shared executor."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def derive_shared_candidate(local_lock: dict, local_record: dict, local_trace) -> dict:
    """Copy the executed LC4d candidate and change only its spatial executor."""
    if local_lock.get("status") != "L0_PASS":
        raise ValueError("SHARED_EXECUTOR_NOT_IDENTIFIABLE: LC4d lock did not pass")
    gate = local_record.get("gate") or {}
    if gate.get("verdict") != "OFFSET_LATENCY_REPAIR_INSUFFICIENT":
        raise ValueError("SHARED_EXECUTOR_NOT_IDENTIFIABLE: unexpected LC4d verdict")
    if float(gate.get("onset_ms", np.nan)) != 11000.0:
        raise ValueError("SHARED_EXECUTOR_NOT_IDENTIFIABLE: LC4d onset anchor drifted")
    numerical = local_record.get("numerical") or {}
    if (bool(numerical.get("numerical_unsafe", True))
            or float(numerical.get("clip_frac_max", np.inf)) != 0.0):
        raise ValueError("SHARED_EXECUTOR_NOT_IDENTIFIABLE: LC4d was numerically unsafe")
    current = np.asarray(local_trace["adap_current"], dtype=float)
    dt = float(np.asarray(local_trace["trace_dt_ms"]).reshape(-1)[0])
    nz = np.flatnonzero(current > 0.0)
    if not nz.size:
        raise ValueError("SHARED_EXECUTOR_NOT_IDENTIFIABLE: LC4d actuator never engaged")
    first_ms = float(nz[0] * dt)
    if first_ms != 11830.0:
        raise ValueError("SHARED_EXECUTOR_NOT_IDENTIFIABLE: first-current anchor drifted")
    candidate = dict(local_lock["candidate"])
    changed = {"m_hill_spatial_mode": "shared"}
    candidate.update(changed)
    return dict(
        candidate=candidate,
        changed_fields=changed,
        local_onset_ms=11000.0,
        local_first_current_ms=first_ms,
        local_first_current_index=int(nz[0]),
        local_trace_dt_ms=dt,
    )


def adjudicate_shared_screen(*, local_record: dict, shared_record: dict,
                             local_trace, shared_trace) -> dict:
    """Separate causal-prefix integrity from the scientific shared-offset result."""
    lg = local_record.get("gate") or {}
    sg = shared_record.get("gate") or {}
    lc = np.asarray(local_trace["adap_current"], dtype=float)
    sc = np.asarray(shared_trace["adap_current"], dtype=float)
    ldt = float(np.asarray(local_trace["trace_dt_ms"]).reshape(-1)[0])
    sdt = float(np.asarray(shared_trace["trace_dt_ms"]).reshape(-1)[0])
    nz_l = np.flatnonzero(lc > 0.0)
    nz_s = np.flatnonzero(sc > 0.0)
    first_l = float(nz_l[0] * ldt) if nz_l.size else float("nan")
    first_s = float(nz_s[0] * sdt) if nz_s.size else float("nan")
    boundary = first_l

    rate_l = np.asarray(local_trace["rate_E"])
    rate_s = np.asarray(shared_trace["rate_E"])
    rdt_l = float(np.asarray(local_trace["rate_dt_ms"]).reshape(-1)[0])
    rdt_s = float(np.asarray(shared_trace["rate_dt_ms"]).reshape(-1)[0])
    af_l = np.asarray(local_trace["af"])
    af_s = np.asarray(shared_trace["af"])
    adt_l = float(np.asarray(local_trace["af_bin_ms"]).reshape(-1)[0])
    adt_s = float(np.asarray(shared_trace["af_bin_ms"]).reshape(-1)[0])
    nr = int(round(boundary / rdt_l)) if np.isfinite(boundary) else 0
    na = int(round(boundary / adt_l)) if np.isfinite(boundary) else 0

    prefix = dict(
        local_first_current_ms=first_l,
        shared_first_current_ms=first_s,
        first_current_time_equal=bool(first_l == first_s == 11830.0),
        current_zero_before_boundary=bool(nz_s.size and np.all(sc[:nz_s[0]] == 0.0)),
        rate_grid_equal=bool(rdt_l == rdt_s),
        af_grid_equal=bool(adt_l == adt_s),
        rate_prefix_equal=bool(nr > 0 and np.array_equal(rate_l[:nr], rate_s[:nr])),
        af_prefix_equal=bool(na > 0 and np.array_equal(af_l[:na], af_s[:na])),
    )
    prefix_ok = bool(all(prefix.values()))
    local_negative = bool(lg.get("verdict") == "OFFSET_LATENCY_REPAIR_INSUFFICIENT")
    shared_gate_pass = bool(sg.get("passed", False))
    if not prefix_ok:
        verdict = "CAUSAL_PREFIX_MISMATCH"
    elif shared_gate_pass and local_negative:
        verdict = "SPATIALLY_SHARED_OFFSET_CANDIDATE"
    elif sg.get("verdict") == "TERMINATOR_PREVENTS_QUALIFYING_ENTRY":
        verdict = ("SHARED_EXECUTOR_OVERFAST" if float(np.max(sc, initial=0.0)) > 0.0
                   else "SHARED_EXECUTOR_NEVER_ENGAGED")
    else:
        verdict = "SHARED_EXECUTOR_OFFSET_NEGATIVE"

    spatial = {}
    for name in ("H_core_A", "H_core_B", "H_axial", "H_off_axis",
                 "y_core_A", "y_core_B", "y_axial", "y_off_axis"):
        key = f"snapshot_{name}"
        if key in shared_trace:
            arr = np.asarray(shared_trace[key], dtype=float)
            spatial[f"shared_final_{name}"] = float(arr[-1]) if arr.size else float("nan")
        if key in local_trace:
            arr = np.asarray(local_trace[key], dtype=float)
            spatial[f"local_final_{name}"] = float(arr[-1]) if arr.size else float("nan")

    return dict(
        schema="fcxr-lc4e-spatial-executor-screen-1.0",
        verdict=verdict,
        passed=bool(verdict == "SPATIALLY_SHARED_OFFSET_CANDIDATE"),
        prefix=prefix,
        local_control=dict(verdict=lg.get("verdict"), onset_ms=lg.get("onset_ms"),
                           offset_ms=lg.get("offset_ms"), bout_ms=lg.get("bout_ms")),
        shared=dict(verdict=sg.get("verdict"), onset_ms=sg.get("onset_ms"),
                    offset_ms=sg.get("offset_ms"), bout_ms=sg.get("bout_ms"),
                    n_returning_before_onset=sg.get("n_returning_before_onset"),
                    max_current=float(np.max(sc, initial=0.0))),
        spatial=spatial,
        claim_boundary=("one development seed causal architecture screen; a positive result only "
                        "authorises the locked nominal lifecycle gate"),
    )
