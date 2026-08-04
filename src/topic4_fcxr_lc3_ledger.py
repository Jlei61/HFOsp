"""Per-event ledger for FCXR-LC3 no-kick trajectories.

A frozen coordinate deletes the history the bearing question is about: how many
interictal events, carrying how much load, were needed before the system crossed
into a sustained high state.  This module turns a detected event list plus the
retained full-field slow snapshots into that ledger.

Everything here is pure.  No file access, no substrate, no simulation -- each
input already exists in memory when a reconnaissance row finishes, so persisting
the ledger costs no extra integration step.

Design of record:
``docs/superpowers/specs/2026-08-04-topic4-fcxr-lc3-event-driven-pivot-design.md``
"""
from __future__ import annotations

import numpy as np

LEDGER_SCHEMA = "fcxr-lc3-event-ledger-1.0"
REGION_KEYS = ("core_A", "core_B", "axial", "off_axis")
ACCUMULATION_BAR = 3
SLOW_VARS = ("D", "H", "X", "y")


def _slice(n, t_on_ms, t_off_ms, step_ms):
    """Inclusive sample window [t_on, t_off] on a series sampled every step_ms."""
    if not (np.isfinite(step_ms) and float(step_ms) > 0.0):
        raise ValueError("step_ms must be finite and positive")
    i0 = max(0, int(round(float(t_on_ms) / float(step_ms))))
    i1 = min(int(n), int(round(float(t_off_ms) / float(step_ms))) + 1)
    return i0, i1


def event_dose_af(af, af_bin_ms, event, floor_af) -> float:
    """Active-fraction dose of one event, on the series the detector runs on.

    The floor is the frozen LC1 baseline's ``floor_af``; nothing is recalibrated
    here.  Sub-floor samples contribute zero rather than a negative credit.
    """
    a = np.asarray(af, dtype=float)
    i0, i1 = _slice(a.size, event["t_on"], event["t_off"], af_bin_ms)
    if i1 <= i0:
        return 0.0
    return float(np.clip(a[i0:i1] - float(floor_af), 0.0, None).sum() * float(af_bin_ms))


def event_dose_rate(rate_hz, dt_ms, event, r_base_hz) -> float:
    """Population-rate dose of one event at the full integration step.

    Deliberately not computed from the decimated 10 ms trace that reaches the
    NPZ: an interictal event lasts 8-19 ms, so decimation leaves one or two
    samples per event and the integral becomes noise.
    """
    r = np.asarray(rate_hz, dtype=float)
    i0, i1 = _slice(r.size, event["t_on"], event["t_off"], dt_ms)
    if i1 <= i0:
        return 0.0
    return float(np.clip(r[i0:i1] - float(r_base_hz), 0.0, None).sum() * float(dt_ms))


def regional_means(field, masks) -> dict:
    """Regional means over the four registered regions, plus the whole-array mean.

    The whole-array value is carried as context only.  The slow-vector probes
    measured a mean X drift of +0.033/s while both cores were depleting, because
    85% of cells sit off-axis -- a mean-only readout inverts that result.
    """
    a = np.asarray(field, dtype=float)
    if set(masks) != set(REGION_KEYS):
        raise ValueError(f"masks must be exactly {REGION_KEYS}")
    out = {}
    for key in REGION_KEYS:
        mask = np.asarray(masks[key], dtype=bool)
        if mask.shape != a.shape:
            raise ValueError(f"mask {key} shape {mask.shape} != field {a.shape}")
        if not mask.any():
            raise ValueError(f"region {key} is empty")
        out[key] = float(a[mask].mean())
    out["all"] = float(a.mean())
    return out


def snapshot_table(snapshots, dt_ms, masks) -> list:
    """Regional D/H/X/y for every retained full-field snapshot, ordered in time.

    ``D = 1 - z`` by the registered convention.  Ties on time are broken by label
    so the ordering is deterministic across runs.
    """
    rows = []
    for label, snap in snapshots.items():
        rows.append(dict(
            t_ms=float(snap["step"]) * float(dt_ms), label=str(label),
            D=regional_means(1.0 - np.asarray(snap["z_E"], dtype=float), masks),
            H=regional_means(snap["h_E"], masks),
            X=regional_means(snap["x_E"], masks),
            y=regional_means(snap["y_E"], masks),
        ))
    rows.sort(key=lambda r: (r["t_ms"], r["label"]))
    return rows


def bracketing_snapshots(table, t_on_ms, t_off_ms):
    """Nearest snapshot at or before the event start, and at or after its end."""
    pre = None
    for row in table:
        if row["t_ms"] <= float(t_on_ms):
            pre = row
        else:
            break
    post = next((row for row in table if row["t_ms"] >= float(t_off_ms)), None)
    return pre, post


def classify_entry(n_returning_before_onset, onset_ms) -> str:
    """Entry class, always reported alongside the count and never in place of it.

    ``ONE_SHOT`` is an explosion, not accumulation; ``CUMULATIVE`` needs at least
    ACCUMULATION_BAR returning events before onset.  A cold-started trajectory
    that ignites during its startup transient lands in ONE_SHOT, which is exactly
    the distinction a bare onset time cannot make.
    """
    if onset_ms is None:
        return "NO_ONSET"
    n = int(n_returning_before_onset)
    if n >= ACCUMULATION_BAR:
        return "CUMULATIVE"
    if n == 2:
        return "AMBIGUOUS_2"
    return "ONE_SHOT"


def entry_from_record(record) -> dict:
    """Entry summary for one reconnaissance row, from the ledger when it is present.

    Rows produced before the ledger existed still carry the event list and the bout,
    which is enough for the count and the class but not for the dose or the per-event
    slow state.  Those are reported as ``None`` with the reason attached, so a
    reconstructed summary can never be mistaken for a measured one.
    """
    ledger = record.get("event_ledger")
    if ledger:
        return dict(
            source="ledger",
            entry_class=ledger["entry_class"],
            n_returning_before_onset=ledger["n_returning_before_onset"],
            n_events_before_onset=ledger["n_events_before_onset"],
            onset_ms=ledger["onset_ms"], offset_ms=ledger["offset_ms"],
            Q_af_to_onset=ledger["Q_af_to_onset"],
            Q_rate_to_onset=ledger["Q_rate_to_onset"],
            first_non_returning_index=ledger["first_non_returning_index"],
            unavailable=[],
        )
    bout = (record.get("lifecycle") or {}).get("bout")
    win_ms = 1000.0
    onset_ms = None if bout is None else float(bout[0]) * win_ms
    offset_ms = None if bout is None else float(bout[1] + 1) * win_ms
    events = record.get("events") or []
    before = [e for e in events
              if onset_ms is None or float(e["t_off_ms"]) < onset_ms]
    n_ret = sum(1 for e in before if e.get("returned"))
    non_ret = next((i for i, e in enumerate(events, start=1)
                    if not e.get("returned")), None)
    return dict(
        source="reconstructed_from_events",
        entry_class=classify_entry(n_ret, onset_ms),
        n_returning_before_onset=n_ret, n_events_before_onset=len(before),
        onset_ms=onset_ms, offset_ms=offset_ms,
        Q_af_to_onset=None, Q_rate_to_onset=None,
        first_non_returning_index=non_ret,
        unavailable=[
            "Q_af_to_onset / Q_rate_to_onset: the stored rate is decimated to 10 ms "
            "and an interictal event is 8-19 ms",
            "per-event regional D/H/X: only the selected landmark snapshots were kept",
        ],
    )


def _delta(pre, post):
    if pre is None or post is None:
        return None
    return {var: {key: float(post[var][key] - pre[var][key]) for key in post[var]}
            for var in SLOW_VARS}


def _phase(event, onset_ms, offset_ms):
    if onset_ms is None or float(event["t_off"]) < float(onset_ms):
        return "pre_onset"
    if offset_ms is not None and float(event["t_on"]) > float(offset_ms):
        return "post_offset"
    return "ictal"


def build_event_ledger(*, events, af, af_bin_ms, floor_af, rate_hz, dt_ms,
                       r_base_hz, table, onset_ms, offset_ms, total_ms,
                       r_base_definition="pre-onset quiet median population rate of this run") -> dict:
    """How many events, carrying how much load, and the slow state they left behind.

    Both doses are always present; neither is a sufficient report on its own.
    Every slow readout is regional, with the whole-array mean carried as context.
    """
    rate = np.asarray(rate_hz, dtype=float)
    rows, q_af, q_rate = [], 0.0, 0.0
    for k, ev in enumerate(events, start=1):
        d_af = event_dose_af(af, af_bin_ms, ev, floor_af)
        d_rate = event_dose_rate(rate, dt_ms, ev, r_base_hz)
        q_af += d_af
        q_rate += d_rate
        i0, i1 = _slice(rate.size, ev["t_on"], ev["t_off"], dt_ms)
        pre, post = bracketing_snapshots(table, ev["t_on"], ev["t_off"])
        rows.append(dict(
            index=k, t_on_ms=float(ev["t_on"]), t_off_ms=float(ev["t_off"]),
            dur_ms=float(ev["dur_ms"]), peak_ext=float(ev["peak_ext"]),
            returned=bool(ev["returned"]),
            peak_rate_hz=float(rate[i0:i1].max()) if i1 > i0 else 0.0,
            dose_af=d_af, dose_rate=d_rate, Q_af=q_af, Q_rate=q_rate,
            phase=_phase(ev, onset_ms, offset_ms),
            pre=None if pre is None else dict(
                t_ms=pre["t_ms"], lag_ms=float(ev["t_on"]) - pre["t_ms"],
                **{v: pre[v] for v in SLOW_VARS}),
            post=None if post is None else dict(
                t_ms=post["t_ms"], lag_ms=post["t_ms"] - float(ev["t_off"]),
                **{v: post[v] for v in SLOW_VARS}),
            delta=_delta(pre, post),
        ))
    before = [r for r in rows if r["phase"] == "pre_onset"]
    n_ret_before = sum(1 for r in before if r["returned"])
    non_ret = next((r["index"] for r in rows if not r["returned"]), None)
    after = [r for r in rows if r["phase"] == "post_offset" and r["returned"]]
    iei = [after[i]["t_on_ms"] - after[i - 1]["t_on_ms"] for i in range(1, len(after))]
    resolved = onset_ms is not None
    return dict(
        schema=LEDGER_SCHEMA,
        calibration=dict(
            floor_af=float(floor_af), af_bin_ms=float(af_bin_ms), dt_ms=float(dt_ms),
            r_base_hz=float(r_base_hz), accumulation_bar=ACCUMULATION_BAR,
            r_base_definition=str(r_base_definition),
        ),
        onset_ms=(None if onset_ms is None else float(onset_ms)),
        offset_ms=(None if offset_ms is None else float(offset_ms)),
        total_ms=float(total_ms),
        n_events=len(rows), n_returning=sum(1 for r in rows if r["returned"]),
        n_events_before_onset=len(before), n_returning_before_onset=n_ret_before,
        entry_class=classify_entry(n_ret_before, onset_ms),
        Q_af_to_onset=(before[-1]["Q_af"] if before and resolved else None),
        Q_rate_to_onset=(before[-1]["Q_rate"] if before and resolved else None),
        first_non_returning_index=non_ret,
        events=rows,
        post_offset=dict(
            n_returning=len(after),
            durations_ms=[r["dur_ms"] for r in after],
            participation=[r["peak_ext"] for r in after],
            iei_ms=iei,
        ),
    )
