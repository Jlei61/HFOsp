"""Consumer of the model agent's ``frozen_state_registry.json``.

The evaluation package never trains a state.  It reads anchor-aligned frozen
state trajectories, aligns them to its own 300 s grid by absolute time, and
marks anchors without a state as missing for *every* arm so pairing survives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REGISTRY_FORMAT = "group_event_state_v0_3_2_frozen_state_registry"
ANCHOR_TIME_KEYS = ("anchor_time", "anchor_times", "t_anchor")
ANCHOR_STATE_KEYS = ("anchor_state", "anchor_states", "state_anchor")
EVENT_TIME_KEYS = ("event_time", "event_times", "t_event")
EVENT_PRE_KEYS = ("event_pre_state", "event_state_pre", "pre_event_state")
ALIGN_TOLERANCE_SECONDS = 1e-3

EXPECTED_SCHEMA: dict[str, Any] = {
    "format": REGISTRY_FORMAT,
    "written_by": "Agent 2 (v032_eval) -- what the evaluator consumes",
    "top_level": {
        "format": REGISTRY_FORMAT,
        "status": "complete | partial",
        "source_commit": "git sha of the model code that produced the states",
        "state_dim": "int (optional; per-seed value wins)",
        "partition": "optional dict; if present must use boundary_fractions [0.6, 0.7, 0.8] on recorded time",
        "patients": {
            "<subject>": {
                "status": "complete | partial | failed",
                "seeds": {
                    "<seed>": {
                        "status": "complete",
                        "arrays_path": "absolute path to an .npz",
                        "checkpoint": "path", "checkpoint_sha256": "sha", "selected_epoch": "int",
                        "state_dim": "int",
                        "selection_phase": "must be dev_val (70-80%)",
                        "open_loop": "true -- anchor states never read events at/after the anchor",
                    }
                },
            }
        },
    },
    "arrays_npz": {
        "anchor_time": "(A,) float64 absolute epoch seconds of every anchor on the 300 s grid (all phases)",
        "anchor_state": "(A, D) float state evolved open-loop to each anchor",
        "event_time": "(N,) float64 absolute epoch of each retained interictal event (optional)",
        "event_pre_state": "(N, D) state immediately before each event, i.e. S(t_e^-) (optional)",
        "event_post_state": "(N, D) state after the event content entered (optional, unused by H1)",
    },
    "alignment": {
        "rule": "anchors matched by absolute time within 1e-3 s; unmatched grid anchors are missing for every arm",
        "event_fallback": "if event_pre_state is absent, the last anchor at or before the event time (same segment) is held; flagged anchor_held_state",
    },
    "aliases_accepted": {
        "anchor_time": list(ANCHOR_TIME_KEYS), "anchor_state": list(ANCHOR_STATE_KEYS),
        "event_time": list(EVENT_TIME_KEYS), "event_pre_state": list(EVENT_PRE_KEYS),
        "patients": ["patients", "subjects"],
    },
}


@dataclass
class StateBundle:
    subject: str
    seed: str
    anchor_state: np.ndarray                 # (A, D) aligned to the evaluator grid; NaN rows missing
    event_pre_state: np.ndarray | None       # (N, D) aligned to evaluator events; NaN rows missing
    event_state_mode: str                    # "event_pre_state" | "anchor_held_state" | "unavailable"
    n_anchor_matched: int
    n_anchor_missing: int
    n_event_matched: int
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def state_dim(self) -> int:
        return int(self.anchor_state.shape[1])


def write_expected_schema(shared_dir: Path) -> Path:
    from .contract import atomic_json

    return atomic_json(Path(shared_dir) / "frozen_state_registry.expected_schema.json", EXPECTED_SCHEMA)


def _first_key(container: Mapping[str, Any], keys: tuple[str, ...], what: str) -> Any:
    for key in keys:
        if key in container:
            return container[key]
    raise KeyError(f"frozen state arrays lack {what}; accepted keys {list(keys)}")


def load_registry(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if payload.get("format") != REGISTRY_FORMAT:
        raise ValueError(
            f"{path}: format {payload.get('format')!r} != {REGISTRY_FORMAT!r}; "
            "refusing to guess the model agent's schema"
        )
    patients = payload.get("patients", payload.get("subjects"))
    if not isinstance(patients, Mapping):
        raise ValueError(f"{path}: no 'patients' mapping")
    payload["patients"] = patients
    return payload


def complete_seed_entries(registry: Mapping[str, Any], subject: str) -> dict[str, dict[str, Any]]:
    entry = registry["patients"].get(subject)
    if not entry:
        return {}
    seeds = entry.get("seeds", {})
    return {
        str(seed): dict(spec) for seed, spec in seeds.items()
        if spec.get("status", "complete") == "complete" and spec.get("arrays_path")
    }


def align_by_time(source_times: np.ndarray, source_values: np.ndarray, target_times: np.ndarray,
                  *, tolerance: float = ALIGN_TOLERANCE_SECONDS) -> tuple[np.ndarray, int]:
    """Row of ``source_values`` whose time matches each target time; NaN otherwise."""

    st = np.asarray(source_times, dtype=np.float64)
    sv = np.asarray(source_values, dtype=np.float64)
    tt = np.asarray(target_times, dtype=np.float64)
    if st.ndim != 1 or sv.ndim != 2 or sv.shape[0] != st.size:
        raise ValueError("source times/values shapes disagree")
    order = np.argsort(st, kind="stable")
    st_sorted = st[order]
    pos = np.searchsorted(st_sorted, tt)
    out = np.full((tt.size, sv.shape[1]), np.nan)
    matched = 0
    for cand in (pos - 1, pos):
        ok = (cand >= 0) & (cand < st_sorted.size)
        idx = np.clip(cand, 0, max(st_sorted.size - 1, 0))
        close = ok & (np.abs(st_sorted[idx] - tt) <= tolerance)
        rows = np.flatnonzero(close & ~np.isfinite(out[:, 0]))
        out[rows] = sv[order[idx[rows]]]
        matched += rows.size
    return out, int(matched)


def anchor_held_event_states(anchor_times: np.ndarray, anchor_segment: np.ndarray,
                             anchor_state: np.ndarray, event_times: np.ndarray,
                             event_segment: np.ndarray) -> np.ndarray:
    """State at the last anchor at or before each event in the same segment."""

    out = np.full((event_times.size, anchor_state.shape[1]), np.nan)
    for seg in np.unique(event_segment):
        a_idx = np.flatnonzero(anchor_segment == seg)
        e_idx = np.flatnonzero(event_segment == seg)
        if a_idx.size == 0 or e_idx.size == 0:
            continue
        a_t = anchor_times[a_idx]
        order = np.argsort(a_t)
        a_idx = a_idx[order]
        a_t = a_t[order]
        pos = np.searchsorted(a_t, event_times[e_idx], side="right") - 1
        ok = pos >= 0
        out[e_idx[ok]] = anchor_state[a_idx[pos[ok]]]
    return out


def load_state_bundle(spec: Mapping[str, Any], *, subject: str, seed: str,
                      grid_times: np.ndarray, grid_segment: np.ndarray,
                      event_times: np.ndarray, event_segment: np.ndarray) -> StateBundle:
    path = Path(spec["arrays_path"])
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: np.asarray(data[k]) for k in data.files}
    a_time = _first_key(arrays, ANCHOR_TIME_KEYS, "anchor times")
    a_state = _first_key(arrays, ANCHOR_STATE_KEYS, "anchor states")
    if a_state.ndim != 2 or a_state.shape[0] != a_time.size:
        raise ValueError(f"{path}: anchor_state must be (A, D) aligned with anchor_time")
    anchor_state, n_matched = align_by_time(a_time, a_state, grid_times)
    event_pre = None
    mode = "unavailable"
    n_event = 0
    ev_key = next((k for k in EVENT_PRE_KEYS if k in arrays), None)
    et_key = next((k for k in EVENT_TIME_KEYS if k in arrays), None)
    if ev_key is not None and et_key is not None:
        event_pre, n_event = align_by_time(arrays[et_key], arrays[ev_key], event_times)
        mode = "event_pre_state"
    elif np.isfinite(anchor_state).any():
        event_pre = anchor_held_event_states(grid_times, grid_segment, anchor_state,
                                             event_times, event_segment)
        n_event = int(np.isfinite(event_pre[:, 0]).sum())
        mode = "anchor_held_state"
    return StateBundle(
        subject=subject, seed=str(seed), anchor_state=anchor_state, event_pre_state=event_pre,
        event_state_mode=mode, n_anchor_matched=int(n_matched),
        n_anchor_missing=int(grid_times.size - n_matched), n_event_matched=int(n_event),
        provenance={"arrays_path": str(path), **{k: v for k, v in spec.items() if k != "arrays_path"}},
    )
