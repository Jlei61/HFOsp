"""The published H registry must be consumable by the model agent's reader."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.topic5_group_event_state.v032_eval.state_registry import (
    EXPECTED_SCHEMA, REGISTRY_FORMAT, align_by_time, anchor_held_event_states, load_registry,
)


def _reader_like_agent1(registry_path: Path, subject: str, t_anchor: np.ndarray, horizons):
    """Mirror of v032_model.history_baseline.load_agent2_history_baseline (read-only contract)."""

    registry = json.loads(Path(registry_path).read_text())
    subjects = registry.get("subjects", registry)
    entry = subjects.get(subject)
    assert entry is not None
    per_horizon = entry.get("horizons", entry)
    out = {}
    for horizon in horizons:
        key = str(int(horizon))
        spec = per_horizon.get(key) or per_horizon.get(f"{key}s")
        assert spec is not None, key
        arrays_path = spec.get("arrays") or spec.get("path") or spec.get("npz")
        with np.load(arrays_path, allow_pickle=False) as data:
            a_key = next(k for k in ("anchor_time", "t_anchor", "anchor_epoch") if k in data)
            m_key = next(k for k in ("log_mu_h", "log_mu_H", "log_mu") if k in data)
            t_theirs = np.asarray(data[a_key], dtype=np.float64)
            values = np.asarray(data[m_key], dtype=np.float64)
        assert t_theirs.shape == values.shape
        order = np.argsort(t_theirs, kind="stable")
        t_sorted, v_sorted = t_theirs[order], values[order]
        pos = np.clip(np.searchsorted(t_sorted, t_anchor), 0, t_sorted.size - 1)
        assert np.all(np.abs(t_sorted[pos] - t_anchor) <= 1e-3)
        aligned = v_sorted[pos]
        assert np.isfinite(aligned).all()
        out[int(horizon)] = (aligned, spec.get("nb_log_dispersion"))
    return out


def test_history_registry_matches_model_agent_reader_contract(tmp_path):
    t = np.arange(10) * 300.0 + 1e9
    log_mu = np.log(np.linspace(5, 50, 10))
    arrays = tmp_path / "p_history_H_strong_1800s.npz"
    np.savez(arrays, anchor_time=t, log_mu_h=log_mu)
    registry = {"format": "group_event_state_v0_3_2_history_baseline_registry",
                "subjects": {"p": {"horizons": {"1800": {"arrays": str(arrays), "nb_log_dispersion": 1.2}}}}}
    path = tmp_path / "history_baseline_registry.json"
    path.write_text(json.dumps(registry))
    out = _reader_like_agent1(path, "p", t[::-1], (1800.0,))
    aligned, disp = out[1800]
    assert np.allclose(aligned, log_mu[::-1]) and disp == 1.2


def test_frozen_state_registry_loader_refuses_unknown_format(tmp_path):
    path = tmp_path / "frozen_state_registry.json"
    path.write_text(json.dumps({"format": "something_else", "patients": {}}))
    try:
        load_registry(path)
    except ValueError as exc:
        assert "refusing to guess" in str(exc)
    else:
        raise AssertionError("unknown format must be refused")
    path.write_text(json.dumps({"format": REGISTRY_FORMAT, "subjects": {"p": {"seeds": {}}}}))
    assert "p" in load_registry(path)["patients"]
    assert EXPECTED_SCHEMA["format"] == REGISTRY_FORMAT


def test_alignment_marks_missing_anchors_for_every_arm():
    src_t = np.array([0.0, 300.0, 900.0])
    src_v = np.array([[1.0], [2.0], [4.0]])
    grid = np.array([0.0, 300.0, 600.0, 900.0])
    aligned, n = align_by_time(src_t, src_v, grid)
    assert n == 3
    assert np.isnan(aligned[2, 0]) and aligned[3, 0] == 4.0


def test_anchor_held_event_state_uses_last_anchor_at_or_before_event_in_same_segment():
    anchor_t = np.array([300.0, 600.0, 900.0, 5000.0])
    anchor_seg = np.array([0, 0, 0, 1])
    anchor_state = np.array([[1.0], [2.0], [3.0], [9.0]])
    event_t = np.array([100.0, 650.0, 900.0, 4900.0, 5100.0])
    event_seg = np.array([0, 0, 0, 1, 1])
    held = anchor_held_event_states(anchor_t, anchor_seg, anchor_state, event_t, event_seg)
    assert np.isnan(held[0, 0])          # before any anchor of its segment
    assert held[1, 0] == 2.0             # anchor 600 held at 650
    assert held[2, 0] == 3.0             # anchor exactly at the event time is allowed (state excludes that event)
    assert np.isnan(held[3, 0]) and held[4, 0] == 9.0
