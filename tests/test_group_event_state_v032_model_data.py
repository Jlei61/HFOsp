"""Task 4: history baseline provider + subject bundle (design §1/§8, D1-D4)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.topic5_group_event_state.v02.timeline import CoverageSegment
from src.topic5_group_event_state.v03.partition import PHASE_NAMES, nested_time_partition
from src.topic5_group_event_state.v032_model.data import SubjectBundle, bundle_from_arrays
from src.topic5_group_event_state.v032_model.history_baseline import (
    HistoryBaseline,
    fit_provisional_history_baseline,
    load_agent2_history_baseline,
    load_endpoint_eligibility,
)


def _toy_timeline(seed=0):
    """Segments 0-10000 s and 20000-30000 s; anchors every 300 s; three horizons."""

    rng = np.random.default_rng(seed)
    segments = [CoverageSegment(0, 0, 0.0, 10_000.0), CoverageSegment(1, 1, 20_000.0, 30_000.0)]
    events = np.sort(np.concatenate([
        rng.uniform(0.0, 10_000.0, 600), rng.uniform(20_000.0, 30_000.0, 500)]))
    event_segment = (events >= 20_000.0).astype(np.int64)
    horizons = (300.0, 1800.0, 7200.0)
    t_anchor = np.concatenate([np.arange(300.0, 10_000.0, 300.0), np.arange(20_300.0, 30_000.0, 300.0)])
    seg_of = (t_anchor >= 20_000.0).astype(np.int64)
    seg_stop = np.where(seg_of == 0, 10_000.0, 30_000.0)
    seg_start = np.where(seg_of == 0, 0.0, 20_000.0)
    a, h = t_anchor.size, len(horizons)
    eligible = np.zeros((a, h), bool)
    lo = np.zeros((a, h), np.int64)
    hi = np.zeros((a, h), np.int64)
    for i, horizon in enumerate(horizons):
        eligible[:, i] = t_anchor + horizon <= seg_stop
        lo[:, i] = np.searchsorted(events, t_anchor, side="left")
        hi[:, i] = np.searchsorted(events, t_anchor + horizon, side="left")
    pos = np.searchsorted(events, t_anchor, side="left") - 1
    in_seg = (pos >= 0) & (events[np.clip(pos, 0, None)] >= seg_start)
    last = np.where(in_seg, pos, -1)
    x = rng.normal(size=(a, 6))
    names = ("rate_tau60", "rate_tau1800", "clock_sin_day", "log_time_since_prev_seizure", "f4", "f5")
    grid = SimpleNamespace(
        t_anchor=t_anchor, segment_index=seg_of, session_id=seg_of, last_event_pos=last,
        eligible=eligible, window_lo=lo, window_hi=hi, horizons_seconds=horizons,
        seconds_since_last_event=np.where(last >= 0, t_anchor - events[np.clip(last, 0, None)], np.inf),
        n_anchors=a,
    )
    tl = SimpleNamespace(
        subject="toy", segments=segments, event_times=events, event_segment=event_segment,
        stream_positions=np.arange(events.size), grid=grid,
        baseline=SimpleNamespace(x=x, names=names),
        config=SimpleNamespace(horizons_seconds=horizons),
    )
    return tl, nested_time_partition(segments)


def test_d1_anchor_mask_requires_phase_eligibility_and_window_inside_phase():
    tl, part = _toy_timeline()
    bundle = bundle_from_arrays(
        tl, part, x_raw=np.zeros((tl.event_times.size, 2), np.float32),
        feature_names=("a", "b"), history=HistoryBaseline({}, {}, "none", {}),
        eligibility=None, fingerprint={},
    )
    for phase_index, phase in enumerate(PHASE_NAMES):
        m = bundle.anchor_mask(phase, 1800.0)
        lo_b, hi_b = part.bounds(phase)
        assert np.all(bundle.t_anchor[m] >= lo_b) and np.all(bundle.t_anchor[m] + 1800.0 <= hi_b + 1e-6)
        assert np.all(bundle.eligible[m, 1])
        assert np.all(bundle.anchor_phase[m] == phase_index)
    assert bundle.train_event_mask().sum() == int((bundle.event_phase == 1).sum()) > 0
    # counts are the events strictly inside [t, t+h)
    i = np.flatnonzero(bundle.anchor_mask("state_train", 1800.0))[0]
    manual = int(((tl.event_times >= bundle.t_anchor[i]) & (tl.event_times < bundle.t_anchor[i] + 1800.0)).sum())
    assert bundle.counts[i, 1] == manual
    assert bundle.effective_independent_windows("state_train", 1800.0) >= 1


def test_d2_provisional_h_is_fit_on_state_train_and_selected_on_dev_val_only():
    tl, part = _toy_timeline()
    base = fit_provisional_history_baseline(tl, part, (300.0, 1800.0), seed=0)
    assert base.source == "provisional_local"
    assert set(base.log_mu) == {300, 1800}
    assert base.log_mu[1800].shape == (tl.grid.t_anchor.size,)
    assert np.isfinite(base.log_mu[1800]).all()
    assert all("seizure" not in n for n in base.meta["feature_names_used"])
    # perturbing development-test counts must not change any prediction
    tl2, _ = _toy_timeline()
    test_anchor = part.labels_of(tl2.grid.t_anchor) == 3
    tl2.grid.window_hi[test_anchor] = tl2.grid.window_hi[test_anchor] + 40
    tl2.event_times = np.concatenate([tl2.event_times, np.full(40, 29_999.0)])
    base2 = fit_provisional_history_baseline(tl2, part, (300.0, 1800.0), seed=0)
    assert np.allclose(base.log_mu[1800], base2.log_mu[1800])


def test_d3_missing_agent2_registry_is_not_silently_replaced(tmp_path):
    result, reason = load_agent2_history_baseline(
        tmp_path / "absent.json", "toy", np.arange(3.0), (1800.0)
    )
    assert result is None and "missing" in reason


def test_d4_agent2_registry_arrays_align_on_anchor_time(tmp_path):
    t_anchor = np.array([0.0, 300.0, 600.0, 900.0])
    npz = tmp_path / "toy_1800.npz"
    np.savez(npz, anchor_time=t_anchor[::-1], log_mu_h=np.log(np.array([4.0, 3.0, 2.0, 1.0])))
    registry = {
        "format": "toy",
        "subjects": {"toy": {"horizons": {"1800": {"arrays": str(npz), "nb_log_dispersion": 1.5,
                                                    "fit_scope": "state_train"}}}},
    }
    path = tmp_path / "history_baseline_registry.json"
    path.write_text(json.dumps(registry))
    base, reason = load_agent2_history_baseline(path, "toy", t_anchor, (1800.0,))
    assert base is not None, reason
    assert base.source == "agent2_registry"
    assert np.allclose(base.log_mu[1800], np.log([1.0, 2.0, 3.0, 4.0]))
    assert base.nb_log_dispersion[1800] == 1.5
    bad, reason = load_agent2_history_baseline(path, "toy", t_anchor + 1.0, (1800.0,))
    assert bad is None and "align" in reason
    assert load_endpoint_eligibility(tmp_path / "none.json", "toy") is None
    (tmp_path / "e.json").write_text(json.dumps({"subjects": {"toy": {"7200": {"eligible": False}}}}))
    assert load_endpoint_eligibility(tmp_path / "e.json", "toy") == {"7200": {"eligible": False}}
