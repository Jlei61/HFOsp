"""Stream batching is a throughput device; it must not change the arithmetic."""

import copy
import json

import numpy as np
import pytest
import torch

from src.topic5_group_event_state.dataset import SubjectSequence
from src.topic5_group_event_state.train import (
    ENDPOINTS,
    GroupEventStateModel,
    TrainConfig,
    _data_shape,
    build_arms,
    run_sequence,
    run_streams,
)


def _write_fake_dataset(root, n_events=192, n_contacts=4, n_bands=2, n_views=2,
                        n_ctx=32, n_env=16, n_bg=3, n_sessions=2, seed=0):
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    specs = {
        "waveform": ((n_events, n_contacts, n_views, n_ctx), np.float16),
        "band_envelope": ((n_events, n_contacts, n_bands, n_env), np.float16),
        "band_features": ((n_events, n_contacts, n_bands, 5), np.float32),
        "cross_band_lag": ((n_events, n_contacts, 1), np.float32),
        "participation": ((n_events, n_contacts), np.bool_),
        "contact_ok": ((n_events, n_contacts), np.bool_),
        "relative_delay": ((n_events, n_contacts), np.float32),
        "tied_group_id": ((n_events, n_contacts), np.int16),
        "legacy_rank": ((n_events, n_contacts), np.int16),
        "background": ((n_events, n_contacts, n_bg), np.float32),
    }
    part = rng.random((n_events, n_contacts)) > 0.4
    part[:, 0] = True
    for name, (shape, dtype) in specs.items():
        if dtype == np.bool_:
            arr = part.copy() if name == "participation" else np.ones(shape, dtype=bool)
        elif name == "relative_delay":
            arr = (rng.random(shape) * 0.05).astype(dtype)
            arr[~part] = np.nan
        elif np.issubdtype(dtype, np.integer):
            arr = rng.integers(0, n_contacts, shape).astype(dtype)
        else:
            arr = rng.standard_normal(shape).astype(dtype)
        np.save(root / f"{name}.npy", arr)

    t = np.cumsum(rng.exponential(5.0, n_events)) + 1000.0
    session = np.zeros(n_events, dtype=np.int32)
    session[n_events // n_sessions :] = 1
    scalars = {
        "t_abs": t,
        "dt_prev": np.r_[np.nan, np.diff(t)].astype(np.float32),
        "session_of_event": session,
        "session_start": np.zeros(n_events, bool),
        "block_of_event": np.zeros(n_events, np.int32),
        "row_of_event": np.arange(n_events, dtype=np.int32),
        "has_waveform": np.ones(n_events, bool),
        "core_seconds": np.full(n_events, 0.25, np.float32),
        "background_age": np.full(n_events, 30.0, np.float32),
        "is_ictal": np.zeros(n_events, bool),
        "time_to_next_seizure": np.full(n_events, np.inf),
        "time_since_prev_seizure": np.full(n_events, np.inf),
        "interictal_index": np.arange(n_events),
    }
    with (root / "scalars.npz").open("wb") as fh:
        np.savez(fh, **scalars)
    index = {
        "format": "test", "subject": "fake_1", "dataset": "fake",
        "n_events": n_events, "n_contacts": n_contacts,
        "native_rate_hz": 1024.0, "detector_reference": "x", "montage_provenance": "x",
        "bipolar_equals_detector": False, "views": ["detector", "shaft_car"],
        "bands": ["ied_low", "ripple"], "band_available": [True, True],
        "band_feature_names": ["a", "b", "c", "d", "e"],
        "cross_band_pairs": [["ied_low", "ripple"]],
        "n_context_samples": n_ctx, "n_core_samples": 16, "core_seconds_nominal": 0.25,
        "envelope_bins": n_env, "background_feature_names": ["p", "q", "r"],
        "contacts": [{"lagpat_label": f"A{i+1}", "detector_label": f"A{i+1}",
                      "anode": f"A{i+1}", "cathode": None, "shaft": "A", "number": i + 1}
                     for i in range(n_contacts)],
        "tie_tolerance_seconds": 0.01,
        "sessions": [{"session_id": 0, "start_index": 0, "stop_index": n_events // n_sessions},
                     {"session_id": 1, "start_index": n_events // n_sessions, "stop_index": n_events}],
        "split_bounds_on_interictal_index": {
            "train": [0, int(n_events * 0.7)],
            "val": [int(n_events * 0.7), int(n_events * 0.8)],
            "test": [int(n_events * 0.8), n_events],
        },
        "split_fractions": [0.7, 0.1, 0.2], "n_seizures": 0, "seizures": [],
        "n_blocks": 1, "source_shards": [],
        "arrays": {n: {"file": f"{n}.npy", "shape": list(sp[0]), "dtype": np.dtype(sp[1]).name}
                   for n, sp in specs.items()},
    }
    (root / "index.json").write_text(json.dumps(index))
    return root


def test_stream_batching_reproduces_segment_by_segment_totals(tmp_path):
    root = _write_fake_dataset(tmp_path / "fake_1")
    seq = SubjectSequence(root)
    device = torch.device("cpu")
    arm = build_arms()["a3_delay_group_state"]
    torch.manual_seed(0)
    model = GroupEventStateModel(arm, _data_shape(seq), None, seq.history.shape[1])
    reference = copy.deepcopy(model.state_dict())

    lo, hi, n_streams, chunk = 0, 128, 4, 16
    cfg_multi = TrainConfig(chunk_events=chunk, n_streams=n_streams, amp=False)
    frozen = torch.optim.SGD(model.parameters(), lr=0.0)
    _means, extra_multi = run_streams(model, seq, lo, hi, device, cfg_multi, frozen)

    pooled = {k: [0.0, 0.0] for k in ENDPOINTS}
    cfg_single = TrainConfig(chunk_events=chunk, n_streams=1, amp=False)
    for a, b in seq.streams(lo, hi, n_streams):
        model.load_state_dict(reference)
        opt = torch.optim.SGD(model.parameters(), lr=0.0)
        _m, extra = run_sequence(
            model, seq, a, b, device, cfg_single, train=True, optimizer=opt
        )
        for k, (total, count) in extra["loss_totals"].items():
            pooled[k][0] += total
            pooled[k][1] += count

    for k in ENDPOINTS:
        got_total, got_count = extra_multi["loss_totals"][k]
        want_total, want_count = pooled[k]
        assert got_count == pytest.approx(want_count), f"{k}: observation count differs"
        assert got_total == pytest.approx(want_total, rel=1e-4, abs=1e-4), f"{k}: loss differs"


def test_padded_stream_slots_contribute_no_observations(tmp_path):
    # 100 events into 8 streams leaves ragged tails that must be masked out.
    root = _write_fake_dataset(tmp_path / "fake_2", n_events=100, seed=3)
    seq = SubjectSequence(root)
    device = torch.device("cpu")
    arm = build_arms()["a3_delay_group_state"]
    torch.manual_seed(1)
    model = GroupEventStateModel(arm, _data_shape(seq), None, seq.history.shape[1])
    frozen = torch.optim.SGD(model.parameters(), lr=0.0)
    cfg = TrainConfig(chunk_events=8, n_streams=8, amp=False)
    _means, extra = run_streams(model, seq, 0, 70, device, cfg, frozen)
    n_participation = extra["loss_totals"]["participation"][1]
    assert n_participation == pytest.approx(70 * seq.index["n_contacts"])
