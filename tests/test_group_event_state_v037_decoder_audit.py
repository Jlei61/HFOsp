from __future__ import annotations

import json

import numpy as np

from scripts.audit_group_event_state_v037_decoder_provenance import audit_subject


def _json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_decoder_audit_distinguishes_selection_safe_from_strict(tmp_path, monkeypatch) -> None:
    subject, fit = "p", "fit"
    monkeypatch.setitem(
        __import__(audit_subject.__module__, fromlist=["V035_DECODER_FITS"]).V035_DECODER_FITS,
        subject,
        fit,
    )
    decoder = tmp_path / "decoder"
    inputs = tmp_path / "inputs"
    cache = decoder / "cache" / fit
    cache.mkdir(parents=True)
    np.savez(cache / "events.npz", event_abs_time=np.array([5.0, 25.0, 65.0, 75.0]), split=np.array([0, 1, 2, -1]))
    np.savez(cache / "events_raw.npz", event_abs_time=np.array([5.0]))
    np.savez(cache / "plane.npz", x=np.array([1]))
    np.savez(cache / "train_only_modes.npz", x=np.array([1]))
    _json(cache / "provenance.json", {"scope": "own_a", "v034_recorded_time_split": {"rule": "test"}})
    manifest = inputs / subject / "manifest_v3.json"
    _json(manifest, {"report": {"phase_boundaries_epoch": {"20pct": 20.0, "60pct": 60.0, "70pct": 70.0, "80pct": 80.0}}})
    for seed in (0, 1, 2):
        unit = decoder / "formal_units" / fit / "L3_LOCAL_PLUS_LEARNED_LR" / f"seed{seed}"
        unit.mkdir(parents=True)
        _json(unit / "metrics.json", {"best_checkpoint_eligible": True, "target_values_read": False, "converged": True})
        _json(unit / "DONE.json", {"done": True})
        (unit / "weights.pt").write_bytes(f"seed{seed}".encode())
    row = audit_subject(subject, decoder, inputs)
    assert row["selection_safe"] is True
    assert row["strict_pre_state_fit"] is False
    assert row["status"] == "SELECTION_SAFE_ONLY"
