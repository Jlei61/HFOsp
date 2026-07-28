import json
import os

import scripts.analyze_topic4_zm_phasec0 as A
import scripts.run_topic4_zm_phasec0_parallel as C


def _write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle)


def _identity_payload(manifest_sha="m"):
    return {
        "status": "complete",
        "manifest_sha256": manifest_sha,
        "spike_metrics": {
            "firing": {"rho80_active_core_median": 0.85},
            "isi": {
                "refractory_isi_fraction": 0.90,
                "isi_cv2": {"median": 0.05},
            },
            "pairwise_5ms": {
                "observed_median": 0.0,
                "null_q97_5": 0.05,
            },
            "spatial": {
                "active_grid_fraction": {"median": 0.30},
            },
        },
    }


def _gain_payload(rate, manifest_sha="m"):
    return {
        "status": "complete",
        "manifest_sha256": manifest_sha,
        "core_rate_hz": rate,
        "noise_bank_sha": "paired",
    }


def _populate_saturated_family(root, monkeypatch):
    old = A.OUT
    A.OUT = str(root)
    manifest = {
        "manifest_sha256": "m",
    }
    monkeypatch.setattr(A, "MANIFEST_PATH", str(root / "phasec_manifest.json"))
    monkeypatch.setattr(A.PCC, "validate_manifest", lambda value: None)
    _write(A.MANIFEST_PATH, manifest)
    try:
        for seed in A.SEEDS:
            for phase in A.PHASES:
                for noise in A.NOISES:
                    _write(A._identity_path("dt", seed, phase, noise), _identity_payload())
            for state in A.GAIN_STATES:
                for noise in A.NOISES:
                    _write(
                        A._gain_path("dt", seed, state, noise, 0.0, 0),
                        _gain_payload(100.0),
                    )
                    for delta in A.DELTAS:
                        gain = 20.0 if state == "pre_entry__natural" else 2.0
                        _write(
                            A._gain_path("dt", seed, state, noise, delta, -1),
                            _gain_payload(100.0 + gain * delta),
                        )
                        _write(
                            A._gain_path("dt", seed, state, noise, delta, +1),
                            _gain_payload(100.0 - gain * delta),
                        )
        return A.analyze("dt", A.SEEDS)
    finally:
        A.OUT = old


def test_phasec0_expected_matrix_has_all_locked_cells():
    identities, gains = A.expected_paths("dt", A.SEEDS)
    assert len(identities) == 3 * 2 * 3
    assert len(gains) == 3 * 3 * 3 * 5
    rows = C.tasks(("identity", "gain"))
    assert len(rows) == len(identities) + len(gains)
    assert len({row["key"] for row in rows}) == len(rows)


def test_old_scalar_fixture_is_blocked_not_promoted(tmp_path, monkeypatch):
    out = _populate_saturated_family(tmp_path, monkeypatch)
    assert out["n_missing"] == 0
    assert out["aggregate"]["verdict"] == "C0_no_evidence"
    assert all(
        row["klass"] == "C0_blocked_observables"
        for row in out["seed_rows"]
    )


def test_phasec0_missing_part_fails_closed(tmp_path, monkeypatch):
    old = A.OUT
    A.OUT = str(tmp_path)
    monkeypatch.setattr(A, "MANIFEST_PATH", str(tmp_path / "phasec_manifest.json"))
    monkeypatch.setattr(A.PCC, "validate_manifest", lambda value: None)
    _write(A.MANIFEST_PATH, {"manifest_sha256": "m"})
    try:
        out = A.analyze("dt", A.SEEDS)
    finally:
        A.OUT = old
    assert out["aggregate"]["verdict"] == "C0_no_evidence"
    assert out["n_missing"] == 153
