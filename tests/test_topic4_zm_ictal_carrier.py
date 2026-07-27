"""Carrier-gate metric extraction (spec 2026-07-24 §1-§4). Tested on SYNTHETIC signals so the
sustained-carrier vs HFO-burst-train discrimination and the Nyquist/provenance contracts are verified
without a multi-minute SNN run. The verdict LOGIC itself is tested in test_topic4_zm_carrier_verdict.py;
here we test that a synthetic sustained carrier extracts to gate-pass metrics and a synthetic burst train
extracts to gate-fail metrics.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import topic4_zm_ictal_carrier as C  # noqa: E402
from src.topic4_zm_carrier_verdict import ictal_carrier_verdict, is_sustained  # noqa: E402


# ---------------------------------------------------------------- CM1 decimation + Nyquist gate
def test_decimate_10k_to_2k_and_nyquist_gate():
    fs_in = 10000.0
    t = np.arange(int(1.0 * fs_in)) / fs_in
    lfp = np.stack([np.abs(np.sin(2 * np.pi * 60 * t)) + 1.0,        # 2 contacts, rectified + DC
                    np.abs(np.sin(2 * np.pi * 90 * t)) + 1.0], axis=1)
    ds, fs_out = C.decimate_lfp(lfp, fs_in)
    assert abs(fs_out - 2000.0) < 1e-9
    assert ds.shape[1] == 2 and ds.shape[0] == lfp.shape[0] // 5
    C.assert_nyquist(fs_out)                                          # 1000 Hz > 150 -> ok
    try:
        C.assert_nyquist(200.0)                                       # Nyquist 100 < 150 -> must raise
    except ValueError:
        return
    raise AssertionError("assert_nyquist must raise when Nyquist <= 150 Hz")


def test_native_lfp_sample_rate_tracks_dt_for_resolution_confirmation():
    assert C.lfp_sample_hz(0.1) == 10_000.0
    assert C.lfp_sample_hz(0.05) == 20_000.0
    x = np.ones((20_000, 2))
    ds, fs_out = C.decimate_lfp(x, fs_in=C.lfp_sample_hz(0.05))
    assert ds.shape == (4_000, 2)
    assert fs_out == 4_000.0


# ---------------------------------------------------------------- CM2-CM5 observed band envelopes
def _rect_carrier(fs, dur_s, f=60.0, amp=1.0, dc=1.0):
    t = np.arange(int(dur_s * fs)) / fs
    return np.abs(amp * np.sin(2 * np.pi * f * t)) + dc


def _rect_train(fs, dur_s, f=60.0, on_ms=80.0, period_ms=600.0, amp=1.0, dc=1.0):
    t = np.arange(int(dur_s * fs)) / fs
    sig = np.abs(amp * np.sin(2 * np.pi * f * t)) + dc
    gate = ((t * 1000.0) % period_ms) < on_ms                        # 80 ms on / 520 ms off -> gaps > 250 ms
    return np.where(gate, sig, dc)                                    # between bursts: DC only (no 60 Hz)


def test_sustained_carrier_gives_ge2_sustained_contacts_train_gives_fewer():
    fs = 2000.0
    pre = np.full(int(0.5 * fs), 1.0)                                 # 0.5 s quiet baseline (DC only)
    car = np.concatenate([pre, _rect_carrier(fs, 5.0)])              # onset at 500 ms, 5 s carrier
    trn = np.concatenate([pre, _rect_train(fs, 5.0)])
    lfp_car = np.stack([car, car, car], axis=1)                     # 3 contacts
    lfp_trn = np.stack([trn, trn, trn], axis=1)
    obs_car = C.compute_observed_metrics(lfp_car, fs, onset_ms=500.0)
    obs_trn = C.compute_observed_metrics(lfp_trn, fs, onset_ms=500.0)
    assert obs_car["n_sustained_contacts"] >= 2, obs_car
    assert obs_trn["n_sustained_contacts"] < 2, obs_trn
    assert is_sustained(obs_car["best_macro"])
    assert not is_sustained(obs_trn.get("best_macro") or {})


# ---------------------------------------------------------------- CM6 source fine rates
def test_fine_rates_recover_core_and_surround_hz():
    dt_ms = 0.1
    NE = 100
    n = 20000                                                        # 2000 ms
    core = np.zeros(NE, bool); core[:20] = True
    spk = np.zeros((n, NE), bool)
    # core cells fire every 10 ms (100 Hz), surround silent
    spk[::100, :20] = True
    r = C.fine_rates(spk, core, dt_ms, bin_ms=5.0)
    assert abs(np.median(r["core"]) - 100.0) < 20.0                  # ~100 Hz core
    assert r["surround"].max() == 0.0
    assert 0.0 < r["active_frac"].max() <= 1.0


# ---------------------------------------------------------------- end-to-end source metrics
def _sustained_core_raster(NE=200, n=40000, dt_ms=0.1, core_n=40):
    """A synthetic sustained core carrier: core cells fire densely (every ~5 ms) from 300 ms to the end;
    surround recruits a bit. -> gate A should read sustained + recruitment."""
    core = np.zeros(NE, bool); core[:core_n] = True
    spk = np.zeros((n, NE), bool)
    onset = int(300 / dt_ms)
    for tt in range(onset, n, int(5 / dt_ms)):                       # dense 200 Hz-ish core bursts
        spk[tt, :core_n] = True
        if tt > onset + int(500 / dt_ms):                           # surround recruits after 500 ms (spread)
            spk[tt, core_n:core_n + 30] = True
    return spk, core


def test_sustained_core_raster_reads_as_sustained_source_macroepisode():
    dt_ms = 0.1
    spk, core = _sustained_core_raster()
    NE = spk.shape[1]
    rng = np.random.default_rng(0)
    posE = rng.random((NE, 2)) * 20.0
    src_xy = posE[:40].mean(axis=0); axis_unit = np.array([1.0, 0.0])
    src = C.compute_source_metrics(spk, core, posE, src_xy, axis_unit, 20.0, dt_ms,
                                   runaway_early_stop_ms=None)
    assert is_sustained(src["macro"]), src["macro"]
    assert src["has_recruitment"] is True
    assert src["whole_field_flash"] is False


# ---------------------------------------------------------------- CM11 provenance atomic + resume
def test_provenance_atomic_manifest_no_overwrite_and_resume(tmp_path):
    mpath = str(tmp_path / "carrier_gate_seed1.json")
    rec_bare = dict(arm="bare", status="complete", output_files=["bare_seed1.npz"], seed=1)
    rec_sg = dict(arm="sg", status="complete", output_files=["sg_seed1.npz"], seed=1)
    C.write_arm_to_manifest(mpath, rec_bare)
    C.write_arm_to_manifest(mpath, rec_sg)                           # must NOT clobber bare
    man = json.load(open(mpath))
    assert set(man["arms"].keys()) == {"bare", "sg"}
    assert man["arms"]["bare"]["output_files"] == ["bare_seed1.npz"]
    # resume: bare is complete in the manifest -> arm_completed True; a never-run arm -> False
    assert C.arm_completed(man, "bare") is True
    assert C.arm_completed(man, "sgh") is False


def test_git_sha_and_file_hash(tmp_path):
    f = tmp_path / "x.npz"
    f.write_bytes(b"hello")
    h = C.sha256_file(str(f))
    assert len(h) == 64 and h == C.sha256_file(str(f))               # deterministic
    sha = C.git_sha(os.path.join(os.path.dirname(__file__), ".."))
    assert isinstance(sha, str) and len(sha) >= 7                    # a real short/long SHA
