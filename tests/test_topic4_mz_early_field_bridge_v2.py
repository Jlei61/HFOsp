"""Contract tests for the MZ early-field bridge V2 (z+m, tau_adp=500) — task §6/§8/§12.

Two groups:
  A. frozen-candidate contract  (config-only; no engine, no SNN run)
  B. slow-off reuse fail-closed  (imports the runner; uses a FAKE substrate + tmp fixtures, no SNN run)

Every test FAILS if the corresponding invariant is broken.
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
for _p in (os.path.join(ROOT, "scripts"), os.path.join(ROOT, "src", "snn_engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

V2_CONFIG = os.path.join(ROOT, "config", "topic4_mz_early_field_bridge_v2_zm.yaml")
CALIB = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_slowvars", "calibration.json")


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(open(V2_CONFIG))


# ============================================================ A. frozen-candidate contract
def test_candidate_frozen_values(cfg):
    c = cfg["candidate"]["cfg"]
    assert cfg["candidate"]["label"] == "zA_q75_tz5000__mA0p001_tau500"
    assert c["use_z"] is True and c["use_m"] is True
    assert c["I_th_EI"] == 95.19851312666987
    assert c["tau_z"] == 5000.0
    assert c["tau_adp"] == 500.0
    assert c["eta_m"] == 0.007451594355587098
    assert cfg["candidate"]["A_target"] == 0.001


def test_A_target_not_in_engine_cfg(cfg):
    """A_target is a derivation label, NOT an MZSlowVarsConfig field — it must stay OUT of candidate.cfg
    so the MZSlowVarsConfig(**cfg) splat does not TypeError, and the splat must actually construct."""
    from mz_slow_vars import MZSlowVarsConfig
    c = cfg["candidate"]["cfg"]
    assert "A_target" not in c
    engine_fields = {f for f in MZSlowVarsConfig.__dataclass_fields__}
    assert set(c).issubset(engine_fields), f"non-engine keys in candidate.cfg: {set(c) - engine_fields}"
    mz = MZSlowVarsConfig(**c)                      # must not raise
    assert mz.use_m is True and mz.eta_m > 0.0 and mz.tau_adp == 500.0


def test_eta_m_matches_committed_calibration(cfg):
    """eta_m must equal A_target * I_EE_scale / peak_m_tau2000 read from the committed calibration
    (never hand-copied). The committed tau500 runs used the tau2000 peak, NOT peak_m.tau500."""
    calib = json.load(open(CALIB))
    a_target = float(cfg["candidate"]["A_target"])
    derived = a_target * float(calib["I_EE_scale"]) / float(calib["peak_m"]["tau2000"])
    assert abs(float(cfg["candidate"]["cfg"]["eta_m"]) - derived) < 1e-12
    # and it must NOT equal the tau500-peak normalization (which would break onset reproduction)
    derived_tau500 = a_target * float(calib["I_EE_scale"]) / float(calib["peak_m"]["tau500"])
    assert abs(float(cfg["candidate"]["cfg"]["eta_m"]) - derived_tau500) > 1e-3


def test_slowoff_T_decoupled_from_native_T(cfg):
    """V2 native runs T=20000 (task §6) but the slow-off templates come from T=15000 so they are
    identical to V1 (clean paired comparison)."""
    assert cfg["T_ms"] == 20000.0
    assert cfg["slowoff_T_ms"] == 15000.0


def test_readout_settings_identical_to_v1(cfg):
    """The V1<->V2 comparison isolates m ONLY if every readout/null/window/detector setting matches V1."""
    v1 = yaml.safe_load(open(os.path.join(ROOT, "config", "topic4_mz_early_field_bridge.yaml")))
    for k in ("event_detector", "timing", "split", "onset", "windows", "source_grid",
              "core_excluded", "participation_audit", "nulls"):
        assert cfg[k] == v1[k], f"V2 readout block '{k}' differs from V1"


# ============================================================ B. slow-off reuse fail-closed (task §8/§12)
def _fake_substrate(names, contacts):
    return {"reg": {"montage_sheet": SimpleNamespace(names=list(names), contacts=np.asarray(contacts, float))}}


def _write_reuse_dir(root, seed, *, names, contacts, shas, dt=0.1, elig_a=True, elig_b=True, status="complete"):
    sd = os.path.join(root, "per_seed", f"seed{seed}")
    os.makedirs(sd, exist_ok=True)
    np.savez_compressed(os.path.join(sd, "slowoff.npz"),
                        names=np.array(names, object), contacts=np.asarray(contacts, np.float32),
                        qmed=np.zeros(len(names), np.float32), qmad=np.zeros(len(names), np.float32),
                        src_quiet_ref=np.zeros(576, np.float32), bin_w=1.0, floor=0.0, bar=0.02, af_max=0.1,
                        af=np.zeros(100, np.float32), times=np.arange(100, dtype=np.float32),
                        r20=np.zeros(10, np.float32))
    np.savez_compressed(os.path.join(sd, "templates.npz"),
                        contact_A=np.zeros(len(names), np.float32), contact_B=np.zeros(len(names), np.float32),
                        contact_A_train=np.zeros(len(names), np.float32),
                        contact_B_train=np.zeros(len(names), np.float32),
                        source_A=np.zeros(576, np.float32), source_B=np.zeros(576, np.float32))
    json.dump({"status": status, "provenance": {"engine_shas": shas, "dt": dt}},
              open(os.path.join(sd, "bridge_metrics.json"), "w"))
    json.dump({"seed": seed, "n_returning": 38}, open(os.path.join(sd, "slowoff.json"), "w"))
    tj = {"templates": {"A_to_B": {"contact": {"eligible": elig_a, "n_train": 4, "n_heldout": 3, "n_shared": 8},
                                   "source": {"eligible": elig_a, "n_train": 4, "n_heldout": 3, "n_shared": 8}},
                        "B_to_A": {"contact": {"eligible": elig_b, "n_train": 13, "n_heldout": 13, "n_shared": 15},
                                   "source": {"eligible": elig_b, "n_train": 13, "n_heldout": 13, "n_shared": 15}}}}
    json.dump(tj, open(os.path.join(sd, "templates.json"), "w"))
    return sd


NAMES = [f"A{i}" for i in range(1, 8)] + [f"B{i}" for i in range(1, 9)]        # 15 contacts
CONTACTS = np.column_stack([np.linspace(0, 14, 15), np.zeros(15)])
SHAS = {"kick_probe.py": "aaa", "params.py": "bbb", "model.py": "ccc",
        "connectivity.py": "ddd", "connectivity_rot.py": "eee", "lfp.py": "fff"}


def test_reuse_passes_when_everything_matches(tmp_path):
    import run_topic4_mz_early_field_bridge as R
    _write_reuse_dir(str(tmp_path), 1, names=NAMES, contacts=CONTACTS, shas=SHAS)
    S = _fake_substrate(NAMES, CONTACTS)
    sd = R.verify_slowoff_reuse(str(tmp_path), 1, S, expected_shas=SHAS)   # must NOT raise
    assert sd.endswith(os.path.join("per_seed", "seed1"))


def test_reuse_fail_closed_missing_artifacts(tmp_path):
    import run_topic4_mz_early_field_bridge as R
    S = _fake_substrate(NAMES, CONTACTS)
    with pytest.raises(FileNotFoundError):
        R.verify_slowoff_reuse(str(tmp_path), 1, S, expected_shas=SHAS)     # nothing written


def test_reuse_fail_closed_sha_mismatch(tmp_path):
    import run_topic4_mz_early_field_bridge as R
    bad = dict(SHAS, **{"model.py": "XXXX"})
    _write_reuse_dir(str(tmp_path), 1, names=NAMES, contacts=CONTACTS, shas=bad)
    S = _fake_substrate(NAMES, CONTACTS)
    with pytest.raises(ValueError, match="engine SHA"):
        R.verify_slowoff_reuse(str(tmp_path), 1, S, expected_shas=SHAS)


def test_reuse_fail_closed_contact_order_mismatch(tmp_path):
    import run_topic4_mz_early_field_bridge as R
    _write_reuse_dir(str(tmp_path), 1, names=NAMES, contacts=CONTACTS, shas=SHAS)
    S = _fake_substrate(list(reversed(NAMES)), CONTACTS)                     # same set, different order
    with pytest.raises(ValueError, match="NAME/order"):
        R.verify_slowoff_reuse(str(tmp_path), 1, S, expected_shas=SHAS)


def test_reuse_fail_closed_coord_mismatch(tmp_path):
    import run_topic4_mz_early_field_bridge as R
    _write_reuse_dir(str(tmp_path), 1, names=NAMES, contacts=CONTACTS, shas=SHAS)
    S = _fake_substrate(NAMES, CONTACTS + 0.5)                               # names match, coords shifted
    with pytest.raises(ValueError, match="COORDINATE"):
        R.verify_slowoff_reuse(str(tmp_path), 1, S, expected_shas=SHAS)


def test_reuse_fail_closed_ineligible_templates(tmp_path):
    import run_topic4_mz_early_field_bridge as R
    _write_reuse_dir(str(tmp_path), 1, names=NAMES, contacts=CONTACTS, shas=SHAS, elig_a=False)
    S = _fake_substrate(NAMES, CONTACTS)
    with pytest.raises(ValueError, match="eligible"):
        R.verify_slowoff_reuse(str(tmp_path), 1, S, expected_shas=SHAS)


def test_reuse_fail_closed_incomplete_status(tmp_path):
    import run_topic4_mz_early_field_bridge as R
    _write_reuse_dir(str(tmp_path), 1, names=NAMES, contacts=CONTACTS, shas=SHAS, status="failed")
    S = _fake_substrate(NAMES, CONTACTS)
    with pytest.raises(ValueError, match="status"):
        R.verify_slowoff_reuse(str(tmp_path), 1, S, expected_shas=SHAS)
