"""Task 3 gate (spec rev3.1 §3.1/§2.2): exact snapshot -> restore -> continue, and freeze semantics.

The hard gate for the whole line: a continuous run and a split run (checkpoint at t_f, serialize,
reload, continue) must be byte-identical -- raster, rates, current-based virtual SEEG, every final
simulator array, and the RNG progression. Forked at three NATURAL fast phases (trough / rising /
peak), never at a manufactured state.

`test_every_current_affecting_state_is_load_bearing` is the sensitivity proof: for each snapshot
field the inventory calls current-affecting, corrupting just that field must break the continuation.
It simultaneously validates the inventory's `current_effect` column against the running engine.
"""
import copy
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "src", "snn_engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from kick_probe import simulate_kick  # noqa: E402
from lfp import LFPRecorder  # noqa: E402
from params import Params  # noqa: E402
from connectivity import place_neurons, build_connectivity  # noqa: E402
from slow_field import SpatialSlowField, SpatialSlowFieldConfig  # noqa: E402

import src.topic4_zm_checkpoint as CK  # noqa: E402
import src.topic4_zm_fork_state as FS  # noqa: E402

SEED = 1
T_TOTAL = 260.0
DT = 0.1


def _substrate():
    p = Params(L=1.0, density=400.0, T=T_TOTAL, dt=DT, seed=SEED, nu_ext_ratio=1.0)
    rng = np.random.default_rng(SEED)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity(p, pos, labels, NE, NI, rng, verbose=False)
    return p, net, pos, labels, NE, NI


def _slow(pos, NE, N, L):
    core = np.linalg.norm(pos[:NE] - np.array([L / 2, L / 2]), axis=1) <= 0.3
    cfg = SpatialSlowFieldConfig(use_qI=False, use_gK=False, use_z=True, use_m=True,
                                 tau_z=200.0, I_th_EI=0.6, tau_adp=200.0, eta_m=0.5,
                                 use_SG=True, alpha_G=16.0, r50_psi=0.05, n_grid=16)
    return SpatialSlowField(N, 18.0, pos[:NE], pos[NE:], L, core_mask_E=core, cfg=cfg)


def _run(p, net, pos, NE, NI, *, T, ckpt=None, slow_wrap=None, vth_shift=0.0, es=None):
    N = NE + NI
    vth = np.full(N, 18.0 + vth_shift)
    vth[:5] = 16.0 + vth_shift
    slow = _slow(pos, NE, N, p.L)
    if slow_wrap is not None:
        slow = FS.FreezeWrapper(slow, slow_wrap)
    rec = LFPRecorder(p, net["pos"], net["labels"])
    net["rng"] = np.random.default_rng(SEED)
    pp = Params(**{**p.__dict__, "T": T})
    res = simulate_kick(pp, net, 5.0, slow=slow, kick_center=np.array([p.L / 2, p.L / 2]),
                        r_kick=0.3, t_kick=50.0, V_th_per_neuron=vth, lfp_recorder=rec,
                        verbose=False, zm_ckpt=ckpt, **(es or {}))
    return res, slow


@pytest.fixture(scope="module")
def reference():
    """One continuous run + snapshots at three natural fast phases."""
    p, net, pos, labels, NE, NI = _substrate()
    probe, _ = _run(p, net, pos, NE, NI, T=T_TOTAL)
    r = probe["rate_E"]
    n = r.size
    lo, hi = int(0.35 * n), int(0.75 * n)
    seg = r[lo:hi]
    peak = lo + int(np.argmax(seg))
    trough = lo + int(np.argmin(seg))
    rising = max(lo, peak - 15)
    forks = sorted({int(trough), int(rising), int(peak)})
    ck = CK.ZMCheckpoint(snapshot_steps=forks, dump_ext=True)
    full, full_slow = _run(p, net, pos, NE, NI, T=T_TOTAL, ckpt=ck)
    return dict(p=p, net=net, pos=pos, NE=NE, NI=NI, full=full, snaps=ck.snapshots, forks=forks,
                full_slow=full_slow)


def _continue(reference, state, n_steps, *, slow_wrap=None, ckpt_kw=None, es=None):
    p, net, pos, NE, NI = (reference[k] for k in ("p", "net", "pos", "NE", "NI"))
    ck = CK.ZMCheckpoint(initial_state=state, dump_ext=True, **(ckpt_kw or {}))
    res, slow = _run(p, net, pos, NE, NI, T=n_steps * DT, ckpt=ck, slow_wrap=slow_wrap, es=es)
    return res, slow, ck


# ---------------------------------------------------------------- exact resume
@pytest.mark.parametrize("phase", [0, 1, 2])
def test_split_resume_is_byte_identical_at_three_natural_fast_phases(reference, phase):
    tf = reference["forks"][phase]
    n_rest = reference["full"]["E_spk_bool"].shape[0] - tf
    res, _, _ = _continue(reference, reference["snaps"][tf], n_rest)
    ref = reference["full"]
    assert res["E_spk_bool"].shape[0] == n_rest
    assert np.array_equal(res["E_spk_bool"], ref["E_spk_bool"][tf:]), "raster diverged"
    assert np.array_equal(res["rate_E"], ref["rate_E"][tf:])
    assert np.array_equal(res["rate_I"], ref["rate_I"][tf:])
    assert np.array_equal(res["lfp_trace"], ref["lfp_trace"][tf:]), "current-vSEEG diverged"
    assert np.array_equal(res["zm_ext_nu"], ref["zm_ext_nu"][tf:]), "external drive diverged"
    assert np.array_equal(res["zm_ext_sum"], ref["zm_ext_sum"][tf:])


def test_final_simulator_arrays_and_rng_match_after_resume(reference):
    tf = reference["forks"][1]
    n_rest = reference["full"]["E_spk_bool"].shape[0] - tf
    _, _, ck = _continue(reference, reference["snaps"][tf], n_rest,
                         ckpt_kw=dict(return_final_state=True))
    t0 = reference["forks"][0]
    _, _, ck_full = _continue(reference, reference["snaps"][t0],
                              reference["full"]["E_spk_bool"].shape[0] - t0,
                              ckpt_kw=dict(return_final_state=True))
    a, b = ck.final_state, ck_full.final_state
    assert a is not None and b is not None
    for k in sorted(set(a) & set(b)):
        if k == "rng_state":
            assert a[k] == b[k], "RNG state progressed differently"
        else:
            assert np.array_equal(a[k], b[k]), f"final state {k} differs"


def test_resume_is_reproducible_not_an_rng_coincidence(reference):
    tf = reference["forks"][2]
    n = 300
    r1, _, _ = _continue(reference, reference["snaps"][tf], n)
    r2, _, _ = _continue(reference, copy.deepcopy(reference["snaps"][tf]), n)
    assert np.array_equal(r1["E_spk_bool"], r2["E_spk_bool"])


def test_ring_slot_phase_mutation_breaks_parity(reference):
    """The absolute-step arithmetic (`tg = t + t_start`) is load-bearing: resuming with the delay-ring
    phase off by one must diverge, proving the parity above is not accidentally insensitive."""
    tf = reference["forks"][1]
    n = 400
    good, _, _ = _continue(reference, reference["snaps"][tf], n)
    bad_state = dict(reference["snaps"][tf])
    bad_state["t"] = np.asarray(int(bad_state["t"]) + 1)
    bad, _, _ = _continue(reference, bad_state, n)
    assert not np.array_equal(good["E_spk_bool"], bad["E_spk_bool"])


def test_every_current_affecting_state_is_load_bearing(reference):
    """Corrupt one snapshot field at a time. Fields the inventory calls current-affecting MUST break
    the continuation; fields it calls `current_effect=none` must not be silently load-bearing."""
    inv = {r["name"]: r for r in FS.build_state_inventory()}
    tf = reference["forks"][1]
    n = 250
    good, _, _ = _continue(reference, reference["snaps"][tf], n)
    checked = 0
    for key in sorted(reference["snaps"][tf]):
        if key in ("t", "rng_state"):
            continue  # covered by their own tests above
        row = inv.get(key)
        if row is None:
            pytest.fail(f"snapshot key {key!r} has no inventory row")
        st = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in reference["snaps"][tf].items()}
        a = np.asarray(st[key], dtype=float)
        st[key] = np.asarray(a * 0.0 + 0.5) if a.ndim else np.asarray(float(a) + 0.5)
        got, _, _ = _continue(reference, st, n)
        differs = not np.array_equal(good["E_spk_bool"], got["E_spk_bool"])
        if row["current_effect"] in ("direct", "indirect"):
            assert differs, f"{key}: inventory says current-affecting but corrupting it changed nothing"
            checked += 1
        else:
            assert not differs, f"{key}: inventory says current_effect=none but it changed the raster"
    assert checked >= 10, f"only {checked} current-affecting fields exercised"


def test_early_stop_controller_state_is_carried_and_matters(reference):
    """`_es_ema`/`_es_run` never touch the membrane, but they decide WHEN the loop truncates. With
    the runaway early-stop armed, a continuation restored one step short of the trigger must stop
    earlier than the same continuation restored with the counter cleared."""
    tf = reference["forks"][1]
    es = dict(early_stop_runaway=True, es_thresh_hz=1.0, es_dur_ms=20.0)
    n = 600
    near = dict(reference["snaps"][tf])
    near["_es_ema"] = np.asarray(50.0)          # already far above threshold
    near["_es_run"] = np.asarray(float(int(round(es["es_dur_ms"] / DT)) - 1))
    cleared = dict(near)
    cleared["_es_run"] = np.asarray(0.0)
    a, _, _ = _continue(reference, near, n, es=es)
    b, _, _ = _continue(reference, cleared, n, es=es)
    assert a["E_spk_bool"].shape[0] < b["E_spk_bool"].shape[0], \
        "the early-stop counter was not carried across the checkpoint"
    assert a["E_spk_bool"].shape[0] == 1


# ---------------------------------------------------------------- freeze semantics
def test_freeze_keeps_the_spatial_field_and_the_current_effect(reference):
    tf = reference["forks"][1]
    n = 400
    snap = reference["snaps"][tf]
    dyn, _, _ = _continue(reference, snap, n, slow_wrap=FS.FreezePolicy.for_arm("dynamic_replay"))
    frz, slow, _ = _continue(reference, snap, n, slow_wrap=FS.FreezePolicy.for_arm("freeze_zm"))
    assert np.array_equal(dyn["E_spk_bool"], reference["full"]["E_spk_bool"][tf:tf + n]), \
        "the all-False freeze policy must be a pure pass-through"
    # z/m held element-wise at the snapshot field (not a mean, not a reset)
    assert np.array_equal(slow.inner.z, snap["slow.z"])
    assert np.array_equal(slow.inner.m, snap["slow.m"])
    assert slow.inner.z.std() > 0, "a frozen field must keep its spatial structure"
    # ...and they are still READ every step, so the trajectory differs from dynamic z/m
    assert not np.array_equal(dyn["E_spk_bool"], frz["E_spk_bool"])


def test_primary_sg_freeze_holds_the_whole_pool_family(reference):
    tf = reference["forks"][1]
    snap = reference["snaps"][tf]
    _, slow, _ = _continue(reference, snap, 300, slow_wrap=FS.FreezePolicy.for_arm("freeze_all"))
    assert float(slow.inner.S_G) == float(snap["slow.S_G"])
    assert float(slow.inner.mu_G) == float(snap["slow.mu_G"])
    assert np.array_equal(slow.inner.rE_fast, snap["slow.rE_fast"])


def test_output_only_sg_freeze_is_a_separate_non_primary_arm(reference):
    tf = reference["forks"][1]
    snap = reference["snaps"][tf]
    pol = FS.FreezePolicy.for_arm("freeze_sg_output_only")
    assert not pol.is_primary, "mixed-semantics freeze must not be usable as a primary arm"
    _, slow, _ = _continue(reference, snap, 300, slow_wrap=pol)
    assert float(slow.inner.S_G) == float(snap["slow.S_G"])
    assert float(slow.inner.mu_G) != float(snap["slow.mu_G"]), "sensor state must keep drifting"


def test_arm_freeze_table_is_exactly_the_spec_matrix():
    assert set(FS.ARM_FREEZE_TABLE) == {"dynamic_replay", "freeze_z", "freeze_zm", "freeze_zsg",
                                        "freeze_all", "dynamic_z_only"}
    t = FS.ARM_FREEZE_TABLE
    assert t["freeze_z"] == dict(freeze_z=True, freeze_m=False, freeze_sg_family=False)
    assert t["freeze_zm"] == dict(freeze_z=True, freeze_m=True, freeze_sg_family=False)
    assert t["freeze_zsg"] == dict(freeze_z=True, freeze_m=False, freeze_sg_family=True)
    assert t["freeze_all"] == dict(freeze_z=True, freeze_m=True, freeze_sg_family=True)
    assert t["dynamic_z_only"] == dict(freeze_z=False, freeze_m=True, freeze_sg_family=True)
    with pytest.raises(ValueError):
        FS.FreezePolicy.for_arm("freeze_everything_please")


# ---------------------------------------------------------------- serialization contract
def test_npz_roundtrip_and_fail_closed_rejections(reference, tmp_path):
    tf = reference["forks"][0]
    state = reference["snaps"][tf]
    path = str(tmp_path / "snap.npz")
    man = CK.save_state_npz(state, dict(config_sha="CFG", engine_sha="ENG", dt=DT, seed=SEED), path)
    back, man2 = CK.load_state_npz(path, expected_config_sha="CFG", expected_engine_sha="ENG",
                                   expected_dt=DT)
    assert man2["state_hash"] == man["state_hash"]
    for k in state:
        if k == "rng_state":
            assert CK._decode_rng(back[k]) == CK._decode_rng(state[k])
        else:
            assert np.array_equal(back[k], state[k]), k
    with pytest.raises(ValueError, match="config_sha"):
        CK.load_state_npz(path, expected_config_sha="OTHER")
    with pytest.raises(ValueError, match="engine_sha"):
        CK.load_state_npz(path, expected_engine_sha="OTHER")
    with pytest.raises(ValueError, match="dt"):
        CK.load_state_npz(path, expected_dt=0.05)


def test_npz_has_no_object_arrays(reference, tmp_path):
    path = str(tmp_path / "snap2.npz")
    CK.save_state_npz(reference["snaps"][reference["forks"][0]], dict(dt=DT), path)
    with np.load(path, allow_pickle=False) as z:   # raises if anything was pickled
        for k in z.files:
            assert z[k].dtype != np.dtype("O")


def test_resumed_state_can_be_reloaded_from_disk_and_still_matches(reference, tmp_path):
    tf = reference["forks"][2]
    n = 300
    path = str(tmp_path / "snap3.npz")
    CK.save_state_npz(reference["snaps"][tf], dict(dt=DT), path)
    loaded, _ = CK.load_state_npz(path)
    a, _, _ = _continue(reference, reference["snaps"][tf], n)
    b, _, _ = _continue(reference, loaded, n)
    assert np.array_equal(a["E_spk_bool"], b["E_spk_bool"])
