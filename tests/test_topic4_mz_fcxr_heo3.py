"""FCXR-HEO3 TDD — joint sliding-window gate + source-space audit (HEO2.1 review P1-b/P1-c closeout).

The point of these instruments is to REJECT things HEO2.1's whole-window summaries would have accepted:
an on-off burst train must not read as 'phase-dispersed', and one loud core seen by every contact must
not read as whole-tissue recruitment. Each test encodes one of those bad-data regressions.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import src.topic4_mz_fcxr_heo3 as H3  # noqa: E402

FS = 1000.0


def _sines(n, f, C=15, phases=None, fs=FS):
    t = np.arange(n) / fs
    ph = np.zeros(C) if phases is None else np.asarray(phases, float)
    return np.stack([np.sin(2 * np.pi * f * t + ph[c]) for c in range(C)], axis=1)


# ============================== per-window spectral field ==============================
def test_band_power_windows_1s_default_shape_and_band_placement():
    x = _sines(4000, 16.0)
    bp, t = H3.band_power_windows(x, FS)                       # defaults: 1s window / 100ms hop
    assert bp.shape[1:] == (15, 6)
    assert bp.shape[0] > 25 and np.all(np.diff(t) > 0)
    # a 16 Hz tone belongs in 13-30 (index 3), NOT in 1-4 (0) or 8-13 (2)
    assert bp[:, 0, 3].mean() > 50 * bp[:, 0, 0].mean()
    assert bp[:, 0, 3].mean() > 50 * bp[:, 0, 2].mean()


def test_short_window_leaks_and_empties_low_bands_so_gate_must_not_use_250ms():
    """REGRESSION for the H3.0 instrument bug: at 250 ms the spectrogram has no bin below 4 Hz (1-4
    band empty) and the wide main lobe leaks a 16 Hz peak into 8-13, faking 'broadband'. 1 s must not."""
    x = _sines(4000, 16.0)
    bp250, _ = H3.band_power_windows(x, FS, win_ms=250.0, hop_ms=50.0)
    bp1s, _ = H3.band_power_windows(x, FS, win_ms=1000.0, hop_ms=100.0)
    assert bp250[:, 0, 0].mean() == 0.0                        # 1-4 Hz unmeasurable at 250 ms
    assert bp1s[:, 0, 0].mean() > 0.0                          # measurable at 1 s
    leak250 = bp250[:, 0, 2].mean() / bp250[:, 0, 3].mean()    # 8-13 relative to the true 13-30 peak
    leak1s = bp1s[:, 0, 2].mean() / bp1s[:, 0, 3].mean()
    assert leak250 > 20 * leak1s                               # 250 ms leaks far more into 8-13


# ============================== phase order parameter ==============================
def test_order_parameter_high_for_in_phase_low_for_scattered():
    n = 4000
    sync = _sines(n, 16.0)                                            # all contacts in phase
    rng = np.random.default_rng(0)
    scat = _sines(n, 16.0, phases=rng.uniform(0, 2 * np.pi, 15))      # phases scattered
    R_sync = H3.phase_order_parameter(sync, FS)[500:-500].mean()
    R_scat = H3.phase_order_parameter(scat, FS)[500:-500].mean()
    assert R_sync > 0.95
    assert R_scat < 0.6
    assert R_sync > R_scat


def test_order_parameter_traveling_wave_looks_dispersed_but_is_organized():
    """REGRESSION for the H3.0 instrument bug: the FCXR 16 Hz reference is a travelling wave (phase
    span ~186°). Instantaneous alignment R reads LOW (looks 'dispersed') even though the wave is
    perfectly organized — so R must NOT be the gate's desync criterion; pairwise PLV must be, and it
    must read HIGH here."""
    n = 4000
    ramp = np.linspace(0.0, np.pi, 15)
    x = _sines(n, 16.0, phases=ramp)
    R = H3.phase_order_parameter(x, FS)[500:-500].mean()
    assert R < 0.8                                             # alignment is low ...
    phi = H3.band_phase(x, FS)
    tcen = np.arange(1.0, 3.0, 0.25)
    plv = H3.pairwise_plv_windows(phi, tcen, dt_ms=1.0, win_ms=1000.0)
    assert np.nanmean(plv) > 0.95                              # ... but the ORGANIZATION is intact


def test_pairwise_plv_drops_only_when_phase_relationships_become_inconsistent():
    n, fs = 8000, FS
    t = np.arange(n) / fs
    rng = np.random.default_rng(1)
    # organized: fixed lags -> high PLV. drifting: each contact at a slightly different frequency.
    fixed = np.stack([np.sin(2 * np.pi * 16.0 * t + p) for p in np.linspace(0, np.pi, 15)], axis=1)
    drift = np.stack([np.sin(2 * np.pi * f * t + p) for f, p in
                      zip(rng.uniform(13.0, 25.0, 15), rng.uniform(0, 2 * np.pi, 15))], axis=1)
    tcen = np.arange(2.0, 6.0, 0.5)
    plv_fixed = np.nanmean(H3.pairwise_plv_windows(H3.band_phase(fixed, fs), tcen, 1.0, 1000.0))
    plv_drift = np.nanmean(H3.pairwise_plv_windows(H3.band_phase(drift, fs), tcen, 1.0, 1000.0))
    assert plv_fixed > 0.95
    assert plv_drift < 0.6
    assert plv_drift < plv_fixed


def test_pairwise_plv_active_gate_blocks_silence_faking_desync():
    """An on-off burst train must not read as desynchronized just because of its silent gaps."""
    n, fs = 8000, FS
    t = np.arange(n) / fs
    x = np.stack([np.sin(2 * np.pi * 16.0 * t + p) for p in np.linspace(0, np.pi, 15)], axis=1)
    env = ((np.arange(n) // 400) % 2).astype(float)            # 400ms on / 400ms off
    burst = x * env[:, None] + 0.001 * np.random.default_rng(0).standard_normal((n, 15))
    phi = H3.band_phase(burst, fs)
    tcen = np.arange(2.0, 6.0, 0.5)
    plv_all = np.nanmean(H3.pairwise_plv_windows(phi, tcen, 1.0, 1000.0))
    plv_act = np.nanmean(H3.pairwise_plv_windows(phi, tcen, 1.0, 1000.0, active=env > 0.5))
    assert plv_act > plv_all                                   # silence dilutes -> active gate restores
    assert plv_act > 0.9                                       # within bursts it is still organized


# ============================== the joint gate (bad-data regressions) ==============================
def _gate(bp, ref, rate, plv, **kw):
    return H3.joint_target_windows(bp, ref, rate, plv, fs_win=20.0, **kw)


def test_joint_gate_requires_all_four_simultaneously():
    nw, C, B = 50, 15, 6
    ref = np.ones((C, B))
    hot = np.full((nw, C, B), 10.0)                    # all bands 10x baseline -> +10 dB
    rate = np.full(nw, 100.0); plv = np.full(nw, 0.3)
    assert _gate(hot, ref, rate, plv)["frac_target"] == 1.0
    # drop each criterion in turn -> target must fail
    assert _gate(hot, ref, np.full(nw, 5.0), plv)["frac_target"] == 0.0        # low energy
    assert _gate(hot, ref, rate, np.full(nw, 0.99))["frac_target"] == 0.0      # still phase-locked
    assert _gate(hot, ref, rate, np.full(nw, np.nan))["frac_target"] == 0.0    # silent -> not desync
    narrow = np.ones((nw, C, B)); narrow[:, :, 3] = 10.0                       # only beta up
    assert _gate(narrow, ref, rate, plv)["frac_target"] == 0.0                 # not broadband
    few = np.ones((nw, C, B)); few[:, :5, :] = 10.0                            # only 5 contacts up
    assert _gate(few, ref, rate, plv)["frac_target"] == 0.0                    # not recruited


def test_joint_gate_longest_run_measures_persistence_not_total():
    """Two regimes with the SAME total target fraction but different persistence must score
    differently — this is the whole point of HEO3 (sustain, not merely reach)."""
    nw, C, B = 40, 15, 6
    ref = np.ones((C, B)); hot = np.full((C, B), 10.0)
    def _mk(mask):
        bp = np.ones((nw, C, B)); bp[mask] = hot
        rate = np.where(mask, 100.0, 1.0); plv = np.where(mask, 0.3, 0.99)
        return _gate(bp, ref, rate, plv)
    flicker = np.zeros(nw, bool); flicker[::2] = True          # 20 scattered target windows
    block = np.zeros(nw, bool); block[:20] = True              # 20 consecutive target windows
    a, b = _mk(flicker), _mk(block)
    assert abs(a["frac_target"] - b["frac_target"]) < 1e-9     # same total
    assert a["longest_run_ms"] < b["longest_run_ms"] / 5       # very different persistence
    assert b["longest_run_ms"] == 1000.0                       # 20 windows @ 20/s = 1000 ms


def test_joint_gate_reports_which_criterion_blocks():
    nw, C, B = 30, 15, 6
    ref = np.ones((C, B)); narrow = np.ones((nw, C, B)); narrow[:, :, 3] = 10.0
    out = _gate(narrow, ref, np.full(nw, 100.0), np.full(nw, 0.3))
    assert out["frac_by_criterion"]["recruited"] == 1.0        # beta alone still counts as recruited
    assert out["frac_by_criterion"]["broadband"] == 0.0        # but not as broadband
    assert out["frac_by_criterion"]["high_energy"] == 1.0
    assert out["frac_by_criterion"]["desynchronized"] == 1.0


# ============================== source-space audit (bad-data regressions) ==============================
def _grid(n=20):
    xs, ys = np.meshgrid(np.linspace(0, 19, n), np.linspace(0, 19, n))
    return np.stack([xs.ravel(), ys.ravel()], axis=1)


def test_build_regions_partitions_every_cell_exactly_once():
    pos = _grid()
    reg = H3.build_regions(pos, [2.0, 10.0], [17.0, 10.0], core_r=3.0)
    masks = [reg[k] for k in ("core_source", "core_sink", "axis_corridor", "off_axis")]
    stacked = np.stack(masks)
    assert np.all(stacked.sum(axis=0) == 1)                    # a partition: exactly one region per cell
    assert reg["core_source"].sum() > 0 and reg["core_sink"].sum() > 0
    assert reg["axis_corridor"].sum() > 0 and reg["off_axis"].sum() > 0
    assert 0.0 <= reg["axis_coord"].min() and reg["axis_coord"].max() <= 1.0


def test_participation_ratio_separates_one_loud_core_from_uniform():
    """THE P1-b regression: 15/15 sensor coverage can hide 'one core is loud'. PR must expose it.
    The load-bearing case is a loud core PLUS weak activity everywhere — there the active FRACTION is
    1.0 (so any active-fraction-style metric says 'whole tissue recruited') while the activity is in
    fact concentrated in 5% of cells. PR must report the concentration, not the active count."""
    N = 400
    uniform = np.ones(N)
    one_core = np.zeros(N); one_core[:20] = 1.0                # 5% of cells carry everything
    loud_core_weak_rest = np.full(N, 0.01); loud_core_weak_rest[:20] = 1.0   # EVERY cell is active
    assert H3.participation_ratio(uniform) == 1.0
    assert abs(H3.participation_ratio(one_core) - 0.05) < 1e-9
    assert H3.participation_ratio(one_core) < H3.participation_ratio(uniform)
    assert float((loud_core_weak_rest > 0).mean()) == 1.0      # active fraction is blind here ...
    assert H3.participation_ratio(loud_core_weak_rest) < 0.15  # ... but PR exposes the concentration
    assert H3.participation_ratio(np.zeros(N)) == 0.0          # silence -> 0, not nan


def test_source_space_audit_detects_core_only_vs_whole_field():
    pos = _grid(20)                                            # NE=400
    reg = H3.build_regions(pos, [2.0, 10.0], [17.0, 10.0], core_r=3.0)
    nsteps = 400                                               # dt=1ms -> 400ms
    core_only = np.zeros((nsteps, len(pos)), bool)
    core_only[::4, reg["core_source"]] = True                  # only the source core fires
    whole = np.zeros((nsteps, len(pos)), bool)
    whole[::4, :] = True                                       # everything fires
    a = H3.source_space_audit(core_only, pos, reg, dt=1.0, win_ms=200.0, hop_ms=100.0)[0]
    b = H3.source_space_audit(whole, pos, reg, dt=1.0, win_ms=200.0, hop_ms=100.0)[0]
    assert a["participation_ratio"] < 0.2                      # one core -> low PR ...
    assert b["participation_ratio"] > 0.95                     # ... whole field -> ~1
    assert a["rate_off_axis"] == 0.0 and a["rate_core_source"] > 0
    assert b["rate_off_axis"] > 0
    assert a["centroid_axis_coord"] < 0.2                      # centroid sits at the source end


def test_pairwise_plv_is_undefined_not_low_when_window_is_silent():
    """REGRESSION: a silent window must yield nan (undefined), never a low PLV that the gate would
    read as 'desynchronized'. Silence is absence of evidence, not evidence of desync."""
    n, fs = 4000, FS
    noise = 0.001 * np.random.default_rng(3).standard_normal((n, 15))
    phi = H3.band_phase(noise, fs)
    active = np.zeros(n, bool)                                  # nothing is active
    plv = H3.pairwise_plv_windows(phi, np.arange(1.0, 3.0, 0.5), 1.0, 1000.0, active=active)
    assert np.all(np.isnan(plv))
    # and the gate must treat nan as NOT desynchronized
    nw, C, B = len(plv), 15, 6
    out = H3.joint_target_windows(np.full((nw, C, B), 10.0), np.ones((C, B)),
                                  np.full(nw, 100.0), plv, fs_win=10.0)
    assert out["frac_by_criterion"]["desynchronized"] == 0.0


# ============== HEO3 H3.1 engine: per-cell recovery time / strength + mean-field control ==============
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", "src", "snn_engine"))
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402


def _mk_m(NE=6, N=8, **kw):
    base = dict(use_m=True, eta_m=0.5, tau_adp=250.0)
    base.update(kw)
    return MZSlowVars(N, 18.0, MZSlowVarsConfig(**base), NE=NE, core_mask_E=np.zeros(NE, bool))


def test_tau_adp_E_gives_per_cell_recovery_times():
    """Patchy recovery time: cells with a shorter tau must decay faster, in the SAME run."""
    NE = 6
    tau = np.array([100.0, 100.0, 100.0, 800.0, 800.0, 800.0])
    mz = _mk_m(NE=NE, tau_adp_E=tau)
    mz.m[:NE] = 10.0                                           # same starting adaptation everywhere
    for _ in range(200):                                       # 20 ms at dt=0.1, no spikes
        mz.step(np.zeros(8, bool), None, 0.1)
    fast, slow = mz.m[:3], mz.m[3:6]
    assert np.all(fast < slow)                                 # short tau -> decayed further
    assert abs(fast[0] - 10.0 * np.exp(-20.0 / 100.0)) < 0.05  # matches the analytic decay
    assert abs(slow[0] - 10.0 * np.exp(-20.0 / 800.0)) < 0.05


def test_eta_m_E_scales_adaptation_current_per_cell():
    NE, N = 4, 6
    eta = np.array([0.0, 0.25, 0.5, 1.0])
    mz = MZSlowVars(N, 18.0, MZSlowVarsConfig(use_m=True, eta_m=0.5, tau_adp=250.0, eta_m_E=eta),
                    NE=NE, core_mask_E=np.zeros(NE, bool))
    mz.m[:NE] = 2.0
    out = mz.apply_currents(np.zeros(N), np.zeros(N), None)
    assert np.allclose(out[:NE], -eta * 2.0)                   # per-cell eta drives per-cell current


def test_load_compensated_patch_holds_steady_state_K_fixed():
    """THE H3.1 design invariant: eta_i = eta0*tau0/tau_i makes each cell's steady-state adaptation
    current independent of its recovery time, so a patchy tau field varies TIMING, not LOAD."""
    tau0, eta0 = 250.0, 0.4
    tau = np.array([125.0, 250.0, 500.0])                      # the H3.1 range (4x spread)
    eta = eta0 * tau0 / tau
    assert np.allclose(eta * tau, eta0 * tau0)                 # steady-state m ~ r*tau -> eta*m ~ eta*tau
    NE, N, dt = 3, 5, 0.1
    mz = MZSlowVars(N, 18.0, MZSlowVarsConfig(use_m=True, eta_m=eta0, tau_adp=tau0,
                                              tau_adp_E=tau, eta_m_E=eta),
                    NE=NE, core_mask_E=np.zeros(NE, bool))
    spk = np.zeros(N, bool); spk[:NE] = True
    for i in range(30000):                                     # 3000 ms = 6x the longest tau -> equilibrated
        mz.step(spk if i % 10 == 0 else np.zeros(N, bool), None, dt)
    cur = eta * mz.m[:NE]                                      # steady-state adaptation current per cell
    assert cur.max() / cur.min() < 1.02                        # equal load despite the 4x tau spread
    assert np.allclose(cur, eta0 * tau0, rtol=0.02)            # and equal to the uniform-arm load


def test_m_mean_field_removes_inter_cell_differences():
    """Control arm: the population-mean m applied to every cell -> pure temporal modulation."""
    NE, N = 6, 8
    mz = _mk_m(NE=NE, m_mean_field=True)
    spk = np.zeros(N, bool); spk[0] = spk[1] = True            # only 2 cells spike
    mz.step(spk, None, 0.1)
    assert np.allclose(mz.m[:NE], mz.m[0])                     # every E cell carries the same m
    assert mz.m[0] > 0                                         # and it is the population mean (2/6)
    assert abs(mz.m[0] - 2.0 / NE) < 1e-9
    mz_off = _mk_m(NE=NE)                                      # without the control: cells differ
    mz_off.step(spk, None, 0.1)
    assert not np.allclose(mz_off.m[:NE], mz_off.m[0])


def test_per_cell_adaptation_fields_validate():
    import pytest
    with pytest.raises(ValueError):
        _mk_m(tau_adp_E=np.array([100.0, -1.0, 100.0, 100.0, 100.0, 100.0]))   # tau must be > 0
    with pytest.raises(ValueError):
        _mk_m(eta_m_E=np.array([0.1, -0.2, 0.1, 0.1, 0.1, 0.1]))               # eta must be >= 0
    with pytest.raises(ValueError):
        MZSlowVars(8, 18.0, MZSlowVarsConfig(use_m=False, tau_adp_E=np.ones(6)),
                   NE=6, core_mask_E=np.zeros(6, bool))                        # requires use_m


# ============== HEO3 H3.1: patch field + alternation metric ==============
def test_patch_field_stripes_alternate_and_preserve_load():
    pos = _grid(40)
    src, snk = np.array([3.5, 8.5]), np.array([16.5, 8.5])
    tau, eta, pid = H3.build_patch_field(pos, src, snk, patch_w=4.35, tau_fast=125.0,
                                         tau_slow=500.0, tau0=250.0, eta0=0.4)
    assert set(np.unique(tau)) == {125.0, 500.0}
    assert np.allclose(eta * tau, 0.4 * 250.0)                 # load invariant holds cell by cell
    # stripes run perpendicular to the axis: cells at the same axis coord share a patch id
    proj = ((pos - src) @ (snk - src)) / np.linalg.norm(snk - src)
    for lo in (0.0, 5.0, 9.0):
        band = (proj >= lo) & (proj < lo + 4.0)                # inside one stripe width
        if band.sum() > 5:
            assert len(np.unique(pid[band])) == 1
    assert 0 < pid.mean() < 1                                  # both patch types present


def test_patch_shuffle_preserves_histogram_but_destroys_organization():
    pos = _grid(40)
    src, snk = np.array([3.5, 8.5]), np.array([16.5, 8.5])
    kw = dict(patch_w=4.35, tau_fast=125.0, tau_slow=500.0, tau0=250.0, eta0=0.4)
    tau, eta, _ = H3.build_patch_field(pos, src, snk, **kw)
    tau_s, eta_s, _ = H3.build_patch_field(pos, src, snk, shuffle_seed=7, **kw)
    assert np.allclose(np.sort(tau), np.sort(tau_s))            # identical histogram/mean/variance
    assert abs(tau.mean() - tau_s.mean()) < 1e-12 and abs(tau.std() - tau_s.std()) < 1e-12
    assert np.allclose(eta_s * tau_s, 0.4 * 250.0)              # load invariant survives the shuffle
    # organization: neighbours share tau in the striped field, far less so once shuffled
    proj = ((pos - src) @ (snk - src)) / np.linalg.norm(snk - src)
    order = np.argsort(proj)
    same = np.mean(tau[order][:-1] == tau[order][1:])
    same_s = np.mean(tau_s[order][:-1] == tau_s[order][1:])
    assert same > 0.9 and same_s < 0.7


def test_region_alternation_is_negative_only_when_regions_take_turns():
    def rows(a, b):
        return [{"rate_core_source": x, "rate_core_sink": y} for x, y in zip(a, b)]
    n = 40
    t = np.arange(n)
    alt_a = 50 + 40 * np.sin(2 * np.pi * t / 10)               # antiphase: they take turns
    alt_b = 50 - 40 * np.sin(2 * np.pi * t / 10)
    assert H3.region_alternation(rows(alt_a, alt_b)) < -0.9
    # REGRESSION for the shares tautology: a purely COMMON-DRIVEN pair (in phase, different amplitude,
    # so the shares are NOT constant) must score POSITIVE. Computing this on shares a/(a+b), b/(a+b)
    # returns -1 here — shares sum to 1 and are anticorrelated by construction — which would have
    # reported "perfect alternation" for every arm.
    tog_a = 50 + 40 * np.sin(2 * np.pi * t / 10)
    tog_b = 0.8 * tog_a + 2.0
    assert H3.region_alternation(rows(tog_a, tog_b)) > 0.9     # common drive -> POSITIVE, not -1
    assert np.isnan(H3.region_alternation(rows(np.zeros(n), np.zeros(n))))   # silence -> undefined
    assert np.isnan(H3.region_alternation(rows(np.full(n, 5.0), np.full(n, 5.0))))  # constant -> undefined


def test_stripe_phase_must_centre_on_cores_to_separate_the_two_regions():
    """REGRESSION for the H3.1 P0: with phase_shift=0 and patch_w=D/3 both core centres fall on stripe
    BOUNDARIES, so each core is ~50/50 fast/slow and the two regions never get different recovery
    times — the manipulation silently does not test the hypothesis. phase_shift=w/2 fixes it."""
    n = 60
    src, snk = np.array([3.5, 8.5]), np.array([16.5, 8.5])
    D = float(np.linalg.norm(snk - src)); w = D / 3
    # cells packed inside each core (radius 1.5) plus a corridor line
    core_s = np.stack([np.linspace(src[0] - 1.4, src[0] + 1.4, n), np.full(n, 8.5)], axis=1)
    core_k = np.stack([np.linspace(snk[0] - 1.4, snk[0] + 1.4, n), np.full(n, 8.5)], axis=1)
    pos = np.vstack([core_s, core_k])
    kw = dict(patch_w=w, tau_fast=125.0, tau_slow=500.0, tau0=250.0, eta0=0.4)
    tau0_, _, _ = H3.build_patch_field(pos, src, snk, **kw)                      # boundary-aligned (bug)
    tau_c, _, _ = H3.build_patch_field(pos, src, snk, phase_shift=w / 2, **kw)   # centred (correct)
    slow_src0, slow_snk0 = (tau0_[:n] == 500.0).mean(), (tau0_[n:] == 500.0).mean()
    slow_srcC, slow_snkC = (tau_c[:n] == 500.0).mean(), (tau_c[n:] == 500.0).mean()
    assert 0.3 < slow_src0 < 0.7 and 0.3 < slow_snk0 < 0.7      # bug: both cores ~half-and-half
    assert slow_srcC == 0.0 and slow_snkC == 1.0                # fixed: source all fast, sink all slow
    assert abs(np.mean(tau_c == 125.0) - np.mean(tau0_ == 125.0)) < 0.35  # same marginal mix, different placement
