"""Shared constants and helpers for the current-Fig.5-network Z state milestone.

Design contract: docs/archive/topic4/fig5_current_network_z_state_milestone_design_2026-09-15.md
Reference network/runs: results/topic4_sef_hfo/fig5_preentry_event_audit_20260914 (eta_M=0.0005, tau_M=1 s).

Nothing here changes the native equations; the producer chain
(run_topic4_m_parameter_modes -> run_topic4_weak_fast_recurrence -> src.topic4_raster_protocol_engine)
is imported unchanged and only wrapped by observers.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['TOPIC4_MANUAL_ARM'] = 'manual_hard'
import sys
import json
import hashlib
import pickle
import time
import copy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import run_topic4_m_parameter_modes as core          # noqa: E402  (installs SOURCE/src paths)
old = core.old                                        # run_topic4_weak_fast_recurrence
import checkpoint as ckpt                             # noqa: E402  worktree engine checkpoint module

OUT = ROOT / 'results/topic4_sef_hfo/fig5_current_network_z_state_v1'
REF = ROOT / 'results/topic4_sef_hfo/fig5_preentry_event_audit_20260914'
SUBSTRATE = ROOT / 'results/topic4_sef_hfo/historical_manual_hard_native_z_v1'
SEEDS = [9108401, 9108402]
DEVICE = {9108401: 0, 9108402: 1}
MAIN_SEED = 9108401
Z_TIMES_MS = [8000, 9000, 9300, 9420, 9870, 10370]
HISTORY_MS = [8000, 10370]
ANCHOR_MS = 10370
REPLAY_END_MS = 12500
DENSE_WINDOW_MS = (8800, 10400)
W2_SEED = 9108501                    # reserved by the design; verified unused before dispatch
W2_DRIVE_SEED_OFFSET = 500000        # same convention as the 2026-09-09 boundary audit
CONTINUATION_MS = 10000
EXTENSION_MS = 10000
FIGURE_TIMES_S = {1: 1.235, 2: 4.025, 3: 9.420, 4: 10.370}
HIGH_ONSET_S = 9.870
HIGH_CONFIRM_S = 10.070
PY = '/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python'
LD = '/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib'
NE, NI = 32000, 8000
DT_MS = 0.1

read = core.read
write = core.write
save_pickle = core.save_pickle
load_pickle = core.load_pickle
sha = core.sha


def now():
    return time.time()


def ms_to_step(ms):
    step = int(round(ms / DT_MS))
    assert abs(step * DT_MS - ms) < 1e-9, ms
    return step


def reference_protocol():
    return read(REF / 'protocol.json')


def reference_job(seed):
    return read(REF / 'jobs' / f'eta0.0005_s{seed}.json')


def check_reference_sources():
    """The producer chain must be byte-identical to what produced the reference runs."""
    p = reference_protocol()
    changed = [path for path, digest in p['source_hashes'].items() if sha(path) != digest]
    for key in ('producer', 'wrapper'):
        if sha(p[key]) != p[key + '_sha256']:
            changed.append(p[key])
    if changed:
        raise RuntimeError('Reference producer sources changed: ' + ', '.join(changed))
    return p


def first_only_tracker(tr, rate, sec, rescue=True):
    """Verbatim copy of run_topic4_fig5_half_m_first_entry.first_only (the reference tracker)."""
    tr['rates'].append(float(rate)); tr['rates'] = tr['rates'][-200:]
    tr['high_bins'] = tr['high_bins'] + 1 if rate >= 200 else 0
    if tr['high_bins'] >= 20 and not tr['entries']:
        tr['entries'].append(dict(onset_s=sec - .2, confirmation_s=sec))
        tr['phase'] = 'HIGH'; tr['last_entry_s'] = sec; tr['stop_s'] = sec + 2.


def observe_only_tracker(tr, rate, sec, rescue=True):
    """Record high-rate entries/recoveries but never shorten the horizon (continuations run their full length)."""
    tr['rates'].append(float(rate)); tr['rates'] = tr['rates'][-200:]
    tr['high_bins'] = tr['high_bins'] + 1 if rate >= 200 else 0
    if tr['high_bins'] >= 20 and tr['phase'] in ('PRE_ENTRY', 'RECOVERED'):
        tr['entries'].append(dict(onset_s=sec - .2, confirmation_s=sec))
        tr['phase'] = 'HIGH'; tr['last_entry_s'] = sec
    if tr['phase'] == 'HIGH' and len(tr['rates']) == 200 and sec >= tr['last_entry_s'] + 2:
        x = np.asarray(tr['rates']).reshape(2, 100)
        if bool(np.all(x.mean(1) < 50) and np.all((x < 5).mean(1) >= .2)):
            tr['recoveries'].append(dict(start_s=sec - 2, confirmation_s=sec, mechanism='NATIVE'))
            tr['phase'] = 'RECOVERED'


def state_to_arrays(state):
    """Flatten an engine checkpoint dict (as produced by checkpoint.capture) for bitwise comparison."""
    out = {}
    for key, value in state.items():
        if isinstance(value, dict):
            for sub, sv in value.items():
                out[f'{key}.{sub}'] = sv
        else:
            out[key] = value
    return out


def compare_states(a, b):
    """Return a list of keys that differ between two engine checkpoint dicts (bitwise for arrays)."""
    fa, fb = state_to_arrays(a), state_to_arrays(b)
    diffs = []
    for key in sorted(set(fa) | set(fb)):
        if key not in fa or key not in fb:
            diffs.append(key); continue
        x, y = fa[key], fb[key]
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and x.shape == y.shape
                    and x.dtype == y.dtype and np.array_equal(x, y)):
                diffs.append(key)
        elif x != y:
            diffs.append(key)
    return diffs


def pickle_engine_to_state(engine):
    """The reference pickle stores checkpoint.capture output directly; return it as-is (dict)."""
    assert engine['schema'] == ckpt.CHECKPOINT_SCHEMA
    return engine


def resolved_module_identity():
    """Actual module files loaded by the producer chain (worktree copies shadow ROOT/src for some)."""
    import src.topic4_raster_protocol_engine as eng
    import src.snn_engine.mz_slow_vars as mz
    import src.topic4_cuda_ordered_scatter as cos_
    import src.topic4_serial_spike_scatter as ser
    import src.topic4_zm_ictal_transition as zmt
    import src.topic4_spatial_ou_drive as oud
    import params, model, lfp
    mods = [core, old, eng, mz, cos_, ser, ckpt, params, model, lfp, zmt, oud]
    return {m.__name__: dict(file=m.__file__, sha256=sha(m.__file__)) for m in mods}


class FrozenSlow:
    """Instance-level patch: freeze Z and/or M *state updates* while keeping their current effects.

    apply_currents is untouched, so I_net = I_E - z*I_I - eta_m*m is still applied with the frozen
    values every step; only the state update in step() is suppressed for the frozen variable(s).
    """
    def __init__(self, slow, freeze_z, freeze_m):
        self.slow = slow; self.freeze_z = bool(freeze_z); self.freeze_m = bool(freeze_m)
        self.original_step = slow.step
        slow.step = self.step

    def step(self, spk, labels, dt):
        s = self.slow
        z_keep = s.z.copy() if self.freeze_z else None
        m_keep = s.m.copy() if self.freeze_m else None
        self.original_step(spk, labels, dt)
        if z_keep is not None:
            s.z[:] = z_keep
        if m_keep is not None:
            s.m[:] = m_keep


def frozen_application_check():
    """Actual-application check required by the design (section 3): frozen state still acts on current."""
    cfg = old.MZSlowVarsConfig(use_z=True, use_m=True, tau_z=5000., I_th_EI=old.THRESHOLD, tau_adp=1000., eta_m=.0005)
    rng = np.random.default_rng(7)
    live = old.ReleaseZ(8, 18., cfg, NE=6); fz = old.ReleaseZ(8, 18., cfg, NE=6); fzm = old.ReleaseZ(8, 18., cfg, NE=6)
    z0 = rng.uniform(.5, .9, 8); z0[6:] = 1.; m0 = rng.uniform(0, 30, 8); m0[6:] = 0.
    for s in (live, fz, fzm):
        s.z[:] = z0; s.m[:] = m0
    FrozenSlow(fz, True, False); FrozenSlow(fzm, True, True)
    for k in range(3000):
        ie, ii = rng.uniform(0, 300, (2, 8)); sp = rng.random(8) < .1
        a, b, c = live.apply_currents(ie, ii), fz.apply_currents(ie, ii), fzm.apply_currents(ie, ii)
        assert np.array_equal(b, ie - fz.z * ii - cfg.eta_m * fz.m)
        assert np.array_equal(c, ie - z0 * ii - cfg.eta_m * m0)
        live.step(sp, None, .1); fz.step(sp, None, .1); fzm.step(sp, None, .1)
        assert np.array_equal(fz.z, z0) and np.array_equal(fzm.z, z0) and np.array_equal(fzm.m, m0)
        assert np.array_equal(fz.m, live.m)
    assert not np.array_equal(live.z, z0) and np.max(live.m) > 0
    return dict(status='PASS', steps=3000, frozen_Z_still_multiplies_GABA=True, frozen_M_still_subtracts_current=True,
                dynamic_M_identical_to_live_under_frozen_Z=True, use_z_use_m_left_true=True)


def transplant_state(physical, anchor, z_field=None, m_field=None):
    """History state + anchor clock/input/RNG (+ replacement Z and/or M fields).

    Physical fast state (V, ref, filters, delay rings, M, I_I_last) comes from `physical`;
    the delay rings are rolled so that the remaining-delay order is preserved on the anchor clock.
    Time, OU state, RNG streams and the external spatial OU drive come from `anchor`.
    """
    state = copy.deepcopy(physical)
    old_step = int(physical['step']); new_step = int(anchor['step'])
    for key in ('ring_sE', 'ring_sI'):
        a = physical[key]; n = len(a)
        state[key] = np.roll(a, (new_step - old_step) % n, axis=0)
        for delay in range(n):
            assert np.array_equal(a[(old_step + delay) % n], state[key][(new_step + delay) % n])
    for key in ('step', 'absolute_time_ms', 'xi', 'rng_state', 'external_drive', 'ras_keep', 'es_ema', 'es_run'):
        state[key] = copy.deepcopy(anchor[key])
    state['slow']['step_index'] = int(anchor['slow']['step_index'])
    if z_field is not None:
        assert z_field.shape == state['slow']['z'].shape
        state['slow']['z'] = np.array(z_field, copy=True)
    if m_field is not None:
        assert m_field.shape == state['slow']['m'].shape
        state['slow']['m'] = np.array(m_field, copy=True)
    for key in ('V', 'ref', 's_E', 'I_E', 's_I', 'I_I'):
        assert np.array_equal(state[key], physical[key])
    return state


def replace_future_innovations(state, seed):
    """W2: keep the anchor's xi and spatial-OU field/cache/clock; replace only the RNG streams."""
    state = copy.deepcopy(state)
    state['rng_state'] = np.random.default_rng(int(seed)).bit_generator.state
    state['external_drive']['rng_state'] = np.random.default_rng(int(seed) + W2_DRIVE_SEED_OFFSET).bit_generator.state
    return state
