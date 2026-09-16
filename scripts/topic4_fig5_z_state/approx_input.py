#!/usr/bin/env python3
"""Exact expected external input (global OU + spatial OU) on the common future clock for W1/W2 and the two replays.

The engine draws one normal (global OU) and N Poisson counts per step from one RNG; the Poisson means depend only on
the external field, so stepping the same RNG with the same means reproduces the exact xi(t) and spatial field that
every native continuation on that stream received.  Outputs the conditional mean rate per coarse cell (per ms) at
0.1-ms resolution; the Poisson realisation itself stays inside the transfer variance of the approximation.
"""
import argparse
import numpy as np
from common import *  # noqa: F401,F403
from params import compute_nu_theta

APPROX = OUT / 'approx'
REPLAY_RUN = OUT / 'replay' / 'runs' / f'eta0.0005_s{MAIN_SEED}'


def cell_maps(s, grids):
    maps = {g: old.spatial_cell_index(s.positions_e, n_grid=g, sheet_l_mm=s.params.L) for g in grids}
    counts = {g: np.bincount(c, minlength=g * g).astype(float) for g, c in maps.items()}
    return maps, counts


def generate(state, start_ms, duration_ms, tag, grids=(20, 10), verify_fields=None):
    started = time.time()
    s, tr, frozen, identity = old.setup(MAIN_SEED); p = s.params; n = NE + NI
    rng = np.random.default_rng(); rng.bit_generator.state = state['rng_state']; xi = float(state['xi'])
    drive = old.make_external_drive(s, tr['spatial_ou'], MAIN_SEED); ckpt.restore_external_drive(state, drive)
    nu_sig = float(p.nu_ext_ratio * compute_nu_theta(p)[0])
    ou_a = np.exp(-p.dt / p.tau_n); sigma_xi = p.sigma_n * 1e-3 * np.sqrt(p.tau_n / 2.); ou_b = sigma_xi * np.sqrt(1 - ou_a * ou_a)
    maps, counts = cell_maps(s, grids)
    steps = ms_to_step(duration_ms)
    out = {g: np.empty((steps, g * g), np.float32) for g in grids}
    out_var = {g: np.empty((steps, g * g), np.float32) for g in grids}
    glob = np.empty(steps, np.float32); xis = np.empty(steps, np.float32)
    checks = []
    for k in range(steps):
        tm = start_ms + k * p.dt
        xi = ou_a * xi + ou_b * rng.standard_normal(); nu_now = max(nu_sig + xi, 0.)
        vec = np.full(n, nu_now); delta = np.asarray(drive.step(tm), float); vec[:NE] = np.maximum(vec[:NE] + delta, 0.)
        rng.poisson(vec * p.dt, size=n)
        for g, cells in maps.items():
            mean = np.bincount(cells, weights=vec[:NE], minlength=g * g) / counts[g]
            out[g][k] = mean; out_var[g][k] = np.maximum(np.bincount(cells, weights=vec[:NE] ** 2, minlength=g * g) / counts[g] - mean ** 2, 0.)
        glob[k] = nu_now; xis[k] = xi
        if verify_fields is not None and k in verify_fields:
            ref = verify_fields[k]
            err = float(np.max(np.abs(out[20][k] - ref['drive_mean']))); checks.append(dict(step=k, max_abs_cell_rate_error=err, xi_equal=bool(ref['xi'] == np.float32(xi))))
            assert err < 1e-6, (k, err)
    folder = APPROX / 'input'; folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / f'{tag}.npz', start_ms=start_ms, dt_ms=p.dt, steps=steps, nu_sig_per_ms=nu_sig,
             **{f'cell_mean_{g}': out[g] for g in grids}, **{f'cell_var_{g}': out_var[g] for g in grids},
             global_rate_per_ms=glob, xi=xis, **{f'count_e_{g}': counts[g] for g in grids})
    end_state = dict(rng_state=rng.bit_generator.state, xi=xi, drive_rng_state=drive._rng.bit_generator.state,
                     drive_cached=drive._cached.copy(), drive_field_state=drive._state.copy(), drive_next_step=drive._next_step, drive_last_step=drive._last_step)
    write(folder / f'{tag}.json', dict(tag=tag, start_ms=start_ms, duration_ms=duration_ms, steps=steps, seconds=time.time() - started,
          nu_sig_per_ms=nu_sig, verification_against_replay_record=checks, identity=identity,
          scope='Conditional mean of the external Poisson rate per coarse cell; same RNG consumption order as the engine.'))
    return end_state


def end_state_matches(end_state, native_state):
    return dict(rng_state_equal=end_state['rng_state'] == native_state['rng_state'], xi_equal=float(end_state['xi']) == float(native_state['xi']),
                drive_rng_equal=end_state['drive_rng_state'] == native_state['external_drive']['rng_state'],
                drive_cached_equal=bool(np.array_equal(end_state['drive_cached'], native_state['external_drive']['cached'])),
                drive_field_equal=bool(np.array_equal(end_state['drive_field_state'], native_state['external_drive']['field_state'])))


def main(which, duration_ms):
    anchor = ckpt.load(REPLAY_RUN / 'checkpoints' / f't{ANCHOR_MS}ms.npz')
    if which == 'W1':
        # Verify against the replay's own recorded drive (10.37 -> 12.5 s) at 1-ms samples.
        verify = {}
        for path in sorted((REPLAY_RUN / 'fields').glob('*.npz')):
            with np.load(path) as a:
                for j, st in enumerate(a['in_step']):
                    if st >= ms_to_step(ANCHOR_MS):
                        k = int(st) - ms_to_step(ANCHOR_MS)
                        verify[k] = dict(drive_mean=a['drive_mean'][j], xi=a['xi'][int(st) - int(a['start_step'])])
        end = generate(anchor, ANCHOR_MS, duration_ms, 'W1', verify_fields=verify)
        # End-state check against the replay end (12.5 s) when duration reaches it.
        if duration_ms >= REPLAY_END_MS - ANCHOR_MS:
            pass
        write(APPROX / 'input' / 'W1_end_state.json', dict(note='compare with a native W1 continuation endpoint after M1', xi=float(end['xi'])))
        save_pickle(APPROX / 'input' / 'W1_end_state.pkl', end)
    elif which == 'W2':
        state = replace_future_innovations(anchor, W2_SEED)
        end = generate(state, ANCHOR_MS, duration_ms, 'W2')
        save_pickle(APPROX / 'input' / 'W2_end_state.pkl', end)
    elif which in ('replay_s9108401', 'replay_s9108402'):
        seed = int(which.split('_s')[1]); run = OUT / 'replay' / 'runs' / f'eta0.0005_s{seed}'
        st = ckpt.load(run / 'checkpoints' / 't8000ms.npz') if (run / 'checkpoints' / 't8000ms.npz').exists() else None
        assert st is not None, 'replay checkpoint at 8000 ms required'
        verify = {}
        for path in sorted((run / 'fields').glob('*.npz')):
            with np.load(path) as a:
                for j, s_ in enumerate(a['in_step']):
                    if s_ >= ms_to_step(8000):
                        verify[int(s_) - ms_to_step(8000)] = dict(drive_mean=a['drive_mean'][j], xi=a['xi'][int(s_) - int(a['start_step'])])
        end = generate(st, 8000, duration_ms, which, verify_fields=verify)
        save_pickle(APPROX / 'input' / f'{which}_end_state.pkl', end)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('which', choices=['W1', 'W2', 'replay_s9108401', 'replay_s9108402'])
    ap.add_argument('--duration-ms', type=float, default=20000.); a = ap.parse_args()
    main(a.which, a.duration_ms)
