#!/usr/bin/env python3
"""Run the reduced model under the native protocols: frozen-Z continuations (W1/W2), extensions, fixed Z+M,
real-Z(t) path replays, deterministic (OU-off) trajectories and grid sensitivity.

Initial states are projected from the SAME native transplanted checkpoints (design 6.3): no per-condition search.
"""
import argparse
import subprocess
import numpy as np
from common import *  # noqa: F401,F403
from approx_system import ReducedModel, APPROX
import native_continue as nc
import readouts as R


def variant_dir(version):
    return APPROX / version


def load_input(tag):
    return np.load(APPROX / 'input' / f'{tag}.npz')


def recent_native_rates(folder, start_step, model):
    """Last 10 ms of native E spikes per cell (0.1 ms) and I regional rates, from the chunk holding start_step."""
    e_counts = []; i_rate_cell = None
    for path in sorted((Path(folder) / 'chunks').glob('*.npz')):
        if '.tmp.' in path.name:
            continue
        with np.load(path) as a:
            s, e = int(a['start_step']), int(a['end_step'])
            if s < start_step <= e:
                k = start_step - s
                f = a['field_0p1ms'][max(0, k - 100):k].astype(float)              # (<=100, 400) 20x20 native cells
                if model.grid == 20:
                    e_counts = f
                else:
                    cells20 = np.load(APPROX / 'coarse_20/geometry.npz')['cell_e']; cells_g = model.cell_e
                    mapping = np.zeros((400, model.n)); np.add.at(mapping, (cells20, cells_g), 1.)
                    mapping = mapping / np.maximum(mapping.sum(1, keepdims=True), 1)
                    e_counts = f @ mapping
                reg = a['regions_1ms'][max(0, k // 10 - 10):k // 10].astype(float)     # (10, 6)
                g = np.load(Path(folder).parent.parent / 'geometry.npz'); rc = g['region_counts'].astype(float)
                i_reg = reg[:, 3:].sum(0) / rc[3:] / (len(reg) * 1.)                  # per ms per neuron by region
                break
    return np.asarray(e_counts), i_reg


def i_region_of_cells(model):
    """Majority 1.75-mm region of each cell's I neurons."""
    centers = np.asarray(read(SUBSTRATE / 'substrate.json')['centers_mm'])
    geo = np.load(APPROX / f'coarse_{model.grid}/geometry.npz'); pos = geo['positions_i']
    d = np.linalg.norm(pos[:, None] - centers[None], axis=2); g = np.full(len(pos), 2); g[d[:, 0] < 1.75] = 0; g[(d[:, 1] < 1.75) & (d[:, 1] < d[:, 0])] = 1
    out = np.zeros(model.n, int)
    for c in range(model.n):
        idx = np.flatnonzero(model.cell_i == c); out[c] = np.bincount(g[idx], minlength=3).argmax()
    return out


def run(job, version, variant):
    """job: dict(name, mode in {continuation, extension, path, deterministic}, ...)."""
    started = time.time(); folder = variant_dir(version) / 'runs' / job['name']; folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'result.json').exists():
        return read(folder / 'result.json')
    model = ReducedModel(variant); n, K = model.n, model.K
    dt = model.dt; mode = job['mode']
    if mode == 'extension':
        parent = variant_dir(version) / 'runs' / job['parent']
        st = load_pickle(parent / 'end_state.pkl'); model.load_state_dict(st); start_step = int(read(parent / 'result.json')['end_step'])
        input_tag = job['input']; z_path = None
    elif mode in ('continuation', 'fixed_zm'):
        njob = read(nc.NATIVE / 'jobs' / (job['native'] + '.json'))
        state = nc.build_state(njob)
        start_step = int(state['step']); rep = nc.REPLAY_RUN
        e_counts, i_reg = recent_native_rates(rep, ms_to_step(njob['history_ms']), model)
        model.project_native_state(state, e_counts, i_reg[i_region_of_cells(model)])
        model.freeze_m = bool(njob['freeze_m']); input_tag = njob['future']; z_path = None
    elif mode in ('path', 'deterministic'):
        rep = OUT / 'replay' / 'runs' / f"eta0.0005_s{job['seed']}"
        state = ckpt.load(rep / 'checkpoints' / f"t{job['start_ms']}ms.npz"); start_step = int(state['step'])
        e_counts, i_reg = recent_native_rates(rep, start_step, model)
        model.project_native_state(state, e_counts, i_reg[i_region_of_cells(model)])
        model.freeze_m = False
        if mode == 'path':
            input_tag = job['input']; z_path = job['z_path']       # path: dict(step->z field file) handled below
        else:
            input_tag = None; z_path = None
            if job.get('z_source_ms') is not None:
                zs = nc.replay_checkpoint(job['z_source_ms'])['slow']['z'][:NE]; model.set_slow_fields(zs, state['slow']['m'][:NE])
    else:
        raise ValueError(mode)
    steps = ms_to_step(job['duration_ms'])
    if input_tag is not None:
        inp = load_input(input_tag); off = start_step - ms_to_step(float(inp['start_ms'])); assert off >= 0 and off + steps <= int(inp['steps']), (off, steps)
        nu_e = inp[f'cell_mean_{model.grid}']; nu_i_all = inp['global_rate_per_ms']
    frames = steps // 10
    fields = np.empty((frames, 2, n), np.float32); unit_rates = np.empty((steps // 100, n * K), np.float32)
    m_rec = np.empty((steps // 1000, n * K), np.float32); cur = np.empty((frames, 4, n), np.float32)
    z_series = None
    if z_path is not None:
        z_series = np.load(z_path); z_steps = z_series['zm_step']; z_vals = z_series['z']; zi = 0
    for k in range(steps):
        if input_tag is not None:
            ne_ = nu_e[off + k].astype(float); ni_ = float(nu_i_all[off + k])
        else:
            ne_ = np.full(n, model.nu_sig); ni_ = model.nu_sig
        if z_series is not None:
            while zi + 1 < len(z_steps) and z_steps[zi + 1] <= start_step + k:
                zi += 1
            if z_steps[zi] == start_step + k:
                model.set_z_field(z_vals[zi])
        re, ri, mu_u, ex_e, inh_u = model.step(ne_, ni_)
        if (k + 1) % 10 == 0:
            f = (k + 1) // 10 - 1; fields[f, 0] = re * 1000.; fields[f, 1] = ri * 1000.
            cur[f] = np.stack([model.cAE, model.cGE, model.cAI, model.cGI])
        if (k + 1) % 100 == 0:
            unit_rates[(k + 1) // 100 - 1] = model.r_u * 1000.
        if (k + 1) % 1000 == 0:
            m_rec[(k + 1) // 1000 - 1] = model.m_u
        if (k + 1) % 10000 == 0:
            write(folder / 'progress.json', dict(status='RUNNING', step=k + 1, steps=steps, seconds=time.time() - started,
                  mean_E_hz=float(np.average(re, weights=model.count_e) * 1000.), mean_I_hz=float(np.average(ri, weights=model.count_i) * 1000.)))
    np.savez_compressed(folder / 'fields.npz', fields_hz=fields, unit_rates_hz=unit_rates, m_units=m_rec, currents=cur,
                        count_e=model.count_e, count_i=model.count_i, z_u=model.z_u, z2_u=model.z2_u, theta_u=model.theta_u,
                        w_u=model.w_u, start_step=start_step, frame_ms=1., dt_ms=dt)
    save_pickle(folder / 'end_state.pkl', model.state_dict())
    res = readout_from_fields(model, fields, start_step)
    res.update(status='COMPLETE', job=job, version=version, variant=model.v, seconds=time.time() - started, start_step=start_step,
               end_step=start_step + steps, mean_Z_units=float(np.average(model.z_u, weights=model.w_u * np.repeat(model.count_e, K))))
    write(folder / 'result.json', res); write(folder / 'progress.json', dict(status='COMPLETE'))
    return res


def readout_from_fields(model, fields, start_step, region_counts=None):
    """Apply the native event/quiet/spatial definitions to the model's cell rates (expected counts per 1 ms)."""
    n = model.n; T = fields.shape[0]
    counts_e_1ms = fields[:, 0].astype(float) / 1000. * model.count_e[None, :]        # expected E spikes per cell per 1 ms
    all_e_1ms = counts_e_1ms.sum(1)
    r10 = R.rate_10ms(all_e_1ms, NE)
    cell_rate10 = fields[:T // 10 * 10, 0].astype(float).reshape(-1, 10, n).mean(1)
    seps, events = R.find_events(r10); high = R.high_rate_entry(r10)
    nb = len(r10); tail_lo = max(0, nb - R.TAIL_S * 1000 // R.BIN_MS); w = model.count_e
    tail = R.window_stats(r10, seps, events, cell_rate10, w, tail_lo, nb, high)
    subs = [R.window_stats(r10, seps, events, cell_rate10, w, tail_lo + k * 100, tail_lo + (k + 1) * 100, high) for k in range(R.TAIL_S)]
    cat, reasons, sens = R.classify(tail, subs, high, r10[tail_lo:])
    t0 = start_step * DT_MS / 1000.

    def rates(lo, hi):
        fe = fields[lo * 10:hi * 10, 0].astype(float); fi = fields[lo * 10:hi * 10, 1].astype(float)
        out = dict(all_E_hz=float(np.average(fe, axis=1, weights=model.count_e).mean()), all_I_hz=float(np.average(fi, axis=1, weights=model.count_i).mean()))
        for key, name in (('175_0', 'readout175_E_coreA_hz'), ('175_1', 'readout175_E_coreB_hz'), ('175_2', 'readout175_E_surround_hz'),
                          ('15_0', 'core15_E_coreA_hz'), ('15_1', 'core15_E_coreB_hz'), ('15_2', 'core15_E_surround_hz')):
            out[name] = float(np.average(fe, axis=1, weights=model.region_w[key]).mean())
        return out
    tail['rates'] = rates(tail_lo, nb)
    for k, s in enumerate(subs):
        s['rates'] = rates(tail_lo + k * 100, tail_lo + (k + 1) * 100)
    ev_all = [dict(e, start_s=t0 + e['start_bin'] * R.BIN_MS / 1000., end_s=t0 + e['end_bin'] * R.BIN_MS / 1000.) for e in events if e['qualifies']]
    return dict(category=cat, category_reasons=reasons, category_sensitivity=sens,
                high_rate=None if high is None else dict(onset_s=t0 + high['onset_bin'] * R.BIN_MS / 1000., confirmation_s=t0 + high['confirmation_bin'] * R.BIN_MS / 1000.),
                high_rate_reached='REACHED' if high is not None else 'NOT_REACHED_IN_WINDOW', start_s=t0, end_s=t0 + nb * R.BIN_MS / 1000.,
                tail_window_s=[t0 + tail_lo * R.BIN_MS / 1000., t0 + nb * R.BIN_MS / 1000.], tail=tail, subwindows=subs, events_all=ev_all,
                n_events_all=len(ev_all), scope='Model expected rates through the native event/quiet/spatial definitions; no spike-count noise.')


def worker(version, name):
    vd = variant_dir(version); variant = read(vd / 'variant.json'); job = read(vd / 'jobs' / (name + '.json'))
    try:
        run(job, version, variant)
    except Exception as exc:
        write(vd / 'runs' / name / 'failure.json', dict(error=repr(exc), time=time.time())); raise


def launch(version, name):
    env = dict(os.environ, LD_LIBRARY_PATH=LD, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    folder = variant_dir(version) / 'runs' / name; folder.mkdir(parents=True, exist_ok=True)
    log = (folder / 'worker.log').open('a')
    child = subprocess.Popen([PY, '-u', str(Path(__file__).resolve()), 'worker', '--version', version, '--name', name],
                             cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    log.close(); return child


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['worker', 'launch']); ap.add_argument('--version', required=True); ap.add_argument('--name', required=True)
    a = ap.parse_args()
    if a.mode == 'worker':
        worker(a.version, a.name)
    else:
        print('launched', launch(a.version, a.name).pid)
