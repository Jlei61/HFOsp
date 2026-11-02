#!/usr/bin/env python3
"""Paired frozen-substrate assays. No changes to historical SNN code."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / '.worktrees/topic4-substrate-autapse-fix'
ZROOT = ROOT / '.worktrees/topic4-dual-core-z-bifurcation'
OUT = ROOT / 'results/topic4_sef_hfo/rate_model_dynamics_validation_v1'
sys.path[:0] = [str(SOURCE), str(SOURCE / 'src/snn_engine')]
import numpy as np
from scipy import sparse
import src
src.__path__.append(str(ZROOT / 'src'))
src.__path__.append(str(ROOT / 'src'))
from src.topic4_zm_ictal_transition import build_substrate, make_external_drive
from src.topic4_multidimensional_parameters import apply_parameters
from src.topic4_graph_edge_flow import array_sha256
from src.topic4_patient_zm_meanfield import (
    build_patient_coarse_model, save_patient_coarse_model, spatial_cell_index)
from src.topic4_dual_core_spatial_z_delay import build_coarse_delay_operators
from src.topic4_rate_validation_engine import simulate_kick


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def setup(seed):
    frozen = read(ROOT / 'config/topic4_rate_model_dynamics_validation_v1.json')
    for relative, info in frozen['source_records'].items():
        if hashlib.sha256((SOURCE / relative).read_bytes()).hexdigest() != info['sha256']:
            raise RuntimeError('Source changed: ' + relative)
    ec = next(x for x in frozen['inputs'] if x['path'].endswith('execution_config.json'))
    config = read(ec['path'])
    transition = read(SOURCE / config['inputs']['transition_config']['path'])
    c = frozen['candidate']; m = c['mechanisms']; n = c['node_mapping']
    s = build_substrate(transition, config['reference']['base_substrate_candidate_id'], 6101,
        cache_dir=config['network_cache'], artifact_root=ROOT,
        topology_seed=6101, dynamics_seed=seed,
        network_cache_record=config['corrected_networks']['6101'],
        ee_dose=m['g_EE'], etoi_dose=m['g_EtoI'], node_candidate_override=c['node_field'],
        node_depth_shrinkage=n['signed_depth_shrinkage'], node_gain=n['node_gain'],
        ee_ellipse_angle_deg=m['ellipse_angle_deg'], ee_ellipse_aspect_ratio=m['ellipse_aspect_ratio'],
        ee_ellipse_reference_angle_deg=m['ellipse_reference_angle_deg'],
        ee_ellipse_reference_aspect_ratio=m['ellipse_reference_aspect_ratio'])
    pa = apply_parameters(s, c['dynamic_parameters'])
    identity = {name: array_sha256(np.asarray(value, np.float32)) for name, value in (
        ('positions_E_sha256', s.positions_e), ('h_sha256', s.h_e),
        ('delta_vtheta_sha256', s.delta_vtheta), ('vtheta_sha256', s.vtheta))}
    for kind in ('ampa', 'gaba'):
        d = pa['sparse_pathways'][kind + '_by_delay']
        identity[kind + '_topology_sha256'] = d['topology_sha256']
        identity[kind + '_values_sha256'] = d['after_sha256']
    assert all(frozen['static_array_identity'][k] == v for k, v in identity.items())
    return s, transition, frozen, identity


def prepare():
    started = time.time(); s, tr, frozen, identity = setup(7101)
    write(OUT / 'reconstruction.json', {'status': 'STATIC_RECONSTRUCTION_PASS',
          'identity': identity, 'spatial_ou': tr['spatial_ou'],
          'params': {k: v for k, v in vars(s.params).items() if isinstance(v, (int, float, str, bool))}})
    for grid in (10, 20):
        folder = OUT / f'coarse_{grid}'; folder.mkdir(exist_ok=True)
        model = build_patient_coarse_model(s, n_grid=grid, threshold_groups=8)
        save_patient_coarse_model(folder / 'model.npz', model)
        ops = build_coarse_delay_operators(s, model)
        for key in ('ee', 'ei', 'ie', 'ii'):
            sparse.save_npz(folder / f'delay_{key}.npz', getattr(ops, f'w_{key}_history'))
        cells = spatial_cell_index(s.positions_e, n_grid=grid, sheet_l_mm=s.params.L)
        np.savez_compressed(folder / 'geometry.npz', cell_e=cells,
            positions_e=s.positions_e, vtheta=s.vtheta[:s.n_e], h_e=s.h_e,
            contact_xy=s.contact_xy)
        write(folder / 'prepared.json', {'dt_ms': ops.dt_ms, 'max_delay_steps': ops.max_delay_steps,
            'grid': grid, 'threshold_groups': 8, 'graph_identity': identity,
            'tau_r_ampa_ms': s.params.tau_r_AMPA, 'tau_r_gaba_ms': s.params.tau_r_GABA})
        print('prepared', grid, flush=True)
    write(OUT / 'preparation_complete.json', {'status': 'COMPLETE', 'seconds': time.time() - started,
        'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2})


class RecordDrive:
    """Read-only projection of the exact external OU process; no extra RNG calls."""
    def __init__(self, s, transition, seed, steps):
        self.drive = make_external_drive(s, transition['spatial_ou'], seed)
        from params import compute_nu_theta
        self.base = s.params.nu_ext_ratio * compute_nu_theta(s.params)[0]
        self.maps = {g: spatial_cell_index(s.positions_e, n_grid=g, sheet_l_mm=s.params.L) for g in (10, 20)}
        self.counts = {g: np.bincount(c, minlength=g*g) for g, c in self.maps.items()}
        self.values = {g: np.empty((steps, g*g), np.float32) for g in self.maps}
        self.variance = {g: np.empty((steps, g*g), np.float32) for g in self.maps}
        self.index = 0
        self.global_rate = np.empty(steps, np.float32)
        self.global_xi = np.empty(steps, np.float32)
        self.current = {g: np.empty((steps // 20, 3, g*g), np.float32) for g in self.maps}

    def step(self, t):
        return self.drive.step(t)

    def observe_input(self, t, nu_vec, xi):
        values = nu_vec[:len(self.maps[10])]
        for g, cells in self.maps.items():
            mean = np.bincount(cells, weights=values, minlength=g*g) / self.counts[g]
            self.values[g][self.index] = mean
            self.variance[g][self.index] = np.maximum(
                np.bincount(cells, weights=values**2, minlength=g*g) / self.counts[g] - mean**2, 0.)
        self.index += 1
        self.global_rate[self.index-1] = nu_vec[-1]
        self.global_xi[self.index-1] = xi

    def observe_current(self, t, ie, ii, voltage):
        step = self.index - 1
        if step % 20:
            return
        for g, cells in self.maps.items():
            for j, value in enumerate((ie, ii, voltage)):
                self.current[g][step//20, j] = np.bincount(cells, weights=value[:len(cells)], minlength=g*g) / self.counts[g]


def run_snn(args):
    started = time.time(); s, tr, frozen, identity = setup(args.seed)
    s.params.T = args.duration; s.net['rng'] = np.random.default_rng(args.seed)
    centers = frozen['candidate']['node_field']['centers_mm']
    center = {'A': centers[0], 'B': centers[1], 'surround': [10., 5.]}[args.site]
    steps = int(round(args.duration / s.params.dt))
    drive = RecordDrive(s, tr, args.seed, steps)
    result = simulate_kick(s.params, s.net, KICK_BOOST=args.dose,
        t_kick=args.onset, kick_center=center, r_kick=.75, slow=None,
        t_kick2=args.second, KICK_BOOST2=args.dose if args.second is not None else 0.,
        V_th_per_neuron=s.vtheta, external_e_rate_drive=drive,
        input_observer=drive.observe_input, current_observer=drive.observe_current,
        early_stop_runaway=False, dump_i_spikes=True)
    spikes = result['E_spk_bool']; dt = s.params.dt
    cm = s.extras['cmrun']; active, active_dt = cm.active_fraction(spikes, dt, cm.BIN_MS)
    arrays = {'rate_e_hz': result['rate_E'], 'rate_i_hz': result['rate_I'],
              'active_fraction': active, 'active_dt_ms': active_dt, 'dt_ms': dt,
              'external_i_rate': drive.global_rate, 'global_xi': drive.global_xi}
    stride = int(round(2. / dt)); nframes = len(spikes) // stride
    for grid, cells in drive.maps.items():
        counts = np.zeros((nframes, grid*grid), np.float32)
        cells_i = spatial_cell_index(s.positions_i, n_grid=grid, sheet_l_mm=s.params.L)
        ni = np.bincount(cells_i, minlength=grid*grid)
        counts_i = np.zeros_like(counts)
        for frame in range(nframes):
            mass = spikes[frame*stride:(frame+1)*stride].sum(axis=0)
            counts[frame] = np.bincount(cells, weights=mass, minlength=grid*grid)
            mass_i = result['I_spk_bool'][frame*stride:(frame+1)*stride].sum(axis=0)
            counts_i[frame] = np.bincount(cells_i, weights=mass_i, minlength=grid*grid)
        arrays[f'field_e_hz_{grid}'] = counts / drive.counts[grid][None, :] * 500.
        arrays[f'field_i_hz_{grid}'] = counts_i / ni[None, :] * 500.
        arrays[f'field_counts_{grid}'] = counts
        arrays[f'external_rate_{grid}'] = drive.values[grid]
        arrays[f'external_within_cell_variance_{grid}'] = drive.variance[grid]
        arrays[f'current_voltage_means_{grid}'] = drive.current[grid]
        kick_mask = np.linalg.norm(s.positions_e - center, axis=1) <= .75
        arrays[f'stimulus_fraction_{grid}'] = np.bincount(cells, weights=kick_mask, minlength=grid*grid) / drive.counts[grid]
    name = args.name
    folder = OUT / 'snn'; folder.mkdir(exist_ok=True)
    np.savez_compressed(folder / f'{name}.npz', **arrays)
    check = None
    if args.dose == 0 and args.seed in (7101, 7102):
        old = np.load(next(r['arrays']['path'] for r in frozen['reference_runs'] if r['dynamics_seed'] == args.seed))
        count_old = old['sheet_activity_counts'][:nframes].reshape(nframes, -1)
        count_new = arrays['field_counts_20']
        # Stored map orientation is checked explicitly, never optimized by agreement.
        old_active = old['active_fraction'][:len(active)]
        check = {'active_fraction_equal_at_saved_precision': bool(np.array_equal(active.astype(old_active.dtype), old_active)),
                 'active_fraction_max_absolute_roundoff': float(np.max(np.abs(active-old_active))),
                 'native_count_shapes': [list(count_old.shape), list(count_new.shape)],
                 'native_counts_equal': bool(np.array_equal(count_old, count_new))}
    write(folder / f'{name}.json', {'status': 'COMPLETE', 'args': vars(args),
        'static_identity_pass': True, 'prefix_check': check,
        'mean_e_hz_after_500ms': float(np.mean(result['rate_E'][int(500/dt):])) if args.duration > 500 else None,
        'peak_rss_gib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        'wall_seconds': time.time()-started, 'stimulus_duration_ms': 18.,
        'stimulus_n_e': int(kick_mask.sum()), 'stimulus_center_mm': center,
        'external_input': 'Exact historical OU; same Poisson generator as historical engine; added rate after clipping.',
        'paired_randomness': 'Local spatial OU has an independent seed and is paired. Global OU shares the Poisson RNG and can diverge across doses; its actual trace is recorded separately.'})
    print(name, check, 'finished', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['prepare', 'snn'])
    p.add_argument('--name', default='prefix'); p.add_argument('--seed', type=int, default=7101)
    p.add_argument('--duration', type=float, default=600.); p.add_argument('--site', choices=['A', 'B', 'surround'], default='A')
    p.add_argument('--dose', type=float, default=0.); p.add_argument('--onset', type=float, default=1000.)
    p.add_argument('--second', type=float)
    args = p.parse_args()
    if args.mode == 'prepare': prepare()
    else: run_snn(args)
