"""Round-1 -> Fig.5 development handoff. Never selects on confirmation scores."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from src.snn_engine.mz_slow_vars import MZSlowVars, MZSlowVarsConfig

COMPLETE = 'ROUND1_COMPLETE_AWAITING_SCIENTIFIC_REVIEW'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(str((a.shape, a.dtype.str)).encode())
    h.update(memoryview(a).cast('B'))
    return h.hexdigest()


def verify_hashes(hashes, root=None):
    for name, expected in hashes.items():
        path = Path(name)
        if not path.is_absolute():
            path = Path(root) / path
        if sha(path) != expected:
            raise RuntimeError(f'locked input changed: {path}')


def verify_lock(path):
    lock = read(path)
    verify_hashes(lock['hashes'])
    return lock


def choose_development_candidate(selection, confirmation, nominees, seeds):
    """Choose using selection alone; failed confirmation never triggers a runner-up."""
    eligible = [r for r in selection['candidates']
                if r['domain'] in ('whole_sheet', 'interior') and r['selection_eligible']
                and r['J_round1'] is not None and math.isfinite(r['J_round1'])]
    if not eligible:
        raise ValueError('NO_QUALIFIED_SELECTION_CANDIDATE')
    winner = min(eligible, key=lambda r: (r['J_round1'], r['candidate_id']))
    cid = winner['candidate_id']
    candidate = next((r for r in nominees['candidates'] if r['candidate_id'] == cid), None)
    if candidate is None:
        raise RuntimeError('selection winner not nominated before confirmation')
    row = next(r for r in confirmation['candidates'] if r['candidate_id'] == cid)
    units = row['units']
    if (len(units) != len(seeds) or sorted(u['seed'] for u in units) != sorted(seeds)
            or not row['selection_eligible'] or not row['all_components_estimable']
            or row['n_returned'] < 20 or any(u['runaway'] for u in units)
            or any(u.get('J_direction') is None or not math.isfinite(u['J_direction']) for u in units)):
        raise ValueError('PRESELECTED_CANDIDATE_FAILED_CONFIRMATION_NO_RESELECTION')
    mechanism = candidate['mechanisms']
    if mechanism['g_EE'] != 0 or mechanism['g_EtoI'] != 0 or mechanism['Z_M'] != 'off':
        raise RuntimeError('selected upstream substrate is not VTH-only')
    if candidate['node_field']['target_count'] != 1499:
        raise RuntimeError('VTH neuron budget changed')
    return candidate, winner, row


def available_gib():
    return next(float(l.split()[1]) / 1024**2 for l in Path('/proc/meminfo').read_text().splitlines()
                if l.startswith('MemAvailable:'))


def proc_memory(pid):
    try:
        lines = Path(f'/proc/{pid}/status').read_text().splitlines()
    except (FileNotFoundError, ProcessLookupError):
        return {}
    return {l.split(':')[0]: float(l.split()[1]) / 1024**2 for l in lines
            if l.startswith(('VmRSS:', 'VmPeak:', 'VmHWM:'))}


def admission_slots(available, reserve, budget, running_rss, maximum):
    outstanding = sum(max(0., budget - rss) for rss in running_rss)
    return max(0, min(maximum - len(running_rss),
                      math.floor((available - reserve - outstanding) / budget)))


def parameter_grid(plan):
    reference = plan['zm_reference']
    configs = []
    for factor in plan['threshold_factors']:
        for tau in plan['tau_z_ms']:
            configs.append(dict(reference, config_id=f'grid_i{factor:g}_tz{tau:g}',
                                I_th_EI=reference['I_th_EI'] * factor, tau_z=tau,
                                family='threshold_tau_z', threshold_factor=factor))
    for parameter, factors in plan['sensitivity_factors'].items():
        for factor in factors:
            configs.append(dict(reference, config_id=f'sensitivity_{parameter}_{factor:g}',
                                family=parameter, **{parameter: reference[parameter] * factor}))
    configs.extend([dict(reference, config_id='slow_off', family='control', use_z=False, use_m=False),
                    dict(reference, config_id='clamp_z', family='control', use_z=False)])
    return configs


def region_labels(pos, h, centers):
    nearest = np.argmin(((np.asarray(pos)[:, None] - np.asarray(centers)[None])**2).sum(axis=2), axis=1)
    return np.where(np.asarray(h) >= .5, nearest, 2).astype(np.int64)


class Fig5RegionSlowVars(MZSlowVars):
    """Only adds recorders; all per-neuron equations and thresholds are inherited."""
    def __init__(self, *args, region_E, **kwargs):
        super().__init__(*args, **kwargs)
        self.region_E = np.asarray(region_E)
        if self.region_E.shape != (self.NE,) or set(np.unique(self.region_E)) != {0, 1, 2}:
            raise ValueError('three nonempty E regions required')
        self._region_trace = {k: [] for k in ('z', 'm', 'disinhibition', 'adaptation')}

    def _record_trace(self, spikes, dt):
        super()._record_trace(spikes, dt)
        z, m, inhibitory = self.z[:self.NE], self.m[:self.NE], self._I_I_last[:self.NE]
        for key, values in [('z', z), ('m', m), ('disinhibition', (1 - z) * inhibitory),
                            ('adaptation', self.cfg.eta_m * m)]:
            self._region_trace[key].append([float(np.mean(values[self.region_E == r])) for r in range(3)])

    def region_arrays(self):
        return {f'region_{k}': np.asarray(v, dtype=np.float32) for k, v in self._region_trace.items()}


def make_slow(substrate, cfg, regions):
    params = {k: cfg[k] for k in MZSlowVarsConfig.__dataclass_fields__}
    return Fig5RegionSlowVars(substrate.n_e + substrate.n_i, substrate.params.V_th,
                             MZSlowVarsConfig(**params), NE=substrate.n_e,
                             core_mask_E=regions < 2, region_E=regions)


def binned_rates(spikes, regions, dt, bin_ms=20.):
    """Chunk time to avoid an additional full-duration spike/region copy."""
    width = int(round(bin_ms / dt))
    n = len(spikes) // width
    counts = np.bincount(regions, minlength=3)
    rates = np.empty((n, 4))
    for i in range(n):
        by_neuron = spikes[i * width:(i + 1) * width].sum(axis=0)
        rates[i, 0] = by_neuron.mean() * 1000 / bin_ms
        rates[i, 1:] = np.bincount(regions, weights=by_neuron, minlength=3) / counts * 1000 / bin_ms
    return (np.arange(n) + .5) * bin_ms, rates


def classify_trajectory(time_ms, rates, horizon_ms, bin_ms=20.):
    """Tonic is an operational 1-s plateau; no event = right-censored observation."""
    rates = np.asarray(rates)
    if rates.ndim != 2 or rates.shape[1] != 4 or not np.isfinite(rates).all():
        raise ValueError('invalid population/core A/core B/surround rate array')
    n_tail = int(round(1000 / bin_ms))
    gate_time = None
    def plateau(tail):
        return (len(tail) == n_tail and tail[:, 0].mean() >= 300
                and np.all(tail[:, 1:].mean(axis=0) >= 250)
                and abs(tail[n_tail // 2:, 0].mean() - tail[:n_tail // 2, 0].mean()) <= 5)
    terminal_pass = plateau(rates[-n_tail:])
    for end in range(n_tail, len(rates) + 1) if terminal_pass else ():
        tail = rates[end - n_tail:end]
        if plateau(tail):
            gate_time = float(time_ms[end - 1] + bin_ms / 2)
            break
    onset = None
    if gate_time is not None:
        # Last crossing into the consecutive high-rate segment preceding the validated plateau.
        end = int(round(gate_time / bin_ms)) - 1
        start = end
        while start > 0 and rates[start - 1, 0] >= 250:
            start -= 1
        onset = float(time_ms[start] - bin_ms / 2)
    observed = float(time_ms[-1] + bin_ms / 2) if len(time_ms) else 0.
    status = 'TONIC_RUNAWAY' if onset is not None else (
        'NO_TONIC_WITHIN_HORIZON' if observed >= horizon_ms - bin_ms else 'EARLY_STOP_UNRESOLVED')
    return {'classification': status, 'runaway': onset is not None, 'onset_ms': onset,
            'tonic_gate_confirmed_ms': gate_time, 'observed_ms': observed, 'horizon_ms': horizon_ms,
            'right_censored': status == 'NO_TONIC_WITHIN_HORIZON',
            'latency_observed_ms': onset if onset is not None else observed,
            'qualified_pretransition': onset is not None and onset >= 2000.,
            'tail_population_hz': float(rates[-n_tail:, 0].mean()) if len(rates) else None}


def critical_state(slow_arrays, onset_ms):
    if onset_ms is None:
        return None
    t = slow_arrays['slow_time_ms']
    idx = max(0, int(np.searchsorted(t, onset_ms, side='right')) - 1)
    return {k: np.asarray(slow_arrays[k][idx]).tolist() for k in
            ('region_z', 'region_m', 'region_disinhibition', 'region_adaptation')}


def choose_mechanism_config(configs, results, required=2):
    """First prespecified non-control config with >=2/3 qualifying pilot trajectories."""
    for cfg in configs:
        if cfg['family'] == 'control':
            continue
        rows = [r for r in results if r['job']['config']['config_id'] == cfg['config_id']]
        if len(rows) == 3 and sum(r['trajectory']['qualified_pretransition'] for r in rows) >= required:
            return cfg
    return None
