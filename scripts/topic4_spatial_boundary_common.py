"""Shared definitions for the bounded sustained-recruitment audit."""
from validate_topic4_fixed_rate_base import ROOT, read, write
import numpy as np

OUT = ROOT / 'results/topic4_sef_hfo/sustained_spatial_recruitment_boundary_v1'
OLD = ROOT / 'results/topic4_sef_hfo/z_transition_bifurcation_audit_v1'
REFERENCE = ROOT / 'results/topic4_sef_hfo/autonomous_z_manual_restore_v1'


def checkpoint_path(tm):
    local = OUT / 'checkpoints' / f't{tm}ms.npz'
    return local if local.exists() else OLD / 'checkpoints' / f't{tm}ms.npz'


def runs(mask, bin_ms):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return (np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)) * bin_ms


def observables(field_counts, cell_counts, window_ms=1000):
    """Native E spikes in 1-ms bins; cells weighted by neuron count, not equally."""
    counts = np.asarray(field_counts[-window_ms:], float)
    weights = np.asarray(cell_counts, float)
    assert len(counts) % 10 == 0 and np.all(weights > 0)
    rate1 = counts / weights[None, :] * 1000
    global1 = counts.sum(1) / weights.sum() * 1000
    g5 = global1.reshape(-1, 5).mean(1)
    r10 = rate1.reshape(-1, 10, len(weights)).mean(1)
    quiet = {}
    for threshold in (.5, 1., 2.):
        for bin_ms in (5, 10):
            g = global1.reshape(-1, bin_ms).mean(1)
            gaps = runs(g < threshold, bin_ms)
            quiet[f'{threshold:g}Hz_{bin_ms}ms'] = {
                'fraction': float(np.mean(g < threshold)),
                'gap_durations_ms': gaps.tolist(),
                'gaps_at_least_20ms': int(np.sum(gaps >= 20)),
                'longest_gap_ms': float(gaps.max(initial=0)),
            }
    spatial = {}
    for threshold in (50., 100., 200.):
        duty = np.mean(r10 > threshold, axis=0)
        footprint = np.average(r10 > threshold, axis=1, weights=weights)
        spatial[f'{threshold:g}Hz'] = {
            'mean_instantaneous_fraction': float(footprint.mean()),
            'peak_instantaneous_fraction': float(footprint.max()),
            'persistent_fraction_duty80': float(np.average(duty >= .8, weights=weights)),
            'persistent_fraction_duty90': float(np.average(duty >= .9, weights=weights)),
            'cell_duty': duty.tolist(),
        }
    local_var = np.average(r10.var(0), weights=weights)
    return {
        'window_ms': window_ms, 'E_mean_hz': float(global1.mean()),
        'E_min_5ms_hz': float(g5.min()), 'E_max_5ms_hz': float(g5.max()),
        'quiet': quiet, 'spatial': spatial,
        'global_to_local_variance_ratio_10ms': float(np.average(r10, axis=1, weights=weights).var() / max(local_var, 1e-12)),
        'maximum_cell_sd_10ms_hz': float(r10.std(0).max()),
        'scope': 'Finite-window descriptors; local variability alone does not establish a sustained oscillation or bifurcation.',
    }
