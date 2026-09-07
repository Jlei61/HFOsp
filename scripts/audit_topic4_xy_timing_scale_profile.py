#!/usr/bin/env python3
"""Offline timing-scale diagnostic; never selects or qualifies a substrate."""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_topic4_joint_xy_replicated_search as run

SCALES = [.25, .4, .5, .67, .8, 1., 1.25, 1.5, 2.]


def main():
    out = run.OUT / 'timing_scale_profile_audit'
    out.mkdir(exist_ok=True)
    source_paths = [run.OUT / 'baseline_scores.json']
    stage = run.OUT / 'rounds/001'
    ids = run.read(stage / 'race_nomination.json')['candidate_ids']
    source_paths += [stage / f'combined_{cid}.json' for cid in ids]
    rows = [r for p in source_paths for r in run.read(p)['candidates']
            if len(r['units']) == 8]
    if len({r['candidate_id'] for r in rows}) != len(rows):
        raise RuntimeError('duplicate geometry')
    locks = {**run.read(run.OUT / 'objective_contract.json')['source_hashes'],
             **run.read(run.OUT / 'analysis_input_lock.json')['hashes']}
    run.v1.runtime.verify_amendment(locks)
    plan = run.read(run.CONFIG)
    obj = run.KernelObjective(run.v1, run.OUT, run.KERNEL)
    cal = run.read(run.OUT / 'patient_calibration.json')
    results = []
    for row in rows:
        if sorted(u['seed'] for u in row['units']) != list(range(2511, 2519)):
            raise RuntimeError('incomplete common seed set')
        tables = []
        for unit in row['units']:
            path = Path(unit['worker_path'])
            if run.sha(path) != unit['worker_sha256']:
                raise RuntimeError('worker metadata changed')
            meta = run.read(path)
            if run.sha(Path(meta['arrays']['path'])) != meta['arrays']['sha256']:
                raise RuntimeError('worker arrays changed')
            tables.append(run.v1.read_worker(path, obj, plan)[0])
        t = np.concatenate(tables)
        assert len(t) == row['n_events']
        valid = np.isfinite(t)
        first = np.min(np.where(valid, t, np.inf), axis=1)
        first = np.where(valid.any(axis=1), first, 0.)
        profile = []
        for scale in SCALES:
            scaled = first[:, None] + (t-first[:, None])*scale
            np.testing.assert_array_equal(np.isfinite(scaled), valid)
            metric = obj.metrics(scaled)
            for key in ['support', 'rank_space']:
                np.testing.assert_allclose(metric['kernel_distances'][key],
                    row['kernel_distances'][key], atol=1e-7, rtol=1e-6)
            if scale == 1:
                for key, value in row['kernel_distances'].items():
                    np.testing.assert_allclose(metric['kernel_distances'][key], value,
                                               atol=1e-7, rtol=1e-6)
            profile.append({'scale': scale, 'kernel_distances': metric['kernel_distances'],
                            'D_lag': metric['D_lag']})
        size = next((n for n in sorted(map(int, cal['samples'])) if n >= len(t)), 256)
        results.append({'candidate_id': row['candidate_id'], 'n_events': len(t),
                        'profile': profile, 'kernel_thresholds': cal['samples'][str(size)]['kernel_q95'],
                        'native_centroid_span_median_ms': float(np.nanmedian(np.nanmax(t, axis=1)-np.nanmin(t, axis=1))),
                        'best_timing_grid_scale': min(profile, key=lambda p:p['kernel_distances']['timing_space'])['scale'],
                        'best_joint_grid_scale': min(profile, key=lambda p:p['kernel_distances']['joint'])['scale']})
        print(row['candidate_id'], 'profile complete', flush=True)
    common = [{'scale': s, 'mean_timing_space_distance': float(np.mean([
        r['profile'][i]['kernel_distances']['timing_space'] for r in results]))}
        for i,s in enumerate(SCALES)]
    run.v1.runtime.verify_amendment(locks)
    result = {'status': 'OFFLINE_TIMING_SCALE_PROFILE_COMPLETE', 'scale_grid': SCALES,
        'n_geometries': len(results), 'n_networks_each': 8, 'results': results,
        'equal_geometry_mean_timing_profile': common,
        'patient_centroid_span_median_ms': float(np.nanmedian(np.nanmax(obj.patient,axis=1)-np.nanmin(obj.patient,axis=1))),
        'claim_boundary': 'Post-hoc diagnostic transformation of saved event times only. No resimulation, '
                         'no re-packing, no validated parameter mapping, no qualification or independent confirmation. '
                         'Minimum grid distances are optimistic training diagnostics, not validated model fits.',
        'live_search_changed': False, 'heldout_opened': False,
        'participation_and_rank_invariance_checked': True, 'native_distance_parity_checked': True,
        'source_hashes': {str(p):run.sha(p) for p in source_paths + [Path(__file__),run.CONFIG,
            run.OUT/'patient_calibration.json',stage/'race_nomination.json']}}
    run.write(out / 'summary.json', result)
    print('COMMON PROFILE', common, flush=True)


if __name__ == '__main__':
    main()
