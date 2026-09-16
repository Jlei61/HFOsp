#!/usr/bin/env python3
"""Paired, count-matched loss-sensitivity audit on TRAINING blocks only.

The full-data null distance is not compared with a small-sample floor as if
sample sizes were equal. True and perturbed pseudo-models share event indices.
No running search, source lock, target, score, or acceptance rule is changed.
"""
from pathlib import Path
import argparse
import sys
import json
import time
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.run_topic4_xy_research import training_contract, read, write, sha
from src.topic4_joint_xy import joint_features, projections, projected_quantiles, observable_groups

OUT = ROOT/'results/topic4_sef_hfo/joint_rank_space_dual_core_search/loss_power_audit'


def rank_input(features, count):
    mask = features[:, :count] * 2*np.sqrt(count)
    rank = features[:, count:2*count] * 4*np.sqrt(count) - 1
    return np.column_stack([mask, np.where(mask > .5, 2*rank-1, 0.)])/np.sqrt(count)


def rff_map(values, weights, phases):
    output = np.empty((len(values), len(phases)), np.float32)
    for a in range(0, len(values), 1024):
        output[a:a+1024] = np.sqrt(2/len(phases))*np.cos(values[a:a+1024]@weights+phases)
    return output


def make_features(t, xy, groups, count, specs):
    f = joint_features(t, xy, groups)
    blocks = {'joint_sw': f, 'support_sw': f[:, :count], 'rank_sw': f[:, count:2*count],
              'lag_sw': f[:, 2*count:3*count], 'space_sw': f[:, 3*count:]}
    result = {k: v@specs[k]['axes'] for k, v in blocks.items()}
    result['rank_rff_mmd2'] = rff_map(rank_input(f, count), specs['rank_rff_mmd2']['weights'],
                                     specs['rank_rff_mmd2']['phases'])
    return result


def quantiles(projected):
    return np.quantile(projected, (np.arange(256)+.5)/256, axis=0)


def distances(mapped, idx, reference):
    return {k: float(np.sum((v[idx].mean(axis=0)-reference[k])**2)) if k == 'rank_rff_mmd2'
            else float(np.abs(quantiles(v[idx])-reference[k]).mean()) for k, v in mapped.items()}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--draws', type=int, default=48)
    args = parser.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    training, _ = training_contract(); t = training['onsets_ms']; n, c = t.shape
    direction_path = ROOT/'results/topic4_sef_hfo/vth_dual_core_xy_research/direction_objective_v2.json'
    xy = np.asarray(read(direction_path)['contact_xy_mm']); groups = training['groups']
    f = joint_features(t, xy, groups)
    specs = {k: {'axes': projections(d, count=128)} for k, d in
             [('joint_sw', f.shape[1]), ('support_sw', c), ('rank_sw', c), ('lag_sw', c), ('space_sw', 8)]}
    rng = np.random.default_rng(2026090604)
    rank = rank_input(f, c); a = rng.integers(n, size=4096); b = rng.integers(n, size=4096)
    bandwidth = np.median(np.linalg.norm(rank[a]-rank[b], axis=1))
    specs['rank_rff_mmd2'] = {'weights': rng.normal(size=(2*c, 1024))/bandwidth,
                             'phases': rng.uniform(0, 2*np.pi, size=1024)}
    first = np.nanmin(t, axis=1)[:, None]
    shuffled = t.copy()
    for row in shuffled:
        valid = np.flatnonzero(np.isfinite(row)); row[valid] = rng.permutation(row[valid])
    assert np.array_equal(np.isfinite(shuffled), np.isfinite(t))
    drop = t.copy(); drop[rng.random(t.shape) < .1] = np.nan
    variants = {'true': t, 'rank_shuffle': shuffled, 'time_stretch_1_5': first+1.5*(t-first),
                'reverse_time': first+np.nanmax(t-first, axis=1)[:, None]-(t-first),
                'drop_10pct_contacts': drop}
    mapped = {name: make_features(values, xy, groups, c, specs) for name, values in variants.items()}
    del f, rank
    samples = []; blocks = training['block_ids']; unique = np.unique(blocks)
    for draw in range(args.draws):
        chosen = rng.choice(unique, len(unique)//2, replace=False)
        left = np.isin(blocks, chosen); pool = np.flatnonzero(left)
        reference = {k: v[~left].mean(axis=0) if k == 'rank_rff_mmd2' else quantiles(v[~left])
                     for k, v in mapped['true'].items()}
        for count in (16, 64, 256):
            index = rng.choice(pool, size=count, replace=False)
            for name, m in mapped.items():
                samples.append({'draw': draw, 'n': count, 'perturbation': name, **distances(m, index, reference)})
        if draw % 8 == 0:
            print(f'Paired loss audit {draw+1}/{args.draws}', flush=True)
            write(OUT/'status.json', {'status': 'RUNNING', 'draws_complete': draw+1})
    summary = []
    for count in (16, 64, 256):
        for metric in specs:
            original = np.array([r[metric] for r in samples if r['n'] == count and r['perturbation'] == 'true'])
            cutoff = float(np.quantile(original, .95))
            for name in variants:
                if name == 'true': continue
                altered = np.array([r[metric] for r in samples if r['n'] == count and r['perturbation'] == name])
                summary.append({'n': count, 'metric': metric, 'perturbation': name,
                    'patient_q95': cutoff, 'true_median': float(np.median(original)),
                    'altered_median': float(np.median(altered)),
                    'paired_delta_median': float(np.median(altered-original)),
                    'fraction_detected_at_empirical_q95': float(np.mean(altered > cutoff)),
                    'fraction_altered_exceeds_paired_true': float(np.mean(altered > original))})
    write(OUT/'paired_draws.json', {'rows': samples})
    write(OUT/'summary.json', {'status': 'TRAINING_LOSS_POWER_AUDIT_COMPLETE', 'rows': summary,
        'draws': args.draws, 'rng_seed': 2026090604, 'rank_kernel_bandwidth': float(bandwidth),
        'patient_training_sha256': training['sha256'], 'producer_sha256': sha(Path(__file__)),
        'geometry_sha256': sha(direction_path), 'heldout_opened': False,
        'running_objective_changed': False,
        'inference': 'Descriptive sensitivity, not independent test power: q95 is estimated from these same true draws. Perturbations are paired by events and use disjoint reference blocks.'})
    # Diagnostic reranking on old trajectories only; do not update the live search.
    search = OUT.parent; baseline = read(search/'baseline_scores.json')['candidates']
    plan = read(ROOT/'config/topic4_joint_xy_adaptive_v1.json')
    reference = {k: v.mean(axis=0) if k == 'rank_rff_mmd2' else quantiles(v) for k, v in mapped['true'].items()}
    candidates = []
    for row in baseline:
        tables = []
        for unit in row['units']:
            p = Path(unit['worker_path']); meta = read(p)
            if sha(p) != unit['worker_sha256'] or sha(meta['arrays']['path']) != meta['arrays']['sha256']:
                raise RuntimeError('historical worker changed')
            with np.load(meta['arrays']['path']) as z:
                table, _ = observable_groups(z['contact_envelope'], float(z['contact_envelope_dt_ms']), **plan['observation'])
                tables.append(table)
        values = np.concatenate(tables)
        if not len(values): continue
        m = make_features(values, xy, groups, c, specs)
        candidates.append({'candidate_id': row['candidate_id'], 'n': len(values),
            **distances(m, np.arange(len(values)), reference),
            'original_exploration_score': row['exploration_score'],
            'original_D_order': row['D_order'], 'original_D_lag': row['D_lag'],
            'original_direction_distance': row['direction_distance']})
    write(OUT/'candidate_diagnostic_distances.json', {'rows': candidates, 'not_live_search_ranking': True})
    write(OUT/'status.json', {'status': 'COMPLETE', 'draws_complete': args.draws, 'n_candidates': len(candidates)})
    print(json.dumps([r for r in summary if r['n'] in (16, 256) and r['perturbation'] == 'rank_shuffle'], indent=2))


if __name__ == '__main__': main()
