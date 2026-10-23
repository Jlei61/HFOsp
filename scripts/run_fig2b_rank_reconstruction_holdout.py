#!/usr/bin/env python3
"""Evaluate frozen Figure 2B templates against held-out complete rank sequences."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import sys
import zlib

import numpy as np
from scipy.stats import wilcoxon
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.plot_topic5_interictal_template_direction_rose import _subject_dir
from src.interictal_propagation import load_subject_propagation_events
from src.interictal_rank_holdout import (
    METHODS, prepare_reconstruction, closest_template_error, shuffle_participating_orders,
)

FROZEN = ROOT / 'results/topic1/direction_distribution_holdout_20261022'
OUT = ROOT / 'results/topic1/rank_reconstruction_holdout_20261022'
SEED = 20261022


def jsonable(v):
    if isinstance(v, dict):
        return {k: jsonable(w) for k,w in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [jsonable(w) for w in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v) if np.isfinite(v) else None
    return v


def write_json(path, obj):
    path.write_text(json.dumps(jsonable(obj), ensure_ascii=False, indent=2)+'\n')


def paired_summary(a, b):
    a, b = np.asarray(a), np.asarray(b)
    delta = a-b
    rng = np.random.default_rng(SEED)
    medians = np.median(delta[rng.integers(0, len(a), (10000, len(a)))], axis=1)
    return dict(n=len(a), a_median=float(np.median(a)), b_median=float(np.median(b)),
                median_a_minus_b=float(np.median(delta)),
                paired_difference_ci95=np.quantile(medians, [.025, .975]),
                n_b_lower=int((delta>0).sum()), n_equal=int((delta==0).sum()),
                wilcoxon_two_sided_p=float(wilcoxon(a,b).pvalue) if np.any(delta) else 1.)


def stars(p):
    return '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else 'n.s.'


def run_patient(sid, out, frozen_root, n_shuffles):
    with threadpool_limits(limits=1):
        return _run_patient(sid, Path(out), Path(frozen_root), n_shuffles)


def _run_patient(sid, out, frozen_root, n_shuffles):
    frozen_path = frozen_root/'per_subject'/f'{sid}.json'
    frozen = json.loads(frozen_path.read_text())
    dataset, subject = sid.split('_', 1)
    events = load_subject_propagation_events(_subject_dir(dataset,subject))
    ranks, bools, blocks = events['ranks'], events['bools'], events['block_ids']
    assert list(map(str,events['channel_names'])) == frozen['channel_names']
    assert len(blocks) == frozen['n_events']
    assert jsonable(events['block_boundaries']) == frozen['source_block_inventory']
    fold_rows, arrays = [], {}
    cache = np.load(frozen_root/'per_subject'/f'{sid}_heldout.npz')
    for f, old in enumerate(frozen['folds']):
        assert old['fold'] == f
        train_blocks, test_blocks = old['train_blocks'], old['test_blocks']
        assert not set(train_blocks)&set(test_blocks)
        test = np.flatnonzero(np.isin(blocks, test_blocks))
        np.testing.assert_array_equal(test, cache[f'f{f}_test_indices'])
        assert np.isin(blocks, train_blocks).sum() == old['n_train']
        templates = {m: np.asarray(old['models'][m]['templates'], float) for m in METHODS}
        truth, mask, predicted, eligible, coverage = prepare_reconstruction(ranks[:,test],bools[:,test],templates)
        evaluated = closest_template_error(truth,mask,predicted)
        null = np.zeros((2,n_shuffles),float)
        if f<2:
            generators = [np.random.default_rng(SEED+zlib.crc32(sid.encode())+f*10000+d) for d in range(n_shuffles)]
            for start in range(0,len(truth),8192):
                selection = slice(start,start+8192)
                for d,rng in enumerate(generators):
                    shuffled = shuffle_participating_orders(truth[selection],mask[selection],rng)
                    null[:,d] += closest_template_error(shuffled,mask[selection],predicted[:,selection])['mae'].sum(axis=1)
            null /= len(truth)
        row = dict(fold=f, split=old['split'], n_train=old['n_train'],
                   train_blocks=train_blocks,test_blocks=test_blocks,coverage=coverage,
                   metrics={m: dict(mae=float(evaluated['mae'][i].mean()),
                                    mse=float(evaluated['mse'][i].mean()),
                                    assignment_counts=np.bincount(evaluated['assignments'][i],minlength=2))
                            for i,m in enumerate(METHODS)})
        if f<2:
            row['null_mae_draws']={m:null[i] for i,m in enumerate(METHODS)}
        fold_rows.append(row)
        arrays[f'f{f}_test_indices']=test
        arrays[f'f{f}_eligible']=eligible
        arrays[f'f{f}_test_blocks']=blocks[test]
        for i,m in enumerate(METHODS):
            for metric in ('mae','mse','assignments'):
                arrays[f'f{f}_{m}_{metric}']=evaluated[metric][i]
        print(f'{sid} fold {f}: {len(truth)}/{len(test)} sequences; MAE '+
              ' / '.join(f'{row["metrics"][m]["mae"]:.6f}' for m in METHODS),flush=True)
    cache.close()
    record=dict(subject_id=sid,dataset=dataset,n_events=len(blocks),n_channels=len(frozen['channel_names']),
                n_evaluated=sum(r['coverage']['n_evaluated'] for r in fold_rows[:2]),
                frozen_model_path=str(frozen_path),frozen_model_sha256=hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
                source_inventory_verified=True,folds=fold_rows,n_shuffles=n_shuffles)
    for m in METHODS:
        for metric in ('mae','mse'):
            record[f'{m}_{metric}']=float(np.mean([r['metrics'][m][metric] for r in fold_rows[:2]]))
        record[f'{m}_event_weighted_mae']=float(np.average([r['metrics'][m]['mae'] for r in fold_rows[:2]],
                                                   weights=[r['coverage']['n_evaluated'] for r in fold_rows[:2]]))
        record[f'{m}_chronological_mae']=fold_rows[2]['metrics'][m]['mae']
        record[f'{m}_null_mae_draws']=np.mean([r['null_mae_draws'][m] for r in fold_rows[:2]],axis=0)
        record[f'{m}_null_mae']=float(record[f'{m}_null_mae_draws'].mean())
    out.joinpath('per_subject').mkdir(exist_ok=True)
    np.savez_compressed(out/'per_subject'/f'{sid}_heldout.npz',**arrays)
    write_json(out/'per_subject'/f'{sid}.json',record)
    return record


def summarize(records, failures, subjects, n_shuffles, out):
    records.sort(key=lambda r:subjects.index(r['subject_id']))
    metrics={key:paired_summary([r[f'rank_sequence_{key}'] for r in records],
                               [r[f'rank_trajectory_{key}'] for r in records])
             for key in ('mae','mse','event_weighted_mae','chronological_mae')}
    comparisons={'sequence_vs_trajectory':dict(metrics['mae'])}
    for m in METHODS:
        comparisons[f'{m}_null_vs_real']=paired_summary([r[f'{m}_null_mae'] for r in records],
                                                        [r[f'{m}_mae'] for r in records])
    ordered=sorted(comparisons,key=lambda k:comparisons[k]['wilcoxon_two_sided_p'])
    adjusted=np.maximum.accumulate([min(1.,(len(ordered)-i)*comparisons[k]['wilcoxon_two_sided_p']) for i,k in enumerate(ordered)])
    for k,p in zip(ordered,adjusted):
        comparisons[k].update(holm_p=float(p),stars=stars(p))
    summary=dict(status='complete' if not failures else 'partial',primary='normalized_rank_mae',
                 task='complete held-out sequence reconstruction with nearest frozen template',
                 n_requested=len(subjects),n_complete=len(records),failures=failures,n_shuffles=n_shuffles,
                 n_events=sum(r['n_events'] for r in records),n_evaluated=sum(r['n_evaluated'] for r in records),
                 statistical_unit='patient',metrics=metrics,comparisons=comparisons,
                 by_dataset={d:paired_summary([r['rank_sequence_mae'] for r in records if r['dataset']==d],
                                              [r['rank_trajectory_mae'] for r in records if r['dataset']==d])
                             for d in sorted({r['dataset'] for r in records})},
                 human_visual_acceptance='PENDING_AUTHOR_VISUAL_REVIEW')
    write_json(out/'summary.json',summary)
    keys=[k for k,v in records[0].items() if isinstance(v,(str,int,float,bool))]
    with (out/'patients.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=keys,extrasaction='ignore')
        writer.writeheader();writer.writerows(records)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--frozen',type=Path,default=FROZEN)
    parser.add_argument('--subjects',nargs='+')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--n-shuffles',type=int,default=100)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.n_shuffles<1: raise ValueError('at least one null draw is required')
    subjects=args.subjects or json.loads((args.frozen/'run_manifest.json').read_text())['subjects']
    args.output.mkdir(parents=True,exist_ok=True)
    sources=[Path(__file__),ROOT/'src/interictal_rank_holdout.py',ROOT/'docs/fig2b_rank_reconstruction_holdout.md']
    snap=args.output/'source_snapshot';snap.mkdir(exist_ok=True)
    hashes={}
    for path in sources:
        data=path.read_bytes();(snap/path.name).write_bytes(data)
        hashes[str(path.relative_to(ROOT))]=hashlib.sha256(data).hexdigest()
    write_json(args.output/'run_manifest.json',dict(subjects=subjects,seed=SEED,n_shuffles=args.n_shuffles,
               frozen_training_root=str(args.frozen),source_sha256=hashes,
               null='within held-out event rank-to-participating-channel permutation; both methods shared draws'))
    records,failures,pending=[],[],[]
    for sid in subjects:
        path=args.output/'per_subject'/f'{sid}.json'
        if args.resume and path.exists():
            r=json.loads(path.read_text())
            if r['n_shuffles']==args.n_shuffles:
                records.append(r);continue
        pending.append(sid)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_patient,sid,str(args.output),str(args.frozen),args.n_shuffles):sid for sid in pending}
        for future in as_completed(futures):
            sid=futures[future]
            try: records.append(future.result())
            except Exception as exc:
                failures.append(dict(subject_id=sid,error=repr(exc)))
                print(f'FAILED {sid}: {exc!r}',flush=True)
    if not records: raise RuntimeError(failures)
    summary=summarize(records,failures,subjects,args.n_shuffles,args.output)
    print(json.dumps(jsonable(summary),indent=2),flush=True)
    if failures: raise SystemExit(1)


if __name__=='__main__':
    main()
