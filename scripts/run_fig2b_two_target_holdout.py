#!/usr/bin/env python3
"""Matched rank and direction targets, alternative errors and block holdouts."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
import hashlib
import json
from pathlib import Path
import sys
import zlib

import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.plot_topic5_interictal_template_direction_rose import _subject_dir
from scripts.run_fig2b_rank_reconstruction_holdout import FROZEN,OUT as RANK_SOURCE,SEED,write_json,jsonable,paired_summary,stars
from src.interictal_propagation import load_subject_propagation_events
from src.interictal_direction_distribution_holdout import fit_template_model
from src.interictal_rank_holdout import METHODS,normalized_order,prepare_reconstruction
from src.interictal_two_target_holdout import rank_error_suite,direction_error_suite,block_direction_null,holm_adjust
from src.lagpat_rank_audit import mask_phantom_ranks
from src.seeg_coord_loader import load_subject_coords
from src.topic5_interictal_direction_rose import fit_event_directions_3d

OUT=ROOT/'results/topic1/rank_direction_holdout_20261022'
METRICS=('rank_mae','rank_rmse','pair_order_error','angular_mae_deg','cosine_loss')
SPLITS=('alternating_a','alternating_b','chronological','random_0','random_1','random_2')


def align_training_templates(models):
    """Align the two labels by training template order alone, never test outcomes."""
    a=np.asarray(models[METHODS[0]]['templates'],float)
    b=np.asarray(models[METHODS[1]]['templates'],float)
    common=np.isfinite(a).all(axis=0)&np.isfinite(b).all(axis=0)
    if common.sum()<2: raise ValueError('insufficient common template channels')
    aa=normalized_order(a[:,common],np.ones((2,common.sum()),bool))
    bb=normalized_order(b[:,common],np.ones((2,common.sum()),bool))
    flip=bool(np.mean(abs(aa-bb[::-1]))<np.mean(abs(aa-bb)))
    if flip:
        for key in ('templates','axes','train_counts'):
            models[METHODS[1]][key]=np.asarray(models[METHODS[1]][key])[::-1].copy()
    return flip


def run_patient(sid,out,n_shuffles):
    with threadpool_limits(limits=1):
        return _run_patient(sid,Path(out),n_shuffles)


def _run_patient(sid,out,n_shuffles):
    old_path=FROZEN/'per_subject'/f'{sid}.json'
    old=json.loads(old_path.read_text())
    previous_rank=json.loads((RANK_SOURCE/'per_subject'/f'{sid}.json').read_text())
    assert previous_rank['n_shuffles']==n_shuffles
    dataset,subject=sid.split('_',1)
    events=load_subject_propagation_events(_subject_dir(dataset,subject))
    ranks,bools,blocks=events['ranks'],events['bools'],events['block_ids']
    names=list(map(str,events['channel_names']))
    assert names==old['channel_names'] and len(blocks)==old['n_events']
    assert jsonable(events['block_boundaries'])==old['source_block_inventory']
    coords=np.asarray(load_subject_coords(dataset,subject,names).coords_array_in_requested_order)
    directions=np.full((len(blocks),3),np.nan)
    with np.load(FROZEN/'per_subject'/f'{sid}_heldout.npz') as cache:
        for f in (0,1): directions[cache[f'f{f}_test_indices']]=cache[f'f{f}_test_directions']
    sample=np.unique(np.linspace(0,len(blocks)-1,min(128,len(blocks)),dtype=int))
    checked=fit_event_directions_3d(mask_phantom_ranks(ranks[:,sample],bools[:,sample],normalize=True),coords)['directions']
    np.testing.assert_allclose(checked,directions[sample],atol=1e-12,equal_nan=True)
    splits=[]
    for f in range(3):
        splits.append((np.asarray(old['folds'][f]['train_blocks']),np.asarray(old['folds'][f]['test_blocks'])))
    unique=np.unique(blocks)
    for seed in range(3):
        shuffled=np.random.default_rng(SEED+seed+zlib.crc32(sid.encode())).permutation(unique)
        splits.append((np.sort(shuffled[:len(unique)//2]),np.sort(shuffled[len(unique)//2:])))
    rows,arrays=[],{}
    for f,(train_blocks,test_blocks) in enumerate(splits):
        assert not set(train_blocks)&set(test_blocks)
        train=np.flatnonzero(np.isin(blocks,train_blocks));test=np.flatnonzero(np.isin(blocks,test_blocks))
        if f<3:
            models={m:{k:np.asarray(v,float) if isinstance(v,list) else v for k,v in old['folds'][f]['models'][m].items()} for m in METHODS}
        else:
            models={m:fit_template_model(ranks[:,train],bools[:,train],directions[train],coords,method=m) for m in METHODS}
            for model in models.values(): model.pop('labels',None)
        flipped=align_training_templates(models)
        truth,mask,pred,eligible,coverage=prepare_reconstruction(ranks[:,test],bools[:,test],{m:models[m]['templates'] for m in METHODS})
        evaluated=rank_error_suite(truth,mask,pred)
        assignments=evaluated.pop('assignments')
        finite=np.isfinite(directions[test][eligible]).all(axis=1)
        axes=np.asarray([models[m]['axes'] for m in METHODS])
        direct=direction_error_suite(directions[test][eligible][finite],axes,assignments[:,finite])
        all_angles=direct.pop('all_angles');evaluated.update(direct)
        idx=np.arange(finite.sum())
        diagnostics=dict(
            equal_rank_loss_fraction=float(np.mean(abs(evaluated['rank_mae'][0]-evaluated['rank_mae'][1])<1e-12)),
            identical_template_orders_fraction=float(np.mean(np.all(abs(pred[0]-pred[1])<1e-12,axis=(1,2)))),
            template_order_mae=float(np.mean(np.sum(abs(pred[0]-pred[1]),axis=2)/mask.sum(axis=1)[:,None])),
            rank_assignment_agreement=float(np.mean(assignments[0]==assignments[1])),
            mean_template_axis_shift_deg=float(np.degrees(np.arccos(np.clip(np.sum(axes[0]*axes[1],axis=1),-1,1))).mean()),
            axis_only_angular_mae_deg=float(all_angles[1,idx,assignments[0,finite]].mean()),
            assignment_only_angular_mae_deg=float(all_angles[0,idx,assignments[1,finite]].mean()))
        row=dict(fold=f,split=SPLITS[f],n_train=len(train),train_blocks=train_blocks,test_blocks=test_blocks,
                 coverage=coverage,n_direction=int(finite.sum()),models=models,aligned_trajectory_labels_flipped=flipped,
                 metrics={m:{k:float(v[i].mean()) for k,v in evaluated.items()} for i,m in enumerate(METHODS)},
                 diagnostics=diagnostics)
        if f<3:
            for m in METHODS:
                np.testing.assert_allclose(row['metrics'][m]['rank_mae'],previous_rank['folds'][f]['metrics'][m]['mae'],atol=1e-12)
        if f<2:
            null,null_info=block_direction_null(all_angles,assignments[:,finite],blocks[test][eligible][finite],n_shuffles,
                                                SEED+f*10000+zlib.crc32(sid.encode()))
            row['direction_null_draws']={m:null[i] for i,m in enumerate(METHODS)}
            row['direction_null_info']=null_info
        for i,m in enumerate(METHODS):
            for key,values in evaluated.items(): arrays[f'f{f}_{m}_{key}']=values[i]
            arrays[f'f{f}_{m}_assignments']=assignments[i]
        arrays[f'f{f}_test_indices']=test
        arrays[f'f{f}_rank_eligible']=eligible
        arrays[f'f{f}_direction_valid']=finite
        arrays[f'f{f}_test_directions']=directions[test][eligible][finite]
        rows.append(row)
        print(f'{sid} {SPLITS[f]}: MAE '+ ' / '.join(f'{row["metrics"][m]["rank_mae"]:.5f}' for m in METHODS)+
              '; angle '+ ' / '.join(f'{row["metrics"][m]["angular_mae_deg"]:.3f}' for m in METHODS),flush=True)
    record=dict(subject_id=sid,dataset=dataset,n_events=len(blocks),n_rank=sum(r['coverage']['n_evaluated'] for r in rows[:2]),
                n_direction=sum(r['n_direction'] for r in rows[:2]),n_shuffles=n_shuffles,folds=rows,
                frozen_model_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),
                raw_inventory_verified=True,cached_direction_sample_verified=len(sample))
    record['combined']={m:{k:float(np.mean([r['metrics'][m][k] for r in rows[:2]])) for k in METRICS} for m in METHODS}
    record['null']={m:dict(rank_mae=previous_rank[f'{m}_null_mae'],
                           angular_mae_deg=float(np.mean([r['direction_null_draws'][m] for r in rows[:2]]))) for m in METHODS}
    record['diagnostics']={k:float(np.mean([r['diagnostics'][k] for r in rows[:2]])) for k in rows[0]['diagnostics']}
    out.joinpath('per_subject').mkdir(exist_ok=True)
    np.savez_compressed(out/'per_subject'/f'{sid}_heldout.npz',**arrays)
    write_json(out/'per_subject'/f'{sid}.json',record)
    return record


def summarize(records,failures,subjects,out):
    records.sort(key=lambda r:subjects.index(r['subject_id']))
    audit=[]
    for f,split in [(-1,'alternating_combined')]+list(enumerate(SPLITS)):
        for metric in METRICS:
            values={m:[r['combined'][m][metric] if f<0 else r['folds'][f]['metrics'][m][metric] for r in records] for m in METHODS}
            audit.append(dict(split=split,metric=metric,**paired_summary(values[METHODS[0]],values[METHODS[1]])))
    for row,p in zip(audit,holm_adjust([r['wilcoxon_two_sided_p'] for r in audit])):
        row.update(holm_35_p=float(p),stars=stars(p))
    primary={}
    for target in ('rank_mae','angular_mae_deg'):
        comparisons={'sequence_vs_trajectory':paired_summary([r['combined'][METHODS[0]][target] for r in records],
                                                             [r['combined'][METHODS[1]][target] for r in records])}
        for m in METHODS:
            comparisons[f'{m}_null_vs_real']=paired_summary([r['null'][m][target] for r in records],
                                                           [r['combined'][m][target] for r in records])
        primary[target]=dict(comparisons=comparisons)
    family=[row for d in primary.values() for row in d['comparisons'].values()]
    for row,p in zip(family,holm_adjust([r['wilcoxon_two_sided_p'] for r in family])):
        row.update(holm_p=float(p),stars=stars(p))
    diagnostics={key:dict(median=float(np.median([r['diagnostics'][key] for r in records])),
                           minimum=float(min(r['diagnostics'][key] for r in records)),
                           maximum=float(max(r['diagnostics'][key] for r in records))) for key in records[0]['diagnostics']}
    for key in ('axis_only_angular_mae_deg','assignment_only_angular_mae_deg'):
        delta=[r['combined'][METHODS[0]]['angular_mae_deg']-r['diagnostics'][key] for r in records]
        diagnostics[key]['median_improvement_from_sequence']=float(np.median(delta))
    summary=dict(status='complete' if not failures else 'partial',n_requested=len(subjects),n_complete=len(records),failures=failures,
                 n_events=sum(r['n_events'] for r in records),n_rank=sum(r['n_rank'] for r in records),
                 n_direction=sum(r['n_direction'] for r in records),primary=primary,audit=audit,diagnostics=diagnostics,
                 followup_status='EXPLORATORY_AFTER_RANK_NULL_RESULT',statistical_unit='patient',
                 main_family_size=6,audit_family_size=len(audit),human_visual_acceptance='PENDING_AUTHOR_VISUAL_REVIEW')
    write_json(out/'summary.json',summary)
    with (out/'metric_holdout_comparisons.csv').open('w',newline='') as handle:
        w=csv.DictWriter(handle,fieldnames=audit[0].keys());w.writeheader();w.writerows(audit)
    patient_rows=[]
    for r in records:
        patient_rows.append(dict(subject_id=r['subject_id'],dataset=r['dataset'],n_events=r['n_events'],n_rank=r['n_rank'],n_direction=r['n_direction'],
                                **{f'{m}_{k}':v for m,d in r['combined'].items() for k,v in d.items()},
                                **{f'{m}_null_{k}':v for m,d in r['null'].items() for k,v in d.items()},**r['diagnostics']))
    with (out/'patients.csv').open('w',newline='') as handle:
        w=csv.DictWriter(handle,fieldnames=patient_rows[0].keys());w.writeheader();w.writerows(patient_rows)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=OUT);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--subjects',nargs='+');p.add_argument('--n-shuffles',type=int,default=100);p.add_argument('--resume',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    subjects=args.subjects or json.loads((FROZEN/'run_manifest.json').read_text())['subjects']
    sources=[Path(__file__),ROOT/'src/interictal_two_target_holdout.py',ROOT/'src/interictal_rank_holdout.py',
             ROOT/'src/interictal_direction_distribution_holdout.py',ROOT/'scripts/run_timing_plus_space_rank_readback.py',
             ROOT/'docs/fig2b_two_target_holdout.md']
    snap=args.output/'source_snapshot';snap.mkdir(exist_ok=True);hashes={}
    for source in sources:
        data=source.read_bytes();(snap/source.name).write_bytes(data);hashes[str(source.relative_to(ROOT))]=hashlib.sha256(data).hexdigest()
    write_json(args.output/'run_manifest.json',dict(subjects=subjects,seed=SEED,random_seeds=[0,1,2],n_shuffles=args.n_shuffles,
               source_sha256=hashes,frozen_training_root=str(FROZEN),rank_null_source=str(RANK_SOURCE),
               main_direction_assignment='nearest normalized rank MAE template; same as rank task'))
    records=[];pending=[];failures=[]
    for sid in subjects:
        path=args.output/'per_subject'/f'{sid}.json'
        if args.resume and path.exists(): records.append(json.loads(path.read_text()))
        else: pending.append(sid)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_patient,sid,str(args.output),args.n_shuffles):sid for sid in pending}
        for future in as_completed(futures):
            try: records.append(future.result())
            except Exception as exc:
                failures.append(dict(subject_id=futures[future],error=repr(exc)))
                print('FAILED',futures[future],repr(exc),flush=True)
    if not records: raise RuntimeError(failures)
    summary=summarize(records,failures,subjects,args.output)
    print(json.dumps(jsonable(summary),indent=2),flush=True)
    if failures: raise SystemExit(1)


if __name__=='__main__': main()
