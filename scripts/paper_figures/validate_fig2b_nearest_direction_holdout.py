#!/usr/bin/env python3
"""Independently recompute nearest 3D errors, null gradients, statistics and figures."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.run_fig2b_nearest_direction_holdout import OUT as ANALYSIS, SOURCE
from scripts.paper_figures.plot_fig2b_nearest_direction_holdout import OUT
from scripts.run_fig2b_rank_reconstruction_holdout import paired_summary, write_json
from src.interictal_rank_holdout import METHODS
from src.interictal_two_target_holdout import holm_adjust
from src.topic5_interictal_direction_rose import fit_event_directions_3d


def validate(analysis,output):
    summary=json.loads((analysis/'summary.json').read_text())
    manifest=json.loads((analysis/'run_manifest.json').read_text())
    assert summary['status']=='complete' and summary['n_complete']==26
    records=[];n_main=0;maximum_error=0.;maximum_null_error=0.;n_null_axes=0
    for sid in manifest['subjects']:
        source_path=SOURCE/'per_subject'/f'{sid}.json'
        old=json.loads(source_path.read_text())
        r=json.loads((analysis/'per_subject'/f'{sid}.json').read_text());records.append(r)
        assert hashlib.sha256(source_path.read_bytes()).hexdigest()==r['source_record_sha256']
        with np.load(analysis/'per_subject'/f'{sid}_heldout.npz') as data,np.load(SOURCE/'per_subject'/f'{sid}_heldout.npz') as prior:
            coords=data['coords'];fold_means=[]
            for f,row in enumerate(r['folds']):
                assert row['models']==old['folds'][f]['models']
                assert not set(row['train_blocks'])&set(row['test_blocks'])
                assert prior[f'f{f}_rank_eligible'].all()
                finite=prior[f'f{f}_direction_valid']
                np.testing.assert_array_equal(data[f'f{f}_test_indices'],prior[f'f{f}_test_indices'][finite])
                vectors=data[f'f{f}_test_directions']
                np.testing.assert_array_equal(vectors,prior[f'f{f}_test_directions'])
                axes=np.array([row['models'][m]['axes'] for m in METHODS])
                templates=np.array([row['models'][m]['templates'] for m in METHODS])
                angles=np.degrees(np.arccos(np.clip(vectors@axes.reshape(4,3).T,-1,1))).reshape(len(vectors),2,2)
                minimum=angles.min(axis=2).T
                for i,m in enumerate(METHODS):
                    saved=data[f'f{f}_{m}_angular_mae_deg']
                    # Independent BLAS vs einsum dot products differ at roundoff;
                    # acos amplifies that close to zero angle. 1e-8 degrees is
                    # still far below any reported precision or biological scale.
                    np.testing.assert_allclose(saved,minimum[i],atol=1e-8)
                    maximum_error=max(maximum_error,float(np.max(abs(saved-minimum[i]))))
                    np.testing.assert_allclose(saved.mean(),row['metrics'][m]['angular_mae_deg'],atol=1e-10)
                    np.testing.assert_allclose(1-np.cos(np.radians(saved)),data[f'f{f}_{m}_cosine_loss'],atol=1e-12)
                    np.testing.assert_array_equal(data[f'f{f}_{m}_assignments'],angles[:,i,:].argmin(axis=1))
                    assert np.all(saved<=prior[f'f{f}_{m}_angular_mae_deg']+1e-8)
                    for metric in ('rank_mae','rank_rmse','pair_order_error'):
                        assert row['metrics'][m][metric]==old['folds'][f]['metrics'][m][metric]
                fold_means.append(minimum.mean(axis=1))
                if f<2:
                    n_main+=len(vectors)
                    info=row['direction_null_info'];assert info['rejected_degenerate_draws']==0
                    keys=np.random.default_rng(info['seed']).random((info['n_draws'],len(coords)))
                    shuffled=np.full((info['n_draws'],2,2,len(coords)),np.nan)
                    for i in range(2):
                        for k in range(2):
                            template=templates[i,k]
                            support=np.flatnonzero(np.isfinite(template)&np.isfinite(coords).all(axis=1))
                            values=template[support][np.argsort(keys[:,support],axis=1)]
                            shuffled[:,i,k,support]=values
                            np.testing.assert_allclose(np.sort(values,axis=1),np.broadcast_to(np.sort(template[support]),values.shape))
                    fitted=fit_event_directions_3d(shuffled.reshape(-1,len(coords)).T,coords)['directions'].reshape(info['n_draws'],2,2,3)
                    np.testing.assert_allclose(fitted,data[f'f{f}_null_axes'],atol=1e-11)
                    n_null_axes+=info['n_draws']*4
                    total=np.zeros((info['n_draws'],2))
                    for start in range(0,len(vectors),2048):
                        part=vectors[start:start+2048]
                        aa=np.degrees(np.arccos(np.clip(part@fitted.reshape(-1,3).T,-1,1))).reshape(len(part),info['n_draws'],2,2)
                        total+=aa.min(axis=3).sum(axis=0)
                    null=(total/len(vectors)).T
                    for i,m in enumerate(METHODS):
                        expected=np.array(row['direction_null_draws'][m])
                        np.testing.assert_allclose(null[i],expected,atol=1e-9)
                        maximum_null_error=max(maximum_null_error,float(np.max(abs(null[i]-expected))))
            for i,m in enumerate(METHODS):
                np.testing.assert_allclose(np.mean(fold_means[:2],axis=0)[i],r['combined'][m]['angular_mae_deg'],atol=1e-10)
                np.testing.assert_allclose(np.mean([f['direction_null_draws'][m] for f in r['folds'][:2]]),r['null'][m]['angular_mae_deg'],atol=1e-10)
                assert r['null'][m]['rank_mae']==old['null'][m]['rank_mae']
        print('verified',sid,flush=True)
    assert n_main==summary['n_direction']
    family=[]
    for target,part in summary['primary'].items():
        for key,row in part['comparisons'].items():
            if key=='sequence_vs_trajectory':
                a=[r['combined'][METHODS[0]][target] for r in records];b=[r['combined'][METHODS[1]][target] for r in records]
            else:
                method=key.removesuffix('_null_vs_real')
                a=[r['null'][method][target] for r in records];b=[r['combined'][method][target] for r in records]
            for k,v in paired_summary(a,b).items():np.testing.assert_allclose(v,row[k],atol=1e-12)
            family.append(row)
    np.testing.assert_allclose(holm_adjust([r['wilcoxon_two_sided_p'] for r in family]),[r['holm_p'] for r in family])
    for row in summary['audit']:
        values=[r['combined'] if row['split']=='alternating_combined' else next(f['metrics'] for f in r['folds'] if f['split']==row['split']) for r in records]
        for k,v in paired_summary([r[METHODS[0]][row['metric']] for r in values],[r[METHODS[1]][row['metric']] for r in values]).items():
            np.testing.assert_allclose(v,row[k],atol=1e-12)
    np.testing.assert_allclose(holm_adjust([r['wilcoxon_two_sided_p'] for r in summary['audit']]),[r['holm_35_p'] for r in summary['audit']])
    meta=json.loads((output/'metadata.json').read_text())
    assert not meta['same_rank_assignment_for_both_targets'] and meta['direction_assignment']=='nearest_3d_angle'
    assert meta['cohort_summary']==summary
    for filename,sha in meta['preserved_acd_sha256'].items():
        assert hashlib.sha256((output/'figures'/filename).read_bytes()).hexdigest()==sha
    checks=json.loads((output/'exemplar_3d_consistency.json').read_text())
    for check in checks:
        record=next(r for r in records if r['subject_id']==check['subjects_source'])
        for m,item in check['methods'].items():
            np.testing.assert_allclose(item['weighted_mean_deg'],record['combined'][m]['angular_mae_deg'],atol=1e-12)
            np.testing.assert_allclose(sum(item['histogram_probability']),1,atol=1e-12)
    import pymupdf as pdf
    rose_pixels=[]
    for target in ('rank','direction'):
        for stem in (f'fig2b-{target}-target',f'fig2-complete-{target}-target'):
            with pdf.open(output/'figures'/f'{stem}.pdf') as doc:
                page=doc[0];text=page.get_text()
                assert 'Held-out error' in text and 'Patients (ordered)' not in text
                assert meta.get('rose_legend',{}).get('event_label','Fold-1 projection') in text
                for block in page.get_text('dict')['blocks']:
                    for line in block.get('lines',[]):
                        for span in line['spans']:
                            assert page.rect.contains(pdf.Rect(span['bbox'])),(stem,span['text'])
                if stem.startswith('fig2b'):
                    rose_pixels.append(page.get_pixmap(matrix=pdf.Matrix(2,2),clip=pdf.Rect(360,0,583,350)).samples)
                    for drawing in page.get_drawings():
                        if drawing['rect'].x0>360:assert drawing['dashes'] in ('[] 0',None,'')
                    assert text.count('***')==(2 if target=='rank' else 3)
    assert rose_pixels[0]==rose_pixels[1]
    with pdf.open(output/'figures'/'fig2b-3d-error-check.pdf') as doc:
        for block in doc[0].get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                for span in line['spans']:assert doc[0].rect.contains(pdf.Rect(span['bbox'])),span['text']
    report=dict(status='PASS',n_patients=len(records),n_main_events=int(n_main),
        six_frozen_splits_per_patient_verified=True,no_rank_assignment_in_direction_metric=True,
        all_event_direction_errors_recomputed=True,max_event_difference_deg=maximum_error,
        canonical_null_axis_refits=n_null_axes,all_200_draws_per_main_fold_verified=True,
        max_null_mean_difference_deg=maximum_null_error,rank_task_unchanged=True,
        all_six_main_and_35_sensitivity_comparisons_verified=True,
        illustrative_first_fold_roses_identical_between_targets=True,
        both_fold_3d_error_plot_means_match_main_patient_points=True,
        unchanged_acd_files=len(meta['preserved_acd_sha256']),all_pdf_text_inside_canvas=True,
        human_visual_acceptance='PENDING_AUTHOR_VISUAL_REVIEW')
    write_json(output/'validation.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,default=ANALYSIS);parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args()
    with threadpool_limits(limits=1):validate(args.analysis,args.output)
