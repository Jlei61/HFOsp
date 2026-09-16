#!/usr/bin/env python3
"""Freeze the development evaluator and test patient-only perturbation controls."""
from pathlib import Path
import json
import pickle
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_topic4_xy_research import training_contract, read, write, sha
from src.topic4_interictal_pilot_evaluation import PilotEvaluator

OUT=ROOT/'results/topic4_sef_hfo/multidimensional_interictal_pilot_round1'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'evaluation_manifest.json').exists():
        manifest=read(OUT/'evaluation_manifest.json')
        if sha(OUT/'evaluator.pkl') != manifest['evaluator_sha256']:
            raise RuntimeError('frozen evaluator changed')
        print(manifest['status']);return
    training,_=training_contract()
    xy_path=ROOT/'results/topic4_sef_hfo/vth_dual_core_xy_research/direction_objective_v2.json'
    ev=PilotEvaluator(training['onsets_ms'],training['block_ids'],read(xy_path)['contact_xy_mm'],training['groups'])
    print('mode discovery',ev.k_scan,flush=True)
    rng=np.random.default_rng(2026090620)
    probe=ev.patient[ev.index['PROBE']]
    labels,_,_=ev.classify(probe)
    # Whole PROBE blocks provide a genuine block identity for the operational
    # mode-presence diagnostic. PROBE is development, never called final test.
    units=ev.blocks[ev.index['PROBE']]
    baseline=ev.metrics(probe,units)
    controls={'patient_disjoint_blocks':baseline}
    first=np.nanmin(probe,axis=1)[:,None]
    variants={'time_stretch_1_5':first+1.5*(probe-first),
              'patient_resolution_noise_0_5ms':probe+rng.normal(0,.5,probe.shape)}
    shuffle=probe.copy()
    for row in shuffle:
        finite=np.flatnonzero(np.isfinite(row));row[finite]=rng.permutation(row[finite])
    variants['order_shuffle_same_mask']=shuffle
    for name,t in variants.items():
        controls[name]=ev.metrics(t,units)
    if ev.modes_stable:
        mask=labels==0
        controls['delete_other_modes']=ev.metrics(probe[mask],units[mask])
        # Duplicate only for an explicit synthetic mixture-control assay.
        idx=np.r_[np.arange(len(probe)),np.tile(np.flatnonzero(mask),3)]
        controls['change_mixture_only']=ev.metrics(probe[idx],units[idx])
    abnormal=first+8*(probe-first)
    controls['inject_50pct_time_outliers']=ev.metrics(np.concatenate([probe,abnormal]),np.tile(units,2))
    collapsed=np.empty_like(probe)
    for m in range(ev.k):
        ix=np.flatnonzero(labels==m)
        if len(ix):
            # One actual event per mode, repeated only in this collapse control.
            collapsed[ix]=probe[ix[0]]
    controls['collapse_each_mode_to_one_event']=ev.metrics(collapsed,units)
    controls['empty']=ev.metrics(np.empty((0,probe.shape[1])))
    controls['unreadable']=ev.metrics(np.full((20,probe.shape[1]),np.nan))
    checks={
        'stable_modes_discovered':ev.modes_stable,
        'patient_self_support_above_50pct':baseline['supported_fraction']>.5,
        'patient_self_unsupported_below_20pct':baseline['unsupported_fraction']<.2,
        'order_shuffle_worsens_rank_kernel':controls['order_shuffle_same_mask']['kernel_distances']['rank_space']>baseline['kernel_distances']['rank_space'],
        'shuffle_preserves_participation_kernel':abs(controls['order_shuffle_same_mask']['kernel_distances']['support']-baseline['kernel_distances']['support'])<1e-10,
        'time_stretch_worsens_timing_kernel':controls['time_stretch_1_5']['kernel_distances']['timing_space']>baseline['kernel_distances']['timing_space'],
        'outlier_injection_reduces_support':controls['inject_50pct_time_outliers']['supported_fraction']<baseline['supported_fraction'],
        'empty_not_estimable':controls['empty']['status']=='NOT_ESTIMABLE_NO_EVENTS',
        'unreadable_not_supported':controls['unreadable']['supported_fraction']==0,
    }
    if ev.modes_stable:
        checks['delete_mode_reduces_coverage']=controls['delete_other_modes']['operational_mode_coverage']<baseline['operational_mode_coverage']
    path=OUT/'evaluator.pkl'
    with open(path,'wb') as f:pickle.dump(ev,f,protocol=pickle.HIGHEST_PROTOCOL)
    write(OUT/'patient_evaluation_controls.json',{'checks':checks,'controls':controls,
        'scope':'development directional sanity checks; not independent power or a calibrated final acceptance rule',
        'qualification_remaining':['joint model acceptance calibration','independent upstream provenance for final test',
            'formal reasonable-noise/collapse/mixture power','observer sensitivity','known-geometry retrieval with new networks']})
    manifest={**ev.manifest(),'status':'DEVELOPMENT_PILOT_SANITY_PASS' if all(checks.values()) else 'DEVELOPMENT_PILOT_SANITY_FAILED',
        'pilot_dispatch_allowed':all(checks.values()),'full_evaluator_qualified':False,
        'patient_training_path':str(training['path']),'patient_training_sha256':training['sha256'],
        'patient_xy_path':str(xy_path),'patient_xy_sha256':sha(xy_path),
        'evaluator_sha256':sha(path),'controls_sha256':sha(OUT/'patient_evaluation_controls.json'),
        'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'src/topic4_interictal_pilot_evaluation.py']},
        'heldout_patient_opened':False,'fig5_released':False}
    write(OUT/'evaluation_manifest.json',manifest)
    print(json.dumps({'status':manifest['status'],'checks':checks,'K':ev.k}),flush=True)


if __name__=='__main__':main()
