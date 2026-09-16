#!/usr/bin/env python3
"""Audit applied VTH maps and distinguish historical clipping from signed maps."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.stats import spearmanr
from src.topic4_core_field import sample_core_quantiles,core_thresholds
from scripts import run_topic4_joint_xy_replicated_search as run


def main():
    sources=[run.OUT/'baseline_scores.json',run.OUT/'rounds/001/scores.json']
    candidates=[x for p in sources for x in run.read(p)['candidates']]
    transition=ROOT/'config/topic4_rev22_dci_transition_execution.json';tc=run.read(transition)
    stage=ROOT/tc['inputs']['stage_config']['path']
    if not stage.exists():stage=run.v1.base.ART/tc['inputs']['stage_config']['path']
    if run.sha(stage)!=tc['inputs']['stage_config']['sha256']:raise RuntimeError('stage changed')
    cfg=run.read(stage);engine=cfg['engine'];seed=cfg['quantile_seed'];expected=None;rows=[]
    out=run.OUT/'threshold_realization_audit';out.mkdir(exist_ok=True)
    for row in candidates:
        center=np.asarray(row['candidate']['node_field']['centers_mm'])
        for u in row['units']:
            path=Path(u['worker_path']);meta=run.read(path)
            if run.sha(path)!=u['worker_sha256'] or run.sha(meta['arrays']['path'])!=meta['arrays']['sha256']:raise RuntimeError('trajectory changed')
            with np.load(meta['arrays']['path']) as z:
                h=z['h'].astype(float);delta=z['delta_vtheta'].astype(float);pos=z['positions_E'].astype(float)
                if np.any(z['edge_coefficients']!=0):raise RuntimeError('non-VTH mechanism changed')
            if expected is None:expected=core_thresholds(sample_core_quantiles(len(h),seed),engine['core_mean'],engine['core_std'])-engine['v_base']
            np.testing.assert_allclose(delta,h*expected,rtol=1e-6,atol=1e-6)
            if not np.all((h==0)|(h==1)) or int(h.sum())!=1499 or np.any(delta[h==0]!=0):raise RuntimeError('core budget or background changed')
            selected=h>0;assignment=((pos[:,None,:]-center[None,:,:])**2).sum(axis=2).argmin(axis=1)
            core=delta[selected];clipped=np.minimum(core,0)
            rows.append({'candidate_id':row['candidate_id'],'seed':u['seed'],'worker_path':str(path),'worker_sha256':run.sha(path),
                'n_core_neurons':int(selected.sum()),'core_mean_threshold_mV':float(engine['v_base']+core.mean()),
                'core_threshold_sd_mV':float(core.std()),'fraction_core_threshold_above_background':float(np.mean(core>0)),
                'mean_signed_depth_mV':float(-core.mean()),'per_core_mean_signed_depth_mV':[float(-delta[selected&(assignment==i)].mean()) for i in (0,1)],
                'same_quantiles_clipped_mean_depth_mV':float(-clipped.mean()),
                'same_quantiles_clipping_additional_mean_depth_mV':float(core.mean()-clipped.mean()),
                'joint_distance':u['metrics']['joint_distance'],'exploration_score':u['metrics']['exploration_score']})
    correlation=[]
    for seed_value in [2511,2512]:
        a=[r for r in rows if r['seed']==seed_value and r['joint_distance'] is not None]
        correlation.append({'seed':seed_value,'n_geometries':len(a),'depth_vs_joint_loss_spearman':float(spearmanr(
            [r['mean_signed_depth_mV'] for r in a],[r['joint_distance'] for r in a]).statistic)})
    summary={'status':'VTH_REALIZATION_AUDIT_COMPLETE','n_geometries':len(candidates),'n_trajectories':len(rows),
        'fixed_parameters':{k:engine[k] for k in ['core_mean','core_std','v_base']},'quantile_seed':seed,
        'all_saved_maps_match_frozen_signed_recipe':True,'all_core_counts_1499':True,'all_noncore_E_unchanged':True,
        'mean_depth_min_median_max_mV':np.quantile([r['mean_signed_depth_mV'] for r in rows],[0,.5,1]),
        'raised_fraction_min_median_max':np.quantile([r['fraction_core_threshold_above_background'] for r in rows],[0,.5,1]),
        'clipping_additional_depth_min_median_max_mV':np.quantile([r['same_quantiles_clipping_additional_mean_depth_mV'] for r in rows],[0,.5,1]),
        'within_seed_correlations_descriptive':correlation,
        'historical_difference':'manual_hard combines two single-core fields with np.minimum, clipping above-baseline draws for disjoint cores; current signed mapping retains those draws.',
        'counterfactual_boundary':'Clipping statistics use identical saved quantiles; no new simulation, no historical rejection-sampler bitwise equivalence claim.',
        'live_search_changed':False,'patient_heldout_opened':False,
        'source_hashes':{str(p):run.sha(p) for p in sources+[stage,transition,Path(__file__),ROOT/'src/topic4_core_field.py',ROOT/'src/topic4_core_field_rev9.py',ROOT/'src/topic4_core_field_runner.py']}}
    run.write(out/'per_trajectory.json',{'rows':rows});run.write(out/'summary.json',summary)
    print({k:v for k,v in summary.items() if k not in ['source_hashes']})


if __name__=='__main__':main()
