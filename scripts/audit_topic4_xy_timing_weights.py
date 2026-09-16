#!/usr/bin/env python3
"""Same windows/masks, alternate timing weights: diagnose observation dependence."""
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.stats import spearmanr
from scripts import run_topic4_joint_xy_kernel_search as run
from src.topic4_xy_timing_observation_audit import reweight_centroids,paired_order_change


def main():
    source=run.OUT/'baseline_scores.json';rows=run.read(source)['candidates'];plan=run.read(run.CONFIG)
    obj=run.KernelObjective(run.v1,run.OUT,run.KERNEL)
    out=run.OUT/'timing_observation_audit';out.mkdir(exist_ok=True)
    legacy=Path('/home/honglab/leijiaxin/HFOsp/ReplayIED/inter_events/epilepsiae_interictal/epilepsiae_packGroupEvents_supressAllSyn_withFreqCenter.py')
    paths=[source,legacy,ROOT/'src/topic4_joint_xy.py',ROOT/'src/topic4_xy_timing_observation_audit.py',Path(__file__),run.KERNEL]
    contract={'source_hashes':{str(p):run.sha(p) for p in paths},'heldout_opened':False,'live_search_modified':False,
        'scope':'Timing sensitivity only; preserve windows and masks. Cubed firing density is not spectrogram amplitude cubed and cannot validate an HFO forward model.',
        'variants':['linear_frozen','cubic_density_sensitivity','q10_subtracted_linear_sensitivity'],
        'no_geometry_or_readout_selected_for_acceptance':True,'created_unix':time.time()}
    run.write(out/'contract.json',contract);output=[]
    for number,row in enumerate(rows):
        variants={k:[] for k in contract['variants']}
        for unit in row['units']:
            p=Path(unit['worker_path']);meta=run.read(p)
            if run.sha(p)!=unit['worker_sha256'] or run.sha(meta['arrays']['path'])!=meta['arrays']['sha256']:raise RuntimeError('worker drift')
            original,u=run.v1.read_worker(p,obj,plan)
            with np.load(meta['arrays']['path']) as z:env=z['contact_envelope'].astype(float);dt=float(z['contact_envelope_dt_ms'])
            w=u['observation']['windows_ms'];mask=np.isfinite(original)
            a=reweight_centroids(env,dt,w,mask)
            np.testing.assert_allclose(a,original,rtol=1e-10,atol=1e-8,equal_nan=True)
            variants['linear_frozen'].append(a)
            variants['cubic_density_sensitivity'].append(reweight_centroids(env,dt,w,mask,power=3))
            baseline=np.quantile(env[:,int(round(plan['observation']['burnin_ms']/dt)):],.1,axis=1)
            variants['q10_subtracted_linear_sensitivity'].append(reweight_centroids(env,dt,w,mask,baseline=baseline))
        variants={k:np.concatenate(v) for k,v in variants.items()};raw=variants['linear_frozen']
        for name,t in variants.items():
            m=obj.metrics(t);mask_equal=np.array_equal(np.isfinite(t),np.isfinite(raw))
            output.append({'candidate_id':row['candidate_id'],'variant':name,'n_events':len(t),
                'participation_preserved':mask_equal,**paired_order_change(raw,t),
                'median_event_span_ms':float(np.median(np.nanmax(t,axis=1)-np.nanmin(t,axis=1))) if len(t) else None,
                **{k:m[k] for k in ('joint_distance','exploration_score','D_order','D_lag','direction_distance')}})
        if number%20==0:
            run.write(out/'status.json',{'status':'RUNNING','n_candidates_complete':number+1,'total':len(rows)})
    run.write(out/'paired_candidate_results.json',{'rows':output})
    stats=[];raw=[r for r in output if r['variant']=='linear_frozen']
    for name in contract['variants']:
        alt=[r for r in output if r['variant']==name];valid=[i for i in range(len(raw)) if raw[i]['joint_distance'] is not None and alt[i]['joint_distance'] is not None]
        x=np.array([raw[i]['joint_distance'] for i in valid]);y=np.array([alt[i]['joint_distance'] for i in valid])
        top=lambda a:set(r['candidate_id'] for r in sorted([r for r in a if r['joint_distance'] is not None],key=lambda r:r['exploration_score'])[:10])
        stats.append({'variant':name,'loss_rank_spearman_vs_linear':float(spearmanr(x,y).statistic),
            'top10_overlap':len(top(raw)&top(alt)),
            'median_strict_order_reversal_fraction':float(np.median([r['strict_order_reversal_fraction'] for r in alt if r['strict_order_reversal_fraction'] is not None])),
            'median_event_span_ms_across_candidates':float(np.median([r['median_event_span_ms'] for r in alt if r['median_event_span_ms'] is not None])),
            'all_participation_preserved':all(r['participation_preserved'] for r in alt)})
    run.write(out/'summary.json',{'status':'PAIRED_TIMING_AUDIT_COMPLETE','n_candidates':len(rows),'variants':stats,
        'patient_median_event_span_ms':float(np.median(np.nanmax(obj.patient,axis=1)-np.nanmin(obj.patient,axis=1))),
        'claim_boundary':contract['scope'],'live_search_modified':False})
    run.write(out/'status.json',{'status':'COMPLETE','n_candidates_complete':len(rows)})
    print(stats)


if __name__=='__main__':main()
