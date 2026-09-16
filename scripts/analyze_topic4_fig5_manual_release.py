#!/usr/bin/env python3
"""Run-level summaries and independent checks for the manual release figure."""
from pathlib import Path
import json
import csv
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/topic4_sef_hfo/fig5_manual_core_release_v1'
PREVIOUS=ROOT/'results/topic4_sef_hfo/historical_manual_hard_native_z_v1'
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def main():
    a=np.load(OUT/'runs/continuous_refill_release.npz');r=read(OUT/'runs/continuous_refill_release.json')
    old=np.load(PREVIOUS/'trajectory.npz')
    release=r['release_ms']/1000;refill=r['restore_start_ms']/1000;onset=r['first_trigger_ms']/1000
    n=int(round(release*10000))
    checks={
        'entire_pre_release_E_rate_bitwise_equal':bool(np.array_equal(a['rate_e_hz'][:n],old['rate_e_hz'][:n])),
        'entire_pre_release_I_rate_bitwise_equal':bool(np.array_equal(a['rate_i_hz'][:n],old['rate_i_hz'][:n])),
        'entire_pre_release_sample_spikes_bitwise_equal':bool(np.array_equal(a['sample_spikes'][:n],old['sample_spikes'][:n])),
        'readout_time_grid_05ms':bool(np.allclose(np.diff(a['lfp_time_ms']),.5)),
        'readout_finite':bool(np.isfinite(a['lfp_raw']).all() and np.isfinite(a['lfp_effective']).all()),
        'applied_inhibition_proxy_not_above_unscaled_proxy':bool(np.all(a['lfp_effective']<=a['lfp_raw']+1e-10)),
        'M_stays_zero':bool(np.all(a['z_stats'][:,10]==0)),
        'Z_range':bool(np.all((a['z_field_5ms']>=0)&(a['z_field_5ms']<=1))),
        'field_E_count_conservation':bool(np.array_equal(a['field_e_count_1ms'].sum(1),np.rint(a['rate_e_hz']*32000*.0001).astype(np.int64).reshape(-1,10).sum(1))),
    }
    zt=a['z_time_ms']/1000;z=a['z_stats'][:,0]
    k=np.flatnonzero(np.isclose(zt,release,atol=1e-8))[0]
    checks['refill_endpoint_all_Z_one']=bool(np.all(a['z_field_5ms'][k]==1))
    lr=np.flatnonzero(np.isclose(a['lfp_time_ms'],r['release_ms'],atol=1e-8))[0]
    checks['two_readout_definitions_equal_at_refill_endpoint']=bool(np.array_equal(a['lfp_raw'][lr],a['lfp_effective'][lr]))
    checks['native_Z_depletes_after_release']=bool(z[k+1:].min()<.99)
    assert all(checks.values()),checks
    e=a['rate_e_hz'].reshape(-1,50).mean(1);i=a['rate_i_hz'].reshape(-1,50).mean(1)
    t=(np.arange(len(e))+.5)*.005
    regions=a['region_spikes_1ms'].reshape(-1,5,6).sum(1)/a['region_counts'][None,:]/.005
    end=r['duration_ms']/1000
    periods=[('baseline',.5,1),('pre_refill_high',onset,refill),('after_release',release+.5,release+1.5),('late',end-1,end)]
    stage=[]
    for label,lo,hi in periods:
        sel=(t>=lo)&(t<hi);values=e[sel]
        stage.append(dict(label=label,window_s=[lo,hi],E_mean_hz=float(values.mean()),I_mean_hz=float(i[sel].mean()),
                          E_quiet_fraction=float(np.mean(values<1)),E_cv=float(values.std()/max(values.mean(),1e-12)),
                          regional_EI_mean_hz=regions[sel].mean(0).tolist(),Z_mean=float(z[(zt>=lo)&(zt<hi)].mean())))
    # Oscillatory-looking paths are tested for dependence on the observation bin;
    # these pulse counts/phase areas are descriptions, not limit-cycle detection.
    loops=[]
    for bin_ms in (2,5,10):
        b=round(bin_ms/.1);nn=len(a['rate_e_hz'])//b
        er=a['rate_e_hz'][:nn*b].reshape(-1,b).mean(1);ir=a['rate_i_hz'][:nn*b].reshape(-1,b).mean(1)
        tt=(np.arange(nn)+.5)*bin_ms/1000
        for label,lo,hi in [('early_self_limited',1,max(1.5,onset-1)),('pre_refill_high',onset,refill),('post_release',release+.5,min(release+5,tt[-1]))]:
            sel=(tt>=lo)&(tt<hi);ee=er[sel];ii=ir[sel]
            sm=gaussian_filter1d(ee,max(1,5/bin_ms));peaks,_=find_peaks(sm,prominence=15,distance=max(1,round(80/bin_ms)))
            de=ee-ee.mean();di=ii-ii.mean()
            area=float(np.sum(de[:-1]*np.diff(di)-di[:-1]*np.diff(de))/2)
            loops.append(dict(bin_ms=bin_ms,stage=label,n_population_peaks=int(len(peaks)),signed_EI_path_area_hz2=area,
                              quiet_fraction=float(np.mean(ee<1)),E_mean_hz=float(ee.mean()),E_cv=float(ee.std()/max(ee.mean(),1e-12))))
    rows=[];input_pairs=[]
    byseed={}
    protocol=read(OUT/'protocol.json')
    for job in protocol['jobs']:
        rr=read(OUT/'runs'/(job['name']+'.json'));aa=np.load(OUT/'runs'/(job['name']+'.npz'))
        assert rr['status']=='COMPLETE' and rr['frozen_identity']==r['frozen_identity']
        latent=rr['first_trigger_ms'];observed=latent is not None and latent<=24000
        counts10=np.rint(aa['rate_e_hz']*32000*.0001).astype(np.int64).reshape(-1,100).sum(1)
        high10=(counts10/32000/.01)>=200.
        first=np.flatnonzero(np.convolve(high10.astype(int),np.ones(20,int),'valid')==20)
        recomputed=float((first[0]+20)*10) if len(first) else None
        assert recomputed==latent,(job['name'],recomputed,latent)
        er5=aa['rate_e_hz'].reshape(-1,50).mean(1)
        width=100
        nonquiet=np.convolve((er5>=1.).astype(int),np.ones(width,dtype=int),'valid')
        average=np.convolve(er5,np.ones(width)/width,'valid')
        hits=np.flatnonzero((nonquiet==width)&(average>=20.))
        quiet_loss_s=float((hits[0]+width)*.005) if len(hits) else None
        rows.append(dict(name=job['name'],seed=job['seed'],tau_z_ms=job['tau_z_ms'],I_th=job['threshold'],
                         transition_by_24s=int(observed),transition_time_s=latent/1000 if observed else '',
                         first_500ms_without_quiet_s=quiet_loss_s if quiet_loss_s is not None and quiet_loss_s<=24 else '',
                         time_without_transition_restricted_24s=latent/1000 if observed else 24.,
                         observation_horizon_s=24.,M_enabled=False))
        inp=aa['input_summary']
        if job['seed'] in byseed:
            base=byseed[job['seed']];n=min(len(base),len(inp));equal=bool(np.array_equal(base[:n],inp[:n]));assert equal
            input_pairs.append(dict(seed=job['seed'],name=job['name'],checked_input_samples=n,exact_equal=equal))
        else:byseed[job['seed']]=inp.copy()
    with (OUT/'transition_times.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(OUT/'analysis.json',dict(stages=stage,oscillation_observation_sensitivity=loops,
          state_scope='Self-limited population events versus sustained high-rate activity; no limit-cycle or Hopf conclusion from projected loops.',
          all_trigger_times_ms=r['all_trigger_times_ms'],Z_mean_at_release=float(z[k]),final_Z_mean=float(z[-1]),
          physical_parameters='Historical manual threshold field on current C fast carrier; native global/spatial OU and Poisson; M off.',
          auxiliary_observable='First 500ms with every 5-ms E-rate bin >=1Hz and window mean >=20Hz; descriptive loss-of-quiet readout, not a validated seizure classifier.',
          censoring_scope='Not reaching the 200Hz/200ms trigger by 24s does not establish persistent interictal activity or exclude a lower-rate sustained state.',
          inference_unit='Noise realization on one fixed topology, n=3 per parameter cell.'))
    fr=np.load(PREVIOUS/'native/frozen_t8000.npz')
    frozen_e=fr['rate_e_hz'].reshape(-1,50).mean(1)
    natural_e=e[(t>=8)&(t<10)]
    old_e=old['rate_e_hz'].reshape(-1,50).mean(1)
    old_t=(np.arange(len(old_e))+.5)*.005
    released_e=e[(t>=release)&(t<14.18)];held_e=old_e[(old_t>=release)&(old_t<14.18)]
    def summary(v):return dict(mean_E_hz=float(v.mean()),quiet_fraction=float(np.mean(v<1)),peak_5ms_hz=float(v.max()))
    fr94=np.load(PREVIOUS/'native/frozen_t9400.npz')
    n94=int(round((refill-9.4)/.005))
    f94=fr94['rate_e_hz'].reshape(-1,50).mean(1)[:n94]
    write(OUT/'paired_control_summary.json',dict(
        before_transition=dict(window_s=[8,10],native=summary(natural_e),frozen_Z=summary(frozen_e)),
        near_transition=dict(window_s=[9.4,refill],native=summary(e[(t>=9.4)&(t<refill)]),frozen_Z=summary(f94)),
        after_refill=dict(window_s=[release,14.18],released_Z=summary(released_e),held_Z=summary(held_e)),
        checkpoint_replay_qa=read(PREVIOUS/'native_resume_qa.json'),
        scope='Matched full-state and future-input finite continuations; these comparisons do not establish asymptotic attractors.'))
    write(OUT/'artifact_qa.json',dict(status='PASS',checks=checks,paired_input_checks=input_pairs,
          slow_update_qa=read(OUT/'slow_qa.json'),all_parameter_runs_complete=len(rows),
          first_entry_times_independently_recomputed_from_spike_counts=True,
          interpretation='Numerical and provenance QA, not human scientific acceptance.'))
    print(json.dumps(dict(checks=checks,stages=stage,all_trigger_times_ms=r['all_trigger_times_ms']),indent=2))


if __name__=='__main__':main()
