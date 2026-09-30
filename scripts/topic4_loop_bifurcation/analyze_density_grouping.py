#!/usr/bin/env python3
"""Native correspondence of the single finer-group density diagnostic."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys,argparse,time,warnings
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from campaign import ROOT,REPO,read,write,sha
from compare_density_resolution import readouts,window_stats,WINDOWS,BASE,REPLAY
sys.path.insert(0,str(REPO/'scripts/topic4_interictal_surrogate'))
from evaluate_readouts import summarize
from direction_validation_v2 import arrivals,trailing5,describe,score,spatial_comparison,CONTRACT
from interictal_common import smooth,safe

OUT=ROOT/'density_spatial_grouping'
OLD=REPO/'results/topic4_sef_hfo/interictal_spatial_surrogate_6101_20260916'


def main():
    assert read(OUT/'result.json')['status']=='COMPLETE'
    assert read(OUT/'result.json')['engine_sha256_unchanged']
    op=dict(np.load(OUT/'operators/geometry.npz'));geo=dict(np.load(REPLAY/'geometry.npz'))
    with np.load(OUT/'trajectory.npz') as z:
        t=z['time_ms'];field=z['field_E_Hz'].astype(float);count=z['cell_counts']
        assert np.array_equal(count,geo['cell_e_counts']) and np.array_equal(t,np.arange(12500)+1.)
        ee=z['population_E'];size=z['group_sizes'];rate=z['group_rate_Hz'].astype(float)
        meanZ=np.average(z['group_Z'][:,ee],axis=1,weights=size[ee])
        assert np.array_equal(size,op['group_size']) and np.array_equal(ee,op['population']==0)
    ev,summary,whole,sm=readouts(t,field,count,'g40_R2048_num927611')
    weighted=np.average(rate[:,ee],axis=1,weights=size[ee]);error=float(abs(whole-weighted).max());assert error<1e-4
    primary=window_stats(ev,1000,9420);D=float(1-meanZ[9869]);nativeD=read(BASE/'native_reference/checkpoint_projections.json')['9870']['D']
    gate=dict(self_limited_events=bool(primary['n'] and 50<=primary['median_duration_ms']<=200 and summary['quiet_fraction']>=.15),
        two_core_participation=bool(primary['n'] and primary['both_cores']/primary['n']>=.5),
        surround_recruitment=bool(primary['n'] and .3<=primary['median_area']<=1.),
        propagation=bool(primary['n'] and primary['forward']>0 and primary['reverse']>0 and 5<=primary['median_extent_mm']<=20),
        entry=bool(summary['high_onset_ms'] is not None and 7000<=summary['high_onset_ms']<=13000),D_track=bool(abs(D-nativeD)<=.05))
    complete=[]
    for e in ev:
        a=int(np.searchsorted(t,e['start_ms']));b=a+int(e['duration_ms'])
        if a>=20 and b+20<=len(sm) and (sm[a-20:a]<5).all() and (sm[b:b+20]<5).all():complete.append(e)
    dynamics=dict(summary=summary,D9870=D,original_A4_checks=gate,n_original_A4_pass=sum(gate.values()),
        windows={f'{a}-{b}':window_stats(ev,a,b) for a,b in WINDOWS},
        complete_quiet_bounded_windows={f'{a}-{b}':window_stats([e for e in complete if e['start_ms']+e['duration_ms']<=b],a,b) for a,b in WINDOWS},
        quiet_by_window={f'{a}-{b}':float((sm[(t>=a)&(t<b)]<5).mean()) for a,b in WINDOWS},
        events=[{k:v for k,v in e.items() if k!='onset'} for e in ev])
    weights=op['contact_rate_weights'];assert np.allclose(weights.sum(0),1) and not weights[~ee].any()
    raw=rate[:8000].reshape(4000,2,-1).sum(1)*.001@weights;env=smooth(raw)
    contract=read(OLD/'observer_firing.json');names=contract['contact_names'];assert names==geo['contact_names'].tolist()
    contact=summarize(env,contract,'firing',start_ms=500,stop_ms=8000)
    ids=contact['primary_event_ids'];mu=np.array(contact['centroids_ms'],float).reshape(-1,15)
    sf=trailing5(field[:8000]);maps=np.array([arrivals(sf,contact['observation']['events'][i]['window_ms']) for i in ids]).reshape(-1,400)
    centers=geo['centers_mm'];axis=centers[1]-centers[0];axis/=np.linalg.norm(axis);yy,xx=np.mgrid[:20,:20]
    coord=(np.c_[xx.ravel()+.5,yy.ravel()+.5]-centers[0])@axis;rho=[]
    for a in maps:
        ok=np.isfinite(a);rho.append(spearmanr(coord[ok],a[ok]).statistic if ok.sum()>=5 and len(np.unique(a[ok]))>1 else np.nan)
    classes=['A_to_B','B_to_A','weak_or_complex','not_estimable'];conditional={};labels={}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        for cut in CONTRACT['field_axis_cutoff_sensitivity']:
            ll=['not_estimable' if not np.isfinite(x) else 'A_to_B' if x>=cut else 'B_to_A' if x<=-cut else 'weak_or_complex' for x in rho]
            labels[str(cut)]=ll;conditional[str(cut)]={c:describe(mu[np.asarray(ids)[np.asarray(ll)==c]],names) for c in classes}
        parent=read(ROOT/'density_contact_replay/comparison.json');parentdir=read(ROOT/'density_contact_replay/direction_comparison.json')
        comparisons=[]
        for other in ['native9108401','native9108402','R8192_num927611']:
            reference=parent['rows'][other]['firing'];direction=parentdir['rows'][other]['firing']
            comparisons.append(dict(reference=other,overall=score(contact['summary'],reference['summary'],names),
                within_direction={cut:{c:score(conditional[cut][c],direction['conditional'][cut][c],names) for c in classes} for cut in labels}))
    oldd=read(ROOT/'density_spatial_resolution/comparison.json')['rows']
    write(OUT/'comparison.json',safe(dict(status='COMPLETE',dynamics=dynamics,contacts=contact,
        direction=dict(axis_rho=rho,labels=labels,conditional=conditional,counts={cut:{c:v['N'] for c,v in d.items()} for cut,d in conditional.items()}),
        contact_comparisons=comparisons,reference_dynamics={r['name']:{k:r[k] for k in ['summary','D9870','windows','quiet_by_window']} for r in oldd if r['name'] in ['native9108401','native9108402','R8192_num927611']},
        qa=dict(weighted_field_rate_error_Hz=error,correct_3479_group_contact_weights=True,frozen_contact_observer=True),
        producer_sha256=sha(__file__),source_sha256=sha(OUT/'trajectory.npz'),
        limitations='One fine-grid numericalstream; forcing identity is in the source contract. Not a numericalconvergence certificate. RawcurrentLFP not observed in this run; gatedgroupcurrent cannot replace it. Baseline G/Koff only.',
        model_promoted=False,formal_bifurcation_allowed=False,human_review='PENDING')))
    np.savez_compressed(OUT/'readout_arrays.npz',time_ms=t,smoothed_allE_Hz=sm,mean_Z=meanZ,
        firing_envelope=env,contact_names=names,contact_event_arrivals_ms=maps)
    print(safe(dict(summary=summary,primary=primary,D9870=D,A4=gate,contact_count=contact['summary']['N'],direction_counts={c:v['N'] for c,v in conditional['0.3'].items()})),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');p.add_argument('--root',type=Path,default=OUT);a=p.parse_args()
    OUT=a.root
    while a.wait and not (OUT/'result.json').exists():time.sleep(30)
    main()
