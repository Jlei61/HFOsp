"""Longer unchanged-field controls after the64.83s Core-A quiet return.

Compare complete, explicitly concatenated histories. Absence of a return is
right censoring, never a bifurcation certificate or a stable-attractor label.
"""
from common import OUT,np,read,write,log,model
import onset_state_continuation as flow
import core_a_resource_branch as local
from scipy.ndimage import uniform_filter1d
from pathlib import Path
import argparse

TYPE=OUT/'core_a_bifurcation_type_20260924'
DEST=TYPE/'censoring_controls'
flow.DEST=DEST;local.DEST=DEST


def register():
    DEST.mkdir(exist_ok=True);assert not(DEST/'conditions.json').exists()
    sources={'above_SN':TYPE/'fold_attractor_contrast/above',
             'strong_depletion':TYPE/'depleted_coarse'}
    coordinates={'above_SN':read(TYPE/'fold_attractor_contrast/contract.json')['coordinates']['above'],
        'strong_depletion':next(iter(read(TYPE/'contract.json')['coordinates'].values()))}
    conditions={};fields={}
    for label,source in sources.items():
        assert read(source/'local_state_audit.json')['status']=='AUDIT_PASS'
        fields[label]=np.load(source/'final_state.npz')['syn'][5]
        prior=20000 if label=='above_SN' else 30000
        conditions[label]=dict(label=label,field=label,initial=str(source/'final_state.npz'),
            source_dt_ms=.05,dt_ms=.05,duration_ms=70000-prior,previous_elapsed_ms=0,
            prior_same_field_elapsed_ms=prior)
    np.savez_compressed(DEST/'fields.npz',**fields)
    write(DEST/'conditions.json',conditions)
    write(DEST/'contract.json',dict(
        question='Does Core A regain a qualified quiet interval on the other side of the certified equilibrium SN or at the stronger depletion endpoint, after observation comparable to the64.83s counterexample?',
        trigger='The D_A=.34315764 trajectory returned to qualified Core-A quiet at64.83s. Its preceding60s absence of quiet was censoring, not permanent activity.',
        coordinates=coordinates,conditions=conditions,
        equations='Identical full3479-group spatial rate flow, original physical delays, locked response and constant mean input. Each original full terminal state is resumed exactly; all Z fixed, all M dynamic, no future innovations or new coefficients.',
        budget='One50s extension above the equilibrium SN and one40s stronger-depletion extension; each actual history reaches70s. These are continuations, not independent replicates.',
        primary='All qualified Core-A quiet intervals,10ms smoothing and<5Hz for>=20ms. Report each uninterrupted activity interval with both boundaries and right-censoring. Original global recruitment stays separate.',
        acceptance='Independent fullgroup-to-region/space reconstruction and Z identity, original source chronology and complete-state source. A return refutes permanence of that episode; no return provides a lower bound on duration only. No bifurcation type from threshold crossings.',model_promoted=False))


def audit(label):
    local.audit(label)
    s=model(40);W=flow.regional_weights(s)
    prefix=[TYPE/'fold_attractor_contrast/above'] if label=='above_SN' else [OUT/'core_a_resource_bifurcation_20260923/coreA_depleted',TYPE/'depleted_coarse']
    chunks=[];sources=[];zfixed=np.load(DEST/'fields.npz')[label]
    for folder in prefix+[DEST/label]:
        if (folder/'trajectory.npz').exists():
            paths=[folder/'trajectory.npz']
        else:
            jobs=read(folder/'jobs.json');assert jobs['status']=='COMPLETE'
            paths=[folder/f'block{b:02d}.npz' for b in jobs['completed_blocks']]
        for path in paths:
            z=np.load(path);r=z['group_rate_hz'].astype(float)
            assert np.array_equal(z['Z'],zfixed)
            rr=r@W.T
            if 'regional_rate_hz' in z:assert np.max(abs(rr-z['regional_rate_hz']))<1e-9
            chunks.append(rr);sources.append(str(path))
    rates=np.concatenate(chunks);assert len(rates)==70000
    sm=uniform_filter1d(rates[:,1],10,mode='nearest')
    edges=np.diff(np.r_[False,sm<5,False].astype(int))
    quiet=[[int(a),int(b)] for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)) if b-a>=20]
    activities=[];start=0
    for a,b in quiet:
        if a>start:activities.append(dict(start_ms=start,end_ms=a,duration_ms=a-start,left_censored=start==0,right_censored=False))
        start=b
    if start<len(sm):activities.append(dict(start_ms=start,end_ms=len(sm),duration_ms=len(sm)-start,left_censored=start==0,right_censored=True))
    result=dict(status='AUDIT_PASS',label=label,coordinates=read(DEST/'contract.json')['coordinates'][label],
        observed_ms=len(rates),all_Z_held=True,all_M_dynamic=True,quiet_intervals_ms=quiet,activities=activities,
        sources=sources,scope='Finite actual history; censored activity duration cannot establish asymptotic persistence or a bifurcation.',model_promoted=False)
    np.savez_compressed(DEST/label/'joined70s_regional.npz',time_ms=np.arange(1,len(rates)+1),regional_rate_hz=rates,Core_A_smoothed_hz=sm)
    write(DEST/label/'joined70s_audit.json',result);log('CENSORING CONTROL AUDIT',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','check','run','audit'])
    p.add_argument('--label',choices=['above_SN','strong_depletion']);p.add_argument('--device',type=int,default=0);a=p.parse_args()
    {'register':register,'check':lambda:flow.check(a.device),'run':lambda:flow.run(a.label,a.device),'audit':lambda:audit(a.label)}[a.command]()
