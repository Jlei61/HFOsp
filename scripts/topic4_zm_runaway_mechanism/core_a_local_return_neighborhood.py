"""One targeted intermediate Core-A Z field, same actual complete history.

This tests low-activity access near the late-return counterexample while
invariant solutions are being corrected. Censoring never identifies a type.
"""
from common import OUT,np,read,write,log,model
from core_a_equilibrium_branch import Family
import onset_state_continuation as flow
import core_a_resource_branch as local
from fine_rate_frozen_Z_fields import capture
from scipy.ndimage import uniform_filter1d
import argparse

TYPE=OUT/'core_a_bifurcation_type_20260924'
DEST=TYPE/'local_return_neighborhood'
flow.DEST=DEST;local.DEST=DEST
LABEL='DA0348158'


def register():
    DEST.mkdir(exist_ok=True);assert not (DEST/'contract.json').exists()
    source=TYPE/'sustained_A_long_window/below_extend_to60s'
    assert read(source/'local_state_audit.json')['status']=='AUDIT_PASS'
    base=dict(np.load(source/'final_state.npz'));s=model(40);family=Family(s)
    d=.3481576412548057
    z,tm=family.field(d)
    assert np.array_equal(z[~family.A],base['syn'][5,~family.A])
    actual=float(1-np.average(z[family.A],weights=s.sizes[family.A]));assert abs(actual-d)<1e-12
    c=dict(label=LABEL,field=LABEL,initial=str(source/'final_state.npz'),source_dt_ms=.05,dt_ms=.05,
           duration_ms=70000,previous_elapsed_ms=0)
    np.savez_compressed(DEST/'fields.npz',**{LABEL:z});write(DEST/'conditions.json',{LABEL:c})
    write(DEST/'contract.json',dict(
        question='Between the observed late-return D_A=.34315764 and70s right-censored D_A=.36315764, does a nearer D_A=.34815764 field still access a qualified Core-A low-activity interval from the same original high-activity history?',
        reason='Aperiodic finite-time expansion and a64s self-terminating episode preclude replacing invariant-set analysis with a short no-return classification. This one neighboring field complements, rather than certifies, periodic/manifold classification.',
        coordinates={LABEL:dict(D_A=actual,Z_A=1-actual,native_interpolation_time_ms=tm)},conditions={LABEL:c},
        physical_scope='Same full3479-group0.5mm rate field, original graph, physical delays, locked response and private variance. Only original within-Core-A Z is changed using audited native pattern interpolation. All outside-A Z bitwise native9s. All Z held during each trajectory, all E M dynamic; no external future count innovations.',
        initial='Exact original D_A=.34315764 state at60s, from which the unchanged-field control returns at64.83s. All fast, M and delay initial states are identical; only within-A Z changes. The new exposure clock starts atzero, not concatenated as same-field exposure with the prior60s.',
        budget='One70s deterministic trajectory atone newly registered neighboring field. Single complete history, no independent-replicate claim or automatic parameter expansion.',
        readout='Whole-record10ms-smoothed Core-A rate, quiet<5Hz foratleast20ms, complete and censored high-activity intervals. Threshold1/10/50Hz diagnostics, last-window summaries only secondary. Each actual return refutes permanence of its preceding episode; no return is right-censoring.',
        acceptance='Exact10ms replay and original deterministic flux checks before launch; bitwise paired initial state except Core-A Z. Independent reconstruction from all groups, space, M and source identities after completion. No Hopf, saddle-node, period fold, basin or crisis label from this record alone.',model_promoted=False))
    log('LOCAL RETURN FIELD REGISTERED',actual,1-actual)


def check(device):
    flow.check(device)
    e=flow.build(device);c=read(DEST/'conditions.json')[LABEL]
    source=dict(np.load(c['initial']));z=flow.initialize(e,c);state=capture(e)
    A=e.s.E&(e.s.geo['group_region']==0)
    assert np.array_equal(z[~A],source['syn'][5,~A])
    for key,value in state.items():
        if key=='syn':assert np.array_equal(value[:5],source[key][:5])
        else:assert np.array_equal(value,source[key]),key
    qa=read(DEST/'implementation_check.json');qa.update(only_core_A_Z_changed=True,full_initial_state_bitwise_except_core_A_Z=True)
    write(DEST/'implementation_check.json',qa)


def audit():
    local.audit(LABEL);folder=DEST/LABEL;jobs=read(folder/'jobs.json')
    s=model(40);W=flow.regional_weights(s);rates=[];paths=[]
    for b in jobs['completed_blocks']:
        path=folder/f'block{b:02d}.npz';z=np.load(path)
        q=z['group_rate_hz'].astype(float)@W.T
        assert np.max(abs(q-z['regional_rate_hz']))<1e-9
        rates.append(q);paths.append(str(path))
    expected=read(DEST/'conditions.json')[LABEL]['duration_ms']
    rates=np.concatenate(rates);assert len(rates)==expected
    sm=uniform_filter1d(rates[:,1],10,mode='nearest');thresholds=[]
    for threshold in [1.,5.,10.,50.]:
        edge=np.diff(np.r_[False,sm<threshold,False].astype(int))
        quiet=[[int(a),int(b)] for a,b in zip(np.flatnonzero(edge==1),np.flatnonzero(edge==-1)) if b-a>=20]
        activities=[];start=0
        for a,b in quiet:
            if a>start:activities.append(dict(start_ms=start,end_ms=a,duration_ms=a-start,left_censored=start==0,right_censored=False))
            start=b
        if start<len(sm):activities.append(dict(start_ms=start,end_ms=len(sm),duration_ms=len(sm)-start,left_censored=start==0,right_censored=True))
        thresholds.append(dict(threshold_hz=threshold,quiet_intervals_ms=quiet,activities=activities,
                               time_fraction_below_after2s=float(np.mean(sm[2000:]<threshold))))
    np.savez_compressed(folder/'whole_record_regional.npz',time_ms=np.arange(1,expected+1),regional_rate_hz=rates,Core_A_smoothed_hz=sm)
    field=read(DEST/'conditions.json')[LABEL]['field']
    write(folder/'whole_record_audit.json',dict(status='AUDIT_PASS',coordinates=read(DEST/'contract.json')['coordinates'][field],
          observed_ms=len(rates),sources=paths,thresholds=thresholds,
          scope='One matched-history deterministic intervention. Any no-return duration is censored. Finite observation does not certify an invariant set or bifurcation type.',model_promoted=False))
    log('LOCAL RETURN WHOLE AUDIT',thresholds)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','check','run','audit']);p.add_argument('--device',type=int,default=0)
    a=p.parse_args();{'register':register,'check':lambda:check(a.device),'run':lambda:flow.run(LABEL,a.device),'audit':audit}[a.command]()
