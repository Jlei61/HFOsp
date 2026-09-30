"""Extend the onset-side actual trajectory without resetting any state.

The20s high-A window is censored, not a permanence certificate. A longer
same-field run tests the specific alternative of another long transient.
"""
from common import OUT,np,read,write,log,model
import onset_state_continuation as flow
import core_a_resource_branch as local
from scipy.ndimage import uniform_filter1d
import argparse

DEST=OUT/'core_a_bifurcation_type_20260924/sustained_A_long_window'
SOURCE=DEST.parent/'fold_attractor_contrast/below'
flow.DEST=DEST;local.DEST=DEST
LABEL='below_extend_to60s'


def register():
    assert read(SOURCE/'local_state_audit.json')['status']=='AUDIT_PASS'
    DEST.mkdir(exist_ok=True);assert not(DEST/'conditions.json').exists()
    old=read(SOURCE.parent/'contract.json')['coordinates']['below']
    field=np.load(SOURCE/'final_state.npz')['syn'][5]
    np.savez_compressed(DEST/'fields.npz',below=field)
    c=dict(label=LABEL,field='below',initial=str(SOURCE/'final_state.npz'),
        dt_ms=.05,source_dt_ms=.05,duration_ms=40000,previous_elapsed_ms=20000)
    write(DEST/'conditions.json',{LABEL:c})
    write(DEST/'contract.json',dict(
        question='At D_A=.3431576413, is the observed20s sustained-Core-A episode only a long transient that returns to quiet during a longer unchanged-field continuation?',
        source=str(SOURCE),coordinates={'below':old},conditions=[c],
        equations='Same original full spatial deterministic conditional drift, native Core-A-only Z field, outside-A Z native9s. All Z held and every M remains dynamic. Resume exact20s terminal state including all fast states, M and delay history. No noise, field, parameter or initial-history change.',
        readout='Original10ms-smoothed regional E rate, >=20ms quiet below5Hz. Retain all completed and censored A episodes, both-core and surrounding activity and complete5s states. Source20s and new40s records concatenate chronologically with no reset.',
        interpretation='A later complete quiet interval refutes treating the preceding20s as permanent A activity. No quiet by60s strengthens finite-duration evidence but does not prove an asymptotic attractor or name a crisis.',
        budget='One40s extension. No parameter expansion or independent replicate is counted.',model_promoted=False))


def audit():
    local.audit(LABEL)
    assert read(DEST/LABEL/'local_state_audit.json')['status']=='AUDIT_PASS'
    s=model(40);W=flow.regional_weights(s);R=[];sources=[]
    for folder in [SOURCE,DEST/LABEL]:
        jobs=read(folder/'jobs.json');assert jobs['status']=='COMPLETE'
        for index in jobs['completed_blocks']:
            path=folder/f'block{index:02d}.npz';z=np.load(path)
            R.append(z['group_rate_hz'].astype(float)@W.T);sources.append(str(path))
    R=np.concatenate(R);assert len(R)==60000
    sm=uniform_filter1d(R[:,1],10,mode='nearest')
    flags=sm<5;edge=np.diff(np.r_[False,flags,False].astype(int))
    starts=np.flatnonzero(edge==1);ends=np.flatnonzero(edge==-1)
    quiet=[[int(a),int(b)] for a,b in zip(starts,ends) if b-a>=20]
    windows=[]
    for a in range(0,60000,5000):
        r=sm[a:a+5000];windows.append(dict(window_ms=[a,a+5000],mean_Core_A_hz=float(R[a:a+5000,1].mean()),
            min_Core_A_10ms_hz=float(r.min()),quiet_fraction=float(np.mean(r<5))))
    result=dict(status='AUDIT_PASS',total_same_field_ms=60000,D_A=read(DEST/'contract.json')['coordinates']['below']['D_A'],
        all_Z_held=True,all_M_dynamic=True,quiet_intervals_ms=quiet,
        quiet_after_original20s=any(b>20000 for a,b in quiet),windows=windows,sources=sources,
        scope='Same-history finite continuation. An observed return is conclusive for nonpermanence of that episode; absence of return is right censoring, not an asymptotic bifurcation certificate.',model_promoted=False)
    np.savez_compressed(DEST/'joined60s_regional.npz',time_ms=np.arange(1,60001),regional_rate_hz=R,Core_A_smoothed_hz=sm)
    write(DEST/'joined60s_audit.json',result);log('CORE A60S AUDIT',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','check','run','audit'])
    p.add_argument('--device',type=int,default=1);a=p.parse_args()
    {'register':register,'check':lambda:flow.check(a.device),'run':lambda:flow.run(LABEL,a.device),'audit':audit}[a.command]()
