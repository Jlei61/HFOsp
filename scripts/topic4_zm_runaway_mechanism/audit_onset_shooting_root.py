"""Independent fixed-time returns from the corrected full state.

No section search or Anderson code is used. Phase-shifted closure and a
half-period test distinguish an interpolated seed from an invariant orbit
and test whether the claimed period is merely a double traversal.
"""
from common import np, read, write, log
from onset_state_continuation import DEST, build
from onset_period_return import dynamical_state, errors
from fine_rate_frozen_Z_fields import capture, restore
from datetime import datetime
import argparse, os, time


def fixed_return(e,base,T):
    restore(e,base);n=int(np.floor(T/e.dt));a=T/e.dt-n;chunk=round(10/e.dt)
    for _ in range(n//chunk):e.chunk()
    for _ in range(n%chunk):e.step()
    e.cp.cuda.get_current_stream().synchronize()
    left=capture(e);u=dynamical_state(left)
    e.step();e.cp.cuda.get_current_stream().synchronize();right=capture(e);v=dynamical_state(right)
    assert int(left['clock'][0])==int(base['clock'][0])+n
    assert np.array_equal(right['syn'][5],base['syn'][5])
    value={k:(1-a)*u[k]+a*v[k] for k in u}
    return errors(dynamical_state(base),value,e.s.sizes/e.s.sizes.sum())


def audit(label,name,device):
    folder=DEST/label/name;result=read(folder/'result.json')
    assert result['status']=='NUMERICAL_SHOOTING_ROOT'
    out=folder/'independent_validation';out.mkdir(exist_ok=True);assert not (out/'jobs.json').exists()
    write(out/'contract.json',dict(created_local=datetime.now().astimezone().isoformat(),
        method='Independent fixed-time integration from saved corrected state and its100ms/200ms phase shifts. No section search/mixing. FullT andhalfT atinitialphase; fullT atshiftedphases.',
        gate='Same-phase combinedrelative RMS<1e-7 and allblocks<1e-6. Shiftedphase andhalfperiodreported separately; cannot waive mesh/Floquet requirements.',
        budget='3.5periods plus300ms, no adaptive orbit refit.',model_promoted=False))
    write(out/'jobs.json',dict(status='RUNNING',pid=os.getpid()))
    e=build(device,dt=result['dt_ms']);base={k:v for k,v in np.load(folder/'latest_state.npz').items()}
    T=result['period_ms'];rows=[];start=time.time()
    for shift,fraction in [(0,1),(0,.5),(100,1),(200,1)]:
        restore(e,base)
        for _ in range(shift//10):e.chunk()
        state=capture(e);r=fixed_return(e,state,T*fraction)
        row=dict(phase_shift_ms=shift,period_fraction=fraction,**r);rows.append(row)
        write(out/'returns.json',rows);log('SHOOTING INDEPENDENT',label,shift,fraction,r['combined_relative_rms'])
    first=rows[0];passed=first['combined_relative_rms']<1e-7 and max(v['relative_rms'] for v in first['blocks'].values())<1e-6
    write(out/'result.json',dict(status='SAME_PHASE_ROOT_VERIFIED' if passed else 'SAME_PHASE_ROOT_CHECK_FAILED',
        dt_ms=e.dt,period_ms=T,returns=rows,seconds=time.time()-start,
        scope='Shiftedphase andhalfperiod are diagnostics. No samebranch meshconvergence, Floquet orcriticalpoint certificate.',model_promoted=False))
    write(out/'jobs.json',dict(status='COMPLETE',pid=os.getpid()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--name',default='shooting_corrector_dt0p05')
    p.add_argument('--device',type=int,default=0);a=p.parse_args();audit(a.label,a.name,a.device)
