"""Construct a solver guess for tonic A with a self-limited B burst cycle.

This is explicitly an artificial initial guess, not an observed trajectory or
accepted periodic state. Both sources have the identical physical Z field.
Only the initial Core A coordinates are transplanted; all subsequent flow
uses the unchanged full delayed equations and dynamic M.
"""
from common import OUT,np,read,write,log
from onset_state_continuation import build,regional_weights
from onset_variational_return import Coordinates
from onset_poincare_corrector import SectionReturn,regional_rate_section
from onset_cubic_section import CubicSectionReturn
from fine_rate_frozen_Z_fields import capture,restore
from onset_period_return import errors,dynamical_state
import argparse,os

ROOT=OUT/'core_a_bifurcation_type_20260924'


def main(device,mode):
    out=ROOT/('periodic_predictors/tonic_A_burst_B_'+mode)
    out.mkdir(parents=True,exist_ok=True);assert not(out/'jobs.json').exists()
    cycle=ROOT/'near_returns/mid_lower/coarse_below_SN_positive_cubic'
    tonic=ROOT/'near_returns/below_high_A/exact_short/state10115.npz'
    assert read(cycle/'result.json')['status']=='NUMERICAL_PERIODIC_ROOT'
    profile=np.load(cycle/'actual_orbit_profile/profile.npz')
    b=profile['regional_rate_hz'][:,2]
    crossings=np.flatnonzero((b[:-1]>50)&(b[1:]<=50))+1
    assert len(crossings)==1
    shift=float(profile['time_ms'][crossings[0]])
    write(out/'contract.json',dict(
        question='Can the same full equations support a cycle with tonic Core A and self-limited Core B, distinct from the already found low-A burst cycle?',
        source_cycle=str(cycle/'latest_state.npz'),source_tonic=str(tonic),
        phase_shift_ms=shift,phase_selection='Unique descending Core B 50Hz crossing in the already recorded low-A cycle.',
        construction=('Artificial root-solver guess: Core A coordinates from actual sustained-A source, all other coordinates from the phase-shifted low-A cycle.' if mode=='tonic_A' else 'Artificial root-solver guess: retain the actual sustained-A state including its surrounding feedback, replace only Core B coordinates by the descending-B phase of the short burst cycle.')+' All five synaptic/M,42 local and canonical delay-history coordinates of the chosen region are transplanted. Complete Z fields agree bitwise. No equation, connection, delay, M law, or parameter is altered.',
        subsequent_flow='All Z held, all M dynamic, same full3479-group autonomous conditional drift.',
        return_selection='First descending crossing of the same B rate plane after50ms, scanned up to800ms. This only chooses a numerical period seed, never a periodicity or stability classification.',
        status='ARTIFICIAL_SOLVER_GUESS_NOT_DATA',model_promoted=False))
    jobs=dict(status='RUNNING',pid=os.getpid());write(out/'jobs.json',jobs)
    e=build(device);base=dict(np.load(cycle/'latest_state.npz'));high=dict(np.load(tonic))
    assert np.array_equal(base['syn'][5],high['syn'][5])
    restore(e,base);n=round(shift/e.dt);whole,tail=divmod(n,round(10/e.dt))
    for _ in range(whole):e.chunk()
    for _ in range(tail):e.step()
    e.cp.cuda.get_current_stream().synchronize();base=capture(e)
    low=base
    if mode=='burst_B':base=high;other=low;region=1
    else:other=high;region=0
    c=Coordinates(base,e.s);x=c.pack(base).reshape(-1,e.s.P)
    h=c.pack(other).reshape(-1,e.s.P);core=e.s.E&(e.s.geo['group_region']==region)
    before=x.copy();x[:,core]=h[:,core]
    A=SectionReturn(base,e,float(profile['period_ms']),20.)
    state=A.state(x.ravel());state['rate']=state['history'][int(state['clock'][0])%len(state['history'])].copy()
    assert A.admissible(x.ravel())
    assert np.array_equal(x[:,~core],before[:,~core])
    np.savez_compressed(out/'state.npz',**state)
    W=regional_weights(e.s);B=W[2];level=float(state['rate']@B*1000)
    restore(e,state);rows=[];previous=level;chosen=None;last=0
    for tm in range(10,801,10):
        e.chunk();tick=int(e.local.clock.get()[0]);depth=len(state['history'])
        # The phase-shifted source need not lie on the10ms recorder boundary.
        # Read actual physical history samples in chronological order.
        index=(tick-round(1/e.dt)*np.arange(9,-1,-1))%depth
        rate=e.local.history[e.cp.asarray(index)].get()*1000
        regional=rate@W.T
        for k,r in enumerate(regional):
            t=tm-9+k;current=float(r[2]);rows.append([t,*r.tolist()])
            if t>50 and previous>level>=current:
                chosen=float(t-1+(previous-level)/(previous-current));break
            previous=current
        if chosen is not None:break
    np.savez_compressed(out/'guess_trajectory.npz',columns=['time_ms','global_hz','A_hz','B_hz','S_hz'],values=rows)
    result=dict(status='NO_RETURN_IN_DECLARED_WINDOW',B_level_hz=level,phase_shift_ms=shift,
        all_M_dynamic=True,all_Z_held=True,model_promoted=False)
    if chosen is not None:
        R=CubicSectionReturn(state,e,chosen,min(10.,chosen*.2));regional_rate_section(R,1,-1)
        try:
            y,meta=R(R.xref)
            error=errors(dynamical_state(state),dynamical_state(R.state(y)),e.s.sizes/e.s.sizes.sum())
            result.update(status='SOLVER_SEED_WITH_ACTUAL_RETURN_NOT_A_PERIODIC_ORBIT',**meta,**error)
        except RuntimeError as exc:
            result.update(status='SECTION_WINDOW_MISS_NOT_A_BIFURCATION',estimated_period_ms=chosen,error=str(exc))
    write(out/'result.json',result);jobs.update(status='COMPLETE');write(out/'jobs.json',jobs);log('TONIC A BURST B SEED',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1)
    p.add_argument('--mode',choices=['tonic_A','burst_B'],default='burst_B');a=p.parse_args();main(a.device,a.mode)
