"""Check a proposed onset-side period against all persistent model states.

This supplies a shooting seed and a closure diagnostic at the current step.
It does not compute a Floquet multiplier or certify a continuous-time orbit.
"""
from common import np, read, write, log
from onset_state_continuation import DEST, build
from fine_rate_frozen_Z_fields import capture, restore
from datetime import datetime
from scipy.optimize import minimize_scalar
import argparse


def dynamical_state(state):
    tick = int(state['clock'][0])
    return dict(syn=state['syn'][:5], local=state['local'],
                history=state['history'][(tick-np.arange(len(state['history']))) % len(state['history'])])


def errors(reference, current, weights):
    blocks = [('AMPA', 'syn', slice(0,2)), ('GABA', 'syn', slice(2,4)),
              ('M', 'syn', slice(4,5)), ('covariance', 'local', slice(0,6)),
              ('input_memory', 'local', slice(6,42)),
              ('delay_refractory_history', 'history', slice(None))]
    rows = {}
    for name,key,indices in blocks:
        a,b = reference[key][indices],current[key][indices]
        rms = float(np.sqrt(((a*a)@weights).mean()))
        err = float(np.sqrt((((b-a)**2)@weights).mean()))
        rows[name] = dict(reference_rms=rms, difference_rms=err,
                          relative_rms=err/max(rms,1e-12))
    score = float(np.sqrt(np.mean([v['relative_rms']**2 for v in rows.values()])))
    return dict(blocks=rows, combined_relative_rms=score)


def run(label, device):
    folder = DEST/label
    jobs=read(folder/'jobs.json')
    assert jobs['status'] == 'COMPLETE'
    seed = read(folder/'rate_period_seed.json')
    out = folder/'period_return'
    out.mkdir(exist_ok=True)
    assert not (out/'result.json').exists()
    write(out/'contract.json', dict(created_local=datetime.now().astimezone().isoformat(),
        question='Does the candidate recurrence also close synapses, all local response memories, dynamic M and the complete delay/refractory history?',
        source=str(folder/'final_state.npz'), period_seed=seed,
        scope='One proposed return at the declared trajectory step; time interpolation only locates a shooting seed. No stability, time-step convergence or bifurcation certificate.',
        budget='One period plus1ms of actual unchanged conditional-drift integration.'))
    e=build(device,dt=jobs['condition'].get('dt_ms',.05))
    initial={k:v for k,v in np.load(folder/'final_state.npz').items()}
    restore(e,initial)
    assert not e.noise and not e.transport.drive_on
    assert np.all(e.transport.pars[19].get()==0) and np.all(e.transport.pars[20].get()==1)
    base=dynamical_state(initial)
    w=e.s.sizes/e.s.sizes.sum()
    T=seed['period_ms']; lo=int(np.floor((T-.5)/e.dt)); hi=int(np.ceil((T+.5)/e.dt))
    whole=lo//round(10/e.dt)
    for _ in range(whole): e.chunk()
    for _ in range(lo-whole*round(10/e.dt)): e.step()
    e.cp.cuda.get_current_stream().synchronize()
    records=[]; states=[]
    for n in range(lo,hi+1):
        if n>lo:
            e.step();e.cp.cuda.get_current_stream().synchronize()
        state=capture(e);current=dynamical_state(state)
        assert int(state['clock'][0])==int(initial['clock'][0])+n
        assert np.array_equal(state['syn'][5],initial['syn'][5])
        row=errors(base,current,w);row['elapsed_ms']=n*e.dt
        records.append(row);states.append(current)
    def interpolated(T):
        q=T/e.dt-lo;k=min(int(q),len(states)-2);a=q-k
        return {name:(1-a)*states[k][name]+a*states[k+1][name] for name in base}
    fit=minimize_scalar(lambda T:errors(base,interpolated(T),w)['combined_relative_rms'],
                        bounds=(lo*e.dt,hi*e.dt),method='bounded',options={'xatol':1e-9})
    fitted=errors(base,interpolated(fit.x),w)
    result=dict(status='RETURN_DIAGNOSTIC_COMPLETE',label=label,dt_ms=e.dt,
                seed_period_ms=T,grid_returns=records,interpolated_period_ms=float(fit.x),
                interpolated_return=fitted,
                scope='Full persistent-state closure diagnostic at one numerical step. The interpolation is not a solved periodic boundary-value problem; no certified stability or bifurcation.',
                model_promoted=False)
    write(out/'result.json',result)
    log('ONSET FULL STATE RETURN',label,float(fit.x),fitted)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--device',type=int,default=1)
    a=p.parse_args();run(a.label,a.device)
