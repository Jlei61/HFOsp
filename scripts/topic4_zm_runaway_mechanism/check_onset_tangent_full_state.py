"""Independent full-state finite differences of the onset variational flow."""
from common import np, read, write, log
from onset_state_continuation import DEST,build
from fine_rate_frozen_Z_fields import capture,restore
from onset_tangent_cuda import Tangent
import argparse


def check(device):
    folder=DEST/'tangent_implementation'
    assert read(folder/'result.json')['status']=='PASS'
    assert not (folder/'full_state_check.json').exists()
    e=build(device)
    reference={k:v for k,v in np.load(DEST/'lower_endpoint/final_state.npz').items()}
    restore(e,reference)
    # A second phase, reached by the actual unmodified flow.
    for _ in range(10):e.chunk()
    base=capture(e);t=Tangent(e);t.graph()
    rng=np.random.default_rng(92317)
    d={key:rng.uniform(-1,1,base[key].shape)*base[key] for key in ['syn','local','history']}
    d['syn'][5]=0;d['syn'][4,~e.s.E]=0
    # Keep a finite central perturbation safely within the refractory simplex.
    clock=int(base['clock'][0]);h=base['history'];depth=len(h)
    slack=np.ones(e.s.P)
    for mask,ref in [(e.s.E,2.),(~e.s.E,1.)]:
        used=h[(clock+1-np.arange(1,round(ref/e.dt)))%depth][:,mask].sum(0)*e.dt
        slack[mask]=np.minimum(1.,np.maximum(1.-used,0.))
    d['history']*=slack
    restore(e,base);t.reset()
    t.syn[:]=e.cp.asarray(d['syn'][:5]);t.local[:]=e.cp.asarray(d['local']);t.history[:]=e.cp.asarray(d['history'])
    e.cp.cuda.get_current_stream().synchronize();t.chunk()
    tangent=dict(syn=t.syn.get(),local=t.local.get(),history=t.history.get())
    rows=[]
    for eps in [1e-4,5e-5,2.5e-5,1.25e-5]:
        states=[]
        for sign in [-1,1]:
            perturbed=dict(base)
            for key in d:perturbed[key]=base[key]+sign*eps*d[key]
            restore(e,perturbed);e.chunk();state=capture(e)
            assert all(np.isfinite(state[key]).all() for key in tangent)
            states.append(state)
        errors={}
        for key,v in tangent.items():
            fd=(states[1][key]-states[0][key])/(2*eps)
            if key=='syn':fd=fd[:5]
            errors[key]=float(np.linalg.norm(fd-v)/max(np.linalg.norm(v),1e-12))
        rows.append(dict(epsilon=eps,relative_errors=errors))
        log('ONSET FULL STATE TANGENT',eps,errors)
    passed=max(rows[-1]['relative_errors'].values())<1e-4
    write(folder/'full_state_check.json',dict(status='PASS' if passed else 'FAIL',
        phase_advance_ms=100,dt_ms=e.dt,propagation_ms=10,rows=rows,
        perturbed=['AMPA','GABA','M','six raw covariances','36 input memories','full delayed/refractory rate history'],
        scope='Implementation verification; no orbit stability or onset classification.'))
    assert passed


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);a=p.parse_args();check(a.device)
