"""Check physical-delay indexing, J/J-squared scaling and a full Heun step.

The CPU reference reads the original sparse delayed matrices directly. It
does not reuse the CUDA arrival kernel or the Fourier-periodic operator.
This is an implementation test, not a native SNN correspondence assay.
"""
from rate_field import *
import argparse


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0)
    args=p.parse_args();s=RateField();rng=np.random.default_rng(912071);rows=[]
    raw=[sparse.load_npz(s.folder/f'{name}.npz').tocoo() for name in
         ['mean_ampa','mean_gaba','variance_ampa','variance_gaba']]
    assert np.all(s.Z==1), 'This CUDA implementation is for the frozen Z=1 model'
    for J,dt in [(.6,.1),(.942,.05),(1.3,.1),(2.,.05)]:
        r=rng.uniform(.0001,.015,s.P)
        y=s.equilibrium_state(r,J)*rng.uniform(.85,1.15,(9,s.P))
        e=RateIntegrator(s,J,dt=dt,initial=y,device=args.device);cp=e.cp
        history=rng.uniform(.0001,.015,(e.depth,s.P))
        e.history[:]=cp.asarray(history);e.tick=e.depth+3
        def arrival(tick):
            result=[]
            for k,a in enumerate(raw):
                src=a.col%s.P;di=a.col//s.P
                core=s.E[a.row]&s.E[src]&(s.geo['group_region'][a.row]<2)&(s.geo['group_region'][a.row]==s.geo['group_region'][src])
                scale=np.where(core,J**(1 if k==0 else 2),1.) if k in (0,2) else 1.
                values=a.data*scale*history[(tick-(di+1)*e.factor)%e.depth,src]
                result.append(np.bincount(a.row,weights=values,minlength=s.P))
            return np.array(result)
        a0=arrival(e.tick);a1=arrival(e.tick+1);e.arrivals(e.tick)
        got=e.arr.get();arrival_error=np.linalg.norm(got-a0)/np.linalg.norm(a0)
        f=s.rhs(y,a0);pred=y+dt*f;expected=y+.5*dt*(f+s.rhs(pred,a1))
        e.step();actual=e.y.get()
        step_error=np.linalg.norm(actual-expected)/np.linalg.norm(expected)
        increment_error=np.linalg.norm(actual-expected)/np.linalg.norm(expected-y)
        by_state=np.max(abs(actual-expected),axis=1)
        output_error=float(np.max(abs(e.history[e.tick%e.depth].get()-s.output(expected))))
        assert arrival_error<1e-12,(J,dt,arrival_error)
        assert increment_error<1e-10,(J,dt,increment_error)
        assert output_error<1e-12,(J,dt,output_error)
        rows.append(dict(J_EE_core=J,dt_ms=dt,history_depth=e.depth,
            history='Nonconstant independent positive rates, every group and delay slot',
            arrivals_relative_error=arrival_error,state_relative_error=step_error,
            increment_relative_error=increment_error,max_abs_state_error=by_state,
            output_history_max_abs_error=output_error))
        print('NONCONSTANT DELAY CHECK',rows[-1],flush=True)
        del e;cp.get_default_memory_pool().free_all_blocks()
    write(RATE_OUT/'model_audit_20260918/nonconstant_delay_history_checks.json',
          dict(status='PASS',rows=rows,equations_changed=False,
               scope='CPU/GPU physical-delay arrivals and one full Heun step at four parameter/step pairs. Tests implementation identity; not long-time convergence, SNN equivalence, or bifurcation completeness.'))


if __name__=='__main__':main()
