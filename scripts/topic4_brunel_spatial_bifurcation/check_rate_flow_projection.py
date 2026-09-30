"""Check the finite-history flow used to remove neutral phase growth."""
from rate_finite_lyapunov import *
from types import SimpleNamespace


def main(device):
    s=RateField();folder=RATE_OUT/'periodic_completion/finite_lyapunov/periodic_control'
    initial=np.load(folder/'state.npz');z=np.load(RATE_OUT/'periodic_completion/orbits/refined_J0.942000000_N1536.npz')
    r=z['r'];T=float(z['T']);J=float(z['J']);N=len(r);cf=np.fft.rfft(r,axis=0)/N
    lam=2j*np.pi*np.arange(len(cf))/T;factor=np.full(len(cf),2.);factor[[0,-1]]=1.
    y=initial['final_state'];w=np.array([1000.,1000.,1.,1.,1.,1.,.1,.1,1.])[:,None];rows=[]
    for dt in [.1,.05]:
        D=s.prep['max_delay_steps']*round(.1/dt);age=np.arange(D+1)*dt
        phase=np.exp(-age[:,None]*lam[None,:]);rh=(phase@(cf*factor[:,None])).real
        analytic=(phase@(cf*(factor*lam)[:,None])).real
        history=np.empty_like(rh);expected=np.empty_like(rh);slots=(-np.arange(D+1))%(D+1)
        history[slots]=rh;expected[slots]=analytic;history[0]=s.output(y)
        e=RateIntegrator(s,J,dt,initial=y,history=history,device=device)
        fy,fh=flow_direction(SimpleNamespace(engine=e));fy,fh=fy.get(),fh.get()
        norm=np.sqrt(np.sum((fy*w)**2)+np.sum((expected*1000)**2))
        err=np.linalg.norm((fh-expected)*1000)/norm
        coef=(np.sum((fy*w)**2)+np.sum(expected*fh)*1e6)/(np.sum((fy*w)**2)+np.sum(fh**2)*1e6)
        rem=np.sqrt(np.sum(((1-coef)*fy*w)**2)+np.sum(((expected-coef*fh)*1000)**2))/norm
        rows.append(dict(dt_ms=dt,history_flow_relative_defect=err,analytic_flow_remaining_after_projection=rem,
            current_rate_derivative_max_error=float(max(abs(fh[0]-expected[0])))))
        del e
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    reduction=rows[0]['history_flow_relative_defect']/rows[1]['history_flow_relative_defect']
    assert 3<reduction<5 and rows[-1]['history_flow_relative_defect']<1e-3
    write(RATE_OUT/'periodic_completion/finite_lyapunov/flow_projection_checks.json',dict(status='PASS',rows=rows,
        step_halving_error_reduction=reduction,scope='Exact Fourier history-flow derivative versus numerical full-history flow; local flow from the unchanged RHS. A periodic Lyapunov control is still required.'))
    print('FLOW PROJECTION CHECK',rows,'reduction',reduction,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);main(p.parse_args().device)
