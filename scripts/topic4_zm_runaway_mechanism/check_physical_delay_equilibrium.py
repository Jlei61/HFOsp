"""One numerical equilibrium/flow identity test, not an onset branch search."""
from common import OUT, np, read, write, log
from physical_delay_conditional_drift import PhysicalDelayConditionalDrift
from datetime import datetime
import argparse, os

DEST = OUT/'physical_delay_conditional_drift_interface'


def root():
    assert read(DEST/'implementation_check.json')['status'] == 'PASS'
    path = DEST/'baseline_equilibrium_contract.json'; assert not path.exists()
    write(path, dict(created_local=datetime.now().astimezone().isoformat(),
        question='Does the repaired private-variance analytic interface supply a baseline fixed point that the actual corresponding integration map preserves?',
        model='Current conditioned39, g40, physical private-Q delay split, constant original mean drive; complete Z=1 held, M dynamic.',
        budget='One baseline Newton solve, default uncoupled-response initial guess, at most40iterations; if converged verify direct residual and actualflow atdt.05/.025. No parameter sweep, onset label, modelpromotion, response fit or branch search.',
        rationale='Earlier failed roots were a different full-Q object and a high-activity initial guess. This is a baseline identity test of the repaired conditional drift; it does not waive native or local validation failures.'))
    write(DEST/'baseline_equilibrium_jobs.json', dict(status='RUNNING_ROOT', pid=os.getpid()))
    s = PhysicalDelayConditionalDrift(); s.set_Z(np.ones(s.P))
    initial = s.phi(*s.moments(np.zeros(s.P)))['rate']
    r, ok, trace = s.solve(initial, maxiter=40, tol=1e-11, verbose=True)
    residual = float(abs(s.residual(r)).max())
    np.savez_compressed(DEST/'baseline_equilibrium.npz', r=r, Z=s.Z, initial_guess=initial, trace=trace)
    result = dict(status='ROOT_PASS' if ok else 'ROOT_FAIL', residual_per_ms=residual,
        iterations=len(trace), global_rate_hz=s.global_rate(r), regional_rates_hz=s.regional_rates(r),
        D=s.D, stability='NOT_COMPUTED', model_promoted=False)
    write(DEST/'baseline_equilibrium_result.json', result)
    write(DEST/'baseline_equilibrium_jobs.json', dict(status=result['status'], pid=os.getpid()))
    log('PHYSICAL PRIVATE BASELINE EQUILIBRIUM', result)


def flow(device):
    assert read(DEST/'baseline_equilibrium_result.json')['status'] == 'ROOT_PASS'
    from physical_delay_count_rate import PhysicalDelayCountEngine
    from refractory_rate_response import covariance_matrices
    from nonlinear_rate_response import normalized_input, SCALE
    import gc
    data=np.load(DEST/'baseline_equilibrium.npz'); r=data['r']; Z=data['Z']; rows=[]
    assert np.array_equal(Z, np.ones(len(Z)))
    s=PhysicalDelayConditionalDrift(); s.set_Z(Z)
    a,b,qa,qb=s.matrices()
    F=s.tm*s.area[0]*(a@r)+s.private_mu; G=s.tm*s.area[1]*(b@r)
    ve=s.tm*s.area[0]**2*(qa@r)+s.private_ve; vi=s.tm*s.area[1]**2*(qb@r)
    syn=np.array([F,F,G,G,.5*s.E*r,Z]); x=np.column_stack([F-Z*G-syn[4],ve,Z**2*vi])
    u=normalized_input(x,s.theta)/SCALE
    write(DEST/'baseline_equilibrium_jobs.json',dict(status='RUNNING_FLOW',pid=os.getpid()))
    for dt in [.05,.025]:
        e=PhysicalDelayCountEngine(dt=dt,device=device,count_sampling=False,constant_input=True)
        cp=e.cp; assert not e.noise and not e.transport.drive_on
        assert e.local.history.data.ptr == e.transport.history.data.ptr
        state=np.zeros((42,s.P))
        for pop,mask in [('E',s.E),('I',~s.E)]:
            A,B,C,_,_=covariance_matrices(pop,dt)
            for ch,v in enumerate([ve,vi]):
                state[3*ch:3*ch+3,mask]=np.linalg.solve(np.eye(3)-A[ch],B[ch])[:,None]*v[mask]
        for ch in range(3): state[6+12*ch:6+12*(ch+1)]=u[:,ch]
        e.syn[:]=cp.asarray(syn); e.local.state[:]=cp.asarray(state)
        e.local.history[:]=cp.asarray(r); e.local.rate[:]=cp.asarray(r); e.emitted[:]=cp.asarray(r)
        e.local.physical[:]=cp.asarray(np.array([x[:,0],ve,vi])); e.local.clock.fill(0)
        e.transport.pars[19].fill(0); e.transport.pars[20].fill(1)
        cp.cuda.get_current_stream().synchronize(); e.step(); cp.cuda.get_current_stream().synchronize()
        error=[float(np.max(abs(actual-target))) for actual,target in
               [(e.local.rate.get(),r),(e.syn.get(),syn),(e.local.state.get(),state)]]
        assert error[0]<1e-9 and max(error[1:])<1e-8, (dt,error)
        for _ in range(round(10/dt)-1): e.step()
        cp.cuda.get_current_stream().synchronize()
        tail=float(np.max(abs(e.transport.history.get()-r)))
        assert tail<1e-7, (dt,tail)
        assert np.array_equal(e.syn[5].get(),Z)
        rows.append(dict(dt_ms=dt,one_step_rate_error_per_ms=error[0],one_step_syn_error_mV=error[1],
            one_step_local_error=error[2],ten_ms_history_error_per_ms=tail,Z_held=True,M_dynamic=True))
        log('PHYSICAL PRIVATE BASELINE FLOW PASS',rows[-1])
        del e; gc.collect(); cp.get_default_memory_pool().free_all_blocks()
    write(DEST/'baseline_equilibrium_flow_check.json',dict(status='PASS',rows=rows,D=0.,
        global_rate_hz=s.global_rate(r),scope='One baseline equilibrium identity at two steps in the actual corrected private-Q engine. No stability or bifurcation claim; native/local correspondence still required.',model_promoted=False))
    write(DEST/'baseline_equilibrium_jobs.json',dict(status='COMPLETE_ROOT_AND_FLOW_PASS',pid=os.getpid()))


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('command', choices=['root','flow']); p.add_argument('--device',type=int,default=0); a=p.parse_args()
    root() if a.command=='root' else flow(a.device)
