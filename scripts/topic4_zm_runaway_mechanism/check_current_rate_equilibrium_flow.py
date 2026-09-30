"""One operating-state check of the new static interface in the actual engine.

This is a numerical identity test, not native validation or continuation.
"""
from common import OUT, np, write, read, log
from current_rate_characteristic import CurrentRateCharacteristic
from fine_rate_frozen_Z_fields import native_field
from refractory_rate_response import covariance_matrices
from nonlinear_rate_response import normalized_input, SCALE
from datetime import datetime
import argparse,os

DEST=OUT/'current_rate_analysis_interface'


def root():
    assert read(DEST/'static_implementation_check.json')['status']=='PASS'
    assert read(DEST/'temporal_implementation_check.json')['status']=='PASS'
    c=DEST/'single_equilibrium_flow_contract.json';assert not c.exists()
    write(c,dict(created_local=datetime.now().astimezone().isoformat(),
        question='Does one equilibrium of the current full-diffusion expected-rate equations remain invariant under the actual spatial integration step?',
        physical='Unchanged g40 original graph, constant original mean input, native9870ms full Z held; M dynamic and initialized at its equilibrium. Current conditioned39 response, no oldv3 gain.',
        solver='One Newton correction from the last1s mean rate of the completed native9870 held-field count trajectory; tolerance1e-11perms,max40iterations. Count result is initial guess only, not target or input.',
        scope='One equilibrium and numerical flow test atdt.05/.025; no branch, eigenvalue search, modelpromotion, fit or native onset claim. If correction fails, retain failure and stop.'))
    jobs=dict(status='RUNNING_ROOT',pid=os.getpid());write(DEST/'flow_jobs.json',jobs)
    s=CurrentRateCharacteristic(40);s.set_Z(native_field(s,9870))
    z=np.load(OUT/'fine_rate_frozen_Z_fields/native_Z9870_held/trajectory.npz')
    initial=z['group_rate_hz'][-1000:].astype(float).mean(0)/1000
    r,ok,trace=s.solve(initial,tol=1e-11,maxiter=40,verbose=True)
    err=float(abs(s.residual(r)).max())
    np.savez_compressed(DEST/'native9870_fullQ_equilibrium.npz',r=r,Z=s.Z,initial_guess=initial,trace=trace)
    result=dict(status='ROOT_PASS' if ok else 'ROOT_FAIL',residual_per_ms=err,iterations=len(trace),
        global_rate_hz=s.global_rate(r),regional_rates_hz=s.regional_rates(r),D=s.D,
        stability='NOT_COMPUTED',model_promoted=False)
    write(DEST/'single_equilibrium_root.json',result);write(DEST/'flow_jobs.json',dict(status=result['status'],pid=os.getpid()))
    log('CURRENT SINGLE EQUILIBRIUM',result)


def flow(device):
    assert read(DEST/'single_equilibrium_root.json')['status']=='ROOT_PASS'
    from refractory_spatial_resolution import FineEngine
    from fine_rate_frozen_Z_fields import capture
    import gc
    z=np.load(DEST/'native9870_fullQ_equilibrium.npz');r=z['r'];Z=z['Z'];rows=[]
    write(DEST/'flow_jobs.json',dict(status='RUNNING_FLOW_CHECK',pid=os.getpid()))
    for dt in [.05,.025]:
        e=FineEngine(dt=dt,drive='mean',noise=False,device=device);s=e.s;cp=e.cp;e.graph()
        s.set_Z(Z);a,b,qa,qb=s.matrices();F=s.tm*s.area[0]*(a@r)+s.private_mu;G=s.tm*s.area[1]*(b@r)
        ve=s.tm*s.area[0]**2*(qa@r)+s.private_ve;vi=s.tm*s.area[1]**2*(qb@r)
        syn=np.array([F,F,G,G,.5*s.E*r,Z]);state=np.zeros((42,s.P))
        x=np.column_stack([F-Z*G-syn[4],ve,Z**2*vi]);u=normalized_input(x,s.theta)/SCALE
        for pop,mask in [('E',s.E),('I',~s.E)]:
            A,B,C,_,_=covariance_matrices(pop,dt)
            for ch,v in enumerate([ve,vi]):
                fixed=np.linalg.solve(np.eye(3)-A[ch],B[ch])
                state[3*ch:3*ch+3,mask]=fixed[:,None]*v[mask]
        for ch in range(3):state[6+12*ch:6+12*(ch+1)]=u[:,ch]
        e.syn[:]=cp.asarray(syn);e.local.state[:]=cp.asarray(state)
        e.local.history[:]=cp.asarray(r);e.transport.history[:]=cp.asarray(r)
        e.local.rate[:]=cp.asarray(r);e.emitted[:]=cp.asarray(r)
        e.local.physical[:]=cp.asarray(np.array([x[:,0],ve,vi]));e.local.clock.fill(0)
        e.transport.pars[19].fill(0);e.transport.pars[20].fill(1)
        cp.cuda.get_current_stream().synchronize();e.step();cp.cuda.get_current_stream().synchronize()
        err_rate=float(np.max(abs(e.local.rate.get()-r)));err_syn=float(np.max(abs(e.syn.get()-syn)))
        err_local=float(np.max(abs(e.local.state.get()-state)))
        assert err_rate<1e-9 and err_syn<1e-8 and err_local<1e-8,(dt,err_rate,err_syn,err_local)
        # Subsequent10ms protects against an incorrectly populated history buffer.
        for _ in range(round(10/dt)-1):e.step()
        cp.cuda.get_current_stream().synchronize()
        history_error=float(np.max(abs(e.transport.history.get()-r)))
        final_rate_error=float(np.max(abs(e.local.rate.get()-r)))
        assert history_error<1e-7 and final_rate_error<1e-7,(dt,history_error,final_rate_error)
        rows.append(dict(dt_ms=dt,one_step_rate_error_per_ms=err_rate,one_step_syn_error_mV=err_syn,
            one_step_local_state_error=err_local,ten_ms_history_error_per_ms=history_error,
            ten_ms_final_rate_error_per_ms=final_rate_error,full_Z_held=bool(np.array_equal(e.syn[5].get(),Z)),
            full_diffusion=True,M_dynamic=True))
        assert rows[-1]['full_Z_held'];log('CURRENT EQUILIBRIUM FLOW PASS',rows[-1])
        del e;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    write(DEST/'single_equilibrium_flow_check.json',dict(status='PASS',rows=rows,
        source='native9870_fullQ_equilibrium.npz',scope='Numerical equilibrium identity atonefield andtwosteps, not stability or native correspondence.'))
    write(DEST/'flow_jobs.json',dict(status='COMPLETE_ROOT_AND_FLOW_CHECK_PASS',pid=os.getpid()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['root','flow']);p.add_argument('--device',type=int,default=1);a=p.parse_args()
    root() if a.command=='root' else flow(a.device)
