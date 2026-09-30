"""Actual corrected-engine identity of the last converged low-branch seed."""
from common import OUT, np, read, write, log
from physical_delay_conditional_drift import PhysicalDelayConditionalDrift
from physical_delay_count_rate import PhysicalDelayCountEngine
from transient_response_network import install
from refractory_rate_response import covariance_matrices
from nonlinear_rate_response import normalized_input, SCALE
import gc

DEST=OUT/'transient_native_low_homotopy_20260923'


def main():
    result=read(DEST/'result.json');assert result['status']=='COMPLETE'
    s=PhysicalDelayConditionalDrift();checks=[]
    for row in result['rows']:
        z=np.load(DEST/f'point{row["index"]:03d}.npz');s.set_Z(z['Z'])
        err=float(abs(s.residual(z['r'])).max());assert err<1e-11
        checks.append(dict(index=row['index'],residual_per_ms=err))
    source=DEST/f'point{result["rows"][-1]["index"]:03d}.npz'
    data=np.load(source);r=data['r'];Z=data['Z'];s.set_Z(Z)
    a,b,qa,qb=s.matrices()
    F=s.tm*s.area[0]*(a@r)+s.private_mu;G=s.tm*s.area[1]*(b@r)
    ve=s.tm*s.area[0]**2*(qa@r)+s.private_ve;vi=s.tm*s.area[1]**2*(qb@r)
    syn=np.array([F,F,G,G,.5*s.E*r,Z]);x=np.column_stack([F-Z*G-syn[4],ve,Z**2*vi])
    u=normalized_input(x,s.theta)/SCALE;rows=[]
    for dt in [.05,.025]:
        e=PhysicalDelayCountEngine(dt=dt,device=1,count_sampling=False,constant_input=True);install(e)
        cp=e.cp;state=np.zeros((42,s.P))
        for pop,mask in [('E',s.E),('I',~s.E)]:
            A,B,C,_,_=covariance_matrices(pop,dt)
            for ch,v in enumerate([ve,vi]):
                state[3*ch:3*ch+3,mask]=np.linalg.solve(np.eye(3)-A[ch],B[ch])[:,None]*v[mask]
        for ch in range(3):state[6+12*ch:6+12*(ch+1)]=u[:,ch]
        e.syn[:]=cp.asarray(syn);e.local.state[:]=cp.asarray(state)
        e.local.history[:]=cp.asarray(r);e.local.rate[:]=cp.asarray(r);e.emitted[:]=cp.asarray(r)
        e.local.physical[:]=cp.asarray(np.array([x[:,0],ve,vi]));e.local.clock.fill(0)
        e.transport.pars[19].fill(0);e.transport.pars[20].fill(1)
        cp.cuda.get_current_stream().synchronize();e.step();cp.cuda.get_current_stream().synchronize()
        errors=[float(abs(actual-target).max()) for actual,target in
            [(e.local.rate.get(),r),(e.syn.get(),syn),(e.local.state.get(),state)]]
        assert errors[0]<1e-9 and max(errors[1:])<1e-8,(dt,errors)
        for _ in range(round(10/dt)-1):e.step()
        cp.cuda.get_current_stream().synchronize()
        tail=float(abs(e.local.history.get()-r).max());assert tail<1e-7
        assert np.array_equal(e.syn[5].get(),Z)
        rows.append(dict(dt_ms=dt,one_step_errors=errors,ten_ms_rate_history_error_per_ms=tail,
            transient_correction_installed=True,Z_held=True,M_dynamic=True))
        log('CORRECTED LOW ROOT FLOW',rows[-1])
        del e;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    write(DEST/'independent_audit.json',dict(status='PASS',all_saved_root_residuals=checks,
        flow_source=str(source),flow_checks=rows,scope='Static roots and stationary flow identity only; stability and any turning-point classification remain uncomputed.',model_promoted=False))


if __name__=='__main__':main()
