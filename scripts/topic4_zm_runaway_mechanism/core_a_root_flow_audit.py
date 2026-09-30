"""Independently embed an actual static root in the complete current flow."""
from common import OUT,np,read,write,log
from physical_delay_conditional_drift import PhysicalDelayConditionalDrift
from onset_state_continuation import build
from refractory_rate_response import covariance_matrices
from nonlinear_rate_response import normalized_input,SCALE
from pathlib import Path
import argparse,gc


def main(a):
    source=Path(a.source);data=np.load(source);s=PhysicalDelayConditionalDrift();s.set_Z(data['Z']);r=data['r']
    residual=float(abs(s.residual(r)).max());assert residual<1e-11
    aa,b,qa,qb=s.matrices();F=s.tm*s.area[0]*(aa@r)+s.private_mu;G=s.tm*s.area[1]*(b@r)
    ve=s.tm*s.area[0]**2*(qa@r)+s.private_ve;vi=s.tm*s.area[1]**2*(qb@r)
    syn=np.array([F,F,G,G,.5*s.E*r,s.Z]);x=np.column_stack([F-s.Z*G-syn[4],ve,s.Z**2*vi]);u=normalized_input(x,s.theta)/SCALE;rows=[]
    for dt in [.05,.025]:
        e=build(a.device,dt=dt);cp=e.cp;state=np.zeros((42,s.P))
        for pop,mask in [('E',s.E),('I',~s.E)]:
            A,B,C,_,_=covariance_matrices(pop,dt)
            for ch,value in enumerate([ve,vi]):state[3*ch:3*ch+3,mask]=np.linalg.solve(np.eye(3)-A[ch],B[ch])[:,None]*value[mask]
        for ch in range(3):state[6+12*ch:6+12*(ch+1)]=u[:,ch]
        e.syn[:]=cp.asarray(syn);e.local.state[:]=cp.asarray(state);e.local.history[:]=cp.asarray(r)
        e.local.rate[:]=cp.asarray(r);e.emitted[:]=cp.asarray(r);e.local.physical[:]=cp.asarray(np.array([x[:,0],ve,vi]));e.local.clock.fill(0)
        e.transport.pars[19].fill(0);e.transport.pars[20].fill(1);cp.cuda.get_current_stream().synchronize()
        e.step();cp.cuda.get_current_stream().synchronize()
        errors={k:float(abs(actual-target).max()) for k,actual,target in [('rate_per_ms',e.local.rate.get(),r),('syn',e.syn.get(),syn),('local',e.local.state.get(),state)]}
        assert errors['rate_per_ms']<1e-9 and max(errors.values())<1e-8,errors
        for _ in range(round(10/dt)-1):e.step()
        cp.cuda.get_current_stream().synchronize();tail=float(abs(e.local.history.get()-r).max());assert tail<1e-7,tail
        assert np.array_equal(e.syn[5].get(),s.Z)
        rows.append(dict(dt_ms=dt,one_step_errors=errors,ten_ms_rate_error_per_ms=tail,Z_held=True,M_dynamic=True));log('CORE A ROOT FLOW AUDIT',rows[-1])
        del e;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    write(source.parent/'flow_audit.json',dict(status='PASS',source=str(source),static_residual_per_ms=residual,rows=rows,
        scope='Actual corrected complete-model stationary flow identity at two steps; stability and onset attribution are separate.',model_promoted=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--device',type=int,default=0);main(p.parse_args())
