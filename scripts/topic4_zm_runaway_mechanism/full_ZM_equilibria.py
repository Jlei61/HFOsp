"""Stationary solutions with both Z and M autonomous in the frozen rate model.

Eliminate Z by its stationary resource equation and M by m=.5*E*r, then
solve the full steady rate equations. This is separate from held-Z branches.
Finding roots alone does not certify stability or basin structure.
"""
from common import *
from scipy.special import ndtr
from scipy.sparse.linalg import spsolve
from dynamics_v3 import THRESHOLD_Z


def main():
    s=model();mat=s.matrices();destination=OUT/'full_ZM_equilibria';destination.mkdir(exist_ok=True)
    def residual(r):
        ig=s.tm*s.area[1]*(mat[1]@r);vg=s.tm*s.area[1]**2*(mat[3]@r)
        sd=np.sqrt(np.maximum(s.tm*vg/(2*s.tau[1]),1e-20))
        Z=np.where(s.E,ndtr((THRESHOLD_Z-ig)/sd),1.)
        s.set_Z(Z,source='Autonomous Z stationary equation, not a prescribed parameter path')
        return r-s.phi(*s.moments(r))['rate']
    baseline=np.load(OUT/'periodic/rate_seed_N1024.npz')['r'].mean(0)
    residual(baseline);A=s.characteristic(baseline,0.,dynamic_z=True).real
    direction=np.random.default_rng(624).normal(size=s.P)*baseline
    errors=[]
    expected=A@direction
    for h in [1e-4,5e-5,1e-5]:
        fd=(residual(baseline+h*direction)-residual(baseline-h*direction))/(2*h)
        errors.append(dict(h=h,relative_error=float(np.linalg.norm(fd-expected)/np.linalg.norm(expected))))
    assert min(v['relative_error'] for v in errors)<1e-6,errors
    write(destination/'jacobian_check.json',dict(status='PASS',rows=errors,
        method='Analytic full dynamic-Z characteristic at lambda=0 vs finite difference of stationary-Z elimination'))
    late=np.load(OUT/'runs/endpoint_D0.1429804_dt0.05_rate_dynamicZ/trajectory.npz')['group_rate_hz'][-4000:].mean(0)/1000
    seeds=[('low',np.full(s.P,.001)),('cycle_mean',baseline),('released_high',late),('high',.9/s.ref)]
    rows=[];roots=[]
    for label,seed in seeds:
        r=np.array(seed,float);history=[]
        for iteration in range(60):
            f=residual(r);error=float(np.max(abs(f)));history.append(error)
            log('FULL ZM ROOT',label,iteration,error)
            if error<1e-11:break
            A=s.characteristic(r,0.,dynamic_z=True).real
            dr=spsolve(A,-f);norm0=np.linalg.norm(f)
            for back in range(22):
                trial=r+dr*2.**-back
                if trial.min()<0 or np.any(trial>1/s.ref):continue
                ff=residual(trial)
                if np.linalg.norm(ff)<norm0:r=trial;break
            else:break
        error=float(np.max(abs(residual(r))));passed=error<1e-11
        q=dict(seed=label,status='ROOT_CONVERGED' if passed else 'NEWTON_UNRESOLVED',
            residual_per_ms=error,iterations=history,global_E_hz=s.global_rate(r),D=s.D,
            region_Z=(np.array(s.regional_rates(s.Z))/1000).tolist(),stability='NOT_ESTABLISHED')
        if passed:
            Y=s.equilibrium_state(r);rhs,rr=s.rhs(Y,np.array([m@r for m in mat]),dynamic_z=True)
            q['full_state_rhs_max']=float(np.max(abs(rhs)))
            assert q['full_state_rhs_max']<1e-8
            duplicate=next((i for i,old in enumerate(roots) if np.linalg.norm(old-r)/max(np.linalg.norm(r),1e-12)<1e-6),None)
            q['duplicate_of_root_index']=duplicate
            if duplicate is None:roots.append(r.copy())
            file=destination/f'{label}.npz';np.savez_compressed(file,r=r,Z=s.Z,D=s.D,state=Y,residual=error)
            q['path']=str(file)
        rows.append(q)
        write(destination/'result.json',dict(status='RUNNING',rows=rows,Z='dynamic',M='dynamic'))
    write(destination/'result.json',dict(status='COMPLETE',rows=rows,distinct_roots_found=len(roots),
        Z='dynamic',M='dynamic',model='Unchanged frozen spatial rate equations',
        claim='Only stationary roots found from the named seeds. No completeness, stability, separatrix or onset bifurcation follows from root convergence alone.'))


if __name__=='__main__':main()
