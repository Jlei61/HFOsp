"""Bounded equilibrium search near the new high-B-background H1 endpoint.

Initialization is taken from a computed cycle. The solved equations, weights,
input and geometry remain frozen. A failed least-squares solve is not a root.
"""
from rate_field import *
from scipy.optimize import least_squares
import argparse
import os


def main():
    p=argparse.ArgumentParser();p.add_argument('--max-nfev',type=int,default=80);a=p.parse_args()
    s=RateField();folder=RATE_OUT/'periodic_completion'
    source=folder/'orbits/arcAconnectionStage4_0159_N2048.npz'
    z=np.load(source);J=float(z['J']);r=z['r'];reg=np.array([s.regional_rates(v) for v in r])
    seeds=[('cycle_mean',r.mean(0)),('low_A_phase',r[np.argmin(reg[:,0])])]
    output=folder/'H1_highB_equilibrium_trust_region_probe.json'
    assert not output.exists(),'Preserve an existing bounded search'
    rows=[];accepted=[]
    def record(status,**kw):
        write(output,dict(status=status,pid=os.getpid(),source=str(source),J_EE_core=J,rows=rows,
            equations_changed=False,scope='Bounded positive equilibrium search at a single parameter. A converged root requires a fresh residual and finite-difference Jacobian check; no spectral or branch identity claim.',**kw))
    for name,seed in seeds:
        calls=0
        def fun(x):
            nonlocal calls
            calls+=1;f=s.residual(x,J)*1000
            if calls%10==1:record('SEARCHING',seed=name,evaluations=calls,current_residual_Hz=float(abs(f).max()))
            return f
        def jac(x):return s.jacobian(x,J)*1000
        fit=least_squares(fun,np.clip(seed,1e-12,1/s.ref-1e-12),jac=jac,
            bounds=(np.zeros(s.P),1/s.ref),method='trf',tr_solver='lsmr',x_scale='jac',
            max_nfev=a.max_nfev,ftol=None,xtol=1e-12,gtol=1e-10,
            tr_options=dict(atol=1e-10,btol=1e-10,maxiter=500))
        residual=float(abs(s.residual(fit.x,J)).max()*1000)
        good=residual<1e-7 and fit.x.min()>=0 and np.all(fit.x<=1/s.ref)
        row=dict(seed=name,scipy_termination=int(fit.status),evaluations=int(fit.nfev),
            maximum_residual_Hz=residual,status='EQUILIBRIUM_CANDIDATE' if good else 'FAILED_TRIAL',
            candidate_regional_rates_Hz=s.regional_rates(fit.x) if good else None,
            failed_trial_regional_rates_Hz=s.regional_rates(fit.x) if not good else None)
        if good:
            direction=np.random.default_rng(3915).normal(size=s.P)*1e-6
            exact=s.jacobian(fit.x,J)@direction
            numeric=(s.residual(fit.x+1e-3*direction,J)-s.residual(fit.x-1e-3*direction,J))/(2e-3)
            error=float(np.linalg.norm(numeric-exact)/np.linalg.norm(exact))
            row['Jacobian_action_relative_error']=error
            row['status']='EQUILIBRIUM_VERIFIED' if error<1e-5 else 'EQUILIBRIUM_JACOBIAN_CHECK_REQUIRED'
            accepted.append(fit.x)
        rows.append(row);record('SEED_FINISHED',seed=name);print(row,flush=True)
    if accepted:
        np.savez_compressed(folder/'H1_highB_equilibrium_trust_region_roots.npz',rates=np.array(accepted),J=J)
    record('SEARCH_FINISHED',verified_roots=sum(q['status']=='EQUILIBRIUM_VERIFIED' for q in rows))


if __name__=='__main__':main()
