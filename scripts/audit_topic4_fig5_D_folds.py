"""Refine old turning points with a smooth transfer and verify fold conditions."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,argparse
import numpy as np
from scipy.linalg import eig,solve,svdvals
from scipy.optimize import root_scalar
from topic4_fig5_D_physical_model import Equilibrium,Characteristic,OUT,ROOT,smooth
from numpy.polynomial.hermite import hermgauss

def audit(name,path,q=1.,gh_nodes=15):
    a=np.load(path);base=a['r_hz'];D0=float(a['s'] if 's' in a else a['D']);eq=Equilibrium(q);m=eq.m
    if gh_nodes!=15:
        gh_x,gh_w=hermgauss(gh_nodes);m.gh_x=gh_x;m.gh_w=gh_w/np.sqrt(np.pi);smooth(m)
    if 'right_mode' in a:v=a['right_mode']
    else:
        f,j=eq.evaluate(base,D0,True);e,vs=eig(j);v=vs[:,np.argmin(abs(e))].real
    v/=np.linalg.norm(v);saved={}
    def at(t):
        if t in saved:return saved[t]
        r=base+t*v;D=D0
        for it in range(12):
            f,j=eq.evaluate(r,D,True);h=1e-6
            fp=(eq.evaluate(r,D+h)-eq.evaluate(r,D-h))/(2*h)
            B=np.zeros((801,801));B[:800,:800]=j;B[:800,800]=fp;B[800,:800]=v
            ff=np.r_[f,np.dot(r-base,v)-t]
            if max(abs(ff))<2e-10:break
            delta=solve(B,-ff,check_finite=False);r+=delta[:800];D+=delta[-1]
        tangent=solve(B,np.r_[np.zeros(800),1.],check_finite=False)
        assert max(abs(ff))<1e-7,(name,t,max(abs(ff)))
        saved[t]=(tangent[-1],r,D)
        print(name,'CONTROL',t,'D',D,'dD/da',tangent[-1],flush=True)
        return saved[t]
    sol=root_scalar(lambda t:at(t)[0],x0=0.,x1=.1,xtol=1e-6,maxiter=15)
    _,r,D=at(sol.root);f,j=eq.evaluate(r,D,True)
    es,left,right=eig(j,left=True,right=True);order=np.argsort(abs(es));idx=order[0]
    v=right[:,idx].real;v/=np.linalg.norm(v);w=left[:,idx].real;w/=w@v
    checks=[]
    for h in (.4,.2,.1,.05,.02,.01):
        # Differencing analytic Jacobians is more accurate than cancellation
        # of three nearly identical large residuals at the high-rate folds.
        jp=eq.evaluate(r+h*v,D,True)[1];jm=eq.evaluate(r-h*v,D,True)[1]
        coef=float(.5*w@((jp-jm)@v/(2*h)))
        checks.append(dict(step_hz=h,quadratic_coefficient=coef))
    hd=1e-6;trans=float(w@(eq.evaluate(r,D+hd)-eq.evaluate(r,D-hd))/(2*hd))
    c=Characteristic(q)
    if gh_nodes!=15:
        c.m.gh_x=gh_x;c.m.gh_w=gh_w/np.sqrt(np.pi);smooth(c.m)
    c.at(r,D);sv=svdvals(c.matrix(0.));h=1e-3
    # Simplicity in dynamic lambda, not just in the stationary residual.
    ev,ll,rr=eig(c.matrix(0.),left=True,right=True);ix=np.argmin(abs(ev));rv=rr[:,ix];lv=ll[:,ix];lv/=np.vdot(lv,rv).conjugate()
    dynamic_derivative=complex(np.vdot(lv,(c.matrix(h)-c.matrix(-h))@rv/(2*h)))
    energy=v[:m.n]**2*m.count_e;shares=[float(np.sum(v[:m.n]**2*m.region_w[f'175_{i}'])/energy.sum()) for i in range(3)]
    last=[x['quadratic_coefficient'] for x in checks[-3:]]
    converged=(max(last)-min(last))/max(abs(np.mean(last)),1e-20)<.02
    passed=bool(converged and abs(trans)>1e-5 and abs(last[-1])>1e-7 and abs(es[idx])<1e-6 and abs(es[order[1]])>1e-4 and sv[-1]<1e-7 and sv[-2]>1e-5 and abs(dynamic_derivative)>1e-7)
    report=dict(name=name,q_ie=q,GH_nodes=gh_nodes,D=D,mean_e_hz=float(np.average(r[:m.n],weights=m.count_e)),
      type='SADDLE_NODE_OF_EQUILIBRIA' if passed else 'UNRESOLVED',residual_max_hz=float(max(abs(f))),
      critical_stationary_eigenvalue=[es[idx].real,es[idx].imag],second_eigenvalue_modulus=float(abs(es[order[1]])),
      transversality=trans,quadratic_checks=checks,quadratic_converged=bool(converged),
      characteristic_zero_singular_values=sv[-2:].tolist(),dynamic_lambda_derivative=[dynamic_derivative.real,dynamic_derivative.imag],
      mode_E_energy_A_B_surround=shares,source=str(path),full_stability='PENDING',
      physical=bool(0<=D<=1),numerical_primitive='C2 Hermite antiderivative, independent integration audited')
    np.savez_compressed(OUT/f'{name}.npz',r_hz=r,D=D,right_mode=v,left_mode=w)
    (OUT/f'{name}.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name');p.add_argument('--path');p.add_argument('--q',type=float,default=1.);p.add_argument('--gh-nodes',type=int,default=15);a=p.parse_args()
    if a.path:audit(a.name,a.path,a.q,a.gh_nodes)
    else:
        for name,sub in [('old_F0','fig5_z_bifurcation_preview_20260915/low_fold_state.npz'),('old_TP','fig5_z_branch_extension_20260915/physical_secondary_fold_state.npz'),('old_TP2','fig5_D_fast_slow_20260916/target_stationary_fold_state.npz')]:
            audit(name,ROOT/'results/topic4_sef_hfo'/sub)
