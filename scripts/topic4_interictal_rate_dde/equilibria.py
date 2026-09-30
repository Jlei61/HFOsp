"""Newton equilibria and verified characteristic roots of the spatial rate DDE.

This is a mathematical calculation for a candidate, not an acceptance of its
SNN correspondence. Root candidates from zero-delay ODE are refined against
all physical delays. Missing roots must not be called stable by omission.
"""
from common import *
from model import RateSystem
from scipy import sparse
from scipy.sparse.linalg import spsolve,eigs,ArpackNoConvergence
import argparse

def solve(s,J,r):
    trace=[]
    for it in range(50):
        F=s.equilibrium_residual(r,J);err=float(np.max(abs(F)));trace.append(err)
        if err<1e-10:return r,trace,True
        d=spsolve(s.equilibrium_jacobian(r,J).tocsc(),-F)
        if not np.isfinite(d).all():break
        step=1.
        for back in range(30):
            trial=r+step*d
            if trial.min()>=0 and np.all(trial<1/s.tref) and np.max(abs(s.equilibrium_residual(trial,J)))<err:
                r=trial;break
            step*=.5
        else:break
    return r,trace,False

def ode_jacobian(s,r,J):
    gain=s.phi(s.stationary_drive(r,J))[1];tau=s.response_tau(s.stationary_drive(r,J))[0]
    a,b=s.coupling(J);I=sparse.eye(s.P);zero=sparse.csr_matrix((s.P,s.P));dg=lambda x:sparse.diags(np.broadcast_to(x,(s.P,)))
    blocks=[[zero]*6 for _ in range(6)]
    blocks[0]=[dg(-1/tau),zero,dg(gain/tau),zero,dg(-gain*s.Z/tau),dg(-.0005*gain/tau)]
    blocks[1]=[dg(s.tm*s.area[0]/s.rise[0])@a,dg(-1/s.rise[0]),zero,zero,zero,zero]
    blocks[2]=[zero,dg(1/s.decay[0]),dg(-1/s.decay[0]),zero,zero,zero]
    blocks[3]=[dg(s.tm*s.area[1]/s.rise[1])@b,zero,zero,dg(-1/s.rise[1]),zero,zero]
    blocks[4]=[zero,zero,zero,dg(1/s.decay[1]),dg(-1/s.decay[1]),zero]
    blocks[5]=[dg(s.E.astype(float)),zero,zero,zero,zero,dg(-.001)]
    return sparse.bmat(blocks,format='csr')

def refine_root(s,r,J,lam,v):
    v=np.asarray(v,complex);pivot=np.argmax(abs(v));v/=v[pivot]
    constraint=sparse.csr_matrix((np.ones(1),(np.zeros(1,int),[pivot])),shape=(1,s.P))
    for it in range(20):
        if abs(lam)>2 or lam.real<-.5:return None
        M=s.characteristic(lam,r,J);res=M@v;err=np.linalg.norm(res)/np.linalg.norm(v)
        if err<1e-9:
            return dict(root_per_ms=lam,eigenvector=v/np.linalg.norm(v),relative_residual=float(err),iterations=it)
        col=s.characteristic(lam,r,J,True)@v
        A=sparse.bmat([[M,sparse.csr_matrix(col[:,None])],[constraint,sparse.csr_matrix((1,1))]],format='csc')
        delta=spsolve(A,np.r_[-res,0j]);v+=delta[:-1];lam+=delta[-1]
        if not np.isfinite(v).all() or not np.isfinite(lam):return None
    return None

def main(args):
    s=RateSystem(grid=args.grid,tau_e=5.,tau_i=5.,adaptive_tau=args.adaptive_tau);dest=BASE/'equilibria'/args.label;dest.mkdir(parents=True,exist_ok=True)
    s.Z[s.E]=args.z
    write(dest/'contract.json',dict(model='Spatial rate DDE',grid=args.grid,rate_groups=s.P,particle_count=0,
        Z_E=args.z,private_input='Original stationary Poisson statistics integrated in Phi',external_common_noise='Zero fluctuation for autonomous skeleton',
        J_values=args.values,state_dependent_tau=args.adaptive_tau,validation='Equilibrium residual and exact delay characteristic residual',
        limitation='Zero-delay eigenvalues provide search seeds only. No stable label without a converged rightmost-spectrum audit; no SNN mechanism claim before correspondence.'))
    r=s.phi(np.zeros(s.P))[0];rows=[]
    for J in args.values:
        r,trace,ok=solve(s,J,r)
        row=dict(J_EE_core=J,success=ok,residual_per_ms=trace[-1],newton_iterations=len(trace),newton_trace=trace)
        if ok:
            size=s.geo['group_size'];reg=s.geo['group_region'];row['core_rates_hz']=[float(np.average(r[s.E&(reg==k)],weights=size[s.E&(reg==k)])*1000) for k in range(3)]
            roots=[]
            if args.spectrum:
                try:ev,vec=eigs(ode_jacobian(s,r,J),k=8,which='LR',maxiter=2000,tol=1e-8)
                except ArpackNoConvergence as e:ev=e.eigenvalues;vec=e.eigenvectors
                row['zero_delay_seed_eigenvalues_per_ms']=[[q.real,q.imag] for q in ev]
                for k,lam in enumerate(ev):
                    if abs(lam+.001)<1e-7 or lam.imag<0:continue
                    q=refine_root(s,r,J,lam,vec[:s.P,k])
                    if q is None or any(abs(q['root_per_ms']-old['root_per_ms'])<1e-6 for old in roots):continue
                    roots.append(q)
                row['exact_delay_roots']=[dict(real_per_s=q['root_per_ms'].real*1000,imag_per_s=q['root_per_ms'].imag*1000,
                    frequency_hz=abs(q['root_per_ms'].imag)*1000/(2*np.pi),relative_residual=q['relative_residual']) for q in roots]
            np.savez_compressed(dest/f'J{J:g}.npz',rates_per_ms=r,Z=s.Z,**{f'eigenvector{k}':q['eigenvector'] for k,q in enumerate(roots)})
        rows.append(row);write(dest/'result.json',dict(status='RUNNING' if J!=args.values[-1] else 'COMPLETE',rows=rows,stability_complete=False))
        print({k:v for k,v in row.items() if k not in ['newton_trace','zero_delay_seed_eigenvalues_per_ms']},flush=True)
        if not ok:break

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--grid',type=int,default=20);ap.add_argument('--z',type=float,default=1.)
    ap.add_argument('--values',type=float,nargs='+',default=[0,.5,1.]);ap.add_argument('--label',required=True)
    ap.add_argument('--adaptive-tau',action='store_true');ap.add_argument('--spectrum',action='store_true');main(ap.parse_args())
