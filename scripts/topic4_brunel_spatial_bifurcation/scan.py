"""Continuation of the actual spatial mean-field equilibrium equations."""
from common import *
from model import SpatialBrunel
from scipy.sparse.linalg import eigs
import argparse

def main(args):
    s=SpatialBrunel(args.grid);dest=OUT/f'g{args.grid}'/args.label;dest.mkdir(parents=True,exist_ok=True)
    r=None;rows=[]
    for J in np.arange(args.start,args.end+args.step/2,args.step):
        r,ok,trace=s.solve(float(J),r)
        row=dict(J_EE_core=J,converged=ok,residual_per_ms=trace[-1],rates_hz=s.regional_rates(r))
        if ok:
            jac=s.jacobian(r,float(J));ev,vec=eigs(jac,k=4,sigma=0,tol=1e-9)
            row['stationary_jacobian_eigenvalues']=ev
            np.savez_compressed(dest/f'J{J:.6f}.npz',rates=r,eigenvalues=ev,eigenvectors=vec)
        rows.append(row);write(dest/'result.json',dict(status='RUNNING',rows=rows,
            meaning='Stationary branch. Static Jacobian eigenvalues are not temporal growth rates.',
            space_cells=args.grid**2,rate_groups=s.P))
        print(row,flush=True)
        if not ok:break
    write(dest/'result.json',dict(status='COMPLETE',rows=rows,space_cells=args.grid**2,rate_groups=s.P,
        meaning='Stationary branch; dynamic susceptibility analysis required for stability'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--grid',type=int,default=20);p.add_argument('--start',type=float,default=0)
    p.add_argument('--end',type=float,default=1.8);p.add_argument('--step',type=float,default=.05);p.add_argument('--label',default='stationary_scan');main(p.parse_args())
