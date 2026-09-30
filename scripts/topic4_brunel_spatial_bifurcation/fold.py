"""Augmented stationary fold solve F=0, F_r v=0, v^T v=1."""
from common import *
from model import SpatialBrunel
from scipy.sparse.linalg import eigs,spsolve
import argparse

def main(args):
    s=SpatialBrunel(args.grid);dest=OUT/args.label if getattr(args,'label',None) else OUT/f'g{args.grid}'/'fold';dest.mkdir(parents=True,exist_ok=True)
    if getattr(args,'seed',None):
        initial=np.load(args.seed);r=initial['rates'];J=float(initial['J']);r,ok,_=s.solve(J,r)
    else:r,ok,_=s.solve(args.start);J=args.start
    assert ok
    ev,vec=eigs(s.jacobian(r,J),k=1,sigma=0,tol=1e-11);v=vec[:,0].real;v/=np.linalg.norm(v)
    trace=[]
    for k in range(18):
        A=s.jacobian(r,J);F=s.residual(r,J);aug=np.r_[F,A@v,v@v-1]
        err=float(abs(aug).max());trace.append(dict(iteration=k,J=J,error=err));print(trace[-1],flush=True)
        if err<1e-9:break
        h=1e-6;H=(s.jacobian(r+h*v,J)-s.jacobian(r-h*v,J))/(2*h)
        jj=(s.jacobian(r,J+h)-s.jacobian(r,J-h))@v/(2*h);pj=s.parameter_derivative(r,J)
        zero=sparse.csr_matrix((s.P,s.P));col=lambda x:sparse.csr_matrix(x[:,None])
        mat=sparse.bmat([[A,col(pj),zero],[H,col(jj),A],[sparse.csr_matrix((1,s.P)),sparse.csr_matrix((1,1)),sparse.csr_matrix(2*v[None,:])]],format='csc')
        step=spsolve(mat,-aug);alpha=1.
        for back in range(25):
            rr=r+alpha*step[:s.P];j=J+alpha*step[s.P];vv=v+alpha*step[s.P+1:]
            f=np.r_[s.residual(rr,j),s.jacobian(rr,j)@vv,vv@vv-1]
            if rr.min()>=-1e-12 and np.linalg.norm(f)<np.linalg.norm(aug):break
            alpha*=.5
        else:raise RuntimeError('fold Newton line search failed')
        r=rr;J=j;v=vv
    assert err<1e-9
    A=s.jacobian(r,J);el,vl=eigs(A.T,k=1,sigma=0,tol=1e-11);left=vl[:,0].real;left/=left@v
    H=(s.jacobian(r+1e-6*v,J)-s.jacobian(r-1e-6*v,J))/(2e-6)
    size=s.geo['group_size'];reg=s.geo['group_region'];energy=size*abs(v)**2;energy/=energy.sum()
    out=dict(J_EE_core=J,rates_hz=s.regional_rates(r),equilibrium_residual=float(abs(s.residual(r,J)).max()),
        eigenvector_residual=float(np.linalg.norm(A@v)),transversality=float(left@s.parameter_derivative(r,J)),quadratic_coefficient=float(left@(H@v)/2),
        regional_energy=[float(energy[s.E&(reg==k)].sum()) for k in range(3)],inhibitory_energy=float(energy[~s.E].sum()),
        grid=args.grid,space_cells=args.grid**2,rate_groups=s.P,trace=trace,
        classification='Nondegenerate stationary saddle-node of the spatial Brunel diffusion closure; not a certified native-SNN onset')
    np.savez_compressed(dest/'fold.npz',rates=r,J=J,right=v,left=left)
    write(dest/'result.json',out);print(out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--grid',type=int,default=20);p.add_argument('--start',type=float,default=1.016)
    p.add_argument('--seed');p.add_argument('--label');main(p.parse_args())
