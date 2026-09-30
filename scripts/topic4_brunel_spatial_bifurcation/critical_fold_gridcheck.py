"""Refine selected coarse-grid folds on the existing 0.5-mm projection."""
from common import *
from model import SpatialBrunel
from scipy.sparse.linalg import spsolve,eigs
import argparse
BASE=OUT/'critical_revision'

def solve_fold(s,r,J,v,projected=False):
    v=v/np.linalg.norm(v);trace=[]
    for k in range(24):
        A=s.jacobian(r,J);aug=np.r_[s.residual(r,J),A@v,v@v-1]
        err=float(abs(aug).max());trace.append(dict(iteration=k,J=J,residual=err));print(trace[-1],flush=True)
        if err<1e-9:return r,J,v,trace,'CONVERGED'
        h=1e-6;H=(s.jacobian(r+h*v,J)-s.jacobian(r-h*v,J))/(2*h)
        pj=s.parameter_derivative(r,J);jj=(s.jacobian(r,J+h)-s.jacobian(r,J-h))@v/(2*h)
        col=lambda x:sparse.csr_matrix(x[:,None]);P=s.P
        mat=sparse.bmat([[A,col(pj),sparse.csr_matrix((P,P))],[H,col(jj),A],
            [sparse.csr_matrix((1,P)),sparse.csr_matrix((1,1)),sparse.csr_matrix(2*v[None,:])]],format='csc')
        step=spsolve(mat,-aug);alpha=1.
        for back in range(24):
            rr=r+alpha*step[:P];j=J+alpha*step[P];vv=v+alpha*step[P+1:]
            if (rr.min()>=-1e-10 or projected) and .7<j<1.4:
                rr=np.maximum(rr,0.)
                f=np.r_[s.residual(rr,j),s.jacobian(rr,j)@vv,vv@vv-1]
                if np.linalg.norm(f)<np.linalg.norm(aug):break
            alpha*=.5
        else:return r,J,v,trace,'LINE_SEARCH_STOP'
        r,J,v=rr,j,vv
    return r,J,v,trace,'ITERATION_LIMIT'

def main(args):
    s=SpatialBrunel(40);old=np.load(OUT/'operators/g20/geometry.npz');g=s.geo
    T=sparse.coo_matrix((1/g['group_size'][g['cell_group']],(g['cell_group'],old['cell_group'])),shape=(s.P,len(old['group_size']))).tocsr()
    rows=[]
    for name in args.names:
        dest=BASE/args.label/name;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'result.json').exists():rows.append(read(dest/'result.json'));continue
        z=np.load(OUT/'expanded/folds'/name/'fold.npz');r0=T@z['rates'];v0=T@z['right'];J0=float(z['J'])
        r,J,v,trace,status=solve_fold(s,r0,J0,v0,args.projected)
        q=dict(name=name,status=status,J_coarse=J0,J_fine=J,rates_coarse_hz=read(OUT/'expanded/folds'/name/'result.json')['rates_hz'],
            rates_fine_hz=s.regional_rates(r),residual=float(abs(s.residual(r,J)).max()),trace=trace,
            seed_projection='Average each original-neuron coarse-group rate onto its fine group; no re-fitting of network parameters',projected_newton=args.projected)
        if status=='CONVERGED':
            A=s.jacobian(r,J);_,left=eigs(A.T,k=1,sigma=0,tol=1e-10);left=left[:,0].real;left/=left@v
            H=(s.jacobian(r+1e-6*v,J)-s.jacobian(r-1e-6*v,J))/(2e-6)
            en=g['group_size']*v**2;en/=en.sum()
            q.update(eigenvector_residual=float(np.linalg.norm(A@v)),transversality=float(left@s.parameter_derivative(r,J)),
                quadratic_coefficient=float(left@(H@v)/2),regional_energy=[float(en[s.E&(g['group_region']==k)].sum()) for k in range(3)])
            np.savez_compressed(dest/'fold.npz',rates=r,J=J,right=v,left=left)
        write(dest/'result.json',q);rows.append(q);print(name,status,J,flush=True)
    write(BASE/args.label/'result.json',dict(status='COMPLETE',rows=rows,meaning='Selected-fold refinement across two spatial grids, not an enumeration of every fine-grid branch'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--names',nargs='+',default=['low_arclength_0','low_arclength_2','low_arclength_4','low_arclength_v2_0'])
    p.add_argument('--projected',action='store_true');p.add_argument('--label',default='gridcheck');main(p.parse_args())
