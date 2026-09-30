"""Beyn contour extraction of multiple spatial characteristic roots.

All retained eigenpairs are refined in the original nonlinear characteristic
matrix. Contour roots complement, rather than replace, the full Nyquist count.
"""
from common import *
from model import SpatialBrunel
from response import characteristic
from scipy.sparse.linalg import splu,spsolve
import argparse

def extract(s,r,J,n=24,m=24,corners=None):
    nodes,weights=np.polynomial.legendre.leggauss(n)
    corners=np.array([-.015+.001j,.05+.001j,.05+.12j,-.015+.12j]) if corners is None else np.asarray(corners)
    rng=np.random.default_rng(79731);V=rng.normal(size=(s.P,m))+1j*rng.normal(size=(s.P,m))
    A0=np.zeros_like(V);A1=A0.copy()
    for i in range(4):
        a=corners[i];b=corners[(i+1)%4]
        for x,w in zip(nodes,weights):
            z=(a+b)/2+(b-a)/2*x;dz=(b-a)/2*w/(2j*np.pi)
            M=characteristic(s,r,J,z);X=splu(M.tocsc()).solve(V)
            A0+=dz*X;A1+=dz*z*X
    U,S,Vh=np.linalg.svd(A0,full_matrices=False);rank=int(np.sum(S>S[0]*1e-7))
    U=U[:,:rank];B=(U.conj().T@A1@Vh[:rank].conj().T)/S[:rank][None,:]
    eig,v=np.linalg.eig(B);return eig,U@v,S

def refine(s,r,J,lam,v):
    pivot=np.argmax(abs(v));v=v/v[pivot];P=s.P
    constraint=sparse.csr_matrix((np.ones(1),([0],[pivot])),shape=(1,P))
    for it in range(16):
        M=characteristic(s,r,J,lam);res=M@v;err=np.linalg.norm(res)/np.linalg.norm(v)
        if err<1e-9:return lam,v/np.linalg.norm(v),float(err)
        h=1e-6;dm=(characteristic(s,r,J,lam+h)-characteristic(s,r,J,lam-h))/(2*h)
        mat=sparse.bmat([[M,sparse.csr_matrix((dm@v)[:,None])],[constraint,sparse.csr_matrix((1,1))]],format='csc')
        d=spsolve(mat,np.r_[-res,0j]);v+=d[:-1];lam+=d[-1]
        if abs(lam)>.5:return None
    return None

def main(args):
    s=SpatialBrunel(args.grid,response=args.response);suffix='_calibrated' if args.response=='calibrated' else ''
    dest=OUT/f'g{args.grid}'/('contour_roots'+suffix);dest.mkdir(exist_ok=True)
    records=[];r=None
    for J in args.values:
        r,ok,_=s.solve(J,r);assert ok
        ev,vec,S=extract(s,r,J,n=args.nodes);roots=[]
        for k,lam in enumerate(ev):
            if not (-.017<lam.real<.055 and 0<lam.imag<.125):continue
            q=refine(s,r,J,lam,vec[:,k])
            if q is None:continue
            l,v,err=q
            if not (-.015<l.real<.05 and .001<l.imag<.12):continue
            if any(abs(l-old[0])<1e-6 for old in roots):continue
            roots.append(q)
        roots.sort(key=lambda q:-q[0].real);rows=[];weights=s.geo['group_size'];reg=s.geo['group_region']
        for l,v,err in roots:
            energy=weights*abs(v)**2;energy/=energy.sum()
            rows.append(dict(lambda_per_ms=l,frequency_hz=l.imag*1000/(2*np.pi),residual=err,
                regional_energy=[float(energy[s.E&(reg==k)].sum()) for k in range(3)]))
        write(dest/f'J{J:.6f}.json',dict(J_EE_core=J,roots=rows,singular_values=S,contour_nodes_per_edge=args.nodes))
        np.savez_compressed(dest/f'J{J:.6f}.npz',rates=r,roots=np.array([q[0] for q in roots]),vectors=np.array([q[1] for q in roots]))
        records.append(dict(J_EE_core=J,roots=rows));write(dest/'result.json',dict(status='COMPLETE' if J==args.values[-1] else 'RUNNING',rows=records))
        print('CONTOUR',J,len(roots),[(complex(q[0]),q[2]) for q in roots],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--grid',type=int,default=20);p.add_argument('--nodes',type=int,default=24)
    p.add_argument('--values',type=float,nargs='+',default=[.88,.90,.92,.94,.945,.95,.96,.98,1.]);p.add_argument('--response',default='shifted_white');main(p.parse_args())
