"""High-precision setup of the same stationary colored-Poisson Galerkin map.

Raw monomial Gram matrices are ill conditioned at large polynomial degree.
All moments, Cholesky factors and polynomial similarity transforms are formed
in arbitrary precision. Only the resulting orthonormal matrices are rounded
to FP64. This changes arithmetic, not noise statistics or the physical map.
The frame is stationary: it must not be used with a time-varying intensity.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
from math import comb
import json
import time
import argparse
import numpy as np
import mpmath as mp

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/topic4_sef_hfo/kinetic_population_bifurcation_20260916'


def powers(degree):
    return [(n-b,b) for n in range(degree+1) for b in range(n+1)]


def array(x):
    return np.asarray(x.tolist(), dtype=float)


def construct(population, degree, nu, digits=70):
    started = time.time()
    with open(OUT/'operators/selected_g40_theta0.25/prepared.json') as f:
        p = json.load(f)['params']
    # Keep exactly the same floating-point physical coefficients as the legacy
    # implementation; extra precision is used for basis construction only.
    ar = np.exp(-.1/p['tau_r_AMPA']); ad = np.exp(-.1/p['tau_d_AMPA'])
    jump = p[f'tau_m_{population}']/p['tau_r_AMPA']*p[f'J_ext_{population}']
    with mp.workdps(digits):
        B = mp.matrix([[float(ar),0.],[float((1-ad)*ar),float(ad)]])
        d = mp.matrix([float(jump),float(jump*(1-ad))]); lam = mp.mpf(float(nu*.1))
        mean = mp.lu_solve(mp.eye(2)-B,d*lam)
        # Centered physical cumulants by triangular degree blocks.
        mon = powers(2*degree+1); kappa = {(0,0):mp.mpf(0),(1,0):mp.mpf(0),(0,1):mp.mpf(0)}
        for n in range(2,2*degree+2):
            for b in range(n+1):
                a=n-b
                rhs=lam*d[0]**a*d[1]**b
                rhs+=sum(mp.mpf(comb(b,j))*B[0,0]**a*B[1,0]**j*B[1,1]**(b-j)*kappa[a+j,b-j]
                         for j in range(1,b+1))
                kappa[a,b]=rhs/(1-B[0,0]**a*B[1,1]**b)
        L=mp.cholesky(mp.matrix([[kappa[2,0],kappa[1,1]],[kappa[1,1],kappa[0,2]]]))
        Li=L**-1
        standardized={}
        for a,b in mon:
            standardized[a,b]=sum(mp.mpf(comb(b,j))*Li[0,0]**a*Li[1,0]**j*Li[1,1]**(b-j)*kappa[a+j,b-j]
                                  for j in range(b+1))
        moments={(0,0):mp.mpf(1)}
        for a,b in mon[1:]:
            if a:
                moments[a,b]=sum(mp.mpf(comb(a-1,i)*comb(b,j))*standardized[i+1,j]*moments[a-1-i,b-j]
                                 for i in range(a) for j in range(b+1))
            else:
                moments[a,b]=sum(mp.mpf(comb(b-1,j))*standardized[0,j+1]*moments[0,b-1-j] for j in range(b))
        selected=powers(degree); size=len(selected); index={v:i for i,v in enumerate(selected)}
        def matrix(dx,dy):
            return mp.matrix([[moments[a+c+dx,b+e+dy] for c,e in selected] for a,b in selected])
        G=matrix(0,0); R=mp.cholesky(G).T; Ri=R**-1
        C=Ri.T*(mean[1]*G+L[1,0]*matrix(1,0)+L[1,1]*matrix(0,1))*Ri
        W=Li*B*L; h=Li*d
        innovation=[mp.mpf(1),mp.mpf(0)]
        for n in range(2,degree+1):
            innovation.append(sum(mp.mpf(comb(n-1,j-1))*lam*innovation[n-j] for j in range(2,n+1)))
        T=mp.zeros(size)
        for row,(a,b) in enumerate(selected):
            for i in range(a+1):
                for j in range(b+1):
                    for ell in range(b-j+1):
                        x=i+j; y=ell; k=a+b-x-y
                        T[row,index[x,y]]+=mp.mpf(comb(a,i)*comb(b,j)*comb(b-j,ell))*W[0,0]**i*h[0]**(a-i)*W[1,0]**j*W[1,1]**ell*h[1]**(b-j-ell)*innovation[k]
        Aorth=Ri.T*T*R.T
        # In degree-ordered orthogonal polynomials the affine noise map is
        # triangular. Its diagonal is known exactly even when a dense FP64
        # eigensolver is sensitive to repeated/non-normal decay modes.
        diagonal_error=float(max(abs(Aorth[i,i]-B[0,0]**a*B[1,1]**b) for i,(a,b) in enumerate(selected)))
        upper_error=float(max(abs(Aorth[i,j]) for i in range(size) for j in range(i+1,size)))
        ortherr=float(mp.norm(Ri.T*G*Ri-mp.eye(size),p=mp.inf))
        symerr=float(mp.norm(C-C.T,p=mp.inf))
        stationary=mp.matrix([moments[x] for x in selected])
        stationarity=float(mp.norm(T*stationary-stationary,p=mp.inf)/max(1,mp.norm(stationary,p=mp.inf)))
        Cf=array(C); Af=array(Aorth)
        nodes,U=np.linalg.eigh((Cf+Cf.T)*.5); U*=np.where(U[0]<0,-1.,1.)
        A=U.T@Af@U; mass=U[0].copy()
        expected=np.sort([ar**a*ad**b for a,b in selected]); ev=np.linalg.eigvals(A)
        qa=dict(degree=degree,digits=digits,seconds=time.time()-started,
            triangular_noise_diagonal_error_high_precision=diagonal_error,
            triangular_noise_upper_error_high_precision=upper_error,
            orthogonality_error_high_precision=ortherr,multiplication_symmetry_error_high_precision=symerr,
            raw_moment_stationarity_relative_error=stationarity,
            noise_eigenvalue_max_abs_error=float(np.max(abs(np.sort(ev.real)-expected))),
            noise_eigenvalue_max_imaginary=float(np.max(abs(ev.imag))),
            mass_left_error=float(np.max(abs(mass@A-mass))),stationary_right_error=float(np.max(abs(A@mass-mass))),
            mass_norm_error=float(abs(mass@mass-1)),gram_condition_fp64=float(np.linalg.cond(array(G))),
            orthogonal_similarity_roundtrip_max_error=float(np.max(abs(U@A@U.T-Af))),
            operator_norm=float(np.linalg.norm(A,2)),current_min_mv=float(nodes.min()),current_max_mv=float(nodes.max()))
        qa['pass']=max(qa['mass_left_error'],qa['stationary_right_error'],qa['orthogonal_similarity_roundtrip_max_error'])<1e-12 and max(ortherr,diagonal_error,upper_error)<1e-30 and qa['operator_norm']<1+1e-12
        qa['validation']='Exact triangular noise spectrum and orthogonal similarity backward error; dense FP64 eigenvalue error is diagnostic, not the acceptance test for repeated non-normal modes.'
        assert qa['pass'],qa
        return dict(A=A,nodes=nodes,mass=mass,mean=array(mean).ravel(),U=U,L=array(L),R=array(R),Ri=array(Ri)),qa


class StationaryBasis:
    def __init__(self,population,degree,nu,digits=70):
        self.degree=degree;self.nu=nu;self.selected=powers(degree)
        folder=OUT/'stationary_noise_basis';folder.mkdir(parents=True,exist_ok=True)
        stem=f'{population}_degree{degree}_nu{nu:.15g}_digits{digits}'
        target=folder/(stem+'.npz');meta=folder/(stem+'.json')
        if target.exists() and meta.exists():
            a=dict(np.load(target));self.qa=json.loads(meta.read_text())
        else:
            a,self.qa=construct(population,degree,nu,digits)
            np.savez_compressed(target,**a);meta.write_text(json.dumps(self.qa,indent=2)+'\n')
        assert self.qa['pass']
        self.A=a['A'];self.mean=a['mean'];self.frame={k:a[k] for k in ('nodes','mass','U','L','R','Ri')}

    def advance(self,nu):
        assert nu==self.nu, 'Stationary basis requires constant private Poisson intensity'
        return self.A.copy()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--degree',type=int,default=12)
    ap.add_argument('--digits',type=int,default=70);args=ap.parse_args()
    nu=json.loads((OUT/'operators/selected_g40_theta0.25/prepared.json').read_text())['nu_ext_per_ms']
    b=StationaryBasis('E',args.degree,nu,args.digits);print(json.dumps(b.qa,indent=2),flush=True)
