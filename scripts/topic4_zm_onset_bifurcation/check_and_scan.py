"""Check Z/M derivatives, then follow both physical-D equilibrium ends."""
from zm_model import *
from scipy.sparse.linalg import eigs
import argparse


def main(a):
    s=ZMRate(a.grid,mode=a.mode,m_current=a.m_current)
    out=DEST/f'g{a.grid}'/a.label;out.mkdir(parents=True,exist_ok=True)
    r,ok,tr=s.solve_D(0.);assert ok
    rng=np.random.default_rng(230917);v=rng.normal(size=s.P);v/=np.linalg.norm(v)
    D=.005;r0=r.copy();h=1e-7
    J=s.jacobian(r0,D);fd=(s.residual(r0+h*v,D)-s.residual(r0-h*v,D))/(2*h)
    jerr=np.linalg.norm(J@v-fd)/np.linalg.norm(fd)
    deriv=s.parameter_derivative(r0,D);fdD=(s.residual(r0,D+h)-s.residual(r0,D-h))/(2*h)
    derr=np.linalg.norm(deriv-fdD)/np.linalg.norm(fdD)
    diff=s.characteristic(r0,D,0.)+s.jacobian(r0,D)
    cerr=float(abs(diff.data).max()) if diff.nnz else 0.
    assert jerr<1e-6 and derr<1e-6 and cerr<1e-10,(jerr,derr,cerr)
    write(out/'checks.json',dict(jacobian_relative_error=jerr,D_derivative_relative_error=derr,
        characteristic_zero_equals_negative_stationary_jacobian=cerr,mode=a.mode,m_current_mv=a.m_current))
    rows=[]
    for direction in ('increasing','decreasing'):
        r=None
        if direction=='decreasing':r=.85/s.ref
        values=np.linspace(0,1,a.points)
        if direction=='decreasing':values=values[::-1]
        for D in values:
            rr,ok,tr=s.solve_D(float(D),r)
            row=dict(D=float(D),direction=direction,converged=ok,residual_hz=tr[-1]*1000)
            if ok:
                r=rr
                row.update(global_E_hz=s.global_rate(r),regional_hz=s.regional_rates(r))
                ev=eigs(s.jacobian(r,float(D)),k=2,sigma=0,return_eigenvectors=False,tol=1e-8)
                row['stationary_eigenvalues']=ev
                np.savez_compressed(out/f'{direction}_D{D:.6f}.npz',r=r,D=D,Z=s.Z,
                    M=1000*s.E*r if a.mode=='dynamic_M' else a.m_current*s.m_shape/.0005)
            rows.append(row)
            write(out/'result.json',dict(status='RUNNING',rows=rows,mode=a.mode,
                temporal_stability='NOT_YET_COMPUTED',bifurcation_type='NOT_YET_ESTABLISHED'))
            print(row,flush=True)
            if not ok:break
    write(out/'result.json',dict(status='COARSE_EQUILIBRIUM_SCAN_COMPLETE',rows=rows,mode=a.mode,
        temporal_stability='NOT_YET_COMPUTED',bifurcation_type='NOT_YET_ESTABLISHED',
        note='Failed solves mark continuation gaps; they are not bifurcations.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--grid',type=int,default=20);p.add_argument('--points',type=int,default=101)
    p.add_argument('--mode',choices=['dynamic_M','frozen_M'],default='dynamic_M');p.add_argument('--m-current',type=float,default=0)
    p.add_argument('--label',default='conditional_dynamic_M');main(p.parse_args())
