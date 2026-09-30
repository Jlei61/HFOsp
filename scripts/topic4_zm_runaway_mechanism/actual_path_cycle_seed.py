"""Periodic BVP seed from the actual 7750-ms frozen-Z regular trajectory.

Uses its rates only as a Newton initial guess. The converged orbit is solved in
the unchanged autonomous rate equations with Z held and M dynamic.
"""
from native_path import *
from compact_periodic import CompactExactGalerkin
from native_cycles import save
from scipy.signal import find_peaks,resample
from scipy.interpolate import CubicSpline
import argparse


def main(a):
    s=model();path=attach_fine_rate_entry_path(s)
    if a.from_orbit:
        source=Path(a.from_orbit);z=np.load(source)
        assert float(z['residual'])<2e-8
        T=float(z['T']);D=float(z['D']);r=resample(z['r'],a.N,axis=0)
        s.set_D(D);assert np.max(abs(s.Z-z['Z']))<1e-12
        left=right=None
    else:
        source=OUT/'runs/actual_fine_Z_t7750_matched_history_dt005/trajectory.npz'
        z=np.load(source);g=z['global_E_hz'];peaks=find_peaks(g,height=20,distance=150)[0]
        assert len(peaks)>=3
        def refine(k):
            return k+.5*(g[k-1]-g[k+1])/(g[k-1]-2*g[k]+g[k+1])
        left,right=map(refine,peaks[-2:]);T=right-left
        assert 150<T<400
        r=CubicSpline(np.arange(len(g)),z['group_rate_hz']/1000)(left+np.arange(a.N)*T/a.N)
        D=float(path['D'][0]);s.set_D(D)
        assert np.max(abs(s.Z-z['Z_source']))<1e-12
    dest=OUT/'periodic'/a.label;dest.mkdir(parents=True,exist_ok=True)
    write(dest/'seed_provenance.json',dict(source=str(source),source_Z_time_ms=7750,
          D=D,estimated_T_ms=T,last_peaks_ms=[left,right],N=a.N,M=a.M,
          spatial_parameter_path='actual 7750,7760,7770,7780-ms Z fields',
          distinction='Separate conditional slice; never combined with old affine branch'))
    if a.prepare_only:return
    CompactExactGalerkin.harmonic_block=33
    o=CompactExactGalerkin(s,a.N,a.M,a.device)
    o.cache_mean_operators=False;o.cp.fft.config.get_plan_cache().set_size(0)
    o.host_krylov=a.host_krylov
    if a.host_krylov:assert read(OUT/'host_krylov_check.json')['status']=='PASS'
    sol=o.solve(r,T,D,maxiter=18,tol=2e-8,restart=60)
    assert sol['residual']<2e-8,sol['residual']
    row=save(s,sol,dest,'seed')
    row.update(nonlinear_samples=a.M,family='fine')
    write(dest/'result.json',dict(status='CONVERGED_SEED',rows=[row],bifurcation_type='NOT_ESTABLISHED'))
    log('ACTUAL PATH CYCLE SEED',row)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--N',type=int,default=2049)
    p.add_argument('--M',type=int,default=8192);p.add_argument('--device',type=int,default=0)
    p.add_argument('--label',default='actual_fine_Z_seed_G2049_M8192')
    p.add_argument('--from-orbit',help='Accepted lower-resolution orbit as a numerical initial guess')
    p.add_argument('--host-krylov',action='store_true')
    p.add_argument('--prepare-only',action='store_true');main(p.parse_args())
