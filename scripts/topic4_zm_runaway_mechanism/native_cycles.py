"""Conditional cycles on the native checkpoint Z path, with the same dynamic M."""
from native_path import *
from periodic_v3 import PeriodicV3
from scipy.signal import find_peaks, resample
from scipy.interpolate import CubicSpline
import argparse


def save(s,sol,out,name):
    out.mkdir(parents=True,exist_ok=True)
    s.set_D(sol['D']);p=out/f'{name}.npz'
    np.savez_compressed(p,r=sol['r'],T=sol['T'],D=sol['D'],Z=s.Z,residual=sol['residual'])
    g=sol['r'][:,s.E]@s.mean_weights*1000
    q=dict(path=str(p),D=sol['D'],T_ms=sol['T'],N=len(g),mean_hz=float(g.mean()),min_hz=float(g.min()),
           max_hz=float(g.max()),residual=sol['residual'],iterations=sol['history'],
           minimum_group_rate_hz=float(sol['r'].min()*1000),stability='NOT_ESTABLISHED',Z='held',M='dynamic',Z_path=s.z_source)
    write(p.with_suffix('.json'),q);return q


def seed(a):
    s=model();(attach_rate_entry_path if a.family=='rate' else attach_native_path)(s)
    if a.stream:
        from streaming_periodic import StreamPeriodic
        o=StreamPeriodic(s,a.N,a.device)
    else:o=PeriodicV3(s,a.N,a.device)
    if a.from_orbit:
        z=np.load(a.from_orbit);r=resample(z['r'],a.N,axis=0);T=float(z['T']);D=float(z['D']) if a.D is None else a.D
    else:
        source='rate_Z7700_carried' if a.family=='rate' else 'native_Z9000_carried'
        z=np.load(OUT/f'runs/{source}/trajectory.npz');g=z['global_E_hz'];peaks=find_peaks(g,height=20,distance=150)[0]
        left,right=peaks[-2:]
        def precise(k):
            return k+.5*(g[k-1]-g[k+1])/(g[k-1]-2*g[k]+g[k+1])
        left,right=precise(left),precise(right);T=right-left
        r=CubicSpline(np.arange(len(g)),z['group_rate_hz']/1000)(left+np.arange(a.N)*T/a.N)
        D=float(1-z['Z_source'][s.E]@s.mean_weights) if a.D is None else a.D
    sol=o.solve(r,T,D,maxiter=18,tol=2e-8,restart=40)
    save(s,sol,OUT/'periodic',a.label or f'seed_N{a.N}')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--N',type=int,default=256);p.add_argument('--device',type=int,default=1)
    p.add_argument('--from-orbit');p.add_argument('--D',type=float);p.add_argument('--label');p.add_argument('--family',choices=['native','rate'],default='native')
    p.add_argument('--stream',action='store_true');seed(p.parse_args())
