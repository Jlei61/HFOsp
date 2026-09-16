"""True periodic solutions seeded from full-state recurrent trajectories."""
from common import *
from periodic import Orbit
from explore import recurrence
from scipy.interpolate import CubicSpline
from scipy.signal import resample
import numpy as np,argparse

def solve_from_trajectory(path,N=2048,branch=None):
    z=np.load(path);r=z['r'];dt=float(z['dt']);g=float(z['g']);s=System();rec=recurrence(r,dt)
    if rec.get('period_ms') is None or rec['recurrence_error']>.005:raise RuntimeError(('Not sufficiently recurrent',g,rec))
    T=rec['period_ms'];end=(len(r)-1)*dt;tt=np.arange(len(r))*dt
    guess=CubicSpline(tt,r)(end-T+np.arange(N)*T/N)
    return save_solve(s,g,guess,T,N,dict(initialization=str(path),initial_recurrence=rec),branch)

def save_solve(s,g,guess,T,N,meta=None,branch=None):
    r,T,err,hist=Orbit(s,g,N).solve(guess,T,maxiter=24)
    row=dict(g=g,N=N,T_ms=T,residual=err,initialization=meta)
    if err>1e-8:write(f'failed_g{g:.8f}_N{N}.json',row);raise RuntimeError(('Orbit failed',row))
    rr=resample(r,2*N,axis=0);F=Orbit(s,g,2*N).evaluate(np.r_[(rr/.01).ravel(),np.log(T)],rr,np.zeros_like(rr))
    row.update(offgrid_defect_hz=float(abs(F[:-1]).max()*10),minimum_hz=(rr.min(0)*1000).tolist(),mean_hz=(r.mean(0)*1000).tolist(),maximum_hz=(rr.max(0)*1000).tolist())
    dest=OUT/'periodic'
    if branch:dest=dest/branch
    dest.mkdir(parents=True,exist_ok=True);path=dest/f'g{g:.8f}_N{N}.npz'
    np.savez_compressed(path,r=r,T=T,g=g,residual=err,history=hist)
    (dest/f'g{g:.8f}_N{N}.json').write_text(json.dumps(row,indent=2)+'\n');print('SOLVED',json.dumps(row),flush=True)
    return path,row

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--trajectory');ap.add_argument('--from-orbit');ap.add_argument('--g',type=float,nargs='+');ap.add_argument('--N',type=int,default=2048);ap.add_argument('--branch');a=ap.parse_args()
    if a.trajectory:solve_from_trajectory(Path(a.trajectory),a.N,a.branch);return
    s=System();z=np.load(a.from_orbit);r=resample(z['r'],a.N,axis=0);T=float(z['T'])
    for g in a.g:
        path,row=save_solve(s,g,r,T,a.N,dict(initialization=a.from_orbit),a.branch);z=np.load(path);r=z['r'];T=float(z['T']);a.from_orbit=str(path)

if __name__=='__main__':main()
