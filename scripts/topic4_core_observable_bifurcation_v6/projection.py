"""Locate a crossing of two observables of DIFFERENT periodic orbits."""
from common import *
from periodic import Orbit
from scipy.signal import resample
from scipy.optimize import brentq,minimize_scalar
import numpy as np

def main():
    s=System();N=2048
    sources={'stable':V4/'periodic/tonic/g1.38000000_N2048.npz','unstable':V5/'means/tonic_to_A200/mean200.000000.npz'}
    cache={}
    def solve(g,name):
        key=(float(g),name)
        if key in cache:return cache[key]
        z=np.load(sources[name]);r,T,err,hist=Orbit(s,g,N).solve(resample(z['r'],N,axis=0),float(z['T']),maxiter=24)
        if err>1e-8:raise RuntimeError((g,name,err))
        dense=resample(r,16384,axis=0);freq=2j*np.pi*np.arange(N//2+1);R=np.fft.rfft(r[:,0]);k=np.argmin(dense[:,0]);center=k/16384
        def rate(x):
            a=R*np.exp(freq*x);return float((a[0].real+a[-1].real+2*a[1:-1].real.sum())/N)
        fit=minimize_scalar(rate,bounds=(center-2/16384,center+2/16384),method='bounded',options={'xatol':1e-14})
        row=dict(g=float(g),T_ms=T,mean_hz=(r.mean(0)*1000).tolist(),min_hz=(dense.min(0)*1000).tolist(),max_hz=(dense.max(0)*1000).tolist(),refined_A_min_hz=fit.fun*1000,residual=err)
        dest=OUT/'periodic'/('projection_'+name);dest.mkdir(parents=True,exist_ok=True);path=dest/f'g{g:.12f}.npz';np.savez_compressed(path,r=r,T=T,g=g,residual=err,N=N)
        row['source']=str(path);cache[key]=row;return row
    def target(g):
        a=solve(g,'stable');b=solve(g,'unstable');value=b['mean_hz'][0]-a['refined_A_min_hz'];print('CROSSING_TEST',g,value,flush=True);return value
    g=brentq(target,1.378,1.3815,xtol=2e-12);a=solve(g,'stable');b=solve(g,'unstable')
    write('projection_intersection.json',dict(g=float(g),stable=a,unstable=b,observable_difference_hz=target(g),meaning='unstable A mean equals stable A minimum; distinct full-network periodic orbits, not a branch intersection'))
    print('PROJECTION_INTERSECTION',g,a,b,flush=True)
if __name__=='__main__':main()
