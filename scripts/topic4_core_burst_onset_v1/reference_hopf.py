"""Reproduce the scalar delay-rate example in Brunel & Hakim (2008), Fig. 2.

This is a literature benchmark, never an inferred reduction of our spatial SNN.
"""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,json
from scipy.optimize import brentq
from scipy.special import lambertw
from numba import njit
OUT=Path('/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/core_burst_onset_brunel_v1_20260915')

@njit(cache=True)
def simulate(K,dt,T):
    tau=10.;delay=2.;r0=1+np.tanh(1.);slope=1-np.tanh(1.)**2
    J=K/slope;drive=1+J*r0;lag=round(delay/dt);n=round(T/dt)
    y=np.full(n+lag+1,r0+.002)
    for i in range(lag+1,n+lag+1):
        f=(-y[i-1]+1+np.tanh(drive-J*y[i-1-lag]))/tau
        predictor=y[i-1]+dt*f
        g=(-predictor+1+np.tanh(drive-J*y[i-lag]))/tau
        y[i]=y[i-1]+dt*(f+g)/2
    return y[lag+1:]

def main():
    theta=brentq(lambda a:np.tan(a)+5*a,np.pi/2+1e-5,np.pi-1e-5)
    kc=np.sqrt(1+(5*theta)**2);freq=theta/2/(2*np.pi)*1000
    ks=np.linspace(7.5,9.5,81);eigen=[];low=[];high=[]
    for k in ks:
        roots=np.array([lambertw(-k*2/10*np.exp(.2),j)/2-.1 for j in range(-6,7)])
        root=roots[np.argmax(roots.real)];eigen.append([root.real*1000,abs(root.imag)*1000/2/np.pi])
        y=simulate(k,.01,6000.)[-100000:];low.append(y.min());high.append(y.max())
    # Verify the exact characteristic equation at the analytically found onset.
    lam=1j*theta/2;res=abs(10*lam+1+kc*np.exp(-2*lam));assert res<1e-9
    checks=[]
    for k in (8.4,8.8):
        a=simulate(k,.01,6000.);b=simulate(k,.005,6000.)
        aa=(a[-100000:].max()-a[-100000:].min())/2
        bb=(b[-200000:].max()-b[-200000:].min())/2
        checks.append(dict(K=k,amplitude_dt001=float(aa),amplitude_dt0005=float(bb),difference=float(abs(aa-bb))))
    traces={f'trace_{k:g}':simulate(k,.005,600.)[::4] for k in (8.4,8.8)}
    np.savez_compressed(OUT/'literature_hopf_benchmark.npz',K=ks,eigen=eigen,low=low,high=high,**traces)
    d=dict(source='https://www.phys.ens.psl.eu/~hakim/08chaosnbvh.pdf',source_figure=2,
        equation='10 dr/dt = -r + 1+tanh(I_ext - J r(t-2)); time in ms',
        r0=float(1+np.tanh(1.)),J='K / sech(1)^2',I_ext='1 + J*r0',
        critical_K=float(kc),critical_frequency_hz=float(freq),characteristic_residual=float(res),dt_checks=checks,
        scope='Literal literature delay-rate example. Coupling and external mean co-vary to hold r0 fixed. Not calibrated to spatial SNN; no K-to-EE conversion.')
    (OUT/'literature_hopf_benchmark.json').write_text(json.dumps(d,indent=2)+'\n')
    print(json.dumps(d))

if __name__=='__main__':main()
