"""Stable local Taylor transport of the unchanged parabolic-cylinder ODE.

This is numerical special-function evaluation, NOT a network-state or
probability-density discretization. It removes cancellation at negative
integration bounds and supports dynamic eigenvalue calculation.
"""
import numpy as np
from numba import njit
from stable_response import tail_terms,cylinder_ratios

@njit(cache=True)
def advance(y,u,p,z,h):
    a=np.zeros(36,np.complex128);a[0]=u;a[1]=p
    for k in range(34):
        a[k+2]=(2*y*(k+1)*a[k+1]+2*(k+z)*a[k])/((k+2)*(k+1))
    change=0j;dp=0j
    for k in range(35,0,-1):
        change=change*h+a[k];dp=dp*h+k*a[k]
    change*=h
    return u+change,dp,change

@njit(cache=True)
def transport_ratios(lo,hi,z):
    y=min(lo,-max(12.,2*abs(z)));_,p=tail_terms(y,z);u=1.+0j
    while y<lo-1e-14:
        h=min(.15,2/max(abs(y),1.),lo-y)
        u,p,_=advance(y,u,p,z,h);y+=h
    p0=p/u;u=1.+0j;p=p0;delta=0j;y=lo
    while y<hi-1e-14:
        h=min(.15,2/max(abs(y),1.),hi-y)
        u,p,change=advance(y,u,p,z,h);delta+=change;y+=h
    q0=2*lo*p0+2*z;qh=2*hi*p+2*z*u
    return (p-p0)/delta/(1+z),(qh-q0)/delta/(2+z)

@njit(cache=True)
def white_transport(lam,lo,hi,sigma,tm,rate):
    out=np.empty((2,len(lo)),np.complex128)
    for i in range(len(lo)):
        a,b=transport_ratios(lo[i],hi[i],lam*tm[i])
        out[0,i]=rate[i]/sigma[i]*a;out[1,i]=rate[i]/sigma[i]**2*b
    return out

def verify():
    rows=[]
    for lo,hi in [(-30.,-29.7),(-10.,-9.5),(-5.6,-5.35),(-3.,-2.5),(-1.,0.),(.5,2.),(3.,4.)]:
        for z in [2e-7+0j,-2e-4+0j,.01+.2j,.2+1j,4j,12j]:
            a=np.array(transport_ratios(lo,hi,z));b=np.array(cylinder_ratios(lo,hi,z))
            err=float(np.max(abs(a-b)/np.maximum(abs(b),1e-15)))
            rows.append(dict(lo=lo,hi=hi,z=[z.real,z.imag],relative_error=err))
    assert max(x['relative_error'] for x in rows)<2e-7
    return rows

if __name__=='__main__':
    import json
    from pathlib import Path
    rows=verify();dest=Path(__file__).resolve().parents[2]/'results/topic4_sef_hfo/fig5_zm_onset_bifurcation_20260917'
    (dest/'response_transport_checks.json').write_text(json.dumps(dict(status='PASS',reference='60-digit same parabolic-cylinder function',rows=rows),indent=2)+'\n')
    print('max relative error',max(x['relative_error'] for x in rows))
