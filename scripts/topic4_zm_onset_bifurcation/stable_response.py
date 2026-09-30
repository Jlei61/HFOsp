"""Numerical evaluation of the SAME Eq.36 response at negative bounds.

The Taylor series about zero suffers catastrophic cancellation in the
strong-drive tail. Use the equivalent parabolic-cylinder function there;
a near-zero-frequency asymptotic expansion handles far-negative bounds.
This does not change the model's response formula or fitted parameters.
"""
import numpy as np
import mpmath as mp
from numba import njit

@njit(cache=True)
def clog1p(z):
    if abs(z)>.01:return np.log(1+z)
    out=0j;power=z
    for k in range(1,14):
        out+=power/k;power*=-z
    return out

@njit(cache=True)
def cexpm1(z):
    if abs(z)>.01:return np.exp(z)-1
    out=0j;power=z
    for k in range(1,14):
        out+=power;power*=z/(k+1)
    return out

@njit(cache=True)
def tail_terms(y,z):
    term=1.+0j;dev=0j;der=0j;last=1e300
    for k in range(1,140):
        term*=-(z+2*k-2)*(z+2*k-1)/(4*k*y*y)
        if abs(term)>last:break
        dev+=term;der+=(-2*k/y)*term;last=abs(term)
        if abs(term)<1e-28:break
    return dev,-z/y+der/(1+dev)

@njit(cache=True)
def tail_ratios(lo,hi,z):
    a,pa=tail_terms(lo,z);b,pb=tail_terms(hi,z)
    logratio=-z*np.log(hi/lo)+clog1p(b)-clog1p(a)
    delta=cexpm1(logratio)
    qa=2*lo*pa+2*z;qb=2*hi*pb+2*z
    return (pb+(pb-pa)/delta)/(1+z),(qb+(qb-qa)/delta)/(2+z)

def cylinder_ratios(lo,hi,z):
    with mp.workdps(60):
        z=mp.mpc(z);lo=mp.mpf(float(lo));hi=mp.mpf(float(hi))
        def values(y):
            fac=mp.power(2,z/2)*mp.exp(y*y/2)
            u=fac*mp.pcfd(-z,-mp.sqrt(2)*y)
            du=2*y*u+mp.sqrt(2)*fac*mp.pcfd(1-z,-mp.sqrt(2)*y)
            ddu=2*y*du+2*z*u
            return u,du,ddu
        a=values(lo);b=values(hi);den=b[0]-a[0]
        return complex((b[1]-a[1])/den/(1+z)),complex((b[2]-a[2])/den/(2+z))

def install(response_module):
    original=response_module.white
    def stable_white(lam,lo,hi,sigma,tm,rate):
        ans=original(lam,lo,hi,sigma,tm,rate)
        for i in np.flatnonzero(lo < -3.):
            z=complex(lam)*tm[i]
            if hi[i] < -8. and abs(z)<.05:
                p,q=tail_ratios(lo[i],hi[i],z)
            else:p,q=cylinder_ratios(lo[i],hi[i],z)
            ans[0,i]=rate[i]/sigma[i]*p;ans[1,i]=rate[i]/sigma[i]**2*q
        return ans
    response_module.white=stable_white

def verify():
    records=[]
    for lo,hi in [(-50.,-49.8),(-20.,-19.7),(-10.,-9.6),(-8.8,-8.1)]:
        for z in [2e-7+0j,2e-4+0j,-2e-4+0j,.002j]:
            a=np.array(tail_ratios(lo,hi,z));b=np.array(cylinder_ratios(lo,hi,z))
            error=float(np.max(abs(a-b)/np.maximum(abs(b),1e-15)))
            records.append(dict(lo=lo,hi=hi,z=[z.real,z.imag],relative_error=error))
    assert max(x['relative_error'] for x in records)<1e-8
    return records
