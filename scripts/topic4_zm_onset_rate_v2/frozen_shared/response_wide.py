"""Stable Eq.36 evaluation beyond the low-rate Taylor domain.

Integrate the logarithmic derivative U'/U and log U. A continued fraction of
parabolic-cylinder functions initializes the recessive solution at negative y.
This avoids cancellation at strongly mean-driven operating points.
"""
import numpy as np
from scipy.integrate import solve_ivp

def initial_ratio(z,y,n=300):
    x=-np.sqrt(2)*y;den=complex(x)
    for k in range(n,0,-1):den=x+(z+k)/den
    return np.sqrt(2)*z/den

def white_wide(lam,lo,hi,sigma,tm,rate):
    answer=np.empty((2,len(rate)),complex)
    for tau in np.unique(tm):
        mask=tm==tau;a=lo[mask];b=hi[mask];z=complex(lam)*tau
        start=min(-4.,float(a.min())-2.)
        def fun(y,state):
            R=state[0]
            return [2*y*R+2*z-R*R,R]
        sol=solve_ivp(fun,(start,float(b.max())),np.array([initial_ratio(z,start),0j]),
                      method='DOP853',rtol=2e-11,atol=1e-13*min(1.,abs(z)),dense_output=True)
        if not sol.success:raise RuntimeError(sol.message)
        va=sol.sol(a);vb=sol.sol(b);den=np.expm1(vb[1]-va[1])
        first=vb[0]+(vb[0]-va[0])/den
        sa=2*a*va[0]+2*z;sb=2*b*vb[0]+2*z
        second=sb+(sb-sa)/den
        answer[0,mask]=rate[mask]/sigma[mask]*first/(1+z)
        answer[1,mask]=rate[mask]/sigma[mask]**2*second/(2+z)
    return answer
