#!/usr/bin/env python3
"""Identify whether a high-branch complex crossing bounds the uniform-Z plateau."""
from analyze_topic4_e_only_bifurcation import Family,OUT,write
from audit_topic4_e_only_stability import count_unstable
from scipy.linalg import eigvals
from scipy.optimize import root
import numpy as np


def main():
    s=Family('uniform').s;cache={}
    def fun(x):
        q,f=x;r,err,ok=s.solve(q,np.full(200,.45));assert ok,(q,err)
        ev=eigvals(s.char_refined(2j*np.pi*f/1000,r,q,.1));v=ev[np.argmin(abs(ev))];cache['r']=r
        return [v.real,v.imag]
    sol=root(fun,[.75,30.],tol=1e-9);err=float(max(abs(np.array(fun(sol.x)))));assert err<1e-7,err
    q,freq=map(float,sol.x);assert .7<q<.79114 and freq>0
    r=cache['r'];tracks=[]
    for delta in [-.0005,.0005]:
        qq=q+delta;rr,err,ok=s.solve(qq,r);assert ok
        def follow(x):
            ev=eigvals(s.char_refined(complex(*x)/1000,rr,qq,.1));v=ev[np.argmin(abs(ev))];return [v.real,v.imag]
        ans=root(follow,[0,2*np.pi*freq],tol=1e-9);error=float(max(abs(np.array(follow(ans.x)))));assert error<1e-7
        count=count_unstable(s,rr,qq,(3200,6400))
        if count['status']!='PASS':count=count_unstable(s,rr,qq,(6400,12800))
        tracks.append({'q':qq,'real_per_s':float(ans.x[0]),'frequency_hz':float(ans.x[1]/2/np.pi),'count':count})
    row={'status':'COMPLETE','kind':'uniform','branch':'high','q':q,'frequency_hz':freq,'residual':err,'E_mean_hz':float(np.average(r[:100],weights=s.m.count_e)*1000),'track':tracks,
         'scope':'One high-branch complex crossing. Only a 0-to-2 root count establishes this as the first local loss of equilibrium stability; does not establish nonlinear cycle criticality or native SNN bifurcation.'}
    np.savez_compressed(OUT/'reduced_bifurcation/uniform_high_crossing.npz',r=r,q=q,frequency_hz=freq)
    write(OUT/'high_crossing_status.json',row);print(row,flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        write(OUT/'high_crossing_status.json',{'status':'FAILED','error':repr(exc)});raise
