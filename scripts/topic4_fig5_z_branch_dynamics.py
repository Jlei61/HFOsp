#!/usr/bin/env python3
"""Transfer helpers and instantaneous-variance sensitivity diagnostics.

The compiled transfer evaluates the same quadrature and antiderivative table.
It changes transfer evaluation speed only. This file's trajectory diagnostic
uses instantaneous variance; the actual filtered-v1 dynamics are implemented in
topic4_fig5_z_frozen_v1.py. Native-SNN correspondence remains unvalidated.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys,json,time,argparse
from pathlib import Path
import numpy as np
from numba import njit
from topic4_fig5_z_bifurcation_preview import Equilibrium,OUT,SOURCE
import siegert_table


@njit(cache=True)
def primitive(x,xs,gs):
    lo,hi=xs[0],xs[-1]
    if x<lo:
        return gs[0]-(np.log(abs(x))-np.log(abs(lo))-.25*(1/x**2-1/lo**2))/np.sqrt(np.pi)
    if x>26.:
        return gs[-1]+np.exp(x*x)/x*(1+1/(2*x*x)+3/(4*x**4))-np.exp(26.**2)/26.*(1+1/(2*26.**2)+3/(4*26.**4))
    u=(x-lo)/(xs[1]-lo)
    i=min(max(int(u),0),len(xs)-2)
    return gs[i]+(x-xs[i])/(xs[i+1]-xs[i])*(gs[i+1]-gs[i])


@njit(cache=True)
def transfer(mu,ex,inh,theta,tm,ref,reset,rise_decay,w2,xx,ww,xs,gs):
    out=np.zeros(len(mu))
    for i in range(len(mu)):
        sigma=np.sqrt(max(ex[i],1e-12))
        sigg=np.sqrt(max(inh[i]*w2,0.))
        shift=mu[i]-1.0325*np.sqrt(max(ex[i]*rise_decay/tm,1e-16))
        for k in range(len(xx)):
            mean=shift+np.sqrt(2.)*sigg*xx[k]
            integral=primitive((theta[i]-mean)/sigma,xs,gs)-primitive((reset-mean)/sigma,xs,gs)
            den=ref+tm*np.sqrt(np.pi)*integral
            if np.isfinite(den) and den>0:
                out[i]+=ww[k]*min(1/den,1/ref)
    return out


@njit(cache=True)
def primitive_slope(x,xs,gs):
    if x<xs[0]:return -(1/x+.5/x**3)/np.sqrt(np.pi)
    if x>26.:return np.exp(x*x)*(2-3.75/x**6)
    i=min(max(int((x-xs[0])/(xs[1]-xs[0])),0),len(xs)-2)
    return (gs[i+1]-gs[i])/(xs[i+1]-xs[i])


@njit(cache=True)
def transfer_gains(mu,ex,inh,theta,tm,ref,reset,rise_decay,w2,xx,ww,xs,gs):
    du=np.zeros(len(mu));de=np.zeros(len(mu));di=np.zeros(len(mu))
    for i in range(len(mu)):
        var=max(ex[i],1e-12);sig=np.sqrt(var);sigg=np.sqrt(max(inh[i]*w2,0.))
        shift=mu[i]-1.0325*np.sqrt(max(ex[i]*rise_decay/tm,1e-16))
        exshift=-.51625*np.sqrt(rise_decay/(tm*ex[i])) if ex[i]>1e-12 else 0.
        for k in range(len(xx)):
            mean=shift+np.sqrt(2.)*sigg*xx[k];a=(reset-mean)/sig;b=(theta[i]-mean)/sig
            den=ref+tm*np.sqrt(np.pi)*(primitive(b,xs,gs)-primitive(a,xs,gs))
            if np.isfinite(den) and ref<den<1e150:
                pa=primitive_slope(a,xs,gs);pb=primitive_slope(b,xs,gs);factor=tm*np.sqrt(np.pi)/den**2
                pm=factor*(pb-pa)/sig;pv=factor*(b*pb-a*pa)/(2*var)
                du[i]+=ww[k]*pm
                de[i]+=ww[k]*(pv+pm*exshift) if ex[i]>1e-12 else 0.
                if inh[i]>0:di[i]+=ww[k]*pm*xx[k]*np.sqrt(w2/(2*inh[i]))
    return du,de,di


def accelerate(m):
    xs,gs=siegert_table.table()
    def pe(mu,ex,inh):
        return transfer(mu,np.repeat(ex,m.K),inh,m.theta_u,m.te,m.tref_e,m.v_reset,m.ra+m.ta,m.w2cv_e,m.gh_x,m.gh_w,xs,gs)
    def pi(mu,ex,inh):
        return transfer(mu,ex,inh,np.full(m.n,m.theta_i),m.ti,m.tref_i,m.v_reset,m.ra+m.ta,m.w2cv_i,m.gh_x,m.gh_w,xs,gs)
    m.phi_e=pe;m.phi_i=pi


def initialize(eq,r,s):
    m=eq.m;n=eq.n
    eq.evaluate(r,s);u=eq.last['r_u'].copy();m.reset()
    re,ri=r[:n]/1000,r[n:]/1000
    m.r_u=u;m.r_i=ri.copy();m.m_u=m.tau_M*u
    m.z_u,m.z2_u=eq.z(s)
    m.gAE=m.te*m.gaA*(m.w_ee@re+m.je*m.nu_sig);m.cAE=m.gAE.copy()
    m.gGE=m.te*m.gaG*(m.w_ei@ri);m.cGE=m.gGE.copy()
    m.gAI=m.ti*m.gaA*(m.w_ie@re+m.ji*m.nu_sig);m.cAI=m.gAI.copy()
    m.gGI=m.ti*m.gaG*(m.w_ii@ri);m.cGI=m.gGI.copy()
    m.hE[:]=re;m.hI[:]=ri


def qa():
    eq=Equilibrium();m=eq.m;rng=np.random.default_rng(35)
    mu=rng.uniform(-100,300,m.n*m.K);ex=rng.uniform(.1,300,m.n);inh=rng.uniform(0,400,m.n*m.K)
    mui=mu[:m.n];exi=ex.copy();inhi=inh[:m.n]
    ref=m.phi_e(mu,ex,inh);refi=m.phi_i(mui,exi,inhi)
    accelerate(m)
    report=dict(e_transfer_max_error_hz=float(max(abs(m.phi_e(mu,ex,inh)-ref))*1000),i_transfer_max_error_hz=float(max(abs(m.phi_i(mui,exi,inhi)-refi))*1000))
    seed=np.load(OUT/'extended_low_seed_s-0.05.npz');initialize(eq,seed['r_hz'],-.05)
    t=time.time()
    for _ in range(100):m.step(np.full(m.n,m.nu_sig),m.nu_sig)
    report['seconds_per_step']=(time.time()-t)/100
    report['equilibrium_drift_hz']=float(max(abs(np.r_[m.cell_rate_e(),m.r_i]*1000-seed['r_hz'])))
    (OUT/'compiled_transfer_qa.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)


def run(s,duration,name,seed_file=None):
    eq=Equilibrium();m=eq.m;accelerate(m)
    seed=np.load(seed_file or OUT/'extended_low_seed_s-0.05.npz')
    initialize(eq,seed['r_hz'],float(seed['s']));m.z_u,m.z2_u=eq.z(s)
    nu=np.full(m.n,m.nu_sig);t=time.time();rates=[];samples=[]
    folder=OUT/'deterministic';folder.mkdir(exist_ok=True)
    for i in range(round(duration/m.dt)):
        m.step(nu,m.nu_sig)
        if i%10==9:
            re=m.cell_rate_e()*1000
            rates.append([np.average(re,weights=m.count_e),m.region_rate(re,'175_0'),m.region_rate(re,'175_1')])
            samples.append(np.r_[re,m.r_i*1000])
        if i%5000==4999:
            print(name,(i+1)*m.dt,'ms',rates[-1],time.time()-t,flush=True)
            np.savez_compressed(folder/f'{name}.npz',rates_hz=rates,r_hz=samples,s=s,dt_ms=1.,end_r_u=m.r_u,end_m_u=m.m_u)
    np.savez_compressed(folder/f'{name}.npz',rates_hz=rates,r_hz=samples,s=s,dt_ms=1.,end_r_u=m.r_u,end_m_u=m.m_u)
    print('COMPLETE',name,time.time()-t,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode');p.add_argument('--s',type=float,default=0.);p.add_argument('--duration',type=float,default=4000.);p.add_argument('--name',default='s0');p.add_argument('--seed');a=p.parse_args()
    if a.mode=='qa':qa()
    else:run(a.s,a.duration,a.name,a.seed)
