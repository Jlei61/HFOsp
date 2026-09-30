#!/usr/bin/env python3
"""Autonomous frozen-v1 trajectory, with paired sparse products for speed only."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,time,json,pickle
import numpy as np
from numba import njit
from scipy import sparse
from topic4_fig5_z_bifurcation_preview import Equilibrium,OUT as BASE
from topic4_fig5_z_branch_dynamics import accelerate,initialize

OUT=BASE/'frozen_filtered_v1'


@njit(cache=True)
def paired_product(w,v,col,ptr,x):
    y=np.zeros(len(ptr)-1);z=np.zeros_like(y)
    for i in range(len(y)):
        a=0.;b=0.
        for k in range(ptr[i],ptr[i+1]):
            xx=x[col[k]];a+=w[k]*xx;b+=v[k]*xx
        y[i]=a;z[i]=b
    return y,z


class Part:
    def __init__(self,pair,index):self.pair=pair;self.index=index
    def __matmul__(self,x):
        if self.index==0:self.pair['result']=paired_product(self.pair['w'],self.pair['v'],self.pair['col'],self.pair['ptr'],x)
        return self.pair['result'][self.index]


def make(fast=True,s=0.):
    eq=Equilibrium();m=eq.m;accelerate(m);m.v['variance']='filtered'
    m.vops={k:sparse.load_npz(m.folder/f'vdelay_{k}.npz') for k in m.ops}
    seed=np.load(BASE/'extended_low_seed_s-0.05.npz');initialize(eq,seed['r_hz'],-.05)
    re=m.cell_rate_e();ri=m.r_i
    for k,x,a,b in [('ee',m.v_ee@re+m.je**2*m.nu_sig,m.arA,m.adA),('ie',m.v_ie@re+m.ji**2*m.nu_sig,m.arA,m.adA),('ei',m.v_ei@ri,m.arG,m.adG),('ii',m.v_ii@ri,m.arG,m.adG)]:
        for j,c in enumerate([a*a,b*b,a*b]):m.y[k][j]=c/(1-c)*x
    m.z_u,m.z2_u=eq.z(s)
    if fast:
        for k in list(m.ops):
            a,b=m.ops[k],m.vops[k]
            assert np.array_equal(a.indices,b.indices) and np.array_equal(a.indptr,b.indptr)
            pair=dict(w=a.data,v=b.data,col=a.indices,ptr=a.indptr)
            m.ops[k]=Part(pair,0);m.vops[k]=Part(pair,1)
    return eq


def qa():
    a=make(True);b=make(False);nu=np.full(a.m.n,a.m.nu_sig);t=time.time()
    for _ in range(50):a.m.step(nu,a.m.nu_sig);b.m.step(nu,b.m.nu_sig)
    names=['r_u','r_i','m_u','gAE','cAE','gGE','cGE','hE','hI']
    report=dict(max_state_difference=float(max(np.max(abs(getattr(a.m,k)-getattr(b.m,k))) for k in names)))
    t=time.time()
    for _ in range(100):a.m.step(nu,a.m.nu_sig)
    report['seconds_per_step']=(time.time()-t)/100
    (OUT/'paired_sparse_qa.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)


def run(s,duration):
    eq=make(True,s);m=eq.m;nu=np.full(m.n,m.nu_sig);r=[];t=time.time();folder=OUT/'deterministic';folder.mkdir(exist_ok=True)
    for i in range(round(duration/m.dt)):
        m.step(nu,m.nu_sig)
        if i%10==9:r.append(np.r_[m.r_u,m.r_i].astype(np.float32)*1000)
        if i%2500==2499:print('GUIDE',s,(i+1)*m.dt,'ms','R',np.average(m.cell_rate_e(),weights=m.count_e)*1000,'seconds',time.time()-t,flush=True)
    np.savez_compressed(folder/f's{s:g}.npz',r_hz=r,dt_ms=1.,s=s)
    with (folder/f's{s:g}_end.pkl').open('wb') as f:pickle.dump(m.state_dict(),f)
    print('GUIDE COMPLETE',s,duration,time.time()-t,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode');p.add_argument('--s',type=float,default=0.);p.add_argument('--duration',type=float,default=2000.);a=p.parse_args();OUT.mkdir(exist_ok=True)
    if a.mode=='qa':qa()
    else:run(a.s,a.duration)
