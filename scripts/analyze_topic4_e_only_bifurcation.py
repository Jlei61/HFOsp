#!/usr/bin/env python3
"""Frozen E-only Z fast-subsystem branches, independently of old all-GABA roots."""
from topic4_e_only_z_rate import EOnlySystem
from topic4_e_only_z_tangent import sample_spectrum
from validate_topic4_fixed_rate_base import ROOT,read,write
from scipy.optimize import root
from scipy.linalg import eig,eigvals
import numpy as np
import time
import argparse

OUT=ROOT/'results/topic4_sef_hfo/z_transition_bifurcation_audit_v1'


class Family:
    def __init__(self,kind='uniform',tau=20.611550480127335):
        self.s=EOnlySystem(tau=tau);self.kind=kind;self.tau=tau
        self.minimum=.3
        if kind=='native_path':
            source=np.load(OUT/'external_input.npz');cell=source['cell_e'];counts=source['count_e']
            from checkpoint import load
            profiles=[np.ones(100)];means=[1.]
            for tm in [8000,9400,9800,10180,10680]:
                state=load(OUT/'checkpoints'/f't{tm}ms.npz');z=state['slow']['z'][:32000]
                profiles.append(np.bincount(cell,weights=z,minlength=100)/counts);means.append(float(z.mean()))
            order=np.argsort(means);self.means=np.array(means)[order];self.profiles=np.array(profiles)[order]
            self.minimum=float(self.means[0])

    def field(self,q):
        if self.kind=='uniform':return q
        return np.array([np.interp(q,self.means,self.profiles[:,i]) for i in range(100)])

    def F(self,r,q):return self.s.F(r,self.field(q))
    def jac(self,r,q):return self.s.jac(r,self.field(q))
    def solve(self,q,r):return self.s.solve(self.field(q),r)


def trace(kind='uniform',tau=20.611550480127335):
    started=time.time();f=Family(kind,tau);s=f.s;rows=[];states=[]
    maxq=1.15 if kind=='uniform' else 1.
    for branch,qs,initial in [('low',np.linspace(maxq,f.minimum,171),np.full(200,.00005)),
                              ('high',np.linspace(f.minimum,maxq,171),np.full(200,.45))]:
        r=initial.copy()
        for q in qs:
            rr,err,ok=f.solve(float(q),r)
            if not ok:
                rows.append({'branch':branch,'q':float(q),'valid':False,'residual':err});break
            r=rr
            ev=eigvals(f.jac(r,q));j=np.argmin(abs(ev))
            rows.append({'branch':branch,'q':float(q),'valid':True,'residual':err,
                         'E_mean_hz':float(np.average(r[:100],weights=s.m.count_e)*1000),
                         'I_mean_hz':float(np.average(r[100:],weights=s.m.count_i)*1000),
                         'nearest_static_eigenvalue_real':float(ev[j].real),
                         'nearest_static_eigenvalue_imag':float(ev[j].imag)})
            states.append((branch,float(q),r.copy()))
    folder=OUT/'reduced_bifurcation';folder.mkdir(exist_ok=True);stem=f'{kind}_tau{tau:g}'
    np.savez_compressed(folder/f'{stem}_branches.npz',branch=np.array([x[0] for x in states]),q=np.array([x[1] for x in states]),r=np.array([x[2] for x in states]))
    write(folder/f'{stem}_branches.json',{'status':'COMPLETE','kind':kind,'tau_ms':tau,'rows':rows,'seconds':time.time()-started,
          'scope':'Equilibrium continuation only; static residual Jacobian is not delayed dynamical stability',
          'parameter':'Uniform E-target Z' if kind=='uniform' else 'Neuron-weighted mean of piecewise-interpolated actual native Z profiles; only observed field range'})
    return rows


def fold(branch,kind='uniform',tau=20.611550480127335):
    family=Family(kind,tau);s=family.s;n=200;folder=OUT/'reduced_bifurcation';stem=f'{kind}_tau{tau:g}'
    a=np.load(folder/f'{stem}_branches.npz');sel=np.flatnonzero(a['branch']==branch);idx=sel[-1];r0=a['r'][idx];q0=float(a['q'][idx])
    ev,V=eig(family.jac(r0,q0));v=V[:,np.argmin(abs(ev))].real;v/=np.linalg.norm(v)
    def fun(y):
        r,q,v=y[:n],y[n],y[n+1:]
        return np.r_[family.F(r,q),family.jac(r,q)@v,v@v-1]
    sol=root(fun,np.r_[r0,q0,v],tol=2e-9);r,q,v=sol.x[:n],float(sol.x[n]),sol.x[n+1:]
    residual=float(np.max(abs(fun(sol.x))))
    upper=1. if kind=='native_path' else 1.2
    if residual>=1e-7 or not family.minimum<=q<=upper or r.min()<-1e-8:
        write(folder/f'{stem}_{branch}_fold.json',{'status':'UNRESOLVED','residual':residual,'q':q,'message':str(sol.message)});return
    ev,L,R=eig(family.jac(r,q),left=True,right=True);j=np.argmin(abs(ev));v=R[:,j].real;v/=np.linalg.norm(v);w=L[:,j].real;w/=w@v
    fq=(family.F(r,q+1e-5)-family.F(r,q-1e-5))/2e-5
    scales=[1e-7,2e-7,5e-7] if branch=='low' else [2e-5,5e-5,1e-4]
    curves=[float(w@(family.F(r+h*v,q)-2*family.F(r,q)+family.F(r-h*v,q))/(h*h)) for h in scales]
    nondegenerate=(abs(w@fq)>1e-8 and min(abs(np.array(curves)))>1e-6 and
                   len(set(np.sign(curves)))==1 and np.sort(abs(ev))[1]>1e-5)
    row={'status':'FOLD_CONDITIONS_PASS' if nondegenerate else 'DEGENERACY_UNRESOLVED','q':q,'kind':kind,'tau_ms':tau,'branch':branch,'residual':residual,
         'E_mean_hz':float(np.average(r[:100],weights=s.m.count_e)*1000),'zero_eigenvalue':float(ev[j].real),
         'next_smallest_eigenvalue_abs':float(np.sort(abs(ev))[1]),'parameter_transversality':float(w@fq),
         'quadratic_steps':scales,'quadratic_coefficients':curves,
         'scope':'Fast reduced equilibrium fold; requires full delayed stability to label attracting boundary; not a native SNN fold proof'}
    np.savez_compressed(folder/f'{stem}_{branch}_fold.npz',r=r,q=q,v=v,w=w)
    write(folder/f'{stem}_{branch}_fold.json',row)
    return row


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--kind',choices=['uniform','native_path'],default='uniform');parser.add_argument('--tau',type=float,default=20.611550480127335);parser.add_argument('--fold',choices=['low','high']);args=parser.parse_args()
    if args.fold:fold(args.fold,args.kind,args.tau)
    else:trace(args.kind,args.tau)
