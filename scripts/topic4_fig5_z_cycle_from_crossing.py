#!/usr/bin/env python3
"""Amplitude continuation of periodic invariant waveforms from a v1 crossing."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import json,argparse,time
import numpy as np
from scipy.linalg import eig
from scipy.sparse.linalg import LinearOperator,gmres
from scipy.signal import resample
from topic4_fig5_z_frozen_v1 import Orbit,Characteristic,OUT,BASE,rate_filters

SCALE=.001


def seed(N,core='a'):
    filename='core_b_crossing_state.npz' if core=='b' else 'oscillatory_crossing_state.npz'
    c=Characteristic();a=np.load(OUT/filename);s=float(a['s']);om=float(a['omega_per_s']);r=a['r_hz'];c.at(r,s)
    vals,vec=eig(c.matrix(1j*om));v=vec[:,np.argmin(abs(vals))];m=c.m;n=m.n;K=m.K;U=n*K
    W=c.weights(1j*om/1000);le,li,lm,*_=rate_filters(m,1j*om/1000)
    ve,vi=v[:n],v[n:];z,z2=c.z,c.z2
    mu=m.te*(np.repeat(W['ee']@ve,K)-z*np.repeat(W['ei']@vi,K))
    ex=m.te*np.repeat(W['vee']@ve,K);inh=m.te*z2*np.repeat(W['vei']@vi,K)
    ru=(c.u*mu+c.v*ex+c.w*inh)/(1/le+c.u*m.eta_M*lm)
    vf=np.r_[ru,vi];idx=int(np.argmax(abs(ru)));vf*=np.exp(-1j*np.angle(vf[idx]));vf/=abs(vf[idx])
    theta=2*np.pi*np.arange(N)/N;wave=(np.exp(1j*theta)[:,None]*vf).real;phase=(1j*np.exp(1j*theta)[:,None]*vf).real
    c.eq.evaluate(r,s);req=np.r_[c.eq.last['r_u'],r[n:]/1000]
    return req,wave,phase,2*np.pi/om*1000,s


class Branch:
    def __init__(self,N,core='a'):
        self.req,self.wave,self.phase,self.T0,self.s0=seed(N,core);self.N=N;self.o=Orbit(self.s0,N)
        self.core=core
        self.pp=self.phase/np.sum(self.phase**2);self.ap=self.wave/np.sum(self.wave**2)

    def set_s(self,s):
        self.o.s=s;self.o.z,self.o.z2=self.o.eq.z(s)

    def solve(self,r,T,s,amp_hz,maxiter=12):
        y=np.r_[(r/SCALE).ravel(),np.log(T),s];hist=[];dest=getattr(self,'destination',OUT/('periodic_from_crossing_b' if self.core=='b' else 'periodic_from_crossing'));dest.mkdir(parents=True,exist_ok=True)
        for it in range(maxiter):
            r=y[:-2].reshape(self.N,-1)*SCALE;T=float(np.exp(y[-2]));s=float(y[-1]);self.set_s(s)
            f,jr=self.o.evaluate_fixed(r,T,True)
            phase=float(np.sum((r-self.req)*self.pp)/SCALE);amp=float(np.sum((r-self.req)*self.ap)/SCALE-amp_hz)
            F=np.r_[(f/SCALE).ravel(),phase,amp];err=float(max(abs(F)));hist.append(err)
            print('AMPLITUDE',amp_hz,'ITER',it,'s',s,'T',T,'res',err,flush=True)
            np.savez_compressed(dest/f'candidate_amp{amp_hz:g}_N{self.N}.npz',r=r,T=T,s=s,control_amplitude_hz=amp_hz,residual_hz=err,history=hist,N=self.N)
            if err<5e-7:break
            eps=1e-5;self.set_s(s+eps);fp=self.o.evaluate_fixed(r,T);self.set_s(s-eps);fm=self.o.evaluate_fixed(r,T);self.set_s(s)
            fs=(fp-fm)/(2*eps)/SCALE;ft=jr.period_derivative/SCALE
            def mv(dy):
                dr=dy[:-2].reshape(self.N,-1)
                out=jr(dr*SCALE)/SCALE+ft*dy[-2]+fs*dy[-1]
                return np.r_[out.ravel(),np.sum(dr*self.pp),np.sum(dr*self.ap)]
            J=LinearOperator((len(y),len(y)),matvec=mv,dtype=float)
            preconditioner=None
            if getattr(self,'use_preconditioner',False):
                from topic4_fig5_z_cycle_preconditioner import bordered_inverse
                try:
                    inv=bordered_inverse(self.o,jr,ft,fs,self.pp,self.ap)
                    preconditioner=LinearOperator(J.shape,matvec=inv,dtype=float)
                except ValueError as exc:print('PRECONDITIONER FALLBACK',str(exc),flush=True)
            nit=[0]
            def count(_):nit[0]+=1
            dy,info=gmres(J,-F,M=preconditioner,rtol=min(1e-5,max(1e-8,err*.001)),atol=1e-10,restart=45,maxiter=8,callback=count,callback_type='pr_norm')
            print('KRYLOV ITERATIONS',nit[0],flush=True)
            print('AMPLITUDE GMRES',info,np.linalg.norm(J@dy+F),flush=True)
            for back in range(10):
                yy=y+2.**(-back)*dy
                if abs(yy[-2]-y[-2])>.3 or abs(yy[-1]-y[-1])>.1:continue
                rr=yy[:-2].reshape(self.N,-1)*SCALE;tt=float(np.exp(yy[-2]));ss=float(yy[-1]);self.set_s(ss)
                ff=self.o.evaluate_fixed(rr,tt)
                trial=np.r_[(ff/SCALE).ravel(),np.sum((rr-self.req)*self.pp)/SCALE,np.sum((rr-self.req)*self.ap)/SCALE-amp_hz]
                if np.linalg.norm(trial)<np.linalg.norm(F):y=yy;break
            else:break
        r=y[:-2].reshape(self.N,-1)*SCALE;T=float(np.exp(y[-2]));s=float(y[-1]);self.set_s(s)
        defect=float(abs(self.o.evaluate_fixed(r,T)).max()*1000)
        phase_error=abs(float(np.sum((r-self.req)*self.pp)*1000))
        control_error=abs(float(np.sum((r-self.req)*self.ap)*1000-amp_hz))
        error=max(defect,phase_error,control_error)
        np.savez_compressed(dest/f'amp{amp_hz:g}_N{self.N}.npz',r=r,T=T,s=s,control_amplitude_hz=amp_hz,residual_hz=error,
                            map_defect_hz=defect,phase_error_hz=phase_error,control_error_hz=control_error,history=hist,N=self.N)
        print('AMPLITUDE RESULT',amp_hz,s,T,error,flush=True)
        return r,T,s,error


def main():
    p=argparse.ArgumentParser();p.add_argument('--N',type=int,default=64);p.add_argument('--amplitudes',default='.05,.1,.25,.5,1,2');p.add_argument('--orbit');p.add_argument('--core',choices=['a','b'],default='a');a=p.parse_args();b=Branch(a.N,a.core)
    last=0.
    if a.orbit:
        z=np.load(a.orbit);r=resample(z['r'],a.N,axis=0);T=float(z['T']);s=float(z['s'])
        last=float(z['control_amplitude_hz'])
    else:r=None;T=b.T0;s=b.s0
    for amp in map(float,a.amplitudes.split(',')):
        if r is None:r=b.req[None,:]+amp/1000*b.wave
        else:r=r+(amp-last)/1000*b.wave
        r,T,s,error=b.solve(r,T,s,amp)
        if error>1e-5:break
        last=amp


if __name__=='__main__':main()
