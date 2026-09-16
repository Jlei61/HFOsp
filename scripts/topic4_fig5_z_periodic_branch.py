#!/usr/bin/env python3
"""Fourier/Newton-Krylov solver for an instantaneous-variance sensitivity model.

All 3200 E threshold/Z units, 400 I cells, physical delay bins, synaptic poles,
and dynamic per-unit M are retained. This is NOT the frozen, filtered-variance v1.
Only periodic, residual-verified solutions
may be entered into a periodic branch figure.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,json,time
import numpy as np
from scipy.sparse.linalg import LinearOperator,gmres
from scipy.signal import find_peaks,resample
from scipy.interpolate import CubicSpline
from numba import njit
from topic4_fig5_z_bifurcation_preview import Equilibrium,OUT
from topic4_fig5_z_branch_dynamics import transfer
import siegert_table


@njit(cache=True)
def frequency_matrix(rows,cols,weights,n,delays,omega):
    out=np.zeros((len(omega),n,n),np.complex128)
    derivative=np.zeros_like(out)
    for k in range(len(omega)):
        phase=np.exp(-1j*omega[k]*delays)
        for p in range(len(weights)):
            d=cols[p]//n;value=weights[p]*phase[d]
            out[k,rows[p],cols[p]%n]+=value
            derivative[k,rows[p],cols[p]%n]+=value*(1j*omega[k]*delays[d])
    return out,derivative


class Orbit:
    def __init__(self,s,N):
        self.eq=Equilibrium();self.m=m=self.eq.m;self.s=s;self.N=N;self.n=m.n;self.U=m.n*m.K
        self.z,self.z2=self.eq.z(s);self.k=np.arange(N//2+1);self.xs,self.gs=siegert_table.table()
        self.thE=np.tile(m.theta_u,N);self.thI=np.full(N*m.n,m.theta_i)
        self.coos={k:o.tocoo() for k,o in m.ops.items()};self.cached=None

    def kernels(self,T):
        if self.cached is not None and T==self.cached[0]:return self.cached[1:5]
        m=self.m;om=2*np.pi*self.k/T;z=1j*om
        kernels={};hp={}
        for key,c in self.coos.items():
            print('KERNEL',key,T,flush=True)
            w,wp=frequency_matrix(c.row,c.col,c.data,m.n,(np.arange(m.D)+1)*m.dt,om)
            ex=key[-1]=='e';gain=m.gaA if ex else m.gaG;rise=m.ra if ex else m.rg;decay=m.ta if ex else m.tg
            kernels[key]=w*(gain/((1+z*rise)*(1+z*decay)))[:,None,None]
            hp[key]=wp*(gain/((1+z*rise)*(1+z*decay)))[:,None,None]+kernels[key]*(z*rise/(1+z*rise)+z*decay/(1+z*decay))[:,None,None]
        LE=1/(1+z*5);LI=1/(1+z*2.5);LM=m.tau_M/(1+z*m.tau_M)
        self.cached=(T,kernels,LE,LI,LM,hp,LE*z*5/(1+z*5),LI*z*2.5/(1+z*2.5),LM*z*m.tau_M/(1+z*m.tau_M))
        return self.cached[1:5]

    def filt(self,r,L):
        return np.fft.irfft(np.fft.rfft(r,axis=0)*L[:,None],n=self.N,axis=0)

    def inputs(self,ru,ri,H,LM):
        m=self.m;n=m.n;N=self.N
        re=(ru*m.w_u).reshape(N,n,m.K).sum(2)
        fe=np.fft.rfft(re,axis=0);fi=np.fft.rfft(ri,axis=0)
        def mean(key,r):return np.fft.irfft(np.einsum('kij,kj->ki',H[key],r),n=N,axis=0)
        cae=m.te*mean('ee',fe);cge=m.te*mean('ei',fi)
        cai=m.ti*mean('ie',fe);cgi=m.ti*mean('ii',fi)
        mu=np.repeat(cae,m.K,axis=1)-self.z*np.repeat(cge,m.K,axis=1)-m.eta_M*self.filt(ru,LM)
        ex=m.te*(re@m.v_ee.T);inh=self.z2*np.repeat(m.te*(ri@m.v_ei.T),m.K,axis=1)
        return [mu,ex,inh,cai-cgi,m.ti*(re@m.v_ie.T),m.ti*(ri@m.v_ii.T)]

    def phi(self,vals):
        m=self.m;mu,ex,inh,mui,ei,ii=vals
        e=transfer(mu.ravel(),np.repeat(ex,m.K,axis=1).ravel(),inh.ravel(),self.thE,m.te,m.tref_e,m.v_reset,m.ra+m.ta,m.w2cv_e,m.gh_x,m.gh_w,self.xs,self.gs).reshape(self.N,self.U)
        i=transfer(mui.ravel(),ei.ravel(),ii.ravel(),self.thI,m.ti,m.tref_i,m.v_reset,m.ra+m.ta,m.w2cv_i,m.gh_x,m.gh_w,self.xs,self.gs).reshape(self.N,m.n)
        return e,i

    def evaluate_fixed(self,r,T,gains=False):
        m=self.m;H,LE,LI,LM=self.kernels(T)
        vals=self.inputs(r[:,:self.U],r[:,self.U:],H,LM)
        vals[0]+=m.te*m.gaA*m.je*m.nu_sig;vals[1]+=m.te*m.je**2*m.nu_sig
        vals[3]+=m.ti*m.gaA*m.ji*m.nu_sig;vals[4]+=m.ti*m.ji**2*m.nu_sig
        pe,pi=self.phi(vals)
        residual=r-np.c_[self.filt(pe,LE),self.filt(pi,LI)]
        if not gains:return residual
        ge,gi=self.gains(vals)
        def derivative(dr):
            dvals=self.inputs(dr[:,:self.U],dr[:,self.U:],H,LM)
            de=ge[0]*dvals[0]+ge[1]*np.repeat(dvals[1],m.K,axis=1)+ge[2]*dvals[2]
            di=sum(a*b for a,b in zip(gi,dvals[3:]))
            return dr-np.c_[self.filt(de,LE),self.filt(di,LI)]
        hp,lep,lip,lmp=self.cached[5:]
        partial=self.inputs(r[:,:self.U],r[:,self.U:],hp,lmp)
        derivative.period_derivative=-np.c_[self.filt(ge[0]*partial[0],LE)+self.filt(pe,lep),self.filt(gi[0]*partial[3],LI)+self.filt(pi,lip)]
        derivative.transfer_gains=(ge,gi)
        derivative.reference_phi=(pe,pi)
        return residual,derivative

    def gains(self,vals):
        ge=[];gi=[];h=2e-4
        for k in range(3):
            hi=[x.copy() for x in vals];lo=[x.copy() for x in vals]
            hi[k]+=h;lo[k]-=h;hi[k+3]+=h;lo[k+3]-=h
            ep,ip=self.phi(hi);em,im=self.phi(lo);ge.append((ep-em)/(2*h));gi.append((ip-im)/(2*h))
        return ge,gi

    def solve(self,r,T,maxiter=8):
        # Unknown log-period; finite-difference period column includes every kernel.
        ref=r.copy();dref=np.fft.irfft(2j*np.pi*self.k[:,None]*np.fft.rfft(ref,axis=0),n=self.N,axis=0)
        phase=dref/np.sum(dref*dref)*.01;y=np.r_[(r/.01).ravel(),np.log(T)];history=[]
        for it in range(maxiter):
            T=np.exp(y[-1]);r=y[:-1].reshape(self.N,-1)*.01
            f,dr=self.evaluate_fixed(r,T,True)
            F=np.r_[(f/.01).ravel(),np.sum((r-ref)*phase)/.01];err=float(max(abs(F)))
            history.append(err);print('ORBIT',self.s,self.N,it,'T',T,'residual',err,flush=True)
            if getattr(self,'progress_path',None) is not None:
                np.savez_compressed(self.progress_path,r=r,T=T,s=self.s,residual_hz=err*10,history=history,N=self.N)
            if err<1e-8:break
            fp=dr.period_derivative/.01
            def mv(dy):return np.r_[(dr(dy[:-1].reshape(self.N,-1)*.01)/.01+fp*dy[-1]).ravel(),np.sum(dy[:-1].reshape(self.N,-1)*phase)]
            J=LinearOperator((len(y),len(y)),matvec=mv,dtype=float)
            dy,info=gmres(J,-F,rtol=max(1e-6,min(1e-3,err*.01)),atol=1e-10,restart=35,maxiter=6)
            print('GMRES',info,np.linalg.norm(J@dy+F),flush=True)
            for back in range(9):
                yy=y+2.**(-back)*dy
                if abs(yy[-1]-y[-1])>.3:continue
                rr=yy[:-1].reshape(self.N,-1)*.01
                ff=self.evaluate_fixed(rr,np.exp(yy[-1]))
                new=np.r_[(ff/.01).ravel(),np.sum((rr-ref)*phase)/.01]
                if np.linalg.norm(new)<np.linalg.norm(F):y=yy;break
            else:break
        r=y[:-1].reshape(self.N,-1)*.01;T=float(np.exp(y[-1]));error=float(abs(self.evaluate_fixed(r,T)).max()*1000)
        return r,T,error,history


def main():
    p=argparse.ArgumentParser();p.add_argument('--s',type=float,default=0.);p.add_argument('--N',type=int,default=128);p.add_argument('--iterations',type=int,default=6);p.add_argument('--orbit');a=p.parse_args()
    o=Orbit(a.s,a.N);m=o.m
    if a.orbit:
        z=np.load(a.orbit);r=resample(z['r'],a.N,axis=0);T=float(z['T'])
    else:
        z=np.load(OUT/'deterministic/s0_from_low.npz');macro=z['r_hz']/1000
        pk,_=find_peaks(z['rates_hz'][:,1],height=20,distance=100);start,end=pk[-2:];T=float(end-start)
        base=CubicSpline(np.arange(start-5,end+6)-start,macro[start-5:end+6])(np.arange(a.N)*T/a.N)
        r=np.c_[np.repeat(base[:,:m.n],m.K,axis=1),base[:,m.n:]]
    t=time.time();r,T,err,h=o.solve(r,T,a.iterations)
    folder=OUT/'periodic';folder.mkdir(exist_ok=True);path=folder/f's{a.s:g}_N{a.N}.npz'
    np.savez_compressed(path,r=r,T=T,s=a.s,residual_hz=err,history=h,N=a.N)
    print('RESULT',str(path),T,err,'seconds',time.time()-t,flush=True)


if __name__=='__main__':main()
