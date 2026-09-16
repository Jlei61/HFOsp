#!/usr/bin/env python3
"""Actual frozen v1: exact 0.1-ms rate/M/filter map, including variance kernels.

The Fourier solution is an invariant periodic waveform sampled by the original
autonomous map. No claim of a finite integer-step period is implied.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,json,time,gc
import numpy as np
from scipy import sparse
from scipy.signal import resample
from scipy.linalg import eigvals
from scipy.optimize import root
from topic4_fig5_z_periodic_branch import Orbit as InstantOrbit,frequency_matrix
from topic4_fig5_z_characteristic import Characteristic as InstantCharacteristic
from topic4_fig5_z_bifurcation_preview import OUT as BASE
from topic4_fig5_z_branch_dynamics import transfer_gains

OUT=BASE/'frozen_filtered_v1'


def filters(m,lam,ex):
    q=np.exp(-lam*m.dt);a,b=(m.arA,m.adA) if ex else (m.arG,m.adG)
    rise=m.ra if ex else m.rg
    mean=(m.dt/rise)*(1-b)/((1-a*q)*(1-b*q))
    meanp=mean*lam*m.dt*q*(a/(1-a*q)+b/(1-b*q))
    c=np.array([a*a,b*b,a*b]);sg=np.array([1.,1.,-2.]);norm=np.sum(sg*c/(1-c))
    var=sum(sg[k]*c[k]/(1-c[k]*q) for k in range(3))/norm
    varp=sum(sg[k]*c[k]**2*lam*m.dt*q/(1-c[k]*q)**2 for k in range(3))/norm
    return mean,meanp,var,varp


def rate_filters(m,lam):
    q=np.exp(-lam*m.dt);be=m.dt/5;bi=m.dt/2.5;am=1-m.dt/m.tau_M
    le=be*q/(1-(1-be)*q);li=bi*q/(1-(1-bi)*q);lm=m.dt/(1-am*q)
    return le,li,lm,le*lam*m.dt/(1-(1-be)*q),li*lam*m.dt/(1-(1-bi)*q),lm*am*lam*m.dt*q/(1-am*q)


class Orbit(InstantOrbit):
    def __init__(self,s,N):
        super().__init__(s,N)
        self.vcoos={k:sparse.load_npz(self.m.folder/f'vdelay_{k}.npz').tocoo() for k in self.coos}
        (OUT/'periodic').mkdir(parents=True,exist_ok=True)
        self.progress_path=OUT/f'periodic/candidate_s{s:g}_N{N}.npz'

    def gains(self,vals):
        m=self.m;mu,ex,inh,mui,ei,ii=vals
        ge=transfer_gains(mu.ravel(),np.repeat(ex,m.K,axis=1).ravel(),inh.ravel(),self.thE,m.te,m.tref_e,m.v_reset,m.ra+m.ta,m.w2cv_e,m.gh_x,m.gh_w,self.xs,self.gs)
        gi=transfer_gains(mui.ravel(),ei.ravel(),ii.ravel(),self.thI,m.ti,m.tref_i,m.v_reset,m.ra+m.ta,m.w2cv_i,m.gh_x,m.gh_w,self.xs,self.gs)
        return [x.reshape(self.N,self.U) for x in ge],[x.reshape(self.N,m.n) for x in gi]

    def kernels(self,T):
        if self.cached is not None and T==self.cached[0]:return self.cached[1:5]
        m=self.m;om=2*np.pi*self.k/T;lam=1j*om;H={};HP={}
        for key,c in self.coos.items():
            print('EXACT KERNEL',key,T,flush=True)
            v=self.vcoos[key];me,mp,va,vp=filters(m,lam,key[-1]=='e')
            for tag,co,f,fp in [(key,c,me,mp),('v'+key,v,va,vp)]:
                w,wp=frequency_matrix(co.row,co.col,co.data,m.n,(np.arange(m.D)+1)*m.dt,om)
                H[tag]=w*f[:,None,None];HP[tag]=wp*f[:,None,None]+w*fp[:,None,None]
        le,li,lm,lep,lip,lmp=rate_filters(m,lam)
        self.cached=(T,H,le,li,lm,HP,lep,lip,lmp)
        return self.cached[1:5]

    def inputs(self,ru,ri,H,LM):
        m=self.m;n=m.n;N=self.N
        re=(ru*m.w_u).reshape(N,n,m.K).sum(2);fe=np.fft.rfft(re,axis=0);fi=np.fft.rfft(ri,axis=0)
        def conv(key,r):return np.fft.irfft(np.einsum('kij,kj->ki',H[key],r),n=N,axis=0)
        cae=m.te*conv('ee',fe);cge=m.te*conv('ei',fi);cai=m.ti*conv('ie',fe);cgi=m.ti*conv('ii',fi)
        mu=np.repeat(cae,m.K,axis=1)-self.z*np.repeat(cge,m.K,axis=1)-m.eta_M*self.filt(ru,LM)
        ex=m.te*conv('vee',fe);inh=self.z2*np.repeat(m.te*conv('vei',fi),m.K,axis=1)
        return [mu,ex,inh,cai-cgi,m.ti*conv('vie',fe),m.ti*conv('vii',fi)]

    def evaluate_fixed(self,r,T,gains=False):
        result=super().evaluate_fixed(r,T,gains)
        if not gains:return result
        f,derivative=result;ge,gi=derivative.transfer_gains;pe,pi=derivative.reference_phi
        _,_,le,li,lm,hp,lep,lip,lmp=self.cached
        p=self.inputs(r[:,:self.U],r[:,self.U:],hp,lmp)
        de=ge[0]*p[0]+ge[1]*np.repeat(p[1],self.m.K,axis=1)+ge[2]*p[2]
        di=sum(a*b for a,b in zip(gi,p[3:]))
        derivative.period_derivative=-np.c_[self.filt(de,le)+self.filt(pe,lep),self.filt(di,li)+self.filt(pi,lip)]
        return f,derivative


class Characteristic(InstantCharacteristic):
    def __init__(self):
        super().__init__()
        self.vcoos={k:sparse.load_npz(self.m.folder/f'vdelay_{k}.npz').tocoo() for k in self.coos}

    def weights(self,lam):
        if lam in self.weights_cache:return self.weights_cache[lam]
        m=self.m;n=m.n;phase=np.exp(-lam*(np.arange(m.D)+1)*m.dt);W={}
        for key,c in self.coos.items():
            me,mp,va,vp=filters(m,lam,key[-1]=='e')
            for tag,co,f in [(key,c,me),('v'+key,self.vcoos[key],va)]:
                index=co.row*n+co.col%n;v=co.data*phase[co.col//n]
                a=np.bincount(index,weights=v.real,minlength=n*n)+1j*np.bincount(index,weights=v.imag,minlength=n*n)
                W[tag]=a.reshape(n,n)*f
        if len(self.weights_cache)>50:self.weights_cache.clear()
        self.weights_cache[lam]=W
        return W

    def matrix(self,lam_per_s):
        m=self.m;n=m.n;lam=complex(lam_per_s)/1000;W=self.weights(lam)
        le,li,lm,*_=rate_filters(m,lam);den=1/le+self.u*m.eta_M*lm
        def avg(a):return (a*m.w_u).reshape(n,m.K).sum(1)
        am=avg(self.u/den);av=avg(self.v/den);az=avg(self.u*self.z/den);aw=avg(self.w*self.z2/den)
        ee=am[:,None]*m.te*W['ee']+av[:,None]*m.te*W['vee']
        ei=-az[:,None]*m.te*W['ei']+aw[:,None]*m.te*W['vei']
        ie=li*(self.ui[:,None]*m.ti*W['ie']+self.vi[:,None]*m.ti*W['vie'])
        ii=li*(-self.ui[:,None]*m.ti*W['ii']+self.wi[:,None]*m.ti*W['vii'])
        return np.eye(2*n)-np.block([[ee,ei],[ie,ii]])

    def rhp_count(self,spacing=4.,omega=None):
        # Count multipliers outside |zeta|=1, not roots of the continuum model.
        m=self.m;nyquist=np.pi/m.dt*1000
        be=m.dt/5;am=1-m.dt/m.tau_M;ar=1-be
        # Elimination poles of the local rate/M block must all be stable.
        roots=np.array([np.roots([1.,-(ar+am-be*u*m.eta_M*m.dt),ar*am]) for u in self.u])
        assert np.max(abs(roots))<1
        freq=np.unique(np.r_[0,np.geomspace(.01,10,15),np.arange(10,202,spacing),np.geomspace(202,nyquist,45)])
        cache={}
        def val(z):
            if z not in cache:cache[z]=np.linalg.slogdet(self.matrix(z))
            return cache[z]
        def piece(a,b,depth=0):
            va,la=val(a);vb,lb=val(b);mid=(a+b)/2;vm,lm=val(mid)
            p1=np.angle(vm/va);p2=np.angle(vb/vm)
            if depth<12 and (max(abs(p1),abs(p2))>.35 or abs(lm-(la+lb)/2)>.15):
                l=piece(a,mid,depth+1);r=piece(mid,b,depth+1);return l[0]+r[0],max(l[1],r[1])
            return float(p1+p2),float(max(abs(p1),abs(p2)))
        phase=0.;maximum=0.
        for a,b in zip(freq[:-1],freq[1:]):
            dp,mx=piece(1j*a,1j*b);phase+=dp;maximum=max(maximum,mx)
        return dict(unstable_multiplier_count=int(round(-phase/np.pi)),raw_count=-phase/np.pi,
                    contour_evaluations=len(cache),max_phase_step=maximum,base_spacing_per_s=spacing,
                    local_elimination_pole_max_modulus=float(np.max(abs(roots))))


def main():
    p=argparse.ArgumentParser();p.add_argument('--s',type=float,default=0.);p.add_argument('--N',type=int,default=128);p.add_argument('--iterations',type=int,default=12);p.add_argument('--orbit',required=True);a=p.parse_args()
    (OUT/'periodic').mkdir(parents=True,exist_ok=True);o=Orbit(a.s,a.N);z=np.load(a.orbit);r=resample(z['r'],a.N,axis=0)
    t=time.time();r,T,err,h=o.solve(r,float(z['T']),a.iterations)
    np.savez_compressed(OUT/f'periodic/s{a.s:g}_N{a.N}.npz',r=r,T=T,s=a.s,residual_hz=err,history=h,N=a.N,
                        model='FROZEN_FILTERED_V1_EXACT_MAP',dt_ms=o.m.dt)
    print('FROZEN V1 RESULT',a.s,T,err,'seconds',time.time()-t,flush=True)


if __name__=='__main__':main()
