"""Physical-D working-point exploration of the Fig.5 filtered rate/M map.

The only biological tuning is an I->E synaptic weight multiplier. Its second
moment scales quadratically. A C2 primitive of erfcx(-x) replaces the old
piecewise-linear primitive; this is an explicitly audited numerical refinement.
Z is a frozen per-neuron field; M remains dynamic in all characteristic kernels.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.special import erfcx
from scipy.interpolate import CubicHermiteSpline
from numba import njit
from topic4_fig5_z_bifurcation_preview import Equilibrium as OriginalEquilibrium, ROOT
from topic4_fig5_z_frozen_v1 import Characteristic as OriginalCharacteristic, Orbit as OriginalOrbit
from topic4_fig5_z_branch_dynamics import primitive as old_primitive

OUT=ROOT/'results/topic4_sef_hfo/fig5_D_physical_bifurcation_20260916'
OUT.mkdir(parents=True,exist_ok=True)
_TABLE=None

def table():
    global _TABLE
    if _TABLE is None:
        x=np.arange(-60.,26.+.00025,.0005)
        f=erfcx(-x);df=2*x*f+2/np.sqrt(np.pi)
        p=CubicHermiteSpline(x,f,df).antiderivative()
        p.c[-1]-=p(0.)
        _TABLE=(x,np.ascontiguousarray(p.c.T))
    return _TABLE

@njit(cache=True)
def primitive(x,xs,gs):
    if x<xs[0]:
        return gs[0,4]-(np.log(abs(x))-np.log(abs(xs[0]))-.25*(1/x**2-1/xs[0]**2))/np.sqrt(np.pi)
    if x>xs[-1]:
        if x>26.6:return np.inf
        hi=xs[-1];h=hi-xs[-2];c=gs[-1]
        end=((((c[0]*h+c[1])*h+c[2])*h+c[3])*h+c[4])
        return end+np.exp(x*x)/x*(1+1/(2*x*x)+3/(4*x**4))-np.exp(hi*hi)/hi*(1+1/(2*hi*hi)+3/(4*hi**4))
    i=min(max(int((x-xs[0])/(xs[1]-xs[0])),0),len(xs)-2)
    h=x-xs[i];c=gs[i]
    return ((((c[0]*h+c[1])*h+c[2])*h+c[3])*h+c[4])

@njit(cache=True)
def primitive_slope(x,xs,gs):
    if x<xs[0]:return -(1/x+.5/x**3)/np.sqrt(np.pi)
    if x>xs[-1]:return np.exp(x*x)*(2-3.75/x**6)
    i=min(max(int((x-xs[0])/(xs[1]-xs[0])),0),len(xs)-2)
    h=x-xs[i];c=gs[i]
    return ((4*c[0]*h+3*c[1])*h+2*c[2])*h+c[3]

@njit(cache=True)
def transfer(mu,ex,inh,theta,tm,ref,reset,rise_decay,w2,xx,ww,xs,gs):
    out=np.zeros(len(mu))
    for i in range(len(mu)):
        sigma=np.sqrt(max(ex[i],1e-12));sigg=np.sqrt(max(inh[i]*w2,0.))
        shift=mu[i]-1.0325*np.sqrt(max(ex[i]*rise_decay/tm,1e-16))
        for k in range(len(xx)):
            mean=shift+np.sqrt(2.)*sigg*xx[k]
            integral=primitive((theta[i]-mean)/sigma,xs,gs)-primitive((reset-mean)/sigma,xs,gs)
            den=ref+tm*np.sqrt(np.pi)*integral
            if np.isfinite(den) and den>0:out[i]+=ww[k]*min(1/den,1/ref)
    return out

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

def smooth(m):
    xs,gs=table()
    argsE=(m.theta_u,m.te,m.tref_e,m.v_reset,m.ra+m.ta,m.w2cv_e,m.gh_x,m.gh_w,xs,gs)
    argsI=(np.full(m.n,m.theta_i),m.ti,m.tref_i,m.v_reset,m.ra+m.ta,m.w2cv_i,m.gh_x,m.gh_w,xs,gs)
    m.phi_e=lambda mu,ex,inh:transfer(mu,np.repeat(ex,m.K),inh,*argsE)
    m.phi_i=lambda mu,ex,inh:transfer(mu,ex,inh,*argsI)
    m.gains_e=lambda mu,ex,inh:transfer_gains(mu,np.repeat(ex,m.K),inh,*argsE)
    m.gains_i=lambda mu,ex,inh:transfer_gains(mu,ex,inh,*argsI)

class Equilibrium(OriginalEquilibrium):
    def __init__(self,q_ie=1.):
        super().__init__();m=self.m;self.q_ie=float(q_ie)
        m.w_ei*=q_ie;m.v_ei*=q_ie**2;m.ops['ei']=m.ops['ei']*q_ie
        smooth(m)
        # Preserve the observed path to its final field, then deplete that field
        # proportionally to zero. No extrapolated neuron may acquire Z>1 or Z<0.
        self.fields=np.vstack([self.fields,np.zeros_like(self.fields[0])])
        self.times.append(np.nan);self.prepare_path()

    def full_z(self,D):
        k=int(np.clip(np.searchsorted(self.ss,D)-1,0,len(self.ss)-2))
        a=(D-self.ss[k])/(self.ss[k+1]-self.ss[k])
        return (1-a)*self.fields[k]+a*self.fields[k+1]

    def evaluate(self,rhz,D,jac=False):
        m=self.m;n=self.n;re,ri=np.asarray(rhz[:n])/1000,np.asarray(rhz[n:])/1000
        z,z2=self.z(D)
        cae=m.te*m.gaA*(m.w_ee@re+m.je*m.nu_sig);cge=m.te*m.gaG*(m.w_ei@ri)
        cai=m.ti*m.gaA*(m.w_ie@re+m.ji*m.nu_sig);cgi=m.ti*m.gaG*(m.w_ii@ri)
        mu0=np.repeat(cae,m.K)-z*np.repeat(cge,m.K)
        exe=m.te*(m.v_ee@re+m.je*m.je*m.nu_sig);inhe=z2*np.repeat(m.te*(m.v_ei@ri),m.K)
        exi=m.ti*(m.v_ie@re+m.ji*m.ji*m.nu_sig);inhi=m.ti*(m.v_ii@ri)
        r=m.phi_e(mu0,exe,inhe)
        for _ in range(30):
            mu=mu0-m.eta_M*m.tau_M*r;phi=m.phi_e(mu,exe,inhe)
            u=m.gains_e(mu,exe,inhe)[0];step=(r-phi)/(1+m.eta_M*m.tau_M*u);r-=step
            if max(abs(step))<1e-14:break
        mu=mu0-m.eta_M*m.tau_M*r
        pe=(r*m.w_u).reshape(n,m.K).sum(1);pi=m.phi_i(cai-cgi,exi,inhi)
        f=np.r_[pe,pi]*1000-rhz
        self.last=dict(r_u=r,mu=mu,ex=exe,inh=inhe,local_error_hz=float(max(abs(r-m.phi_e(mu,exe,inhe)))*1000))
        if not jac:return f
        u,v,w=m.gains_e(mu,exe,inhe);den=1+m.eta_M*m.tau_M*u
        avg=lambda a:(a*m.w_u).reshape(n,m.K).sum(1)
        jee=avg(u/den)[:,None]*m.te*m.gaA*m.w_ee+avg(v/den)[:,None]*m.te*m.v_ee
        jei=-avg(u*z/den)[:,None]*m.te*m.gaG*m.w_ei+avg(w*z2/den)[:,None]*m.te*m.v_ei
        ui,vi,wi=m.gains_i(cai-cgi,exi,inhi)
        jie=ui[:,None]*m.ti*m.gaA*m.w_ie+vi[:,None]*m.ti*m.v_ie
        jii=-ui[:,None]*m.ti*m.gaG*m.w_ii+wi[:,None]*m.ti*m.v_ii
        return f,np.block([[jee,jei],[jie,jii]])-np.eye(2*n)

class Characteristic(OriginalCharacteristic):
    def __init__(self,q_ie=1.):
        self.eq=Equilibrium(q_ie);self.m=self.eq.m
        self.coos={k:o.tocoo() for k,o in self.m.ops.items()}
        self.vcoos={k:sparse.load_npz(self.m.folder/f'vdelay_{k}.npz').tocoo() for k in self.coos}
        self.vcoos['ei'].data*=q_ie**2;self.weights_cache={}

    def at(self,r,D):
        m=self.m;self.r=r;self.s=D;self.z,self.z2=self.eq.z(D);self.eq.evaluate(r,D)
        p=self.eq.last;self.u,self.v,self.w=m.gains_e(p['mu'],p['ex'],p['inh'])
        re,ri=r[:m.n]/1000,r[m.n:]/1000
        mu=m.ti*(m.gaA*(m.w_ie@re+m.ji*m.nu_sig)-m.gaG*(m.w_ii@ri))
        ex=m.ti*(m.v_ie@re+m.ji*m.ji*m.nu_sig);inh=m.ti*(m.v_ii@ri)
        self.ui,self.vi,self.wi=m.gains_i(mu,ex,inh)
        return self

class Orbit(OriginalOrbit):
    def __init__(self,D,N,q_ie=1.):
        self.eq=Equilibrium(q_ie);self.m=m=self.eq.m;self.s=D;self.N=N;self.n=m.n;self.U=m.n*m.K
        self.z,self.z2=self.eq.z(D);self.k=np.arange(N//2+1);self.xs,self.gs=table()
        self.thE=np.tile(m.theta_u,N);self.thI=np.full(N*m.n,m.theta_i)
        self.coos={k:o.tocoo() for k,o in m.ops.items()};self.cached=None
        self.vcoos={k:sparse.load_npz(m.folder/f'vdelay_{k}.npz').tocoo() for k in self.coos}
        self.vcoos['ei'].data*=q_ie**2

    def phi(self,vals):
        m=self.m;mu,ex,inh,mui,ei,ii=vals
        e=transfer(mu.ravel(),np.repeat(ex,m.K,axis=1).ravel(),inh.ravel(),self.thE,m.te,m.tref_e,m.v_reset,m.ra+m.ta,m.w2cv_e,m.gh_x,m.gh_w,self.xs,self.gs)
        i=transfer(mui.ravel(),ei.ravel(),ii.ravel(),self.thI,m.ti,m.tref_i,m.v_reset,m.ra+m.ta,m.w2cv_i,m.gh_x,m.gh_w,self.xs,self.gs)
        return e.reshape(self.N,self.U),i.reshape(self.N,m.n)

    def gains(self,vals):
        m=self.m;mu,ex,inh,mui,ei,ii=vals
        ge=transfer_gains(mu.ravel(),np.repeat(ex,m.K,axis=1).ravel(),inh.ravel(),self.thE,m.te,m.tref_e,m.v_reset,m.ra+m.ta,m.w2cv_e,m.gh_x,m.gh_w,self.xs,self.gs)
        gi=transfer_gains(mui.ravel(),ei.ravel(),ii.ravel(),self.thI,m.ti,m.tref_i,m.v_reset,m.ra+m.ta,m.w2cv_i,m.gh_x,m.gh_w,self.xs,self.gs)
        return [x.reshape(self.N,self.U) for x in ge],[x.reshape(self.N,m.n) for x in gi]
