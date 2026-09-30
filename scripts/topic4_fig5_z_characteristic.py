#!/usr/bin/env python3
"""Exact-delay characteristic of the continuous v1 rate/M equations."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
from scipy.linalg import eigvals
from scipy.optimize import root
from topic4_fig5_z_bifurcation_preview import Equilibrium,OUT
from topic4_fig5_z_branch_dynamics import accelerate


class Characteristic:
    def __init__(self):
        self.eq=Equilibrium();self.m=m=self.eq.m;accelerate(m)
        self.coos={k:(o.tocoo()) for k,o in m.ops.items()};self.weights_cache={}

    def at(self,r,s):
        q=self.eq;m=self.m;n=m.n;q.evaluate(r,s)
        self.r=r;self.s=s;self.z,self.z2=q.z(s)
        mu=q.last['mu'];ex=q.last['ex'];inh=q.last['inh'];h=2e-4
        self.u=(m.phi_e(mu+h,ex,inh)-m.phi_e(mu-h,ex,inh))/(2*h)
        self.v=(m.phi_e(mu,ex+h,inh)-m.phi_e(mu,ex-h,inh))/(2*h)
        self.w=(m.phi_e(mu,ex,inh+h)-m.phi_e(mu,ex,inh-h))/(2*h)
        re,ri=r[:n]/1000,r[n:]/1000
        mui=m.ti*(m.gaA*(m.w_ie@re+m.ji*m.nu_sig)-m.gaG*(m.w_ii@ri))
        ei=m.ti*(m.v_ie@re+m.ji*m.ji*m.nu_sig);ii=m.ti*(m.v_ii@ri)
        self.ui=(m.phi_i(mui+h,ei,ii)-m.phi_i(mui-h,ei,ii))/(2*h)
        self.vi=(m.phi_i(mui,ei+h,ii)-m.phi_i(mui,ei-h,ii))/(2*h)
        self.wi=(m.phi_i(mui,ei,ii+h)-m.phi_i(mui,ei,ii-h))/(2*h)
        return self

    def weights(self,lam):
        if lam in self.weights_cache:return self.weights_cache[lam]
        m=self.m;n=m.n;phase=np.exp(-lam*(np.arange(m.D)+1)*m.dt);W={}
        for key,c in self.coos.items():
            index=c.row*n+c.col%n;v=c.data*phase[c.col//n]
            a=np.bincount(index,weights=v.real,minlength=n*n)+1j*np.bincount(index,weights=v.imag,minlength=n*n)
            ex=key[-1]=='e';gain=m.gaA if ex else m.gaG;rise=m.ra if ex else m.rg;decay=m.ta if ex else m.tg
            W[key]=a.reshape(n,n)*gain/((1+lam*rise)*(1+lam*decay))
        if len(self.weights_cache)>100:self.weights_cache.clear()
        self.weights_cache[lam]=W
        return W

    def matrix(self,lam_per_s):
        m=self.m;n=m.n;lam=complex(lam_per_s)/1000;W=self.weights(lam)
        den=1+lam*5+self.u*m.eta_M/(lam+1/m.tau_M)
        def avg(a):return (a*m.w_u).reshape(n,m.K).sum(1)
        am=avg(self.u/den);av=avg(self.v/den);az=avg(self.u*self.z/den);aw=avg(self.w*self.z2/den)
        ee=am[:,None]*m.te*W['ee']+av[:,None]*m.te*m.v_ee
        ei=-az[:,None]*m.te*W['ei']+aw[:,None]*m.te*m.v_ei
        ie=(self.ui[:,None]*m.ti*W['ie']+self.vi[:,None]*m.ti*m.v_ie)/(1+lam*2.5)
        ii=(-self.ui[:,None]*m.ti*W['ii']+self.wi[:,None]*m.ti*m.v_ii)/(1+lam*2.5)
        return np.eye(2*n)-np.block([[ee,ei],[ie,ii]])

    def refine(self,guess):
        def fun(x):
            ev=eigvals(self.matrix(complex(*x)),check_finite=False)
            v=ev[np.argmin(abs(ev))]
            return [v.real,v.imag]
        sol=root(fun,[guess.real,guess.imag],tol=1e-8)
        lam=complex(*sol.x);err=float(np.linalg.norm(fun(sol.x)))
        return lam,err

    def rhp_count(self,spacing=3.,omega=3000.):
        """Argument-principle count on the right half-plane, using conjugacy.

        A large-frequency bound excludes roots outside the integration circle.
        Repeat with a denser contour before accepting a stability label.
        """
        m=self.m;n=m.n;U=n*m.K;O=omega/1000
        rep=lambda a:np.repeat(a,m.K)
        ee=m.w_ee.sum(1);ei=m.w_ei.sum(1);ie=m.w_ie.sum(1);ii=m.w_ii.sum(1)
        bE=(m.te*abs(self.u)*(m.gaA*rep(ee)/(O*O*m.ra*m.ta)+abs(self.z)*m.gaG*rep(ei)/(O*O*m.rg*m.tg))+
            m.te*abs(self.v)*rep(m.v_ee.sum(1))+m.te*abs(self.w*self.z2)*rep(m.v_ei.sum(1)))/(O*5-1-abs(self.u)*m.eta_M/O)
        bI=(m.ti*abs(self.ui)*(m.gaA*ie/(O*O*m.ra*m.ta)+m.gaG*ii/(O*O*m.rg*m.tg))+
            m.ti*abs(self.vi)*m.v_ie.sum(1)+m.ti*abs(self.wi)*m.v_ii.sum(1))/(O*2.5-1)
        bound=float(max(((bE*m.w_u).reshape(n,m.K).sum(1)).max(),bI.max()))
        assert bound<1.,bound
        freqs=np.unique(np.r_[0,np.geomspace(.01,10,15),np.arange(10,202,spacing),np.geomspace(202,omega,24)])
        nodes=list(1j*freqs)+list(omega*np.exp(1j*np.linspace(np.pi/2,0,17))[1:])
        cache={}
        def val(z):
            if z not in cache:cache[z]=np.linalg.slogdet(self.matrix(z))
            return cache[z]
        def piece(a,b,depth=0):
            va,la=val(a);vb,lb=val(b);mid=(a+b)/2;vm,lm=val(mid)
            p1=np.angle(vm/va);p2=np.angle(vb/vm)
            if depth<12 and (max(abs(p1),abs(p2))>.35 or abs(lm-(la+lb)/2)>.15):
                left=piece(a,mid,depth+1);right=piece(mid,b,depth+1)
                return left[0]+right[0],max(left[1],right[1])
            return float(p1+p2),float(max(abs(p1),abs(p2)))
        phase=0.;largest=0.
        for a,b in zip(nodes[:-1],nodes[1:]):
            dp,large=piece(a,b);phase+=dp;largest=max(largest,large)
        count=-phase/np.pi
        return dict(unstable_root_count=int(round(count)),raw_count=count,outer_radius_per_s=omega,
                    exterior_coupling_bound=bound,contour_evaluations=len(cache),max_phase_step=largest,base_spacing_per_s=spacing)
