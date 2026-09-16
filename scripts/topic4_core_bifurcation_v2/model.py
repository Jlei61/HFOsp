"""Graph-projected colored-Siegert delay-rate closure; rates/ms, time/ms, voltage/mV.

The exact Jacobian belongs to these deterministic population equations, not to
the sample-path Jacobian of 40,000 spiking neurons. No fitting to burst labels.
"""
from pathlib import Path
import os,json
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import numpy as np
from scipy.special import erfcx
from scipy.optimize import root
OUT=Path('/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/core_burst_bifurcation_v2_20260915')

class System:
    def __init__(self,groups=32,tau_scale=1.,colored=True,threshold_rule='gaussian'):
        self.cfg=json.loads((OUT/'projection.json').read_text());p=self.cfg['params'];self.p=p;self.n=6
        with np.load(OUT/'projected_graph.npz') as z:
            self.W=z['W'];self.Q=z['Q'];self.count=z['count'];self.nu=z['nu'];self.mask=z['stochastic_fraction']
            thresholds=z['vtheta'];region=z['region']
        self.threshold=np.zeros((6,groups));self.tw=np.zeros((6,groups))
        for i in range(6):
            v=np.sort(thresholds[region==i])
            if threshold_rule=='equal_mass':
                chunks=np.array_split(v,min(groups,len(v)))
                for j,c in enumerate(chunks):self.threshold[i,j]=c.mean();self.tw[i,j]=len(c)/len(v)
                self.threshold[i,len(chunks):]=v[-1]
            else:
                # Gaussian quadrature for the discrete EMPIRICAL distribution.
                # Lanczos on diag(Vth), with full reorthogonalization, retains
                # the low-threshold tail much better than quantile-bin means.
                from scipy.linalg import eigh_tridiagonal
                if v.std()<1e-12:self.threshold[i]=v[0];self.tw[i,0]=1.;continue
                cen=v.mean();width=v.std();xx=(v-cen)/width
                q=np.ones(len(v))/np.sqrt(len(v));prev=np.zeros(len(v));beta=0.;basis=[];aa=[];bb=[]
                for k in range(min(groups,len(np.unique(v)))):
                    basis.append(q.copy());z=xx*q-beta*prev;alpha=q@z;z-=alpha*q
                    for _ in range(2):
                        for bq in basis:z-=bq*(bq@z)
                    bn=np.linalg.norm(z);aa.append(alpha)
                    if k==groups-1 or bn<1e-13:break
                    bb.append(bn);prev=q;q=z/bn;beta=bn
                nodes,vec=eigh_tridiagonal(np.array(aa),np.array(bb[:len(aa)-1]))
                self.threshold[i,:len(nodes)]=cen+width*nodes;self.tw[i,:len(nodes)]=vec[0]**2
                self.threshold[i,len(nodes):]=v[-1]
        self.tm=np.where(np.arange(6)<3,p['tau_m_E'],p['tau_m_I'])
        self.ref=np.where(np.arange(6)<3,p['tau_ref_E'],p['tau_ref_I'])
        self.tr=np.where(np.arange(6)<3,5.,2.5)*tau_scale
        self.rise=np.where(np.arange(6)<3,p['tau_r_AMPA'],p['tau_r_GABA'])
        self.decay=np.where(np.arange(6)<3,p['tau_d_AMPA'],p['tau_d_GABA'])
        self.dt=p['dt'];self.delay=(np.arange(len(self.W))+1)*self.dt
        self.area=self.dt/(self.rise*(1-np.exp(-self.dt/self.rise)))
        self.sign=np.where(np.arange(6)<3,1.,-1.)
        self.jext=np.where(np.arange(6)<3,p['J_ext_E'],p['J_ext_I'])
        self.ext_mu=self.tm*self.area[0]*self.jext*self.nu
        self.ext_var=self.tm*self.jext**2*self.nu*self.mask
        self.colored=colored
        self.x,self.qw=np.polynomial.legendre.leggauss(24)

    def weights(self,g):
        scale=np.ones((6,6));scale[0,0]=scale[1,1]=g
        return self.W*scale,self.Q.sum(0)*scale**2

    def phi(self,mu,ve,vi):
        variance=np.maximum(ve+vi,1e-14);sigma=np.sqrt(variance)
        shift=1.0325*np.sqrt(np.maximum((ve*(self.rise[0]+self.decay[0])+vi*(self.rise[-1]+self.decay[-1]))/self.tm,0)) if self.colored else 0.
        mean=mu-shift
        a=(self.p['V_reset']-mean[:,None])/sigma[:,None];b=(self.threshold-mean[:,None])/sigma[:,None]
        u=(a+b)[...,None]/2+(b-a)[...,None]/2*self.x
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            integral=(b-a)/2*np.sum(self.qw*erfcx(-u),axis=-1)
            rates=1/(self.ref[:,None]+self.tm[:,None]*np.sqrt(np.pi)*integral)
        rates=np.where(np.isfinite(rates),rates,0.)
        return np.sum(rates*self.tw,axis=1)

    def moments(self,r,g):
        W,Q=self.weights(g);W=W.sum(0)
        return (self.ext_mu+self.tm*(W@(self.area*self.sign*r)),
            self.ext_var+self.tm*(Q[:,:3]@r[:3]),self.tm*(Q[:,3:]@r[3:]))

    def gains(self,vals):
        ds=[]
        for k in range(3):
            step=1e-5*np.maximum(abs(vals[k]),1.)
            hi=[a.copy() for a in vals];lo=[a.copy() for a in vals]
            hi[k]+=step;lo[k]-=step
            ds.append((self.phi(*hi)-self.phi(*lo))/(2*step))
        return ds

    def F(self,r,g):return self.phi(*self.moments(r,g))-r

    def blocks(self,r,g):
        W,Q=self.weights(g);u,v,h=self.gains(self.moments(r,g))
        mean=self.tm[:,None]*W.sum(0)*(self.area*self.sign)[None,:]
        var=self.tm[:,None]*Q*np.c_[np.tile(v[:,None],(1,3)),np.tile(h[:,None],(1,3))]
        return u,mean,var

    def jac(self,r,g):
        u,W,Q=self.blocks(r,g);return u[:,None]*W+Q-np.eye(6)

    def solve(self,g,initial):
        sol=root(lambda r:self.F(r,g),initial,jac=lambda r:self.jac(r,g),tol=1e-11)
        err=float(abs(self.F(sol.x,g)).max())
        return sol.x,err, bool(err<1e-9 and sol.x.min()>-1e-10 and np.all(sol.x<1/self.ref+1e-9))

    def characteristic(self,lam,r,g,pre=None):
        # Exact physical delays and two synaptic poles; native DC area held fixed.
        u,_,var=self.blocks(r,g) if pre is None else pre
        W,_=self.weights(g)
        filt=self.area*self.sign/((1+lam*self.rise)*(1+lam*self.decay))
        coupling=self.tm[:,None]*np.einsum('d,dij->ij',np.exp(-lam*self.delay),W)*filt[None,:]
        return np.diag(1+lam*self.tr)-var-u[:,None]*coupling

if __name__=='__main__':
    from scipy.linalg import eigvals
    s=System();r=np.array([.0005,.0005,.00001,.0002,.0002,.00001]);rows=[]
    for g in np.arange(.2,1.501,.025):
        rr,err,ok=s.solve(float(g),r)
        if not ok:print('FAILED',g,err,rr);break
        r=rr;rows.append(dict(g=float(g),r_hz=(r*1000).tolist(),static_leading=float(eigvals(s.jac(r,g)).real.max()),residual=err))
    (OUT/'initial_branch.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows[::4],indent=2))
