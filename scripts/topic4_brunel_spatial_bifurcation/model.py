"""Spatial extension of Bachschmid-Romano et al. 2026 Eqs. 13--30.

Same realized graph, physical delays, empirical threshold subgroups, and
external Poisson mean as topology 6101. The two low-threshold cores invalidate
the homogeneous Fourier reduction; all rate groups remain in the operator.

Units: ms, mV, rate/ms. Z is prescribed at the interictal baseline 1; M retains
its original activity feedback. Common external OU modulation is set to its
mean when defining the autonomous stationary branch, NOT private variance.
"""
from common import *
from scipy.special import erfcx
from scipy.sparse.linalg import spsolve
from functools import lru_cache

class SpatialBrunel:
    def __init__(self,grid=20,quadrature=48,area_correction=True,response='shifted_white'):
        self.folder=OUT/f'operators/g{grid}';self.prep=read(self.folder/'prepared.json')
        self.geo=dict(np.load(self.folder/'geometry.npz'));self.p=self.prep['params'];self.grid=grid
        g=self.geo;p=self.p;self.P=len(g['group_size']);self.E=g['population']==0
        self.tm=np.where(self.E,p['tau_m_E'],p['tau_m_I']);self.ref=np.where(self.E,p['tau_ref_E'],p['tau_ref_I'])
        self.theta=g['threshold_mv'];self.Z=np.ones(self.P);self.nu=self.prep['nu_ext_per_ms']
        self.response_method=response
        self.response_fit=read(OUT/'local_response_fit/result.json')['rows'] if response.startswith('calibrated') else None
        self.variance_fit=read(OUT/'local_response_fit/variance_result.json')['rows'] if response=='calibrated_full' else None
        self.jext=np.where(self.E,p['J_ext_E'],p['J_ext_I']);self.rise=np.array([p['tau_r_AMPA'],p['tau_r_GABA']])
        self.decay=np.array([p['tau_d_AMPA'],p['tau_d_GABA']]);self.tau=self.rise+self.decay
        self.area=.1/(self.rise*(1-np.exp(-.1/self.rise))) if area_correction else np.ones(2)
        self.nodes,self.weights=np.polynomial.legendre.leggauss(quadrature)
        self.raw=[]
        for name in ('mean_ampa','mean_gaba','variance_ampa','variance_gaba'):
            a=sparse.load_npz(self.folder/f'{name}.npz').tocoo();row=a.row;col=a.col%self.P;di=a.col//self.P
            core=self.E[row]&self.E[col]&(g['group_region'][row]<2)&(g['group_region'][row]==g['group_region'][col])
            # Cache one row/column edge with its full distribution of delays.
            keys,inv=np.unique(row.astype(np.int64)*self.P+col,return_inverse=True)
            delay_matrix=sparse.coo_matrix((a.data,(inv,di)),shape=(len(keys),self.prep['max_delay_steps'])).tocsr()
            r=keys//self.P;c=keys%self.P
            mask=self.E[r]&self.E[c]&(g['group_region'][r]<2)&(g['group_region'][r]==g['group_region'][c])
            self.raw.append((r,c,mask,delay_matrix))
        self.delays=(np.arange(self.prep['max_delay_steps'])+1)*.1

    @lru_cache(maxsize=6)
    def matrices(self,J,lam=0.):
        result=[];phase=np.exp(-lam*self.delays)
        for k,(r,c,mask,delay) in enumerate(self.raw):
            values=delay@phase
            if k in (0,2): values=values*np.where(mask,J**(1 if k==0 else 2),1.)
            result.append(sparse.coo_matrix((values,(r,c)),shape=(self.P,self.P)).tocsr())
        return result

    def moments(self,r,J):
        a,b,qa,qb=self.matrices(J)
        mu=self.tm*(self.area[0]*(a@r+self.jext*self.nu)-self.Z*self.area[1]*(b@r))-.5*self.E*r
        ve=self.tm*self.area[0]**2*(qa@r+self.jext**2*self.nu)
        vi=self.tm*(self.Z*self.area[1])**2*(qb@r)
        return mu,ve,vi

    def phi(self,mu,ve,vi,details=False):
        q=np.maximum(ve+vi,1e-12);sigma=np.sqrt(q)
        # Eq.18 is a variance-weighted HARMONIC effective correlation time.
        teff=q/np.maximum(ve/self.tau[0]+vi/self.tau[1],1e-12)
        shift=1.0325*np.sqrt(teff/self.tm)
        low=(self.p['V_reset']-mu)/sigma+shift;high=(self.theta-mu)/sigma+shift
        x=(low+high)[:,None]/2+(high-low)[:,None]/2*self.nodes
        with np.errstate(over='ignore'):
            integ=(high-low)/2*np.sum(self.weights*erfcx(-x),axis=1)
            rate=1/(self.ref+self.tm*np.sqrt(np.pi)*integ)
        if details:return rate,low,high,sigma,teff
        return rate

    def gains(self,moments):
        ds=[]
        for k in range(3):
            step=1e-5*np.maximum(abs(moments[k]),1.)
            hi=list(moments);lo=list(moments);hi[k]=hi[k]+step;lo[k]=lo[k]-step
            ds.append((self.phi(*hi)-self.phi(*lo))/(2*step))
        return ds

    def residual(self,r,J): return self.phi(*self.moments(r,J))-r

    def jacobian(self,r,J):
        a,b,qa,qb=self.matrices(J);gm,ge,gi=self.gains(self.moments(r,J))
        K=sparse.diags(gm*self.tm)@(self.area[0]*a-sparse.diags(self.Z*self.area[1])@b)
        K+=sparse.diags(ge*self.tm*self.area[0]**2)@qa+sparse.diags(gi*self.tm*(self.Z*self.area[1])**2)@qb
        return K-sparse.diags(1+.5*gm*self.E)

    def parameter_derivative(self,r,J):
        h=1e-5*max(1,abs(J));return (self.residual(r,J+h)-self.residual(r,J-h))/(2*h)

    def solve(self,J,r=None,tol=1e-11):
        if r is None:r=self.phi(*self.moments(np.zeros(self.P),J))
        r=np.maximum(r.copy(),0.);trace=[]
        for iteration in range(60):
            f=self.residual(r,J);error=float(abs(f).max());trace.append(error)
            if error<tol:return r,True,trace
            step=spsolve(self.jacobian(r,J).tocsc(),-f);alpha=1.
            for back in range(30):
                trial=r+alpha*step
                if trial.min()>=-tol and np.all(trial<1/self.ref):
                    trial=np.maximum(trial,0.)
                    if abs(self.residual(trial,J)).max()<error:
                        r=trial;break
                alpha*=.5
            else:return r,False,trace
        return r,False,trace

    def regional_rates(self,r):
        region=self.geo['group_region'];size=self.geo['group_size']
        return [float(np.average(r[self.E&(region==i)],weights=size[self.E&(region==i)])*1000) for i in range(3)]
