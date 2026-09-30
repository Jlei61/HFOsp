"""Continuous spatial rate DDE with explicit stationary and tangent equations.

Rates are spikes/cell/ms; currents mV; time ms. Private Poisson variability is
integrated in the empirical transfer. No member ages, voltages or sampled
spikes are evolved. Z is a fixed interictal field for autonomous continuation.
"""
from common import *
from scipy import sparse
from scipy.optimize import root
from numba import njit


@njit(cache=True)
def transfer_eval(drive,theta,nu,tk,nk,xk,coef,tref):
    value=np.zeros(len(drive));derivative=value.copy()
    for k in range(len(drive)):
        t=theta[k];n=nu[k];d=drive[k];a=0;b=0
        while a<len(tk)-2 and t>=tk[a+1]:a+=1
        while b<len(nk)-2 and n>=nk[b+1]:b+=1
        ft=min(1.,max(0.,(t-tk[a])/(tk[a+1]-tk[a]))) if len(tk)>1 else 0.
        fn=min(1.,max(0.,(n-nk[b])/(nk[b+1]-nk[b])))
        x=np.arcsinh(d/8);j=0
        while j<len(xk)-2 and x>=xk[j+1]:j+=1
        f=min(1.,max(0.,(x-xk[j])/(xk[j+1]-xk[j])))
        v=0.;dv=0.
        for i in range(2 if len(tk)>1 else 1):
            for l in range(2):
                y=0.;dy=0.
                for z in range(6):dy=dy*f+y;y=y*f+coef[a+i,b+l,j,z]
                weight=((ft if i else 1-ft) if len(tk)>1 else 1.)*(fn if l else 1-fn)
                v+=weight*y;dv+=weight*dy
        r=1/(tref+np.exp(min(700.,v)));value[k]=r
        derivative[k]=-r*(1-tref*r)*dv/(xk[j+1]-xk[j])/np.sqrt(64+d*d) if x>=xk[0] and x<=xk[-1] else 0.
    return value,derivative


class RateSystem:
    def __init__(self,grid=20,tau_e=5.,tau_i=2.5,z_field=None):
        self.folder=BASE/f'operators/g{grid}';self.prep=read(self.folder/'prepared.json');self.geo=dict(np.load(self.folder/'geometry.npz'))
        g=self.geo;p=self.prep['params'];self.P=len(g['group_size']);self.p=p;self.grid=grid
        self.pop=g['population'];self.E=self.pop==0;self.tm=np.where(self.E,p['tau_m_E'],p['tau_m_I'])
        self.tref=np.where(self.E,p['tau_ref_E'],p['tau_ref_I']);self.tr=np.where(self.E,tau_e,tau_i)
        self.jext=np.where(self.E,p['J_ext_E'],p['J_ext_I']);self.nu0=self.prep['nu_ext_per_ms']
        self.rise=np.array([p['tau_r_AMPA'],p['tau_r_GABA']]);self.decay=np.array([p['tau_d_AMPA'],p['tau_d_GABA']])
        self.area=DT/(self.rise*(1-np.exp(-DT/self.rise)))
        self.Z=np.ones(self.P) if z_field is None else np.asarray(z_field,float)
        self.raw=[]
        for kind in ['ampa','gaba']:
            m=sparse.load_npz(self.folder/f'delay_{kind}.npz').tocoo()
            self.raw.append((m.row,m.col%self.P,m.col//self.P,m.data))
        rr,cc,dd,ww=self.raw[0]
        self.core_ee=self.E[rr]&self.E[cc]&(g['group_region'][rr]<2)&(g['group_region'][rr]==g['group_region'][cc])
        self.tables=[read(BASE/f'transfer/{pop}.json') for pop in ['E','I']]
        self.theta=g['threshold_mv'];self.delay=(np.arange(self.prep['max_delay_steps'])+1)*DT
        self.parameter='J_EE_core'

    def phi(self,drive,nu=None):
        nu=np.full(self.P,self.nu0) if nu is None else np.broadcast_to(nu,(self.P,))
        out=np.zeros(self.P);grad=out.copy()
        for p,tab in enumerate(self.tables):
            mask=self.pop==p
            out[mask],grad[mask]=transfer_eval(np.asarray(drive)[mask],self.theta[mask],nu[mask],
                np.array(tab['theta']),np.array(tab['nu']),np.array(tab['input_knots']),np.array(tab['coefficients']),tab['ref_ms'])
        return out,grad

    def coupling(self,J,lam=0.,derivative=False):
        out=[]
        for k,(row,col,di,w) in enumerate(self.raw):
            factor=np.where(self.core_ee,J,1.) if k==0 else 1.
            d=(di+1)*DT;val=w*factor*np.exp(-lam*d)
            if derivative:val*=-d
            out.append(sparse.coo_matrix((val,(row,col)),shape=(self.P,self.P)).tocsr())
        return out

    def stationary_drive(self,r,J):
        a,b=self.coupling(J)
        return self.tm*(self.area[0]*(a@r)-self.Z*self.area[1]*(b@r))-.5*self.E*r

    def equilibrium_residual(self,r,J):return self.phi(self.stationary_drive(r,J))[0]-r

    def equilibrium_jacobian(self,r,J):
        a,b=self.coupling(J);gain=self.phi(self.stationary_drive(r,J))[1]
        K=sparse.diags(self.tm)@(self.area[0]*a-sparse.diags(self.Z*self.area[1])@b)-sparse.diags(.5*self.E)
        return sparse.diags(gain)@K-sparse.eye(self.P)

    def equilibrium_state(self,r,J):
        a,b=self.coupling(J);qa=self.tm*self.area[0]*(a@r);qg=self.tm*self.area[1]*(b@r)
        qe=self.tm*self.area[0]*self.jext*self.nu0
        return np.array([r,qa,qa,qg,qg,qe,qe,1000*self.E*r])

    def rhs(self,state,delayed_ampa,delayed_gaba,nu=None):
        r,qa,ia,qg,ig,qe,ie,M=state
        nu=np.full(self.P,self.nu0) if nu is None else np.asarray(nu)
        private_mean=self.tm*self.area[0]*self.jext*nu
        f=self.phi(ia-self.Z*ig-.0005*M+ie-private_mean,nu)[0]
        return np.array([(f-r)/self.tr,(-qa+self.tm*self.area[0]*delayed_ampa)/self.rise[0],(qa-ia)/self.decay[0],
            (-qg+self.tm*self.area[1]*delayed_gaba)/self.rise[1],(qg-ig)/self.decay[1],
            (-qe+private_mean)/self.rise[0],(qe-ie)/self.decay[0],self.E*r-M/1000])

    def jvp(self,state,delta,delayed_a_delta,delayed_g_delta,nu=None):
        nu=np.full(self.P,self.nu0) if nu is None else np.asarray(nu)
        r,qa,ia,qg,ig,qe,ie,M=state;dr,da,dia,dg,dig,de,die,dm=delta
        gain=self.phi(ia-self.Z*ig-.0005*M+ie-self.tm*self.area[0]*self.jext*nu,nu)[1]
        return np.array([(gain*(dia-self.Z*dig-.0005*dm+die)-dr)/self.tr,
            (-da+self.tm*self.area[0]*delayed_a_delta)/self.rise[0],(da-dia)/self.decay[0],
            (-dg+self.tm*self.area[1]*delayed_g_delta)/self.rise[1],(dg-dig)/self.decay[1],
            -de/self.rise[0],(de-die)/self.decay[0],self.E*dr-dm/1000])

    def characteristic(self,lam,r,J,derivative=False):
        """Exact characteristic of these continuous DDEs, rates-only Schur form.

        The independent private-input filter poles are known negative real
        modes. Elimination is valid away from filter poles, which are checked
        separately rather than mistaken for nonlinear eigenvalues.
        """
        gain=self.phi(self.stationary_drive(r,J))[1];a,b=self.coupling(J,lam)
        fa=self.area[0]/((1+lam*self.rise[0])*(1+lam*self.decay[0]))
        fg=self.area[1]/((1+lam*self.rise[1])*(1+lam*self.decay[1]))
        if derivative:
            da,db=self.coupling(J,lam,True)
            dfa=-fa*(self.rise[0]/(1+lam*self.rise[0])+self.decay[0]/(1+lam*self.decay[0]))
            dfg=-fg*(self.rise[1]/(1+lam*self.rise[1])+self.decay[1]/(1+lam*self.decay[1]))
            coupling=sparse.diags(self.tm)@(dfa*a+fa*da-sparse.diags(self.Z)@(dfg*b+fg*db))
            diagonal=self.tr-gain*.5*self.E*1000/(1+1000*lam)**2
        else:
            coupling=sparse.diags(self.tm)@(fa*a-sparse.diags(self.Z*fg)@b)
            diagonal=1+lam*self.tr+gain*.5*self.E/(1+1000*lam)
        return sparse.diags(diagonal)-sparse.diags(gain)@coupling

