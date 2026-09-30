"""Analytic gains of the unchanged frozen quadrature, including tiny variances."""
from common import *
sys.path.append(str(ROOT/'scripts/topic4_core_network_bifurcation_v7'))
from analytic_gains import gains
from periodic import Orbit as BaseOrbit
from scipy.sparse.linalg import LinearOperator


class Orbit(BaseOrbit):
    def evaluate(self,y,ref,phase,derivative=False):
        s=self.s;r=y[:-1].reshape(self.N,6)*.01;T=np.exp(y[-1]);H,Hp,L,Lp=self.kernels(T)
        mu=s.ext_mu+self.mean(r,H);ve=s.ext_var+(r[:,:3]@self.Q[:,:3].T)*s.tm;vi=(r[:,3:]@self.Q[:,3:].T)*s.tm
        phi=self.phi(mu,ve,vi)
        residual=np.r_[((r-self.filt(phi,L))/.01).ravel(),np.sum((r-ref)*phase)/.01]
        if not derivative:return residual
        gg=gains(s,mu,ve,vi);u,v,h=gg
        colp=-(self.filt(u*self.mean(r,Hp),L)+self.filt(phi,Lp))/.01
        def mv(dy):
            dr=dy[:-1].reshape(self.N,6)*.01;dmu=self.mean(dr,H)
            dve=(dr[:,:3]@self.Q[:,:3].T)*s.tm;dvi=(dr[:,3:]@self.Q[:,3:].T)*s.tm
            jj=(dr-self.filt(u*dmu+v*dve+h*dvi,L))/.01+colp*dy[-1]
            return np.r_[jj.ravel(),np.sum(dr*phase)/.01]
        return residual,LinearOperator((len(y),len(y)),matvec=mv),dict(mu=mu,ve=ve,vi=vi,gains=gg)
