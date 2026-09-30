"""Temporal characteristic of the current FULL-diffusion expected-rate model.

Z is a prescribed full spatial field; M remains dynamic. A constant mean
external drive defines the autonomous object. Discrete mode matches the
actual endpoint-arrival, implicit refractory and old-M update convention.
Neither this interface nor a root of it validates the native correspondence.
"""
from current_rate_equilibrium import CurrentRateEquilibrium, np, DT0
from refractory_rate_response import covariance_matrices, TAUS
from scipy import sparse


def local_factors(pop, lam, dt=None):
    """Input-history bank, covariance response and refractory integral."""
    ref=2. if pop=='E' else 1.
    bank=np.empty(12,complex);cov=np.ones(3,complex)
    A,B,C,G,Q=covariance_matrices(pop,.05 if dt is None else dt)
    if dt is None:
        for j,tau in enumerate(TAUS):bank[3*j:3*j+3]=(1+lam*tau)**(-np.arange(1,4))-1
        for ch in range(2):cov[ch+1]=C[ch]*np.linalg.solve(lam*np.eye(3)-G[ch],Q[ch])[2]
        K=ref if lam==0 else -np.expm1(-lam*ref)/lam
    else:
        z=np.exp(-lam*dt)
        for j,tau in enumerate(TAUS):
            b=dt/tau;e=np.exp(-b)
            L=e*np.array([[1,0,0],[b,1,0],[.5*b*b,b,1.]])
            F=np.array([1-e,1-e*(1+b),1-e*(1+b+.5*b*b)])
            bank[3*j:3*j+3]=np.linalg.solve(np.eye(3)-L*z,F)-1
        for ch in range(2):cov[ch+1]=C[ch]*np.linalg.solve(np.eye(3)-A[ch]*z,B[ch])[2]
        assert abs(round(ref/dt)*dt-ref)<1e-12
        K=dt*np.exp(-lam*dt*np.arange(round(ref/dt))).sum()
    return bank,cov,K


class CurrentRateCharacteristic(CurrentRateEquilibrium):
    def local_gain(self, operating, lam, dt=None):
        out=np.empty((self.P,3),complex)
        for pop,mask in [('E',self.E),('I',~self.E)]:
            bank,cov,K=local_factors(pop,lam,dt)
            grad=operating['feature_gradient'][mask]
            ugrad=operating['input_normalization_gradient'][mask]
            loggain=operating['base_gradient'][mask]+ugrad*(grad[:,:3]+grad[:,3:].reshape(-1,3,12)@bank)
            # r/(1+rho*K); reciprocal hazard avoids overflow near saturation.
            invrho=DT0*np.exp(-operating['log_hazard'][mask])
            factor=operating['rate'][mask]*invrho/(invrho+K)
            out[mask]=factor[:,None]*loggain*cov
        return out

    def temporal_components(self, lam, dt=None):
        if dt is None:
            H=1/((1+lam*self.rise)*(1+lam*self.decay))
            M=.5*self.E/(1+1000*lam)
        else:
            z=np.exp(-lam*dt);H=[]
            for tr,td in zip(self.rise,self.decay):
                a=np.exp(-dt/tr);d=np.exp(-dt/td);b=tr/(tr-td)*(a-d)
                H.append(((1-d-b)+b*z*(1-a)/(1-a*z))/(1-d*z))
            H=np.array(H)
            e=np.exp(-dt/1000)
            # The physical step reads M[k-1], finish updates M[k] from r[k].
            M=.5*self.E*z*(1-e)/(1-e*z)
        a,b,qa,qb=self.matrices(lam)
        U=sparse.diags(self.tm)@(self.area[0]*H[0]*a-sparse.diags(self.Z*self.area[1]*H[1])@b)-sparse.diags(M)
        VE=sparse.diags(self.tm*self.area[0]**2)@qa
        VI=sparse.diags(self.Z**2*self.tm*self.area[1]**2)@qb
        return U,VE,VI

    def characteristic(self, r, lam=0., dt=None, operating=None):
        op=self.local_operating(*self.moments(r)) if operating is None else operating
        chi=self.local_gain(op,lam,dt)
        feedback=sum(sparse.diags(chi[:,ch])@v for ch,v in enumerate(self.temporal_components(lam,dt)))
        return (sparse.eye(self.P,format='csc')-feedback).tocsc()
