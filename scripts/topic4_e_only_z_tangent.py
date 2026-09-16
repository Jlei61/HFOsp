"""Full delayed fast-rate tangent for a frozen E-only Z field."""
from check_topic4_delayed_rate_spectrum import Tangent as GlobalTangent
from scipy.linalg import lu_factor,lu_solve
from scipy.signal import lfilter
from scipy.sparse.linalg import LinearOperator,eigs
import numpy as np


class EOnlyTangent(GlobalTangent):
    def drive(self,h):
        o=self.s.ops;q=self.s.qvec(self.q)
        return np.array([o['ee']@h[0].ravel(),q*(o['ei']@h[1].ravel()),o['ie']@h[0].ravel(),o['ii']@h[1].ravel()])

    def inverse(self,z):
        lu=lu_factor(self.s.characteristic(z,self.r,self.q));n=self.n;D=self.D;q=self.s.qvec(self.q)
        weights=z**(-np.arange(1,D+1));Wr={k:(v@weights).reshape(n,n) for k,v in self.s.block_ops.items()}
        def solve(b):
            br,bg,bc,bh=self.unpack(b)
            hc=lfilter([1/z],[1,-1/z],bh,axis=1)
            dc=self.drive(hc);gc=(bg+self.B*dc)/(z-self.ar)
            cc=(bc+(1-self.ad)*(z*gc-bg))/(z-self.ad)
            r=lu_solve(lu,br/self.alpha+self.u*self.signed(z*cc-bc))
            h=hc+weights[None,:,None]*r.reshape(2,1,n)
            dr=np.array([Wr['ee']@r[:n],q*(Wr['ei']@r[n:]),Wr['ie']@r[:n],Wr['ii']@r[n:]])
            g=gc+self.B*dr/(z-self.ar);c=(bc+(1-self.ad)*(z*g-bg))/(z-self.ad)
            return np.r_[r,g.ravel(),c.ravel(),h.ravel()]
        return LinearOperator((self.size,self.size),matvec=solve,dtype=complex)


def sample_spectrum(s,r,q,frequencies=(0,4,20,50),k=4):
    tangent=EOnlyTangent(s,r,q);v0=np.random.default_rng(46).normal(size=tangent.size)
    z=1.001+.003j;x=tangent.inverse(z)@v0
    err=np.linalg.norm(z*x-tangent.action(x)-v0)/np.linalg.norm(v0);assert err<1e-8,err
    rows=[]
    for f in frequencies:
        shift=np.exp((.00003+2j*np.pi*f/1000)*s.dt)
        values,vectors=eigs(tangent.inverse(shift),k=k,which='LM',tol=2e-9,maxiter=700,ncv=24,v0=v0.astype(complex))
        for j,mu in enumerate(values):
            multiplier=shift-1/mu;lam=np.log(multiplier)/s.dt*1000
            residual=np.linalg.norm(tangent.action(vectors[:,j])-multiplier*vectors[:,j])/np.linalg.norm(vectors[:,j])
            if all(abs(complex(row['real_per_s'],row['imag_per_s'])-lam)>1e-3 for row in rows):
                rows.append({'real_per_s':float(lam.real),'imag_per_s':float(lam.imag),
                             'frequency_hz':float(abs(lam.imag)/2/np.pi),'full_map_residual':float(residual)})
    rows.sort(key=lambda row:row['real_per_s'],reverse=True)
    return {'dimension':tangent.size,'inverse_identity_error':float(err),'roots':rows,
            'scope':'Frozen E-only Z fast rate subsystem with native delays; not native spiking SNN Jacobian or autonomous slow-Z spectrum',
            'coverage':'Shift-targeted eigenvalues; a full unstable-root count requires an independent contour test'}
