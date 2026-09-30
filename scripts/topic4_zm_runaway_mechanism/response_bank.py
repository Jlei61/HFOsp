"""Exploratory finite-dimensional response bank, separate from frozen v3.

Five stable filters for each physical input moment. The static LIF curve is
unchanged. Dimensionless coefficients multiply normalized moment differences:
mu_eff = mu + sum b_j (mu-mu_j) + sum e_j (vE-vE_j)/gap
                                      + sum i_j (vI-vI_j)/gap.
Coefficients are read at the existing filtered operating state, not fitted
against network trajectories. This is not an accepted network replacement.
"""
from common import *
from scipy.interpolate import make_interp_spline
from transfer_spline import cox_de_boor

DEST=OUT/'response_bank_candidate'


class BankTable:
    def __init__(self,x,sE,sI,values):
        self.x=np.array(x);self.u=np.arcsinh(x);self.sEmax=max(sE);self.sImax=max(sI)
        self.sE=np.r_[-np.asarray(sE)[::-1],sE] if sE[0]>0 else np.r_[-np.asarray(sE)[:0:-1],sE]
        self.sI=np.r_[-np.asarray(sI)[::-1],sI] if sI[0]>0 else np.r_[-np.asarray(sI)[:0:-1],sI]
        v=np.concatenate([values[:,:,::-1,:],values],axis=2) if sE[0]>0 else np.concatenate([values[:,:,:0:-1,:],values],axis=2)
        v=np.concatenate([v[:,:,:,::-1],v],axis=3) if sI[0]>0 else np.concatenate([v[:,:,:,:0:-1],v],axis=3)
        coefficients=[]
        for field in v:
            c=field.copy()
            for axis,grid in enumerate([self.u,self.sE,self.sI]):
                c=np.moveaxis(c,axis,0);spl=make_interp_spline(grid,c.reshape(len(grid),-1),k=3)
                c=spl.c.reshape(c.shape);c=np.moveaxis(c,0,axis)
                setattr(self,['tu','tE','tI'][axis],spl.t)
            coefficients.append(c)
        self.c=np.asarray(coefficients)

    def evaluate(self,mu,ve,vi,theta):
        mu,ve,vi,theta=np.broadcast_arrays(mu,ve,vi,theta);shape=mu.shape
        sc=theta.ravel()-11.;assert np.all(sc>0)
        u=np.clip(np.arcsinh((mu.ravel()-11)/sc),self.u[0],self.u[-1])
        e=np.minimum(np.sqrt(np.maximum(ve.ravel(),0))/sc,self.sEmax)
        h=np.minimum(np.sqrt(np.maximum(vi.ravel(),0))/sc,self.sImax)
        out=np.empty((15,mu.size))
        for start in range(0,mu.size,2048):
            end=min(mu.size,start+2048);sl=slice(start,end)
            iu,bu,_,_=cox_de_boor(self.tu,3,u[sl]);ie,be,_,_=cox_de_boor(self.tE,3,e[sl]);ii,bi,_,_=cox_de_boor(self.tI,3,h[sl])
            a=iu[:,None]-3+np.arange(4);b=ie[:,None]-3+np.arange(4);d=ii[:,None]-3+np.arange(4)
            co=self.c[:,a[:,:,None,None],b[:,None,:,None],d[:,None,None,:]]
            out[:,sl]=np.einsum('pnijk,ni,nj,nk->pn',co,bu,be,bi)
        return out.reshape((3,5)+shape)


def tables():
    z=np.load(DEST/'coefficients.npz')
    return {pop:BankTable(z['x_'+pop],z['sE_'+pop],z['sI_'+pop],z['values_'+pop]) for pop in 'EI'}
