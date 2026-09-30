"""Check equilibrium/Jacobian tractability, without claiming a bifurcation.

Static equations eliminate the finite auxiliary, synaptic and M states
algebraically. Their dynamics are retained in the actual candidate simulator.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[name]='1'
from common import *
from rate_unit import RateTransfer
from scipy import sparse
from scipy.optimize import root
import time


class Equilibrium:
    def __init__(self,D=0.):
        self.geo=dict(np.load(OUT/'operators/groups.npz'));self.P=len(self.geo['cell']);self.NE=int(self.geo['nE'])
        self.transfer=RateTransfer();self.D=D;self.prep=read(OUT/'operators/params.json')
        cell=self.geo['cell'];p=self.prep['params'];pop=self.geo['population']
        sums=[]
        for name in ('ampa','gaba'):
            o=sparse.load_npz(OUT/f'operators/{name}_delay.npz').tocoo()
            sums.append(sparse.coo_matrix((o.data,(o.row,o.col%800)),shape=(800,800)).tocsr())
        R=sparse.coo_matrix((self.geo['weight'],(cell,np.arange(self.P))),shape=(800,self.P)).tocsr()
        tm=np.where(pop==0,20.,10.);dca=tm/.7*.1/(1-np.exp(-.1/.7));dcg=tm*.1/(1-np.exp(-.1))
        # D=0 preflight only: no spatial-path approximation is introduced here.
        assert D==0.
        self.A=(sparse.diags(dca)@sums[0][cell]@R-sparse.diags(dcg)@sums[1][cell]@R).toarray()
        self.A[np.arange(self.NE),np.arange(self.NE)]-=.5 # eta_M*tau_M; rates are per ms
        theta=self.geo['theta'];hi=np.clip(np.searchsorted(self.transfer.theta,theta),1,7);lo=hi-1
        self.mix=(theta-self.transfer.theta[lo])/(self.transfer.theta[hi]-self.transfer.theta[lo])
        self.lo=np.where(pop==0,lo,8);self.hi=np.where(pop==0,hi,8);self.mix=np.where(pop==0,self.mix,0.)

    def evaluate(self,r,jac=False):
        raw=self.A@r;x=np.clip(raw,self.transfer.x[0],self.transfer.x[-1]);index=np.arange(self.P)
        values=self.transfer.f(x);F=(1-self.mix)*values[self.lo,index]+self.mix*values[self.hi,index]
        f=F-r
        if not jac:return f
        der=self.transfer.f.derivative()(x)
        gain=(1-self.mix)*der[self.lo,index]+self.mix*der[self.hi,index]
        gain[(raw<self.transfer.x[0])|(raw>self.transfer.x[-1])]=0.
        return gain[:,None]*self.A-np.eye(self.P)


def main():
    folder=OUT/'analysis_preflight';folder.mkdir(exist_ok=True);model=Equilibrium();rng=np.random.default_rng(2309)
    r=np.full(model.P,.0001);v=rng.normal(size=model.P);v/=np.linalg.norm(v);eps=1e-8
    actual=(model.evaluate(r+eps*v)-model.evaluate(r-eps*v))/(2*eps);pred=model.evaluate(r,True)@v
    derivative_error=float(np.linalg.norm(actual-pred)/np.linalg.norm(actual));assert derivative_error<1e-5
    started=time.time();result=root(model.evaluate,r,jac=lambda r:model.evaluate(r,True),method='hybr',options={'maxfev':180,'xtol':1e-10})
    residual=float(np.max(abs(model.evaluate(result.x)))*1000)
    report=dict(status='PREFLIGHT_ROOT_FOUND' if residual<1e-5 and result.x.min()>-1e-9 else 'PREFLIGHT_ROOT_NOT_CONVERGED',
        D=0.,rate_unknowns=model.P,continuous_model_states=read(OUT/'operators/definition.json')['total_continuous_states'],
        analytic_jacobian_directional_relative_error=derivative_error,wall_s=time.time()-started,
        maximum_rate_residual_hz=residual,minimum_rate_hz=float(result.x.min()*1000),
        scipy_success=bool(result.success),message=str(result.message),evaluations=result.nfev,
        scientific_scope='Rate-model analysis capability only; no native correspondence, stability, or critical type claimed')
    write(folder/'equilibrium.json',report);np.savez_compressed(folder/'equilibrium.npz',rate_per_ms=result.x)
    print(report)


if __name__=='__main__':main()
