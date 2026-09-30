"""Independent source/target, delay, transfer and local-flow checks."""
from common import *
from network import SpatialRate
from rate_unit import RateTransfer
import cupy as cp
from scipy import sparse


def main():
    rows=[];rng=np.random.default_rng(230991)
    for kind in ('linear','nonlinear'):
        m=SpatialRate(kind);history=rng.uniform(0,.1,size=(m.depth,800));m.history.set(history);tick=431
        delay_error=[]
        for name,op,out in zip(('ampa','gaba'),m.ops,m.arrive):
            m.module.get_function('delayed')((800,),(128,),(*op,m.history,out,np.int32(tick),np.int32(m.depth)))
            matrix=sparse.load_npz(OUT/f'operators/{name}_delay.npz')
            values=history[(tick-np.arange(m.depth))%m.depth].ravel()
            delay_error.append(float(np.max(abs(matrix@values-cp.asnumpy(out)))))
        # A stationary rate state must remain exactly stationary under constant
        # current. This also checks CUDA PCHIP coefficient and theta indexing.
        u=rng.uniform(-30,200,size=800);tr=RateTransfer();F=[];CV=[]
        for c,theta,pop in zip(m.geo['cell'],m.geo['theta'],m.geo['population']):
            f,cv=tr.row(u[c],theta,pop);F.append(float(f));CV.append(float(cv))
        F=np.array(F);CV=np.array(CV);pop=m.geo['population'];tm=np.where(pop==0,20.,10.)
        alpha=m.parameters[2*pop]*F*(CV/.22)**2+m.parameters[2*pop+1]*np.exp(-F*tm)/tm
        m.r.set(F);m.aux.set(-alpha/2 if kind=='nonlinear' else np.zeros(m.P));m.ia.set(u)
        m.module.get_function('rates')(((m.P+127)//128,),(128,),(m.r,m.aux,m.z,m.m,m.output,m.cell,m.theta,
            np.int32(m.NE),np.int32(m.P),m.ia,m.ig,m.xs,np.int32(len(m.xs)),m.fs,m.cvs,m.pars,DT,np.int32(kind=='nonlinear'),np.int32(0),m.ith,5.,-1.,0.))
        err=float(np.max(abs(cp.asnumpy(m.r)-F)))
        q=dict(model=kind,maximum_delay_error_mv_per_ms=max(delay_error),stationary_rate_error_per_ms=err)
        assert max(delay_error)<1e-9 and err<1e-12,q
        q['pass']=True;rows.append(q)
    write(OUT/'network_operator_qa.json',dict(status='PASS',rows=rows,scope='Numerical and source-operator checks, not model dynamics acceptance'))
    print(rows)


if __name__=='__main__':main()
