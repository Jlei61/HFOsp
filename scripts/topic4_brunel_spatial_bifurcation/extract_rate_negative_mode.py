"""Propagate a checked negative Floquet vector into an antiperiodic BVP seed.

Small successive graphs avoid allocating an additional full-period CUDA
graph. The local states and physical delay history are carried continuously.
The extracted profile is only an initial guess for a later -1 root solve.
"""
from rate_floquet import *
from scipy.interpolate import CubicSpline


def main():
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--label',required=True)
    p.add_argument('--samples',type=int,default=512);p.add_argument('--dt',type=float,default=.05)
    p.add_argument('--device',type=int,default=0);a=p.parse_args()
    source=Path(a.source);q=read(source);z=np.load(source.with_suffix('.npz'))
    vals=np.array([complex(*v) for v in q['multipliers']]);ids=np.flatnonzero((vals.real<-1)&(abs(vals.imag)<1e-8))
    assert len(ids);index=ids[np.argmax(abs(vals[ids]))];mu=float(vals[index].real)
    assert q['residuals'][index]/abs(mu)<1e-6
    s=RateField();N=a.samples;n=int(np.ceil(q['T_ms']/a.dt/N))*N
    dtmax=q['T_ms']/n*(1+1e-12)
    m=Monodromy(s,q['orbit'],dtmax,a.device,capture=False);assert m.n==n and n%N==0
    local=z['local_vectors'][:,index];history=z['rate_history_vectors'][:,index].reshape(-1,s.P)
    assert np.linalg.norm(local.imag)+np.linalg.norm(history.imag)<1e-10
    old=-np.arange(1,len(history)+1)*float(z['dt']);new=-np.arange(1,m.D+1)*m.dt
    h=CubicSpline(old[::-1],history.real[::-1],axis=0,extrapolate=True)(new)
    initial=np.r_[local.real,h.ravel()];v=initial.copy();samples=[];stride=n//N
    status=PERIODIC_OUT/(a.label+'_worker.json')
    for j in range(N):
        y=v[:9*s.P].reshape(9,s.P);samples.append(s.alpha*y[0]+(1-s.alpha)*y[1])
        v=m.interval_matvec(v,j*stride,(j+1)*stride)
        if j%64==0:
            write(status,dict(pid=os.getpid(),status='PROPAGATING_FULL_STATE',sample=j,total=N))
            print('NEGATIVE MODE SAMPLE',j,N,flush=True)
    growth=np.log(abs(mu))/m.T;t=np.arange(N)*m.T/N
    u=np.array(samples)*np.exp(-growth*t[:,None]);u/=np.linalg.norm(u)
    defect=float(np.linalg.norm(v-mu*initial)/np.linalg.norm(mu*initial))
    assert defect<.02,defect
    path=PERIODIC_OUT/(a.label+'.npz');save_periodic_array(path,u=u,J=m.J,T=m.T)
    out=dict(status='ANTIPERIODIC_SEED_ONLY',source=str(source),source_index=int(index),
        source_multiplier=mu,samples=N,dt_ms=m.dt,J_EE_core=m.J,T_ms=m.T,
        full_state_endpoint_mode_relative_defect=defect,seed=str(path),
        scope='Carried full-state tangent trajectory with exp(log|mu| t/T) removed. This is a seed, not a -1 root or a new PD certificate.')
    write(PERIODIC_OUT/(a.label+'.json'),out);write(status,dict(pid=os.getpid(),status='COMPLETE',result=out));print(out,flush=True)


if __name__=='__main__':main()
