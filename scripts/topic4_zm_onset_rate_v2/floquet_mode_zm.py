"""Convert a full monodromy eigenvector into a periodic rate eigenfunction.

This enables independent Fourier BVP refinement and spatial attribution over
the entire cycle rather than at an arbitrary single phase of the burst.
"""
from floquet_zm import *
from spectral_floquet_zm import SpectralFloquet


def main():
    p=argparse.ArgumentParser();p.add_argument('orbit');p.add_argument('floquet')
    p.add_argument('--index',type=int,required=True);p.add_argument('--N',type=int,default=256)
    p.add_argument('--dt',type=float,default=.1);p.add_argument('--device',type=int,default=0);p.add_argument('--label',required=True)
    a=p.parse_args();s=ZMSpatialRate();m=Monodromy(s,a.orbit,a.dt,a.device);z=np.load(a.floquet);cp=m.cp
    mu=complex(z['multipliers'][a.index]);lam=np.log(mu)/m.T
    assert abs(float(z['dt'])-m.dt)<1e-12
    v=np.r_[z['local_vectors'][:,a.index],z['rate_history_vectors'][:,a.index]]
    pieces=[]
    for part in [v.real,v.imag]:
        x=cp.asarray(part);m.y[:]=x[:9*s.P].reshape(9,s.P);m.hist[m.order]=x[9*s.P:].reshape(m.D,s.P)
        m.hist[0]=m.pars[3]*m.y[0]+(1-m.pars[3])*m.y[1]
        series=cp.empty((m.n,s.P))
        for j in range(m.n):
            series[j]=m.pars[3]*m.y[0]+(1-m.pars[3])*m.y[1];m.step(j)
        pieces.append(series.get())
    rate=pieces[0]+1j*pieces[1]
    periodic=rate*np.exp(-lam*np.arange(m.n)[:,None]*m.dt)
    u=resample(periodic,a.N,axis=0);u/=abs(u).max()
    del m;cp.get_default_memory_pool().free_all_blocks()
    solver=SpectralFloquet(s,a.orbit,a.N,a.device);lam,u,history=solver.refine(lam,u)
    energy=s.sizes*np.mean(abs(u)**2,axis=0);energy/=energy.sum();reg=s.geo['group_region']
    out=PERIODIC_OUT/'spectral_floquet';out.mkdir(exist_ok=True)
    row=dict(orbit=a.orbit,D=solver.depletion,T_ms=solver.T,N=a.N,lambda_per_ms=lam,
             multiplier=np.exp(lam*solver.T),residual=history[-1],status='CONVERGED' if history[-1]<2e-9 else 'NOT_CONVERGED',
             E_energy=[float(energy[s.E&(reg==k)].sum()) for k in range(3)],I_energy=float(energy[~s.E].sum()),
             monodromy_multiplier=mu,monodromy_source=a.floquet,monodromy_index=a.index,
             spatial_energy='Cell-count-weighted squared periodic rate eigenfunction, averaged over the whole cycle')
    write(out/f'{a.label}_N{a.N}.json',row)
    np.savez_compressed(out/f'{a.label}_N{a.N}.npz',u=u,lam=lam,D=solver.depletion,T=solver.T)
    print('MODE',row,flush=True)


if __name__=='__main__':main()
