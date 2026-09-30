"""Independent full-state monodromy check of a real periodic Floquet mode."""
from rate_floquet import *
from rate_real_floquet_mode import RealFloquet
import gc


def harmonic_mode(f,u,growth):
    cp=f.cp;s=f.s;K=f.o.K;k=f.kernels(growth)
    cf=cp.fft.rfft(cp.asarray(u),axis=0)/f.N
    z=growth+2j*np.pi*cp.arange(K)[:,None]/f.T
    a,b,qa,qb=[(op@cf.ravel()).reshape(K,s.P) for op in k[0]]
    tm,ref,th,alpha,tf,ts,E=f.o.gpars;drive=cf/k[-2]
    xf=drive/(1+z*tf);xs=drive/(1+z*ts)
    qe=tm*s.area[0]*a/(1+z*s.rise[0]);ie=qe/(1+z*s.decay[0])
    qi=tm*s.area[1]*b/(1+z*s.rise[1]);ii=qi/(1+z*s.decay[1])
    ve=tm*s.area[0]**2*qa/(1+z*s.tau[0]/2)
    vi=tm*s.area[1]**2*qb/(1+z*s.tau[1]/2);m=.5*E*cf/(1+z*1000)
    factor=cp.full((K,1),2.);factor[0]=1;factor[-1]=1
    states=cp.stack([xf,xs,qe,ie,qi,ii,ve,vi,m])*factor[None,:,:]
    return dict(states=states,rate=cf*factor,z=z[:,0],cp=cp,alpha=alpha)


def at_time(data,D,dt,origin=0.):
    cp=data['cp'];z=data['z']
    local=cp.sum(data['states']*cp.exp(origin*z)[None,:,None],axis=1).real
    history=(cp.exp((origin-cp.arange(1,D+1)[:,None]*dt)*z[None,:])@data['rate']).real
    return cp.r_[local.ravel(),history.ravel()].get()


def recover_mode(f,u,growth,D,dt):
    data=harmonic_mode(f,u,growth);v=at_time(data,D,dt)
    local=v[:9*f.s.P].reshape(9,f.s.P);alpha=f.s.alpha
    error=float(np.max(abs(alpha*local[0]+(1-alpha)*local[1]-u[0])))
    assert error<1e-9,error
    return v,error


def main():
    p=argparse.ArgumentParser();p.add_argument('--label',required=True)
    p.add_argument('--N',type=int,required=True);p.add_argument('--dt',type=float,nargs='+',default=[.05,.025,.0125])
    p.add_argument('--device',type=int,default=0);a=p.parse_args();s=RateField()
    q=read(PERIODIC_OUT/f'{a.label}_N{a.N}.json')
    assert q['status']=='EIGENPAIR_CONVERGED_CHECKS_PENDING'
    z=np.load(PERIODIC_OUT/f'{a.label}_mode_N{a.N}.npz');u=z['u'];growth=float(z['lam'])
    for dtmax in a.dt:
        dt=q['T_ms']/int(np.ceil(q['T_ms']/dtmax));D=int(np.ceil(s.delays[-1]/dt))+1
        f=RealFloquet(s,q['orbit'],a.N,a.device);v,error=recover_mode(f,u,growth,D,dt)
        cp=f.cp;del f;gc.collect();cp.get_default_memory_pool().free_all_blocks()
        m=Monodromy(s,q['orbit'],dtmax,a.device)
        assert len(v)==m.dim and abs(dt-m.dt)<1e-14
        mu=np.exp(growth*m.T);mv=m.matvec(v);phase=m.phase_vector()
        ph=m.matvec(phase)
        row=dict(label=a.label,orbit=q['orbit'],N=a.N,dt_ms=m.dt,J_EE_core=m.J,T_ms=m.T,
            expected_multiplier=mu,independent_rayleigh_multiplier=float(v@mv/(v@v)),
            full_state_mode_relative_defect=float(np.linalg.norm(mv-mu*v)/np.linalg.norm(v)),
            phase_relative_defect=float(np.linalg.norm(ph-phase)/np.linalg.norm(phase)),
            output_reconstruction_error=error,
            scope='Direct propagation of this eigenfunction in all nine states and physical delay history. Not a complete spectrum. Strong inherited instability can amplify whole-period integration error.')
        write(PERIODIC_OUT/f'{a.label}_monodromy_check_N{a.N}_dt{dtmax:g}.json',row)
        print('REAL MODE CHECK',row,flush=True)
        del m;gc.collect();cp.get_default_memory_pool().free_all_blocks()


if __name__=='__main__':main()
