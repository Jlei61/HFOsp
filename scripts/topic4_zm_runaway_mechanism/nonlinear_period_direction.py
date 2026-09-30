"""Check an unvalidated Floquet direction using independent nonlinear flows.

The orbit, network and Z are unchanged. This is an error diagnostic, not a
replacement for periodicity, phase invariance or Floquet convergence gates.
"""
from periodic_flow_closure import *
from scipy.interpolate import CubicSpline


def main(a):
    s=model();attach_native_path(s)
    z=np.load(a.orbit);sol=dict(r=z['r'],T=float(z['T']),D=float(z['D']))
    s.set_D(sol['D']);assert np.max(abs(s.Z-z['Z']))<1e-12
    spectral=np.load(a.spectrum);metadata=read(Path(a.spectrum).with_suffix('.json'))
    assert Path(metadata['orbit']).resolve()==Path(a.orbit).resolve()
    vector=spectral['vectors'][:,a.index].real;claimed=spectral['multipliers'][a.index]
    assert abs(claimed.imag)<1e-10
    olddt=metadata['dt_ms'];dy=vector[:14*s.P].reshape(14,s.P).copy();dy[11]=0.
    oldhist=vector[14*s.P:].reshape(-1,s.P)
    o=StreamPeriodic(s,len(sol['r']),a.device);o.cache_mean_operators=False
    Y,_=orbit_states(o,sol,len(sol['r']));initial=Y[0].copy()
    scale=np.maximum(Y[:-1].std(0),np.abs(initial)*1e-3+1e-8)
    amplitude=1e-4/max(np.max(np.abs(dy)/scale),1e-20)
    dr0=(s.output(initial+1e-3*dy)-s.output(initial-1e-3*dy))/(2e-3)
    hs=CubicSpline(np.arange(len(oldhist)+1)*olddt,np.vstack([dr0,oldhist]),axis=0)
    del Y,o;gc.collect()
    import cupy as cp
    cp.get_default_memory_pool().free_all_blocks();rows=[]
    for dt in a.dt:
        depth=round(s.delays[-1]/dt)+1;lags=np.arange(depth)*dt
        hbase=fourier_value(sol['r'],-lags,sol['T']);dh=hs(lags)
        v=np.r_[dy.ravel(),dh[1:].ravel()]
        steps=round(sol['T']/dt)
        for fraction in a.fraction:
            eps=amplitude*fraction;ends=[]
            for sign in [-1,1]:
                history=np.empty_like(hbase)
                history[(-np.arange(depth))%depth]=hbase+sign*eps*dh
                e=EndpointIntegrator(s,dt=dt,initial=initial+sign*eps*dy,
                    history=history,dynamic_z=False,device=a.device)
                block=StepBlock(e,128);rest=steps%128
                finalblock=StepBlock(e,rest) if rest else None
                for _ in range(steps//128):block.run()
                if finalblock:finalblock.run()
                yy=e.y.get();hh=e.history.get()[(-np.arange(1,depth))%depth]
                ends.append(np.r_[yy.ravel(),hh.ravel()])
                del e,block,finalblock;gc.collect();cp.get_default_memory_pool().free_all_blocks()
            propagated=(ends[1]-ends[0])/(2*eps)
            rayleigh=float(v@propagated/(v@v))
            rows.append(dict(dt_ms=dt,epsilon=eps,fraction=fraction,
                elapsed_ms=steps*dt,period_offset_ms=steps*dt-sol['T'],
                norm_gain=float(np.linalg.norm(propagated)/np.linalg.norm(v)),
                rayleigh=rayleigh,
                relative_eigendirection_residual=float(np.linalg.norm(propagated-rayleigh*v)/np.linalg.norm(propagated)),
                relative_to_unvalidated_prediction=float(np.linalg.norm(propagated-claimed.real*v)/np.linalg.norm(propagated))))
            write(OUT/'floquet'/f'{a.label}.json',dict(status='RUNNING',rows=rows))
            log('INDEPENDENT NONLINEAR DIRECTION',rows[-1])
    write(OUT/'floquet'/f'{a.label}.json',dict(status='DIAGNOSTIC_COMPLETE',rows=rows,
        orbit=a.orbit,spectrum=a.spectrum,unvalidated_multiplier=float(claimed.real),
        spectrum_phase_valid=metadata['phase_valid'],Z='held',M='dynamic',
        scope='Central differences of nonlinear EndpointHeun full-state/history flow; not accepted Floquet stability or bifurcation classification.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('orbit');p.add_argument('spectrum')
    p.add_argument('--index',type=int,default=0);p.add_argument('--dt',nargs='+',type=float,default=[.00625,.003125])
    p.add_argument('--fraction',nargs='+',type=float,default=[1.,.1])
    p.add_argument('--device',type=int,default=0);p.add_argument('--label',required=True)
    main(p.parse_args())
