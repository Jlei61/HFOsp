"""Impulse response of the frozen-active-set local density Jacobian.

This propagates an infinitesimal density perturbation. It avoids subtracting
two full trajectories after the response has decayed to floating-point noise.
The initial current derivative is still independently refined in epsilon, and
the integrated response is compared with the stationary constant-input slope.
"""
from stationary_response import *


def tangent_code():
    code=CODE.replace('void voltage(\n const double* P,',
        'void tangent_voltage(\n const double* reference, const double* P,')
    old='double pm=P[base+j];if(pm==0.)continue;'
    assert code.count(old)==1;code=code.replace(old,'double pm=P[base+j];')
    old='double slope=0.,f=pm/w[j];if(j>0&&j<nv-1){double dl=(f-P[base+j-1]/w[j-1])/(v-c[j-1]),dr=(P[base+j+1]/w[j+1]-f)/(c[j+1]-v);if(dl*dr>0)slope=copysign(fmin(fabs(dl),fabs(dr)),dl);}'
    new='''double slope=0.,f=pm/w[j];if(j>0&&j<nv-1){
      double rf=reference[base+j]/w[j];
      double rdl=(rf-reference[base+j-1]/w[j-1])/(v-c[j-1]);
      double rdr=(reference[base+j+1]/w[j+1]-rf)/(c[j+1]-v);
      double dl=(f-P[base+j-1]/w[j-1])/(v-c[j-1]);
      double dr=(P[base+j+1]/w[j+1]-f)/(c[j+1]-v);
      if(rdl*rdr>0)slope=fabs(rdl)<=fabs(rdr)?dl:dr;
    }'''
    assert code.count(old)==1;return code.replace(old,new)


def run(args):
    source=Path(args.equilibrium);cfg=read(source/'config.json')
    assert read(source/'status.json')['status']=='FULL_MAP_EQUILIBRIUM_CORRECTED'
    state=dict(np.load(source/'stationary_local_state.npz'))
    folder=source/f'tangent_response_eps{args.epsilon:g}_{args.duration:g}ms'
    folder.mkdir(parents=True,exist_ok=False)
    m=LocalStationaryDensity(state['theta'],state['population'],state['current_mv'],cfg['degree'],cfg['voltage_dv'],args.device,basis_mode=cfg.get('basis_mode','legacy'))
    F=cp.asarray(state['F']);baseline=m.drive.copy()
    m.F[:]=F;m.drive[:]=baseline+args.epsilon;plus_rate=m.map().copy();plus=m.Q.copy()
    m.F[:]=F;m.drive[:]=baseline-args.epsilon;minus_rate=m.map().copy();minus=m.Q.copy()
    perturbation=(plus-minus)/(2*args.epsilon);response=[cp.asnumpy((plus_rate-minus_rate)/(2*args.epsilon))]
    # Current perturbations cannot change the stationary private-noise marginal.
    perturbation-=cp.sum(perturbation,axis=2)[:,:,None]*F/cp.sum(F,axis=2)[:,:,None]
    del plus,minus;reference=cp.ascontiguousarray(cp.matmul(m.A,F));m.drive[:]=baseline
    module=cp.RawModule(code=tangent_code(),options=('--fmad=false',),name_expressions=['tangent_voltage'])
    kernel=module.get_function('tangent_voltage');m.F[:]=perturbation
    started=time.time();last=started;max_marginal=0.
    for k in range(1,round(args.duration/DT)):
        moved=cp.ascontiguousarray(cp.matmul(m.A,m.F))
        kernel((m.P*m.K,),(128,),(reference,moved,m.Q,m.flux,m.de,m.dc,m.dw,m.nodes,m.ratio,m.decay,
            m.drive,m.refs,np.int32(m.K),np.int32(m.nv),np.int32(m.width)))
        response.append(cp.asnumpy(m.flux@m.mass*1000/DT));m.F,m.Q=m.Q,m.F
        if (k+1)%100==0:max_marginal=max(max_marginal,float(cp.max(abs(m.F.sum(2))).get()))
        if time.time()-last>20:
            write(folder/'status.json',dict(status='RUNNING_TANGENT_IMPULSE',pid=os.getpid(),completed_ms=(k+1)*DT,wall_s=time.time()-started))
            print('tangent response',cfg['D'],(k+1)*DT,flush=True);last=time.time()
    h=np.asarray(response);integral=h.sum(0);area=abs(h).sum(0);tail=abs(h[-round(50/DT):]).sum(0)
    with np.load(args.static_reference) as z:static=z['static_derivative_hz_per_mv']
    valid=abs(static)>1e-4;error=abs(integral[valid]-static[valid])/abs(static[valid])
    np.savez_compressed(folder/'susceptibility.npz',kernel_hz_per_mv=h,time_s=np.arange(len(h))*DT/1000.,
        static_derivative_hz_per_mv=static,integrated_derivative_hz_per_mv=integral,tail_absolute_area=tail)
    qa=dict(max_relative_dc_error=float(error.max()),median_relative_dc_error=float(np.median(error)),
        max_tail_fraction_of_absolute_response=float(np.max(tail/np.maximum(area,1e-12))),
        maximum_noise_marginal_tangent_error=max_marginal,epsilon_mv=args.epsilon,
        static_reference=str(args.static_reference),
        derivative='Analytic frozen-minmod transport Jacobian; finite difference initial current impulse',
        usable_for_dynamic_spectrum=bool(error.max()<.01 and np.max(tail/np.maximum(area,1e-12))<.01))
    write(folder/'status.json',dict(status='LOCAL_TANGENT_RESPONSE_COMPLETE',qa=qa,wall_s=time.time()-started));print(qa,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--equilibrium',type=Path,required=True)
    ap.add_argument('--static-reference',type=Path,required=True);ap.add_argument('--epsilon',type=float,default=.001)
    ap.add_argument('--duration',type=float,default=200.);ap.add_argument('--device',type=int,default=1);run(ap.parse_args())
