"""Local dynamic susceptibilities around a corrected density equilibrium.

A one-step current impulse is applied with symmetric signs. Voltage/noise/
refractory memory evolves with the unrelaxed physical map. A separate constant
perturbation verifies the zero-frequency susceptibility. Network M and delay
feedback are incorporated later, not frozen out of the stability problem.
"""
from stationary_response import *


def run_response(args):
    source=Path(args.equilibrium)
    status=read(source/'status.json');assert status['status']=='FULL_MAP_EQUILIBRIUM_CORRECTED'
    config=read(source/'config.json')
    with np.load(source/'stationary_local_state.npz') as f:
        original={k:f[k] for k in f.files}
    P=len(original['theta'])
    folder=source/f'response_eps{args.epsilon:g}_{args.duration:g}ms'
    folder.mkdir(parents=True,exist_ok=False)
    model=LocalStationaryDensity(np.tile(original['theta'],2),np.tile(original['population'],2),
        np.tile(original['current_mv'],2),config['degree'],config['voltage_dv'],args.device,basis_mode=config.get('basis_mode','legacy'))
    equilibrium=cp.asarray(np.tile(original['F'],(2,1,1)))
    model.F[:]=equilibrium
    baseline=cp.asarray(np.tile(original['current_mv'],2));sign=cp.r_[cp.ones(P),-cp.ones(P)]
    model.drive[:]=baseline+args.epsilon*sign
    steps=round(args.duration/DT);response=[];started=time.time();last=started
    diagnostics=[]
    for k in range(steps):
        if k==1:model.drive[:]=baseline
        rates=model.map()
        response.append(cp.asnumpy((rates[:P]-rates[P:])/(2*args.epsilon)))
        model.F,model.Q=model.Q,model.F
        if (k+1)%100==0:
            physical=cp.einsum('k,gkv->gv',model.mass,model.F)
            diagnostics.append(dict(time_ms=(k+1)*DT,
                drive_baseline_error=float(cp.max(abs(model.drive-baseline)).get()),
                mass_error=float(cp.max(abs(physical.sum(1)-1.)).get()),
                maximum_negative_probability=float(cp.maximum(-physical,0.).sum(1).max().get()),
                max_rate_hz=float(cp.max(rates).get()),min_rate_hz=float(cp.min(rates).get())))
        if time.time()-last>20:
            write(folder/'status.json',dict(status='RUNNING_IMPULSE',pid=os.getpid(),
                completed_ms=(k+1)*DT,wall_s=time.time()-started))
            print('impulse',config['D'],(k+1)*DT,'ms',flush=True);last=time.time()
    kernel=np.asarray(response)
    if args.static_reference:
        refcfg=read(args.static_reference.parent.parent/'config.json')
        assert refcfg['D']==config['D'] and refcfg['degree']==config['degree']
        assert refcfg.get('basis_mode','legacy')==config.get('basis_mode','legacy')
        with np.load(args.static_reference) as f:static_derivative=f['static_derivative_hz_per_mv']
        diag=dict(reused_static_reference=str(args.static_reference.resolve()),
                  purpose='Independent impulse-amplitude refinement; the existing constant-input derivative is unchanged')
    else:
        model.F[:]=equilibrium;model.drive[:]=baseline+args.epsilon*sign
        def progress(it,residual,rate):
            write(folder/'status.json',dict(status='RUNNING_STATIC_DERIVATIVE',pid=os.getpid(),iteration=it,
                maximum_stationary_residual=float(residual.max())))
            print('static susceptibility',config['D'],it,residual.max(),flush=True)
        static,_,diag,_=model.solve(8000,progress,tolerance=2e-11,acceleration=args.acceleration)
        static_derivative=(static[:P]-static[P:])/(2*args.epsilon)
    integrated=kernel.sum(0);absolute_area=np.sum(abs(kernel),axis=0)
    tail=np.sum(abs(kernel[-round(50/DT):]),axis=0)
    valid=abs(static_derivative)>1e-4
    qa=dict(static_solver=diag,max_static_derivative_discrepancy=float(np.max(abs(integrated-static_derivative))),
        median_relative_dc_error=float(np.median(abs(integrated[valid]-static_derivative[valid])/abs(static_derivative[valid]))),
        max_relative_dc_error=float(np.max(abs(integrated[valid]-static_derivative[valid])/abs(static_derivative[valid]))),
        max_tail_fraction_of_absolute_response=float(np.max(tail/np.maximum(absolute_area,1e-12))),
        epsilon_mv=args.epsilon,impulse_duration_ms=DT,recorded_ms=args.duration,
        numerical_derivative_refinement='REQUIRES_SECOND_EPSILON',network_stability='NOT_YET_COMPUTED')
    qa['usable_for_dynamic_spectrum']=bool(qa['max_relative_dc_error']<.01 and
        qa['max_tail_fraction_of_absolute_response']<.01 and
        max(r['drive_baseline_error'] for r in diagnostics)<1e-12 and
        max(r['mass_error'] for r in diagnostics)<1e-6)
    write(folder/'impulse_diagnostics.json',diagnostics)
    np.savez_compressed(folder/'susceptibility.npz',kernel_hz_per_mv=kernel,
        time_s=np.arange(steps)*DT/1000.,static_derivative_hz_per_mv=static_derivative,
        integrated_derivative_hz_per_mv=integrated,tail_absolute_area=tail)
    write(folder/'status.json',dict(status='LOCAL_RESPONSE_COMPLETE',qa=qa,wall_s=time.time()-started))
    print(folder,qa,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--equilibrium',type=Path,required=True)
    ap.add_argument('--epsilon',type=float,default=.001);ap.add_argument('--duration',type=float,default=200.)
    ap.add_argument('--device',type=int,default=0);ap.add_argument('--static-reference',type=Path)
    ap.add_argument('--acceleration',action='store_true')
    run_response(ap.parse_args())
