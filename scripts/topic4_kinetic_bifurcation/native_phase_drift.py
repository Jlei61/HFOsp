"""Check native phase drift at several nearby points on a candidate curve.

A generalized-return fixed point alone does not prove an irrational invariant
circle: a time-grid-locked native periodic orbit is an alternative. Several
fractional-step phase initializations are evolved by the unchanged native map.
The resulting signed return fractions and transverse defects distinguish the
next numerical question; finite phase sampling is not a mathematical proof.
"""
from generalized_return_spectrum import *


def run(a):
    root=a.corrected;rcfg=read(root/'config.json');rresult=read(root/'result.json')
    assert rresult['status']=='GENERALIZED_RETURN_CORRECTED'
    source=Path(rcfg['source']);cfg=read(source/'config.json');folder=OUT/'native_phase_drifts'/a.label
    folder.mkdir(parents=True,exist_ok=False)
    m=AutonomousDensity(cfg['D'],cfg['degree'],cfg['voltage_dv'],a.device,basis_mode=cfg.get('basis_mode','legacy'))
    m.restore(source);coords=StateCoordinates(m);m.restore(root/'best_state')
    initial=capture(m);initial_step=m.step_index;x=coords.pack(m);n=rcfg['integer_steps']
    original_normal=cp.asarray(np.load(root/'section_normal.npy'))
    # These six interpolation vectors are only needed between replays. Keep
    # them on the host so they do not compete with the full return on GPU.
    local=[np.zeros(coords.size)]
    for _ in range(5):m.advance_step();local.append(cp.asnumpy(coords.pack(m)-x))
    def reset(v):
        for k,value in initial.items():
            if k not in ('ordered_history','step_index'):getattr(m,k)[:]=value
        m.step_index=initial_step;coords.unpack(v,m)
        for k in ('masserror','maxneg','minflux','positive_emitted','negative_emitted','maximum_lower_mass'):
            getattr(m,k).fill(0.)
        m.minimum_drive_bound.fill(cp.inf);m.diagnostic_start_step=initial_step
    rows=[];started=time.time();last=started
    write(folder/'config.json',dict(corrected_source=str(root.resolve()),D=cfg['D'],phases=a.phases,
        original_native_step_ms=DT,integer_return_steps=n,interpolation_order=5,
        object='Signed native return phase and transverse defect at fractional-step initializations',
        limitation='Initialization and cross-section interpolation are approximations; native evolutions are unchanged. Finite phase sampling does not prove an invariant circle.'))
    for phase in a.phases:
        xa=x.copy();normal=cp.zeros_like(x)
        for c,dc,d in zip(coefficients(phase,5),coefficient_derivatives(phase,5),local):
            vector=cp.asarray(d);xa+=float(c)*vector;normal+=float(dc)*vector
        del vector;normal/=cp.linalg.norm(normal)
        reset(xa);deltas=[];values=[];reference_values=[]
        reset_error=float(cp.linalg.norm(coords.pack(m)-xa).get());assert reset_error<1e-12
        section_geometry=dict(initial_phase=phase,reset_error=reset_error,
            initial_step=m.step_index,normal_cosine_to_original=float(cp.dot(normal,original_normal).get()),
            input_offset_norm=float(cp.linalg.norm(xa-x).get()))
        write(folder/'section_geometry.json',section_geometry)
        for step in range(1,n+6):
            m.advance_step()
            if step>=n:
                d=coords.pack(m)-xa;deltas.append(d);values.append(float(cp.dot(normal,d).get()))
                reference_values.append(float(cp.dot(original_normal,d).get()))
            if time.time()-last>20:
                write(folder/'progress.json',dict(status='RUNNING',pid=os.getpid(),initial_phase=phase,
                    elapsed_ms=step*DT,wall_s=time.time()-started,completed_phases=len(rows)));last=time.time()
                print(a.label,phase,step*DT,flush=True)
        estimates=[]
        gram=np.array([[float(cp.dot(u,v).get()) for v in deltas] for u in deltas])
        write(folder/'latest_section_samples.json',dict(**section_geometry,integer_steps=n,delta_gram=gram,
            values=values,original_normal_values=reference_values,
            full_state_offsets=[float(cp.linalg.norm(d).get()) for d in deltas],diagnostics=m.diagnostics()))
        for order in (3,5):
            try:alpha=crossing(values,order)
            except ValueError as error:
                write(folder/'result.json',dict(status='PHASE_SECTION_BRACKET_FAILED',rows=rows,
                    phase=phase,order=order,error=str(error),samples='latest_section_samples.json',
                    interpretation='No phase drift or locking conclusion; retain scalar section samples for diagnosis'))
                return
            f=cp.zeros_like(x)
            for c,d in zip(coefficients(alpha,order),deltas):f+=float(c)*d
            estimates.append(dict(order=order,return_phase_fraction=alpha,return_time_ms=(n+alpha)*DT,
                weighted_transverse_defect=float(cp.linalg.norm(f).get())))
            del f
        row=dict(initial_phase=phase,native_integer_return_defect=float(cp.linalg.norm(deltas[0]).get()),
            estimates=estimates,diagnostics=m.diagnostics())
        rows.append(row);write(folder/'phase_results.json',dict(status='RUNNING',rows=rows))
        print('native phase result',a.label,row,flush=True)
        del xa,normal,deltas
    fractions=np.array([row['estimates'][-1]['return_phase_fraction'] for row in rows])
    status='SAME_SIGN_DRIFT_AT_TESTED_PHASES' if fractions.min()*fractions.max()>0 else 'PHASE_LOCKING_ALTERNATIVE_REMAINS'
    write(folder/'result.json',dict(status=status,rows=rows,wall_s=time.time()-started,
        minimum_absolute_return_phase_fraction=float(abs(fractions).min()),
        interpretation='Signed finite-phase diagnostic. Neither integer periodicity nor an invariant circle is accepted from this test alone.'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--corrected',type=Path,required=True);ap.add_argument('--label',required=True)
    ap.add_argument('--phases',type=float,nargs='+',default=[0.,.25,.5,.75]);ap.add_argument('--device',type=int,default=0)
    run(ap.parse_args())
