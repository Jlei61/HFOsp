"""Bounded long-orbit search with full-state Poincare diagnostics.

Mixed precision is an initial-guess search only. No recurrence or stability is
accepted until the candidate is corrected in the FP64 physical map. The section
uses downward crossings of a 1 ms global rate, with a 50 ms refractory interval.
All density coefficients, M, synaptic currents and ordered delay memory enter
the recurrence distance; a rate-trace resemblance alone is insufficient.
"""
from mixed_precision_pilot import MixedDensity
from autonomous_density import *


STATE_NAMES=('F','history','qa','ia','qg','ig','qe','ie','Z','M')


def capture(model):
    state={k:getattr(model,k).copy() for k in STATE_NAMES}
    # The ring index itself is a coordinate convention, not a physical phase.
    order=(model.step_index-np.arange(model.D)) % model.D
    state['ordered_history']=model.history[order].copy()
    state['step_index']=model.step_index
    return state


def separation(a,b,weights):
    fa,fb=a['F'],b['F']
    scale=cp.maximum(cp.sum((fa*fa+fb*fb)*.5,axis=(1,2)),1e-24)
    d2=cp.sum((fa-fb)**2,axis=(1,2))/scale
    density=float(cp.sqrt(weights@d2).get())
    values={}
    for k in ('qa','ia','qg','ig','M','ordered_history'):
        x,y=a[k],b[k]
        # Rates for delay memory; mV for currents; Hz-equivalent for M.
        if k=='ordered_history':x=x*1000/DT;y=y*1000/DT
        values[k]=float(cp.sqrt(cp.mean((x-y)**2)/cp.maximum(cp.mean((x*x+y*y)*.5),1.)).get())
    return dict(density_relative_RMS=density,other_relative_RMS=values,
                score=max(density,*values.values()))


def save_state(folder,state,config):
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'checkpoint.npz',
        **{k:cp.asnumpy(state[k]) for k in STATE_NAMES},step_index=state['step_index'])
    write(folder/'config.json',config)


def run(args):
    source=Path(args.source);config=read(source/'config.json')
    folder=OUT/'recurrence_searches'/args.label;folder.mkdir(parents=True,exist_ok=False)
    if args.pdf_fp32 or args.conservative_pdf_fp32:
        from fast_search_density import FastSearchDensity,ConservativeFastSearchDensity
        cls=ConservativeFastSearchDensity if args.conservative_pdf_fp32 else FastSearchDensity
        precision='FP32 PDF transport/search only; currents, M and readout FP64; final FP64 validation required'
        if args.conservative_pdf_fp32:precision+='; modal mass restored after voltage transport'
    else:
        cls=AutonomousDensity if args.fp64 else MixedDensity
        precision='FP64' if args.fp64 else 'FP32 noise matrix product only; final FP64 validation required'
    D=config['D'] if args.D is None else args.D
    m=cls(D,config['degree'],config['voltage_dv'],args.device,basis_mode=config.get('basis_mode','legacy'));m.restore(source,allow_D_change=args.D is not None)
    initial_projection=None
    if args.normalize_initial_noise_marginal:
        values,vectors=np.linalg.eig(cp.asnumpy(m.transition));at=np.argmin(abs(values-1.))
        marginal=vectors[:,at].real;marginal/=cp.asnumpy(m.mass)@marginal
        assert abs(values[at]-1.)<1e-10 and marginal.min()>0
        total=cp.sum(m.F,axis=2);factor=cp.asarray(marginal)[None,:]/total
        maximum=float(cp.max(abs(factor-1.)).get());assert maximum<1e-3
        m.F*=factor[:,:,None]
        initial_projection=dict(maximum_modal_relative_correction=maximum,
            scope='Initial checkpoint roundoff correction to exact stationary private-noise marginal; all subsequent steps use the declared physical map.')
    initial=m.step_index*DT
    cfg=dict(config,D=D,alpha=m.alpha,source_D=config['D'],initial_ms=initial,duration_ms=args.duration,resumed_from=str(source.resolve()),
        precision=precision,initial_noise_marginal_projection=initial_projection,
        section_downcrossing_hz=args.section,minimum_section_interval_ms=50.,
        retained_sections=args.retain,acceptance='SEARCH_ONLY_NO_PERIODIC_ORBIT_CLAIM')
    write(folder/'config.json',cfg)
    weights=cp.asarray(m.geo['group_size']/40000.)
    count=np.bincount(m.geo['group_cell'],weights=np.where(m.geo['population']==0,m.geo['group_size'],0),minlength=1600)
    traces=[];fields=[];slow=[];block=cp.zeros(m.P);sections=[];comparisons=[]
    prior=None;last_section=-np.inf;best=np.inf;best_pair=None;best_record=None
    started=time.time();last=started;status='COMPLETE'
    for step in range(round(args.duration/DT)):
        block+=m.advance_step()
        if (step+1)%10==0:
            rate=block*1000.;trace=cp.asnumpy(cp.r_[m.e_weights@rate,m.region_weights@rate])
            traces.append(trace);fields.append(cp.asnumpy(cp.bincount(m.observable_cell,weights=rate*m.e_sizes,minlength=1600)))
            block.fill(0.);now=m.step_index*DT
            if prior is not None and prior>args.section>=trace[0] and now-last_section>=50.:
                state=capture(m)
                for old in sections:
                    sep=separation(old,state,weights)
                    row=dict(start_ms=old['step_index']*DT,end_ms=now,lag_ms=now-old['step_index']*DT,**sep)
                    comparisons.append(row)
                    if sep['score']<best:
                        best=sep['score'];best_pair=(old,state);best_record=row
                sections.append(state);sections=sections[-args.retain:];last_section=now
                write(folder/'recurrences.json',dict(best=best_record,comparisons=comparisons,
                    interpretation='Candidate full-state returns; finite section and mixed precision errors remain'))
            prior=float(trace[0])
        if (step+1)%100==0:
            slow.append([m.step_index*DT,float((m.e_weights@m.M).get()),float((m.e_weights@m.Z).get())])
            diag=m.diagnostics()
            if not diag['finite'] or diag['maximum_mass_error']>1e-5 or diag['minimum_step_spike_probability'] < -1e-6:
                status='NUMERICAL_FAILURE';break
        if time.time()-last>20:
            write(folder/'status.json',dict(status='RUNNING',pid=os.getpid(),completed_ms=m.step_index*DT,
                wall_s=time.time()-started,best_return=best_record,diagnostics=m.diagnostics()))
            print('recurrence',args.label,m.step_index*DT,'best',best_record,flush=True);last=time.time()
    rates=np.asarray(traces);field=np.asarray(fields)/np.maximum(count,1)[None,:]
    assert np.allclose(field@count/32000.,rates[:,0],rtol=1e-12,atol=1e-12)
    np.savez_compressed(folder/'trajectory.npz',rate_1ms=rates,field_1ms=field,slow_10ms=np.asarray(slow),count_e=count)
    m.save(folder)
    if best_pair:
        for i,state in enumerate(best_pair):save_state(folder/f'best_pair_{i}',state,cfg)
    write(folder/'status.json',dict(status=status,completed_ms=m.step_index*DT,wall_s=time.time()-started,
        best_return=best_record,diagnostics=m.diagnostics(),bifurcation_acceptance='NOT_ESTABLISHED'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--label',required=True);ap.add_argument('--duration',type=float,default=10000.)
    ap.add_argument('--section',type=float,default=10.);ap.add_argument('--retain',type=int,default=16)
    ap.add_argument('--device',type=int,default=0);ap.add_argument('--D',type=float)
    ap.add_argument('--normalize-initial-noise-marginal',action='store_true')
    group=ap.add_mutually_exclusive_group();group.add_argument('--fp64',action='store_true');group.add_argument('--pdf-fp32',action='store_true');group.add_argument('--conservative-pdf-fp32',action='store_true')
    run(ap.parse_args())
