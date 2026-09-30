"""Original A4 criteria plus explicitly separate strict-event diagnostics."""
from common import OUT, BASE, model, np, read, write, log
from native_readouts import readouts, window_stats
from scipy.ndimage import uniform_filter1d
from datetime import datetime
import argparse

DEST=OUT/'conditioned_refractory_fine_forcing'
PARENT=OUT/'conditioned_refractory_spatial_resolution'


def register():
    assert not (DEST/'readout_contract.json').exists()
    write(DEST/'readout_contract.json',dict(
        created_local=datetime.now().astimezone().isoformat(),
        source_contract=str(BASE/'a4_contract.json'),
        original_primary='Use native_readouts active segments, original1-9.42s event statistics andfull-recordquiet fraction as in originalA4; do not replace with strictcompleteevent subset.',
        original_six_gates=read(BASE/'a4_contract.json')['acceptance'],
        resource_clock='Read D at actualstate_time_ms9870. Legacy a4_acceptance indexed987 in a10ms-endpoint array, i.e.9880; preserve intended9870ms, not that historical index error.',
        role='Actual-count arm is the primary stochastic contrast. Expectedflux still receives exogenous OU; neither is the constant-input deterministic bifurcation skeleton.',
        strict_diagnostic='Also retain previously used extra20msquiet onbothsides andactual754/786core50percentpeak order, without promoting either to a newA4gate.',
        completion='NumericalreadoutPASS, originalA4criteria andlocalresponsevalidation reported separately. No automaticmodelpromotion or bifurcation label.'))


def core_order(t, core, events, sm, label):
    smooth=uniform_filter1d(core,5,axis=0,mode='nearest'); rows=[]
    for e in events:
        if not 1000<=e['start_ms']<9420:continue
        a=int(np.searchsorted(t,e['start_ms']));b=a+int(e['duration_ms'])
        strict=bool(a>=20 and b+20<=len(sm) and (sm[a-20:a]<5).all() and (sm[b:b+20]<5).all())
        seg=smooth[a:b];peak=seg.max(0);eligible=bool((peak>=20).all())
        times=[float(t[a+np.argmax(seg[:,j]>=.5*peak[j])]) for j in [0,1]] if eligible else [None,None]
        rows.append(dict(label=label,start_ms=e['start_ms'],duration_ms=e['duration_ms'],
            strict_complete=strict,eligible_core_pair=eligible,peak_core_hz=peak.tolist(),
            core_halfpeak_time_ms=times,lag_B_minus_A_ms=times[1]-times[0] if eligible else None))
    summary={}
    for key,items in [('original_events',rows),('strict_complete_events',[r for r in rows if r['strict_complete']])]:
        lags=[r['lag_B_minus_A_ms'] for r in items if r['eligible_core_pair']]
        summary[key]=dict(events=len(items),eligible_core_pairs=len(lags),A_first=sum(x>2 for x in lags),
            B_first=sum(x<-2 for x in lags),ties=sum(abs(x)<=2 for x in lags),
            median_lag_ms=float(np.median(lags)) if lags else None)
    return rows,summary


def gates(w,quiet,entry,D,nativeD):
    n=w['n'];duration=w.get('median_duration_ms');area=w.get('median_area');extent=w.get('median_extent_mm')
    return dict(
        self_limited_events=bool(n and duration is not None and 50<=duration<=200 and quiet>=.15),
        two_core_participation=bool(n and w.get('both_cores',0)/n>=.5),
        surround_recruitment=bool(area is not None and .3<=area<=1.),
        propagation=bool(w.get('forward',0)>0 and w.get('reverse',0)>0 and extent is not None and 5<=extent<=20),
        entry=bool(entry is not None and 7000<=entry<=13000),
        D_track=bool(abs(D-nativeD)<=.05))


def audit(partial):
    import audit_refractory_spatial_diagnostic as qa
    assert (DEST/'readout_contract.json').exists()
    qa.DEST=DEST;qa.main(partial=partial,grid=40)
    jobs=read(DEST/'jobs.json');fineqa=read(DEST/('partial_comparison.json' if partial else 'independent_comparison.json'))
    s=model(40);coarse=model(20)
    count=np.bincount(coarse.geo['group_cell'][coarse.E],weights=coarse.sizes[coarse.E],minlength=400)
    nativeD=read(BASE/'native_reference/checkpoint_projections.json')['9870']['D']
    chunks=OUT.parent/'fig5_current_network_z_state_v1/replay/runs/eta0.0005_s9108401/chunks'
    nt=[];nr=[]
    for path in sorted(chunks.glob('*.npz')):
        z=np.load(path);nt.append(z['time_ms']);nr.append(z['regions_1ms'][:,:3])
    nt=np.concatenate(nt);nr=np.concatenate(nr).astype(float)/np.array([754,786,30460])*1000
    sources=[('native',None)]+[(name+'_parent',PARENT/name) for name in ['recorded_drive_expected','recorded_drive_binomial_seed1']]
    sources += [(name+'_fine_forcing',DEST/name) for name in jobs['completed']]
    rows=[];event_rows=[]
    for label,source in sources:
        z=np.load(BASE/'native_reference/seed9108401_readouts.npz' if source is None else source/'trajectory.npz')
        if source is None:
            t=z['t'];field=z['rate_cells'];D9870=nativeD;cores=nr[:,:2]
            assert np.array_equal(t,nt)
            assert np.max(abs(nr@np.array([754,786,30460])/32000-z['allE']))<1e-4
        else:
            t=z['time_ms'];field=z['field_E_hz'];idx=np.flatnonzero(z['state_time_ms']==9870);assert len(idx)==1
            D9870=float(z['D'][idx[0]]);r=z['group_rate_hz'].astype(float)
            cores=np.column_stack([r[:,s.E&(s.geo['group_region']==k)]@s.sizes[s.E&(s.geo['group_region']==k)]/s.sizes[s.E&(s.geo['group_region']==k)].sum() for k in [0,1]])
        events,summary,whole,sm=readouts(t,field.astype(float),count,label)
        primary=window_stats(events,1000,9420);strict=[]
        for ev in events:
            a=int(np.searchsorted(t,ev['start_ms']));b=a+int(ev['duration_ms'])
            if a>=20 and b+20<=len(sm) and (sm[a-20:a]<5).all() and (sm[b:b+20]<5).all():strict.append(ev)
        order,order_summary=core_order(t,cores,events,sm,label);event_rows.extend(order)
        # Reproduce previously completed strict core order as a regression check.
        prior=read(OUT/'native_current_memory/core_lead.json')
        if not label.endswith('_fine_forcing'):
            oldlabel='native' if source is None else source.name
            prev=next(r for r in prior['summary'] if r['label']==oldlabel)
            assert all(prev[k]==order_summary['strict_complete_events'][k] for k in ['events','eligible_core_pairs','A_first','B_first','ties','median_lag_ms'])
        checks=gates(primary,summary['quiet_fraction'],summary['high_onset_ms'],D9870,nativeD)
        edges=np.diff(np.r_[0,(sm<5).astype(int),0]);quiet_spans=[(a,b) for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)) if b-a>=20]
        terminal=None if not quiet_spans or quiet_spans[-1][1]==len(t) else float(t[quiet_spans[-1][1]])
        item=dict(label=label,source=str(source) if source is not None else str(BASE/'native_reference/seed9108401_readouts.npz'),
            original_event_window=primary,original_six_checks=checks,original_six_passed=sum(checks.values()),
            quiet_fraction_full=summary['quiet_fraction'],high_entry_ms=summary['high_onset_ms'],D_at_native9870=D9870,
            strict_complete_window=window_stats(strict,1000,9420),physical_core_order=order_summary,
            terminal_no_return_start_ms=terminal,terminal_right_censored=True,
            tail_global_hz=float(whole[-1000:].mean()),
            tail_persistent_fraction=float((count/count.sum())[(field[-1000:]>50).mean(0)>=.9].sum()))
        if source is not None:
            for key,tm in [('terminal',terminal),('high_entry',summary['high_onset_ms'])]:
                item[key+'_D_interpolated']=None if tm is None else float(np.interp(tm,z['state_time_ms'],z['D']))
        rows.append(item)
    write(DEST/('partial_scientific_comparison.json' if partial else 'scientific_comparison.json'),dict(
        status='PARTIAL_COMPARISON_COMPLETE' if partial else 'COMPARISON_COMPLETE',rows=rows,core_event_rows=event_rows,
        original_gates_source=str(BASE/'a4_contract.json'),readout_contract=str(DEST/'readout_contract.json'),
        scope='One native history and matched rate diagnostics. OriginalA4rules andstrict-event diagnostics are separate. No changedgates, native confidence intervals ornewbifurcationclaim.',
        local_response_validation='FAIL_REMAINS',model_promoted=False,bifurcation_type='NOT_ESTABLISHED'))
    log('FINE FORCING READOUT',[(r['label'],r['original_six_passed'],r['original_six_checks'],r['physical_core_order']['original_events']) for r in rows])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','audit']);p.add_argument('--partial',action='store_true');a=p.parse_args()
    register() if a.command=='register' else audit(a.partial)
