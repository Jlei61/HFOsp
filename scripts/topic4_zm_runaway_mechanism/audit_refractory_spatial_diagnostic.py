"""Read stored spatial trajectories independently; never promote on numerical QA."""
from common import OUT, BASE, model, read, write, log, np
from native_readouts import readouts, window_stats
from closure_network_sensitivity_audit import high_entry
from scipy.ndimage import uniform_filter1d
import argparse

DEST=OUT/'conditioned_refractory_spatial_diagnostic'
WINDOWS=[(500,3000),(4000,8000),(8000,9420),(1000,9420)]

def main(partial=False,grid=20):
    jobs=read(DEST/'jobs.json');contract=read(DEST/'contract.json')
    if not partial:assert jobs['status']=='COMPLETE' and len(jobs['completed'])==len(contract['runs'])
    s=model(grid);coarse=model(20);cell=coarse.geo['group_cell'][coarse.E]
    count=np.bincount(cell,weights=coarse.sizes[coarse.E],minlength=400);w=count/count.sum()
    native=np.load(BASE/'native_reference/seed9108401_readouts.npz')
    nd=read(BASE/'native_reference/checkpoint_projections.json');rows=[]
    for label in ['native']+jobs['completed']:
        if label=='native':
            t=native['t'];field=native['rate_cells'];whole=field.astype(float)@w
            assert np.allclose(whole,native['allE'],rtol=1e-6,atol=1e-4)
            d={k:float(v['D']) for k,v in nd.items()};qa={'source':'Original seed9108401 SNN, original recorded clock.'}
        else:
            z=np.load(DEST/label/'trajectory.npz');t=z['time_ms'];field=z['field_E_hz']
            if grid==40:
                from refractory_spatial_resolution import mapping,projections
                parent,_=mapping(coarse,s);assert np.array_equal(parent,z['parent_g20'])
                P=projections(s,coarse,parent)
                restricted=(P[20][0]@z['group_rate_hz'].astype(float).T).T
                assert np.max(abs(restricted-field))<1e-4
                fine=(P[40][0]@z['group_rate_hz'].astype(float).T).T
                assert np.max(abs(fine-z['field_E_hz_grid40']))<1e-4
            assert np.array_equal(t,np.arange(12500)+1.) and np.array_equal(count,z['cell_counts'])
            assert np.array_equal(z['state_time_ms'],np.arange(10,12501,10))
            whole=field.astype(float)@w;group=z['group_rate_hz'].astype(float)[:,s.E]@s.mean_weights
            error=float(max(abs(whole-group).max(),abs(whole-z['global_E_hz']).max()))
            assert error<1e-4,error
            dd=1-z['Z'].astype(float)[:,s.E]@s.mean_weights;de=float(abs(dd-z['D']).max());assert de<1e-6
            conditional=z['group_expected_rate_hz'];assert np.isfinite(conditional).all()
            # Available mass is 1 minus a binary64 sum of past count fractions.
            # At exactly zero available cells, cancellation can leave a tiny
            # negative diagnostic mean. Emitted counts must still be >=0.
            nref_max=round(float(s.ref.max())/contract['dt_ms'])
            eps=np.finfo(float).eps;gamma=nref_max*eps/(1-nref_max*eps)
            expected_roundoff_bound=4*gamma*1000/contract['dt_ms']
            assert conditional.min()>=-expected_roundoff_bound
            assert z['group_rate_hz'].min()>=0
            assert z['Z'].min()>=0 and z['Z'].max()<=1 and np.array_equal(z['Z'][:,~s.E],np.ones_like(z['Z'][:,~s.E]))
            assert np.isfinite(z['M_current']).all() and z['M_current'].min()>=0
            d={k:float(z['D'][np.flatnonzero(z['state_time_ms']==int(k))[0]]) for k in nd}
            expected=z['group_expected_rate_hz'].astype(float)[:,s.E]@s.mean_weights
            qa=dict(weighted_rate_error_hz=error,D_float32_error=de,Z_physical=True,
                    M_dynamic_peak_mV=float(z['M_current'].max()),
                    expected_emitted_global_mean_difference_hz=float((whole-expected).mean()),
                    final_tick=int(z['final_tick'][0]),Z_and_M_dynamic=True)
            qa['conditional_mean_roundoff']=dict(minimum_hz=float(conditional.min()),negative_values=int((conditional<0).sum()),
                binary64_occupancy_bound_hz=float(expected_roundoff_bound),raw_values_changed=False,emitted_rates_nonnegative=True)
            counts=z['group_rate_hz'].astype(float)*s.sizes[None,:]/1000
            bounds={}
            for pop,mask,span in [('E',s.E,2),('I',~s.E,1)]:
                integrated=counts[:-1,mask]+counts[1:,mask] if span==2 else counts[:,mask]
                excess=integrated-s.sizes[mask];bad=excess>1e-4
                bounds[pop]=dict(window_ms=span,violations=int(bad.sum()),group_windows=bad.size,
                    maximum_excess_spikes=float(excess.max()),fraction=float(bad.mean()))
            qa['emitted_refractory_bound']=bounds
            if 'binomial' in label:
                assert all(v['violations']==0 for v in bounds.values())
                assert np.array_equal(z['final_own_history'],z['final_emitted_history'])
                h=z['final_own_history']*s.sizes[None,:]*float(z['dt_ms'])
                assert abs(h-np.rint(h)).max()<1e-9
                qa['same_actual_refractory_and_synaptic_counts']=True
            assert qa['final_tick']==round(12500/contract['dt_ms'])
        sm=uniform_filter1d(whole,10,mode='nearest');entry=high_entry(t,sm)
        events,_,_,_=readouts(t,field,count,label);complete=[]
        for ev in events:
            a=int(np.searchsorted(t,ev['start_ms']));b=a+int(ev['duration_ms'])
            if a>=20 and b+20<=len(sm) and (sm[a-20:a]<5).all() and (sm[b:b+20]<5).all():complete.append(ev)
        wins={f'{a}-{b}':window_stats(complete,a,b) for a,b in WINDOWS}
        quiet={f'{a}-{b}':float(np.mean(sm[(t>=a)&(t<b)]<5)) for a,b in WINDOWS}
        tail_persistence=float(w[(field[-1000:]>50).mean(0)>=.9].sum())
        row=dict(label=label,high_onset_ms=entry,D_at_native_checkpoints=d,
                 complete_event_windows=wins,quiet_fraction_by_window=quiet,
                 tail_global_mean_hz=float(whole[-1000:].mean()),tail_spatial_persistence=tail_persistence,
                 total_complete_events=len(complete),qa=qa)
        if label!='native':assert entry==read(DEST/label/'result.json')['high_onset_ms']
        rows.append(row)
    result=dict(status='PARTIAL_READOUT_AUDIT_PASS' if partial else 'READOUT_AUDIT_PASS',rows=rows,
                statistical_unit='One native realization and the explicitly registered diagnostic trajectories. No population inference.',
                scope='All Z/M dynamic, unchanged physical graph and frozen local response. Local validation remains FAIL; no accepted replacement or bifurcation claim.',
                completed=jobs['completed'],model_promoted=False)
    write(DEST/('partial_comparison.json' if partial else 'independent_comparison.json'),result)
    log('SPATIAL READOUT',[(r['label'],r['high_onset_ms'],r['D_at_native_checkpoints']['9870'],r['total_complete_events']) for r in rows])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');p.add_argument('--filtered',action='store_true');p.add_argument('--count-consistent',action='store_true');p.add_argument('--fine-grid',action='store_true');a=p.parse_args()
    if a.filtered:DEST=OUT/'conditioned_refractory_external_filter_pair'
    if a.count_consistent:DEST=OUT/'conditioned_refractory_count_consistency'
    if a.fine_grid:DEST=OUT/'conditioned_refractory_spatial_resolution'
    main(a.partial,40 if a.fine_grid else 20)
