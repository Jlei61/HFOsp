"""Consolidate the completed mesh pair without conflating entry definitions."""
from common import OUT,BASE,model,np,read,write,log
from scipy.ndimage import uniform_filter1d

DEST=OUT/'conditioned_refractory_spatial_resolution'


def main():
    jobs=read(DEST/'jobs.json');assert jobs['status']=='COMPLETE' and len(jobs['completed'])==2
    fine=read(DEST/'independent_comparison.json');coarse=read(OUT/'conditioned_refractory_external_filter_pair/independent_comparison.json')
    count=read(OUT/'conditioned_refractory_count_consistency/independent_comparison.json')
    sources=[('native',20,None,fine['rows'][0]),
        ('expected_1mm',20,OUT/'conditioned_refractory_external_filter_pair/recorded_drive_expected',next(r for r in coarse['rows'] if r['label']=='recorded_drive_expected')),
        ('expected_0p5mm',40,DEST/'recorded_drive_expected',next(r for r in fine['rows'] if r['label']=='recorded_drive_expected')),
        ('counts_1mm',20,OUT/'conditioned_refractory_count_consistency/recorded_drive_binomial_seed1',next(r for r in count['rows'] if r['label']=='recorded_drive_binomial_seed1')),
        ('counts_0p5mm',40,DEST/'recorded_drive_binomial_seed1',next(r for r in fine['rows'] if r['label']=='recorded_drive_binomial_seed1'))]
    native_slow=np.load(OUT/'native_input_bridge/runs/native_t8000_inputs_observe/chunks/0000090000_0000095000.npz')
    checkpoints=read(BASE/'native_reference/checkpoint_projections.json')
    for tcheck in [9300,9420]:
        zz=float(np.interp(tcheck,native_slow['slow_time_ms'],native_slow['Z'][:,0]))
        assert abs(1-zz-checkpoints[str(tcheck)]['D'])<1e-12
    rows=[]
    for label,grid,source,audit in sources:
        z=np.load(BASE/'native_reference/seed9108401_readouts.npz' if source is None else source/'trajectory.npz')
        t=z['t'] if source is None else z['time_ms'];r=z['allE'].astype(float) if source is None else z['global_E_hz']
        sm=uniform_filter1d(r,10,mode='nearest');edges=np.diff(np.r_[0,(sm<5).astype(int),0]);spans=[(a,b) for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)) if b-a>=20]
        a,b=spans[-1];terminal=None if b==len(t) else float(t[b]);duration=0. if terminal is None else float(t[-1]-terminal+1)
        row=dict(label=label,grid=grid,high_entry_ms=audit['high_onset_ms'],D_at_native9870=audit['D_at_native_checkpoints']['9870'],
            events=audit['complete_event_windows']['1000-9420'],quiet_fraction=audit['quiet_fraction_by_window']['1000-9420'],
            tail_global_rate_hz=audit['tail_global_mean_hz'],tail_persistent_fraction=audit['tail_spatial_persistence'],
            terminal_no_return_start_ms=terminal,terminal_no_return_duration_ms=duration,terminal_at_least500ms=duration>=500,
            own_high_entry_D=None,own_high_entry_Z_regions=None)
        if source is not None:
            s=model(grid);clock=z['state_time_ms'];state=z['Z'].astype(float)
            row['terminal_D_interpolated']=None if terminal is None else float(np.interp(terminal,clock,z['D']))
            if row['high_entry_ms'] is not None:
                tt=row['high_entry_ms'];k=np.searchsorted(clock,tt);w=(tt-clock[k-1])/(clock[k]-clock[k-1]);zz=(1-w)*state[k-1]+w*state[k]
                row['own_high_entry_D']=float(1-zz[s.E]@s.mean_weights);row['own_high_entry_time_bracket_ms']=clock[k-1:k+1].tolist()
                row['own_high_entry_Z_regions']={name:float(zz[(s.geo['group_region']==j)&s.E]@s.sizes[(s.geo['group_region']==j)&s.E]/s.sizes[(s.geo['group_region']==j)&s.E].sum()) for j,name in enumerate(['Core A','Core B','Surround'])}
        else:
            assert 9000<=terminal<=9500
            row['terminal_D_interpolated']=float(1-np.interp(terminal,native_slow['slow_time_ms'],native_slow['Z'][:,0]))
            k=np.searchsorted(native_slow['slow_time_ms'],terminal);row['terminal_D_time_bracket_ms']=native_slow['slow_time_ms'][k-1:k+1].tolist()
            row['own_high_entry_D']=audit['D_at_native_checkpoints']['9870']
            row['own_high_entry_D_note']='9870ms actual checkpoint,1.5ms after readout9868.5ms,not interpolated.'
        rows.append(row)
    write(DEST/'consolidated_comparison.json',dict(status='DIAGNOSTIC_COMPLETE_NATIVE_EQUIVALENCE_FAIL',rows=rows,
        definitions=dict(high_entry='10ms-smoothed globalE>=200Hz for200ms; finite-time readout, not a bifurcation.',
            terminal='After last >=20ms quiet episode (globalE<5Hz), no qualifying return through12.5s; explicitly report duration and flag>=500ms. Right-censored observation, no asymptotic stability claim.',
            event='All complete events satisfying existing onset/return andduration/peak rules. Same1-9.42s clock, original20x20 readout.',
            direction='Original fixed centroid displacement along A-to-B axis; >1mmforward,<-1mmreverse,otherwiseundirected.',
            resource='D=1-originalE-cellweightedZ. Entry-D differs from D sampled at native clock; rate10ms slow-state interpolation retained withbracket.'),
        same_Z_M_dynamic=True,model_promoted=False,bifurcation_type='NOT_ESTABLISHED',
        execution=dict(fine_runs_expected=2,fine_runs_completed=2,simulation_failures=0,audit_attempt_failures=1,
            audit_failure_resolution='Strict nonnegative conditionalmean assertion found19values down to-1.03e-12Hz fromzero-available-mass cancellation. Corrected audit uses a derived binary64 refractorysummation bound7.11e-10Hz; raw files unchanged,actualemittedrates nonnegative and integer/refractory checks PASS.'),
        conclusion='Finer mesh restores some temporal statistics and highentry but neither expected nor count arm restores bidirectional propagation or native resource entry. Does not resolve localresponse validationFAIL.'))
    log('MESH PAIR COMPLETE',[(r['label'],r['high_entry_ms'],r['own_high_entry_D'],r['terminal_no_return_start_ms'],r['events'].get('n'),r['events'].get('forward'),r['events'].get('reverse')) for r in rows])


if __name__=='__main__':main()
