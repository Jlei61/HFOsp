"""Independent finite-window comparison of count and conditional-drift arms."""
from common import OUT,model,np,read,write,log
from conditional_drift_Z_fields import DEST,SOURCE,native_field
from refractory_spatial_resolution import mapping,projections
from audit_fine_rate_frozen_Z_fields import runs,moving_mean
import argparse


def audit(partial=False):
    jobs=read(DEST/'jobs.json');contract=read(DEST/'contract.json')
    if not partial:assert jobs['status']=='COMPLETE' and jobs['completed']==contract['arms']
    assert read(DEST/'implementation_check.json')['status']=='PASS'
    s=model(40);coarse=model(20);parent,_=mapping(coarse,s);P,count=projections(s,coarse,parent)[20];w=count/count.sum()
    rows=[];baseline={r['label']:r for r in read(SOURCE/'independent_comparison.json')['rows']}
    for label in jobs['completed']:
        z=np.load(DEST/label/'trajectory.npz');worker=read(DEST/label/'result.json')
        r=z['group_rate_hz'].astype(float);field=z['field_E_hz'].astype(float);whole=z['global_E_hz'];t=z['time_ms']
        assert np.array_equal(z['group_rate_hz'],z['group_expected_rate_hz'])
        assert np.array_equal(t,np.arange(9001,12501.)) and np.array_equal(z['state_time_ms'],np.arange(9010,12501.,10))
        assert np.isfinite(r).all() and r.min()>=0
        assert np.isfinite(z['M_current']).all() and z['M_current'].min()>=0
        errors=dict(field=float(np.max(abs((P@r.T).T-field))),global_rate=float(np.max(abs(r[:,s.E]@s.mean_weights-whole))),
            field_global=float(np.max(abs(field@w-whole))))
        assert max(errors.values())<1e-4
        tm=int(label.split('_')[1][1:]);expected=native_field(s,tm)
        assert np.array_equal(z['Z'],np.broadcast_to(expected.astype('f4'),z['Z'].shape))
        assert np.array_equal(z['final_synaptic_slow_state'][5],expected)
        assert np.array_equal(z['final_emitted_history'],z['final_own_history'])
        history=z['final_own_history'];tick=int(z['final_tick'][0]);assert tick==250000 and float(z['dt_ms'])==.05
        chronological=history[(tick-len(history)+1+np.arange(len(history)))%len(history)]
        occupancy=[]
        for mask,ref in [(s.E,2),(~s.E,1)]:
            used=np.lib.stride_tricks.sliding_window_view(chronological[:,mask],round(ref/.05),axis=0).sum(-1)*.05
            assert used.min()>=-1e-10 and used.max()<=1+1e-9
            occupancy.append(float(used.max()))
        sm=moving_mean(whole,10);high=next((float(t[a]) for a,b in runs(sm>=200) if b-a>=200),None)
        quiet=[[float(t[a]),float(t[b-1]+1)] for a,b in runs(sm<5) if b-a>=20]
        episodes=[];complete=[]
        for a,b in runs(sm>=5):
            if b-a<20 or sm[a:b].max()<20:continue
            whole_event=bool(a>=20 and b+20<=len(sm) and (sm[a-20:a]<5).all() and (sm[b:b+20]<5).all())
            ev=dict(start_ms=float(t[a]),duration_ms=b-a,peak_hz=float(sm[a:b].max()),
                reaches_left_boundary=bool(a==0),reaches_right_boundary=bool(b==len(t)),complete_with_20ms_quiet=whole_event)
            episodes.append(ev)
            if whole_event:complete.append(ev)
        persistent=float(w[(field[-1000:]>50).mean(0)>=.9].sum())
        assert high==worker['high_onset_ms'] and abs(persistent-worker['tail_persistent_fraction'])<1e-12
        near=int(np.count_nonzero(abs(sm-5)<1e-9))
        if not near:
            assert [(e['start_ms'],e['duration_ms']) for e in complete]==[(e['start_ms'],e['duration_ms']) for e in worker['complete_events']]
        row=dict(label=label,D=float(1-expected[s.E]@s.mean_weights),high_onset_ms=high,quiet_intervals_ms=quiet,
            complete_events=complete,active_episodes=episodes,tail_global_hz=float(whole[-1000:].mean()),
            tail_quiet_fraction=float((sm[-1000:]<5).mean()),tail_persistent_fraction=persistent,
            count_control={k:baseline[label][k] for k in ['initial_D','high_onset_ms','complete_events','quiet_intervals_ms','tail_global_hz','tail_persistent_fraction']},
            numerical=dict(weighted_errors_hz=errors,maximum_refractory_occupancy_E_I=occupancy,near_quiet_threshold_bins=near),
            M_dynamic=True,Z_held=True,private_diffusion_unchanged=True)
        rows.append(row)
    write(DEST/('partial_comparison.json' if partial else 'independent_comparison.json'),dict(
        status='PARTIAL_READOUT_AUDIT_PASS' if partial else 'READOUT_AUDIT_PASS',rows=rows,
        unit='One common own-model9000ms history and future external realization; matched interventions in count innovations at three full native Z fields.',
        scope='Finite-time noise-removal diagnostic; recorded external OU remains. Not deterministic ensemble average, constant-input bifurcation or native acceptance.',
        model_promoted=False,bifurcation_type='NOT_ESTABLISHED'))
    log('PRIVATE DRIFT INDEPENDENT AUDIT',[(r['label'],len(r['complete_events']),r['high_onset_ms'],r['tail_global_hz']) for r in rows])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args();audit(a.partial)
