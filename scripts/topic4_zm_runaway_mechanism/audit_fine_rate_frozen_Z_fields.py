"""Independent readout of the fixed same-history spatial-Z experiment.

No model fitting, extra simulation, acceptance waiver or bifurcation naming.
"""
from common import OUT, np, model, read, write, log
from refractory_spatial_resolution import mapping, projections
from fine_rate_frozen_Z_fields import DEST, SOURCE, native_field
import argparse


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def moving_mean(x, width):
    # scipy uniform_filter1d's even-window convention: five past, four future.
    pad = [(width // 2, width - 1 - width // 2)] + [(0, 0)] * (x.ndim - 1)
    p = np.pad(x, pad, mode='edge')
    return np.lib.stride_tricks.sliding_window_view(p, width, axis=0).mean(-1)


def initial_transplant_audit(s, state):
    """Separate a finite parameter jump from a quasistatic branch experiment."""
    syn=state['syn'];assert int(state['clock'][0])==180000
    regions=s.geo['group_region'];rows=[]
    for tm in [9000,9420,9870]:
        target=native_field(s,tm);delta=(syn[5]-target)*syn[3]
        for region,name in enumerate(['Core A','Core B','Surround']):
            mask=s.E&(regions==region);w=s.sizes[mask]/s.sizes[mask].sum()
            rows.append(dict(native_Z_time_ms=tm,region=name,own_Z=float(w@syn[5,mask]),
                native_Z=float(w@target[mask]),instant_net_input_shift_mV=float(w@delta[mask]),
                own_GABA_mean_mV=float(w@syn[3,mask]),M_unchanged_mV=float(w@syn[4,mask]),
                minimum_shift_mV=float(delta[mask].min()),maximum_shift_mV=float(delta[mask].max())))
    result=dict(status='ALGEBRAIC_INITIAL_STATE_AUDIT',tick=180000,rows=rows,
        formula='delta_mu=(Z_own-Z_transplant)*I_G; same current and M at application, before any future update.',
        scope='Initial applied input shift, not integrated response, basin crossing, model mismatch attribution or bifurcation evidence.',network_runs=0)
    path=DEST/'initial_transplant_input_shift.json'
    if path.exists():assert read(path)==result
    else:write(path,result)


def audit(partial=False):
    jobs = read(DEST/'jobs.json'); contract = read(DEST/'contract.json')
    if not partial:
        assert jobs['status'] == 'COMPLETE' and jobs['completed'] == contract['arms']
    assert read(DEST/'replay_qa.json')['status'] == 'PASS'
    s = model(40); coarse = model(20); parent, _ = mapping(coarse, s)
    projection, count = projections(s, coarse, parent)[20]
    original = np.load(SOURCE/'trajectory.npz'); initial = np.load(DEST/'checkpoint9000.npz')
    initial_transplant_audit(s,initial)
    w = count/count.sum(); results = []
    for label in jobs['completed']:
        folder = DEST/label; z = np.load(folder/'trajectory.npz'); worker = read(folder/'result.json')
        t = z['time_ms']; ts = z['state_time_ms']; r = z['group_rate_hz'].astype(float)
        assert np.array_equal(t, np.arange(9001,12501.))
        assert np.array_equal(ts, np.arange(9010,12501.,10))
        assert np.array_equal(z['cell_counts'], count) and np.array_equal(z['parent_g20'], parent)
        field = z['field_E_hz'].astype(float); global_rate = z['global_E_hz']
        errors = dict(field=float(abs((projection@r.T).T-field).max()),
                      global_from_field=float(abs(field@w-global_rate).max()),
                      global_from_groups=float(abs(r[:,s.E]@s.mean_weights-global_rate).max()))
        assert max(errors.values()) < 1e-4, errors
        assert np.isfinite(r).all() and r.min() >= 0
        assert np.isfinite(z['M_current']).all() and z['M_current'].min() >= 0
        assert np.isfinite(z['Z']).all() and z['Z'].min() >= 0 and z['Z'].max() <= 1
        assert np.array_equal(z['Z'][:,~s.E], np.ones_like(z['Z'][:,~s.E]))
        D = 1-z['Z'].astype(float)[:,s.E]@s.mean_weights
        derr = float(abs(D-z['D']).max()); assert derr < 6e-8
        assert int(z['final_tick'][0]) == 250000 and float(z['dt_ms']) == .05
        assert np.array_equal(z['final_own_history'], z['final_emitted_history'])
        h = z['final_own_history']*s.sizes*.05
        assert abs(h-np.rint(h)).max() < 1e-9
        # Every recorded aligned refractory window respects the physical count.
        counts = r*s.sizes/1000
        count_error = float(abs(counts-np.rint(counts)).max()); assert count_error < 1e-4
        for mask, span in [(s.E,2),(~s.E,1)]:
            n = counts[:-1,mask]+counts[1:,mask] if span == 2 else counts[:,mask]
            assert (n-s.sizes[mask]).max() < 1e-4
        if label == 'dynamic_reference':
            for key in ['group_rate_hz','group_expected_rate_hz']:
                assert np.array_equal(z[key],original[key][9000:])
            for key in ['Z','M_current','D']:
                assert np.array_equal(z[key],original[key][900:])
            for key in ['final_synaptic_slow_state','final_local_state','final_own_history','final_emitted_history','final_tick']:
                assert np.array_equal(z[key],original[key])
            field_held = False
        else:
            expected = initial['syn'][5] if label == 'own_Z9000_held' else native_field(s,int(label.split('_')[1][1:]))
            assert np.array_equal(z['Z'],np.broadcast_to(expected.astype('f4'),z['Z'].shape))
            assert np.array_equal(z['final_synaptic_slow_state'][5],expected)
            field_held = True
        sm = moving_mean(global_rate,10)
        high = next((float(t[a]) for a,b in runs(sm>=200) if b-a>=200),None)
        assert high == worker['high_onset_ms'],(label,high,worker['high_onset_ms'])
        quiet = [(a,b) for a,b in runs(sm<5) if b-a>=20]
        region_ids = s.geo['group_region']
        core = np.array([r[:,s.E & (region_ids==region)]@(
            s.sizes[s.E & (region_ids==region)]/s.sizes[s.E & (region_ids==region)].sum())
            for region in [0,1]]).T
        core = moving_mean(core,5); complete = []; episodes = []
        for a,b in runs(sm>=5):
            if b-a<20 or sm[a:b].max()<20: continue
            episodes.append(dict(start_ms=float(t[a]),duration_ms=b-a,peak_hz=float(sm[a:b].max()),
                reaches_left_boundary=bool(a==0),reaches_right_boundary=bool(b==len(t)),
                complete_with_20ms_quiet=bool(a>=20 and b+20<=len(t) and
                    (sm[a-20:a]<5).all() and (sm[b:b+20]<5).all())))
            if a<20 or b+20>len(t) or not (sm[a-20:a]<5).all() or not (sm[b:b+20]<5).all(): continue
            pk=core[a:b].max(0); times=None
            if (pk>=20).all():
                times=[float(t[a+np.argmax(core[a:b,j]>=.5*pk[j])]) for j in [0,1]]
            complete.append(dict(start_ms=float(t[a]),duration_ms=b-a,peak_hz=float(sm[a:b].max()),
                                 core_halfpeak_ms=times,lag_B_minus_A_ms=times[1]-times[0] if times else None))
        # Mark exact-threshold roundoff rather than silently changing a definition.
        near = int(np.count_nonzero(abs(sm-5)<1e-9))
        old_events=[(e['start_ms'],e['duration_ms']) for e in worker['complete_events']]
        new_events=[(e['start_ms'],e['duration_ms']) for e in complete]
        if not near: assert old_events == new_events,(label,old_events,new_events)
        tail=float(w[(field[-1000:]>50).mean(0)>=.9].sum())
        assert abs(tail-worker['tail_persistent_fraction'])<1e-12
        lags=[e['lag_B_minus_A_ms'] for e in complete if e['lag_B_minus_A_ms'] is not None]
        regions={}
        for name,region in [('Core A',0),('Core B',1),('Surround E',2)]:
            mask=s.E & (region_ids==region); weight=s.sizes[mask]/s.sizes[mask].sum()
            regions[name]=dict(final_Z=float(z['Z'][-1,mask]@weight),final_m_mV=float(z['M_current'][-1,mask]@weight))
        results.append(dict(label=label,initial_D=worker['initial_D'],high_onset_ms=high,
            complete_events=complete,active_episodes=episodes,
            quiet_intervals_ms=[[float(t[a]),float(t[b-1]+1)] for a,b in quiet],
            core_order=dict(eligible=len(lags),A_first=sum(v>2 for v in lags),B_first=sum(v< -2 for v in lags),ties=sum(abs(v)<=2 for v in lags)),
            tail_global_hz=float(global_rate[-1000:].mean()),tail_quiet_fraction=float((sm[-1000:]<5).mean()),
            tail_persistent_fraction=tail,regions=regions,Z_held=field_held,M_dynamic=True,
            numerical=dict(weighted_errors_hz=errors,D_float32_error=derr,count_float32_error=count_error,
                           near_quiet_threshold_bins=near,complete_event_roundoff_agreement=old_events==new_events)))
    result=dict(status='PARTIAL_READOUT_AUDIT_PASS' if partial else 'READOUT_AUDIT_PASS',rows=results,
        statistical_unit='One fixed rate-model9s history, one common forcing and innovation realization, five prescribed interventions.',
        interpretation='Finite-time conditional comparison, not bifurcation or basin certification; native late clamp histories differ. No parameter fitting or acceptance waiver.',
        model_promoted=False,bifurcation_type='NOT_ESTABLISHED')
    write(DEST/('partial_comparison.json' if partial else 'independent_comparison.json'),result)
    log('FROZEN Z READOUT',[(r['label'],len(r['complete_events']),r['high_onset_ms'],r['tail_global_hz']) for r in results])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args();audit(a.partial)
