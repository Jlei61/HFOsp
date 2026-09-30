"""Recover the best long-window high-A recurrence in the original full flow.

The source screen uses actual 20--70s records at D_A=.3631576413. This
replay is an initial guess only; neither approximate recurrence nor a
Newton failure establishes a periodic orbit or a bifurcation.
"""
from common import OUT,np,read,write,log
from onset_state_continuation import build,regional_weights
from fine_rate_frozen_Z_fields import capture,restore
from onset_period_return import dynamical_state,errors
import argparse,os,time

BASE=OUT/'core_a_bifurcation_type_20260924'
PARENT=BASE/'near_returns/above70s_A_sustained_B_cycle'
DEST=PARENT/'exact_seed'


def main(device,single_B_cycle=False):
    global DEST
    screen=read(PARENT/'result.json')
    assert screen['status']=='RECURRENCE_SEEDS_ONLY'
    if single_B_cycle:
        checked=read(PARENT/'candidate_B_activity_structure.json')
        assert checked['status']=='RECURRENCE_SCREEN_ACTIVITY_ONLY'
        candidates=[r for r in checked['rows'] if len(r['Core_B_50Hz_20ms_segments_relative_ms'])==1 and r['B_max_hz']>200 and r['B_quiet_fraction']>.1]
        assert candidates
        candidate=min(candidates,key=lambda r:r['score'])
        DEST=PARENT/'exact_single_B_cycle'
    else:
        candidate=min(screen['candidates'],key=lambda r:r['score'])
    first,last=candidate['time1_ms'],candidate['time2_ms']
    assert 20000<first<last<=70000
    times=np.linspace(first,last,5)
    source=BASE/'censoring_controls/above_SN'
    assert read(source/'joined70s_audit.json')['status']=='AUDIT_PASS'
    relative_start=int((first-20000-1)//5000)*5000
    start=20000+relative_start
    statefile=(source/f'checkpoint{relative_start}.npz' if relative_start
               else BASE/'fold_attractor_contrast/above/final_state.npz')
    DEST.mkdir(exist_ok=True);assert not(DEST/'jobs.json').exists()
    write(DEST/'contract.json',dict(question='Does the observed A-continuous/B-intermittent state admit a full periodic skeleton in the stronger local-resource field?',
        source=str(statefile),source_total_elapsed_ms=start,candidate=candidate,
        times_ms=times.tolist(),number_segments=4,dt_ms=.05,
        selection=('Lowest original recurrence score among candidate intervals with exactly one B>50Hz20ms burst, peak>200Hz and>10percent quiet. This only nominates an elementary skeleton; fundamental-period and closure checks remain necessary.' if single_B_cycle else 'Lowest all-group rate/history and dynamic-M recurrence score in audited20--70s, periods150--2000ms, A continuously>50Hz and both endpoints>200Hz. No imposed B waveform or source SNN future spikes.'),
        single_B_cycle_selector=single_B_cycle,
        equations='Unchanged full3479-group spatial conditional rate drift, physical delays, all Z held and all M dynamic; only native Core-A Z pattern differs from native9s.',
        validation='Every replayed1ms3479-group float32 rate must equal its source record bitwise. Save exact full-state quarter-period nodes, not interpolated states. This is a seed, not a closed orbit.',model_promoted=False))
    jobs=dict(status='RUNNING',pid=os.getpid());write(DEST/'jobs.json',jobs)
    e=build(device);base=dict(np.load(statefile));restore(e,base)
    ticks=np.rint((times-start)/e.dt).astype(int)
    assert np.max(abs(ticks*e.dt+start-times))<1e-9
    clock0=int(base['clock'][0]);targets={clock0+int(t):j for j,t in enumerate(ticks)}
    stop=int(np.ceil(last/10))*10
    expected={b:np.load(source/f'block{b:02d}.npz')['group_rate_hz']
              for b in range(relative_start//5000,(stop-20000-1)//5000+1)}
    states={};begin=time.time();W=regional_weights(e.s)
    for tm in range(start+10,stop+1,10):
        now=int(e.local.clock.get()[0]);end=now+round(10/e.dt)
        if any(now<t<=end for t in targets):
            for tick in range(now+1,end+1):
                e.step()
                if tick in targets:
                    e.cp.cuda.get_current_stream().synchronize();state=capture(e);j=targets[tick]
                    assert int(state['clock'][0])==tick
                    assert np.array_equal(state['syn'][5],base['syn'][5])
                    states[j]=state;np.savez_compressed(DEST/f'node{j:02d}.npz',**state)
            e.cp.cuda.get_current_stream().synchronize();r=e.output.get()
        else:r=e.chunk()
        block=(tm-20000-1)//5000;offset=tm-20000-block*5000
        assert np.array_equal(r[:,0].astype('f4'),expected[block][offset-10:offset])
        if tm%1000==0:log('ABOVE CORE A EXACT REPLAY',tm,round(time.time()-begin,1))
    assert len(states)==5
    er=errors(dynamical_state(states[0]),dynamical_state(states[4]),e.s.sizes/e.s.sizes.sum())
    R=np.concatenate([expected[k] for k in sorted(expected)]).astype(float)@W.T
    selected=R[int(first-start):int(last-start)]
    structure=[dict(region=name,mean_hz=float(selected[:,j].mean()),min_hz=float(selected[:,j].min()),
        max_hz=float(selected[:,j].max()),fraction_below5=float(np.mean(selected[:,j]<5)))
        for j,name in enumerate(['Global E','Core A','Core B','Surround'])]
    write(DEST/'result.json',dict(status='EXACT_REPLAY_PASS_RECURRENCE_ONLY',period_ms=last-first,
        times_ms=times.tolist(),full_state_errors=er,actual_state_structure=structure,
        all_original_1ms_group_rates_bitwise=True,all_Z_held=True,all_M_dynamic=True,
        seconds=time.time()-begin,model_promoted=False))
    jobs.update(status='COMPLETE');write(DEST/'jobs.json',jobs)
    log('ABOVE CORE A SEED COMPLETE',er,structure)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);p.add_argument('--single-B-cycle',action='store_true')
    a=p.parse_args();main(a.device,a.single_B_cycle)
