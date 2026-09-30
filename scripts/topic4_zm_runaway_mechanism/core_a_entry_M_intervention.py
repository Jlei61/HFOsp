"""Matched M-clamp diagnostic inside the first long Core-A activity.

This is a fast/slow intervention, not a replacement model or bifurcation
certificate. Both arms retain the same full spatial network and histories.
"""
from common import OUT, np, read, write, log
from onset_state_continuation import build, regional_weights
from fine_rate_frozen_Z_fields import capture, restore
from scipy.ndimage import uniform_filter1d
from datetime import datetime
import argparse, os, time

BASE = OUT/'core_a_bifurcation_type_20260924/reference_stability_gap'
SOURCE = BASE/'actual_D0300/from_interictal_history'
DEST = BASE/'first_long_M_intervention'


def main(device,start_ms=5000,duration_ms=3000):
    global DEST
    assert start_ms in [2800,5000] and duration_ms in [3000,5000]
    if start_ms!=5000:DEST=BASE/f'first_long_M_intervention_from{start_ms}'
    DEST.mkdir(exist_ok=True)
    assert not (DEST/'jobs.json').exists()
    assert read(SOURCE/'whole_record_audit.json')['status'] == 'AUDIT_PASS'
    source = SOURCE/f'checkpoint{start_ms}.npz'
    write(DEST/'contract.json', dict(
        created_local=datetime.now().astimezone().isoformat(),
        question='Does ongoing M accumulation help terminate the first prolonged Core-A activity, or does the fast network end it even when every M is held at its current value?',
        source=str(source) if source.exists() else str(DEST/'replayed_initial.npz'), source_elapsed_ms=start_ms,
        selection=('Existing exact 5s checkpoint lies inside the first complete long activity (2.853--6.143s).' if start_ms==5000 else 'Replay the original complete initial state to2.8s, the10ms-grid quiet point53ms before the originally observed first long activity. This earlier clamp distinguishes M changes before5s from the negative late-clamp result; a5s-only clamp cannot exclude a prior M-induced crossing.'),
        arms=['dynamic_M', 'held_M'], duration_ms=duration_ms, dt_ms=.05,
        intervention='Hold all original Z in both arms. Same complete fast/M/history initial state. Dynamic arm keeps every M law; held arm changes only existing dynamic-M switches to zero, preserving the exact spatial M field. No rate, threshold, connectivity, delay, gain or input changes.',
        observations='Full group rates every1ms and original group M every10ms. Original 10ms smoothing, Core-A quiet<5Hz sustained20ms. Report first complete quiet interval, not permanent-state or attractor membership.',
        decision='If dynamic M ends the activity while held M does not in this matched window, ongoing M evolution contributes to this termination. Equal termination weakens the necessity of M evolution after the declared start. Neither outcome establishes a fold, Hopf, crisis or other onset type.',
        baseline='New matched dynamic control; independently compare it with the same old recorded rate interval at original storage precision. Preserve any discrepancy instead of attributing it to physiology.',
        budget=f'Two{duration_ms}ms continuations and condition-specific10ms repeat checks; additionally exact-source{start_ms}ms replay if no checkpoint exists. No automatic broad clamp campaign.',
        model_promoted=False))
    jobs=dict(status='RUNNING', pid=os.getpid(), completed=[])
    write(DEST/'jobs.json', jobs)
    e=build(device);W=regional_weights(e.s)
    needed=range((start_ms+duration_ms+4999)//5000)
    old_all=np.concatenate([np.load(SOURCE/f'block{j:02d}.npz')['group_rate_hz'] for j in needed])
    old=old_all[start_ms:start_ms+duration_ms]
    if source.exists():base=dict(np.load(source))
    else:
        condition=read(SOURCE.parent/'conditions.json')['from_interictal_history']
        initial=dict(np.load(condition['initial']));restore(e,initial)
        e.syn[5]=e.cp.asarray(np.load(SOURCE/'block00.npz')['Z']);e.transport.pars[19].fill(0)
        e.cp.cuda.get_current_stream().synchronize();replay=[]
        for j in range(start_ms//10):
            replay.append(e.chunk()[:,0].astype('f4'))
            if (j+1)%100==0:log('EARLY M SOURCE REPLAY',(j+1)*10)
        actual=np.concatenate(replay);difference=abs(actual.astype(float)-old_all[:start_ms].astype(float))
        bound=abs(np.spacing(old_all[:start_ms])).astype(float)+1e-12
        passed=bool(np.all(difference<=bound));base=capture(e)
        write(DEST/'source_replay_check.json',dict(status='STORED_PRECISION_PASS' if passed else 'FAIL',
            maximum_rate_difference_hz=float(difference.max()),outside_bound=int((difference>bound).sum()),
            original_initial_state=condition['initial'],no_future_rates_used_as_input=True))
        np.savez_compressed(DEST/'replayed_initial.npz',**base)
        if not passed:
            jobs.update(status='FAILED_SOURCE_REPLAY');write(DEST/'jobs.json',jobs)
            raise AssertionError('Early intervention source replay failed; no arms launched')
    rows=[]; started=time.time()
    try:
        for label in ['dynamic_M', 'held_M']:
            folder=DEST/label; folder.mkdir()
            restore(e,base)
            if label=='held_M':e.transport.pars[20].fill(0)
            e.cp.cuda.get_current_stream().synchronize()
            initial=capture(e)
            for key, value in base.items():
                if key=='parameters':
                    same=value.copy();same[20]=0 if label=='held_M' else value[20]
                    assert np.array_equal(initial[key],same)
                else:assert np.array_equal(initial[key],value),key
            first=e.chunk(); terminal=capture(e); restore(e,initial); repeat=e.chunk(); end=capture(e)
            qa={key:float(np.max(abs(end[key]-value))) for key,value in terminal.items()}
            passed=all(np.all(abs(end[k]-v)<=1e-14+1e-12*abs(v)) for k,v in terminal.items())
            write(folder/'replay_check.json',dict(status='PASS' if passed else 'FAIL',
                max_abs_by_array=qa, output_bitwise=bool(np.array_equal(first,repeat)),
                component_bound='1e-14+1e-12*abs(reference)', scientific_qualification=False))
            assert passed,qa
            restore(e,initial);rates=[];ms=[]
            for j in range(duration_ms//10):
                x=e.chunk(); assert np.array_equal(x[:,0],x[:,1])
                assert np.isfinite(x).all() and x.min()>=0
                rates.append(x[:,0].astype('f4'));ms.append(e.syn[4].get())
                assert np.array_equal(e.syn[5].get(),base['syn'][5])
                if label=='held_M':assert np.array_equal(ms[-1],base['syn'][4])
                if (j+1)%50==0:
                    jobs.update(arm=label,arm_elapsed_ms=(j+1)*10)
                    write(DEST/'jobs.json',jobs);log('CORE A M INTERVENTION',label,(j+1)*10,time.time()-started)
            r=np.concatenate(rates);m=np.array(ms);regional=r.astype(float)@W.T
            sm=uniform_filter1d(regional[:,1],10,mode='nearest');quiet=sm<5
            changes=np.diff(np.r_[False,quiet,False].astype(int));a=np.flatnonzero(changes==1);b=np.flatnonzero(changes==-1)
            intervals=[dict(start_ms=int(x+start_ms),end_ms=int(y+start_ms),duration_ms=int(y-x),
                left_censored=bool(x==0),right_censored=bool(y==len(sm))) for x,y in zip(a,b) if y-x>=20]
            comparison=None
            if label=='dynamic_M':
                err=abs(r.astype(float)-old.astype(float));bound=abs(np.spacing(old)).astype(float)+1e-12
                comparison=dict(status='STORED_PRECISION_PASS' if np.all(err<=bound) else 'STORED_PRECISION_MISMATCH',
                    maximum_rate_difference_hz=float(err.max()),outside_bound=int((err>bound).sum()))
            np.savez_compressed(folder/'trajectory.npz',elapsed_time_ms=np.arange(start_ms+1,start_ms+duration_ms+1.),
                state_time_ms=np.arange(start_ms+10,start_ms+duration_ms+1.,10),group_rate_hz=r,regional_rate_hz=regional,
                M_current=m,Z=base['syn'][5])
            np.savez_compressed(folder/'final_state.npz',**capture(e))
            row=dict(label=label,status='COMPLETE',quiet_intervals=intervals,
                mean_rates_global_A_B_surround_hz=regional.mean(0).tolist(),
                M_initial_global_A_B_surround=(base['syn'][4]@W.T).tolist(),
                M_final_global_A_B_surround=(m[-1]@W.T).tolist(),
                original_replay_comparison=comparison,M_held_bitwise=label=='held_M',
                original_Z_held_bitwise=True,bifurcation_type='NOT_ESTABLISHED')
            write(folder/'result.json',row);rows.append(row)
            jobs['completed'].append(label);write(DEST/'jobs.json',jobs)
        write(DEST/'result.json',dict(status='COMPLETE',rows=rows,
            scope='Finite paired intervention before or inside one long activity, as declared; no fast-subsystem or full-system bifurcation certificate.',model_promoted=False))
        jobs.update(status='COMPLETE');write(DEST/'jobs.json',jobs)
    except BaseException as exc:
        jobs.update(status='FAILED',error=repr(exc));write(DEST/'jobs.json',jobs);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1)
    p.add_argument('--start-ms',type=int,default=5000);p.add_argument('--duration-ms',type=int,default=3000)
    a=p.parse_args();main(a.device,a.start_ms,a.duration_ms)
