"""Full-state replay of an explicitly selected interictal recurrence."""
from common import OUT,np,read,write,log
from onset_state_continuation import build,regional_weights
from fine_rate_frozen_Z_fields import capture,restore
from onset_period_return import dynamical_state,errors
import argparse,os,time
from pathlib import Path

BASE=OUT/'core_a_bifurcation_type_20260924/reference_stability_gap/actual_D0270'
DEST=BASE/'recurrence/exact_nine_burst_seed'


def main(device,destination=None,stored_precision=False,source_base=None,screen_path=None,burst_count=9,segments=6,diagnostic_only=False):
    global DEST
    if destination is not None:DEST=Path(destination).resolve()
    base_path=Path(source_base).resolve() if source_base else BASE
    screen_file=Path(screen_path).resolve() if screen_path else base_path/'recurrence/result.json'
    screen=read(screen_file)
    candidate=next(r for r in screen['best_by_burst_count'] if r['burst_count']==burst_count)
    source=base_path/'from_interictal_history';audit=read(source/'whole_record_audit.json');assert audit['status']=='AUDIT_PASS'
    dt=.05;K=segments;first=round(candidate['time1_ms']/dt)*dt;last=round(candidate['time2_ms']/dt)*dt
    origin=int(first//5000)*5000;checkpoint_end=int(np.ceil(last/5000))*5000
    assert 0<origin<first<last<=audit['observed_ms']
    times=np.rint(np.linspace(first,last,K+1)/dt)*dt
    DEST.mkdir(exist_ok=True);assert not(DEST/'jobs.json').exists()
    write(DEST/'contract.json',dict(candidate=candidate,screen=str(screen_file),coordinates=audit['coordinates'],segments=K,dt_ms=dt,device=device,source=str(source/f'checkpoint{origin}.npz'),times_ms=times.tolist(),diagnostic_only=diagnostic_only,
        diagnostic_scope='When diagnostic_only is true, preserve all recorded differences and complete the original checkpoint comparison; never accept these nodes as a cycle seed or silently pass a failed replay gate.',
        question='Does the actual short-event spatial trajectory admit its observed multi-burst full-state periodic candidate, rather than the unstable one-burst root?',
        selection=f'Explicitly selected {burst_count}-burst candidate from the stored all-group history/M screen. Longer near-returns are not declared distinct periods before checking this candidate.',
        equations='Same original3479-group drift, frozen originalCoreAfield recorded in coordinates andnative9s outsideA, allE M dynamic. No parameter, response or futureinput changes.',
        numerical='K complete-state nodes partition only the shooting equation, not the physical model. All nodes are nearest actual.05ms states; subsequent matching removes segment timing offsets.',
        acceptance=(f'Numerical replay gate: every savedfloat32 rate within one float32 ULP plus1e-12Hz; continue to original{checkpoint_end}ms checkpoint and require original six-block full-float64 state agreement combined<1e-10, eachblock<1e-9. Report exact bitwise differences separately. This is a seed-source check, not a relaxation of periodic closure, stability, phase or mesh gates.' if stored_precision else 'Every replayed1ms all-group rate bitwise matches the original auditedrecord; report full endpoint closure. Recurrence is not a cycle or bifurcation certificate.'),
        reason_for_precision_check=('The earlier D_A=.270 replay differed by onefloat32 ULP at one0.000188Hz sample; independentGPU0/GPU1 andCPUthread1/2 replays agreed bitwise with eachother. That failed attempt is retained. Apply the separately declared complete-float64 checkpoint comparison to this source; no physical or periodic acceptance gate changes.' if stored_precision else None),model_promoted=False))
    write(DEST/'jobs.json',dict(status='RUNNING',pid=os.getpid()))
    e=build(device);base=dict(np.load(source/f'checkpoint{origin}.npz'));restore(e,base)
    clock0=int(base['clock'][0]);targets={clock0+round((tm-origin)/dt):j for j,tm in enumerate(times)}
    expected=np.concatenate([np.load(source/f'block{j:02d}.npz')['group_rate_hz'] for j in range(origin//5000,checkpoint_end//5000)])
    states={};begin=time.time();replay_rows=[]
    stop=checkpoint_end if stored_precision else int(np.ceil(last/10))*10
    for tm in range(origin+10,stop+1,10):
        now=int(e.local.clock.get()[0]);end=now+round(10/dt)
        if any(now<tick<=end for tick in targets):
            for tick in range(now+1,end+1):
                e.step()
                if tick in targets:
                    e.cp.cuda.get_current_stream().synchronize();state=capture(e)
                    assert int(state['clock'][0])==tick and np.array_equal(state['syn'][5],base['syn'][5])
                    j=targets[tick];np.savez_compressed(DEST/f'node{j:02d}.npz',**state);states[j]=state
            e.cp.cuda.get_current_stream().synchronize();r=e.output.get()
        else:r=e.chunk()
        offset=tm-origin
        ref=expected[offset-10:offset];q=r[:,0].astype('f4');exact=np.array_equal(q,ref)
        delta=abs(q.astype(float)-ref.astype(float));bound=np.spacing(ref).astype(float)+1e-12
        replay_rows.append(dict(elapsed_ms=tm,bitwise=bool(exact),differing=int(np.count_nonzero(q!=ref)),max_abs_hz=float(delta.max()),within_stored_precision=bool(np.all(delta<=bound))))
        if not diagnostic_only and not (exact or (stored_precision and np.all(delta<=bound))):
            np.savez_compressed(DEST/'replay_difference.npz',actual=r,expected=ref,elapsed_ms=tm)
            failure=dict(status='FAILED',pid=os.getpid(),elapsed_ms=tm,
                max_output_difference_hz=float(np.max(abs(r[:,0].astype('f4')-ref))),
                differing_recorded_values=int(np.count_nonzero(r[:,0].astype('f4')!=ref)),
                error='Original recorded-precision replay equality failed; no periodic seed accepted.')
            write(DEST/'jobs.json',failure)
            raise AssertionError(failure)
        if tm%1000==0:log('INTERICTAL FULL STATE REPLAY',burst_count,tm,round(time.time()-begin,1))
    assert len(states)==K+1
    er=errors(dynamical_state(states[0]),dynamical_state(states[K]),e.s.sizes/e.s.sizes.sum())
    checkpoint_check=None
    if stored_precision:
        terminal=capture(e);original=dict(np.load(source/f'checkpoint{checkpoint_end}.npz'))
        checkpoint_check=errors(dynamical_state(original),dynamical_state(terminal),e.s.sizes/e.s.sizes.sum())
        write(DEST/'original_checkpoint_comparison.json',checkpoint_check)
        assert np.array_equal(original['syn'][5],terminal['syn'][5]) and np.array_equal(original['clock'],terminal['clock'])
        if not diagnostic_only:
            assert checkpoint_check['combined_relative_rms']<1e-10 and max(v['relative_rms'] for v in checkpoint_check['blocks'].values())<1e-9,checkpoint_check
    all_bitwise=all(r['bitwise'] for r in replay_rows)
    write(DEST/'replay_comparison.json',dict(rows=replay_rows,all_bitwise=all_bitwise,complete_checkpoint=checkpoint_check))
    rate=expected[int(first-origin):int(last-origin)].astype(float)@regional_weights(e.s).T
    write(DEST/'result.json',dict(status='REPLAY_DIFFERENCE_DIAGNOSTIC_ONLY' if diagnostic_only else ('EXACT_REPLAY_PASS_RECURRENCE_ONLY' if all_bitwise else 'REPLAY_AGREEMENT_PASS_RECURRENCE_ONLY'),segments=K,period_ms=last-first,times_ms=times.tolist(),full_state_errors=er,
        all_original_1ms_group_rates_bitwise=all_bitwise,original_checkpoint_comparison=checkpoint_check,all_Z_held=True,all_M_dynamic=True,mean_global_A_B_surround_hz=rate.mean(0).tolist(),seconds=time.time()-begin,model_promoted=False))
    write(DEST/'jobs.json',dict(status='COMPLETE',pid=os.getpid()));log('INTERICTAL FULL STATE SEED',burst_count,er)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0);p.add_argument('--destination')
    p.add_argument('--stored-precision',action='store_true')
    p.add_argument('--source-base');p.add_argument('--screen-path');p.add_argument('--burst-count',type=int,default=9)
    p.add_argument('--segments',type=int,choices=[4,6,9,12,18],default=6)
    p.add_argument('--diagnostic-only',action='store_true')
    a=p.parse_args();main(a.device,a.destination,a.stored_precision,a.source_base,a.screen_path,a.burst_count,a.segments,a.diagnostic_only)
