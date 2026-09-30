"""Same-history conditional Z-field test of the current spatial rate model.

Restores an independently verified own-model 9s state. Future external drive,
random innovation keys and dynamic M are shared. No new response fit and no
bifurcation classification are performed by this finite-time experiment.
"""
from common import OUT, BASE, ROOT, np, read, write, log
from refractory_fine_forcing_pair import ForcedEngine
from refractory_spatial_resolution import projections
from native_readouts import readouts
from scipy.ndimage import uniform_filter1d
from datetime import datetime
import argparse, os, time

DEST=OUT/'fine_rate_frozen_Z_fields'
SOURCE=OUT/'conditioned_refractory_fine_forcing/recorded_drive_binomial_seed1'
CHECKPOINTS=ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/runs/eta0.0005_s9108401/checkpoints'


def arrays(e):
    assert e.local.history.data.ptr == e.transport.history.data.ptr
    return dict(syn=e.syn,local=e.local.state,history=e.transport.history,
        clock=e.local.clock,rate=e.local.rate,physical=e.local.physical,emitted=e.emitted,
        accumulator=e.accumulator,output=e.output,arrivals=e.transport.arr,parameters=e.transport.pars)


def capture(e):
    e.stream.synchronize()
    return {k:v.get() for k,v in arrays(e).items()}


def restore(e, state):
    e.stream.synchronize()
    for k,v in arrays(e).items(): v[:] = e.cp.asarray(state[k])
    e.cp.cuda.get_current_stream().synchronize()


def native_field(s, tm):
    raw=np.load(CHECKPOINTS/f't{tm}ms.npz')['slow__z'][:32000]
    ids=s.geo['cell_group'][:32000]; count=np.bincount(ids,minlength=s.P)
    z=np.ones(s.P); z[s.E]=(np.bincount(ids,weights=raw,minlength=s.P)/np.maximum(count,1))[s.E]
    assert np.array_equal(count[s.E],s.sizes[s.E])
    assert abs(1-z[s.E]@s.mean_weights-read(BASE/'native_reference/checkpoint_projections.json')[str(tm)]['D'])<1e-12
    return z


def register():
    assert read(OUT/'conditioned_refractory_fine_forcing/jobs.json')['status']=='COMPLETE'
    a=read(OUT/'conditioned_refractory_fine_forcing/scientific_comparison.json')
    row=next(x for x in a['rows'] if x['label']=='recorded_drive_binomial_seed1_fine_forcing')
    assert row['original_six_checks']['propagation'] and not row['original_six_checks']['D_track']
    DEST.mkdir(exist_ok=True); assert not (DEST/'contract.json').exists()
    write(DEST/'contract.json',dict(created_local=datetime.now().astimezone().isoformat(),status='REGISTERED_BEFORE_CONDITIONAL_FIELD_RUNS',
        question='With bidirectional propagation restored, is the remaining error primarily the free Z depletion path or also the conditional fast-network response at the actual native spatial Z fields?',
        model='Unchanged conditioned39 refractory-rate response; same0.5mm graph, thresholds, delays, actual-member external field and Binomial counts. Failed voltage-memory candidate is NOT used.',
        initial='Replay the completed count arm from0to9000ms; verify every stored1ms count/conditional-rate sample and10ms Z/M sample at original float32 precision. Save the full own-model fast, synaptic, history, Z/M and clock state.',
        arms=['dynamic_reference','own_Z9000_held','native_Z9000_held','native_Z9420_held','native_Z9870_held'],
        intervention='All forks start from exactly the same saved rate-model9s state. Held arms freeze the entire projected E Z field; I Z stays1. M starts identically and remains dynamic in every arm. Z means are coordinates, not substitutes for spatial fields.',
        input='Same external forcing9-12.5s and same seed1 Philox(group,tick) innovation keys. Actual output counts differ with conditional probabilities, not identical spikes.',
        clock='dt0.05ms; each arm9-12.5s,10ms fullslow samples and1ms rates. Dynamic reference must reproduce the completed trajectory including full final states.',
        readout='Original global10ms-smoothed quiet<5Hz, complete events with20ms quiet before/after, >=200Hz for200ms entry, tail rate and spatial persistent fraction (>50Hz in>=90percent of last1s). Report both-core order and all fields; no new scalar bifurcation threshold.',
        interpretation='If native preentry fields return to self-limited activity and later field sustains, conditional correspondence is supported in this finite window even if free Z clock differs. If not, fast conditional response also fails. Mixed/censored outcomes remain unresolved, never converted into a bifurcation name.',
        comparability='One own-rate history; native late clamps have native histories at their own clamp times. This is NOT identical fast-state transplantation across model classes and does not establish basin completeness or universal necessity/sufficiency.',
        budget='One9s checkpoint replay plus5fixed3.5s arms,26.5s total. No extra seed, extension, newfit, network parameter change, modelpromotion or continuation.',
        prerequisite='Completed original six-gate comparison remains5/6; broad local-response failures remain. This is a discriminating correspondence test before mathematical onset attribution, not an acceptance waiver.'))


def check(device):
    assert (DEST/'contract.json').exists(); e=ForcedEngine(noise=True,seed=1,device=device); e.graph(); s=e.s
    source=np.load(SOURCE/'trajectory.npz'); records=[]
    for _ in range(10): records.append(e.chunk())
    r=np.concatenate(records)
    assert np.array_equal(r[:,0].astype('f4'),source['group_rate_hz'][:100])
    assert np.array_equal(r[:,1].astype('f4'),source['group_expected_rate_hz'][:100])
    state=capture(e); a=e.chunk(); terminal=capture(e); restore(e,state); b=e.chunk()
    assert np.array_equal(a,b)
    assert all(np.array_equal(v,arrays(e)[k].get()) for k,v in terminal.items())
    rows=[]
    for tm in [9000,9420,9870]:
        restore(e,state); z=native_field(s,tm); e.syn[5]=e.cp.asarray(z); e.transport.pars[19].fill(0)
        e.cp.cuda.get_current_stream().synchronize()
        assert np.array_equal(e.syn[4].get(),state['syn'][4])
        assert np.array_equal(e.transport.history.get(),state['history'])
        e.chunk(); assert np.array_equal(e.syn[5].get(),z)
        assert np.array_equal(e.transport.pars[20].get(),state['parameters'][20])
        rows.append(dict(native_time_ms=tm,D=float(1-z[s.E]@s.mean_weights),Z_held_bitwise=True,M_initial_unchanged=True))
    write(DEST/'implementation_check.json',dict(status='PASS',baseline100ms_recorded_precision_bitwise=True,
        replayed10ms_full_state_bitwise=True,rows=rows,scope='Implementation only; toy-prefix clamps are not scientific conditional results.'))
    log('CONDITIONAL FIELD IMPLEMENTATION PASS',rows)


def run(device):
    c=read(DEST/'contract.json'); assert read(DEST/'implementation_check.json')['status']=='PASS'
    assert not (DEST/'jobs.json').exists()
    jobs=dict(status='RUNNING',pid=os.getpid(),stage='checkpoint_replay',expected=5,completed=[])
    write(DEST/'jobs.json',jobs); e=ForcedEngine(noise=True,seed=1,device=device); e.graph(); s=e.s
    source=np.load(SOURCE/'trajectory.npz')
    oldr=source['group_rate_hz']; olde=source['group_expected_rate_hz']; oldz=source['Z']; oldm=source['M_current']
    start=time.time()
    for k in range(900):
        r=e.chunk()
        assert np.array_equal(r[:,0].astype('f4'),oldr[10*k:10*k+10])
        assert np.array_equal(r[:,1].astype('f4'),olde[10*k:10*k+10])
        assert np.array_equal(e.syn[5].get().astype('f4'),oldz[k])
        assert np.array_equal(e.syn[4].get().astype('f4'),oldm[k])
        if (k+1)%100==0:
            jobs['replayed_ms']=(k+1)*10; write(DEST/'jobs.json',jobs)
            log('CONDITIONAL FIELD REPLAY',(k+1)*10,'ms','seconds',round(time.time()-start,1))
    state=capture(e); np.savez_compressed(DEST/'checkpoint9000.npz',**state)
    write(DEST/'replay_qa.json',dict(status='PASS',full_prefix_ms=9000,rate_and_slow_saved_precision_bitwise=True,
        tick=int(state['clock'][0]),source=str(SOURCE/'trajectory.npz')))
    projection=projections(s,e.coarse,e.parent)[20]; P,count=projection
    for label in c['arms']:
        folder=DEST/label; folder.mkdir(); restore(e,state)
        if label!='dynamic_reference':
            z=state['syn'][5].copy() if label=='own_Z9000_held' else native_field(s,int(label.split('_')[1][1:]))
            e.syn[5]=e.cp.asarray(z); e.transport.pars[19].fill(0); e.cp.cuda.get_current_stream().synchronize()
        else: z=state['syn'][5].copy()
        assert np.array_equal(e.syn[4].get(),state['syn'][4])
        assert np.array_equal(e.transport.history.get(),state['history'])
        assert np.array_equal(e.local.state.get(),state['local'])
        write(folder/'initial.json',dict(clock_ms=9000,initial_D=float(1-z[s.E]@s.mean_weights),
            M_dynamic=True,Z_dynamic=label=='dynamic_reference',same_fast_history=True,same_future_innovation_keys=True))
        jobs.update(stage=label,current_arm_ms=0); write(DEST/'jobs.json',jobs)
        R=[]; Z=[]; M=[]; start=time.time()
        for k in range(350):
            x=e.chunk(); assert np.isfinite(x).all() and x[:,0].min()>=0
            zz=e.syn[5].get(); mm=e.syn[4].get()
            if label=='dynamic_reference':
                assert np.array_equal(x[:,0].astype('f4'),oldr[9000+10*k:9010+10*k])
                assert np.array_equal(x[:,1].astype('f4'),olde[9000+10*k:9010+10*k])
                assert np.array_equal(zz.astype('f4'),oldz[900+k])
                assert np.array_equal(mm.astype('f4'),oldm[900+k])
            else: assert np.array_equal(zz,z)
            R.append(x); Z.append(zz); M.append(mm)
            if (k+1)%100==0:
                jobs['current_arm_ms']=(k+1)*10; write(DEST/'jobs.json',jobs)
                log('CONDITIONAL FIELD',label,(k+1)*10,'ms','seconds',round(time.time()-start,1))
        if label=='dynamic_reference':
            for actual,key in [(e.syn,'final_synaptic_slow_state'),(e.local.state,'final_local_state'),
                (e.local.history,'final_own_history'),(e.transport.history,'final_emitted_history'),(e.local.clock,'final_tick')]:
                assert np.array_equal(actual.get(),source[key]),key
        r=np.concatenate(R); zslow=np.array(Z); m=np.array(M); field=(P@r[:,0].T).T
        t=np.arange(9001,12501.); ts=np.arange(9010,12501.,10); D=1-zslow[:,s.E]@s.mean_weights
        events,summary,whole,sm=readouts(t,field,count,label)
        complete=[]
        for ev in events:
            a=int(np.searchsorted(t,ev['start_ms'])); b=a+int(ev['duration_ms'])
            if a>=20 and b+20<=len(sm) and (sm[a-20:a]<5).all() and (sm[b:b+20]<5).all():
                complete.append({k:v for k,v in ev.items() if k!='onset'})
        np.savez_compressed(folder/'trajectory.npz',time_ms=t,state_time_ms=ts,group_rate_hz=r[:,0].astype('f4'),
            group_expected_rate_hz=r[:,1].astype('f4'),field_E_hz=field.astype('f4'),global_E_hz=whole,
            Z=zslow.astype('f4'),M_current=m.astype('f4'),D=D,cell_counts=count,parent_g20=e.parent,
            final_synaptic_slow_state=e.syn.get(),final_local_state=e.local.state.get(),
            final_own_history=e.local.history.get(),final_emitted_history=e.transport.history.get(),final_tick=e.local.clock.get(),dt_ms=e.dt)
        summary.update(status='COMPLETE',complete_events=complete,M_dynamic=True,Z_dynamic=label=='dynamic_reference',
            tail_global_hz=float(whole[-1000:].mean()),tail_quiet_fraction=float(np.mean(sm[-1000:]<5)),
            tail_persistent_fraction=float((count/count.sum())[(field[-1000:]>50).mean(0)>=.9].sum()),
            own_rate_history='common saved9000ms state',initial_D=read(folder/'initial.json')['initial_D'],
            model_promoted=False,bifurcation_type='NOT_ESTABLISHED')
        write(folder/'result.json',summary); jobs['completed'].append(label); write(DEST/'jobs.json',jobs)
        log('CONDITIONAL FIELD COMPLETE',label,summary['high_onset_ms'],len(complete),summary['tail_global_hz'])
    jobs.update(status='COMPLETE',stage='all_fixed_arms_complete'); write(DEST/'jobs.json',jobs)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['register','check','run']); p.add_argument('--device',type=int,default=0); a=p.parse_args()
    {'register':register,'check':lambda:check(a.device),'run':lambda:run(a.device)}[a.command]()
