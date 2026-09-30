"""Remove count innovations alone, preserving the actual private-Q rate model.

This conditional drift is explicitly distinct from the full-Q expected-rate
model. No automatic change of the accepted bifurcation object is made here.
"""
from common import OUT,np,read,write,log
from fine_rate_frozen_Z_fields import DEST as SOURCE, ForcedEngine, capture, restore, native_field
from refractory_spatial_resolution import projections
from native_readouts import readouts
from datetime import datetime
import argparse,os,time

DEST=OUT/'conditional_drift_Z_fields'


def register():
    assert read(SOURCE/'independent_comparison.json')['status']=='READOUT_AUDIT_PASS'
    assert read(OUT/'rate_deterministic_object_audit/result.json')['status']=='READ_ONLY_OBJECT_AUDIT_COMPLETE'
    DEST.mkdir(exist_ok=True);assert not (DEST/'contract.json').exists()
    write(DEST/'contract.json',dict(created_local=datetime.now().astimezone().isoformat(),
        question='Does the observed nativeZfield ordering survive removal of count innovations alone, with the SAME private diffusion operators and shared refractory history?',
        model='Current conditioned39 g40 finite-count rate model; construct with private-Q split, then replace each Binomial count by its conditional mean. Physicalgraph,thresholds,delays,responseweights,externaldrive andZM laws unchanged.',
        mathematical_object='Private-Q deterministic conditional drift under recorded external forcing, NOT existing full-Q expected rate, NOT exact ensemble mean and NOT an autonomous constant-input bifurcation system.',
        initial='Use saved own-rate full9000ms checkpoint without replay. Same complete fast/covariance/refractory/M history and future9-12.5s original external forcing as completedcount interventions.',
        arms=['native_Z9000_held','native_Z9420_held','native_Z9870_held'],duration_ms=3500.,dt_ms=.05,
        control='Completed count runs atsamefields reused exactly. First10ms count prefixes must replay bitwise before intervention implementation accepted. Atcommoninitialstate count and drift conditionalexpectedrate/localupdate must agree; only emission and subsequentcount-drivenM differ.',
        innovation_change='Keep private operators and share local/transport history; set count sampling off BEFORE CUDAgraph capture. Expected future emission drives the same refractory,delayandM equations. No changes to variance allocation.',
        Z='Entire prescribed native spatialfield fixed; originalgroup cellcountweightedD retained, I Z1.',
        M='Dynamic in all conditions, same actual9000ms initialvalue.',
        readout='Same >=200Hz for200ms entry, <5Hz quiet, completeevents with20msquietbefore/after, final1srate andspatialpersistentfraction. Originalthreecountarms arepairedcontrols, notindependentreplicates.',
        interpretation='Preservedordering supports analyzing this conditionaldrift afterconstant-input andequationchecks; alteredordering shows countinnovations matter tothisboundary and forbids transferringfull-Q criticalpoints. Censoring andonehistory limit remain.',
        budget='Three3500ms continuations. Noresponsefit, parametersearch, newseed, extendedobservationorbranchsearch. Readouts and numericalidentity checks do not waive originalmodelacceptancefailures.'))


def prepare(e,state,tm):
    restore(e,state);z=native_field(e.s,tm)
    e.syn[5]=e.cp.asarray(z);e.transport.pars[19].fill(0)
    e.cp.cuda.get_current_stream().synchronize()
    assert np.array_equal(e.syn[4].get(),state['syn'][4])
    assert np.array_equal(e.transport.history.get(),state['history'])
    return z


def check(device):
    assert (DEST/'contract.json').exists();state=np.load(SOURCE/'checkpoint9000.npz')
    e=ForcedEngine(noise=True,seed=1,device=device);e.graph();cp=e.cp;s=e.s
    operators=[a.get() for a in e.transport.ops];rows=[]
    for tm in [9000,9420,9870]:
        e.noise=True;e.graph();z=prepare(e,state,tm)
        reference=np.load(SOURCE/f'native_Z{tm}_held/trajectory.npz');first=e.chunk()
        assert np.array_equal(first[:,0].astype('f4'),reference['group_rate_hz'][:10])
        assert np.array_equal(first[:,1].astype('f4'),reference['group_expected_rate_hz'][:10])
        prepare(e,state,tm);e.step();cp.cuda.get_current_stream().synchronize()
        expected=e.local.rate.get();local=e.local.state.get();physical=e.local.physical.get();sample=e.emitted.get()
        e.noise=False;e.graph();prepare(e,state,tm);e.step();cp.cuda.get_current_stream().synchronize()
        assert np.array_equal(expected,e.local.rate.get())
        assert np.array_equal(local,e.local.state.get()) and np.array_equal(physical,e.local.physical.get())
        assert np.array_equal(expected,e.emitted.get())
        assert e.local.history.data.ptr==e.transport.history.data.ptr
        wanted_m=np.exp(-e.dt/1000)*state['syn'][4]+(1-np.exp(-e.dt/1000))*.5*s.E*expected
        m_error=float(np.max(abs(wanted_m-e.syn[4].get())));assert m_error<1e-12
        assert all(np.array_equal(a,b.get()) for a,b in zip(operators,e.transport.ops))
        assert np.array_equal(e.syn[5].get(),z)
        rows.append(dict(native_Z_time_ms=tm,first10mscount_bitwise=True,
            first_step_conditional_rate_bitwise=True,private_operators_unchanged=True,
            count_changed_groups=int(np.count_nonzero(sample!=expected)),M_expected_update_error=m_error))
    write(DEST/'implementation_check.json',dict(status='PASS',rows=rows,private_split=e.split_qa,
        object='Sampling removed only; private variance and shared self-history retained.',
        native_future_spikes_used=False,model_promoted=False))
    log('PRIVATE DRIFT IMPLEMENTATION PASS',rows)


def run(device):
    c=read(DEST/'contract.json');assert read(DEST/'implementation_check.json')['status']=='PASS'
    assert not (DEST/'jobs.json').exists();jobs=dict(status='RUNNING',pid=os.getpid(),expected=3,completed=[])
    write(DEST/'jobs.json',jobs)
    state=np.load(SOURCE/'checkpoint9000.npz');e=ForcedEngine(noise=True,seed=1,device=device)
    e.noise=False;e.graph();s=e.s;P,count=projections(s,e.coarse,e.parent)[20]
    for label in c['arms']:
        tm=int(label.split('_')[1][1:]);folder=DEST/label;folder.mkdir();z=prepare(e,state,tm)
        write(folder/'initial.json',dict(initial_D=float(1-z[s.E]@s.mean_weights),source=str(SOURCE/'checkpoint9000.npz'),
            Z_field_source_ms=tm,history_clock_ms=9000,private_diffusion=True,count_sampling=False,M_dynamic=True))
        jobs.update(stage=label,current_arm_ms=0);write(DEST/'jobs.json',jobs)
        R=[];Z=[];M=[];start=time.time()
        for k in range(350):
            x=e.chunk();assert np.isfinite(x).all() and x.min()>=-1e-9
            assert np.array_equal(x[:,0],x[:,1])
            R.append(x);Z.append(e.syn[5].get());M.append(e.syn[4].get());assert np.array_equal(Z[-1],z)
            if (k+1)%100==0:
                jobs['current_arm_ms']=(k+1)*10;write(DEST/'jobs.json',jobs)
                log('PRIVATE DRIFT',label,(k+1)*10,'ms seconds',round(time.time()-start,1))
        r=np.concatenate(R);zs=np.array(Z);m=np.array(M);field=(P@r[:,0].T).T
        t=np.arange(9001,12501.);ts=np.arange(9010,12501.,10);D=1-zs[:,s.E]@s.mean_weights
        events,summary,whole,sm=readouts(t,field,count,label);complete=[]
        for ev in events:
            a=int(np.searchsorted(t,ev['start_ms']));b=a+int(ev['duration_ms'])
            if a>=20 and b+20<=len(sm) and (sm[a-20:a]<5).all() and (sm[b:b+20]<5).all():
                complete.append({k:v for k,v in ev.items() if k!='onset'})
        np.savez_compressed(folder/'trajectory.npz',time_ms=t,state_time_ms=ts,group_rate_hz=r[:,0].astype('f4'),
            group_expected_rate_hz=r[:,1].astype('f4'),field_E_hz=field.astype('f4'),global_E_hz=whole,
            Z=zs.astype('f4'),M_current=m.astype('f4'),D=D,cell_counts=count,parent_g20=e.parent,
            final_synaptic_slow_state=e.syn.get(),final_local_state=e.local.state.get(),
            final_own_history=e.local.history.get(),final_emitted_history=e.transport.history.get(),final_tick=e.local.clock.get(),dt_ms=e.dt)
        summary.update(status='COMPLETE',complete_events=complete,initial_D=float(D[0]),M_dynamic=True,Z_dynamic=False,
            tail_global_hz=float(whole[-1000:].mean()),tail_quiet_fraction=float((sm[-1000:]<5).mean()),
            tail_persistent_fraction=float((count/count.sum())[(field[-1000:]>50).mean(0)>=.9].sum()),
            conditional_drift_private_Q=True,count_sampling=False,autonomous_constant_input=False,model_promoted=False,bifurcation_type='NOT_ESTABLISHED')
        write(folder/'result.json',summary);jobs['completed'].append(label);write(DEST/'jobs.json',jobs)
        log('PRIVATE DRIFT COMPLETE',label,summary['high_onset_ms'],len(complete),summary['tail_global_hz'])
    jobs.update(status='COMPLETE',stage='all_arms_complete',current_arm_ms=3500);write(DEST/'jobs.json',jobs)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','check','run']);p.add_argument('--device',type=int,default=1);a=p.parse_args()
    {'register':register,'check':lambda:check(a.device),'run':lambda:run(a.device)}[a.command]()
