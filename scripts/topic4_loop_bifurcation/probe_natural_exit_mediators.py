#!/usr/bin/env python3
"""One verified q-off state and two bounded native mediator interventions."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,copy,shutil,subprocess,time
from pathlib import Path
import numpy as np
from campaign import ROOT,NATIVE,REPO,PYTHON,read,write,sha
import run_topic4_loop_zk_conditional as native
import run_topic4_loop_cuda_override as gpu
from run_topic4_recovery_window import assert_same_state
from native_campaign import STREAMS

OUT=ROOT/'natural_exit_mediator_probes'
INITIAL=ROOT/'exit_state_reconstruction/runs/source10_to16p70/checkpoint.pkl'
REFERENCE=native.SOURCE/'runs'/native.NAME
RECON='unchanged16p7_to16p8'
VERIFY='unchanged16p8_to20'
CONTROLS=['remove_existing_G_tail_once','remove_low_rate_K_retention']


def configure():
    native.OUT=OUT;native.prepare=lambda:read(OUT/'protocol.json')


def prepare():
    OUT.mkdir(exist_ok=True);assert not (OUT/'contract.json').exists()
    assert read(ROOT/'exit_state_continuation_qa/gate.json')['status']=='PASS'
    obs=np.load(ROOT/'native_exit_input_observation/inputs.npz');t=obs['time_ms']/1000;R=obs['global_R_and_s'][:,0]
    i=np.flatnonzero(R<=200)[0];target=np.ceil((t[i]+1e-9)/.02)*.02
    assert target==16.8 and t[i]==16.799
    write(OUT/'contract.json',dict(status='REGISTERED_BEFORE_NATURAL_EXIT_MEDIATOR_PROBES',created_epoch=time.time(),
        question='At the actual firstq-off state, is remainingG memory needed to finishsuppression, and is slowlow-rateK decay needed to provide adequate coreZ recovery time?',
        motivation='StaticheldZ/K highstates relaxG near0, whereas naturalexit retainsG near12. ExistingpairedtauG trajectories show both earlybuildup changes and q-offcarryover; seed8403 has nearlyequalK at a matchedclocktime yetdifferentactivity. Do not continue refining a staticroot before checking its connection to the actualexit mechanism.',
        selection='First20msboundary after q firstcloses in the independently verified16.7-20s originalrecord:16.8s, causalR194.911Hz,Graw11.9675. This is beforefirstR<=5 at16.868s, not the20s retrospectiveexitcheckpoint.',
        state_validation='Unchanged16.7->16.8 reconstruction, exactobservations versusoriginal; then unchanged16.8->20 reachesoriginal20s completeengine bitwise. Only after both pass dispatch interventions.',
        design='Exactlytwo10s nativeinterventions fromthe sameverified16.8s fullstate, futureRNG/OU/drive unchanged. Control1 sets existingglobal_state to0 once, then originalGdynamics continues andmayregenerateG. Control2 preservesinitialstate butuses0.5s instead of5s Kdecay whenR<=5; allother originaldynamics unchanged. Z/K/M are neverclamped andno timer imposesquietactivity.',
        readouts='Originalpairedbaseline16.8-26.8s reused. CausalR,globalG,K,wholeE/coreZlevels andnetZbudgets, firstsustainedlowinterval, Gresourceunblocking, subsequentreactivation andZreferenceattainment. No criterionbasedonlyonwholeEZ.',
        decisions='Gremovalwithcontinuedquiet wouldshowthatGtail atthisspecificstate is not necessary tofinishsuppression; reboundwouldsupportitslocalnecessity. FastKdecaypreventingcoreZ recoverywouldsupportthe retentionmechanism. A null effectwouldrestrict thatclaim tootherstates or mechanisms. Two single-state interventions do not establish universalmediators or formalbifurcation.',
        stop='Two10s branches only after replayQA. No automatic newseed, parametergrid, horizonextension orautonomous-loopcount.',
        producer_sha256=sha(__file__),initial_checkpoint=str(INITIAL),initial_sha256=sha(INITIAL),
        counts_as_autonomous_loop=False,formal_bifurcation_allowed=False))
    protocol=copy.deepcopy(read(NATIVE/'protocol.json'))
    protocol.update(stage='NATURAL_EXIT_MEDIATOR_INTERVENTIONS',created_epoch=time.time(),deadline_epoch=time.time()+86400)
    write(OUT/'protocol.json',protocol);shutil.copy2(NATIVE/'geometry.npz',OUT/'geometry.npz')
    configure();native.make_job(RECON,str(INITIAL),.1,clamp=False,common_input=False)
    shutil.copy2(__file__,OUT/'producer.py')


def check_observations(name):
    folder=OUT/'runs'/name;checks=[]
    assert read(folder/'result.json')['status']=='COMPLETE'
    for stream in STREAMS+['z_budget_chunks']:
        if not (REFERENCE/stream).exists():continue
        for path in sorted((folder/stream).glob('*.npz')):
            start,end=map(int,path.stem.split('_'))
            matches=[q for q in (REFERENCE/stream).glob('*.npz') if int(q.stem.split('_')[0])<=start and int(q.stem.split('_')[1])>=end]
            assert len(matches)==1,(stream,path.name)
            original=matches[0];a,b=map(int,original.stem.split('_'))
            with np.load(path) as x,np.load(original) as y:
                assert set(x.files)==set(y.files),(stream,path.name)
                for key in x.files:
                    if key in ['start_step','end_step']:
                        assert int(x[key])==(start if key=='start_step' else end);continue
                    if key in ['keys','region_names','variables']:want=y[key]
                    else:
                        stride=(b-a)//len(y[key]);assert stride*len(y[key])==b-a
                        assert (start-a)%stride==0 and (end-a)%stride==0
                        want=y[key][(start-a)//stride:(end-a)//stride]
                    assert np.array_equal(x[key],want,equal_nan=x[key].dtype.kind in 'fc'),(stream,path.name,key)
                    checks.append(dict(stream=stream,key=key,bitwise=True))
    assert checks
    write(OUT/f'{name}_observation_qa.json',dict(status='PASS',checks=checks))


def prepare_controls():
    configure();source=OUT/'runs'/RECON/'checkpoint.pkl';old=native.read_pickle(source)
    for name in CONTROLS:
        job=native.make_job(name,str(source),10.,clamp=False,common_input=False)
        folder=OUT/'runs'/name;saved=native.read_pickle(folder/'checkpoint.pkl');expected=copy.deepcopy(old['engine'])
        job.update(diagnostic_intervention=name,counts_as_autonomous_loop=False,no_external_intervention=False)
        if name==CONTROLS[0]:
            expected['global_feedback_response']['global_state']=0.
            saved['engine']['global_feedback_response']['global_state']=0.
            job.update(initial_G_raw_override=0.,initial_G_raw_before=30*old['engine']['global_feedback_response']['global_state'])
        else:job.update(off_tau_s=.5,low_rate_K_decay_override_s=.5)
        saved['job']=job;assert_same_state(saved['engine'],expected)
        native.base.save_pickle(folder/'checkpoint.pkl',saved);write(OUT/'jobs'/f'{name}.json',job)
    write(OUT/'control_initial_state_qa.json',dict(status='PASS',complete_engine_changes_exactly_as_declared=True,
        paired_future_exogenous_state_exact=True,original_endogenous_Z_K_M_fields_carried=True))


def worker(name,device):
    assert sha(__file__)==read(OUT/'contract.json')['producer_sha256'];configure()
    if name==CONTROLS[1]:
        original=native.ConditionalSlow
        class NoRetention(original):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs)
                self.off_tau_ms=500.
        native.ConditionalSlow=NoRetention
    gpu.worker(OUT,name,device)
    if name in CONTROLS:
        folder=OUT/'runs'/name;result=read(folder/'result.json')
        result.update(no_external_intervention=False,diagnostic_only=True,counts_as_autonomous_loop=False,
            diagnostic_intervention=name,global_feedback_may_reactivate=True,Z_K_M_not_clamped=True)
        write(folder/'result.json',result);write(folder/'progress.json',result)


def dispatch(name,device):
    log=(OUT/f'{name}.log').open('w')
    process=subprocess.Popen([PYTHON,__file__,'worker','--name',name,'--device',str(device)],stdout=log,stderr=subprocess.STDOUT)
    return process,log


def supervise():
    assert not (OUT/'supervisor.json').exists();configure()
    def status(stage,**kw):write(OUT/'supervisor.json',dict(status=stage,pid=os.getpid(),updated_epoch=time.time(),**kw))
    try:
        p,log=dispatch(RECON,1);status('RECONSTRUCTING_QOFF_STATE',worker_pid=p.pid)
        code=p.wait();log.close();assert code==0
        check_observations(RECON)
        state=native.read_pickle(OUT/'runs'/RECON/'checkpoint.pkl')['engine'];assert state['step']==168000
        obs=np.load(ROOT/'native_exit_input_observation/inputs.npz');i=1000
        assert state['termination_mechanism']['r_global']==obs['global_R_and_s'][i,0]
        assert state['global_feedback_response']['global_state']==obs['global_R_and_s'][i,1]
        assert state['termination_mechanism']['r_global']<200
        write(OUT/'qoff_state.json',dict(time_s=16.8,causal_R_Hz=float(state['termination_mechanism']['r_global']),
            Graw=float(30*state['global_feedback_response']['global_state']),meanK=float(state['termination_mechanism']['sahp_g'].mean()),meanZ=float(state['slow']['z'][:32000].mean())))
        native.make_job(VERIFY,str(OUT/'runs'/RECON/'checkpoint.pkl'),3.2,clamp=False,common_input=False)
        p,log=dispatch(VERIFY,1);status('VERIFYING_FULL20S_STATE',worker_pid=p.pid)
        code=p.wait();log.close();assert code==0
        check_observations(VERIFY)
        a=native.read_pickle(OUT/'runs'/VERIFY/'checkpoint.pkl')['engine'];b=native.read_pickle(REFERENCE/'states/t20s.pkl')['engine']
        assert a['slow']['kind']=='ConditionalSlow' and b['slow']['kind']=='GlobalResponseSlow'
        a['slow']['kind']=b['slow']['kind'];assert_same_state(a,b)
        write(OUT/'full_state_gate.json',dict(status='PASS',complete20s_engine_bitwise=True,
            only_metadata_normalization='ConditionalSlow withoutclamp versusGlobalResponseSlow classname',
            initial16p8_state_sha256=sha(OUT/'runs'/RECON/'checkpoint.pkl')))
        prepare_controls();jobs=[dispatch(name,device) for device,name in enumerate(CONTROLS)]
        status('RUNNING_TWO_NATIVE_MEDIATOR_CONTROLS',workers=[dict(name=n,pid=p.pid) for n,(p,l) in zip(CONTROLS,jobs)])
        codes=[]
        for p,log in jobs:codes.append(p.wait());log.close()
        assert not any(codes),codes
        status('COMPLETE_TWO_NATIVE_MEDIATOR_CONTROLS_ANALYSIS_PENDING',controls=CONTROLS)
    except Exception as exc:
        status('FAILED',error=repr(exc));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','supervise','worker']);p.add_argument('--name');p.add_argument('--device',type=int,default=1);a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='supervise':supervise()
    else:worker(a.name,a.device)
