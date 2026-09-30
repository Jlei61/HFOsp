#!/usr/bin/env python3
"""Exact revised-Z state continuation30–60s; observation stop changes only."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import argparse,copy,hashlib,pickle,shutil,time
import run_topic4_continuous_resource_recovery as physical
carrier=physical.carrier
ROOT,PARENT=physical.ROOT,physical.PARENT
OUT=PARENT/'revisedZ_continuation_to60'
SOURCE=PARENT/'preserved_global_gain_round7'
SOURCE_NAME='resource_rho0.25_k200_tau10_s9108401'
NAME='revisedZ_k200_tau10_continuation60_s9108401'

def engine_hash(engine):
    return hashlib.sha256(pickle.dumps(engine,protocol=5)).hexdigest()

def checkpoint_policy(sink,target_step,deadline,ignored,now=time.time):
    def callback(step,state):
        try:return sink(step,state)
        except carrier.Stop:
            # Frozen sink saves first, then raises for QA, deadline or the
            # observation endpoint. QA is forbidden for this continuation.
            if now()>=deadline or step>=target_step:raise
            ignored.append(dict(step=int(step),time_s=step*.0001,
                reason='Second-entry observation stop reached; saved state retained, integration continues.'))
    return callback

def verify_stop_policy():
    saved=[];ignored=[]
    def sink(step,state):
        saved.append(step)
        raise carrier.Stop()
    checkpoint_policy(sink,600000,100,ignored,now=lambda:90)(400000,{})
    assert saved==[400000] and ignored[0]['step']==400000
    for step,clock in [(600000,90),(400000,100)]:
        try:checkpoint_policy(sink,600000,100,[],now=lambda:clock)(step,{})
        except carrier.Stop:pass
        else:raise AssertionError('Required horizon/deadline stop swallowed')
    def failure(step,state):raise RuntimeError('independent failure')
    try:checkpoint_policy(failure,600000,100,[],now=lambda:90)(400000,{})
    except RuntimeError:pass
    else:raise AssertionError('Non-stop failure swallowed')

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():return carrier.base.read(OUT/'protocol.json')
    result=carrier.base.read(SOURCE/'runs'/SOURCE_NAME/'result.json')
    assert result['end_s']==30. and len(result['tracker']['entries'])>=2
    source_checkpoint=SOURCE/'runs'/SOURCE_NAME/'checkpoint.pkl'
    saved=carrier.base.load_pickle(source_checkpoint)
    assert saved['engine']['step']==300000 and saved['job']['recovery_ratio']==.25
    assert not saved['job'].get('qa')
    before=engine_hash(saved['engine']);source_job=copy.deepcopy(saved['job'])
    job=copy.deepcopy(source_job);job.update(name=NAME,round=11)
    assert job['horizon_s']==60. and job['device']==0
    saved['job']=job;folder=OUT/'runs'/NAME;folder.mkdir(parents=True,exist_ok=True)
    for subdir in ['chunks','pool_chunks','resource_chunks']:
        target=folder/subdir;target.mkdir(exist_ok=True)
        for path in sorted((source_checkpoint.parent/subdir).glob('*.npz')):
            assert '.tmp.' not in path.name
            os.link(path,target/path.name)
    shutil.copy2(SOURCE/'geometry.npz',OUT/'geometry.npz')
    carrier.base.save_pickle(folder/'checkpoint.pkl',saved)
    restored=carrier.base.load_pickle(folder/'checkpoint.pkl')
    assert engine_hash(restored['engine'])==before
    verify_stop_policy()
    p=copy.deepcopy(carrier.base.read(SOURCE/'protocol.json'))
    p.update(round=11,created_at=time.time(),initial_jobs=[job],
        continuation_producer_sha256=carrier.base.sha(__file__),
        source_checkpoint=str(source_checkpoint),source_checkpoint_sha256=carrier.base.sha(source_checkpoint),
        source_initial_engine_sha256=before,source_job=source_job,
        extension='Same revisedZ rho.25 candidate: full engine/RNG/Z/M/Rg/delays carried from30s. No physical parameter change; observe through originally planned60s rather than stopping after second entry.',
        question='Do finite high/quiet episodes persist to60s, or does the rho.25 candidate also end in a persistent high platform?',
        control='Same trajectory continuation, not an independent noise/network replicate. Count one extra continuation conservatively against night budget; source remains unchanged.')
    carrier.base.write(OUT/'jobs'/(NAME+'.json'),job)
    carrier.base.write(OUT/'continuation_qa.json',dict(status='PASS',
        source_engine_and_copied_engine_pickle_sha256=before,whole_engine_unchanged=True,
        source_parameter_values_unchanged_except_job_name_and_round=True,
        source_prefix_hardlinked_readonly_by_usage=True,intermediate_checkpoint_stop_suppressed=True,
        wall_deadline_and_final_horizon_preserved=True,non_stop_failure_propagates=True,
        source_step=300000,target_step=600000))
    carrier.base.write(OUT/'protocol.json',p)
    return p

def worker():
    assert (OUT/'dispatch_authorization.json').exists()
    p=prepare();assert carrier.base.sha(__file__)==p['continuation_producer_sha256']
    ignored=[];original_wrap=carrier.wrap_simulator
    def wrap(fn,device_index):
        simulator=original_wrap(fn,device_index=device_index)
        def invoke(*args,**kwargs):
            kwargs['checkpoint_sink']=checkpoint_policy(kwargs['checkpoint_sink'],600000,
                p['deadline_epoch'],ignored)
            return simulator(*args,**kwargs)
        return invoke
    old_out,old_prepare=physical.OUT,physical.prepare
    physical.OUT,physical.prepare=OUT,lambda:p
    carrier.wrap_simulator=wrap
    try:physical.worker(NAME)
    finally:
        physical.OUT,physical.prepare=old_out,old_prepare
        carrier.wrap_simulator=original_wrap
    assert carrier.base.sha(p['source_checkpoint'])==p['source_checkpoint_sha256']
    folder=OUT/'runs'/NAME;r=carrier.base.read(folder/'result.json')
    original_reason=r['tracker']['stop_reason']
    if r['status']=='COMPLETE':
        assert r['end_s']==60.
        r['tracker']['stop_reason']='PREDEFINED_CONTINUATION_HORIZON'
    r['display_stop_s']=r['end_s']
    r['continuation']=dict(source_checkpoint=p['source_checkpoint'],start_s=30.,
        source_unchanged=True,full_engine_carried=True,no_physics_or_detector_change=True,
        suppressed_observation_stops=ignored,original_tracker_stop_reason=original_reason,
        note='This diagnostic displays the full continuation; original first-two-entry Fig5 keeps its shorter display.')
    carrier.base.write(folder/'result.json',r);carrier.base.write(folder/'progress.json',r)
    carrier.base.write(OUT/'continuation_complete.json',dict(status='PASS',end_s=r['end_s'],
        termination_status=r['status'],source_unchanged=True,suppressed_stops=ignored))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','worker'])
    parser.add_argument('--producer-script');args=parser.parse_args()
    try:globals()[args.mode]()
    except Exception as exc:
        carrier.base.write(OUT/'failure.json',dict(error=repr(exc),time=time.time()))
        raise
