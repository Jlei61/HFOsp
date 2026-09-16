#!/usr/bin/env python3
"""One numerical replica: same early Z rule, verified lookup,0.5s observations.

No new biological condition or independent sample. The source branch continues.
The frozen worker and all physics files remain unchanged.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import copy
from datetime import datetime
import hashlib
from pathlib import Path
import sys
import time
import numpy as np
import run_topic4_early_z_refill_branches as early
import plot_topic4_m_parameter_modes as fig

core=early.core;ROOT=core.ROOT;WINDOW=early.WINDOW
OUT=WINDOW/'early_Z_lookup_dense'
DISPLAY=WINDOW/'early_Z_lookup_dense_figures'
NAME='early_z_refill_s9108401'


def write(path,value):
    assert path.resolve().is_relative_to(OUT.resolve()) or path.resolve().is_relative_to(DISPLAY.resolve())
    core.write(path,value)


def prepare():
    import psutil
    OUT.mkdir(exist_ok=True);DISPLAY.mkdir(exist_ok=True)
    qa_path=WINDOW/'scatter_lookup_qa/full_network_qa.json';qa=core.read(qa_path)
    assert qa['status']=='PASS' and qa['full_engine_state_bitwise_identical'] and qa['all_observations_bitwise_identical']
    assert core.sha(qa['replacement'])==qa['replacement_sha256']
    source_protocol=core.read(early.OUT/'protocol.json')
    for path,digest in source_protocol['source_hashes'].items():assert core.sha(path)==digest,path
    job=core.read(early.OUT/'jobs'/(NAME+'.json'))
    assert job['eta_m']==.005 and job['tau_M_s']==1 and job['seed']==9108401
    parent=core.load_pickle(early.OUT/'parents/s9108401_10s.pkl')
    assert early.eligibility(parent)=='ELIGIBLE'
    original_engine_digest=early.engine_digest(parent['engine'])
    parent['job']=job
    source=early.OUT/'runs'/NAME
    write(OUT/'source_progress_at_prepare.json',core.read(source/'progress.json'))
    for base in [OUT,DISPLAY]:
        (base/'runs'/NAME/'chunks').mkdir(parents=True,exist_ok=True)
        geometry=base/'geometry.npz'
        if not geometry.exists():geometry.symlink_to((early.SOURCE/'geometry.npz').resolve())
        for path in (source/'chunks').glob('*.npz'):
            lo,hi=map(int,path.stem.split('_'))
            if hi>100000:continue
            target=base/'runs'/NAME/'chunks'/path.name
            if not target.exists():target.symlink_to(path.resolve())
        checkpoint=base/'runs'/NAME/'checkpoint.pkl'
        assert not checkpoint.exists(),'Replica already prepared; do not accidentally restart or overwrite it'
        core.save_pickle(checkpoint,parent)
        assert early.engine_digest(core.load_pickle(checkpoint)['engine'])==original_engine_digest
        protocol=copy.deepcopy(source_protocol);protocol['jobs']=[job];protocol['total']=1
        write(base/'protocol.json',protocol)
    audit=core.read(source/'branch_audit.json')
    audit.update(numerical_replica_of=str(source),independent_sample_added=False,
        source_branch_continues=True,parent_engine_sha256=original_engine_digest,
        observed_checkpoint_interval_s=.5,numerical_QA=str(qa_path))
    write(DISPLAY/'runs'/NAME/'branch_audit.json',audit)
    assert fig.OUT==early.SOURCE
    write(OUT/'plan.json',dict(time=datetime.now().astimezone().isoformat(),source_branch=str(source),
        branch_conditions_changed=False,independent_sample_added=False,included_in_F=False,
        job=job,parent_s=10.,parent_engine_sha256=original_engine_digest,
        source_early_protocol=core.read(early.OUT/'plan.json'),
        numerical_QA=str(qa_path),lookup_sha256=qa['replacement_sha256'],
        observer='Read-only checkpoint callback writes actual observations and complete engine every0.5s; physical callback remains intact.',
        source_branch_continues=True,max_replicas=1,display_source=str(DISPLAY),
        executor=str(Path(__file__).resolve()),executor_sha256=core.sha(__file__),
        deadline=core.read(WINDOW/'window.json')['deadline']))
    assert psutil.virtual_memory().available/2**30>=68
    assert time.time()<datetime.fromisoformat(core.read(WINDOW/'window.json')['deadline']).timestamp()
    return job


def prepare_resume():
    """Resume this same numerical replica from its latest closed half-second.

    Called only while holding the replica lock; original biological branches
    continue independently and none of their files is changed.
    """
    import psutil
    plan=core.read(OUT/'plan.json');job=plan['job']
    for path,digest in core.read(early.OUT/'protocol.json')['source_hashes'].items():
        assert core.sha(path)==digest,path
    assert time.time()<datetime.fromisoformat(core.read(WINDOW/'window.json')['deadline']).timestamp()
    assert psutil.virtual_memory().available/2**30>=68
    source_folder=DISPLAY/'runs'/NAME;dest_folder=OUT/'runs'/NAME
    assert not (dest_folder/'result.json').exists()
    blob=(source_folder/'checkpoint.pkl').read_bytes()
    saved=__import__('pickle').loads(blob)
    assert saved['job']==job
    assert saved['identity']==core.read(early.OUT/'protocol.json')['identity']
    end=int(saved['engine']['step']);assert end%5000==0
    before_digest=early.engine_digest(saved['engine'])
    def closed_end(folder):
        cursor=0
        for path in sorted((folder/'chunks').glob('*.npz')):
            if '.tmp.' in path.name:continue
            lo,hi=map(int,path.stem.split('_'))
            if hi>end:continue
            assert lo==cursor,(str(path),cursor)
            cursor=hi
        return cursor
    assert closed_end(source_folder)==end
    cursor=closed_end(dest_folder)
    for path in sorted((source_folder/'chunks').glob('*.npz')):
        if '.tmp.' in path.name:continue
        lo,hi=map(int,path.stem.split('_'))
        if hi<=cursor or hi>end:continue
        assert lo==cursor
        target=dest_folder/'chunks'/path.name
        assert not target.exists()
        target.symlink_to(path.resolve());cursor=hi
    assert cursor==end and closed_end(dest_folder)==end
    core.save_pickle(dest_folder/'checkpoint.pkl',saved)
    assert early.engine_digest(core.load_pickle(dest_folder/'checkpoint.pkl')['engine'])==before_digest
    history=core.read(OUT/'resume_history.json') if (OUT/'resume_history.json').exists() else {'resumes':[]}
    history['resumes'].append(dict(time=datetime.now().astimezone().isoformat(),step=end,
        parent_engine_sha256=before_digest,source_checkpoint_sha256=hashlib.sha256(blob).hexdigest(),
        executor=str(Path(__file__).resolve()),executor_sha256=core.sha(__file__),
        same_job_and_RNG=True,original_branch_untouched=True,independent_sample_added=False))
    write(OUT/'resume_history.json',history)
    reference=core.read(early.OUT/'runs'/NAME/'progress.json')
    if reference.get('time_s',-1)>end*.0001:
        write(OUT/'resume_reference_progress.json',reference)
    return job


def main(resume=False,use_cuda=False):
    import fcntl
    OUT.mkdir(exist_ok=True);lock=(OUT/'replica.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    job=prepare_resume() if resume else prepare()
    import src.topic4_serial_spike_scatter as serial
    from src.topic4_serial_spike_scatter_lookup import scatter
    serial.scatter=scatter
    core.OUT=OUT;core.tracker_step=early.early_tracker_step
    original_simulate=core.old.simulate_kick
    if use_cuda:
        qa_path=WINDOW/'cuda_ordered_scatter_qa/full_network_qa.json';qa=core.read(qa_path)
        assert qa['status']=='PASS' and qa['full_engine_state_bitwise_identical'] and qa['all_observations_bitwise_identical']
        assert core.sha(qa['replacement'])==qa['replacement_sha256']
        from src.topic4_cuda_ordered_scatter import wrap_simulator
        original_simulate=wrap_simulator(original_simulate,device_index=1)
        write(OUT/'computational_backend.json',dict(backend='CUDA_ordered_incoming',device=1,
            validation=str(qa_path),source=qa['replacement'],source_sha256=qa['replacement_sha256'],
            executor=str(Path(__file__).resolve()),executor_sha256=core.sha(__file__),
            same_numerical_replica=True,independent_sample_added=False))
    preview_end=int(core.load_pickle(OUT/'runs'/NAME/'checkpoint.pkl')['engine']['step'])
    if use_cuda:
        path=DISPLAY/'runs'/NAME/'branch_audit.json';audit=core.read(path)
        audit['cuda_full_state_QA']=str(qa_path)
        audit['numerical_backend_segments']=[[0,10,'original_CPU'],[10,preview_end*.0001,'CPU_slot_lookup'],
            [preview_end*.0001,None,'CUDA_ordered_incoming_device1']]
        audit['same_replica_continued']=True
        write(path,audit)
    def simulate(*args,**kwargs):
        original_sink=kwargs['checkpoint_sink']
        required={'block_start','data','identity','job','prior_wall','slow','started','tracker'}
        assert required.issubset(original_sink.__code__.co_freevars)
        def observe_checkpoint(k,engine):
            nonlocal preview_end
            closure={name:cell.cell_contents for name,cell in zip(original_sink.__code__.co_freevars,original_sink.__closure__)}
            start=closure['block_start'];data=closure['data'];assert start<=preview_end<k
            scales={'time_ms':10,'spikes_1ms':10,'regions_1ms':10,'field_1ms':10,'raster':1,
                'slow_time_ms':50,'Z':50,'M':50,'currents':50,'lfp_time_ms':5,'lfp_raw':5,'inputs':1000}
            assert data.keys()==scales.keys()
            arrays={key:np.asarray(values)[(preview_end-start)//scales[key]:] for key,values in data.items()}
            for key in ['spikes_1ms','regions_1ms','field_1ms']:arrays[key]=arrays[key].astype(np.uint16)
            assert len(arrays['raster'])==k-preview_end
            assert len(arrays['spikes_1ms'])*10==k-preview_end
            assert np.array_equal(arrays['spikes_1ms'][:,0],arrays['field_1ms'].sum(1))
            assert np.array_equal(arrays['spikes_1ms'][:,0],arrays['regions_1ms'][:,:3].sum(1))
            assert np.array_equal(arrays['spikes_1ms'][:,1],arrays['regions_1ms'][:,3:].sum(1))
            arrays.update(start_step=preview_end,end_step=k)
            folder=DISPLAY/'runs'/NAME
            dest=folder/'chunks'/f'{preview_end:010d}_{k:010d}.npz';temporary=dest.with_suffix('.tmp.npz')
            np.savez_compressed(temporary,**arrays);temporary.replace(dest)
            tracker=copy.deepcopy(closure['tracker'])
            tracker['wall_s']=closure['prior_wall']+time.time()-closure['started']
            core.save_pickle(folder/'checkpoint.pkl',dict(job=closure['job'],identity=closure['identity'],
                engine=engine,tracker=tracker,restore_from=closure['slow'].restore_from))
            write(folder/'progress.json',dict(status='COMMITTED_PREFIX_FOLLOWUP_RUNNING',pid=os.getpid(),
                time_s=k*.0001,entries=tracker['entries'],recoveries=tracker['recoveries'],phase=tracker['phase'],
                restore_s=tracker['restore_s'],release_s=tracker['release_s'],
                Z=float(closure['slow'].z[:32000].mean()),adaptation_current=float(.005*closure['slow'].m[:32000].mean()),
                numerical_replica=True,independent_sample_added=False))
            for reference_name,report_name in [('source_progress_at_prepare.json','source_progress_parity.json'),
                                               ('resume_reference_progress.json','source_progress_resume_parity.json')]:
                if not (OUT/reference_name).exists():continue
                reference=core.read(OUT/reference_name)
                if abs(reference.get('time_s',-1)-k*.0001)<1e-8:
                    measured=core.read(folder/'progress.json')
                    for key in ['Z','adaptation_current']:assert abs(measured[key]-reference[key])<1e-12,key
                    assert measured['entries']==reference['entries'] and measured['recoveries']==reference['recoveries']
                    write(OUT/report_name,dict(status='PASS',time_s=k*.0001,
                        quantities=['mean Z','mean applied M current','entry tracker','return tracker'],
                        full_spatial_and_raster_match_not_yet_directly_available=True))
            preview_end=k
            # Delegate without modifying the frozen worker's tracker, buffers,
            # engine or scheduling, including its original stop exception.
            original_sink(k,engine)
        kwargs['checkpoint_sink']=observe_checkpoint
        return original_simulate(*args,**kwargs)
    core.old.simulate_kick=simulate
    write(OUT/'status.json',dict(status='RUNNING',pid=os.getpid(),job=job,source_branch_continues=True,
        numerical_replica_not_new_condition=True))
    r=core.worker(job)
    assert preview_end==round(r['end_s']*10000)
    write(DISPLAY/'runs'/NAME/'result.json',r)
    a=fig.load(DISPLAY/'runs'/NAME);metrics=fig.analyze(a,r)
    fig.render(a,r,metrics,OUT/'candidate/figures',fig.grid_summary())
    write(OUT/'status.json',dict(status='COMPLETE_PENDING_REVIEW',pid=os.getpid(),end_s=r['end_s'],
        metrics=fig.safe(metrics),source_branch_continues=True,independent_sample_added=False))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');parser.add_argument('--cuda',action='store_true')
    args=parser.parse_args()
    try:main(resume=args.resume,use_cuda=args.cuda)
    except Exception as exc:
        OUT.mkdir(exist_ok=True);write(OUT/'status.json',dict(status='FAILED',pid=os.getpid(),error=repr(exc),source_branch_untouched=True))
        raise
