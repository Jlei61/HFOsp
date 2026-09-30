"""Continue the unresolved B-leading endpoint after physical mesh correction."""
from rate_periodic import *
from rate_periodic_accuracy import prepare
from compare_rate_torus_periodic_targets import distances
import subprocess,gc


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1)
    p.add_argument('--after-pid',type=int,nargs='*',default=[])
    p.add_argument('--min-free-gib',type=float,default=12.)
    p.add_argument('--stream-harmonics',action='store_true')
    p.add_argument('--quadratic-predictor',action='store_true')
    p.add_argument('--resume-existing',action='store_true');a=p.parse_args()
    label='arcBleadingConnection_20260920';scripts=Path(__file__).parent
    folder=Path('/data/hfosp/topic4_sef_hfo/interictal_rate_branch_completion_20260920/Bleading_extension')
    folder.mkdir(parents=True,exist_ok=True);worker=folder/'worker.json'
    def status(state,**kw):write(worker,dict(status=state,pid=os.getpid(),label=label,**kw))
    deps={pid:Path(f'/proc/{pid}/cmdline').read_bytes() for pid in a.after_pid
          if Path(f'/proc/{pid}/cmdline').exists()}
    while deps:
        for pid,identity in list(deps.items()):
            try:active=Path(f'/proc/{pid}/cmdline').read_bytes()==identity
            except FileNotFoundError:active=False
            if not active:deps.pop(pid)
        if deps:status('WAITING_DEPENDENCIES',dependencies=list(deps));time.sleep(30)
    def memory():
        while True:
            free=float(subprocess.check_output(['nvidia-smi','-i',str(a.device),
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
            if free>=a.min_free_gib*1024:return
            status('WAITING_GPU_RESOURCE',free_mib=free);time.sleep(30)
    def run(stage,args):
        memory();log=folder/f'{stage}_{time.time_ns()}.log'
        with log.open('w') as out:
            child=subprocess.Popen([sys.executable,'-u',*map(str,args)],stdout=out,stderr=subprocess.STDOUT)
            status(stage,child_pid=child.pid,log=str(log));code=child.wait()
        if code:raise RuntimeError(str(log))
    try:
        s=RateField();weights=s.geo['group_size']/s.geo['group_size'].sum();seeds=[]
        if a.resume_existing:
            saved=read(folder/'seeds.json');assert saved['status']=='COMPLETE'
            seeds=saved['rows'];assert len(seeds)==2
            assert all(q['resolution']['status']=='RESOLUTION_CHECKED' and Path(q['orbit']).exists() for q in seeds)
        for i in ([] if a.resume_existing else [58,59]):
            memory();source=PERIODIC_OUT/'orbits'/f'arcBleadingBridge_{i:04d}_N1024.npz'
            status('PHYSICAL_SEED_REFINEMENT',index=i,orbit=str(source))
            actual,check=prepare(source,a.device,max_N=4096,host_krylov=True,
                check_filter_states=True,harmonic_chunk_size=64,adaptive_memory=True,
                stream_harmonics=a.stream_harmonics)
            if check['status']!='RESOLUTION_CHECKED':
                status('SEED_REFINEMENT_UNRESOLVED',index=i,check=check);return
            old,new=np.load(source),np.load(actual);N=max(len(old['r']),len(new['r']))
            x,y=[resample(z['r']*1000,N,axis=0) for z in [old,new]]
            d,_=distances(x[:,None,:],y,weights)
            norm=np.sqrt(np.mean(np.sum((x-x.mean(0))**2*weights,axis=1)))
            assert abs(float(old['J'])-float(new['J']))<1e-12
            assert abs(float(new['T'])/float(old['T'])-1)<.01 and d[0]/norm<.02
            seeds.append(dict(source=str(source),orbit=str(actual),resolution=check,
                relative_waveform_change=float(d[0]/norm)))
            write(folder/'seeds.json',dict(status='COMPLETE' if len(seeds)==2 else 'RUNNING',rows=seeds))
            import cupy as cp
            gc.collect();cp.get_default_memory_pool().free_all_blocks()
        paths=[q['orbit'] for q in seeds];N=max(len(np.load(f)['r']) for f in paths)
        destination=PERIODIC_OUT/(label+'_continuation.json')
        assert destination.exists()==a.resume_existing,'Existing continuation must be explicitly resumed'
        command=[scripts/'rate_periodic_continue.py',*paths,'--N',N,
            '--steps','80','--ds','.5','--stop-after-turns','2','--tol','2e-11',
            '--label',label,'--device',a.device,'--low-memory','--host-krylov',
            '--linear-normalize','--require-filter-positivity']
        if a.stream_harmonics:command.append('--stream-harmonics')
        if a.quadratic_predictor:command.append('--quadratic-predictor')
        if a.resume_existing:command.append('--resume-existing')
        run('CONTINUATION',command)
        if not read(destination)['rows']:
            status('NO_ACCEPTED_NEW_POINTS',continuation=str(destination));return
        run('ALL_PROFILE_CHECKS',[scripts/'check_rate_extension.py',label,'--device',a.device,
            '--stride','1','--check-filter-states','--prior-segment','arcBleadingBridge',
            '--prior-orbits',*paths])
        q=read(PERIODIC_OUT/(label+'_accuracy.json'))
        strict=all(v['maximum_group_defect_Hz']<.001 for v in q['checks'])
        if not strict:
            q.update(status='RESOLUTION_UNRESOLVED',strict_continuous_defect_Hz=.001)
            write(PERIODIC_OUT/(label+'_accuracy.json'),q)
        write(folder/'checked_segment.json',q)
        if q['status']=='SAMPLED_PASS':
            run('FULL_SPACE_CONNECTION_SCREEN',[scripts/'screen_rate_branch_encounters.py',
                '--accepted-segment',label,'--provisional-family','Bleading',
                '--phase-samples','4096','--include-higher-hopfs'])
        status('BATCH_FINISHED',scientific_status=q['status'],continued_points=q['continued_points'],
            candidate_turns=len(q['turns']),global_connection_confirmed=False)
    except Exception as exc:
        status('STOPPED_WITH_ERROR',error=repr(exc));raise


if __name__=='__main__':main()
