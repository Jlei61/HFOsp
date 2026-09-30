"""Continue physically checked periodic endpoints in the frozen spatial DDE."""
from rate_periodic import *
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1)
    p.add_argument('--steps',type=int,default=160);p.add_argument('--ds',type=float,default=1.2)
    p.add_argument('--label',default='arcAconnectionStage4')
    p.add_argument('--predecessor',help='Continue the checked last two points of an accepted segment')
    p.add_argument('--wait-pids',type=int,nargs='*',default=[])
    p.add_argument('--min-free-gib',type=float,default=14.)
    p.add_argument('--stream-harmonics',action='store_true')
    p.add_argument('--quadratic-predictor',action='store_true')
    p.add_argument('--tol',type=float,default=2e-8)
    p.add_argument('--check-stride',type=int,default=10)
    p.add_argument('--maximum-group-defect',type=float,default=.1)
    p.add_argument('--stop-after-turns',type=int,default=0,
                   help='Pause after this many new sampled parameter reversals for separate root checks')
    a=p.parse_args()
    assert a.min_free_gib>0 and a.check_stride>=1 and a.maximum_group_defect>0 and a.stop_after_turns>=0
    if a.quadratic_predictor:
        assert read(PERIODIC_OUT/'curvature_predictor_same_orbit_check.json')['status']=='PASS'
    label=a.label;worker=PERIODIC_OUT/('H1_stage4_worker.json' if label=='arcAconnectionStage4' else label+'_worker.json')
    if a.predecessor:
        seed_source=PERIODIC_OUT/(a.predecessor+'_accuracy.json');q=read(seed_source)
        assert q['status']=='SAMPLED_PASS'
        paths=q['included_orbits'][-2:];assert len(paths)==2
    else:
        seed_source=PERIODIC_OUT/'H1_stage4_seed_refinement.json';q=read(seed_source)
        assert q['status']=='SEEDS_REFINED' and len(q['rows'])==2
        paths=[v['refined_orbit'] for v in sorted(q['rows'],key=lambda v:v['index'])]
        assert all(v['resolution']['status']=='RESOLUTION_CHECKED' for v in q['rows'])
    N=max(len(np.load(f)['r']) for f in paths);scripts=Path(__file__).parent
    assert not (PERIODIC_OUT/(label+'_continuation.json')).exists(),'Existing segment must be explicitly resumed'
    deps={pid:Path(f'/proc/{pid}/cmdline').read_bytes() for pid in a.wait_pids if Path(f'/proc/{pid}/cmdline').exists()}
    def record(status,**kw):write(worker,dict(status=status,pid=os.getpid(),N=N,seed_source=str(seed_source),rows=[],**kw))
    while deps:
        for pid,identity in list(deps.items()):
            try:active=Path(f'/proc/{pid}/cmdline').read_bytes()==identity
            except FileNotFoundError:active=False
            if not active:deps.pop(pid)
        if deps:record('WAITING_DEPENDENCIES',dependencies=list(deps));time.sleep(30)
    while True:
        free=float(subprocess.check_output(['nvidia-smi','-i',str(a.device),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
        if free>=a.min_free_gib*1024:break
        record('WAITING_GPU_RESOURCE',free_mib=free);time.sleep(30)
    if a.predecessor:
        from rate_periodic_accuracy import prepare
        from compare_rate_torus_periodic_targets import distances
        import gc,cupy as cp
        seed_rows=[];s=RateField();weights=s.geo['group_size']/s.geo['group_size'].sum()
        for source in paths:
            record('VERIFYING_SEED',orbit=str(source))
            corrected,check=prepare(source,a.device,max_N=4096,host_krylov=True,
                check_filter_states=True,harmonic_chunk_size=64,
                stream_harmonics=a.stream_harmonics,adaptive_memory=True)
            assert check['status']=='RESOLUTION_CHECKED',check
            old,new=np.load(source),np.load(corrected);mesh=max(len(old['r']),len(new['r']))
            before,after=[resample(z['r']*1000,mesh,axis=0) for z in [old,new]]
            distance,_=distances(before[:,None,:],after,weights)
            scale=np.sqrt(np.mean(np.sum((before-before.mean(0))**2*weights,axis=1)))
            drift=float(distance[0]/scale);period_drift=abs(float(new['T']/old['T'])-1)
            assert abs(float(new['J']-old['J']))<1e-12 and drift<.02 and period_drift<.01
            seed_rows.append(dict(original_orbit=str(source),refined_orbit=str(corrected),resolution=check,
                relative_waveform_change=drift,relative_period_change=period_drift))
            gc.collect();cp.fft.config.get_plan_cache().clear();cp.get_default_memory_pool().free_all_blocks()
        paths=[row['refined_orbit'] for row in seed_rows];N=max(len(np.load(f)['r']) for f in paths)
        predecessor_source=str(seed_source);seed_source=PERIODIC_OUT/(label+'_seed_refinement.json')
        write(seed_source,dict(status='SEEDS_REFINED',rows=seed_rows,predecessor_source=predecessor_source))
    def run(stage,args):
        log=PERIODIC_OUT/(label+'_'+stage+'.log')
        with log.open('w') as f:
            child=subprocess.Popen([sys.executable,'-u',*map(str,args)],stdout=f,stderr=subprocess.STDOUT)
            record('RUNNING',stage=stage,child_pid=child.pid,log=str(log));code=child.wait()
        if code:record('STOPPED_WITH_ERROR',stage=stage,exit_code=code,log=str(log));raise RuntimeError(log)
    command=[scripts/'rate_periodic_continue.py',*paths,'--N',N,'--steps',a.steps,'--ds',a.ds,
        '--label',label,'--device',a.device,'--low-memory','--linear-normalize','--require-filter-positivity',
        '--tol',a.tol]
    if a.stream_harmonics:command+=['--stream-harmonics','--host-krylov']
    if a.quadratic_predictor:command+=['--quadratic-predictor']
    if a.stop_after_turns:command+=['--stop-after-turns',a.stop_after_turns]
    run('continuation',command)
    continuation=read(PERIODIC_OUT/(label+'_continuation.json'))
    if not continuation['rows']:
        record('STOPPED_PHYSICAL_REFINEMENT_REQUIRED',continuation=continuation);return
    command=[scripts/'check_rate_extension.py',label,'--device',a.device,'--stride',a.check_stride,
        '--prior-segment',a.predecessor or 'arcAconnectionStage3','--prior-orbits',*paths,'--check-filter-states',
        '--maximum-group-defect',a.maximum_group_defect]
    if a.stream_harmonics:command+=['--stream-harmonics']
    run('accuracy',command)
    check=read(PERIODIC_OUT/(label+'_accuracy.json'))
    record('BATCH_FINISHED',segment_status=check['status'],continued_points=check['continued_points'],
        requested_steps=a.steps,continuation_stop=continuation.get('status'),source=str(PERIODIC_OUT/(label+'_accuracy.json')),
        scope='Branch geometry from the named, physically checked predecessor endpoints; no Floquet or global-connection claim from continuation alone.')


if __name__=='__main__':main()
