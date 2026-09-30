"""Refine every profile of a withheld segment at the same parameter values.

The old segment remains withheld until all corrections pass physical-state,
continuous-equation and waveform-identity checks. This is not a new fold or
Floquet classification. Existing artifacts are preserved.
"""
from rate_periodic_accuracy import *
from compare_rate_torus_periodic_targets import distances
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('label')
    p.add_argument('--device',type=int,default=1);p.add_argument('--min-N',type=int,default=1024)
    p.add_argument('--max-N',type=int,default=4096);p.add_argument('--min-free-gib',type=float,default=14.)
    p.add_argument('--host-krylov',action='store_true');p.add_argument('--wait-pids',type=int,nargs='*',default=[])
    a=p.parse_args();source=PERIODIC_OUT/(a.label+'_accuracy.json')
    original=read(source);assert original['status']=='RESOLUTION_UNRESOLVED'
    checkpoint=PERIODIC_OUT/(a.label+'_physical_refinement.json')
    previous=read(checkpoint) if checkpoint.exists() else {}
    rows=previous.get('rows',[]);s=RateField();weights=s.geo['group_size']/s.geo['group_size'].sum()
    def record(status,**kw):
        write(checkpoint,dict(status=status,pid=os.getpid(),source=str(source),rows=rows,**kw))
    deps={pid:Path(f'/proc/{pid}/cmdline').read_bytes() for pid in a.wait_pids
          if Path(f'/proc/{pid}/cmdline').exists()}
    while deps:
        for pid,identity in list(deps.items()):
            try:active=Path(f'/proc/{pid}/cmdline').read_bytes()==identity
            except FileNotFoundError:active=False
            if not active:deps.pop(pid)
        if deps:record('WAITING_DEPENDENCIES',dependencies=list(deps));time.sleep(30)
    def before_mesh(N,stage):
        import cupy as cp
        cp.cuda.Device(a.device).use();gc.collect();cp.fft.config.get_plan_cache().clear()
        cp.get_default_memory_pool().free_all_blocks()
        required=max(a.min_free_gib,4+N/512+(0 if a.host_krylov else N/1024))
        while True:
            free=float(subprocess.check_output(['nvidia-smi','-i',str(a.device),
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
            if free>=required*1024:return
            record('WAITING_GPU_RESOURCE',index=index,N=N,next_stage=stage,free_mib=free,required_gib=required)
            time.sleep(30)
    originals=list(map(Path,original['included_orbits']))
    for index,path in enumerate(originals):
        if any(v['index']==index and v['status']=='PROFILE_REFINED' for v in rows):continue
        record('REFINING',index=index)
        try:
            cache=[read(f) for f in path.parent.glob(path.stem+'_accuracy_N*.json')]
            cache=[q for q in cache if q['status']=='CONVERGED' and q['N']<=a.max_N]
            start=Path(max(cache,key=lambda q:q['N'])['path']) if cache else path
            actual,check=prepare(start,a.device,max_N=a.max_N,min_N=a.min_N,
                host_krylov=a.host_krylov,check_filter_states=True,harmonic_chunk_size=64,before_mesh=before_mesh)
            z=np.load(path);zz=np.load(actual);N=max(len(z['r']),len(zz['r']))
            x=resample(z['r']*1000,N,axis=0);y=resample(zz['r']*1000,N,axis=0)
            d,_=distances(x[:,None,:],y,weights)
            amplitude=np.sqrt(np.mean(np.sum((x-x.mean(0))**2*weights,axis=1)))
            delta=float(d[0]/amplitude);period=abs(float(zz['T'])/float(z['T'])-1)
            same_parameter=abs(float(zz['J'])-float(z['J']))<1e-13
            passed=check['status']=='RESOLUTION_CHECKED' and delta<.02 and period<.01 and same_parameter
            row=dict(index=index,original_orbit=str(path),refined_orbit=str(actual),resolution=check,
                relative_waveform_change=delta,relative_period_change=period,same_parameter=same_parameter,
                status='PROFILE_REFINED' if passed else 'PROFILE_UNRESOLVED')
            rows[:]=[v for v in rows if v['index']!=index];rows.append(row)
            record('PROFILE_FINISHED',index=index);print('REFINED SEGMENT PROFILE',index,row['status'],check.get('N'),delta,flush=True)
            if not passed:record('STOPPED_PROFILE_UNRESOLVED',index=index);return
        except Exception as exc:
            record('STOPPED_WITH_ERROR',index=index,error=repr(exc));raise
        import cupy as cp
        gc.collect();cp.fft.config.get_plan_cache().clear();cp.get_default_memory_pool().free_all_blocks()
    assert len(rows)==len(originals) and all(v['status']=='PROFILE_REFINED' for v in rows)
    rows.sort(key=lambda v:v['index'])
    replacement={str(old):v['refined_orbit'] for old,v in zip(originals,rows)}
    turns=[]
    for turn in original['turns']:
        q=dict(turn)
        for key in ['left','center','right']:
            if q.get(key) in replacement:q[key]=replacement[q[key]]
        turns.append(q)
    revised=dict(original,status='SAMPLED_PASS',included_orbits=[v['refined_orbit'] for v in rows],
        checks=[dict(v['resolution'],index=v['index']) for v in rows],turns=turns,
        physical_profile_check='ALL_SEGMENT_PROFILES_REFINED',physical_refinement_source=str(checkpoint),
        scope='Same-J corrections of every continued profile, physical positivity and off-grid equations checked. Parameter turns remain candidates; no new Floquet or global-connection classification.')
    archive=PERIODIC_OUT/'attempt_history'/f'{a.label}_before_physical_refinement_{time.time_ns()}.json'
    write(archive,original);write(source,revised)
    record('SEGMENT_REFINED',archived_original=str(archive),scope=revised['scope'])


if __name__=='__main__':main()
