"""Checkpointed full-state Floquet coverage of already computed orbit families.

Samples are a numerical survey, not a proof that no crossing occurs between
them. Keep untested intervals explicit; never infer stability from plot style.
"""
from rate_floquet import *
from plot_rate_periodic_completion import families, critical, continuation_breaks
from rate_periodic_accuracy import prepare
import subprocess
import gc

COVER = PERIODIC_OUT/'stability_coverage'


def wait_for_memory(args, i, path):
    """Defer a survey site while another process occupies its GPU."""
    if args.min_free_gib <= 0:return
    while True:
        free_mib=float(subprocess.check_output([
            'nvidia-smi', '-i', str(args.device),
            '--query-gpu=memory.free', '--format=csv,noheader,nounits'],text=True).strip())
        if free_mib >= args.min_free_gib*1024:return
        write(COVER/f'worker{args.shard}.json',dict(status='WAITING_RESOURCE',
            task_index=i,orbit=path,pid=os.getpid(),free_gpu_mib=free_mib,
            required_free_gib=args.min_free_gib))
        print('WAITING_RESOURCE',i,free_mib,'MiB free',flush=True)
        time.sleep(30)


def assess(q):
    raw=np.asarray(q['multipliers'])
    vals=raw[:,0]+1j*raw[:,1] if raw.ndim==2 else raw.astype(complex)
    residual=np.array(q['residuals'])/np.maximum(1,abs(vals))
    neutral=q.get('identified_neutral_index')
    keep=np.ones(len(vals),bool)
    if neutral is not None:keep[neutral]=False
    reliable=(residual<1e-6)&keep
    # Near-unit multipliers need refinement against the phase/timestep error.
    margin=max(.002,4*q['phase_tangent_relative_defect'])
    outside=reliable&(abs(vals)>1+margin)
    if outside.any():return dict(status='UNSTABLE',reliable_outside_count=int(outside.sum()),margin=margin)
    covered=q.get('smallest_returned_transformed_modulus',np.inf)<.9*q.get('filter_coverage_threshold',0.)
    if (neutral is not None and covered and (residual<1e-6).all() and
        q['phase_tangent_relative_defect']<.003 and keep.any() and max(abs(vals[keep]))<1-margin):
        return dict(status='NUMERICALLY_STABLE',reliable_outside_count=0,margin=margin)
    return dict(status='UNRESOLVED',reliable_outside_count=0,margin=margin)


def plan(stride=18):
    fs=families();selected={}
    for family,rr in fs.items():
        # Child branch has only eight local samples and is already unstable at
        # its largest amplitude; retain both ends instead of duplicating all.
        ids=set(range(0,len(rr),stride))|{len(rr)-1}
        if family=='PDchild':ids={0,len(rr)-1}
        # Include samples on both sides of every parameter reversal. These
        # indices are candidate survey sites, never new bifurcation labels.
        j=np.array([q['J_EE_core'] for q in rr]);turn=np.flatnonzero(np.diff(j)[:-1]*np.diff(j)[1:]<0)+1
        breaks=set(continuation_breaks(rr))
        for i in turn:
            if i in breaks or i+1 in breaks:continue
            ids.update([max(0,i-2),min(len(rr)-1,i+2)])
        for i in sorted(ids):
            q=rr[i];path=str(Path(q['path']).resolve());entry=selected.setdefault(path,dict(orbit=path,J_EE_core=q['J_EE_core'],T_ms=q['T_ms'],N=q['N'],memberships=[]))
            entry['memberships'].append(dict(family=family,index=i))
    rows=list(selected.values())
    # Short Hopf-born cycles first establish the stability structure cheaply.
    rows.sort(key=lambda q:(q['T_ms']>350,q['T_ms'],q['J_EE_core']))
    write(COVER/'plan.json',dict(rows=rows,stride=stride,meaning='Full-state monodromy sample sites; incomplete interval coverage until adaptive follow-up and crossing refinement.'))
    print('PLANNED',len(rows),flush=True)


def run(args):
    tasks=read(COVER/'plan.json')['rows'];indices=list(range(args.shard,len(tasks),args.shards))
    if args.limit:indices=indices[:args.limit]
    for i in indices:
        item=tasks[i];path=item['orbit'];tag=Path(path).stem;dest=COVER/(tag+'.json')
        if dest.exists():
            previous_status=read(dest).get('status')
            if previous_status in ['NUMERICALLY_STABLE','UNSTABLE']:continue
            # A resumed frozen survey should finish unattempted sites before
            # repeating an inconclusive calculation with identical settings.
            # Dedicated follow-ups retain responsibility for those sites.
            if args.only_pending and previous_status=='UNRESOLVED':continue
        wait_for_memory(args,i,path)
        if dest.exists():
            prior=read(dest)
            history=COVER/'attempt_history'/f'{tag}_{time.time_ns()}.json'
            history.parent.mkdir(exist_ok=True)
            write(history,prior)
        source=PERIODIC_OUT/'floquet'/f'{tag}_dt{args.dt:g}.json'
        write(COVER/f'worker{args.shard}.json',dict(status='RUNNING',task_index=i,orbit=path,pid=os.getpid()))
        started=time.time()
        try:
            actual,accuracy=prepare(path,args.device)
            write(COVER/(tag+'_resolution.json'),accuracy)
            if accuracy['status']!='RESOLUTION_CHECKED':
                write(dest,dict(status='UNRESOLVED',orbit=path,resolution=accuracy,meaning='Orbit resolution must be repaired before stability classification.'))
                continue
            source=PERIODIC_OUT/'floquet'/f'{actual.stem}_dt{args.dt:g}.json'
            q=read(source) if source.exists() and 'polynomial_filter_rho' in read(source) else compute(actual,args.dt,8,args.device,True,24)
            verdict=assess(q)
            write(dest,dict(**verdict,orbit=path,J_EE_core=item['J_EE_core'],T_ms=item['T_ms'],memberships=item['memberships'],
                analyzed_orbit=str(actual),resolution=accuracy,
                floquet_source=str(source),elapsed_seconds=time.time()-started,
                scope='Full state and delay-history Arnoldi at this orbit; sampled stability, not a continuum completeness certificate.'))
            print('COVERAGE',i,tag,verdict,flush=True)
        except Exception as exc:
            write(dest,dict(status='COMPUTATION_FAILED',orbit=path,error=repr(exc),meaning='Numerical failure is not a bifurcation or stability classification.'))
            print('FAILED',i,repr(exc),flush=True)
            if type(exc).__name__=='OutOfMemoryError':
                # Continuing immediately can turn a shared resource shortage
                # into dozens of misleading failed scientific sites.
                write(COVER/f'worker{args.shard}.json',dict(status='RESOURCE_WAIT_REQUIRED',
                    task_index=i,orbit=path,pid=os.getpid(),error=repr(exc)))
                return
        import cupy as cp
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    write(COVER/f'worker{args.shard}.json',dict(status='COMPLETE',task_indices=indices,pid=os.getpid()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',action='store_true');p.add_argument('--stride',type=int,default=18)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--device',type=int,default=0)
    p.add_argument('--dt',type=float,default=.1);p.add_argument('--limit',type=int)
    p.add_argument('--min-free-gib',type=float,default=10.)
    p.add_argument('--only-pending',action='store_true',
        help='Preserve completed inconclusive sites; run pending/failed sites only')
    a=p.parse_args();COVER.mkdir(exist_ok=True)
    if a.plan:plan(a.stride)
    else:run(a)
