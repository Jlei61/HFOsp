"""Refine constituent rate states at unchanged J; do not relabel bifurcations."""
from rate_periodic_accuracy import *
from compare_rate_torus_periodic_targets import distances
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('labels',nargs='+')
    p.add_argument('--device',type=int,default=0);p.add_argument('--min-free-gib',type=float,default=10)
    p.add_argument('--name',help='Short output identity for a larger bounded batch')
    p.add_argument('--stream-harmonics',action='store_true',help='Exact frequency blocks; do not retain the complex harmonic bank')
    p.add_argument('--wait-pids',type=int,nargs='*',default=[]);a=p.parse_args()
    source=read(PERIODIC_OUT/'rate_filter_state_positivity_audit.json')
    lookup={q['label']:q for q in source['rows']};assert all(k in lookup for k in a.labels)
    s=RateField();w=s.geo['group_size']/s.geo['group_size'].sum();rows=[]
    destination=PERIODIC_OUT/('filter_state_repair_'+(a.name or '_'.join(a.labels))+'.json')
    def status(stage,**kw):
        write(destination,dict(status=stage,pid=os.getpid(),rows=rows,**kw));print(stage,kw,flush=True)
    dependencies={pid:Path(f'/proc/{pid}/cmdline').read_bytes() for pid in a.wait_pids
                  if Path(f'/proc/{pid}/cmdline').exists()}
    while dependencies:
        for pid,identity in list(dependencies.items()):
            try:active=Path(f'/proc/{pid}/cmdline').read_bytes()==identity
            except FileNotFoundError:active=False
            if not active:dependencies.pop(pid)
        if dependencies:status('WAITING_DEPENDENCIES',dependencies=list(dependencies));time.sleep(30)
    for label in a.labels:
        original=Path(lookup[label]['orbit'])
        def before_mesh(N,stage):
            import cupy as cp
            cp.cuda.Device(a.device).use();gc.collect()
            cp.fft.config.get_plan_cache().clear();cp.get_default_memory_pool().free_all_blocks()
            # Frequency-block construction bounds the temporary bank. The
            # remaining harmonic matrices still grow linearly with N.
            # The streamed bank is small, but the fourfold nonlinear
            # residual still owns several dense moment/FFT arrays. A real
            # N8192 check exceeded 8.4 GiB before allocation failure; reserve
            # for these arrays as well, instead of using a 7 GiB estimate.
            required=max(a.min_free_gib,(4+N/1024) if a.stream_harmonics else (4+N/512))
            while True:
                free=float(subprocess.check_output(['nvidia-smi','-i',str(a.device),
                    '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
                if free>=required*1024:break
                status('WAITING_GPU_RESOURCE',label=label,N=N,next_stage=stage,free_mib=free,required_gib=required);time.sleep(30)
        status('REFINING',label=label)
        try:
            cached=[read(f) for f in original.parent.glob(original.stem+'_accuracy_N*.json')]
            cached=[q for q in cached if q['status']=='CONVERGED']
            start=Path(max(cached,key=lambda q:q['N'])['path']) if cached else original
            # Larger N is not necessarily better if an older solve stopped
            # at the loose algebraic tolerance. Reuse an already positive
            # cached profile first; prepare still checks its full residual.
            for candidate in sorted(cached,key=lambda q:q['N']):
                profile=np.load(candidate['path'])
                if filter_state_minima(s,profile['r'],float(profile['T']))['positive']:
                    start=Path(candidate['path']);break
            path,check=prepare(start,a.device,max_N=8192,host_krylov=True,check_filter_states=True,
                               harmonic_chunk_size=64,before_mesh=before_mesh,stream_harmonics=a.stream_harmonics)
            z=np.load(original);zz=np.load(path);N=max(len(z['r']),len(zz['r']))
            x=resample(z['r']*1000,N,axis=0);y=resample(zz['r']*1000,N,axis=0)
            distance,_=distances(x[:,None,:],y,w)
            amplitude=np.sqrt(np.mean(np.sum((x-x.mean(0))**2*w,axis=1)))
            row=dict(label=label,original_orbit=str(original),refined_orbit=str(path),resolution=check,
                relative_waveform_change=float(distance[0]/amplitude),
                relative_period_change=abs(float(zz['T'])/float(z['T'])-1),
                status='PROFILE_REFINED_CRITICAL_MODE_RECHECK_PENDING' if check['status']=='RESOLUTION_CHECKED' else 'PROFILE_UNRESOLVED')
            assert float(z['J'])==float(zz['J'])
            assert row['relative_waveform_change']<.02 and row['relative_period_change']<.01
            rows.append(row)
            # A composite case is explicitly a same-J waveform, so its
            # accepted source can be updated without claiming a new root.
            if label.startswith('composite_') and check['status']=='RESOLUTION_CHECKED':
                f=PERIODIC_OUT/'composite_case_resolution.json';prior=read(f)
                history=PERIODIC_OUT/'stability_coverage/attempt_history'/f'composite_before_filter_refinement_{time.time_ns()}.json'
                write(history,prior)
                for item in prior['rows']:
                    if item['case']==label[-1]:
                        item.update(orbit=str(path),resolution=check,filter_state_refinement_source=str(destination))
                write(f,prior);row['status']='COMPOSITE_WAVEFORM_REFINED'
            status('PROFILE_FINISHED',label=label)
        except Exception as exc:
            status('STOPPED_WITH_ERROR',label=label,error=repr(exc));raise
        import cupy as cp
        gc.collect();cp.fft.config.get_plan_cache().clear();cp.get_default_memory_pool().free_all_blocks()
    status('BATCH_FINISHED',scope='Fixed-J physical waveform refinement. Critical locations and eigenmodes must be rechecked on corrected critical profiles before full acceptance; no parameter or equation change.')


if __name__=='__main__':main()
