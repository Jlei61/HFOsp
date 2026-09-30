"""Resolve the noncritical spectrum on both physical PD3 parent witnesses.

The existing six-mode spectra establish the matched -1 crossing but do not
cover every multiplier outside the unit disk. Reuse that evidence, then
increase spectral coverage before counting changes in unstable dimension.
"""
from complete_rate_positive_stability import *
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0)
    p.add_argument('--after-pids',type=int,nargs='*',default=[])
    p.add_argument('--min-free-gib',type=float,default=4.)
    a=p.parse_args();folder=DEST/'PD3_parent_spectra';folder.mkdir(exist_ok=True)
    worker=folder/'worker.json'
    def status(state,**kw):
        write(worker,dict(status=state,pid=os.getpid(),timestamp=time.time(),**kw))
        print(state,kw,flush=True)
    dependencies={pid:Path(f'/proc/{pid}/cmdline').read_bytes() for pid in a.after_pids
                  if Path(f'/proc/{pid}/cmdline').exists()}
    while dependencies:
        for pid,identity in list(dependencies.items()):
            f=Path(f'/proc/{pid}/cmdline')
            if not f.exists() or f.read_bytes()!=identity:dependencies.pop(pid)
        if dependencies:status('WAITING_DEPENDENCIES',dependencies=list(dependencies));time.sleep(30)
    def gate(side):
        release(a.device)
        while True:
            free=float(subprocess.check_output(['nvidia-smi','-i',str(a.device),
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
            if free>=a.min_free_gib*1024:return
            status('WAITING_GPU_RESOURCE',side=side,free_mib=free);time.sleep(30)
    try:
        witnesses=read(DEST/'PD3_parent_witnesses/result.json')
        assert witnesses['status']=='PHYSICAL_PARENT_CROSSING_CHECKED'
        rows=[]
        for row in witnesses['rows']:
            side=row['side'];path=Path(row['orbit']);physical=row['physical']
            assert Path(physical['orbit']).resolve()==path.resolve()
            assert physical['filter_state_check']['positive'] and physical['maximum_group_defect_Hz']<1e-6
            output=folder/(side+'.json')
            if output.exists() and read(output).get('numerical_unstable_dimension') is not None:
                rows.append(read(output));continue
            attempts=[]
            for nev,steps in [(6,[.05,.025]),(10,[.025,.0125]),(16,[.0125,.00625])]:
                spectra=[];sources=[]
                for k,dt in enumerate(steps):
                    if nev==6:
                        source=Path(row['attempts'][0]['sources'][k]);q=read(source)
                    else:
                        gate(side);tag=f'PD3_parent_{side}_full_k{nev}_20260920'
                        source=PERIODIC_OUT/'poincare_floquet'/f'{tag}_dt{dt:g}.json'
                        status('FULL_PARENT_SPECTRUM',side=side,nev=nev,dt_ms=dt)
                        q=read(source) if source.exists() else compute_return(path,dt,nev,a.device,
                            2*nev+4,stream_harmonics=True,output_label=tag)
                    assert Path(q['orbit']).resolve()==path.resolve()
                    assert abs(q['dt_ms']/dt-1)<.001
                    spectra.append(q);sources.append(str(source))
                classification=paired_modes(*spectra)
                attempts.append(dict(nev=nev,sources=sources,classification=classification))
                result=dict(side=side,orbit=str(path),J_EE_core=row['J_EE_core'],physical=physical,
                    attempts=attempts,**classification)
                write(output,result)
                if classification['numerical_unstable_dimension'] is not None:break
            rows.append(result)
        counts=[q['numerical_unstable_dimension'] for q in rows]
        write(folder/'summary.json',dict(status='PARENT_SPECTRUM_BATCH_FINISHED',rows=rows,
            both_dimensions_resolved=all(n is not None for n in counts),
            sampled_dimension_change=None if any(n is None for n in counts) else counts[1]-counts[0],
            interval_completeness=False,
            scope='Physical parent witnesses bracketing PD3, in the same continuation direction. Extra changes of unstable dimension require separate location and mode tracking. A difference of one is compatible with the known flip but does not exclude compensating crossings inside the interval.'))
        status('PARENT_SPECTRUM_BATCH_FINISHED',dimensions=counts)
    except Exception as exc:
        status('COMPUTATION_FAILED',error=repr(exc));raise


if __name__=='__main__':main()
