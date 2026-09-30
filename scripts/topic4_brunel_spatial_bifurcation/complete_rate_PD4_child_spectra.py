"""Check the unresolved spectrum along the physical PD4 doubled branch.

The largest multiplier changes sign between two checked children. Match and
count the full returned spectrum before interpreting any additional crossing.
"""
from complete_rate_positive_stability import *
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1)
    p.add_argument('--after-pids',type=int,nargs='*',default=[])
    p.add_argument('--min-free-gib',type=float,default=4.)
    a=p.parse_args();folder=DEST/'H2_local_PD';worker=folder/'worker_child_spectra.json'
    def status(state,**kw):
        write(worker,dict(status=state,pid=os.getpid(),timestamp=time.time(),**kw))
        print(state,kw,flush=True)
    def gate():
        release(a.device)
        while True:
            free=float(subprocess.check_output(['nvidia-smi','-i',str(a.device),
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
            if free>=a.min_free_gib*1024:return
            status('WAITING_RESOURCE',free_mib=free);time.sleep(30)
    deps={pid:Path(f'/proc/{pid}/cmdline').read_bytes() for pid in a.after_pids
          if Path(f'/proc/{pid}/cmdline').exists()}
    while deps:
        for pid,identity in list(deps.items()):
            proc=Path(f'/proc/{pid}/cmdline')
            if not proc.exists() or proc.read_bytes()!=identity:deps.pop(pid)
        if deps:status('WAITING_PARENT_WITNESSES',dependencies=list(deps));time.sleep(30)
    try:
        profiles=read(folder/'physical_children.json')['rows'];rows=[]
        for index,profile in enumerate(profiles):
            path=Path(profile['orbit']);physical=profile['physical_check']
            assert Path(physical['orbit']).resolve()==path.resolve()
            assert physical['filter_state_check']['positive'] and physical['maximum_group_defect_Hz']<1e-6
            saved=folder/f'child_full_spectrum_{index}.json'
            if saved.exists() and read(saved).get('numerical_unstable_dimension') is not None:
                rows.append(read(saved));continue
            attempts=[]
            for nev,steps in [(6,[.05,.025]),(10,[.025,.0125])]:
                spectra=[];sources=[]
                for dt in steps:
                    gate();tag=f'PD4_child_index{index}_full_k{nev}_20260920'
                    result_path=PERIODIC_OUT/'poincare_floquet'/f'{tag}_dt{dt:g}.json'
                    status('FULL_CHILD_SPECTRUM',child_index=index,nev=nev,dt_ms=dt)
                    q=read(result_path) if result_path.exists() else compute_return(
                        path,dt,nev,a.device,2*nev+4,stream_harmonics=True,output_label=tag)
                    assert Path(q['orbit']).resolve()==path.resolve()
                    spectra.append(q);sources.append(str(result_path))
                classification=paired_modes(*spectra)
                attempts.append(dict(sources=sources,classification=classification))
                result=dict(child_index=index,orbit=str(path),J_EE_core=profile['J_EE_core'],
                    physical=physical,attempts=attempts,**classification)
                write(saved,result)
                if classification['numerical_unstable_dimension'] is not None:break
            rows.append(result)
        write(folder/'child_full_spectrum_summary.json',dict(status='SAMPLED_SPECTRA_COMPLETE',rows=rows,
            interval_completeness=False,
            scope='Paired-step full-delay spectra at three physical doubled cycles. A change in the largest eigenvalue sign alone is not a bifurcation; matching modes and refined crossings are separate follow-ups. Equal sampled unstable dimensions do not exclude intermediate crossings.'))
        status('SAMPLED_SPECTRA_COMPLETE')
    except Exception as exc:
        status('COMPUTATION_FAILED',error=repr(exc));raise


if __name__=='__main__':main()
