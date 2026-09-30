"""Check a bounded interval of existing H1 points before displaying them."""
from complete_rate_positive_stability import *
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1)
    p.add_argument('--after-pid',type=int,nargs='*',default=[])
    p.add_argument('--min-free-gib',type=float,default=6.)
    p.add_argument('--first-index',type=int,default=49)
    p.add_argument('--last-index',type=int,default=84)
    p.add_argument('--output-name',default='H1_display_extension')
    p.add_argument('--minimum-N',type=int,default=0,
        help='Minimum temporal Fourier mesh; spatial populations and equations are unchanged')
    p.add_argument('--stream-harmonics',action='store_true')
    a=p.parse_args();assert 0<=a.first_index<=a.last_index
    assert 0<=a.minimum_N<=4096
    assert Path(a.output_name).name==a.output_name
    expected=a.last_index-a.first_index+1
    output=DEST/(a.output_name+'.json');worker=DEST/(a.output_name+'_worker.json')
    prior=read(output).get('rows',[]) if output.exists() else []
    done={q['index']:q for q in prior if q['status']=='PASS'};rows=[]
    def status(stage,**kw):
        write(worker,dict(status=stage,pid=os.getpid(),completed=len(rows),expected=expected,**kw))
    deps={pid:Path(f'/proc/{pid}/cmdline').read_bytes() for pid in a.after_pid if Path(f'/proc/{pid}/cmdline').exists()}
    while deps:
        for pid,identity in list(deps.items()):
            f=Path(f'/proc/{pid}/cmdline')
            if not f.exists() or f.read_bytes()!=identity:deps.pop(pid)
        if deps:status('WAITING_DEPENDENCIES',dependencies=list(deps));time.sleep(30)
    for index in range(a.first_index,a.last_index+1):
        if index in done:rows.append(done[index]);continue
        release(a.device)
        def gate(mesh=0,stage='PROFILE'):
            required=a.min_free_gib
            # A larger Newton correction needs more memory than the exact
            # blocked off-grid check. Preserve a separate gate at each mesh
            # instead of assuming the initial free memory is sufficient.
            if stage=='NEWTON' and mesh>=4096:required=max(required,6.)
            while True:
                free=float(subprocess.check_output(['nvidia-smi','-i',str(a.device),
                    '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
                if free>=required*1024:return
                status('WAITING_GPU_RESOURCE',index=index,free_mib=free,
                    next_stage=stage,N=mesh,required_free_gib=required);time.sleep(30)
        gate()
        source=PERIODIC_OUT/'orbits'/f'arcAreturnStrong_{index:04d}_N512.npz'
        meta=read(source.with_suffix('.json'));status('SAME_J_PROFILE_CHECK',index=index,source=str(source))
        try:
            start=best_cached(dict(orbit=str(source),J_EE_core=meta['J_EE_core'],T_ms=meta['T_ms']))
            actual,check=prepare(start,a.device,max_N=4096,min_N=a.minimum_N,check_filter_states=True,
                harmonic_chunk_size=32,adaptive_memory=True,stream_harmonics=a.stream_harmonics,
                host_krylov=a.stream_harmonics,before_mesh=gate)
            # The display contract requires every population's off-grid
            # defect below .001 Hz; the generic survey preparation also
            # serves looser regional-only consumers. Enforce the actual
            # plotting requirement here rather than saving a false PASS.
            while (check['status']=='RESOLUTION_CHECKED' and
                   check['maximum_group_defect_Hz']>=.001 and check['N']<4096):
                actual,check=prepare(actual,a.device,max_N=4096,min_N=2*check['N'],
                    check_filter_states=True,harmonic_chunk_size=32,adaptive_memory=True,
                    stream_harmonics=a.stream_harmonics,host_krylov=a.stream_harmonics,
                    before_mesh=gate)
            old=np.load(source);fine=np.load(actual);coarse=resample(old['r'],len(fine['r']),axis=0)
            drift=float(np.linalg.norm(fine['r']-coarse)/np.linalg.norm(coarse-coarse.mean(0)))
            period=abs(float(fine['T']/old['T'])-1)
            same=abs(float(fine['J']-old['J']))<1e-12 and drift<.01 and period<.001
            passed=(check['status']=='RESOLUTION_CHECKED' and same and
                    check['maximum_group_defect_Hz']<.001)
            row=dict(index=index,source=str(source),orbit=str(actual),status='PASS' if passed else 'REVIEW_REQUIRED',
                check=check,relative_waveform_change=drift,relative_period_change=period,branch_match_pass=same)
        except Exception as exc:
            row=dict(index=index,source=str(source),status='COMPUTATION_FAILED',error=repr(exc))
        rows.append(row)
        write(output,dict(status='RUNNING',pid=os.getpid(),expected=expected,completed=len(rows),rows=rows,
            first_index=a.first_index,last_index=a.last_index))
        print('H1 DISPLAY EXTENSION',index,row['status'],flush=True)
        if row['status']=='COMPUTATION_FAILED':break
    passed=len(rows)==expected and all(q['status']=='PASS' for q in rows)
    write(output,dict(status='PASS' if passed else 'INCOMPLETE',expected=expected,completed=len(rows),rows=rows,
        first_index=a.first_index,last_index=a.last_index,
        scope='Existing H1 continuation, corrected at identical J. Every displayed extension sample must pass fourfold off-grid equation and constituent-filter positivity checks and remain within the original branch waveform/period tolerances. This adds no H1-to-burst connection or interval-wide stability claim.'))
    status('BATCH_FINISHED',scientific_status='PASS' if passed else 'INCOMPLETE')


if __name__=='__main__':main()
