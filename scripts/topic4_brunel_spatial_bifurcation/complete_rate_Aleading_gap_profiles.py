"""Prepare selected, already scheduled A-leading survey profiles for reuse.

Separate waveform correction from the queued full-spectrum calculation.
Every correction keeps the original J and all spatial populations/delays.
"""
from complete_rate_positive_stability import *
import subprocess


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',type=int,default=1)
    parser.add_argument('--min-free-gib',type=float,default=6.)
    parser.add_argument('--indices',type=int,nargs='+',default=[45,44,42])
    parser.add_argument('--output-name',default='corrected_profiles')
    parser.add_argument('--worker-label',default='correction_worker')
    args=parser.parse_args()
    assert Path(args.output_name).name==args.output_name and Path(args.worker_label).name==args.worker_label
    folder=DEST/'Aleading_profile_gaps';folder.mkdir(exist_ok=True)
    worker=folder/(args.worker_label+'.json');plan=read(DEST/'frozen_survey_plan.json');rows=[]
    def status(state,**kw):
        write(worker,dict(status=state,pid=os.getpid(),timestamp=time.time(),completed=len(rows),**kw));print(state,kw,flush=True)
    s=RateField()
    for index in args.indices:
        output=folder/f'corrected_site_{index:03d}.json'
        if output.exists() and read(output).get('status')=='SAME_J_PHYSICAL_PROFILE_CHECKED':
            rows.append(read(output));continue
        item=plan['rows'][index];origin=Path(item['orbit'])
        def gate(mesh=0,stage='PROFILE'):
            release(args.device);required=max(args.min_free_gib,8. if mesh>=8192 else 6.)
            while True:
                free=float(subprocess.check_output(['nvidia-smi','-i',str(args.device),
                    '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
                if free>=required*1024:return
                status('WAITING_GPU_RESOURCE',index=index,N=mesh,stage=stage,
                    free_mib=free,required_free_gib=required);time.sleep(30)
        start=best_cached(item);gate();status('CORRECTING_SAME_J_PROFILE',index=index,source=str(start))
        actual,check=prepare(start,args.device,max_N=8192,min_N=2048,
            check_filter_states=True,harmonic_chunk_size=32,stream_harmonics=True,
            host_krylov=True,before_mesh=gate)
        assert check['status']=='RESOLUTION_CHECKED' and check['filter_state_check']['positive']
        assert check['maximum_group_defect_Hz']<.001
        coarse=np.load(origin);fine=np.load(actual);rr=resample(coarse['r'],len(fine['r']),axis=0)
        drift=float(np.linalg.norm(fine['r']-rr)/np.linalg.norm(rr-rr.mean(0)))
        period=float(abs(float(fine['T']/coarse['T'])-1))
        assert abs(float(fine['J'])-item['J_EE_core'])<1e-12 and drift<.02 and period<.001
        result=dict(status='SAME_J_PHYSICAL_PROFILE_CHECKED',index=index,
            original_orbit=str(origin),orbit=str(actual),resolution=check,
            relative_waveform_change=drift,relative_period_change=period,
            memberships=item['memberships'],
            scope='Physical periodic profile only. The queued original Floquet survey must compute stability on this corrected profile.')
        write(output,result);rows.append(result);release(args.device)
    write(folder/(args.output_name+'.json'),dict(
        status='THREE_SAME_J_PROFILES_CHECKED' if args.indices==[45,44,42] else 'SELECTED_SAME_J_PROFILES_CHECKED',
        selected_indices=args.indices,rows=rows,
        scope='Existing frozen survey points; unchanged spatial model and parameter values. No new critical point or stability claim.'))
    status('COMPLETE')


if __name__=='__main__':main()
