"""Refine the closest cached encounter targets before interpreting distances."""
from complete_rate_positive_stability import *
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0)
    p.add_argument('--min-free-gib',type=float,default=3.5);a=p.parse_args()
    folder=DEST/'Bleading_extension';worker=folder/'nearest_targets_worker.json'
    screen=PERIODIC_OUT/'arcBleadingConnection_20260920_prefix17_encounter_N4096.json'
    plan=read(screen);assert plan['status']=='SCREEN_COMPLETE'
    s=RateField();weights=s.geo['group_size']/s.geo['group_size'].sum();results=[]
    def status(state,**kw):
        write(worker,dict(status=state,pid=os.getpid(),timestamp=time.time(),**kw))
        print(state,kw,flush=True)
    def gate(N,stage):
        required=max(a.min_free_gib,6. if stage=='NEWTON' and N>=4096 else 0.)
        while True:
            free=float(subprocess.check_output(['nvidia-smi','-i',str(a.device),
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
            if free>=required*1024:return
            status('WAITING_RESOURCE',N=N,stage=stage,free_mib=free,required_gib=required);time.sleep(30)
    for target in ['B','single']:
        pair=next(q for q in plan['rows'] if q['families']==['Bleading',target])['nearest'][0]
        origin=Path(pair['second_orbit']);meta=read(origin.with_suffix('.json'))
        start=best_cached(dict(orbit=str(origin),J_EE_core=meta['J_EE_core'],T_ms=meta['T_ms']))
        release(a.device);status('CHECKING_SAME_J_TARGET',family=target,source=str(origin),start=str(start))
        actual,check=prepare(start,a.device,max_N=4096,host_krylov=True,
            check_filter_states=True,harmonic_chunk_size=32,stream_harmonics=True,before_mesh=gate)
        if check['status']=='RESOLUTION_CHECKED' and check['maximum_group_defect_Hz']>=.001:
            actual,check=prepare(actual,a.device,max_N=4096,min_N=2*check['N'],host_krylov=True,
                check_filter_states=True,harmonic_chunk_size=32,stream_harmonics=True,before_mesh=gate)
        before,after=np.load(origin),np.load(actual)
        x,y=[resample(z['r']*1000,4096,axis=0) for z in [before,after]]
        drift,_=distances(x[:,None,:],y,weights)
        amplitude=lambda r:float(np.sqrt(np.mean(np.sum((r-r.mean(0))**2*weights,axis=1))))
        relative_drift=float(drift[0]/amplitude(x))
        period_drift=abs(float(after['T']/before['T'])-1)
        matched=(check['status']=='RESOLUTION_CHECKED' and check['maximum_group_defect_Hz']<.001
            and abs(float(after['J']-before['J']))<1e-12 and relative_drift<.02 and period_drift<.01)
        result=dict(family=target,original_pair=pair,corrected_target=str(actual),resolution=check,
            relative_waveform_change=relative_drift,relative_period_change=period_drift,
            same_branch_refinement_pass=bool(matched))
        if matched:
            source=np.load(pair['first_orbit']);r=resample(source['r']*1000,4096,axis=0)
            distance,phase=distances(r[:,None,:],y,weights)
            result.update(full_group_RMS_difference_Hz=float(distance[0]),
                relative_waveform_difference=float(distance[0]/max(amplitude(r),amplitude(y))),
                common_phase_cycles=float(phase[0]),target_J_EE_core=float(after['J']),
                target_T_ms=float(after['T']),status='MATCHED_TARGET_RECHECKED')
        else:result['status']='TARGET_REFINEMENT_REVIEW_REQUIRED'
        results.append(result)
        write(folder/'nearest_target_refinement.json',dict(status='COMPLETE' if len(results)==2 else 'RUNNING',
            screen=str(screen),rows=results,
            scope='Two fixed nearest candidates from the earlier cached screen, corrected at their own unchanged J; source and target parameters are close but not identical. This tests numerical robustness of those distances only. It is not a new global search or an exclusion of unseen connections.'))
        status(result['status'],family=target,relative_waveform_change=relative_drift,
               revised_distance=result.get('relative_waveform_difference'))
    release(a.device);status('BATCH_FINISHED')


if __name__=='__main__':main()
