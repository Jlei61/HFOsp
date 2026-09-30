"""Paired-step dominant Floquet checks on both sides of the new H1 fold.

The two sides are positions along the cycle branch, not opposite J values.
An unstable leading mode is sufficient to reject stability; failure to find
one with four requested multipliers cannot establish full stability.
"""
from rate_periodic import *
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0)
    p.add_argument('--min-free-gib',type=float,default=8.)
    p.add_argument('--include-root',action='store_true',help='Also check dominant instability at the exact validated fold')
    p.add_argument('--fold-number',type=int,choices=[1,2,3],default=2)
    p.add_argument('--root-only',action='store_true',help='Bound this check to the exact validated root')
    a=p.parse_args()
    number=a.fold_number;label=f'LPC_A_stage4_turn{number}'
    worker=PERIODIC_OUT/f'H1_stage4_fold{number}_neighbors_worker.json';rows=[]
    root=read(PERIODIC_OUT/(label+'_validation.json'))
    assert root['status']=='VALIDATED_CYCLE_FOLD'
    segment=read(PERIODIC_OUT/'arcAconnectionStage4_accuracy.json')
    assert segment['status']=='SAMPLED_PASS'
    def record(status,**kw):
        write(worker,dict(status=status,pid=os.getpid(),rows=rows,**kw))
    def gate():
        while True:
            free=float(subprocess.check_output(['nvidia-smi','-i',str(a.device),
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
            if free>=a.min_free_gib*1024:return
            record('WAITING_GPU_RESOURCE',free_mib=free);time.sleep(30)
    indices=[] if a.root_only else {1:[38,40],2:[50,52],3:[133,135]}[number]
    if a.include_root or a.root_only:indices.append('fold')
    for index in indices:
        path=Path(root['mesh_checks'][-1]['orbit'] if index=='fold' else segment['included_orbits'][index]);checks=[]
        # Each selected profile has already been evaluated off-grid in the
        # accepted segment. Keep its physical-profile evidence explicit.
        profile=root['continuous_defect'] if index=='fold' else next(q for q in segment['checks'] if q['index']==index)
        assert profile['filter_state_check']['positive'] and profile['maximum_group_defect_Hz']<.001
        for dt in [.05,.025]:
            result=PERIODIC_OUT/'floquet'/f'{path.stem}_dt{dt:g}.json'
            if not result.exists():
                gate();log=PERIODIC_OUT/f'H1_fold{number}_neighbor_{index}_dt{dt:g}.log'
                with log.open('w') as out:
                    child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).parent/'rate_floquet.py'),
                        str(path),'--dt',str(dt),'--nev','1' if index=='fold' else '4','--device',str(a.device)],stdout=out,stderr=subprocess.STDOUT)
                    record('COMPUTING',index=index,dt=dt,child_pid=child.pid,log=str(log));code=child.wait()
                if code:record('COMPUTATION_FAILED',index=index,dt=dt,exit_code=code,log=str(log));return
            q=read(result);mu=np.asarray(q['multipliers']);mu=mu[:,0]+1j*mu[:,1]
            k=int(np.argmax(abs(mu)))
            checks.append(dict(source=str(result),dt_ms=q['dt_ms'],multiplier=mu[k],
                relative_eigenpair_residual=q['residuals'][k]/max(1,abs(mu[k])),
                phase_tangent_relative_defect=q['phase_tangent_relative_defect']))
        mu0=complex(*checks[0]['multiplier']) if isinstance(checks[0]['multiplier'],list) else checks[0]['multiplier']
        mu1=complex(*checks[1]['multiplier']) if isinstance(checks[1]['multiplier'],list) else checks[1]['multiplier']
        change=abs(mu1-mu0)/max(1,abs(mu1))
        unstable=all(abs(v['multiplier'])>1.1 and v['relative_eigenpair_residual']<1e-6 for v in checks) and change<.01
        rows.append(dict(index=index,orbit=str(path),J_EE_core=profile['J_EE_core'],T_ms=profile['T_ms'],
            checks=checks,relative_leading_multiplier_change=change,
            status='DOMINANT_INSTABILITY_VERIFIED' if unstable else 'FULL_STABILITY_UNRESOLVED',
            profile_check_source=str(PERIODIC_OUT/(label+'_validation.json' if index=='fold' else 'arcAconnectionStage4_accuracy.json'))))
        record('POINT_FINISHED',index=index)
    write(PERIODIC_OUT/f'H1_stage4_fold{number}_neighbor_stability.json',dict(status='BOUNDED_CHECK_FINISHED',rows=rows,
        critical_point_source=str(PERIODIC_OUT/(label+'_validation.json')),
        exact_root_included=a.include_root or a.root_only,
        scope='Paired time-step leading Floquet evidence at the listed points, including the exact validated root when requested. Instability is point-specific; neither all unstable multipliers nor intervening crossings are counted.'))
    record('BATCH_FINISHED')


if __name__=='__main__':main()
