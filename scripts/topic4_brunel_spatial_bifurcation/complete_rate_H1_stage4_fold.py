"""Verify the new H1-family fold after its running finer-mesh solve."""
from rate_periodic import *
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--after-pid',type=int,required=True)
    p.add_argument('--device',type=int,default=0)
    p.add_argument('--label',choices=['LPC_A_stage4_turn1','LPC_A_stage4_turn2','LPC_A_stage4_turn3'],default='LPC_A_stage4_turn1')
    p.add_argument('--refine-from-N',type=int)
    p.add_argument('--locate-coarse',action='store_true',help='Locate the selected checked stage4 turn before mesh refinement')
    a=p.parse_args()
    label=a.label;number=int(label[-1]);worker=PERIODIC_OUT/('H1_stage4_fold_worker.json' if number==1 else f'H1_stage4_fold{number}_worker.json')
    def record(status,**kw):write(worker,dict(status=status,pid=os.getpid(),label=label,**kw))
    proc=Path(f'/proc/{a.after_pid}/cmdline');identity=proc.read_bytes() if proc.exists() else None
    while identity:
        try:active=proc.read_bytes()==identity
        except FileNotFoundError:active=False
        if not active:break
        record('WAITING_FINE_ROOT',dependency=a.after_pid);time.sleep(30)
    scripts=Path(__file__).parent
    if a.locate_coarse:
        assert a.refine_from_N and a.refine_from_N<2048
        coarse=PERIODIC_OUT/f'{label}_N{a.refine_from_N}.json'
        if not coarse.exists():
            segment=read(PERIODIC_OUT/'arcAconnectionStage4_accuracy.json')
            assert segment['status']=='SAMPLED_PASS'
            turn=segment['turns'][number-1]
            assert read(PERIODIC_OUT/'cached_host_fold_solver_check.json')['status']=='PASS'
            while True:
                free=float(subprocess.check_output(['nvidia-smi','-i',str(a.device),
                    '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
                if free>=10*1024:break
                record('WAITING_COARSE_ROOT_RESOURCE',free_mib=free,required_free_gib=10);time.sleep(30)
            log=PERIODIC_OUT/f'{label}_N{a.refine_from_N}_location.log'
            with log.open('w') as out:
                child=subprocess.Popen([sys.executable,'-u',str(scripts/'rate_cycle_fold_mean.py'),
                    turn['left'],turn['right'],'--label',label,'--N',str(a.refine_from_N),
                    '--core','A','--device',str(a.device),'--low-memory','--linear-normalize',
                    '--cached-host-krylov','--linear-rtol-cap','.00001'],stdout=out,stderr=subprocess.STDOUT)
                record('LOCATING_COARSE_ROOT',child_pid=child.pid,log=str(log));code=child.wait()
            if code:record('COARSE_ROOT_COMPUTATION_FAILED',exit_code=code,log=str(log));return
    fine=PERIODIC_OUT/f'{label}_N2048.json'
    if not fine.exists() and a.refine_from_N:
        coarse=PERIODIC_OUT/f'{label}_N{a.refine_from_N}.json'
        if not coarse.exists():record('COARSE_ROOT_NOT_AVAILABLE');return
        assert read(PERIODIC_OUT/'cached_host_fold_solver_check.json')['status']=='PASS'
        while True:
            free=float(subprocess.check_output(['nvidia-smi','-i',str(a.device),
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
            if free>=10*1024:break
            record('WAITING_FINE_ROOT_RESOURCE',free_mib=free,required_free_gib=10);time.sleep(30)
        source=read(coarse)['orbit'];log=PERIODIC_OUT/f'{label}_N2048_refinement.log'
        with log.open('w') as out:
            child=subprocess.Popen([sys.executable,'-u',str(scripts/'rate_cycle_fold_mean.py'),
                source,source,'--label',label,'--N','2048','--core','A','--device',str(a.device),
                '--low-memory','--linear-normalize','--cached-host-krylov',
                '--linear-rtol-cap','.00001','--radius','.02','--tangent-predictor'],
                stdout=out,stderr=subprocess.STDOUT)
            record('REFINING_FINE_ROOT',child_pid=child.pid,log=str(log));code=child.wait()
        if code:record('FINE_ROOT_COMPUTATION_FAILED',exit_code=code,log=str(log));return
    if not fine.exists():
        record('FINE_ROOT_NOT_AVAILABLE');return
    for steps in [[.05,.025,.0125],[.05,.025,.0125,.00625],[.05,.025,.0125,.00625,.003125]]:
        while True:
            free=float(subprocess.check_output(['nvidia-smi','-i',str(a.device),
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
            if free>=8*1024:break
            record('WAITING_RESOURCE',free_mib=free,required_free_gib=8);time.sleep(30)
        log=PERIODIC_OUT/f'{label}_full_state_dt{steps[-1]:g}.log'
        with log.open('w') as out:
            child=subprocess.Popen([sys.executable,'-u',str(scripts/'validate_rate_mean_fold.py'),label,
                '--device',str(a.device),'--dt',*map(str,steps)],stdout=out,stderr=subprocess.STDOUT)
            record('VERIFYING',child_pid=child.pid,log=str(log));code=child.wait()
        if code:record('VERIFICATION_COMPUTATION_FAILED',exit_code=code,log=str(log));return
        q=read(PERIODIC_OUT/(label+'_validation.json'))
        if q['status']=='VALIDATED_CYCLE_FOLD':
            record('VALIDATED_CYCLE_FOLD',source=str(PERIODIC_OUT/(label+'_validation.json')));return
        checks=q['continuous_defect']
        if not checks['filter_state_check']['positive'] or checks['maximum_group_defect_Hz']>=.001:
            record('PARENT_PROFILE_REFINEMENT_REQUIRED',source=str(PERIODIC_OUT/(label+'_validation.json')));return
    record('MODE_CHECK_UNRESOLVED',source=str(PERIODIC_OUT/(label+'_validation.json')),
        scope='Preserve the located candidate; do not promote it or infer absence from the incomplete check.')


if __name__=='__main__':main()
