"""Checkpoint the bounded remaining-turn refinements in the frozen model."""
from common import *
from rate_periodic import PERIODIC_OUT
from validate_rate_mean_fold import validation_matches_latest_root
import subprocess

CASES={
 'A1':('arcAreturnStrong',130,133,'A','LPC_A_large_return1'),
 'A2':('arcAreturnStrong',144,146,'A','LPC_A_large_return2'),
 'A3':('arcAreturnStrong',151,153,'B','LPC_A_large_return3'),
 'A4':('arcAglobalConnection',59,61,'A','LPC_A_global_turn1'),
 'S1':('arcSingleUpperConnect',9,11,'B','LPC_single_upper1'),
 'S2':('arcSingleUpperConnect',19,21,'B','LPC_single_upper2'),
 'S3':('arcSingleUpperConnect',43,45,'B','LPC_single_upper3'),
 'S4':('arcSingleUpperConnect',60,62,'B','LPC_single_upper4'),
}

# Freeze brackets from the checked continuation segments. The coordinate is
# whichever core mean changes more across the local bracket; no turn is
# promoted merely by entering this catalogue.
SOURCES={}
# Refine existing primary burst folds whose stored coarse profiles or
# curvature have not reached the common acceptance tolerances. These are
# the same named roots, not additional candidate bifurcations.
for key,label,core,N in [('BL','LPC_Bleading_low','A',1024),
                        ('AL','LPC_burst_low','B',512),
                        ('AH','LPC_burst_high','A',512),
                        ('DL','LPC_double_low','B',1536),
                        ('DH','LPC_double_high','A',1536),
                        ('DS1','LPC_double_secondary1','B',2048),
                        ('DS2','LPC_double_secondary2','B',2048),
                        ('DS3','LPC_double_secondary3','B',1024),
                        ('DS4','LPC_double_secondary4','B',2048)]:
    path=PERIODIC_OUT/'orbits'/f'{label}_N{N}.npz'
    CASES[key]=('legacy_primary_burst',None,None,core,label)
    SOURCES[key]=(path,path,N,.02)
for key,label in [('LU','LPC_resonance_upper'),('LL','LPC_resonance_lower')]:
    path=PERIODIC_OUT/'orbits'/f'{label}_N128.npz'
    CASES[key]=('legacy_resonance',None,None,'B',label)
    SOURCES[key]=(path,path,256,.00005)
for segment,key_prefix,label_prefix in [
    ('arcSingleUpperConnect','S','LPC_single_upper'),
    ('arcBtoBurst','B','LPC_B_burst_turn'),
    ('arcBtoBurstFurther','BF','LPC_B_further_turn'),
    ('arcAconnectionFurther','AC','LPC_A_connection'),
    ('arcBconnectionFurther','BC','LPC_B_connection'),
    ('arcAconnectionNext','AN','LPC_A_next'),
    ('arcBconnectionNext','BN','LPC_B_next'),
    ('arcBconnectionStage3','BS','LPC_B_stage3_turn'),
    ('arcBconnectionStage4_20260920','B4','LPC_B_stage4_turn')]:
    source=PERIODIC_OUT/(segment+'_accuracy.json')
    if not source.exists():continue
    checked=read(source)
    if checked['status']!='SAMPLED_PASS':continue
    for i,turn in enumerate(checked['turns'],1):
        key=key_prefix+str(i)
        if key in CASES:continue
        left,right=[read(Path(turn[k]).with_suffix('.json')) for k in ['left','right']]
        change=np.array(right['mean_rates_hz'][:2])-np.array(left['mean_rates_hz'][:2])
        core='AB'[int(np.argmax(abs(change)))];label=label_prefix+str(i)
        CASES[key]=(segment,None,None,core,label)
        # These accepted H2 profiles were repaired individually to N2048.
        # Locate the same local root at 1024 and 2048; the original N256
        # continuation mesh is not sufficient critical-point evidence.
        root_N=1024 if segment=='arcBconnectionStage3' else left['N']
        SOURCES[key]=(Path(turn['left']),Path(turn['right']),root_N,
                      min(.02,max(1e-6,.2*max(abs(change)))))


def main():
    import argparse,shutil
    p=argparse.ArgumentParser();p.add_argument('cases',nargs='+',choices=list(CASES));p.add_argument('--device',type=int,default=0)
    p.add_argument('--min-free-gib',type=float,default=0.)
    p.add_argument('--after-pid',type=int,nargs='*',default=[])
    p.add_argument('--cached-host-krylov',action='store_true')
    p.add_argument('--linear-rtol-cap',type=float,default=.02)
    p.add_argument('--tangent-predictor',action='store_true')
    a=p.parse_args();folder=PERIODIC_OUT/'remaining_turn_refinements';folder.mkdir(exist_ok=True)
    status=folder/('worker_'+'_'.join(a.cases)+'.json');rows=[];scripts=Path(__file__).parent
    dependencies={pid:Path(f'/proc/{pid}/cmdline').read_bytes() for pid in a.after_pid if Path(f'/proc/{pid}/cmdline').exists()}
    while dependencies:
        for pid,identity in list(dependencies.items()):
            try:active=Path(f'/proc/{pid}/cmdline').read_bytes()==identity
            except FileNotFoundError:active=False
            if not active:dependencies.pop(pid)
        if dependencies:
            write(status,dict(status='WAITING_DEPENDENCIES',pid=os.getpid(),dependencies=list(dependencies),rows=rows));time.sleep(30)
    def launch(cmd,label,stage):
        if a.min_free_gib:
            while True:
                free=float(subprocess.check_output(['nvidia-smi','-i',str(a.device),
                    '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
                if free>=a.min_free_gib*1024:break
                write(status,dict(status='WAITING_GPU_RESOURCE',rows=rows,pid=os.getpid(),
                    label=label,stage=stage,free_mib=free,required_free_gib=a.min_free_gib));time.sleep(30)
        storage=Path(os.environ.get('HFOSP_RATE_ARRAY_STORAGE',str(PERIODIC_OUT)))
        storage.mkdir(parents=True,exist_ok=True)
        if (shutil.disk_usage(storage).free<750*1024**2 or
            shutil.disk_usage(PERIODIC_OUT).free<150*1024**2):
            write(status,dict(status='STOPPED_STORAGE_LIMIT',rows=rows,pid=os.getpid(),next_stage=stage));raise RuntimeError('Storage reserve reached; no calculation failure inferred')
        log=folder/(label+'_'+stage+'.log')
        if log.exists():log.replace(log.with_name(log.stem+f'_attempt_{time.time_ns()}.log'))
        with log.open('w') as output:
            child=subprocess.Popen([sys.executable,'-u',*map(str,cmd)],stdout=output,stderr=subprocess.STDOUT)
            write(status,dict(status='RUNNING',rows=rows,pid=os.getpid(),child_pid=child.pid,label=label,stage=stage,log=str(log)))
            code=child.wait()
        print('TURN STAGE',label,stage,'exit',code,flush=True);return code
    for key in a.cases:
        prefix,left,right,core,label=CASES[key];failed=False
        if key in SOURCES:
            first_source,second_source,low_N,radius=SOURCES[key]
        else:
            first_source,second_source=[PERIODIC_OUT/'orbits'/f'{prefix}_{i:04d}_N512.npz' for i in [left,right]]
            low_N=512;radius=.02
        for N in [low_N,2*low_N]:
            dest=PERIODIC_OUT/f'{label}_N{N}.json'
            if dest.exists():
                q=read(dest)
                if abs(q.get('dJ_dcoordinate',q.get('dJ_dlogT',float('inf'))))<1e-7:continue
            if N==low_N:
                first,second=first_source,second_source
                extra=['--radius',str(radius)] if first==second else []
            else:
                first=second=PERIODIC_OUT/'orbits'/f'{label}_N{low_N}.npz';extra=['--radius',str(radius)]
            if key in ['BC2','AH']:
                # The core means turn inside this bracket; period remains
                # monotone and is therefore a regular local coordinate.
                extra=[] if N==low_N else ['--radius','.5']
                cmd=[scripts/'rate_cycle_folds.py',first,second,'--label',label,'--N',N,
                     '--device',a.device,'--low-memory','--linear-normalize','--krylov-restart','180',*extra]
            else:
                if a.cached_host_krylov:
                    check=read(PERIODIC_OUT/'cached_host_fold_solver_check.json')
                    assert check['status']=='PASS'
                cmd=[scripts/'rate_cycle_fold_mean.py',first,second,'--label',label,'--N',N,'--core',core,
                     '--device',a.device,'--low-memory','--linear-normalize',
                     '--linear-rtol-cap',a.linear_rtol_cap,
                     *(['--cached-host-krylov'] if a.cached_host_krylov else ['--host-krylov'] if N>=3072 else []),
                     *(['--tangent-predictor'] if a.tangent_predictor else []),*extra]
            if launch(cmd,label,f'N{N}'):
                rows.append(dict(case=key,label=label,status='REFINEMENT_FAILED',N=N));failed=True;break
        if failed:continue
        validation=PERIODIC_OUT/(label+'_validation.json')
        if not validation_matches_latest_root(label):
            code=launch([scripts/'validate_rate_mean_fold.py',label,'--device',a.device],label,'mode_validation')
            if code:
                rows.append(dict(case=key,label=label,status='VALIDATION_COMPUTATION_FAILED'));continue
        rows.append(dict(case=key,label=label,status=read(validation)['status'],validation=str(validation)))
    write(status,dict(status='BATCH_FINISHED',rows=rows,pid=os.getpid(),scope='Batch completion is separate from the scientific status of each candidate.'))


if __name__=='__main__':main()
