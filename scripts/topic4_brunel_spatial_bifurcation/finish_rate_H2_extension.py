"""Check the completed H2 segment and refine the preceding two missing turns."""
from rate_periodic import *
import subprocess

DEST=Path('/data/hfosp/topic4_sef_hfo/interictal_rate_branch_completion_20260920')


def main():
    p=argparse.ArgumentParser();p.add_argument('--after-pid',type=int,required=True)
    p.add_argument('--device',type=int,default=1);a=p.parse_args()
    scripts=Path(__file__).parent;worker=DEST/'H2_checks_worker.json';rows=[]
    def status(state,**kw):write(worker,dict(status=state,pid=os.getpid(),rows=rows,**kw))
    proc=Path(f'/proc/{a.after_pid}/cmdline');identity=proc.read_bytes() if proc.exists() else None
    while identity and proc.exists() and proc.read_bytes()==identity:
        status('WAITING_DEPENDENCY',dependency=a.after_pid);time.sleep(30)
    label='arcBconnectionStage4_20260920'
    def run(command,stage):
        log=DEST/f'H2_{stage}_{time.time_ns()}.log'
        with log.open('w') as f:
            child=subprocess.Popen([sys.executable,'-u',*map(str,command)],stdout=f,stderr=subprocess.STDOUT)
            status('RUNNING',stage=stage,child_pid=child.pid,log=str(log));code=child.wait()
        rows.append(dict(stage=stage,returncode=code,log=str(log)))
        if code:raise RuntimeError(f'{stage} failed: {log}')
    try:
        previous=PERIODIC_OUT/(label+'_accuracy.json')
        expected=[q['path'] for q in read(PERIODIC_OUT/(label+'_continuation.json'))['rows']]
        if not previous.exists() or read(previous).get('included_orbits')!=expected:
            run([scripts/'check_rate_extension.py',label,'--device',a.device,'--stride','1',
                 '--check-filter-states','--prior-segment','arcBconnectionStage3','--prior-orbits',
                 *[PERIODIC_OUT/'orbits'/f'arcBconnectionStage3_{i:04d}_N256_accuracy_N2048.npz' for i in [98,99]]],
                'all_profile_check')
        check=read(PERIODIC_OUT/(label+'_accuracy.json'))
        write(DEST/'H2_extension_check.json',check)
        if check['status']!='SAMPLED_PASS':
            status('PHYSICAL_REFINEMENT_REQUIRED');return
        run([scripts/'screen_rate_branch_encounters.py','--accepted-segment',label,
             '--provisional-family','B','--include-higher-hopfs'],'encounter_screen')
        run([scripts/'refine_rate_remaining_turns.py','BS1','BS2','--device',a.device],
            'preceding_turns')
        status('BATCH_FINISHED')
    except Exception as exc:
        status('STOPPED_WITH_ERROR',error=repr(exc));raise


if __name__=='__main__':main()
