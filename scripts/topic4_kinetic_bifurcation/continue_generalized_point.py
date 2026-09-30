"""One bounded parameter-continuation step from a corrected return point.

No automatic parameter campaign: prepares one physical Z(D) change and runs
one declared correction. An optional completed result serializes GPU work.
"""
from pathlib import Path
import argparse,json,os,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/topic4_sef_hfo/kinetic_population_bifurcation_20260916'


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def run(a):
    parent=json.load(open(a.corrected/'result.json'));assert parent['status']=='GENERALIZED_RETURN_CORRECTED'
    folder=OUT/'continuation_steps'/a.label;folder.mkdir(parents=True,exist_ok=False)
    period=round(parent['history'][-1]['effective_return_time_ms'],1)
    script=ROOT/'scripts/topic4_kinetic_bifurcation'
    seed=[sys.executable,'-u',str(script/'seed_generalized_continuation.py'),'--corrected',str(a.corrected.resolve()),
          '--D',str(a.D),'--label',a.label,'--device',str(a.device)]
    correction=[sys.executable,'-u',str(script/'correct_generalized_return.py'),'--source',str(OUT/'continuation_seeds'/a.label),
                '--label',a.label,'--period',str(period),'--search-radius',str(a.search_radius),
                '--order','5','--iterations',str(a.iterations),'--device',str(a.device)]
    write(folder/'config.json',dict(parent=str(a.corrected.resolve()),D=a.D,commands=[seed,correction],
        wait_for_result=str(a.wait_for) if a.wait_for else None,scope='One initial guess and one bounded full-map return correction; no stability/type inference'))
    start=time.time()
    if a.wait_for:
        write(folder/'status.json',dict(status='WAITING_FOR_COMPUTE_SLOT',pid=os.getpid(),wait_for=str(a.wait_for)))
        while not a.wait_for.exists():
            if time.time()-start>7200:
                write(folder/'status.json',dict(status='DEPENDENCY_WAIT_TIMEOUT',pid=os.getpid()));return
            time.sleep(10)
    for name,command in [('seed',seed),('correction',correction)]:
        write(folder/'status.json',dict(status='RUNNING',pid=os.getpid(),stage=name))
        with (folder/f'{name}.log').open('w') as f:
            process=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT)
            write(folder/'status.json',dict(status='RUNNING',pid=os.getpid(),stage=name,child_pid=process.pid))
            code=process.wait()
        if code:
            write(folder/'status.json',dict(status='COMMAND_FAILED',stage=name,returncode=code));return
    result=json.load(open(OUT/'generalized_corrections'/a.label/'result.json'))
    write(folder/'status.json',dict(status='BOUNDED_CONTINUATION_FINISHED',correction_status=result['status'],
        best_weighted_return_residual=result['best_weighted_return_residual'],wall_s=time.time()-start,
        next_step='Review root, section selection, interpolation and normal spectrum before any bifurcation classification'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--corrected',type=Path,required=True);ap.add_argument('--D',type=float,required=True)
    ap.add_argument('--label',required=True);ap.add_argument('--wait-for',type=Path);ap.add_argument('--device',type=int,default=0)
    ap.add_argument('--search-radius',type=float,default=10.);ap.add_argument('--iterations',type=int,default=12)
    run(ap.parse_args())
