"""Bounded continuation of the two unresolved Hopf-family endpoints.

Preserve the existing 935-group equations and geometry. Each new segment is
checked independently before becoming eligible for figure inclusion. Turns
and numerical stops are not classified as bifurcations by this worker.
"""
from rate_periodic import *
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=0)
    p.add_argument('--steps',type=int,default=80);p.add_argument('--min-free-gib',type=float,default=10.)
    p.add_argument('--stage',type=int,choices=[1,2,3],default=1)
    a=p.parse_args();script=Path(__file__).resolve().parent;rows=[]
    status=PERIODIC_OUT/('global_connections_worker.json' if a.stage==1 else f'global_connections_stage{a.stage}_worker.json')
    tasks=[('arcAconnectionFurther','arcAglobalConnection',512,.6),
           ('arcBconnectionFurther','arcBtoBurstFurther',256,.25)]
    if a.stage==2:
        tasks=[('arcAconnectionNext','arcAconnectionFurther',512,.6),
               ('arcBconnectionNext','arcBconnectionFurther',256,.25)]
    if a.stage==3:
        tasks=[('arcAconnectionStage3','arcAconnectionNext',512,.6),
               ('arcBconnectionStage3','arcBconnectionNext',256,.25)]
    for label,prior,N,ds in tasks:
        assert not (PERIODIC_OUT/(label+'_continuation.json')).exists(), 'Do not overwrite an existing segment'
        while True:
            free=float(subprocess.check_output(['nvidia-smi','-i',str(a.device),
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
            if free>=a.min_free_gib*1024:break
            write(status,dict(pid=os.getpid(),status='WAITING_RESOURCE',label=label,free_mib=free,rows=rows))
            print('WAITING RESOURCE',label,free,flush=True);time.sleep(30)
        prior_check=read(PERIODIC_OUT/(prior+'_accuracy.json'))
        assert prior_check['status']=='SAMPLED_PASS'
        paths=sorted((PERIODIC_OUT/'orbits').glob(f'{prior}_[0-9][0-9][0-9][0-9]_N{N}.npz'))[-2:]
        assert len(paths)==2
        assert all(path.exists() for path in paths)
        write(status,dict(pid=os.getpid(),status='CONTINUATION',label=label,requested_steps=a.steps,rows=rows))
        subprocess.run([sys.executable,str(script/'rate_periodic_continue.py'),*map(str,paths),
            '--N',str(N),'--steps',str(a.steps),'--ds',str(ds),'--label',label,
            '--device',str(a.device),'--low-memory','--linear-normalize'],check=True)
        write(status,dict(pid=os.getpid(),status='CONTINUOUS_CHECKS',label=label,rows=rows))
        subprocess.run([sys.executable,str(script/'check_rate_extension.py'),label,
            '--device',str(a.device),'--stride','10','--prior-segment',prior],check=True)
        q=read(PERIODIC_OUT/(label+'_accuracy.json'))
        rows.append(dict(label=label,status=q['status'],points=q['continued_points'],
            candidate_turns=len(q['turns']),source=str(PERIODIC_OUT/(label+'_accuracy.json'))))
        print('SEGMENT FINISHED',rows[-1],flush=True)
    write(status,dict(pid=os.getpid(),status='BATCH_FINISHED',rows=rows,
        scope='Bounded continuation and sampled accuracy; global branch identity and Floquet crossing completeness are separate checks.'))


if __name__=='__main__':main()
