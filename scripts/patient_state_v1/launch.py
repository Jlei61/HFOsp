"""Launch explicitly bounded patient-only CPU and GPU queues without touching other tasks."""
import sys,os,subprocess,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.patient_state_v1.common import ROOT,RUN,write_json

def main():
    env=os.environ.copy();env.update(LD_LIBRARY_PATH='/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1')
    logdir=RUN/'logs';logdir.mkdir(parents=True,exist_ok=True)
    tasks=[('real_cpu','cpu_jobs.py',['--wave','real','--workers','16']),
           ('synthetic_cpu','cpu_jobs.py',['--wave','synthetic','--workers','16']),
           ('gpu0_canary','gpu_particles.py',['--gpu','0','--particles','512','--replicates','32']),
           ('gpu1_canary','gpu_particles.py',['--gpu','1','--particles','1024','--replicates','32'])]
    rows=[]
    for name,script,args in tasks:
        cmd=[sys.executable,'-u',str(ROOT/'scripts/patient_state_v1'/script),*args]
        with (logdir/f'{name}.log').open('a') as log:
            p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        rows.append(dict(name=name,pid=p.pid,command=cmd,log=str(logdir/f'{name}.log')))
    write_json(RUN/'launch.json',dict(start_unix=time.time(),goal_start_unix=1788968996,target_end_unix=1789001396,hard_end_unix=1789004996,
                                    status='FIRST_WAVE_RUNNING',tasks=rows,scope='patient-state only; all SNN and Z/M frozen'))
    print(rows,flush=True)

if __name__=='__main__':main()
