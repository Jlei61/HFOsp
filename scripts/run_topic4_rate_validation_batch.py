#!/usr/bin/env python3
"""Complete the bounded V1 diagnostic batch; never dispatch V2/Hopf automatically."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess
import sys
import time
import json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/topic4_sef_hfo/rate_model_dynamics_validation_v1'
PYTHON='/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python'


def work(job):
    name,site,dose=job;logdir=OUT/'logs';logdir.mkdir(exist_ok=True)
    env=dict(os.environ,LD_LIBRARY_PATH='/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with (logdir/(name+'.log')).open('a') as log:
        if not (OUT/'snn'/f'{name}.json').exists():
            subprocess.run([PYTHON,str(ROOT/'scripts/validate_topic4_fixed_rate_base.py'),'snn',
                '--name',name,'--site',site,'--dose',str(dose),'--duration','2400'],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        for grid in (10,20):
            if not (OUT/'rate'/f'{name}_grid{grid}_cascade.json').exists():
                subprocess.run([PYTHON,str(ROOT/'scripts/run_topic4_rate_validation.py'),name,
                    '--grid',str(grid),'--closure','cascade'],cwd=ROOT,env=env,
                    stdout=log,stderr=subprocess.STDOUT,check=True)
    return name


def main():
    jobs=[('sham_7101','A',0.),('A_dose4','A',4.),('A_dose0p25','A',.25),('A_dose1','A',1.),
          ('B_dose0p25','B',.25),('B_dose1','B',1.),('B_dose4','B',4.),('surround_dose4','surround',4.)]
    passed=[];start=time.time()
    def save(status,error=None):
        p=OUT/'v1_batch_status.json';temp=p.with_suffix('.tmp');temp.write_text(json.dumps({
            'status':status,'completed':passed,'total':len(jobs),'elapsed_seconds':time.time()-start,
            'max_workers':2,'error':error,'V2_released':False,'V3_released':False,'Hopf_released':False},indent=2)+'\n');temp.replace(p)
    prefix=json.loads((OUT/'snn/prefix_7101.json').read_text())['prefix_check']
    assert prefix['native_counts_equal'] and prefix['active_fraction_equal_at_saved_precision']
    save('RUNNING_V1_DIAGNOSTICS')
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(work,j):j for j in jobs}
        try:
            for future in as_completed(futures):
                passed.append(future.result());save('RUNNING_V1_DIAGNOSTICS');print('completed',passed[-1],flush=True)
        except Exception as exc:
            for future in futures:future.cancel()
            save('ENGINEERING_ERROR_NO_FURTHER_DISPATCH',repr(exc));raise
    save('V1_BATCH_COMPLETE_PENDING_REDUCTION_REVIEW')
    subprocess.run([PYTHON,str(ROOT/'scripts/analyze_topic4_rate_validation.py')],cwd=ROOT,check=True)


if __name__=='__main__':main()
