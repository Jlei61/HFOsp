import time,json,os,signal,subprocess
from pathlib import Path
import numpy as np
root=Path('/home/honglab/leijiaxin/HFOsp');r=root/'results/topic5_patient_state_inference/e1146_drift_hypothesis_v1/overnight_20260909';old=r/'two_scale_particle_posterior_v1_15';new=r/'two_scale_particle_posterior_v1_15_16chains'
while True:
 m=json.loads((old/'checkpoint.json').read_text())
 if m['iteration']>=25:break
 os.kill(3049730,0);time.sleep(5)
os.kill(3049730,signal.SIGTERM);time.sleep(1)
m=json.loads((old/'checkpoint.json').read_text());z=dict(np.load(old/'checkpoint.npz'));assert len(z['samples'])==m['iteration']+1
new.mkdir(exist_ok=True)
for key in ['samples','loglikes','accepted']:z[key]=z[key][:,:16]
np.savez_compressed(new/'checkpoint.npz',**z);m['chains']=16;m['warmup']=600;m['status']='READY_TO_RESUME_RUNTIME_ADJUSTMENT';(new/'checkpoint.json').write_text(json.dumps(m,indent=2));(new/'runtime_adjustment.json').write_text(json.dumps(dict(reason='Measured 128-filter kernel takes about twice the time of 64 filters; 32 chains x 4 filters would leave too few post-warmup iterations within the goal',preserved_source=str(old),selected_chains='First 16 chain IDs fixed without inspecting scores or samples',likelihood_unchanged='Four independent 1024-particle filters per chain',new_iterations=2000,new_warmup=600,source_iteration=m['iteration']),indent=2));oldm=json.loads((old/'checkpoint.json').read_text());oldm['status']='STOPPED_RUNTIME_PILOT_CONTINUED_FIRST16';(old/'checkpoint.json').write_text(json.dumps(oldm,indent=2))
env=dict(os.environ,LD_LIBRARY_PATH='/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
p=subprocess.Popen(['/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python','-u','scripts/patient_state_v1/two_scale_posterior.py','--chains','16','--iterations','2000','--warmup','600','--output-name',new.name],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=open(r/'logs/two_scale_posterior_16chains.log','w'),stderr=subprocess.STDOUT,start_new_session=True);(new/'launch.json').write_text(json.dumps(dict(pid=p.pid,started=time.time()),indent=2));print(p.pid,flush=True)
