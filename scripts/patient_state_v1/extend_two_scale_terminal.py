"""One bounded continuation of the existing two-scale sampler, never concurrent writers."""
import sys,time,json,subprocess,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.patient_state_v1.common import ROOT,RUN,write_json
import numpy as np
OUT=RUN/'two_scale_terminal_extension';OUT.mkdir(exist_ok=True)
folder=RUN/'two_scale_particle_posterior_v1_15_16chains';oldpid=3050472;deadline=1788968996+8.8*3600
assert not (OUT/'control.json').exists(), 'One extension only'
def live(pid):
    try:return b'two_scale_posterior.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
    except FileNotFoundError:return False
while live(oldpid) and time.time()<deadline:time.sleep(15)
meta=json.loads((folder/'checkpoint.json').read_text())
if live(oldpid) or meta['status']!='COMPLETE' or meta['iteration']!=2000 or time.time()>deadline-900:
    write_json(OUT/'control.json',dict(status='NO_EXTENSION',old_pid=oldpid,metadata=meta));sys.exit(0)
subprocess.run([sys.executable,str(ROOT/'scripts/patient_state_v1/review_posteriors.py')],check=True)
r=next(r for r in json.loads((RUN/'posterior_diagnostics/status.json').read_text())['results'] if r['model']=='two_ou_history')
if r['acceptance_gate_pass']:
    write_json(OUT/'control.json',dict(status='NO_EXTENSION_DIAGNOSTICS_PASS',old_pid=oldpid,diagnostic=r));sys.exit(0)
for n in ['checkpoint.npz','checkpoint.json','contract.json']:shutil.copy2(folder/n,OUT/('before_'+n))
z=dict(np.load(OUT/'before_checkpoint.npz'));assert len(z['samples'])==2001 and np.isfinite(z['samples']).all() and np.isfinite(z['loglikes']).all()
with (OUT/'run.log').open('w') as log:
    p=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/patient_state_v1/two_scale_posterior.py'),'--gpu','0','--chains','16','--iterations','2400','--warmup','600','--output-name','two_scale_particle_posterior_v1_15_16chains'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
write_json(OUT/'control.json',dict(status='RESUMED',old_pid=oldpid,new_pid=p.pid,from_iteration=2000,target_iteration=2400,hard_end_unix=deadline,reason='One fixed continuation because terminal2000draws failed the prespecified posterior numerical gate; same target/RNG/covariance/warmup and unchanged8.8hdeadline',diagnostic_at2000=r))
time.sleep(45)
a=dict(np.load(folder/'checkpoint.npz'));m=json.loads((folder/'checkpoint.json').read_text());c=json.loads((folder/'contract.json').read_text())
assert np.array_equal(a['samples'][:2001],z['samples']) and np.array_equal(a['loglikes'][:2001],z['loglikes']) and np.array_equal(a['accepted'][:2000],z['accepted']) and np.array_equal(a['proposal_cov'],z['proposal_cov']);assert c['hard_end_unix']==deadline and c['warmup']==600
write_json(OUT/'resume_validation.json',dict(status='PASS',all_saved_sample_likelihood_and_acceptance_prefixes_identical=True,proposal_covariance_unchanged=True,warmup_unchanged=True,hard_deadline_unchanged=True,new_pid=p.pid,new_iteration=m['iteration']))
print(json.dumps(dict(status='RESUMED_AND_VALIDATED',new_pid=p.pid)),flush=True)
