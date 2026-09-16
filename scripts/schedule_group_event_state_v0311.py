#!/usr/bin/env python
"""Priority queue for the v0.3.11 compact development package.

Task identity carries subject, split, inputs, family, arm, ablation, seed and
the source version. Claims are atomic; only the claiming worker writes a card.
"""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

ROOT=Path('/data/hfosp_group_event_state_rich_event_identification_v0311')
RUNS=ROOT/'runs';CLAIMS=ROOT/'claims';LOGS=ROOT/'logs'
SUBJECTS=('epilepsiae_1125','epilepsiae_1096','epilepsiae_253')
REFERENCES=('intercept','clock','recent_rate','marked_history','constant_state')


def task_id(t):
    key=json.dumps({k:t[k] for k in sorted(t) if k!='priority'},sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def tag(t):
    s=f"{t['subject']}__{t['split']}__{t['inputs']}__{t['family']}__{t['arm']}__seed{t['seed']}"
    if t.get('target_ablation'):s+='__oldtargets'
    if t.get('crossview_ablation'):s+='__nodelay'
    if t.get('shuffle_marks'):s+='__shuffledmarks'
    return s


def build_queue(seed,updates,source_commit):
    q=[]
    def add(pri,**kw):
        t=dict(seed=seed,max_updates=updates,source_commit=source_commit,
               target_ablation=False,crossview_ablation=False,shuffle_marks=False)
        t.update(kw)
        t['priority']=pri;q.append(t)
    # 1. first patient's complete chain
    for inp in ('P_marks','P_stats'):
        add(1,subject='epilepsiae_1125',split='S-E',inputs=inp,family='I-L-G1',arm='state')
    for ref in REFERENCES:
        add(2,subject='epilepsiae_1125',split='S-E',inputs='P_marks',family='I-L-G1',arm=ref)
    for inp in ('P_marks','P_stats'):
        add(3,subject='epilepsiae_1125',split='S-E',inputs=inp,family='C-N-G1',arm='state')
    # 2. targeted ablations on the first patient
    add(4,subject='epilepsiae_1125',split='S-E',inputs='P_marks',family='I-L-G1',arm='state',
        target_ablation=True)
    add(4,subject='epilepsiae_1125',split='S-E',inputs='P_marks',family='I-L-G1',arm='state',
        crossview_ablation=True)
    # 3. remaining S-E main body and their references
    for sub in SUBJECTS[1:]:
        for inp in ('P_marks','P_stats'):
            add(5,subject=sub,split='S-E',inputs=inp,family='I-L-G1',arm='state')
        for ref in REFERENCES:
            add(6,subject=sub,split='S-E',inputs='P_marks',family='I-L-G1',arm=ref)
        for inp in ('P_marks','P_stats'):
            add(7,subject=sub,split='S-E',inputs=inp,family='C-N-G1',arm='state')
    # 4. S-ID minimal contrast and its references
    for sub in SUBJECTS:
        for inp in ('P_marks','P_stats'):
            add(8,subject=sub,split='S-ID',inputs=inp,family='I-L-G1',arm='state')
    for sub in SUBJECTS:
        for ref in REFERENCES:
            add(9,subject=sub,split='S-ID',inputs='P_marks',family='I-L-G1',arm=ref)
    for t in q:t['id']=task_id(t)
    return sorted(q,key=lambda r:r['priority'])


def claim(t,worker):
    CLAIMS.mkdir(parents=True,exist_ok=True)
    p=CLAIMS/f"{t['id']}.claim.json"
    try:
        fd=os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd,'w') as f:
        json.dump(dict(worker=worker,pid=os.getpid(),started=time.time(),task=t),f)
    return True


def done(t):
    return (RUNS/f'{tag(t)}.card.json').exists()


def run(t,device,worker):
    LOGS.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,str(Path(__file__).resolve().parents[1]/'scripts'/'run_group_event_state_v0311_cell.py'),
         '--subject',t['subject'],'--split',t['split'],'--inputs',t['inputs'],'--family',t['family'],
         '--arm',t['arm'],'--seed',str(t['seed']),'--max-updates',str(t['max_updates']),
         '--extended-updates',str(t['max_updates']),'--device',device,'--out-dir',str(RUNS)]
    if t['target_ablation']:cmd.append('--target-ablation')
    if t['crossview_ablation']:cmd.append('--crossview-ablation')
    if t.get('shuffle_marks'):cmd.append('--shuffle-marks')
    log=LOGS/f"{tag(t)}.log"
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
             NUMEXPR_NUM_THREADS='2')
    with open(log,'w') as f:
        f.write(json.dumps(t)+'\n');f.flush()
        r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
    status='COMPLETE' if r.returncode==0 and done(t) else 'FAILED'
    (CLAIMS/f"{t['id']}.done.json").write_text(json.dumps(dict(status=status,returncode=r.returncode,
                                                              worker=worker,finished=time.time(),task=t)))
    return status


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--device',required=True);ap.add_argument('--worker',required=True)
    ap.add_argument('--seed',type=int,default=20260906);ap.add_argument('--updates',type=int,default=1600)
    ap.add_argument('--deadline-epoch',type=float,default=None)
    ap.add_argument('--print-queue',action='store_true')
    a=ap.parse_args()
    commit=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()
    q=build_queue(a.seed,a.updates,commit)
    RUNS.mkdir(parents=True,exist_ok=True)
    if a.print_queue:
        print(json.dumps([dict(priority=t['priority'],tag=tag(t),id=t['id']) for t in q],indent=1));sys.exit()
    (ROOT/'execution_plan.json').write_text(json.dumps(dict(
        generated=time.time(),source_commit=commit,updates=a.updates,seed=a.seed,
        tasks=[dict(priority=t['priority'],tag=tag(t),id=t['id'],**{k:t[k] for k in
               ('subject','split','inputs','family','arm','target_ablation','crossview_ablation')})
               for t in q]),indent=1))
    while True:
        nxt=None
        for t in q:
            if done(t) or (CLAIMS/f"{t['id']}.claim.json").exists():continue
            if claim(t,a.worker):nxt=t;break
        if nxt is None:
            print(f'[{a.worker}] queue empty',flush=True);break
        if a.deadline_epoch and time.time()>a.deadline_epoch:
            print(f'[{a.worker}] deadline reached before {tag(nxt)}',flush=True)
            (CLAIMS/f"{nxt['id']}.claim.json").unlink(missing_ok=True);break
        print(f'[{a.worker}] start {tag(nxt)}',flush=True)
        s=run(nxt,a.device,a.worker)
        print(f'[{a.worker}] {s} {tag(nxt)}',flush=True)
