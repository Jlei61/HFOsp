#!/usr/bin/env python3
"""Render completed model-training rounds without modifying search or acceptance."""
from pathlib import Path
import fcntl,subprocess,time,json,os,hashlib
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/topic4_sef_hfo/joint_rank_space_dual_core_search_v2'
PYTHON='/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python'
PRODUCER=ROOT/'scripts/paper_figures/plot_topic4_joint_kernel_expanded.py'
SERVICE='codex-t4-joint-kernel-xy-20260906.service'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');os.replace(tmp,path)


def main():
    guard=open(OUT/'figure_watcher.lock','a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    expected=sha(PRODUCER)
    while True:
        for analysis in sorted((OUT/'rounds').glob('*/analysis.json')):
            stage=analysis.parent;marker=stage/'figure_generation.json'
            if marker.exists():
                record=json.loads(marker.read_text())
                if record['analysis_sha256']!=sha(analysis) or record['producer_sha256']!=expected:raise RuntimeError('completed figure inputs changed')
                if not all(Path(p).exists() and sha(Path(p))==h for p,h in record['outputs'].items()):raise RuntimeError('completed figure outputs changed')
                continue
            if sha(PRODUCER)!=expected:raise RuntimeError('producer changed while watcher is active')
            with open(stage/'figure_generation.log','a') as stream:
                subprocess.run([PYTHON,str(PRODUCER),'--round',str(int(stage.name))],cwd=ROOT,check=True,stdout=stream,stderr=subprocess.STDOUT)
            outputs={str(stage/'figures'/f'candidate_after_event_expansion.{ext}'):sha(stage/'figures'/f'candidate_after_event_expansion.{ext}') for ext in ('png','pdf','svg')}
            save(marker,{'status':'RENDERED_PENDING_VISUAL_QA','analysis_sha256':sha(analysis),'producer_sha256':expected,
                'outputs':outputs,'author_acceptance':False,'model_qualification_changed':False,'completed_unix':time.time()})
        state=subprocess.check_output(['systemctl','--user','show',SERVICE,'-p','ActiveState','-p','MainPID'],text=True)
        save(OUT/'figure_watcher_status.json',{'status':'WATCHING_COMPLETED_ROUNDS','model_service':state,'updated_unix':time.time(),'producer_sha256':expected})
        if 'ActiveState=active' not in state and 'ActiveState=activating' not in state:
            save(OUT/'figure_watcher_status.json',{'status':'MODEL_SERVICE_NOT_ACTIVE_RENDERING_FINISHED','model_service':state,'updated_unix':time.time()});return
        time.sleep(10)


if __name__=='__main__':main()
