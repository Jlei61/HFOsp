#!/usr/bin/env python3
"""Finite dependency graph after corrected-input main training; no score gates."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
from collections import defaultdict


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');os.replace(tmp,path)


def run(root,snapshot,python):
    cache={}
    def sha(path):
        path=Path(path);stat=path.stat();key=(str(path),stat.st_size,stat.st_mtime_ns)
        if key not in cache:cache[key]=hashlib.sha256(path.read_bytes()).hexdigest()
        return cache[key]
    source_files=['scripts/supervise_group_event_state_v038_review_repair.py','scripts/train_group_event_state_v039_human.py',
        'scripts/export_group_event_state_v039_frozen_state.py','scripts/audit_group_event_state_v039_human_gradient.py',
        'scripts/train_group_event_state_v039_contact_transfer.py','scripts/probe_group_event_state_v039_expression.py',
        'src/topic5_group_event_state/v039/transition.py','src/topic5_group_event_state/v039/human_data.py',
        'src/topic5_group_event_state/v039/frozen_transfer.py','src/topic5_group_event_state/v039/ridge_transfer.py']
    source_hashes={p:sha(snapshot/p) for p in source_files}
    def job(key,script,args,output,inputs):
        return dict(id=key,argv=[python,'scripts/'+script,*map(str,args)],output=str(output),input_hashes={str(p):sha(p) for p in inputs})
    def launch(name,jobs,cpu=0,wait=()):
        folder=root/name
        if (folder/'manifest.json').exists():return
        plan=dict(scope='Finite v039 corrected-background dependency: '+name,source_root=str(snapshot),source_hashes=source_hashes,
            jobs=jobs,gpus=[] if cpu else [0,1],cpu_workers=cpu,workers_per_gpu=2,
            wait_for_queues_to_finish=[str(root/q/'queue_status.json') for q in wait])
        atomic(folder/'manifest.json',plan)
        process=subprocess.Popen([python,str(snapshot/'scripts/supervise_group_event_state_v038_review_repair.py'),'--manifest',str(folder/'manifest.json'),'--poll-seconds','5'],
            cwd=snapshot,stdout=(folder/'supervisor.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
        (folder/'supervisor.pid').write_text(str(process.pid)+'\n')
    def finished(name):
        p=root/name/'queue_status.json'
        if not p.exists():return False
        state=json.loads(p.read_text())
        if state['status']=='COMPLETE':return True
        rp=root/'runtime_recovery_registry.json'
        if state['status']!='FAILED' or not rp.exists():return False
        for recovery in json.loads(rp.read_text())['recoveries']:
            retry=root/recovery['retry_queue']/'queue_status.json'
            if recovery['original_queue']==name and retry.exists():
                rs=json.loads(retry.read_text())
                if rs['status']=='COMPLETE' and sorted(k for k,v in state['jobs'].items() if v['status']=='FAILED')==sorted(recovery['jobs']) and set(rs['jobs'])==set(recovery['jobs']):return True
        return False
    def selected(directory,views=False):
        groups=defaultdict(list)
        for path in sorted((root/directory).glob('*/card.json')):
            c=json.loads(path.read_text())
            if c['status']!='COMPLETE':raise ValueError('Cannot freeze incomplete upstream '+str(path))
            cfg=c['config'];key=(c['subject'],c['family'],cfg['history_hours'],cfg['seed'],cfg['view'])
            groups[key].append((path,c))
        expected=18 if views else 54
        if len(groups)!=expected or any(len(g)!=2 for g in groups.values()):raise ValueError('Incomplete LR pairs before freezing')
        chosen=[min(g,key=lambda p:(p[1]['stages']['event']['selected_inner'],p[1]['config']['lr'])) for k,g in sorted(groups.items())]
        atomic(root/('view_state_selection.json' if views else 'main_state_selection.json'),dict(status='FROZEN',selection='minimum own trained-view 2h INNER, tie smaller LR; no downstream or SELECTION selection',
            states=[dict(source=str(p),sha256=sha(p),subject=c['subject'],family=c['family'],config=c['config'],inner=c['stages']['event']['selected_inner']) for p,c in chosen]))
        return chosen
    def export_and_gradient(chosen,tag):
        exports=[];gradients=[]
        for p,c in chosen:
            key=p.parent.name;out=root/('frozen_states' if tag=='main' else 'frozen_view_states')/key/'state.npz'
            inputs=[p,Path(c['checkpoint'])]
            exports.append(job(key,'export_group_event_state_v039_frozen_state.py',['--source',p,'--output',out],out.with_suffix('.json'),inputs))
            dest=root/('human_gradient_audit' if tag=='main' else 'view_gradient_audit')/key/'card.json'
            gradients.append(job(key,'audit_group_event_state_v039_human_gradient.py',['--source',p,'--output',dest],dest,inputs))
        launch('frozen_export_'+tag+'_batch',exports,cpu=3)
        launch('gradient_'+tag+'_batch',gradients,cpu=2)
    def views(chosen):
        scores=defaultdict(list)
        for p,c in chosen:
            if c['config']['history_hours']==8.:scores[(c['subject'],c['family'])].append(c['stages']['event']['selected_inner'])
        subjects=sorted({s for s,f in scores});family={s:min('FLN',key=lambda f:(sum(scores[(s,f)])/3,f)) for s in subjects}
        atomic(root/'single_view_family_selection.json',dict(status='FROZEN',family=family,mean_inner={s:{f:sum(scores[(s,f)])/3 for f in 'FLN'} for s in subjects},
            rule='joint H8 per-seed LR-selected INNER mean; before any fine-transfer outcome; corrected background'))
        jobs=[]
        for s in subjects:
            data=root/'human_data_v2'/f'{s}.pt'
            for v in ['count','recruitment']:
                for seed in [20260905,20260906,20260907]:
                    for lr in [.001,.003]:
                        key=f'{s}_{family[s]}_{v}_lr{lr}_seed{seed}';out=root/'human_views'/key/'card.json'
                        args=['--data',data,'--family',family[s],'--view',v,'--history-hours','8.0','--lr',lr,'--seed',seed,'--output',out]
                        jobs.append(job(key,'train_group_event_state_v039_human.py',args,out,[data,data.with_suffix('.json'),root/'single_view_family_selection.json']))
        launch('human_views_batch',jobs,wait=['human_sensitivity_batch'])
    def transfer(chosen,tag):
        expression=[];contact=[]
        for p,c in chosen:
            key=p.parent.name;s=c['subject'];features=root/('frozen_states' if tag=='main' else 'frozen_view_states')/key/'state.npz'
            inputs=[p,features,features.with_suffix('.json'),root/'transfer_data'/f'{s}.npz',root/'transfer_data'/f'{s}.json',root/'frozen_prefix'/f'{s}.npz',root/'frozen_prefix'/f'{s}.json']
            out=root/('expression_transfer' if tag=='main' else 'view_expression_transfer')/key/'card.json'
            expression.append(job(key,'probe_group_event_state_v039_expression.py',['--features',features,'--root',root,'--output',out],out,inputs))
            dest=root/('contact_transfer' if tag=='main' else 'view_contact_transfer')/key/'card.json'
            contact.append(job(key,'train_group_event_state_v039_contact_transfer.py',['--features',features,'--root',root,'--output',dest,'--seed',c['seed'],'--device','cpu'],dest,inputs))
        launch('expression_'+tag+'_batch',expression,cpu=3)
        # These small frozen adapters complete quickly on CPU. They can run
        # independently alongside GPU observer fits once ancestry is frozen.
        launch('contact_'+tag+'_batch',contact,cpu=3)
    main=None;view=None
    while True:
        if finished('human_main_batch') and main is None:
            main=selected('human_main');views(main);export_and_gradient(main,'main')
        if main and finished('frozen_export_main_batch') and all((root/'frozen_prefix'/f'{s}.json').exists() for s in ['epilepsiae_1096','epilepsiae_1125','epilepsiae_253']):
            transfer(main,'main')
        if finished('human_views_batch') and view is None:
            view=selected('human_views',True);export_and_gradient(view,'view')
        if view and finished('frozen_export_view_batch'):
            transfer(view,'view')
        names=['human_main_batch','human_sensitivity_batch','human_views_batch','frozen_export_main_batch','gradient_main_batch','expression_main_batch','contact_main_batch',
               'frozen_export_view_batch','gradient_view_batch','expression_view_batch','contact_view_batch']
        states={name:json.loads((root/name/'queue_status.json').read_text())['status'] if (root/name/'queue_status.json').exists() else 'NOT_STARTED' for name in names}
        recovery_path=root/'runtime_recovery_registry.json';recovery=None
        if recovery_path.exists():
            recovery=json.loads(recovery_path.read_text())
            for item in recovery['recoveries']:
                retry=item['retry_queue'];original=item['original_queue'];rp=root/retry/'queue_status.json'
                states[retry]=json.loads(rp.read_text())['status'] if rp.exists() else 'NOT_STARTED'
                if states[original]=='FAILED' and states[retry]=='COMPLETE':
                    old=json.loads((root/original/'queue_status.json').read_text())
                    failed_jobs=sorted(k for k,v in old['jobs'].items() if v['status']=='FAILED')
                    retried=json.loads(rp.read_text())
                    if failed_jobs!=sorted(item['jobs']) or set(retried['jobs'])!=set(failed_jobs):
                        raise ValueError('Runtime recovery does not cover exactly the failed jobs')
                    states[original]='COMPLETE_AFTER_RUNTIME_RETRY'
        done=all(v in ('COMPLETE','COMPLETE_AFTER_RUNTIME_RETRY') for v in states.values());failed=[k for k,v in states.items() if v=='FAILED']
        atomic(root/'closure_queue_status.json',dict(status='COMPLETE' if done else 'NEEDS_FAILURE_REVIEW' if failed else 'RUNNING',queues=states,updated_at=time.time(),
            entire_scientific_goal_complete=False,failed_queues=failed,runtime_recovery=recovery,source_sha256=sha(__file__)))
        if done:return
        time.sleep(15)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--python',required=True)
    a=p.parse_args();run(a.root,a.snapshot,a.python)
