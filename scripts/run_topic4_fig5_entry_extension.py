#!/usr/bin/env python3
"""Continue the 22 right-censored Fig5 trajectories without resetting state."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import copy
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import psutil
import run_topic4_fig5_log_m_scan as engine
import analyze_topic4_fig5_log_m_scan as analysis

base=engine.base
ROOT=engine.ROOT
PARENT=engine.OUT
OUT=ROOT/'results/topic4_sef_hfo/fig5_log_m_entry_extension_20260915'
STORAGE=Path('/data/hfosp/topic4_sef_hfo/fig5_log_m_entry_extension_20260915')


def prepare():
    if not OUT.exists():
        STORAGE.mkdir(parents=True,exist_ok=True)
        OUT.symlink_to(STORAGE,target_is_directory=True)
    if (OUT/'protocol.json').exists():
        return base.read(OUT/'protocol.json')
    parent=engine.prepare();grid=analysis.collect()
    assert grid['all_complete'] and grid['completed_new']==42
    jobs=[];references=[];sources={}
    for record in grid['records']:
        source=Path(record['source']);job=copy.deepcopy(record['job'])
        if record['event_observed']:
            references.append(dict(job=job,source=str(source),first_entry=record['first_entry'],
                result_sha256=base.sha(source/'result.json')))
            continue
        result=base.read(source/'result.json')
        assert result['elapsed_s']==300 and (source/'checkpoint.pkl').exists()
        job['horizon_s']=1000.
        jobs.append(job)
        sources[job['name']]=dict(source=str(source),original_job=result['job'],
            result_sha256=base.sha(source/'result.json'),checkpoint_sha256=base.sha(source/'checkpoint.pkl'))
    assert len(jobs)==22 and len(references)==26
    protocol=dict(status='DEFINED_BEFORE_CONTINUATIONS',jobs=jobs,references=references,
        sources=sources,new_jobs=22,reused=26,total_realizations=48,total_cells=24,
        eta_M=parent['eta_M'],tau_M_s=parent['tau_M_s'],seeds=parent['seeds'],horizon_s=1000.,
        identity=parent['identity'],source_hashes=parent['source_hashes'],
        producer_sha256=parent['producer_sha256'],extension_producer_sha256=base.sha(__file__),
        parent_protocol=str(PARENT/'protocol.json'),parent_protocol_sha256=base.sha(PARENT/'protocol.json'),
        endpoint=parent['endpoint'],max_workers=12,
        approval='2026-09-15 user requested further simulations to examine entry times; bounded extension of the 22 previously censored trajectories.',
        question='Do any trajectories with no entry by 300 seconds enter by 1000 seconds?',
        intervention='Extend only the stopping horizon from 300 to 1000 seconds. Preserve fast, slow, synaptic, OU and RNG checkpoint state; no reset or parameter change.',
        statistics='Same 24 parameter cells, one fixed topology and two paired noise realizations per cell. Continuations are not new independent samples.',
        interpretation='Restricted mean first confirmation time at 1000 seconds with entry counts; no-entry by 1000 seconds is right-censored, not permanent stability.',
        stop='22 continuations to first confirmation or 1000 seconds, plus four short checkpoint QA runs. No automatic grid expansion.',
        qa='Both seeds: compare a fresh 1.02-second replay with a 1.0-to-1.02-second continuation from the original QA checkpoint, including complete executor state.')
    for path,digest in protocol['source_hashes'].items():assert base.sha(path)==digest,path
    base.write(OUT/'protocol.json',protocol)
    for job in jobs:base.write(OUT/'jobs'/(job['name']+'.json'),job)
    for seed in parent['seeds']:
        old=base.read(PARENT/'jobs'/f'qa_s{seed}.json')
        for kind in ('fresh','resume'):
            job=dict(old,name=f'qa_{kind}_s{seed}',horizon_s=1.02,qa=False)
            base.write(OUT/'jobs'/(job['name']+'.json'),job)
    (OUT/'execution_plan.md').write_text(
        '# Fig5 首次进入时间延长\n\n'
        '问题：原300秒内未进入高态的轨迹，在1000秒内是否会进入？读出仍为全E在10ms分箱下≥200Hz连续200ms，保留起点和确认时间。\n\n'
        '只将22条删失轨迹从原300秒检查点续跑至首次确认或1000秒；26个已有精确进入终点复用。共24个参数格，每格仍是同一拓扑下两个配对噪声，续跑不增加独立样本数。\n\n'
        'E/I网络、Z/M、外部OU、随机数及突触延迟状态原样接续，只改停止时限；两种噪声先做完整状态的短程续跑一致性检查。最多12个worker，数据放/data，原结果保留。\n\n'
        '每个延长终点从连续10ms计数重算核对；图保持正方形，参数和色条均为log轴。旧300秒结果保留，新1000秒图中灰格表示尚未完成延长，斜线仅用于已完成但有删失的格。\n\n'
        '有晚进入则报告实际确认时间和进入数；1000秒仍未进入则报告右删失。τM=1000秒只覆盖一个时间常数，不能据此认定永久稳定或严格分岔。完成22条后停止扩展，交人工审阅。\n')
    return protocol


def configure(protocol):
    engine.OUT=OUT
    engine.prepare=lambda:protocol
    analysis.OUT=OUT


def seed_checkpoint(name,source,job,expected_job,expected_sha=None):
    folder=OUT/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
    cp=folder/'checkpoint.pkl'
    if cp.exists():return
    if expected_sha is not None:assert base.sha(source/'checkpoint.pkl')==expected_sha
    saved=base.load_pickle(source/'checkpoint.pkl')
    assert saved['job']==expected_job
    assert saved['identity']==base.read(OUT/'protocol.json')['identity']
    start=int(saved['engine']['step'])
    assert start==round(expected_job['horizon_s']*10000)
    assert saved['tracker']['first_entry'] is None
    changed=[key for key in set(job)|set(expected_job) if job.get(key)!=expected_job.get(key)]
    allowed={'horizon_s'} if not name.startswith('qa_') else {'horizon_s','name','qa'}
    assert set(changed)<=allowed
    chunks=folder/'chunks';chunks.mkdir(exist_ok=True)
    for path in sorted((source/'chunks').glob('*.npz')):
        if '.tmp.' not in path.name:shutil.copy2(path,chunks/path.name)
    # Only bookkeeping changes: the executor and tracker objects stay intact.
    saved['job']=job
    base.save_pickle(cp,saved)
    base.write(folder/'continuation.json',dict(source=str(source),source_checkpoint_sha256=base.sha(source/'checkpoint.pkl'),
        start_step=start,changed_job_fields=changed,all_executor_and_tracker_state_preserved=True))


def worker(name):
    protocol=prepare()
    assert base.sha(__file__)==protocol['extension_producer_sha256']
    job=base.read(OUT/'jobs'/(name+'.json'))
    if name.startswith('qa_resume_'):
        source=PARENT/'runs'/f'qa_s{job["seed"]}'
        seed_checkpoint(name,source,job,base.read(PARENT/'jobs'/f'qa_s{job["seed"]}.json'))
    elif not name.startswith('qa_'):
        ref=protocol['sources'][name];source=Path(ref['source'])
        assert base.sha(source/'result.json')==ref['result_sha256']
        seed_checkpoint(name,source,job,ref['original_job'],ref['checkpoint_sha256'])
    configure(protocol)
    engine.worker(name)


def same(left,right):
    if isinstance(left,np.ndarray):assert np.array_equal(left,right,equal_nan=True)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:same(left[key],right[key])
    elif isinstance(left,(list,tuple)):
        assert len(left)==len(right)
        for a,b in zip(left,right):same(a,b)
    else:assert left==right,(left,right)


def launch(name):
    folder=OUT/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();lib=str(Path(sys.executable).parent.parent/'lib')
    env['LD_LIBRARY_PATH']=lib+os.pathsep+env.get('LD_LIBRARY_PATH','')
    with (folder/'worker.log').open('a') as log:
        return subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--name',name],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)


def qa(protocol):
    if (OUT/'qa.json').exists():
        assert base.read(OUT/'qa.json')['status']=='PASS'
        return
    children={name:launch(name) for name in [f'qa_{kind}_s{seed}' for seed in protocol['seeds'] for kind in ('fresh','resume')]}
    while children:
        for name,child in list(children.items()):
            code=child.poll()
            if code is not None:
                if code:raise RuntimeError(f'Continuation QA failed: {name}')
                del children[name]
        base.write(OUT/'status.json',dict(status='CHECKING_CONTINUATION',pid=os.getpid(),running={n:c.pid for n,c in children.items()},
            completed_extensions=0,total_extensions=22,pending=22,reused=26,failed=[],updated_at=time.time()))
        if children:time.sleep(5)
    checks=[]
    for seed in protocol['seeds']:
        a=base.load_pickle(OUT/'runs'/f'qa_fresh_s{seed}'/'checkpoint.pkl')
        b=base.load_pickle(OUT/'runs'/f'qa_resume_s{seed}'/'checkpoint.pkl')
        same(a['engine'],b['engine'])
        for key in ('first_entry','high_bins'):same(a['tracker'][key],b['tracker'][key])
        checks.append(dict(seed=seed,entire_executor_state_exact=True,tracker_exact=True,end_step=a['engine']['step']))
    base.write(OUT/'qa.json',dict(status='PASS',checks=checks))


def audit_endpoint(record):
    end=0;rows=[]
    for path in sorted((Path(record['source'])/'chunks').glob('*.npz')):
        if '.tmp.' in path.name:continue
        with np.load(path) as data:
            assert int(data['start_step'])==end;end=int(data['end_step'])
            assert np.array_equal(data['spikes_10ms'][:,0],data['regions_10ms'][:,:3].sum(1))
            rows.append(data['spikes_10ms'][:,0])
    rate=np.concatenate(rows)/320
    bits=np.diff(np.r_[False,rate>=200,False].astype(int))
    spans=[(a,b) for a,b in zip(np.flatnonzero(bits==1),np.flatnonzero(bits==-1)) if b-a>=20]
    if record['event_observed']:
        assert spans
        assert np.isclose(spans[0][0]*.01,record['first_entry']['onset_s'])
        assert np.isclose((spans[0][0]+20)*.01,record['first_entry']['confirmation_s'])
    else:assert not spans and end>=10000000
    return dict(name=record['job']['name'],status='PASS',continuous_from_zero=True,observed=record['event_observed'])


def report(protocol):
    configure(protocol);grid=analysis.collect()
    checks=[audit_endpoint(r) for r in grid['records'] if not r['reused']]
    base.write(OUT/'endpoint_audit.json',dict(checked_extensions=len(checks),checks=checks,all_complete=grid['all_complete']))
    base.write(OUT/'measured_grid.json',analysis.safe(grid))
    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    fig=analysis.plt.figure(figsize=(11,9));ax=analysis.draw(fig,fig.add_gridspec(1,1)[0],grid)
    fig.canvas.draw()
    assert np.isclose(ax.bbox.width,ax.bbox.height) and ax.child_axes[0].get_yscale()=='log'
    for ext in ('png','pdf'):fig.savefig(dest/f'log_m_first_entry.{ext}',dpi=160,bbox_inches='tight')
    analysis.plt.close(fig)
    (dest/'README.md').write_text('### log_m_first_entry.png / .pdf\n延长到1000秒的首次进入图，每格仍是原来的两条噪声轨迹；参数与色条均为log轴，绘图区为正方形。灰格表示延长尚未完成，已完成格显示限制均值和进入数，斜线标出右删失。\n**关注点**：原300秒未进入结果保留，未完成延长不能归为1000秒未进入；该图不证明永久稳定。\n')
    import plot_topic4_fig5_clean_panels as figure
    figure.OUT=OUT/'full_fig5';figure.OUT.mkdir(exist_ok=True)
    versions=[figure.render(row,grid) for row in figure.audit.sources() if row['eta_m']==.0005 and (row['source']/'result.json').exists()]
    base.write(figure.OUT/'delivery_manifest.json',dict(versions=versions,human_review='PENDING'))
    (OUT/'README.md').write_text('# Fig5 首次进入时间延长\n\n'
        f'已完成延长 {grid["completed_new"]}/22 条，复用26个此前已观测到的精确终点。原24格、每格两条噪声不变，观察窗从300秒扩至1000秒。\n\n'
        '[参数图](figures/log_m_first_entry.png) · [执行方案](execution_plan.md) · [实时状态](status.json)\n\n'+
        '\n'.join(f'- {v["name"]}：[完整Fig5 PNG]({v["figure"]}) · [PDF]({v["pdf"]})' for v in versions)+
        '\n\n灰格为延长尚未完成，斜线为已完成但有右删失。完成22条后停止扩展，候选待人工检查。\n')
    return grid['completed_new']


def supervise():
    protocol=prepare();lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    qa(protocol)
    pending=[j['name'] for j in protocol['jobs'] if not (OUT/'runs'/j['name']/'result.json').exists()]
    children={};failed=[];last=-1
    while pending or children:
        for name,child in list(children.items()):
            code=child.poll()
            if code is None:continue
            del children[name]
            if code or not (OUT/'runs'/name/'result.json').exists():failed.append(name)
        available=psutil.virtual_memory().available/2**30
        while pending and not failed and len(children)<protocol['max_workers'] and available>84 and shutil.disk_usage(OUT).free/2**30>40:
            name=pending.pop(0);children[name]=launch(name);available-=4
        completed=sum((OUT/'runs'/j['name']/'result.json').exists() for j in protocol['jobs'])
        progress={}
        for name,child in children.items():
            path=OUT/'runs'/name/'progress.json'
            d=base.read(path) if path.exists() else {}
            progress[name]=dict(pid=child.pid,time_s=d.get('time_s',300.),status=d.get('status','STARTING'))
        base.write(OUT/'status.json',dict(status='DRAINING_FAILURE' if failed else 'RUNNING',pid=os.getpid(),updated_at=time.time(),
            completed_extensions=completed,total_extensions=22,reused=26,running=progress,pending=len(pending),failed=failed))
        if completed!=last:report(protocol);last=completed
        if failed and not children:raise RuntimeError(failed)
        if pending or children:time.sleep(10)
    report(protocol)
    base.write(OUT/'status.json',dict(status='COMPLETE_PENDING_HUMAN_REVIEW',completed_extensions=22,total_extensions=22,reused=26,running={},pending=0,failed=[]))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','worker','report','supervise']);parser.add_argument('--name')
    args=parser.parse_args()
    try:
        if args.mode=='prepare':prepare()
        elif args.mode=='worker':worker(args.name)
        elif args.mode=='report':report(prepare())
        else:supervise()
    except Exception as exc:
        target=OUT/'runs'/args.name/'failure.json' if args.mode=='worker' else OUT/'supervisor_failure.json'
        base.write(target,dict(error=repr(exc),time=time.time()))
        raise
