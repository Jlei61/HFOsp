"""Paired-future SNN test: retained Z-history dose versus persistent recruitment."""
import argparse
import copy
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import recover_fig5_two_gif_bases as rec
from src.topic4_xy_fig5_followup import read,write,sha,available_gib,proc_memory,admission_slots
from src.snn_engine.checkpoint import digest

DATA=ROOT/'results/topic4_sef_hfo/fig5_z_retention_dose_response'
FIG=ROOT/'results/paper-ready-figure/fig5_z_retention_dose_response/figures'
DOSES=[0.,.25,.5,.75,1.]
SEEDS=[7101,7102,7103,7104]


def intervention_state(parent,reference,n_e,dose,seed):
    if not 0<=dose<=1:raise ValueError('dose must lie in [0,1]')
    common=copy.deepcopy(parent)
    common['rng_state']=np.random.default_rng(np.random.SeedSequence([seed,0])).bit_generator.state
    common['external_drive']['rng_state']=np.random.default_rng(np.random.SeedSequence([seed,1])).bit_generator.state
    state=copy.deepcopy(common)
    early=reference['slow']['z'][:n_e];pre=parent['slow']['z'][:n_e]
    state['slow']['z'][:n_e]=early+dose*(pre-early)
    check=copy.deepcopy(state);check['slow']['z']=common['slow']['z'].copy()
    if digest(check)!=digest(common):raise RuntimeError('another state variable changed with dose')
    if not np.all((state['slow']['z']>=0)&(state['slow']['z']<=1)):raise RuntimeError('nonphysical Z')
    return state


def outcome(rate):
    run=0;confirmation=None;onset=None;longest=0
    for i,r in enumerate(rate):
        run=run+1 if r>=250 else 0;longest=max(longest,run)
        if run==25 and confirmation is None:onset=(i-24)*20.;confirmation=(i+1)*20.
    return dict(transition=confirmation is not None,confirmation_ms=confirmation,onset_ms=onset,
                longest_high_ms=20*longest,censor_ms=1500.,terminal_200ms_hz=float(np.mean(rate[-10:])))


def result_path(cid,dose,seed):return DATA/cid/f'lambda_{dose:.2f}_noise_{seed}.json'


def worker(cid,dose,seed):
    path=result_path(cid,dose,seed);path.parent.mkdir(parents=True,exist_ok=True)
    s,cfg,regions,f,job,p=rec.build(cid)
    parent_path=rec.DATA/cid/'checkpoint_pre_onset.npz';reference_path=rec.DATA/cid/'checkpoint_reference.npz'
    parent=rec.old.load_checkpoint(parent_path);reference=rec.old.load_checkpoint(reference_path)
    protocol=read(DATA/'protocol.json')
    for q in [parent_path,reference_path]:
        if sha(q)!=protocol['checkpoint_hashes'][str(q)]:raise RuntimeError('checkpoint changed')
    state=intervention_state(parent,reference,s.n_e,dose,seed)
    result,slow=rec.old.simulate(s,cfg,regions,job,resume=state,duration=1500.)
    arrays=rec.old.trajectory_arrays(s,regions,result,slow)
    write(path,dict(candidate_id=cid,dose=dose,noise_seed=seed,
        protocol_sha256=sha(DATA/'protocol.json'),only_z_changes_across_doses=True,
        mean_Z_at_start=float(state['slow']['z'][:s.n_e].mean()),
        substrate_fingerprint=f,outcome=outcome(arrays['rates_hz'][:,0]),
        arrays=rec.old.save_arrays(path.with_suffix('.npz'),arrays)))


def wilson(success,total):
    p=success/total;z=1.959963984540054;den=1+z*z/total
    center=(p+z*z/(2*total))/den
    half=z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return max(0.,center-half),min(1.,center+half)


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(10.8,7.1),layout='constrained')
    summary=[]
    for row,cid in enumerate(rec.IDS):
        rows=[];probs=[];lo=[];hi=[]
        for dose in DOSES:
            cells=[read(result_path(cid,dose,seed)) for seed in SEEDS]
            for c in cells:
                if c['protocol_sha256']!=sha(DATA/'protocol.json'):raise RuntimeError('mixed contracts')
            n=sum(c['outcome']['transition'] for c in cells);probs.append(n/len(SEEDS));l,h=wilson(n,len(SEEDS));lo.append(l);hi.append(h)
            rows.append(dict(dose=dose,successes=n,n=len(SEEDS),fraction=n/len(SEEDS),wilson95=[l,h],units=cells))
        ax=axes[row,0];ax.errorbar(DOSES,probs,yerr=[np.array(probs)-lo,np.array(hi)-probs],fmt='o-',color='#7c387e',capsize=4,lw=1.5)
        ax.set(ylim=(-.04,1.04),xlim=(-.05,1.05),ylabel='Recruitment probability within 1.5 s',xlabel='Retained Z-history dose λ')
        ax.set_xticks(DOSES,['0','0.25','0.5','0.75','1'])
        name='Base 1 · threshold gain 0.7 / GABA 18 ms' if row==0 else 'Base 2 · threshold gain 1.0 / GABA 24 ms'
        ax.set_title(name,fontsize=10)
        ax=axes[row,1]
        for i,seed in enumerate(SEEDS):
            x=np.asarray(DOSES)+(i-1.5)*.012
            y=[]
            for dose,xx in zip(DOSES,x):
                item=read(result_path(cid,dose,seed))['outcome'];value=item['confirmation_ms']
                y.append(np.nan if value is None else value/1000)
                ax.scatter(xx,1.5 if value is None else value/1000,marker='^' if value is None else 'o',facecolor='none' if value is None else f'C{i}',edgecolor=f'C{i}',s=34)
            ax.plot(x,y,color=f'C{i}',alpha=.5,lw=.8)
        ax.axhline(1.5,color='.5',ls=':',lw=.7)
        ax.set(xlim=(-.05,1.05),ylim=(0,1.67),xlabel='Retained Z-history dose λ',ylabel='Time to confirmed recruitment (s)',title='Same four future-noise realizations')
        ax.text(.02,.98,'△ No qualifying transition by 1.5 s',transform=ax.transAxes,va='top',fontsize=8,color='#555555')
        summary.append(dict(candidate_id=cid,rows=rows))
    fig.suptitle('Does accumulated loss of inhibition promote the runaway transition?\nλ = 0: earlier Z restored; λ = 1: pre-onset Z retained',fontsize=13)
    FIG.mkdir(parents=True,exist_ok=True);stem=FIG/'fig5-E-z-retention-dose-response'
    fig.savefig(stem.with_suffix('.png'),dpi=180);fig.savefig(stem.with_suffix('.pdf'));plt.close(fig)
    write(DATA/'analysis.json',dict(status='COMPLETE_PENDING_VISUAL_REVIEW',results=summary,author_accepted=False))
    (FIG/'README.md').write_text('### fig5-E-z-retention-dose-response.png\n横轴为同一个转变前检查点中保留的 Z 历史变化比例，纵轴为 1.5 s 内达到持续招募标准的频率；右列为相同四组未来噪声的确认时间，未达到者用上三角标出。仅干预起始 Z，M 及全部快状态保留，随后 Z 自然演化。\n**关注点**：每剂量四个新噪声重复，误差棒为 Wilson 95% 区间；这是条件于选定检查点的探索性转变概率，不是患者概率、严格分叉或长期预防证据。\n')


def run():
    DATA.mkdir(parents=True,exist_ok=True)
    protocol=dict(doses=DOSES,future_noise_seeds=SEEDS,duration_ms=1500.,
        dose='Z_lambda = Z_early + lambda*(Z_pre - Z_early), applied per E neuron; I cells unchanged. Earlier state at 1 s, pre-onset state 250 ms before earliest regional recruitment.',
        intervention='Only Z is varied across doses; all other physical state fields retained. Z evolves naturally after intervention, not clamped.',
        randomness='Independent new future innovations in simulator and spatial OU streams; current OU field/cached drive retained. Same innovation seeds paired across every dose. Original future used to select onset is excluded from estimated probabilities.',
        endpoint='Population E rate >=250 Hz continuously for 500 ms, evaluated in 20-ms bins, confirmed within 1.5 s. Confirmation time right-censored at 1.5 s.',
        inference='Four noise replicates per dose, conditional on one graph and checkpoint per base. No patient-level or local bifurcation claim.',
        checkpoint_hashes={str(q):sha(q) for cid in rec.IDS for q in [rec.DATA/cid/'checkpoint_pre_onset.npz',rec.DATA/cid/'checkpoint_reference.npz']})
    if (DATA/'protocol.json').exists():
        if read(DATA/'protocol.json')!=protocol:raise RuntimeError('protocol drift')
    else:write(DATA/'protocol.json',protocol)
    tasks=[(cid,dose,seed) for dose in [0.,1.,.25,.5,.75] for cid in rec.IDS for seed in SEEDS if not result_path(cid,dose,seed).exists()]
    running=[];failures=[]
    while tasks or running:
        rss=[proc_memory(p.pid).get('VmRSS',0.) for p,t,l in running]
        slots=admission_slots(available_gib(),64.,12.,rss,8)
        for _ in range(min(slots,len(tasks))):
            task=tasks.pop(0);cid,dose,seed=task
            log=open(DATA/f'{cid}_{dose}_{seed}.log','a')
            p=subprocess.Popen([sys.executable,__file__,'worker',cid,str(dose),str(seed)],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
            running.append((p,task,log))
        for item in list(running):
            p,task,log=item
            if p.poll() is not None:
                running.remove(item);log.close()
                if p.returncode:failures.append(dict(task=task,returncode=p.returncode));tasks=[]
        write(DATA/'status.json',dict(stage='FAILED_DRAINING' if failures else 'RUNNING',completed=len(list(DATA.glob('*/*.json'))),total=40,pending=len(tasks),running=[dict(pid=p.pid,task=t) for p,t,l in running],failures=failures))
        if tasks or running:time.sleep(15)
    if failures:raise RuntimeError('dose-response worker failed')
    plot();write(DATA/'status.json',dict(stage='COMPLETE_PENDING_VISUAL_REVIEW',completed=40,total=40))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','worker','plot']);p.add_argument('candidate',nargs='?',choices=rec.IDS);p.add_argument('dose',nargs='?',type=float);p.add_argument('seed',nargs='?',type=int);a=p.parse_args()
    resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,12*1024**3))
    if a.mode=='worker':worker(a.candidate,a.dose,a.seed)
    else:globals()[a.mode]()
