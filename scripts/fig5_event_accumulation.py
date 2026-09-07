"""Event-resolved Z accounting and matched Z-history restoration continuations."""
import argparse
import copy
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import recover_fig5_two_gif_bases as rec
from src.topic4_xy_fig5_followup import read,write,sha
from src.snn_engine.checkpoint import digest

DATA=ROOT/'results/topic4_sef_hfo/fig5_event_accumulation'
FIG=ROOT/'results/paper-ready-figure/fig5_event_accumulation/figures'


def source(cid):
    root=rec.DATA/cid
    path=root/'trajectory.json'
    if not path.exists():path=rec.first.OUTPUT/cid/'trajectory.json'
    j=read(path)
    with np.load(j['arrays']['path']) as a:arrays={k:a[k] for k in a.files}
    return j,arrays,read(root/'protocol.json')


def accounting(cid):
    j,a,p=source(cid);onset=p['states_ms']['pre_onset']+250
    original=read(rec.first.OUTPUT/cid/'trajectory.json')
    # Include all detector intervals for accounting; do not discard failures to return.
    events=[e for e in original['population_excursion_diagnostics'] if e['t_on']>=200 and e['t_off']<onset-250]
    rows=[];t=a['slow_time_ms'];z=a['slow_z_core_mean']
    for i,event in enumerate(events[:-1]):
        nxt=events[i+1]['t_on'];before=event['t_on'];after=event['t_off']
        zb,za,zn=np.interp([before,after,nxt],t,z)
        drop=zb-za;recovery=zn-za;residual=zb-zn
        if not np.isclose(drop-recovery,residual,atol=1e-12):raise RuntimeError('Z accounting does not balance')
        rows.append(dict(index=i+1,t_before_ms=before,t_after_ms=after,t_next_ms=nxt,
            returned=event['returned'],Z_before=float(zb),Z_after=float(za),Z_next=float(zn),
            within_event_loss=float(drop),between_event_recovery=float(recovery),residual_loss=float(residual)))
    out=DATA/cid;out.mkdir(parents=True,exist_ok=True)
    report=dict(candidate_id=cid,onset_ms=onset,rows=rows,
        total_within_event_loss=float(sum(r['within_event_loss'] for r in rows)),
        total_between_event_recovery=float(sum(r['between_event_recovery'] for r in rows)),
        net_loss=float(sum(r['residual_loss'] for r in rows)),
        definition='Z at detector onset/off and next onset, interpolated from native slow records. Negative between-event recovery means continuing depletion in the gap; it is not clipped.',
        scope='All prespecified model population detector intervals, including non-returned fragments; not patient IED units or template labels.',
        source_arrays=j['arrays'])
    write(out/'accounting.json',report);return report,a,p


def intervention(cid):
    out=DATA/cid;out.mkdir(parents=True,exist_ok=True)
    s,cfg,regions,f,job,p=rec.build(cid)
    root=rec.DATA/cid
    ckpath=root/'checkpoint_pre_onset.npz';refpath=root/'checkpoint_reference.npz'
    state=rec.old.load_checkpoint(ckpath);reference=rec.old.load_checkpoint(refpath)
    restored=copy.deepcopy(state);restored['slow']['z'][:s.n_e]=reference['slow']['z'][:s.n_e]
    check=copy.deepcopy(restored);check['slow']['z']=state['slow']['z'].copy()
    if digest(check)!=digest(state):raise RuntimeError('Z restoration changed another checkpoint field')
    j,a,_=source(cid)
    protocol=dict(candidate_id=cid,start_ms=float(state['absolute_time_ms']),duration_ms=1500.,
        checkpoint_sha256=sha(ckpath),reference_checkpoint_sha256=sha(refpath),
        intervention='Restore only per-E-neuron Z to the 1-s checkpoint; M, voltage, refractory state, synaptic currents, delay rings and all random states retained. Z evolves naturally thereafter.',
        only_z_modified_verified=True,substrate_fingerprint=f,
        endpoint='At least 500 ms continuously above 250 Hz population E rate in 20-ms bins within 1.5 s. This is persistent recruitment, not stationary plateau or long-term prevention.')
    write(out/'intervention_protocol.json',protocol)
    outcomes={}
    for name,initial in [('natural',state),('restore_z',restored)]:
        target=out/(name+'.json')
        if target.exists():outcomes[name]=read(target);continue
        write(out/'intervention_status.json',dict(stage='RUNNING',branch=name,pid=__import__('os').getpid()))
        result,slow=rec.old.simulate(s,cfg,regions,job,resume=initial,duration=1500.)
        arrays=rec.old.trajectory_arrays(s,regions,result,slow)
        if name=='natural':
            begin=int(round(initial['absolute_time_ms']))
            if not np.array_equal(arrays['lfp'],a['lfp'][begin:begin+1500]):raise RuntimeError('natural sham changed compared with parent')
        high=arrays['rates_hz'][:,0]>=250;longest=run=0
        for flag in high:run=run+1 if flag else 0;longest=max(longest,run)
        outcome=dict(branch=name,persistent_recruitment=longest>=25,longest_high_ms=longest*20,
            terminal_200ms_population_hz=float(arrays['rates_hz'][-10:,0].mean()),
            arrays=rec.old.save_arrays(out/(name+'.npz'),arrays),natural_parent_exact=name=='natural')
        write(target,outcome);outcomes[name]=outcome
        del result,slow,arrays
    write(out/'intervention_result.json',dict(protocol=protocol,outcomes=outcomes,
        interpretation='Z restoration changes only one checkpoint state field, under paired future randomness. Effect supports a role of the Z state; it does not prove that event count alone caused depletion, or identify a bifurcation.'))
    write(out/'intervention_status.json',dict(stage='COMPLETE'))


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    FIG.mkdir(parents=True,exist_ok=True)
    show_intervention=any((DATA/c/'intervention_result.json').exists() for c in rec.IDS)
    fig=plt.figure(figsize=(14 if show_intervention else 10.5,8.3),layout='constrained')
    grid=fig.add_gridspec(2,3 if show_intervention else 2,width_ratios=[1.35,1,1.12] if show_intervention else [1.35,1.2])
    summaries=[]
    for row,cid in enumerate(rec.IDS):
        report,a,p=accounting(cid);summaries.append(report)
        on=report['onset_ms'];t=a['slow_time_ms']/1000
        sub=grid[row,0].subgridspec(2,1,height_ratios=[.5,1],hspace=.04)
        rateax=fig.add_subplot(sub[0]);zax=fig.add_subplot(sub[1],sharex=rateax)
        rt=a['time_ms']/1000;rateax.plot(rt,a['rates_hz'][:,0],color='#666666',lw=.75)
        rateax.set(ylabel='E rate (Hz)',ylim=(0,400));rateax.tick_params(labelbottom=False)
        rateax.text(.97,.94,'Recruitment onset →',ha='right',va='top',transform=rateax.transAxes,color='#bc315b',fontsize=8)
        zax.plot(t,a['slow_z_core_mean'],color='#326ca5',lw=1.5)
        zax.plot(t,a['slow_z_surround_mean'],color='#a4bad0',lw=1.,ls='--',label='Surround')
        events=report['rows']
        for e in events:
            for ax in [rateax,zax]:ax.axvspan(e['t_before_ms']/1000,e['t_after_ms']/1000,color='#edbc75',alpha=.17,lw=0)
        before=np.array([e['t_before_ms']/1000 for e in events]);zb=np.array([e['Z_before'] for e in events])
        zax.scatter(before,zb,s=11,color='#326ca5',zorder=4,label='Core Z before each event')
        for ax in [rateax,zax]:
            ax.axvline(on/1000,color='#bc315b',ls='--',lw=1);ax.set_xlim(.2,(on+400)/1000)
            ax.axvspan(on/1000,(on+400)/1000,color='#bc315b',alpha=.07,lw=0)
        zax.set(ylim=(.72,1.005),ylabel='Inhibitory efficacy Z',xlabel='Time (s)');zax.legend(fontsize=7.5,frameon=False,loc='lower left')
        label='Base 1 · threshold gain 0.7 / GABA 18 ms' if row==0 else 'Base 2 · threshold gain 1.0 / GABA 24 ms'
        rateax.set_title(label+'\nRepeated events and retained Z loss',fontsize=10,pad=9)
        ax=fig.add_subplot(grid[row,1]);n=np.arange(1,len(events)+1)
        loss=np.array([e['within_event_loss'] for e in events]);recovery=np.array([e['between_event_recovery'] for e in events])
        ax.plot(n,100*np.cumsum(loss),color='#c77921',lw=1.4,label='Loss during detected events')
        ax.plot(n,-100*np.cumsum(recovery),color='#777777',lw=1.2,label='Gap contribution (+ loss / − recovery)')
        ax.plot(n,100*np.cumsum(loss-recovery),color='#326ca5',lw=1.7,label='Total retained loss')
        ax.axhline(0,color='.65',lw=.7);ax.set(xlabel='Successive model event intervals',ylabel='Accumulated Z change (percentage points)',title='Where does the accumulation occur?')
        ax.legend(fontsize=7.5,frameon=False,loc='upper left');ax.set_xlim(0,max(n))
        if not show_intervention:continue
        ax=fig.add_subplot(grid[row,2]);res=DATA/cid/'intervention_result.json'
        if res.exists():
            result=read(res)
            for name,color,label in [('natural','#bc315b','Natural continuation'),('restore_z','#326ca5','Restore earlier Z only')]:
                with np.load(result['outcomes'][name]['arrays']['path']) as q:
                    ax.plot((q['time_ms']-result['protocol']['start_ms'])/1000,q['rates_hz'][:,0],color=color,lw=1.2,label=label)
            ax.axhline(250,color='.6',ls=':',lw=.7);ax.set(xlim=(0,1.5),ylim=(0,510),xlabel='Time after intervention (s)',ylabel='Population E rate (Hz)',title='Does restoring Z interrupt recruitment?');ax.legend(fontsize=8,frameon=False)
        else:
            ax.set_facecolor('#f8f8f8');ax.set_xticks([]);ax.set_yticks([])
            ax.set_title('Causal test: restore earlier Z only')
            ax.text(.5,.58,'Paired continuation running',ha='center',va='center',transform=ax.transAxes,fontsize=11,color='#555555')
            ax.text(.5,.37,'Same pre-onset checkpoint\nSame M and future random input\nOnly Z restored to its 1-s spatial state',ha='center',va='center',transform=ax.transAxes,fontsize=9,color='#555555')
    fig.suptitle('E candidate: event-associated accumulation of inhibitory-efficacy loss',fontsize=15)
    stem=FIG/'fig5-E-event-accumulation'
    fig.savefig(stem.with_suffix('.png'),dpi=180);fig.savefig(stem.with_suffix('.pdf'));plt.close(fig)
    write(stem.with_suffix('.json'),dict(status='CANDIDATE',accounting=summaries,causal_test_complete=[(DATA/c/'intervention_result.json').exists() for c in rec.IDS],author_accepted=False))
    (FIG/'README.md').write_text('### fig5-E-event-accumulation.png\n左列为实际群体活动和 Z，保留事件前的 Z 水平；中列把净下降拆为检测事件期间与事件间隔期间的有符号变化。右列为同检查点、同随机过程仅恢复 Z 的延续；未完成时明确标注运行中。\n**关注点**：事件区间来自既有模型群体检测器，不能据此声称患者 IED 次数决定 Z；间隔内若继续下降，必须保留，不能改称恢复。\n')
    print(json.dumps([{k:v for k,v in x.items() if k not in ['rows','source_arrays']} for x in summaries],indent=2))


def run():
    DATA.mkdir(parents=True,exist_ok=True)
    procs=[]
    for cid in rec.IDS:
        if not (DATA/cid/'intervention_result.json').exists():
            log=open(DATA/(cid+'.log'),'a');procs.append((subprocess.Popen([sys.executable,__file__,'intervention',cid],stdout=log,stderr=subprocess.STDOUT),log))
    failed=False
    while procs:
        for process,log in list(procs):
            if process.poll() is not None:
                log.close();procs.remove((process,log));failed |= process.returncode!=0;plot()
        if procs:time.sleep(20)
    write(DATA/'completion.json',dict(status='FAILED' if failed else 'COMPLETE_PENDING_VISUAL_REVIEW'))
    if failed:raise RuntimeError('paired Z restoration worker failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['plot','run','intervention']);p.add_argument('candidate',nargs='?',choices=rec.IDS);a=p.parse_args()
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    if a.mode=='intervention':intervention(a.candidate)
    else:globals()[a.mode]()
