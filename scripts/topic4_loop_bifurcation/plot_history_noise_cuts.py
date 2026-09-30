#!/usr/bin/env python3
"""Finished native history/noise cuts, displayed as finite-time measurements."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,pickle,time
import numpy as np
from campaign import ROOT,read,write,sha

OUT=ROOT/'history_noise_review'


def main(wait=False):
    OUT.mkdir(exist_ok=True)
    while True:
        long=read(ROOT/'native_extensions/extended_analysis_summary.json')
        spatial=read(ROOT/'exit_return_probes/extended_analysis_summary.json')
        if long['status']=='COMPLETE' and spatial['status']=='COMPLETE':break
        if not wait:raise RuntimeError('Long windows or initialG probes still incomplete')
        write(OUT/'progress.json',dict(status='WAITING_NATIVE',pid=os.getpid(),long_complete=long['completed'],spatial_complete=spatial['completed'],updated_epoch=time.time()))
        time.sleep(30)
    base=read(ROOT/'native_slices/extended_analysis_summary.json')['rows']
    br={x['name']:x for x in base};lr={x['name']:x for x in long['rows']};sr={x['name']:x for x in spatial['rows']}
    noise=read(ROOT/'independent_noise_probes/extended_analysis_summary.json')['rows']
    nr={x['name']:x for x in noise}
    for p in (ROOT/'independent_noise_probes/incremental_review').glob('*.json'):
        x=read(p);nr[x['name']]=x
    long_rows=[]
    for label,prefix,histories in [('Entry Z=.74','entry_z0.74_k0.0002',['high','interictal']),
                                  ('Exit K=6','exit_z0.21_k6',['high','recovery']),('Exit K=9','exit_z0.21_k9',['high','recovery'])]:
        for history in histories:
            name=prefix+'_'+history
            for horizon,source in [(30,br),(120,lr)]:
                x=source[name];assert x['full_horizon']
                long_rows.append(dict(condition=label,history=history,observation_s=horizon,mean_Hz=x['tail_mean_Hz'],
                    brief_events_last10s=x['tail_brief_events'],joint_quiet_fraction=x['tail_joint_quiet_fraction']))
    grows=[]
    for name,source in [('exit_z0.21_k9_high',br),('exit_z0.21_k9_recovery',br),
                        ('exit_z0.21_k9_G0_high',sr),('exit_z0.21_k9_G11.495_high',sr),('exit_z0.21_k9_G11.495_recovery',sr)]:
        x=source[name];j=x['job']
        if 'G_raw_override' in j:g=float(j['G_raw_override'])
        else:
            with open(j['source_checkpoint'],'rb') as f:g=30*pickle.load(f)['engine']['global_feedback_response']['global_state']
        grows.append(dict(name=name,history=j['source_history'],initial_Graw=g,mean_Hz=x['tail_mean_Hz'],
            joint_quiet_fraction=x['tail_joint_quiet_fraction'],brief_events_last10s=x['tail_brief_events'],
            drift_Z_K_per_s_allE_A_B_other=x['counterfactual_drift_mean_allE_A_B_other'],
            drift_variable_order=['dZ_per_s','dK_per_s']))
    returns=[]
    for k in [.01,.02,.03]:
        name=f'return_z0.995_k{k:g}_recovery';x=(sr if k==.02 else br)[name]
        rows=[(9108405,x)]
        for seed in [9108402,9108403]:
            name2=name+f'_noise{seed}';assert name2 in nr,name2;rows.append((seed,nr[name2]))
        for seed,x in rows:
            returns.append(dict(K=k,future_noise_source=seed,brief_events_last10s=x['tail_brief_events'],
                mean_Hz=x['tail_mean_Hz'],joint_quiet_fraction=x['tail_joint_quiet_fraction'],
                core_recruitment=x['tail_core_recruitment'],event_metrics=x['tail_event_metrics']))
    report=dict(status='COMPLETE',long_windows=long_rows,initial_G_probes=grows,return_future_noise=returns,
        scope='Allnativeconditionalclamps withfixedt20spatialfieldfamily,notautonomousloops. G/Mandfaststatesdynamic. HeldZ/K,declaredhistories,originalgeometryunchanged.',
        units='Tailmeans andeventcounts fromlast10s. InitialG overridescausalwithinpairedhistory. Futureinputseeds arenot fullnativeindependentseeds;3futureinputs atsameendogenousstate.',
        formal_bifurcation='NOT_ESTABLISHED',human_review='PENDING',source_sha256=sha(__file__))
    write(OUT/'analysis.json',report);plot(report);write(OUT/'progress.json',dict(status='COMPLETE',updated_epoch=time.time()))
    print('NATIVE G PROBES',[(x['history'],x['initial_Graw'],x['mean_Hz']) for x in grows],flush=True)
    print('NATIVE RETURN',[(x['K'],x['future_noise_source'],x['brief_events_last10s']) for x in returns],flush=True)


def plot(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(13.4,4.5),layout='constrained');colors={'high':'#b7672b','interictal':'#287c8e','recovery':'#287c8e'}
    labels=['Entry Z=.74','Exit K=6','Exit K=9']
    for row in report['long_windows']:
        x=labels.index(row['condition'])+(-.16 if row['history']=='high' else .16)
        x+=(-.04 if row['observation_s']==30 else .04)
        axes[0].plot(x,row['mean_Hz'][0],marker='o' if row['observation_s']==30 else 's',color=colors[row['history']],ms=6,
                     markerfacecolor='white' if row['observation_s']==30 else colors[row['history']],linestyle='none')
    axes[0].set_xticks(range(3),['Entry Z=.74\nhigh / interictal','Exit K=6\nhigh / recovery','Exit K=9\nhigh / recovery'],fontsize=9)
    axes[0].set_ylabel('Tail mean E rate (Hz)');axes[0].set_ylim(-10,500);axes[0].set_title('A  Extend observation to 120 s',loc='left',weight='bold')
    axes[0].legend(handles=[Line2D([],[],marker='o',color='.3',mfc='white',ls='none',label='30 s'),Line2D([],[],marker='s',color='.3',ls='none',label='120 s')],frameon=False,ncol=2,fontsize=9)
    for history,marker in [('high','o'),('recovery','x')]:
        rows=sorted([x for x in report['initial_G_probes'] if x['history']==history],key=lambda x:x['initial_Graw'])
        axes[1].plot([x['initial_Graw'] for x in rows],[x['mean_Hz'][0] for x in rows],marker=marker,ls='none',ms=7,color=colors[history],label=history+' history')
    axes[1].set_xlabel('Initial raw G / gL');axes[1].set_ylabel('Tail mean E rate (Hz)');axes[1].set_title('B  Change only initial G',loc='left',weight='bold')
    axes[1].text(.04,.97,'Z=.21, K=9; 30-s window',va='top',transform=axes[1].transAxes,fontsize=9)
    axes[1].set_ylim(-5,100);axes[1].legend(frameon=False,fontsize=9,loc='center right')
    palette=['#595959','#277c9e','#b65c83']
    for seed,color,marker in zip([9108405,9108402,9108403],palette,['o','s','^']):
        rows=[x for x in report['return_future_noise'] if x['future_noise_source']==seed]
        axes[2].plot([x['K'] for x in rows],[x['brief_events_last10s'] for x in rows],marker=marker,color=color,lw=1,label=f'Future input {str(seed)[-4:]}')
    axes[2].set_xticks([.01,.02,.03]);axes[2].set_xlabel('Held mean K / gL');axes[2].set_ylabel('Brief events / last 10 s')
    axes[2].set_title('C  Sparse return near the boundary',loc='left',weight='bold');axes[2].legend(frameon=False,fontsize=8)
    fig.suptitle('Native conditional responses: history, observation time and future noise',weight='bold')
    fig.text(.5,-.025,'Fixed t20 spatial field family. Points are finite-time measurements; connecting lines guide the eye, not certified branches.',ha='center',fontsize=9)
    for ext in ['png','svg']:fig.savefig(ROOT/'figures'/f'native_history_noise_cuts.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig);write(OUT/'figure_metadata.json',dict(agent_visual='PENDING',human_review='PENDING',producer_sha256=sha(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');main(p.parse_args().wait)
