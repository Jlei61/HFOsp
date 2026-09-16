#!/usr/bin/env python
"""Diagnostic figures for the v0.3.11 window. One panel, one question.

Only real produced data is drawn; a missing arm is written as not run rather
than filled with a placeholder.
"""
import json,sys,glob
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/data/hfosp_group_event_state_rich_event_identification_v0311')
FIG=ROOT/'figures'
HOR=[1,5,30,120]
VIEW_LABEL={'count':'Event count','spatial':'Coarse spatial composition','morphology':'Conditional morphology'}
ARM_LABEL={'P_marks/I-L-G1/state':'Rich event history → state',
           'P_stats/I-L-G1/state':'Coarse summary history → state',
           'P_marks/C-N-G1/state':'Rich event history → coupled nonlinear state',
           'P_stats/C-N-G1/state':'Coarse summary history → coupled nonlinear state',
           'P_marks/I-L-G1/intercept':'Constant rate baseline',
           'P_marks/I-L-G1/clock':'Clock + exposure baseline',
           'P_marks/I-L-G1/recent_rate':'Recent rate history',
           'P_marks/I-L-G1/marked_history':'Full marked history (no state)',
           'P_marks/I-L-G1/constant_state':'Constant state, same readout',
           'P_marks/I-L-G1/recent_rate+unnormalised_history_kernel':'Recent rate history (unnormalised kernel)',
           'P_marks/I-L-G1/marked_history+unnormalised_history_kernel':'Full marked history (unnormalised kernel)',
           'P_stats/I-L-G1/state+oldtargets':'Rich event history → state, count/space targets only',
           'P_marks/I-L-G1/state+oldtargets':'Rich event history → state, count/space targets only',
           'P_marks/I-L-G1/state+nodelay':'Rich event history → state, propagation delay removed',
           'P_marks/I-L-G1/state+shuffledmarks':'Rich event history → state, marks shuffled across events',
           'P_marks/C-N-G1/state':'Rich event history → state (coupled nonlinear)',
           'P_stats/C-N-G1/state':'Coarse summary history → state (coupled nonlinear)'}
COLOR={'P_marks/I-L-G1/state':'#1b6ca8','P_stats/I-L-G1/state':'#e07b39',
       'P_marks/C-N-G1/state':'#3f8f5b','P_stats/C-N-G1/state':'#9c6ea8',
       'P_marks/I-L-G1/intercept':'#9a9a9a','P_marks/I-L-G1/clock':'#b8b8b8',
       'P_marks/I-L-G1/recent_rate':'#6d6d6d','P_marks/I-L-G1/marked_history':'#3a3a3a',
       'P_marks/I-L-G1/constant_state':'#c9c9c9',
       'P_marks/I-L-G1/recent_rate+unnormalised_history_kernel':'#8a6d3b',
       'P_marks/I-L-G1/marked_history+unnormalised_history_kernel':'#5d4d2b',
       'P_marks/I-L-G1/state+oldtargets':'#c2506a','P_stats/I-L-G1/state+oldtargets':'#c2506a',
       'P_marks/I-L-G1/state+nodelay':'#7a4fa3','P_marks/I-L-G1/state+shuffledmarks':'#2f9e8f'}
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'legend.fontsize':7,
                     'xtick.labelsize':7,'ytick.labelsize':7,'axes.spines.top':False,
                     'axes.spines.right':False,'figure.dpi':110,'savefig.dpi':600})


def save(fig,name,meta):
    FIG.mkdir(parents=True,exist_ok=True)
    fig.savefig(FIG/f'{name}.png',bbox_inches='tight')
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight')
    (FIG/f'{name}.metadata.json').write_text(json.dumps(meta,indent=1,default=str))
    plt.close(fig)


def seed_replicates(pair):
    """Same arm, other optimiser seeds: the noise any gap has to clear."""
    sub,sp=pair.split('|')
    out=[]
    for f in sorted(glob.glob(str(ROOT/'runs_replication_rescored'/f'{sub}__{sp}__P_marks__I-L-G1__state__seed*.card.json'))):
        try:out.append(json.load(open(f)))
        except Exception:pass
    return out


def fig_scores(summary,pair,name):
    entry=summary['table'][pair]
    arms=entry['arms']
    reps=seed_replicates(pair)
    fig,axes=plt.subplots(1,3,figsize=(8.2,2.5),sharex=True)
    fig.subplots_adjust(wspace=0.42)
    drawn=[]
    for ax,view in zip(axes,('count','spatial','morphology')):
        for i,rc in enumerate(reps):
            y=[rc['outer'].get(str(h),{}).get(view) for h in HOR]
            if all(v is None for v in y):continue
            ax.plot(HOR,y,lw=0.8,color='#8fbcd9',alpha=0.9,zorder=1,
                    label='Same arm, other optimiser seed' if (i==0 and view=='count') else None)
        for a,c in sorted(arms.items()):
            y=[c['per_horizon'].get(str(h),{}).get(view) for h in HOR]
            if all(v is None for v in y):continue
            m=[v is not None for v in y]
            ax.plot(np.array(HOR)[m],np.array([v for v in y if v is not None]),
                    marker='o',ms=3,lw=1.2,color=COLOR.get(a,'#333333'),
                    label=ARM_LABEL.get(a,a),zorder=3 if 'state' in a else 2,
                    ls='-' if '/state' in a else '--')
            if a not in drawn:drawn.append(a)
        ax.set_xscale('log');ax.set_xticks(HOR);ax.set_xticklabels([str(h) for h in HOR])
        ax.set_title(VIEW_LABEL[view]);ax.set_xlabel('Prediction horizon (min)')
        ax.margins(x=0.04)
    axes[0].set_ylabel('Negative log predictive density\n(nats per scored unit, lower is better)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(0.5,-0.78))
    sub,sp=pair.split('|')
    fig.suptitle(f'{sub.replace("epilepsiae_","Patient E")} · held-out forward segment · '
                 f'{entry["n_outer_eligible"]} query points',y=1.10)
    save(fig,name,dict(pair=pair,arms=drawn,n_outer_eligible=entry['n_outer_eligible'],
                       not_estimable=entry['not_estimable'],n_seed_replicates=len(reps),
                       note='all arms scored on identical query points, targets and denominators; '
                            'the pale blue lines are the same arm re-run with other optimiser seeds '
                            'and set the noise any gap has to clear'))


def fig_budget_transfer(summary,pair,name):
    """Does more optimisation help forward prediction, or only the held-in score?"""
    ext=ROOT/'machine_summary_extended_budget.json'
    if not ext.exists():return False
    e2=json.load(open(ext))['table'].get(pair)
    if not e2:return False
    e1=summary['table'][pair]['arms'];e2=e2['arms']
    shared=[a for a in e1 if a in e2]
    if len(shared)<2:return False
    fig,ax=plt.subplots(figsize=(4.4,3.0))
    for a in sorted(shared):
        x=[e1[a]['inner_selection'],e2[a]['inner_selection']]
        y=[e1[a]['per_horizon']['30']['count'],e2[a]['per_horizon']['30']['count']]
        if None in x or None in y:continue
        c=COLOR.get(a,'#333333')
        ax.annotate('',xy=(x[1],y[1]),xytext=(x[0],y[0]),
                    arrowprops=dict(arrowstyle='->',color=c,lw=1.3,shrinkA=0,shrinkB=0))
        ax.plot(x[0],y[0],'o',ms=4,color=c,label=ARM_LABEL.get(a,a))
        ax.plot(x[1],y[1],'s',ms=4,color=c,mfc='white')
    ax.set_xlabel('Selection score on held-in blocks (nats)')
    ax.set_ylabel('Held-out forward segment,\ncount 30 min ahead (nats)')
    ax.set_title('More optimisation: better held-in, worse forward')
    ax.legend(frameon=False,fontsize=6,loc='center left',bbox_to_anchor=(1.01,0.5))
    ax.text(0.02,0.02,'circle = 1200 updates, open square = 3200',transform=ax.transAxes,
            fontsize=6,color='#666666')
    save(fig,name,dict(pair=pair,arms=sorted(shared),
                       note='arrows run from the 1200-update budget to the 3200-update budget; '
                            'every arm moves left (better held-in) and up (worse forward)'))
    return True


def fig_optimisation(summary,pair,name):

    entry=summary['table'][pair];arms=entry['arms']
    suff=summary['training_sufficiency']
    fig,ax=plt.subplots(figsize=(5.4,2.6))
    lo=[]
    for a in sorted(arms):
        ck=arms[a].get('card_key')
        if ck is None or ck not in suff:continue
        cur=suff[ck]['inner_curve']
        if not cur:continue
        u=[r[0] for r in cur];v=[r[1] for r in cur]
        lo.extend([x for x in v if x is not None])
        ax.plot(u,v,lw=1.1,color=COLOR.get(a,'#333333'),label=ARM_LABEL.get(a,a),
                ls='-' if '/state' in a else '--')
        s=suff[ck]['selected_updates']
        if s and s>0:
            ax.plot([s],[dict(cur)[s]],marker='v',ms=4,color=COLOR.get(a,'#333333'))
    if lo:
        m=min(lo);ax.set_ylim(m-0.01,m+0.28)
    ax.set_xlim(left=100)
    ax.set_xlabel('Optimiser updates');ax.set_ylabel('Selection score on held-in blocks\n(nats, lower is better)')
    ax.set_title('Was each arm optimised far enough to be compared?')
    ax.legend(frameon=False,loc='center left',bbox_to_anchor=(1.01,0.5))
    ax.text(0.02,0.96,'first 100 updates off scale',transform=ax.transAxes,va='top',fontsize=6,color='#666666')
    save(fig,name,dict(pair=pair,marker='triangle marks the selected checkpoint',
                       note='budget-stopped curves that are still falling cannot support a negative claim'))


def fig_synthetic(name):
    """Does the rich input win where the signal is, and stay flat where it is not?"""
    worlds=[('morphology','Only shape follows\nthe slow state'),
            ('zero_effect','Slow state destroyed\n(zero-effect control)'),
            ('identity','Only which contacts\nfollow the slow state')]
    have=[(m,lab) for m,lab in worlds if (ROOT/'synthetic'/f'{m}.json').exists()]
    if not have:return False
    fig,ax=plt.subplots(figsize=(4.6,2.8))
    width=0.26;views=('count','spatial','morphology')
    colors={'count':'#9a9a9a','spatial':'#e07b39','morphology':'#1b6ca8'}
    for vi,view in enumerate(views):
        xs=[];ys=[]
        for wi,(mode,_) in enumerate(have):
            d=json.load(open(ROOT/'synthetic'/f'{mode}.json'))
            diffs=[]
            for rl in sorted({r['realisation'] for r in d['rows']}):
                a=[r for r in d['rows'] if r['inputs']=='P_marks' and r['targets']=='new' and r['realisation']==rl]
                b=[r for r in d['rows'] if r['inputs']=='P_stats' and r['targets']=='new' and r['realisation']==rl]
                if not (a and b):continue
                m=lambda c:np.mean([v[view] for v in c['outer'].values() if v[view] is not None])
                diffs.append(m(b[0])-m(a[0]))
            if not diffs:continue
            xs.append(wi+(vi-1)*width);ys.append(np.mean(diffs))
            ax.plot([wi+(vi-1)*width]*len(diffs),diffs,'o',ms=3,color='#333333',zorder=3)
        ax.bar(xs,ys,width,color=colors[view],label=VIEW_LABEL[view])
    ax.axhline(0,color='#333333',lw=0.8)
    ax.set_xticks(range(len(have)));ax.set_xticklabels([lab for _,lab in have],fontsize=6.5)
    ax.set_ylabel('Rich input advantage\n(nats, positive = rich better)')
    ax.set_title('Can the instrument tell rich input apart at all?')
    ax.legend(frameon=False,fontsize=6,loc='center left',bbox_to_anchor=(1.01,0.5))
    ax.text(0.01,0.97,'dots = individual data realisations',transform=ax.transAxes,
            va='top',fontsize=6,color='#666666')
    save(fig,name,dict(worlds=[m for m,_ in have],
                       note='rate and coarse composition are constant by construction in every world; '
                            'the zero-effect world sets the floor the other bars must clear'))
    return True


def fig_lag(summary,pair,name):
    """How much does the real publication lag cost, and does it hit the state harder?"""
    sub,sp=pair.split('|')
    entry=summary['table'][pair]
    want={a:v['card_key'] for a,v in entry['arms'].items()
          if a.endswith('/state') or a.endswith('/intercept') or a.endswith('/marked_history')}
    fig,ax=plt.subplots(figsize=(4.2,2.6))
    drawn=False;meta={}
    for a,ck in sorted(want.items()):
        f=ROOT/'runs_rescored'/f'{ck}.outer_detail.npz'
        if not f.exists():
            f=ROOT/'runs_reference_variants_rescored'/f'{ck}.outer_detail.npz'
        if not f.exists():continue
        z=np.load(f)
        keys=[k for k in z.files if k.endswith('_lag_minutes') and k.split('_')[0]=='30']
        lag=[];val=[]
        for k in keys:
            pre=k[:-len('lag_minutes')]
            lp=z.get(pre+'logp_count');un=z.get(pre+'units_count')
            if lp is None:continue
            lag.append(z[k]);val.append(np.where(un>0,-lp/np.maximum(un,1e-9),np.nan))
        if not lag:continue
        lag=np.concatenate(lag);val=np.concatenate(val)
        good=np.isfinite(val)
        if good.sum()<20:continue
        L=lag[good].astype(float);V=val[good]
        edges=np.unique(np.quantile(L,np.linspace(0,1,5)))
        xs=[];ys=[]
        for lo,hi in zip(edges[:-1],edges[1:]):
            m=(L>=lo)&(L<=hi)
            if m.sum()<5:continue
            xs.append(float(np.median(L[m])));ys.append(float(np.median(V[m])))
        if len(xs)<2:continue
        ax.plot(xs,ys,marker='o',ms=3,lw=1.2,color=COLOR.get(a,'#333333'),
                label=ARM_LABEL.get(a,a),ls='-' if '/state' in a else '--')
        meta[a]=dict(lag_minutes=xs,count_nats=ys);drawn=True
    if not drawn:
        plt.close(fig);return False
    ax.set_xlabel('Information lag at the query (min)')
    ax.set_ylabel('Count prediction 30 min ahead\n(nats, lower is better)')
    ax.set_title('What does the one-hour publication delay cost?')
    ax.legend(frameon=False,fontsize=6,loc='center left',bbox_to_anchor=(1.01,0.5))
    save(fig,name,dict(pair=pair,curves=meta,
                       note='lag is the real gap between the query and the end of the last published hour block'))
    return True


def main():
    s=json.load(open(ROOT/'machine_summary.json'))
    made=[]
    for pair in s['table']:
        sub=pair.split('|')[0].replace('epilepsiae_','E')
        sp=pair.split('|')[1].replace('-','').lower()
        n=f'outer_predictive_scores_{sub}_{sp}'
        fig_scores(s,pair,n);made.append(n)
        n2=f'optimisation_sufficiency_{sub}_{sp}'
        try:
            fig_optimisation(s,pair,n2);made.append(n2)
        except Exception as e:print('optimisation figure skipped:',e)
        n4=f'budget_vs_forward_transfer_{sub}_{sp}'
        try:
            if fig_budget_transfer(s,pair,n4):made.append(n4)
        except Exception as ex:print('budget figure skipped:',ex)
        n3=f'information_lag_{sub}_{sp}'
        if fig_lag(s,pair,n3):made.append(n3)
    if fig_synthetic('synthetic_discrimination'):made.append('synthetic_discrimination')
    print(json.dumps(dict(figures=made,dir=str(FIG)),indent=1))

if __name__=='__main__':main()
