#!/usr/bin/env python3
"""Render actual native extrema and computed reduced equilibria; no invented loci."""
import json
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import find_peaks

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1'
OUT=ROOT/'results/topic4_sef_hfo/fig5_z_bifurcation_preview_20260915'
FIG=OUT/'figures'
COLORS=dict(eq='#456c9e',peak='#238d80',mean='#d08c23',trough='#87bab0')


def native():
    path=np.load(OUT/'spatial_z_path.npz');times=path['time_s'];ss=path['depletion']
    rows=[];series={}
    for source in (8000,9000,9300,9420,9870,10370):
        depletion=float(ss[np.argmin(abs(times-source/1000.))])
        for history in (8000,10370):
            for future in ('W1','W2'):
                name=f'z{source}_h{history}_{future}'
                folder=SOURCE/'native/runs'/name
                rates=[]
                for f in sorted((folder/'chunks').glob('*.npz')):
                    a=np.load(f);rates.append(a['spikes_1ms'][:,0].astype(float))
                counts=np.concatenate(rates)
                rate=counts.reshape(-1,10).sum(1)/32000/.01
                assert len(rate)==1000 and np.isfinite(rate).all()
                tail=rate[-400:]
                peaks=find_peaks(tail,prominence=20.,distance=2)[0]
                troughs=find_peaks(-tail,prominence=20.,distance=2)[0]
                row=dict(name=name,z_source_s=source/1000.,history_s=history/1000.,future=future,
                         depletion=depletion,mean_z=1-depletion,mean_rate_hz=float(tail.mean()),
                         min_rate_hz=float(tail.min()),max_rate_hz=float(tail.max()),
                         peak_rates_hz=tail[peaks].tolist(),trough_rates_hz=tail[troughs].tolist())
                rows.append(row);series[name]=rate
    (OUT/'native_extrema.json').write_text(json.dumps(rows,indent=2)+'\n')
    with (OUT/'native_extrema_summary.csv').open('w') as f:
        keys=[k for k in rows[0] if not k.endswith('rates_hz')]
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        writer.writerows({k:r[k] for k in keys} for r in rows)
    return rows,series


def draw(single=False):
    FIG.mkdir(parents=True,exist_ok=True)
    rows,series=native()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.labelsize':16,
                         'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,
                         'svg.fonttype':'none','axes.linewidth':1.1})
    fig=plt.figure(figsize=(10.5,8.8) if single else (11,10.4))
    gs=fig.add_gridspec(1 if single else 2,3,height_ratios=[1] if single else [3.4,1],hspace=.40,wspace=.22,
                       left=.13,right=.965,bottom=.12 if single else .08,top=.735 if single else .76)
    ax=fig.add_subplot(gs[0,:])
    summaries=[]; all_x=[]
    for fname in ('low_equilibria','recruited_equilibria','core_equilibria'):
        f=OUT/f'{fname}.json'
        if not f.exists():continue
        data=json.loads(f.read_text());pts=data['points']
        if len(pts)<2:continue
        sx=np.array([p['s'] for p in pts]);ry=np.array([p['mean_e_hz'] for p in pts])
        valid=(sx>=0)&(sx<=.32)
        all_x.extend(sx[valid].tolist())
        ax.plot(sx[valid],ry[valid],color=COLORS['eq'],lw=2.3,zorder=3)
        # Open endpoints designate finite computation coverage, not bifurcations.
        ax.plot(sx[[0,-1]],ry[[0,-1]],'o',mfc='white',mec=COLORS['eq'],ms=5,zorder=4)
        summaries.append(dict(file=fname,points=len(pts),max_residual_hz=max(p['residual_max_hz'] for p in pts)))
    representative=[r for r in rows if r['history_s']==8 and r['future']=='W1']
    x=np.array([r['depletion'] for r in representative])
    means=np.array([r['mean_rate_hz'] for r in representative])
    maxima=np.array([r['max_rate_hz'] for r in representative])
    minima=np.array([r['min_rate_hz'] for r in representative])
    # Only the observed Z fields have measurements. No smoothed or fitted envelope.
    for j,r in enumerate(representative):
        sx=r['depletion'];p=np.asarray(r['peak_rates_hz']);v=np.asarray(r['trough_rates_hz'])
        if len(p):ax.scatter(np.full(len(p),sx),p,s=12,color=COLORS['peak'],alpha=.55,zorder=4)
        if len(v):ax.scatter(np.full(len(v),sx),v,s=10,color=COLORS['trough'],alpha=.6,zorder=4)
        ax.plot([sx,sx],[r['min_rate_hz'],r['max_rate_hz']],color=COLORS['peak'],alpha=.35,lw=1)
    ax.scatter(x,maxima,marker='^',s=54,color=COLORS['peak'],zorder=5)
    ax.scatter(x,minima,marker='v',s=48,facecolor='white',edgecolor=COLORS['peak'],zorder=5)
    ax.scatter(x,means,marker='o',s=56,color=COLORS['mean'],edgecolor='white',lw=.6,zorder=6)
    # Other histories/innovations are separate observed outcomes, not confidence intervals.
    for r in rows:
        if r in representative:continue
        ax.scatter(r['depletion'],r['mean_rate_hz'],marker='o',s=25,facecolor='none',edgecolor=COLORS['mean'],alpha=.75,zorder=5)
    mark=representative[3]
    ax.axvline(mark['depletion'],ls=':',color='#888888',lw=1.1,zorder=1)
    ax.text(mark['depletion']+.003,365,'Fig.5 ③ field',fontsize=12,color='#555555')
    xmin=max(0,min(all_x+x.tolist())-.012)
    ax.set_xlim(xmin,.32);ax.set_ylim(-7,390)
    ax.set_xlabel(r'Spatial Z depletion, $s=1-\langle Z\rangle_E$')
    ax.set_ylabel(r'Global E population rate (Hz / neuron)')
    ax.set_xticks([v for v in [0,.05,.1,.15,.2,.25,.3] if v>=xmin]);ax.set_yticks([0,100,200,300])
    top=ax.secondary_xaxis('top',functions=(lambda s:1-s,lambda z:1-z))
    top.set_xlabel(r'Mean Z of the prescribed spatial field');top.set_xticks([1.,.9,.8,.75,.7])
    ax.grid(axis='y',alpha=.12)
    if (OUT/'fold_check.json').exists():
        fold=json.loads((OUT/'fold_check.json').read_text())
        if fold['status']=='NUMERICAL_STATIONARY_FOLD_CONFIRMED':
            ax.plot(fold['s'],fold['mean_e_hz'],'o',ms=7,mfc='white',mec=COLORS['eq'],mew=1.5,zorder=8)
            ax.annotate('Fold of reduced high-rate branch',
                        xy=(fold['s'],fold['mean_e_hz']),xytext=(.225,335),fontsize=11,
                        arrowprops=dict(arrowstyle='-',color='#555555',lw=.9),color='#45566b')
    handles=[Line2D([],[],color=COLORS['eq'],lw=2.3,label='Reduced equilibria; stability unclassified'),
             Line2D([],[],marker='^',color=COLORS['peak'],linestyle='none',label='Native maximum / minimum'),
             Line2D([],[],marker='o',color=COLORS['mean'],linestyle='none',label='Native time mean'),
             Line2D([],[],marker='.',color=COLORS['peak'],linestyle='none',label='Native individual extrema')]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.12,.872),ncol=2,
               frameon=False,fontsize=10.5,columnspacing=1.6,handlelength=2.)
    for j,source in enumerate(() if single else (8000,9420,9870)):
        a=fig.add_subplot(gs[1,j]);r=next(q for q in representative if q['z_source_s']==source/1000)
        t=np.arange(400)*.01
        a.plot(t,series[r['name']][-400:],color='#2e4855',lw=.85)
        a.set(xlim=(0,4),ylim=(-5,300),xticks=[0,2,4],yticks=[0,100,200,300])
        a.set_xlabel('Time within final 4 s (s)',fontsize=12)
        a.set_title(r'$\langle Z\rangle_E=$'+f"{r['mean_z']:.3f}"+('  (③)' if source==9420 else ''),fontsize=13)
        if j==0:a.set_ylabel('Global E rate (Hz)',fontsize=12)
        else:a.set_yticklabels([])
    fig.text(.13,.957,'Spatial Z and collective activity',fontsize=21,weight='bold')
    fig.text(.13,.922,'Computed equilibria + native burst extrema · current Fig.5 network',fontsize=12,color='#454545')
    fig.text(.13,.891,'Exploratory: the v1 reduction does not reproduce native bursting.',fontsize=12,color='#a14436')
    stem='fig5_spatial_z_bifurcation_single_axis' if single else 'fig5_spatial_z_bifurcation_preview'
    for ext in ('png','pdf','svg'):
        fig.savefig(FIG/f'{stem}.{ext}',dpi=200)
    plt.close(fig)
    manifest=dict(source=str(SOURCE),statistical_unit='one frozen-Z continuation',native_runs=24,
                  native_main='history 8 s, W1; final 4 s of a 10 s continuation',native_other='other histories/futures: open mean markers',
                  extrema_definition='10 ms population rate; inner extrema prominence >=20 Hz and distance >=20 ms; absolute min/max always shown',
                  equilibrium_branches=summaries,branch_stability='NOT_COMPUTED',native_irregular_bifurcation_type='NOT_ESTABLISHED',
                  reduced_stationary_fold=json.loads((OUT/'fold_check.json').read_text()) if (OUT/'fold_check.json').exists() else None,
                  status='CANDIDATE_PENDING_AUTHOR_VISUAL_REVIEW',model_correspondence='FAILED_IN_PRIOR_MILESTONE',
                  native_physical_simulations_added=0)
    (OUT/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    print(json.dumps(draw(),indent=2))
    draw(single=True)
