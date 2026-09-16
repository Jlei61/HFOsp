"""Relate full native core timing to contact timing without claiming causality."""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scripts import analyze_topic4_propagation_recovery_night as s


def main():
    cases=[('long','refine_mid_EE075','上移3mm；核内EE×0.75'),
        ('long','refine_mid_EE085_mean095','上移3mm；EE×0.85、输入均值×0.95'),
        ('final_A','refine_midpoint_EE085_A115_mean095','中点；EE×0.85、均值×0.95、左核降幅×1.15')]
    if (s.night.OUT/'analysis_final_B/event_timing.csv').exists():
        cases.extend(('final_B',c['id'],c['display_name']) for c in s.rt.read(s.night.OUT/'final_B_selection.json')['candidates'])
    sources={phase:list(csv.DictReader((s.night.OUT/f'analysis_{phase}/event_timing.csv').open())) for phase,_,_ in cases}
    selected=[];summaries=[]
    for phase,cid,title in cases:
        for seed in ['847101','847102']:
            for mode in ['TA','TB']:
                rr=[r for r in sources[phase] if r['base_id']==cid and r['seed']==seed and r['mode']==mode and r['primary']=='True']
                xy=[]
                for r in rr:
                    try:x=float(r['B_minus_A_t10_ms']);y=float(r['centroid_SCL_minus_ICL_ms'])
                    except (ValueError,KeyError):continue
                    if not np.isfinite([x,y]).all():continue
                    xy.append((x,y));selected.append(dict(phase=phase,candidate=cid,seed=seed,mode=mode,event=int(r['event']),core_B_minus_A_t10_ms=x,SCL_minus_ICL_centroid_ms=y))
                core=np.array([float(r['B_minus_A_t10_ms']) for r in rr]);core=core[np.isfinite(core)]
                summaries.append(dict(candidate=cid,seed=seed,mode=mode,primary_n=len(rr),both_rods_plotted_n=len(xy),
                    A_earlier_t10_fraction=float((core>0).mean()) if len(core) else None,core_t10_n=len(core)))
    dest=s.night.OUT/'core_to_contact_timing';F=dest/'figures';F.mkdir(parents=True,exist_ok=True)
    s.an.writecsv(dest/'events.csv',selected);s.an.writecsv(dest/'run_summary.csv',summaries)
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':9,'pdf.fonttype':42})
    files=[]
    for group,cc in [('working_points',cases[:3]),('correlation',cases[3:])]:
        if not cc:continue
        fig,axes=plt.subplots(1,len(cc),figsize=(5*len(cc),4.8),layout='constrained',squeeze=False)
        axes=axes[0]
        for ax,(phase,cid,title) in zip(axes,cc):
            for si,seed in enumerate(['847101','847102']):
                for mode,color in [('TA','#bd3934'),('TB','#2679b0')]:
                    rr=[r for r in selected if r['candidate']==cid and r['seed']==seed and r['mode']==mode]
                    ax.scatter([r['core_B_minus_A_t10_ms'] for r in rr],[r['SCL_minus_ICL_centroid_ms'] for r in rr],
                        c=color,marker='os'[si],s=19,alpha=.6,linewidths=.3,edgecolor='white')
            ax.axvline(0,c='.5',lw=.8);ax.axhline(0,c='.5',lw=.8)
            ax.set(title=title,xlabel='右核 − 左核：10%累计活动时间 (ms)\n正值表示左核较早',ylabel='SCL − ICL：参与质心中位差 (ms)\n正值表示SCL较晚')
            ax.spines[['top','right']].set_visible(False)
            lines=[]
            for seed in ['847101','847102']:
                rr=[r for r in summaries if r['candidate']==cid and r['seed']==seed]
                lines.append(seed+'：'+', '.join(f"{r['mode']} {r['both_rods_plotted_n']}/{r['primary_n']}" for r in rr))
            ax.text(.02,.98,'\n'.join(lines),transform=ax.transAxes,ha='left',va='top',fontsize=7,bbox=dict(fc='white',ec='none',alpha=.85))
            ax.margins(.15)
        legend=[Line2D([],[],marker='o',ls='',color=c,label=m) for m,c in [('TA','#bd3934'),('TB','#2679b0')]]
        legend += [Line2D([],[],marker=mark,ls='',color='.4',label='噪声'+seed) for mark,seed in [('o','847101'),('s','847102')]]
        fig.legend(handles=legend,loc='outside lower center',ncol=4,fontsize=9)
        fig.suptitle('两核的先后差别，是否传成了患者所需的两杆时序？\n固定图2511，原合格事件；每点一事件，数字为可画两杆事件/该类全部事件。累计活动时间不是因果起源。',fontsize=11)
        for ext in ['png','pdf']:
            name=f'{group}_native_to_contact_timing.{ext}';fig.savefig(F/name,dpi=190);files.append(name)
        plt.close(fig)
    s.rt.write(dest/'manifest.json',dict(status='DESCRIPTIVE_LINK_DIAGNOSTIC',producer=__file__,producer_sha256=s.rt.sha(__file__),
        sources=[str(s.night.OUT/f'analysis_{p}/event_timing.csv') for p in sources],
        definition='x: each core full-window cumulative spike10% time, B-A. y: median actual participating SCL centroid minus median actual participating ICL centroid.',
        exclusion='One-rod events cannot supply y and are explicitly counted; they are not removed from the frozen objective or mode support.',
        boundary='Core cumulative timing does not identify causal source. Different participant sets can change rod medians. Events nested in one topology and two noise runs, no independent-event significance claim.'))
    (F/'README.md').write_text('\n\n'.join(f'### {file}\n\n横轴为每个core在完整事件窗内累积到自身10%发放量的时间差，纵轴为参与SCL与ICL触点各自质心中位数之差；每点是原合格事件，颜色分TA/TB，符号分噪声。单杆事件不能计算纵轴，因此图内同时报告可画事件数和全部事件数，不把它们从训练或支持统计删除。**关注点**：左核较早时SCL是否仍较晚，以及核心时序分离是否足以产生正确接触结构；累计质量不是因果起源，杆间中位受参与集合影响。' for file in files)+'\n')
    print({'output':str(dest),'events':len(selected),'files':len(files)},flush=True)


if __name__=='__main__':main()
