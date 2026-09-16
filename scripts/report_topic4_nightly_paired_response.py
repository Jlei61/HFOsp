#!/usr/bin/env python3
"""Paired diagnostic responses; does not accept a workpoint or infer a global effect."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/topic4_sef_hfo/nightly_central_workpoint'
LABELS={'central_A_anchor1_geom1':'Central parent','central_B_EI_down':'E to I -20%','central_B_IE_down':'I to E -20%','central_B_EI_down_IE_up':'E to I -20%; I to E +15%','central_B_EE_down_EI_down':'E to E -10%; E to I -20%','central_B_GABA_long':'GABA decay: 21.44 to 32 ms','broad_B_geometry':'Broader core positions'}
KEYS=['participation','centroid_structure','local_shape','recruitment','joint_envelope']
KL=['Participation error','Centroid order / lag error','Local shape error','Recruitment error','Full envelope error']

def run():
    a=json.loads((OUT/'batches/A/scores.json').read_text())['candidates']
    b=json.loads((OUT/'batches/B/scores.json').read_text())['candidates']
    parent=next(r for r in a if r['candidate_id']=='central_A_anchor1_geom1')
    rows=[parent]+[r for r in b if not r['control']]
    f=OUT/'figures';f.mkdir(exist_ok=True)
    records=[]
    fig,axs=plt.subplots(3,3,figsize=(16,12),sharey=True)
    titles=KL+['Local width median (ms)','Recruitment span median (ms)','Contacts per event median','Primary event count']
    for j,ax in enumerate(axs.ravel()):
        ax.set_title(titles[j]);ax.axvline(0,color='.65',lw=.7);ax.grid(axis='x',alpha=.15)
        for i,r in enumerate(rows):
            for offset,(uid,u) in zip([-.12,.12],sorted(r['units'].items())):
                ref=parent['units'][uid];eligible=u['observation']['loss'] is not None
                if j<5:
                    v=u['observation'].get('blocks',{}).get(KEYS[j],{}).get('D_off')
                    p=ref['observation']['blocks'][KEYS[j]]['D_off']
                elif j<8:
                    key=['local_width_ms','recruitment_span_ms','n_contacts'][j-5]
                    v=u['distributions'][key]['median'];p=ref['distributions'][key]['median']
                else:v=u['observation']['N'];p=ref['observation']['N']
                # N remains meaningful as observation support even for early runaway.
                if j!=8 and not eligible:v=None
                delta=None if v is None else float(v-p)
                records.append(dict(candidate_id=r['candidate_id'],unit=uid,observable=titles[j],value=v,parent_value=p,paired_delta=delta,physical_status=u['physical_status'],N=u['observation']['N']))
                if delta is not None:
                    ax.scatter(delta,i+offset,marker='o' if '6101' in uid else '^',color='#4079a8' if '6101' in uid else '#b16b43',s=32)
                elif j==0:ax.text(.03,i+offset,'not estimable: '+u['physical_status'],transform=ax.get_yaxis_transform(),fontsize=6,va='center',color='#a33')
        ax.set_xlabel('Change from central parent');ax.set_yticks(range(len(rows)),[LABELS[r['candidate_id']] for r in rows]);ax.set_ylim(len(rows)-.5,-.5)
    fig.suptitle('Paired parameter responses: same topology and dynamics seed\nBlue circles: topology 6101; brown triangles: 6102. Error changes < 0 are improvements.',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.94))
    for ext in ['png','pdf']:fig.savefig(f/f'paired_parameter_responses.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)
    with (OUT/'paired_parameter_responses.csv').open('w') as stream:
        w=csv.DictWriter(stream,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    # Raw ranges complement deltas; event variance is not network replication.
    fig,axs=plt.subplots(1,3,figsize=(15,6),sharey=True)
    pat=list(csv.DictReader((OUT/'batches/B/conditional_distributions.csv').open()))
    for ax,key,title in zip(axs,['local_width_ms','recruitment_span_ms','n_contacts'],['Local width (ms)','Recruitment span (ms)','Participating contacts']):
        ref=next(r for r in pat if r['source']=='patient_TRAIN' and r['mode']=='all' and r['metric']==key)
        ax.axvspan(float(ref['q05']),float(ref['q95']),alpha=.13,color='black',label='Patient TRAIN: 5-95%');ax.axvline(float(ref['median']),color='black',ls='--',label='Patient TRAIN: median')
        for i,r in enumerate(rows):
            for offset,(uid,u) in zip([-.12,.12],sorted(r['units'].items())):
                if u['observation']['loss'] is None:
                    if key=='local_width_ms':ax.text(.1,i+offset,'RUNAWAY: no primary events',transform=ax.get_yaxis_transform(),fontsize=6,color='#a33',va='center')
                    continue
                d=u['distributions'][key];color='#4079a8' if '6101' in uid else '#b16b43'
                ax.plot([d['q05'],d['q95']],[i+offset]*2,color=color,lw=2);ax.scatter(d['median'],i+offset,color=color,marker='o' if '6101' in uid else '^')
        ax.set(title=title,yticks=range(len(rows)),yticklabels=[LABELS[r['candidate_id']] for r in rows],ylim=(len(rows)-.5,-.5));ax.grid(axis='x',alpha=.15)
    axs[0].plot([],[], 'o',color='#4079a8',label='Topology 6101');axs[0].plot([],[], '^',color='#b16b43',label='Topology 6102');axs[0].legend(fontsize=7);fig.suptitle('Event distributions within each network: median and 5-95% range\nPatient reference: weighted 46 TRAIN envelopes; model: all primary events per complete run.')
    fig.tight_layout(rect=(0,0,1,.93))
    for ext in ['png','pdf']:fig.savefig(f/f'paired_observable_ranges.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)
    (f/'README.md').write_text('### paired_parameter_responses.png / .pdf\n\n每个条件相对同一中央双核出发点，分别在拓扑 6101、6102 和固定噪声 842901 下计算变化。蓝圆和棕三角代表两张拓扑；前五项误差下降表示改善，后三项物理观测不预设越大越好。两参数共同改变的行不能当作单参数主效应。\n\n**关注点**：runaway 的拟合与时间指标不可估计，空白不是零；事件数仅表示有效观测支持。这是未被接受的基底上的诊断，不能代替可靠工作点上的机制响应。\n\n### paired_observable_ranges.png / .pdf\n\n每网络显示所有 primary 事件的中位数和 5–95% 范围，背景为患者 46 条 TRAIN 包络的加权参考。横线描述事件变异，不是网络层置信区间；未将 TA/TB 分组均值当作全部事件分布。\n\n**关注点**：同时看局部持续时间、跨接触点招募和参与数量，不能用某一项改善掩盖另一项缺口。\n')
if __name__=='__main__':run()
