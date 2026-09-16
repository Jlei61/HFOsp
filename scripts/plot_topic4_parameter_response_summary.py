"""Compact fixed-network parameter responses from the complete long panel.

Two populations are rendered separately. No pooled-seed curve, significance
claim, new patient fit or new event definition.
"""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scripts import analyze_topic4_propagation_recovery_night as s


def main():
    source=s.night.OUT/'analysis_long';rows=list(csv.DictReader((source/'run_observations.csv').open()))
    counts=list(csv.DictReader((source/'counts.csv').open()))
    lookup={(r['base_id'],r['seed'],r['layer'],r['mode']):r for r in rows};rates={(r['base_id'],r['seed']):r for r in counts}
    ref=s.rt.read(s.an.run.OUT/'analysis/patient_observation_reference.json')
    # These are explicit one-factor contrasts. The two backgrounds in column1
    # share the same EE values; other columns retain the near-SCL geometry.
    columns=[dict(title='核内 E→E 强度',xlabel='相对连接权重',color='#356eb8',
        series=[('靠近上部SCL','-',[(.75,'refine_near_EE075'),(.85,'refine_near_EE085')]),
                ('端点上移3mm','--',[(.75,'refine_mid_EE075'),(.85,'refine_mid_EE085')])]),
        dict(title='左核易激性',xlabel='阈值降低幅度的倍数',color='#d77820',
        series=[('靠近上部SCL','-',[(1.,'refine_near_EE085'),(1.15,'refine_near_EE085_A115')])]),
        dict(title='两核外部输入均值',xlabel='相对到达率均值',color='#269977',
        series=[('靠近上部SCL','-',[(1.,'refine_near_EE085'),(.95,'refine_near_EE085_mean095')])])]
    dest=s.night.OUT/'parameter_response_summary';F=dest/'figures';F.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':9,'pdf.fonttype':42})
    records=[];files=[]
    for layer in ['primary','all_detected']:
        fig,axes=plt.subplots(4,3,figsize=(11,10),layout='constrained',sharex='col')
        for col,c in enumerate(columns):
            for label,ls,points in c['series']:
                for si,seed in enumerate(['847101','847102']):
                    xx=[];yy=[[],[],[],[]]
                    for value,cid in points:
                        r=lookup[(cid,seed,layer,'ALL')];a=lookup[(cid,seed,layer,'TA')];n=int(r['n']);na=int(a['n'])
                        vals=[n,na/n if n else np.nan,float(a['ICL_contact_participation']) if na else np.nan,float(rates[(cid,seed)]['coreBE_rate_hz'])]
                        xx.append(value)
                        for i,v in enumerate(vals):yy[i].append(v)
                        records.append(dict(layer=layer,parameter=c['title'],layout=label,candidate=cid,seed=seed,parameter_value=value,
                            n=n,TA_n=na,TA_fraction=vals[1],TA_ICL_mean_participation=vals[2],native_B_rate_hz=vals[3]))
                    for row in range(4):
                        axes[row,col].plot(xx,yy[row],ls=ls,marker='os'[si],c=c['color'],lw=1.2,ms=5,
                            markerfacecolor=c['color'] if si==0 else 'white')
                    for x,y in zip(xx,yy[0]):axes[0,col].annotate(str(int(y)),(x,y),xytext=(4,4 if si==0 else -10),textcoords='offset points',fontsize=7)
            axes[0,col].set_title(c['title'],color=c['color'],fontsize=12)
            axes[-1,col].set_xlabel(c['xlabel'])
            axes[1,col].axhline(ref['TA']['n']/(ref['TA']['n']+ref['TB']['n']),c='.55',ls=':',lw=1)
            axes[2,col].axhline(ref['TA']['ICL_contact_participation'],c='.55',ls=':',lw=1)
            for row,ax in enumerate(axes[:,col]):
                ax.spines[['top','right']].set_visible(False);ax.margins(x=.2)
                if row in [1,2]:ax.set_ylim(0,1.04)
                else:ax.set_ylim(bottom=0)
                ax.set_xticks(sorted({v for _,_,pts in c['series'] for v,_ in pts}))
        for row,label in enumerate(['每条60秒运行的事件数','TA事件比例','TA事件中ICL的平均参与比例','右核E：完整时段每神经元率 (Hz)']):
            for ax in axes[row]:ax.set_ylabel(label)
        handles=[Line2D([],[],marker='o',color='.3',ls='',label='噪声847101'),Line2D([],[],marker='s',mfc='white',color='.3',ls='',label='噪声847102'),
            Line2D([],[],color='.3',ls='-',label='靠近上部SCL'),Line2D([],[],color='.3',ls='--',label='端点上移3mm（仅左列）'),
            Line2D([],[],color='.55',ls=':',label='患者FIT描述参考')]
        fig.legend(handles=handles,loc='outside lower center',ncol=3,fontsize=8)
        population='原合格孤立窗' if layer=='primary' else '全部检测：仅用于开发诊断'
        fig.suptitle('参数改变怎样影响事件组成、招募与原生活动？\n'+population+'；固定图2511，保留两条噪声；灰点线不构成验收阈值',fontsize=12)
        for ext in ['png','pdf']:
            name=f'{layer}_parameter_response_summary.{ext}';fig.savefig(F/name,dpi=190);files.append(name)
        plt.close(fig)
    s.an.writecsv(dest/'values.csv',records)
    s.rt.write(dest/'manifest.json',dict(status='DESCRIPTIVE_FIXED_NETWORK_RESULT',producer=__file__,producer_sha256=s.rt.sha(__file__),sources=[str(source/'run_observations.csv'),str(source/'counts.csv')],
        statistical_unit='One complete60s run on topology2511. Lines pair the same noise identity; not a claim of other-topology replication.',
        physical_pairing='Within each EE or threshold contrast, graph and per-step actual input match where audited. Geometry or input mean changes alter actual input realizations.',
        population='Primary and all-detection separate figures; the latter has different qualification from patient FIT and is not formal fit acceptance.',
        conditional_missing='TA participation left undefined if no observed TA. More TA labels or more ICL recruitment need not imply correct patient paths.',
        no_claim='Two points are a tested response, not a continuous dose-response law or a parameter optimum. Native rate uses all activity after1500ms, independent of event filtering.'))
    (F/'README.md').write_text('\n\n'.join(f'### {file}\n\n三列分别只改变核内EE、左核阈值降幅或两核外部输入均值，四行显示事件量、TA比例、TA的ICL参与和完整原生右核率。颜色表示参数类别，圆/空方保留两噪声；只有EE列比较两种布局，实/虚线表示位置，原合格与全检测分别成图。**关注点**：EE作用随位置反转，以及更多TA、更多事件或局部参数变化是否伴随招募取舍与非局部活动重排；空值不是零误差，患者灰线仅为描述参考，尚不能由此接受患者传播恢复。' for file in files)+'\n')
    print({'output':str(dest),'files':len(files)},flush=True)


if __name__=='__main__':main()
