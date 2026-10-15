"""Descriptive paired responses from completed frozen runs; no new score or gate."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('/data/hfosp/topic4_sef_hfo/core_propagation_recovery_20260911')
COLORS={'EE':'#356eb8','Vth':'#d77820','input':'#269977','geometry':'#9256ad','correlation':'#ad3f83'}

def value(row,key):
    try:return float(row[key])
    except (KeyError,TypeError,ValueError):return np.nan

def contrasts():
    pairs=[]
    for layout,title in [('near','靠近上部 SCL'),('mid','端点上移 3 mm'),('midpoint','两位置中点')]:
        pre='refine_'+layout+'_'
        for family,a,b,change in [('EE','EE075','EE085','核内 EE：0.75 → 0.85'),
              ('Vth','EE085','EE085_A115','左核阈值降幅：1 → 1.15'),
              ('input','EE085','EE085_mean095','核内输入均值：1 → 0.95'),
              ('Vth','EE085_mean095','EE085_A115_mean095','输入 0.95 下，左核阈值降幅：1 → 1.15'),
              ('input','EE085_A115','EE085_A115_mean095','左核降幅 1.15 下，输入均值：1 → 0.95')]:
            pairs.append((family,pre+a,pre+b,title+'；'+change))
        pairs.append(('Vth',pre+'EE075',pre+'EE075_A115',title+'；EE 0.75 下，左核阈值降幅：1 → 1.15'))
    for recipe,title in [('EE075','EE 0.75'),('EE085','EE 0.85'),('EE085_A115','EE 0.85、左核降幅 1.15'),('EE085_mean095','EE 0.85、输入均值 0.95')]:
        pairs.append(('geometry','refine_near_'+recipe,'refine_mid_'+recipe,'左核位置：靠近 SCL → 端点上移 3 mm；'+title))
    for recipe,title in [('EE075','EE 0.75'),('EE085_mean095','EE 0.85、输入均值 0.95')]:
        pairs.append(('geometry','refine_mid_'+recipe,'refine_midpoint_'+recipe,'左核位置：端点上移 3 mm → 两位置中点；'+title))
    for recipe,title in [('mid_EE075','EE 0.75'),('mid_EE085_mean095','EE 0.85、输入均值 0.95')]:
        parent='refine_'+recipe
        a='coreOU_'+recipe+'_rho050';b='coreOU_'+recipe+'_rho000'
        pairs.extend([('correlation',parent,a,title+'；两核慢输入相关系数：1 → 0.5'),
            ('correlation',parent,b,title+'；两核慢输入相关系数：1 → 0'),
            ('correlation',a,b,title+'；两核慢输入相关系数：0.5 → 0')])
    return pairs

def main(phase):
    source=OUT/('analysis_'+phase)
    with (source/'run_observations.csv').open() as f:rows=list(csv.DictReader(f))
    with (source/'counts.csv').open() as f:counts=list(csv.DictReader(f))
    lookup={(r['base_id'],r['topology_seed'],r['seed'],r['layer'],r['mode']):r for r in rows}
    spec=json.loads((OUT/f'{phase}_units.json').read_text())
    units=sorted({(str(t),str(s)) for _,t,s in spec['units']})
    dest=OUT/('parameter_response_'+phase);F=dest/'figures';F.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':8,'pdf.fonttype':42})
    metrics=[('n','事件数变化'),('SCL_upper_participation','SCL9/8 参与概率变化'),
        ('ICL_contact_participation','ICL 平均参与概率变化'),('participation_mae','参与误差变化\n负值：更接近患者'),
        ('pair_order_probability_mae','触点对顺序误差变化\n负值：更接近患者'),
        ('B_minus_A_t10_ms_median','右核−左核 10% 时间变化 (ms)')]
    records=[];files=[]
    for layer in ['primary','all_detected']:
      for mode in ['ALL','TA','TB']:
        present=[p for p in contrasts() if any((p[1],t,s,layer,mode) in lookup and (p[2],t,s,layer,mode) in lookup for t,s in units)]
        if not present:continue
        fig,axes=plt.subplots(1,len(metrics),figsize=(19,max(4,len(present)*.62+1.9)),sharey=True,layout='constrained')
        for pi,(family,parent,child,title) in enumerate(present):
          for ui,(topo,seed) in enumerate(units):
            a=lookup.get((parent,topo,seed,layer,mode));b=lookup.get((child,topo,seed,layer,mode))
            if a is None or b is None:continue
            offset=(ui-(len(units)-1)/2)*min(.18,.55/max(len(units)-1,1))
            for ax,(key,label) in zip(axes,metrics):
                av=value(a,key);bv=value(b,key);delta=bv-av
                records.append(dict(family=family,contrast=title,parent=parent,child=child,topology_seed=topo,dynamics_seed=seed,layer=layer,mode=mode,
                    observable=key,parent_value=av if np.isfinite(av) else None,child_value=bv if np.isfinite(bv) else None,delta=delta if np.isfinite(delta) else None,
                    parent_events=int(a['n']),child_events=int(b['n'])))
                if np.isfinite(delta):ax.plot(delta,pi+offset,marker=['o','s','^','D'][ui%4],color=COLORS[family],ms=4,ls='none')
            axes[0].text(1.01,pi+offset,f"{a['n']}→{b['n']}",transform=axes[0].get_yaxis_transform(),fontsize=6,va='center')
        for ax,(key,label) in zip(axes,metrics):
            ax.axvline(0,c='.7',lw=.8);ax.grid(axis='y',alpha=.15);ax.set_xlabel(label)
            if key in ('SCL_upper_participation','ICL_contact_participation','participation_mae','pair_order_probability_mae'):ax.set_xlim(-1.02,1.02)
        axes[0].set(yticks=range(len(present)),yticklabels=[p[3] for p in present]);axes[0].invert_yaxis()
        pop='原 primary 集合' if layer=='primary' else '全部检测，仅作开发诊断'
        identity='；'.join(f"{['圆','方','三角','菱形'][i%4]}：图 {t}、噪声 {s}" for i,(t,s) in enumerate(units))
        fig.suptitle(f"{mode}｜单项参数改变后的观测差值：后者 − 前者｜{pop}\n{identity}\n蓝：EE；橙：阈值；绿：均值；紫：位置；洋红：两核输入相关性。每点是一条完整重演；事件数写为前→后，缺失模式的分布指标留空。",fontsize=10)
        stem=f'{mode}_{layer}_paired_changes'
        for ext in ['png','pdf']:
            path=F/f'{stem}.{ext}';fig.savefig(path,dpi=180);files.append(path.name)
        plt.close(fig)
    if records:
        with (dest/'paired_changes.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    native_lookup={(c['base_id'],c['topology_seed'],c['seed']):c for c in counts}
    present=[p for p in contrasts() if any((p[1],t,s) in native_lookup and (p[2],t,s) in native_lookup for t,s in units)]
    native=[]
    if present:
        fig,axes=plt.subplots(2,2,figsize=(13,max(6,len(present)*.75+3)),sharey=True,layout='constrained')
        rates=[('coreAE_rate_hz','左核 E'),('coreBE_rate_hz','右核 E'),('surroundE_rate_hz','核外 E'),('allI_rate_hz','全体 I')]
        for pi,(family,parent,child,title) in enumerate(present):
          for ui,(topo,seed) in enumerate(units):
            a=native_lookup.get((parent,topo,seed));b=native_lookup.get((child,topo,seed))
            if a is None or b is None:continue
            offset=(ui-(len(units)-1)/2)*min(.18,.55/max(len(units)-1,1))
            for ax,(key,label) in zip(axes.flat,rates):
                av=value(a,key);bv=value(b,key);d=bv-av
                if np.isfinite(d):ax.plot(d,pi+offset,marker=['o','s','^','D'][ui%4],color=COLORS[family],ms=5,ls='none')
                native.append(dict(family=family,contrast=title,parent=parent,child=child,topology_seed=topo,dynamics_seed=seed,group=key,parent_hz=av,child_hz=bv,delta_hz=d))
        for ax,(key,label) in zip(axes.flat,rates):
            ax.axvline(0,c='.6',lw=.8);ax.grid(axis='y',alpha=.2);ax.set_title(label)
            ax.set_xlabel('完整启动排除后，每细胞平均放电率变化 (Hz)')
            ax.set(yticks=range(len(present)),yticklabels=[p[3] for p in present])
        axes[0,0].invert_yaxis()
        identity='；'.join(f"{['圆','方','三角','菱形'][i%4]}：图 {t}、噪声 {s}" for i,(t,s) in enumerate(units))
        fig.suptitle('单项参数改变后，活动如何在网络中重新分布？\n'+identity+'\n蓝：EE；橙：阈值；绿：均值；紫：位置；洋红：两核输入相关性。使用全部原生发放，不经过电极读出或事件筛选。',fontsize=11)
        for ext in ['png','pdf']:
            path=F/f'native_group_rate_paired_changes.{ext}';fig.savefig(path,dpi=180);files.append(path.name)
        plt.close(fig)
        with (dest/'native_rate_changes.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(native[0]));w.writeheader();w.writerows(native)
    (dest/'manifest.json').write_text(json.dumps(dict(phase=phase,source=str(source),paired_points=len(records),
        statistical_unit='one complete candidate x topology x dynamics replay; descriptive differences, no significance test',
        random_input='EE and threshold contrasts preserve per-step input where verified by paired_input_streams. Geometry, mean and OU-correlation changes share seeds but not input realizations.',
        no_claim='Positive participation difference is not automatically improvement; patient-reference errors and conditional full distributions are separate.',files=files),indent=2))
    (F/'README.md').write_text('\n\n'.join(f'### {name}\n\n同一拓扑、同一噪声种子下，将明确的一项参数改变前后作差；颜色区分EE、阈值、输入均值、位置和两核慢输入相关性。圆方等形状区分实际运行，事件数列出前后值，未观察到模式时对应分布指标留空；没有把事件当作独立网络，也没有显著性检验。位置、输入均值及两核慢输入相关性改变时，实际随机输入不保证相同；primary与全部检测开发集合分开。**关注点**：一个参数能否在两条重演中产生一致响应，以及招募、顺序误差与事件量是否存在取舍。' for name in files)+'\n')
    if native:
        notes=(F/'README.md').read_text()
        for ext in ['png','pdf']:
            name=f'native_group_rate_paired_changes.{ext}'
            start=notes.index('### '+name);end=notes.find('\n\n### ',start+4)
            if end<0:end=len(notes)
            notes=notes[:start]+f'### {name}\n\n每个参数前后，在同一图和噪声种子下计算左核E、右核E、核外E和全体I的平均放电率差；每个细胞等权，使用完整启动排除后记录的原生发放，无电极读出或事件筛选。颜色代表参数类别，符号代表实际重演；正负只表示活动增加或减少，不是患者拟合优劣。**关注点**：局部参数是否引起非局部活动重排，以及这种响应是否跨图/噪声保留。'+notes[end:]
        (F/'README.md').write_text(notes+'\n')
    print(json.dumps(dict(output=str(dest),files=len(files))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',default='long');args=p.parse_args();main(args.phase)
