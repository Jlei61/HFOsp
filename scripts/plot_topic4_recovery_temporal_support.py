"""Time support diagnostics: fixed six-second bins, no sequence score or gate."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('/data/hfosp/topic4_sef_hfo/core_propagation_recovery_20260911')

def main(phase):
    src=OUT/('analysis_'+phase)
    with (src/'event_timing.csv').open() as f:events=list(csv.DictReader(f))
    with (src/'counts.csv').open() as f:counts=list(csv.DictReader(f))
    spec=json.loads((OUT/f'{phase}_units.json').read_text())
    allowed={(cid,str(t),str(s)) for cid,t,s in spec['units']}
    dest=OUT/('temporal_support_'+phase);F=dest/'figures';F.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':8,'pdf.fonttype':42})
    records=[];files=[]
    for c in counts:
      if (c['base_id'],c['topology_seed'],c['seed']) not in allowed:continue
      selected=[e for e in events if e['candidate']==c['candidate'] and e['seed']==c['seed']]
      end=float(c['duration_ms']);edges=np.r_[np.arange(1500,end,6000),end]
      fig,axes=plt.subplots(2,2,figsize=(12,6),layout='constrained',sharex=True)
      for li,layer in enumerate(['primary','all_detected']):
        ev=[e for e in selected if layer=='all_detected' or e['primary']=='True']
        for mode,color in [('TA','#c9473f'),('TB','#377dab')]:
          vals=[];frac=[]
          for bi,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
            window=[e for e in ev if lo<=float(e['event_time_ms'])<hi];mm=[e for e in window if e['mode']==mode]
            times=np.array(sorted(float(e['event_time_ms']) for e in mm));gaps=np.diff(times)
            rec=dict(candidate=c['candidate'],base_id=c['base_id'],topology_seed=c['topology_seed'],seed=c['seed'],layer=layer,mode=mode,
                start_ms=float(lo),end_ms=float(hi),exposure_s=float((hi-lo)/1000),partial_bin=bool(hi-lo<6000),n_events=len(mm),all_mode_events=len(window),
                proportion=len(mm)/len(window) if window else None,rate_per_s=len(mm)/((hi-lo)/1000),
                within_bin_same_mode_interval_median_ms=float(np.median(gaps)) if len(gaps) else None)
            records.append(rec);vals.append(rec['rate_per_s']);frac.append(rec['proportion'] if rec['proportion'] is not None else np.nan)
          xx=(edges[:-1]+edges[1:])/2000
          axes[li,0].plot(xx,vals,'o-',color=color,label=mode);axes[li,1].plot(xx,frac,'o-',color=color,label=mode)
        axes[li,0].set_ylabel(('原 primary' if layer=='primary' else '全检测开发层')+'\n事件数 / 实际秒数')
        axes[li,1].set_ylabel('该时间段内的模式占比');axes[li,1].set_ylim(-.03,1.03)
        for ax in axes[li]:ax.grid(alpha=.2);ax.legend()
      for ax in axes[-1]:ax.set_xlabel('记录时间 (s)')
      fig.suptitle(f"{c['display_name']}｜图 {c['topology_seed']}、噪声 {c['seed']}\n从 1.5 s 启动排除后每 6 s 分段；末段按实际时长计率。标签仅组织比较，不代表传播恢复。")
      stem=f"{c['base_id']}_{c['topology_seed']}_{c['seed']}"
      for ext in ['png','pdf']:
          path=F/f'{stem}.{ext}';fig.savefig(path,dpi=180);files.append(path.name)
      plt.close(fig)
    if records:
      with (dest/'temporal_support.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    (dest/'manifest.json').write_text(json.dumps(dict(phase=phase,source=str(src),bin_ms=6000,burnin_ms=1500,
        interval='Within-bin intervals between adjacent observed events of the same label, not a complete waiting-time distribution; no cross-bin intervals.',
        interpretation='Descriptive temporal support only; no new sequence loss, no hard requirement that both modes occur in every bin.',files=files),indent=2))
    (F/'README.md').write_text('\n\n'.join(f'### {name}\n\n从启动排除结束的1.5秒开始按连续6秒分段，最后不足6秒的区间使用实际时长计算率；红为TA，蓝为TB。原primary与全检测开发层分别显示，模式缺席的区间计数为零，无事件区间的占比留空。**关注点**：两类标签是否只来自记录开头，以及参数影响是否随时间改变；这不等于恢复模式切换动力学或两条患者路径。' for name in files)+'\n')
    print(json.dumps(dict(output=str(dest),figures=len(files))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',default='long');a=p.parse_args();main(a.phase)
