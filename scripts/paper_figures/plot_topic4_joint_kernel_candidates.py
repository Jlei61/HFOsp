#!/usr/bin/env python3
"""Diagnostic view of pre-nominated event-expansion candidates; not paper Fig5."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scripts import run_topic4_joint_xy_kernel_search as run
from src.topic4_joint_xy import joint_features


def mean_rank(t,xy,groups):
    n=t.shape[1];r=joint_features(t,xy,groups)[:,n:2*n]*4*np.sqrt(n)-1
    return np.nanmean(np.where(np.isfinite(t),r,np.nan),axis=0)


def main():
    out=run.OUT;pool=run.read(out/'baseline_scores.json')['candidates'];plan=run.read(run.CONFIG)
    selected=run.select_racers(pool,0,plan);cal=run.read(out/'patient_calibration.json')
    obj=run.KernelObjective(run.v1,out,run.KERNEL)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(len(selected),3,figsize=(12,3.0*len(selected)),gridspec_kw={'width_ratios':[1,1.25,1.2]})
    fig.subplots_adjust(top=.94,bottom=.055,left=.065,right=.98,hspace=.42,wspace=.33)
    fig.suptitle('Dual-core candidates before common-seed event expansion',fontsize=17,y=.987,weight='bold')
    fig.text(.065,.963,'Fixed VTH budget and global E→E axis; no core-angle or event-direction term in the fitted loss.',fontsize=11)
    patient_rank=mean_rank(obj.patient,obj.xy,obj.groups)
    for i,row in enumerate(selected):
        a,b,c=axes[i];center=np.asarray(row['candidate']['node_field']['centers_mm']);radius=max(u['geometry']['distance_cutoff_mm'] for u in row['units'])
        for xy in center:a.add_patch(Circle(xy,radius,facecolor='#d8a5bf',edgecolor='#9d3766',alpha=.8))
        a.scatter(obj.xy[:,0],obj.xy[:,1],c='black',s=9)
        angle=np.deg2rad(-22.80538396505847);x=np.array([0,20]);a.plot(x,10+(x-10)*np.tan(angle),'--',lw=1,color='#777777')
        a.set(xlim=(0,20),ylim=(0,20),aspect='equal',xlabel='x (mm)',ylabel='y (mm)',title=row['candidate_id'].replace('control_historical_matched','Historical manual control'))
        tables=[run.v1.read_worker(Path(u['worker_path']),obj,plan)[0] for u in row['units']]
        rank=mean_rank(np.concatenate(tables),obj.xy,obj.groups)
        b.plot(patient_rank,'o-',color='black',ms=3,label='Patient training')
        b.plot(rank,'o-',color='#398675',ms=3,label=f'Model: {row["n_events"]} events')
        b.set(ylim=(-.05,1.05),ylabel='Mean participating-contact rank',xlabel='Contact index',xticks=[0,4,8,12,14])
        b.legend(frameon=False,fontsize=8,loc='upper right')
        size=next((n for n in sorted(map(int,cal['samples'])) if n>=row['n_events']),256)
        names=['support','rank_space','timing_space','joint'];floor=cal['samples'][str(size)]['kernel_q95']
        ratio=[row['kernel_distances'][k]/floor[k] for k in names]
        c.bar(range(4),ratio,color=['#8d9eac','#916196','#ca9160','#398675'])
        c.axhline(1,color='black',ls='--',lw=1);c.set(yscale='log',xticks=range(4),xticklabels=['Participation','Rank','Timing','Joint'],ylabel='Distance / patient q95',title='Short-run diagnostics; not qualification')
        c.tick_params(axis='x',labelrotation=15)
        for j,v in enumerate(ratio):c.text(j,v*1.05,f'{v:.1f}',ha='center',va='bottom',fontsize=8)
        c.set_ylim(bottom=min(.7,min(ratio)*.8),top=max(2,max(ratio)*1.7))
        for ax in (b,c):ax.spines[['top','right']].set_visible(False)
    fig.text(.065,.013,'Each geometry initially has two 8 s networks. At least 64 actual events are required before independent confirmation.\n'
        'Circles show the VTH cores; dashed line shows the global E→E orientation, not a localized connection corridor.\n'
        'Rank curves are descriptive averages; the fitted kernel compares complete event patterns. Firing-density proxy is not validated HFO/LFP.',fontsize=9)
    folder=out/'figures';folder.mkdir(exist_ok=True)
    for ext in ('png','pdf','svg'):fig.savefig(folder/f'candidate_event_expansion_review.{ext}',dpi=160)
    plt.close(fig)
    (folder/'README.md').write_text('''### candidate_event_expansion_review.png / .pdf / .svg
展示按预设规则提名补算的六个候选：左列是双核位置与固定电极，虚线只表示整个网络 E→E 连接的共同方向，并不是中央局部连接带；中列比较参与通道的平均 rank，右列比较完整事件核各分量与患者训练波动阈值的比例。当前每个候选只有两个 8 秒网络，不能据此冻结双核。损失使用完整事件分布，中列平均曲线只是诊断摘要，不能证明两种模式已经出现。
**关注点**：位置是否不同、rank／时滞差异是否仍明显，以及额外共同 seed 后结论是否保持；比例超过 1 表示尚未落入相应开发阈值。
''')
    paths=[out/'baseline_scores.json',out/'patient_calibration.json',run.CONFIG,Path(__file__)]
    write=lambda p,x:p.write_text(json.dumps(x,indent=2)+'\n')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    write(out/'candidate_review_metadata.json',{'input_hashes':{str(p):sha(p) for p in paths},'candidate_ids':[r['candidate_id'] for r in selected],
        'output_hashes':{p.name:sha(p) for p in folder.glob('candidate_event_expansion_review.*')},'author_acceptance':False})


if __name__=='__main__':main()
