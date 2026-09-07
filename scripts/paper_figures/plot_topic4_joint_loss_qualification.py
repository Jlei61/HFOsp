#!/usr/bin/env python3
"""Export the training-only loss responsiveness comparison."""
from pathlib import Path
import json
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/topic4_sef_hfo/joint_rank_space_dual_core_search'


def main():
    old=OUT/'loss_power_audit/summary.json';new=OUT/'kernel_qualification/summary.json'
    a=json.loads(old.read_text())['rows'];b=json.loads(new.read_text())['rows']
    plt.rcParams.update({'font.size':11,'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(11.5,5.8))
    fig.subplots_adjust(left=.08,right=.98,top=.71,bottom=.26,wspace=.29)
    fig.suptitle('Can the loss detect disrupted rank and timing patterns?',fontsize=17,weight='bold',y=.96)
    fig.text(.08,.865,'Patient training events only; participation masks and event counts held fixed.',fontsize=11)
    for ax,perturb,title,component in zip(axes,['rank_shuffle','time_stretch_1_5'],
            ['A  Shuffle rank within each event','B  Stretch within-event times by 1.5×'],['rank_space','timing_space']):
        for rows,metric,label,color in [(a,'joint_sw','Current joint projection loss','#555555'),
            (b,'joint','Prospective whole-event kernel','#328366'),
            (b,component,'Prospective component kernel','#9259a1')]:
            values=[next(r['fraction_detected_at_empirical_q95'] for r in rows
                         if r['n']==n and r['metric']==metric and r['perturbation']==perturb) for n in (16,64,256)]
            ax.plot(range(3),values,'o-',color=color,label=label,lw=1.8,ms=5)
        ax.set(title=title,xticks=range(3),xticklabels=['16','64','256'],ylim=(-.03,1.08),
               xlabel='Pseudo-model events',ylabel='Fraction exceeding patient q95')
        ax.spines[['top','right']].set_visible(False)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,fontsize=9,loc='upper center',bbox_to_anchor=(.52,.83),ncol=3)
    fig.text(.08,.06,'48 training-block resamples per setting; q95 estimated from the corresponding unperturbed draws.\n'
             'These are descriptive sensitivity checks, not independent test-power estimates or model-fit results.\n'
             'The prospective kernel uses complete event patterns; it has no target direction or core-angle term.',fontsize=10,linespacing=1.5)
    folder=OUT/'kernel_qualification/figures';folder.mkdir(exist_ok=True)
    for ext in ('png','pdf','svg'):fig.savefig(folder/f'joint_loss_responsiveness.{ext}',dpi=180)
    plt.close(fig)
    (folder/'README.md').write_text('''### joint_loss_responsiveness.png / .pdf / .svg
用患者训练事件构造保持参与掩码和事件数不变的 rank 打乱、时滞拉伸对照，比较当前联合投影损失与候选事件核距离的响应。每个点来自 48 次训练块重采样，阈值来自对应的未扰动样本；这不是独立统计功效估计，也不是模型已经拟合患者的证据。图中不使用 core 位置或指定的传播角度。
**关注点**：候选损失在约 64 个事件时能够识别 rank 和时滞破坏；仅约 16 个事件仍明显不足，下一轮需要对有潜力的几何补足观测。
''')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (folder.parent/'figure_metadata.json').write_text(json.dumps({'inputs':{str(p):sha(p) for p in [old,new,Path(__file__)]},
        'outputs':{p.name:sha(p) for p in folder.iterdir() if p.suffix in ('.png','.pdf','.svg')},
        'author_acceptance':False},indent=2)+'\n')


if __name__=='__main__':main()
