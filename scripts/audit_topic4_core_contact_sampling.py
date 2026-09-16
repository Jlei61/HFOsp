"""Audit geometric readout weights, not activity fractions or causal origins."""
from pathlib import Path
from types import SimpleNamespace
import sys, json, inspect
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.patches import Circle,Ellipse
from src.topic4_streaming_spike_readout import StreamingSpikes
from src.topic4_pdf_font_guard import install
from scripts import analyze_topic4_shape_output_response as an

BASE=Path('/data/hfosp/topic4_sef_hfo/core_recruitment_tradeoff_followup_20260912')
OUT=Path('/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913/core_contact_sampling')
CASES=[('up3__circle__EE_core_to_out_scale_1.25__x_minus075','左移圆核＋向外EE增强25%'),
       ('up3__ellipse4__EI_same_core_scale_0.75__x_minus075','左移椭圆＋核内EI减弱25%')]

def main():
    (OUT/'figures').mkdir(parents=True,exist_ok=True);install()
    an.plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'pdf.fonttype':3})
    fig,axes=an.plt.subplots(2,2,figsize=(14,12),gridspec_kw={'width_ratios':[1.2,1]})
    fig.subplots_adjust(left=.055,right=.97,bottom=.17,top=.88,hspace=.32,wspace=.3)
    rows=[];sources=[];checks=[];colors=['#bd7939','#327fb0','#d9d9d9']
    sigma=inspect.signature(StreamingSpikes).parameters['kernel_width'].default
    for row,(cid,title) in enumerate(CASES):
        p=BASE/'response/units'/cid/'2511_847101/workers/trajectory.json'
        r=an.rt.read(p);assert an.rt.sha(p.with_suffix('.npz'))==r['arrays_sha256']
        ap=an.rt.read(p.parents[1]/'applied_physics.json');c=an.rt.read(BASE/'candidates'/f'{cid}.json')
        with np.load(p.with_suffix('.npz')) as a:
            pos=a['positions_E'];group=a['core_index_E'];xy=a['contact_xy_mm'];names=list(a['contact_names'])
            assert np.array_equal(np.flatnonzero(group==0),a['group_coreAE'])
            assert np.array_equal(np.flatnonzero(group==1),a['group_coreBE'])
        # Call the actual recorder constructor, using its existing spatial-kernel
        # default. No simulation is run and the readout is not changed.
        recorder=StreamingSpikes(1,len(pos),pos,SimpleNamespace(contacts=xy),.05)
        w=recorder.weights;parts=np.stack([w[:,group==0].sum(1),w[:,group==1].sum(1),w[:,group<0].sum(1)],axis=1)
        assert np.allclose(w.sum(1),1,rtol=0,atol=1e-12)
        assert np.allclose(parts.sum(1),1,rtol=0,atol=1e-12)
        for i,name in enumerate(names):
            rows.append(dict(candidate=cid,topology=2511,contact=name,x_mm=xy[i,0],y_mm=xy[i,1],
                coreA_readout_weight=parts[i,0],coreB_readout_weight=parts[i,1],outside_readout_weight=parts[i,2],
                nearest_coreA_neuron_mm=np.linalg.norm(pos[group==0]-xy[i],axis=1).min(),
                nearest_coreB_neuron_mm=np.linalg.norm(pos[group==1]-xy[i],axis=1).min(),kernel_sd_mm=sigma))
        ax=axes[row,0]
        ax.scatter(pos[::12,0],pos[::12,1],s=.8,c='#ddd',rasterized=True,zorder=1)
        for k,col in enumerate(colors[:2]):ax.scatter(pos[group==k,0],pos[group==k,1],s=1.4,c=col,rasterized=True,zorder=2,label='左核E' if k==0 else '右核E')
        aa,bb=ap['threshold']['ellipse_A_semiaxes_mm'];ax.add_patch(Ellipse(c['centers_mm'][0],2*aa,2*bb,angle=ap['threshold']['ellipse_A_angle_deg'],fill=False,ec=colors[0],lw=1.2))
        ax.add_patch(Circle(c['centers_mm'][1],c['radii_mm'][1],fill=False,ec=colors[1],lw=1.2))
        for shaft in ['SCL','ICL']:
            ids=[names.index(n) for n in an.figreview.display.CONTACT_ORDER if n.startswith(shaft)]
            ax.plot(xy[ids,0],xy[ids,1],c='#555',lw=.8,zorder=3)
        for i,name in enumerate(names):
            ax.add_patch(Circle(xy[i],2*sigma,fill=False,ec='#777',ls=':',lw=.65,zorder=4))
            ax.scatter(*xy[i],s=13,fc='white',ec='black',lw=.55,zorder=5)
            ax.annotate(name,xy[i],xytext=(1,8 if name.startswith('SCL') else -11),textcoords='offset points',fontsize=6.5,zorder=6)
        ax.set(xlim=(0,20),ylim=(0,20),aspect='equal',xlabel='x (mm)',ylabel='y (mm)',title=title)
        ax.legend(fontsize=8,frameon=False,loc='upper right');ax.text(.025,.97,'点线圈：读出空间核2σ = 0.5 mm\n仅作尺度标记，无硬截断',transform=ax.transAxes,va='top',fontsize=8)
        ax=axes[row,1];ids=[names.index(n) for n in an.figreview.display.CONTACT_ORDER];left=np.zeros(len(ids))
        for k,(label,col) in enumerate(zip(['左核E','右核E','核外E'],colors)):
            values=parts[ids,k];ax.barh(range(len(ids)),values,left=left,color=col,label=label,height=.74);left+=values
        ax.axhline(3.5,c='#555',lw=.8);ax.set(yticks=range(len(ids)),yticklabels=an.figreview.display.CONTACT_ORDER,ylim=(14.7,-.7),xlim=(0,1),xlabel='该触点空间采样权重的份额',title='实际神经元分组 × 同一个固定读出核')
        ax.text(.98,.99,f'任一触点的最大左核权重：{parts[:,0].max():.2e}',transform=ax.transAxes,ha='right',va='top',fontsize=8)
        sources.append(dict(candidate=cid,trajectory=str(p),arrays_sha256=r['arrays_sha256'],applied_physics=str(p.parents[1]/'applied_physics.json'),applied_sha256=an.rt.sha(p.parents[1]/'applied_physics.json')))
        checks.append(dict(candidate=cid,E_neurons=len(pos),coreA_neurons=int((group==0).sum()),coreB_neurons=int((group==1).sum()),max_coreA_weight=float(parts[:,0].max()),weight_partition_max_error=float(np.max(abs(parts.sum(1)-1)))))
    handles,labels=axes[0,1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.75,.935),ncol=3,frameon=False)
    fig.suptitle('核先活动，不等于电极先看到核：当前几何中的采样不对称',fontsize=17,y=.975)
    fig.text(.055,.047,'这里统计固定读出核对各区域神经元的空间权重，不是事件信号、发放量或解释方差的百分比。基础拓扑2511，静态几何不随这两次噪声改变。\n读出为每触点归一化的高斯采样（σ=0.25mm）；左核几乎没有直接触点覆盖，右核覆盖ICL1–3。左核活动要被看见，通常需要先招募核外组织。\n两个组合还改变形状与连接设置，这张图只解释采样几何，不比较形状的单独因果效应。没有更改读出宽度、事件定义或loss。\n这提供核时序与接触标签不完全对应的解释线索；仍须原生场判断连续传播和双前沿叠加，不能由几何图断言患者真实起源。',fontsize=10)
    for ext in ['png','pdf']:fig.savefig(OUT/'figures'/('core_contact_sampling.'+ext),dpi=160)
    an.plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT/'core_contact_sampling.csv',index=False)
    an.rt.write(OUT/'provenance.json',dict(scope='geometric readout weights, not observed signal fractions',kernel_sd_mm=sigma,implementation=str(ROOT/'src/topic4_streaming_spike_readout.py'),implementation_sha256=an.rt.sha(ROOT/'src/topic4_streaming_spike_readout.py'),producer_sha256=an.rt.sha(Path(__file__)),sources=sources,checks=checks))
    (OUT/'figures/README.md').write_text('### core_contact_sampling.png\n左侧保留20mm平面、实际E核心成员及SEEG布局，点线圆为读出核2σ而非硬采样边界。右侧固定按杆排列，展示各触点归一化空间权重来自左核、右核及核外E的份额；PDF为同图。\n**关注点**：该份额不是实际事件发放或信号贡献；核时序与接触时序之间存在几何与传播环节。\n')
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
