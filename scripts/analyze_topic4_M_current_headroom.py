#!/usr/bin/env python3
"""Frozen-current diagnostic at an actual high state, not a bifurcation diagram."""
from pathlib import Path
import pickle
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Circle

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/topic4_sef_hfo/fig5_m_overnight_exploration_20260913/M_current_headroom'
PARENT=ROOT/'results/topic4_sef_hfo/reset_state_diagnosis_20260911/parents/high_parent.pkl'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with PARENT.open('rb') as stream:parent=pickle.load(stream)
    state=parent['engine'];assert state['step']==755000
    with np.load(ROOT/'results/topic4_sef_hfo/historical_manual_hard_native_z_v1/substrate.npz') as a:
        xy=a['positions_e'];vth=a['vtheta'][:32000];centers=a['centers_mm']
    protocol=json.loads((ROOT/'results/topic4_sef_hfo/m_parameter_modes_fig5_20260913/protocol.json').read_text())
    assert parent['identity']==protocol['identity'] and xy.shape==(32000,2)
    z=state['slow']['z'][:32000];m=state['slow']['m'][:32000]
    pre_m=state['I_E'][:32000]-z*state['I_I'][:32000]
    bins=np.clip(np.floor(xy).astype(int),0,19);cell=bins[:,1]*20+bins[:,0];n=np.bincount(cell,minlength=400)
    plt.rcParams.update({'font.size':14,'axes.labelsize':16,'axes.titlesize':16,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),sharex=True,sharey=True)
    rows=[];fields=[]
    for ax,eta in zip(axes,[.02,.2,2.]):
        current=pre_m-eta*m;headroom=current-vth
        field=np.bincount(cell,weights=headroom,minlength=400)/n
        fields.append(field)
        im=ax.imshow(field.reshape(20,20),origin='lower',extent=[0,20,0,20],
            cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-1000,vcenter=0,vmax=1000),interpolation='nearest')
        for center in centers:ax.add_patch(Circle(center,1.5,fc='none',ec='#2cc9bb',lw=1.5))
        fraction=float(np.mean(headroom>0))
        ax.set(xlabel='x (mm)',xticks=[0,10,20],yticks=[0,10,20],
            title=f'ηM = {eta:g}\n{fraction:.1%} of E cells: net drive > Vth')
        rows.append(dict(eta_M=eta,mean_net_drive=float(current.mean()),mean_M_current=float((eta*m).mean()),
            fraction_E_net_drive_above_Vth=fraction,
            fraction_E_net_drive_above_current_V=float(np.mean(current>state['V'][:32000])),
            net_drive_quantiles=np.quantile(current,[.1,.5,.9]).tolist()))
    axes[0].set_ylabel('y (mm)')
    fig.subplots_adjust(left=.065,right=.88,bottom=.17,top=.84,wspace=.18)
    cax=fig.add_axes([.9,.18,.02,.63]);fig.colorbar(im,cax=cax).set_label('Net drive − Vth (mV equiv.)')
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'frozen_current_M_effect.png',dpi=170,bbox_inches='tight');fig.savefig(folder/'frozen_current_M_effect.pdf',bbox_inches='tight');plt.close(fig)
    result=dict(source=str(PARENT),time_s=75.5,mean_Z=float(z.mean()),mean_M=float(m.mean()),records=rows,
        observable='I_AMPA_i − Z_i I_GABA_i − eta_M M_i − Vth_i at one frozen actual checkpoint; per-cell values averaged into1mm spatial bins.',
        classification='Instantaneous frozen-current margin relative to voltage threshold, not a spike prediction, fixed point, or network bifurcation.',
        omitted_dynamics='No incoming synaptic arrivals, OU/Poisson update, refractory gating, or recurrent feedback is evolved in this algebraic diagnostic.',
        statistical_unit='One actual high-state realization; cells and map bins are descriptive components, not independent experiments.',
        native_constant_parameter_return_claim=False,agent_visual_review='PENDING',human_review='PENDING')
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    np.savez_compressed(OUT/'frozen_current_fields.npz',eta_M=[.02,.2,2.],fields=np.asarray(fields),neuron_counts=n)
    (folder/'README.md').write_text('### frozen_current_M_effect.png / .pdf\n'
        '使用同一个75.5秒真实高态的逐细胞AMPA、GABA、Z、M及阈值，仅代数改变ηM，显示净电流相对阈值的空间分布。'
        '三幅均为固定瞬时状态的计算，真正放电是否退出须由完整续跑判断；未模拟新的网格。\n'
        '**关注点**：这是电流余量诊断，不是固定点、分岔或未来放电预测；蓝色表示瞬时净电流低于阈值。\n')
    text=['# 高态下M电流为何需要足够大','',
        '实际75.5秒状态的平均Z=0.55483、平均M=457.283。保持这些状态不变，当前已储存的M在ηM=0.02时只贡献约9.15mV等效电流，ηM=0.2时约91.46，ηM=2时约914.57。',
        '当回返兴奋已经非常强时，增加十倍适应仍未将多数细胞的瞬时净电流压到阈值以下；增加百倍则改变了绝大多数细胞的电流方向。此处只是固定实际电流/状态的代数诊断，随后突触、膜、延迟、噪声与Z/M演化需以真正SNN续跑确认。','',
        '| ηM | 平均净驱动 | 净驱动超过Vth的E细胞比例 |', '|---|---|---|']
    for r in rows:text.append(f'| {r["eta_M"]} | {r["mean_net_drive"]:.3f} | {r["fraction_E_net_drive_above_Vth"]:.4%} |')
    text+=['','这些细胞比例不能作为独立实验样本数，也不是网络稳定性的判定；没有给出固定点或特征值，更不能由这三张图命名Hopf。强M使初始进入困难的可能性仍需要原恒参数扫描与完整轨迹验证。']
    (OUT/'scientific_review.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(rows))


if __name__=='__main__':main()
