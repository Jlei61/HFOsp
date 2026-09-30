"""Spatial snapshots near certified folds, with a separate multi-fold audit."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from topic4_fig5_D_gap_focus import OUT,SOURCE
import plot_topic4_fig5_D_physical as base

def main():
    base.FIG=OUT/'figures';base.FIG.mkdir(exist_ok=True)
    base.diagram(focus=True)
    audit=json.loads((OUT/'fold_spatial_audit.json').read_text())
    selected=[audit['folds'][k-1] for k in (1,5,7)]
    fig,axes=plt.subplots(2,3,figsize=(10,7.5))
    fig.subplots_adjust(left=.07,right=.84,bottom=.09,top=.87,hspace=.52,wspace=.30)
    for column,row in enumerate(selected):
        im=base.map_panel(axes[0,column],row['cell_rate_hz'],PowerNorm(.6,0,8),
             f"SN{row['number']}\nD = {row['D']:.6f}",showy=column==0)
        mode=base.map_panel(axes[1,column],100*np.array(row['mode_energy_per_cell']),PowerNorm(.5,0,35),
             f"SN{row['number']}",showy=column==0)
    axes[0,0].text(-.29,1.16,'A',transform=axes[0,0].transAxes,fontsize=18,weight='bold')
    axes[1,0].text(-.29,1.09,'B',transform=axes[1,0].transAxes,fontsize=18,weight='bold')
    fig.colorbar(im,cax=fig.add_axes([.89,.57,.018,.27]),label='Equilibrium E rate (Hz)',ticks=[0,2,4,8])
    fig.colorbar(mode,cax=fig.add_axes([.89,.11,.018,.27]),label='Critical mode energy / cell (%)',ticks=[0,10,20,35])
    base.save(fig,'fig_multiple_folds_spatial')
    (OUT/'figures/README.md').write_text('''### fig_D_fold_focus_spatial.png / .pdf / .svg
沿用 q_IE=1.25 的已验证平衡和局部周期支，横轴收窄为 D=0–0.56，删除重复的 B 周期空间快照。③改为 D=0.250220 的中率平衡鞍结，④改为高率支已采样且判稳的最小 D=0.507124 的平衡解；放大窗展示高率支边缘的折返。
两侧分支仍未连接，空白不代表状态必然跳跃；③、④均为平衡解，④没有被标成全局振荡起点。**关注点**：空间快照与各自分支解一一对应，原来 D=1 的远端状态已移除；待用户人工验图。

### fig_multiple_folds_spatial.png / .pdf / .svg
对比原星号目录中的 SN1、SN5、SN7：参数 D 几乎相同，核 B 的临界扰动模式近似相同，但核 A 的平衡活动背景不同。上排为平衡放电率，下排为归一化临界模态能量；两排不是同一个量。
这解释了为什么多个空间平衡态的鞍结会在全局平均率投影上拥挤，不能理解成网络依次经历了三次全局爆发。**关注点**：核 A 背景差异与核 B 临界模式的重复；不据此把整个解集称作已证明的 snaking。
''',encoding='utf8')
    print('PLOTS COMPLETE',flush=True)

if __name__=='__main__':main()
