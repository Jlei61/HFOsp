"""Numerical diagnostic only: a local response kink is not a network fold."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/topic4_sef_hfo/kinetic_population_bifurcation_20260916'


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.labelsize':16,
        'xtick.labelsize':13,'ytick.labelsize':13,'svg.fonttype':'none','pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.8));rows=[]
    styles=[(6,.125,'#c63e32','-'),(6,.0625,'#c63e32','--'),(8,.125,'#287a9e','-'),
            (10,.125,'#26836d','-'),(12,.125,'#8962a5','-')]
    for degree,dv,color,style in styles:
        folder=OUT/'local_threshold_resolution'/f'g395_degree{degree}_dv{dv:g}'
        status=json.loads((folder/'status.json').read_text());assert status['status']=='COMPLETE'
        z=np.load(folder/'curve.npz');u=z['current_mv'];r=z['rate_hz'];slope=np.diff(r)/np.diff(u)
        label=f'{degree}, '+r'$\Delta V$'+f' = {dv:g} mV'
        axes[0].plot(u,r,color=color,ls=style,lw=2,label=label)
        axes[1].plot((u[:-1]+u[1:])/2,slope,color=color,ls=style,lw=2)
        rows.append(dict(degree=degree,dv=dv,central_rate_hz=float(r[20]),
            slope_range=[float(slope.min()),float(slope.max())],
            node_crossings=status['candidate_node_crossings'],source=str(folder)))
    cross=rows[0]['node_crossings'][0]
    for ax,letter in zip(axes,['A','B']):
        ax.axvline(cross,color='black',lw=.8,ls=':')
        ax.set_xlabel('Constant recurrent current (mV)')
        ax.set_xlim(u[0],u[-1]);ax.set_xticks([.83,.85,.87,.89])
        ax.text(0.,1.04,letter,transform=ax.transAxes,fontweight='bold',fontsize=20)
    axes[0].set_ylabel('Stationary E rate (Hz / neuron)')
    axes[1].set_ylabel('Local response slope (Hz / mV)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,fontsize=12)
    fig.subplots_adjust(left=.085,right=.98,top=.90,bottom=.25,wspace=.30)
    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    for suffix in ['png','pdf','svg']:fig.savefig(dest/f'fig_local_threshold_resolution.{suffix}',dpi=220)
    plt.close(fig)
    report=dict(local_group=395,rows=rows,degree6_threshold_node_current=cross,
        finding='The degree-6 slope corner survives voltage-grid refinement but disappears at this current with higher noise degree.',
        implication='The nearby network turning candidate cannot yet be typed as a smooth saddle-node without noise-basis and critical-mode refinement.',
        scope='A local discretization diagnostic, not the requested network bifurcation diagram')
    (OUT/'local_threshold_resolution_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (dest/'README.md').write_text('### fig_local_threshold_resolution.png / .pdf / .svg\n\n'
        '同一细胞组、同一阈值和常值递归电流下，比较噪声表示阶数与电压网格的局部稳态响应；A 为放电率，B 为相邻电流点的响应斜率。'
        '黑色竖线是 degree 6 的一个噪声节点到达阈值的电流，不是网络分岔点；图例首项为噪声阶数。'
        '这是一张数值诊断图，不能替代目标分岔图。\n\n'
        '**关注点**：degree 6 的斜率折角在电压网格减半后仍存在，在提高噪声阶数后移出此处，因此附近网络转折需先通过噪声分辨率验证；本图待用户人工检查。\n')


if __name__=='__main__':main()
