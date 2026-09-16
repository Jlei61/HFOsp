"""One-axis bifurcation figure and matched A/B observations from joint runs.

Reuses existing equilibria, periodic solutions and native trajectories. No
new simulations, parameter fits or bifurcation calculations are performed.
"""
import json, csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import plot as p


def draw_single(ax, rows, cyc, fold, arc):
    p.branches(ax,cyc,fold,arc)
    for row in rows:
        p.number_marker(ax,row['g'],row['mean'],row['number'],row['color'],size=270)
    ax.legend(handles=[
        Line2D([],[],color=p.BLUE,label='Stable equilibrium'),
        Line2D([],[],color=p.RED,ls='--',label='Unstable equilibrium'),
        Line2D([],[],color=p.GREEN,label='Burst maximum / minimum'),
        Line2D([],[],color=p.ORANGE,label='Burst time mean'),
        Line2D([],[],color='black',marker='o',ls='none',label='Native SNN mean (1–4)')],
        frameon=False,loc='upper left',fontsize=11,labelspacing=.5)
    ax.set_xlim(.46,1.23)
    # Direct labels identify the three solved periodic observables on the
    # same rate axis. No inset axes or remapped parameter scale is used.
    end=cyc[-1]
    for y,lab,color in [(end['hi'][0],'Max',p.GREEN),(end['mean'][0],'Mean',p.ORANGE),(end['lo'][0],'Min',p.GREEN)]:
        ax.annotate(lab,xy=(end['g'],y),xytext=(7,4 if lab=='Min' else 0),
                    textcoords='offset points',ha='left',va='bottom' if lab=='Min' else 'center',fontsize=11,color=color)
    ax.set_title('Core burst onset',loc='left',fontweight='bold',fontsize=19,pad=14)


def bifurcation_figures(rows,cyc,fold,arc,book):
    fig,ax=plt.subplots(figsize=(9.3,8.0))
    fig.subplots_adjust(left=.12,right=.96,bottom=.12,top=.91)
    draw_single(ax,rows,cyc,fold,arc)
    assert len(fig.axes)==1 and not ax.child_axes
    p.save(fig,'00_core_bifurcation_single_axis',
        '将上方分岔图独立导出，所有平衡支、周期峰率/均率/谷值及原生编号点共用一个坐标轴。图中没有内嵌窗，线条的计算来源与两核比较见配套说明。',
        '纵轴是联合降阶系统的Core A E率；编号点是原生SNN时间平均率，两种模型层不能混同。',book)
    fig=plt.figure(figsize=(12.8,12.3))
    ax=fig.add_axes([.085,.525,.89,.39]);draw_single(ax,rows,cyc,fold,arc)
    p.native_panel(fig,rows,box=(.085,.09,.89,.285),title_y=.439)
    p.save(fig,'01_single_axis_bifurcation_four_states',
        '保留下方四状态编号对照，将上方统一为一个完整坐标轴，去掉两个内嵌窗。四状态继续显示固定30细胞4–7秒raster，波形为各运行全部720个A核E细胞的2ms率。',
        'IEI CV使用2–20秒；各波形率轴的线性范围有明确刻度，图中不再放灰色说明小字。',book)


def paired_native():
    baseline=p.read(p.V1/'baseline_observables.json'); rows=[]
    for seed in (848101,848102):
        for g in (.5,.7,.85,1.,1.2):
            row=dict(J_EE_core=g,seed=seed,threshold_depth=1,topology=2511,analysis_start_s=2,analysis_end_s=20)
            for core,group in [('A','coreAE'),('B','coreBE')]:
                obs=next(x for x in baseline if x['seed']==seed and x['ee']==g and x['depth']==1 and x['group']==group)
                m=obs['metric']
                row.update({core+'_mean_rate_hz':m['mean_rate_hz'],core+'_n_bursts':m['n_bursts'],
                    core+'_iei_cv':m['cv'],core+'_median_peak_2ms_active_fraction':m['median_peak_active_fraction_2ms'],core+'_label':m['label']})
            path=p.NATIVE/'per_run'/f'ee{g:g}_d1_n1_t2511_s{seed}'/'trajectory.npz'
            with np.load(path) as z:
                names=z['group_names'].tolist()
                for core,group in [('A','coreAE'),('B','coreBE')]:
                    i=names.index(group);r=z['spike_counts_2ms'][1000:10000,i]/int(z['group_sizes'][i])/.002
                    assert abs(r.mean()-row[core+'_mean_rate_hz'])<1e-10
            row['same_joint_trajectory']=str(path);rows.append(row)
    (p.OUT/'paired_native_core_a_b.json').write_text(json.dumps(rows,indent=2)+'\n')
    with (p.OUT/'paired_native_core_a_b.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return rows


def simultaneous_waveforms(book):
    fig,axs=plt.subplots(3,1,figsize=(10,8.5))
    fig.subplots_adjust(left=.105,right=.96,bottom=.095,top=.855,hspace=.50)
    for ax,g,title in zip(axs,(.7,.85,1.),('Irregular bursts','Intermediate','Regular bursts')):
        path=p.NATIVE/'per_run'/f'ee{g:g}_d1_n1_t2511_s848101'/'trajectory.npz'
        with np.load(path) as z:
            names=z['group_names'].tolist(); t=(np.arange(len(z['spike_counts_2ms']))+.5)*.002
            sel=(t>=4)&(t<7)
            for group,color,ls,label in [('coreAE',p.BLUE,'-','Core A E'),('coreBE','#8059a3','--','Core B E')]:
                i=names.index(group);rate=z['spike_counts_2ms'][:,i]/int(z['group_sizes'][i])/.002
                ax.plot(t[sel],rate[sel],color=color,ls=ls,lw=1.4,label=label)
        ax.set(xlim=(4,7),ylim=(0,45 if g==.7 else 440),xlabel='Time (s)',ylabel='Rate (Hz / E cell)')
        ax.set_title(f'{title}   |   {p.JLABEL} = {g:g}',loc='left',fontsize=14)
    fig.text(.105,.955,'Core A and B in the same native SNN run',fontsize=19,weight='bold')
    fig.legend(*axs[0].get_legend_handles_labels(),loc='upper right',bbox_to_anchor=(.965,.928),ncol=2,frameon=False,fontsize=12)
    p.save(fig,'05_native_core_a_b_simultaneous',
        '两核曲线来自同一原生40000细胞网络、同一运行文件及相同4–7秒窗口；固定拓扑2511、阈值幅度1、噪声848101。每核分别按自身E细胞数720/742归一化，未对波形移相或跨运行拼接。',
        '不同条件下的招募幅度和时序可分别变化；率相近不能单独证明同步或固定相位差。',book)


def explain_lines_and_cores(paired,cyc,fold,arc):
    graph=np.load(p.SOURCE/'projected_graph.npz');W=graph['W'].sum(0)
    specs=[]
    for i in (0,1):
        thresholds=graph['vtheta'][graph['region']==i]
        specs.append(dict(core='AB'[i],n_E=int(graph['count'][i]),n_I=int(graph['count'][i+3]),
            E_threshold_mean_mV=float(thresholds.mean()),E_threshold_sd_mV=float(thresholds.std()),
            E_threshold_min_mV=float(thresholds.min()),within_core_EE_weight_sum_per_target=float(W[i,i])))
    assert np.all(W[np.ix_([0,3],[1,4])]==0) and np.all(W[np.ix_([1,4],[0,3])]==0)
    comparisons=[]
    for c in cyc:
        r=c['r'];T=c['T'];lag=((np.argmax(r[:,1])-np.argmax(r[:,0]))/len(r)*T+T/2)%T-T/2
        comparisons.append(dict(J_EE_core=c['g'],period_ms=T,A_mean_hz=float(c['mean'][0]),B_mean_hz=float(c['mean'][1]),
            A_peak_hz=float(c['hi'][0]),B_peak_hz=float(c['hi'][1]),B_minus_A_peak_lag_ms=float(lag),source=c['path']))
    metadata=dict(model_layer='Six-population deterministic delay-rate closure, jointly solved',
        displayed_coordinate='Core A E rate',fixed_point_continuation='pseudo-arclength',
        n_equilibrium_source_points=len(arc),n_visible_equilibrium_points=sum(.46<=x['g']<=1.23 for x in arc),n_periodic_solutions=len(cyc),
        periodic_solver='free-period Fourier collocation with a phase condition',
        stability='full-delay characteristic spectrum for equilibria; Floquet multipliers for periodic orbits',
        line_drawing='straight segments between numerically solved points; no curve fitting to native examples',
        graph_groups=specs,direct_A_B_weights_zero=True,periodic_core_comparisons=comparisons,
        physical_simulations_added=0)
    (p.OUT/'line_construction_and_core_comparison.json').write_text(json.dumps(metadata,indent=2)+'\n')
    tab=[]
    for row in paired[:5]:
        cvA='—' if row['A_iei_cv'] is None else f'{row["A_iei_cv"]:.3f}'
        cvB='—' if row['B_iei_cv'] is None else f'{row["B_iei_cv"]:.3f}'
        tab.append(f'| {row["J_EE_core"]:g} | {row["A_mean_rate_hz"]:.3f} | {row["B_mean_rate_hz"]:.3f} | {row["A_n_bursts"]} / {row["B_n_bursts"]} | {cvA} / {cvB} |')
    peri=next(x for x in comparisons if x['J_EE_core']==1.15)
    doc=f'''# 单坐标轴分岔图：曲线如何计算、两核是否不同

当前展示为[独立的单坐标轴分岔图](figures/00_core_bifurcation_single_axis.png)，所有曲线使用同一组横纵坐标，没有内嵌小图；另保留[下配四状态的版本](figures/01_single_axis_bifurcation_four_states.png)。这次只重画、核查已有数组，未新增SNN仿真或拟合。

## 这些线从哪里来

**曲线是同一个联合六群体确定性时延率系统的数值解；纵轴只取其中Core A E的坐标。** 同时求解的群体为A核E/I、B核E/I、周边E/I。真实网络的逐延迟权重和、平方权重和、阈值经验分布投影到这六群体；共享OU不进入这套确定性方程，私有Poisson通过输入矩进入。核内E→E倍率同时作用于A→A和B→B，其他连接保持原设定。

| 线/点 | 实际计算过程 | 图中纵轴是什么 |
|---|---|---|
| 蓝实线 | 联立求各群体平衡条件，并通过伪弧长延拓跟踪分支；全时延特征谱无正实部根 | 稳定平衡点的A核E率 |
| 红虚线 | 同样解平衡条件并延拓，包括时间积分无法稳定停留的不稳定解；至少有一个正实部特征根 | 不稳定平衡点的A核E率 |
| 绿色上/下边界 | 联立整条六群体周期波形、未知周期和相位条件，用Fourier配点求解；Floquet核验横向稳定 | 一周期内A核E率的最大/最小值 |
| 橙线 | 对同一条已求解周期轨道取时间平均 | A核E的一周期平均率 |
| 编号1–4 | 读取原生SNN的2–20秒spike计数，按720个A核E细胞归一化 | 原生SNN实际时间平均率 |

平衡点是解 $r_i=\\Phi_i(\\mu_i(r),V_{{E,i}}(r),V_{{I,i}}(r))$，而非对SNN的率散点做曲线拟合。平衡态稳定性来自完整时延特征方程 $\\det M(\\lambda)=0$，其中保留传播时延和突触双极点；不是把静态残差Jacobian的非零特征值当作完整动力学谱。绘图线只是已求解点之间的连线。黑点{fold['g']:.8f}处，一个实特征值达到0并发生非退化saddle-node；局部正规形与周期标度、全局回返共同支持SNIC型起始。

平衡源数组包含{len(arc)}个带谱核验的延拓点，图中截取所示横轴范围；周期支包含{len(cyc)}个已求解、带Floquet核验的点。上方平衡分支的空心端点以及周期支最右端只表示当前计算范围，未标为新的分岔。编号点不参与任何曲线拟合；它们都在降阶折点左侧，显示该闭合还未准确预测原SNN的状态边界。

## 联合降阶系统里A、B的区别

此图没有把A/B分开独立求解。低率折点处A核E率为{fold['r_hz'][0]:.6f} Hz，B核E为{fold['r_hz'][1]:.6f} Hz；临界右模态位于A核E方向，B相关慢模态仍衰减。因此“最先失稳的群体”和“大burst最终涉及的群体”有区别。

在1.15的同一条周期解中，A/B共享{peri['period_ms']:.6f} ms周期，但A/B均率为{peri['A_mean_hz']:.3f}/{peri['B_mean_hz']:.3f} Hz，峰率为{peri['A_peak_hz']:.3f}/{peri['B_peak_hz']:.3f} Hz，B峰晚于A峰约{peri['B_minus_A_peak_lag_ms']:.2f} ms。这个64 ms属于该降阶周期解，不能当作原生SNN实测延迟。周期随参数变化时相位差也变化，未设定固定64 ms。

## 同一次原生SNN仿真里A、B的区别

两核确有区别，尤其在中间条件；但不是整个参数范围内一种固定“A态/B态”分工。下表每行来自同一40000细胞网络的一次20秒运行，分析2–20秒；固定拓扑2511、阈值幅度1、噪声848101。每个核按自身E细胞数归一化，事件数是同一18秒窗内计数。

| 核内EE倍率 | A均率 Hz | B均率 Hz | A/B事件数 | A/B事件间隔CV |
|---:|---:|---:|---:|---:|
{chr(10).join(tab)}

在0.85，A的2ms峰值参与比例中位数约62.5%，B约19.0%，差别主要表现在招募强度；另一个噪声seed848102也有A均率9.413 Hz、B均率5.235 Hz，方向一致。只有两个噪声重复，未做总体显著性推断。到1.0，两核的均率、事件数和CV接近；接近的均率或CV不能单独证明逐事件同步。[同时间波形](figures/05_native_core_a_b_simultaneous.png)直接使用同一文件4–7秒的A/B率，没有人为对齐相位。

## 为什么会不同

两个核沿用相同的参数规则，但实际网络不是严格镜像。A/B分别有720/742个E、197/200个I细胞；E阈值经验均值约17.264/17.285 mV，标准差约0.740/0.718 mV，实际局部和周边连接也不同。直接A↔B连接权重为0，耦合经过周边E/I路径。核内有限细胞数、连接与阈值的实现差别，以及随机输入共同提供产生差异的可能来源；本次未通过交换阈值、连接或噪声的消融把原因单独分离。

因此当前可回答“同时计算/仿真时两核会不会不同”：会，中间区间的招募差异明显，规则区间统计量更接近。尚不能回答“究竟哪一项异质性导致A先失稳或burst更强”，也不能据此把A/B分别命名为间期核或发作核。

## 数值与文件

- [逐seed的配对原生统计CSV](paired_native_core_a_b.csv)，统计对象是同一次运行内两核，非独立网络。
- [曲线来源及两核周期数值](line_construction_and_core_comparison.json)。
- 方程与参数：[v2 model.py](../../../scripts/topic4_core_bifurcation_v2/model.py)；平衡延拓：[branches.py](../../../scripts/topic4_core_bifurcation_v2/branches.py)；周期求解：[periodic.py](../../../scripts/topic4_core_bifurcation_v2/periodic.py)；完整数学说明见[v2报告](../core_burst_bifurcation_v2_20260915/scientific_report.md)。
- [本次三页图册](figures/single_axis_and_core_comparison.pdf)。图形自查后仍待用户目视检查。
'''
    (p.OUT/'line_construction_and_core_comparison.md').write_text(doc)


def main():
    p.MANIFEST.clear();p.DESCRIPTIONS.clear()
    rows=p.load_native();cyc=p.load_cycles();fold=p.read(p.SOURCE/'fold.json');arc=p.read(p.SOURCE/'equilibrium_spectrum.json')
    paired=paired_native()
    with PdfPages(p.FIG/'single_axis_and_core_comparison.pdf') as book:
        bifurcation_figures(rows,cyc,fold,arc,book)
        simultaneous_waveforms(book)
    explain_lines_and_cores(paired,cyc,fold,arc)
    (p.OUT/'single_axis_figure_manifest.json').write_text(json.dumps(p.MANIFEST,indent=2)+'\n')
    readme=p.FIG/'README.md';text=readme.read_text();key='## 单坐标轴与两核联合动力学修订'
    if key in text:text=text.split(key)[0].rstrip()+'\n'
    text+='\n'+key+'\n\n'+'\n'.join(p.DESCRIPTIONS)
    text+='\n### single_axis_and_core_comparison.pdf\n本次三页图册，依次为独立分岔图、带四状态的单坐标轴分岔图及原生两核同时间波形。曲线来源、输入条件、两核比较及解释边界见上级目录line_construction_and_core_comparison.md。\n**关注点**：当前上方分岔图只有一个坐标轴；没有内嵌图和灰色说明小字。\n'
    readme.write_text(text)
    print('COMPLETE',json.dumps(p.MANIFEST),flush=True)


if __name__=='__main__':
    main()
