"""Report the new layout, numerical extension and honest native correspondence."""
from extend import ROOT, OUT, SOURCE
import json
import numpy as np
from PIL import Image

def read(p):
    return json.loads(p.read_text())

def main():
    fold=read(SOURCE/'fold.json')
    states=read(OUT/'numbered_native_states.json')
    cycles=read(OUT/'displayed_periodic_branch.json')
    extension=read(OUT/'extended_periodic_branch.json')
    rel=read(OUT/'core_relationship.json')
    base=read(ROOT/'results/topic4_sef_hfo/core_burst_onset_brunel_v1_20260915/baseline_observables.json')
    lines=[]
    state_checks=[]
    for st in states:
        g=st['J_EE_core']; cv='—' if st['cv'] is None else f'{st["cv"]:.3f}'
        row=next(x for x in base if x['ee']==g and x['depth']==1 and x['seed']==848101 and x['group']=='coreAE')
        ps=np.array([e['peak_active_fraction_10ms'] for e in row['events']])
        quantile=np.quantile(ps,[.1,.5,.9]).tolist() if len(ps) else []
        recruit='—' if not len(ps) else f'{100*quantile[0]:.1f}–{100*quantile[2]:.1f}%'
        lines.append(f'| {st["number"]} | {st["name"]} | {g:g} | {st["mean"]:.3f} | {st["n_bursts"]} | {cv} | {recruit} |')
        state_checks.append(dict(number=st['number'],J_EE_core=g,peak_10ms_active_fraction_quantiles=quantile,
                                 parameter_left_of_closure_fold=g<fold['g']))
    periodic_lines=[]
    for c in cycles:
        if c['J_EE_core']<1.15: continue
        periodic_lines.append(f'| {c["J_EE_core"]:g} | {c["period_ms"]:.6f} | {c["core_A_E_mean_hz"]:.4f} | {c["core_A_E_max_hz"]:.3f} | {c["maximum_transverse_modulus"]:.5g} |')
    maxg=max(c['J_EE_core'] for c in cycles)
    endpoint=cycles[-1]
    fchecks=[]
    for p in sorted((OUT/'floquet').glob('g1.17500000_dt*.json')):
        r=read(p); mm=np.array([complex(*v) for v in r['multipliers']]);k=np.argmin(abs(mm-1))
        fchecks.append(dict(dt_ms=r['dt_ms'],phase_error=float(abs(mm[k]-1)),transverse=float(max(abs(np.delete(mm,k))))))
    fchecks.sort(key=lambda x:x['dt_ms'],reverse=True)
    assert len(fchecks)==3, 'Endpoint time-step refinement must finish before the report'
    ratios=[fchecks[k]['phase_error']/fchecks[k+1]['phase_error'] for k in range(2)]
    assert all(3.5<x<4.5 for x in ratios), 'Neutral phase mode must show second-order convergence'
    assert all(x['transverse']<1 for x in fchecks)
    checks=dict(number_of_native_runs_added=0,number_of_extended_orbits=len(extension),
                number_of_displayed_verified_orbits=len(cycles),largest_displayed_J_EE_core=maxg,
                max_extension_residual=max(x['residual'] for x in extension),
                max_extension_offgrid_defect_hz=max(x['offgrid_defect_hz'] for x in extension),
                native_correspondence=state_checks,endpoint_floquet_refinement=fchecks,
                neutral_error_refinement_ratios=ratios,
                direct_core_to_core_edges_absent=rel['core_to_core_direct_weight_sum']==0,
                status='NUMERICAL_AND_SOURCE_CHECKS_PASSED_VISUAL_REVIEW_PENDING')
    assert all(x['parameter_left_of_closure_fold'] for x in state_checks)
    assert checks['max_extension_residual']<1e-8
    assert checks['max_extension_offgrid_defect_hz']<.001
    for item in read(OUT/'figure_manifest.json'):
        with Image.open(OUT/'figures'/f'{item["stem"]}.png') as im:
            im.load();assert list(im.size)==item['pixels']
    (OUT/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    report=f'''# Core分岔与原生四状态的编号对照 v3

本版按用户要求，把主图改为近方形，横轴统一为 $J_{{\\mathrm{{EE,core}}}}$，用简短 Fold 标记临界点；周期峰率、平均率和谷值放大窗内嵌主图，去掉灰色说明小字，下方依次配四个原生SNN状态的波形和raster。分析仍只回答core内burst起始，未进入发作分岔。

**核心判断：当前确定性降阶方程有低率 saddle-node 及稳定周期burst分支；之前的四种原生SNN状态不能作为这条分支上的四种已确证吸引子。原生规则burst也在降阶折点左侧，说明该闭合尚未准确预测原SNN的状态边界。**

## 图的读法

[近方形主图](figures/01_square_bifurcation_four_states_inset.png)上半部分保留经典分岔图：蓝实线是稳定低率平衡点，红虚线是不稳定平衡点；绿色为经过周期方程求解与Floquet核验的周期最大/最小率，橙色为周期平均率。主轴1 Hz以下线性、以上对数，主图内嵌的两幅放大图均用线性率轴。红色上支的空心端点仅表示本次绘图采用的已验证延拓区间结束，不标作新的分岔。

编号1–4的横坐标是实际原生参数，纵坐标是各运行2–20秒的全720个A核E细胞平均率；它们是**原生SNN观测点**，没有投到降阶平衡支或周期支上。下方四状态与上方编号一一对应，显示固定4–7秒窗口、固定30个已记录细胞，无响应排序。独立大图同时给4–7秒和5.0–5.6秒，以便看清raster。raster为实际保存的2ms占用标记，不是可恢复到亚毫秒的逐spike时间。

表中IEI CV与burst数使用完整2–20秒；峰值参与比例是每个事件内最大10ms活跃细胞占比，表列该运行内事件的10–90%分位数。它与2ms波形峰率、单细胞ISI irregularity是不同读出。

| 编号 | 原生状态 | $J_{{\\mathrm{{EE,core}}}}$ | 平均率 Hz | burst数 | IEI CV | 峰10ms参与比例 P10–P90 |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

固定条件：拓扑2511、阈值降低幅度1、噪声848101、Core A。Resting在此指低活动背景，仍有约0.708 Hz/细胞的发放；它不是单细胞电压reset。

## 跨过折点就变为regular bursting吗？

**在当前六群体确定性闭合中，已求解的折点右侧分支是稳定周期burst。** 折点 $J_{{\\mathrm{{EE,core}}}}={fold['g']:.11f}$，A核E率{fold['r_hz'][0]:.9f} Hz；一个实特征值达到零，两条局部分支合并。v2的周期发散、正规形前因子与临界全局回返支持SNIC型起始，局部非退化saddle-node已数值确证。临界点本身不等于有限频率的规则burst；最靠近临界的已求解轨道在1.1263，周期约2.306秒。图上没有用未经求解的插值填满折点和第一个周期点之间的间隙。

**对原生随机SNN，不能用这一个临界值宣称“跨过才regular”。** 该原生例子在1.0已经相当规则（IEI CV约0.128），而降阶低率支在该值仍稳定。即使移除共享OU、保留私有Poisson，原生0.85仍有重复burst，因此不吻合不只是主图两层噪声条件不同造成的。这里尚未证明原SNN的规则化过程对应同一SNIC，更未把它解释为发作。

原生irregular位于编号2（0.7）；编号3（0.85）是事件招募大小与时序仍变动的中间表型。从编号2到3再到4，峰10ms参与比例从局部招募变为小/大事件混合，最后几乎每次全核招募，同时IEI CV降低。这支持“群体事件逐渐规则、招募趋于一致”的描述，**尚不等于发现第三个确定性吸引子或第二次分岔**。近阈值可激发回返、有限规模噪声触发、恢复与同步招募的相互作用是待区分的机制；目前不能将其中一种写成已证明原因。

## Core A、B是什么关系？

A和B是同一张40000神经元网络中的两个空间核，分别含720和742个E细胞；不是resting/burst类别，也不是本图的两个时间阶段。$J_{{\\mathrm{{EE,core}}}}$ 同时缩放A→A和B→B的E→E权重，跨核及core–surround连接保持原设置。符号是无量纲倍率，图的更名没有改变模型物理量。

从真实连接投影检查，两核E/I群体之间的直接连接权重均为零；它们经周边E/I群体间接耦合。线性临界右模态首先集中于A核E群体，B在该低率折点仍有约−22.36 s⁻¹的稳定模态；这是局部小扰动的结论。非线性大burst可以招募另一核。在1.15的已求解周期轨道中，两核共享{rel['period_ms']:.6f} ms的完整周期，B峰相对A峰延后约{rel['B_minus_A_peak_lag_ms']:.2f} ms。相位先后本身不提供因果方向证明，见[两核关系图](figures/03_core_a_b_relationship.png)。

## 右侧分支实际补充及验证

继承v2全部方程、实际连接、阈值经验分布、368个时延箱与突触双极点；不调整阈值、噪声、率响应时间，也不拟合四状态标签。本版新增{len(extension)}个自由周期轨道求解点，已显示的稳定周期分支延伸至{maxg:g}。每个新增轨道用2048配点，同时检验4096网格缺陷；再用完整时延历史变分系统计算Floquet乘子，排除中性相位方向后确认横向稳定。

| $J_{{\\mathrm{{EE,core}}}}$ | 周期 ms | A核E平均率 Hz | A核E峰率 Hz | 最大横向Floquet模 |
|---:|---:|---:|---:|---:|
{chr(10).join(periodic_lines)}

新增轨道最大归一化残差{checks['max_extension_residual']:.3g}，双倍网格最大率等价缺陷{checks['max_extension_offgrid_defect_hz']:.3g} Hz。峰率在细网格上求取，均率直接由周期Fourier零频分量得到；与初值时间积分的粗采样峰率区分。图示两端范围及稳定性均来自真实求解结果。

在新增端点1.175处，将变分积分步长从约0.1降为0.05、0.025 ms，中性相位乘子误差依次为{fchecks[0]['phase_error']:.5g}、{fchecks[1]['phase_error']:.5g}、{fchecks[2]['phase_error']:.5g}，约按四倍下降；最细网格的最大横向乘子模为{fchecks[-1]['transverse']:.9f}，远小于1。

直接延拓到1.18及减小步长到1.1775的两次尝试未收敛，因此本次补充以实际解出的1.175为界，不画未经求解的右侧连线。这个计算区间端点不能命名新分岔，也不表示稳定周期在这里终止，具体残差保留在[尝试记录](right_continuation_attempts.json)。此前时间积分仅提供周期初值，本版未添加原生SNN仿真。详细数值在[新增周期分支](extended_periodic_branch.json)、[全部图示周期点](displayed_periodic_branch.json)、[核验](validation.json)；v2的严格局部谱与402维左右向量保持不变，见[原数学报告](../core_burst_bifurcation_v2_20260915/scientific_report.md)。

## 如何补上原生状态与分岔的机制联系

当前结果已经排除了“把四状态按相同顺序直接放到这个折点两侧”的解释。下一步若要把编号2/3命名为确定的随机动力学状态，应先让降阶系统在同一原生网络、相同输入条件下匹配低率工作点及小扰动动态响应，再检查同一参数下有限扰动回返、噪声缩放和恢复/招募过程；判断依据是能否同时解释原生背景率、burst出现范围和时序，而不是继续增加当前不吻合闭合的小数精度。本版没有开展这一新机制批次。

当前匹配误差的具体来源仍包括六群体共同输入矩近似、瞬时独立spike方差闭合、被投影掉的核内协变以及尚未校准的率响应时间。原生2–20秒有限时窗分类也不是无限时间吸引子的数学定义。

## 交付与复现

- [7页矢量图册](figures/core_bifurcation_four_states_booklet.pdf)；[逐图说明](figures/README.md)。
- [近方形特征值/向量图](figures/02_square_fold_eigenvalues_vectors.png)。
- [四状态编号、参数与源文件](numbered_native_states.json)。
- Producer：`scripts/topic4_core_bifurcation_states_v3/`。按 `extend.py`、`validate_extension.py`、`plot.py`、`report.py` 顺序执行，依赖原v2的数值方程及原生归档数组。

这是待用户目视检查的v3候选；旧v2的图和数值保留。
'''
    (OUT/'scientific_report.md').write_text(report)
    (OUT/'README.md').write_text('# Core分岔与四状态编号对照 v3\n\n入口：[科学说明](scientific_report.md)、[主图](figures/01_square_bifurcation_four_states_inset.png)、[图册](figures/core_bifurcation_four_states_booklet.pdf)。\n\n本版沿用v2六群体确定性方程，增加右侧周期支并对应旧原生四状态；两层模型的边界不吻合被明确显示。旧v2保留，未新增原生SNN仿真；待用户目视检查。\n')
    print('REPORT_AND_VALIDATION_COMPLETE',json.dumps(checks),flush=True)

if __name__=='__main__':
    main()
