"""Report what the branch gap does and does not establish."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import csv,json
import numpy as np
from topic4_fig5_D_gap_focus import OUT,SOURCE
from topic4_fig5_D_physical_model import Equilibrium

def main():
    audit=json.loads((OUT/'fold_spatial_audit.json').read_text())
    folds=audit['folds'];eq=Equilibrium(1.25);m=eq.m
    branches=[]
    for name in ('middle_extension','high'):
        a=np.load(OUT/f'q1.25_{name}_followup.npz')
        meta=json.loads((OUT/f'q1.25_{name}_followup.json').read_text())
        old=np.load(SOURCE/f'q1.25_{name}.npz')
        assert np.array_equal(a['r_hz'][0],old['r_hz'][-1])
        assert a['s'][0]==old['s'][-1]
        errors=[row['residual_max_hz'] for row in meta['points']]
        assert len(a['s'])==180 and max(errors)<2e-6
        branches.append(dict(branch=name,records=len(a['s']),new_points_excluding_seed=len(a['s'])-1,
             D_min=float(min(a['s'])),D_max=float(max(a['s'])),
             endpoint_D=float(a['s'][-1]),endpoint_mean_E_hz=meta['points'][-1]['mean_e_hz'],
             residual_max_hz=max(errors),seconds=meta['seconds'],
             stop_reason='REACHED_PRESET_180_RECORD_LIMIT',
             full_stability='NOT_CLASSIFIED_IN_THIS_FOLLOWUP',
             turn_candidates=int(np.count_nonzero(a['tangent'][:-1,-1]*a['tangent'][1:,-1]<0))))
    gap=[branches[0]['D_max'],branches[1]['D_min']]
    assert gap[0]<gap[1]
    fig=json.loads((OUT/'focus_figure_data.json').read_text())
    selected=fig['spatial_points'];assert len(selected)==4
    assert 'saddle-node' in selected[2]['kind']
    assert 'not a global oscillation' in selected[3]['kind']
    for k in (2,3):
        row=selected[k];a=np.load(row['source'])
        target=a['r_hz'][row['source_index'],:400] if 'source_index' in row else a['r_hz'][:400]
        assert np.array_equal(np.array(row['rate_hz']),target)
        assert abs(np.average(target,weights=m.count_e)-row['displayed_phase_global_E_hz'])<1e-10
    assert all(f['type']=='SADDLE_NODE_OF_EQUILIBRIA' and f['quadratic_converged'] for f in folds)
    separation=min(r['full_state_max_difference_hz'] for r in audit['pairwise_state_distances'])
    assert separation>1e-4
    residuals=[]
    for fold in folds:
        a=np.load(SOURCE/(fold['name']+'.npz'))
        residuals.append(float(np.max(abs(eq.evaluate(a['r_hz'],float(a['D']))))))
    assert max(residuals)<1e-7
    qa=dict(status='PASS',previous_stars_distinct_equilibria=len(folds),
            smallest_pairwise_full_state_max_difference_hz=separation,
            recomputed_fold_max_residual_hz=max(residuals),
            largest_mode_95_percent_cell_count=max(f['mode_95_percent_cells'] for f in folds),
            spatial_total_cells=400,followup_branches=branches,
            unresolved_connection_D_interval=gap,
            selected_fields_exact=True,selected_global_rates_exact=True,
            snapshot_4_D=selected[3]['D'],snapshot_4_kind=selected[3]['kind'],
            new_trace_stability_certified=False,new_turn_candidates_plotted_as_stars=False,
            zero_log_colors='display floor only, original arrays retained',
            native_global_oscillation_onset='NOT_ESTABLISHED',
            human_visual_acceptance='PENDING')
    (OUT/'qa.json').write_text(json.dumps(qa,indent=2)+'\n')
    columns=['number','name','D','mean_e_hz','core_A_rate_hz','core_B_rate_hz',
             'mode_95_percent_cells','mode_peak_xy_mm','type']
    with (OUT/'fold_catalogue.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(folds)
    comparison='\n'.join(f"| SN{f['number']} | {f['D']:.9f} | {f['mean_e_hz']:.6f} | {f['core_A_rate_hz']:.3f} | {f['core_B_rate_hz']:.3f} | {f['mode_95_percent_cells']} |" for f in folds if f['number'] in (1,5,7))
    report=f'''# 分支断口、多鞍结与空间快照位置

## 断口是计算覆盖缺口，不是已经确定的动力学跳变

上一版从低态出发的延拓先保存 180 点，再续接 240 点，最大 D 只到 0.250217；从 D=1 高态反向的独立延拓保存 160 点，最小 D 到 0.506571。三段都触及预设记录数量上限。轨道沿弧长经过多次参数折返，记录点增多不等于 D 单调向前推进。因此横轴 0–1 的展示不等于整个区间已求全。

此前 D=0.25、0.30、0.35、0.40、0.45 的 15 个均匀初值求根均未收敛；失败只能说明这些初始化没有得到根。不能把空白称为无平衡区，也不能把两条分支用一根竖线连接为已经建立的跳变。

本轮分别从两个已存末端沿原切向各追加 180 个记录（各含一个接续起点，共 358 个新增点），保留空间异质性、完整 800 变量残差及曲率信任限制。中率支最大 D={gap[0]:.9f}，高率支最小 D={gap[1]:.9f}，仍未接通；本批均达到记录数量上限后结束。新增延拓均满足平衡残差，但尚未逐段分类完整动态稳定性，因此主图保留已有稳定性证据的分支段，没有将新增转向候选直接画成鞍结星号。

这里没有证据证明两侧必须直接相连，也没有排除其他共存平衡、周期或非周期状态。全局振荡从哪里出现仍未回答。

## 为什么一条投影曲线上有这么多星号

原图 16 个星号对应 16 个不同的空间平衡鞍结解：当前重新计算平衡残差最大 {max(residuals):.3g} Hz，全部仍有此前的简单零根、非零参数横截性及收敛的非零二阶系数证据。完整 800 维状态任意两点的最大分量差最小为 {separation:.6g} Hz，不是重复保存同一个根。

平衡解是一张空间放电率场；纵轴只显示全局平均，丢失了核 A、核 B 与核外活动的组合。不同空间态可以在几乎相同的 D 上发生局部折返。例如：

| 点 | D | 全局 E 率 Hz | 核 A 率 Hz | 核 B 率 Hz | 临界 E 模态 95% 能量格数 |
|---|---:|---:|---:|---:|---:|
{comparison}

三者的临界模式都主要位于核 B，但核 A 的平衡背景明显不同；同一类局部失稳可以出现在不同的空间平衡背景上。SN_H 附近四个折返的临界模式则集中在边缘约 4–6 个格子。所有 16 个点的临界 E 模态 95% 能量只占 4–40 个格子（总共 400 格），不能把这些星号逐一称为“全局同步爆发起点”。局部失稳是否进一步非线性招募全局活动仍需独立证明。

这是当前固定空间划分降阶系统内的数值结论，不等于原生网络拥有同样 16 个生理转变。空间离散与闭合变化下是否保留这些细小折返尚未检验。空间模式分支可出现多个鞍结在其他模型中也有原始研究，例如 [Lloyd and Sandstede, 2011](https://epubs.siam.org/doi/10.1137/100782747)；此处并未据此把本模型直接命名为同宿蛇形分支。

## 本次图形修改

- 保留①低态平衡与②一条稳定 A 周期的空间场，去掉重复的 B 周期快照。
- ③移到中率支鞍结 SN_L：D={selected[2]['D']:.9f}、平均率 {selected[2]['mean_e_hz']:.6f} Hz。显示的是该平衡解本身，不是声称自由仿真会停留在此。
- ④从 D=1 前移到已采样并完整判稳的高率支最小 D={selected[3]['D']:.9f}，平均率 {selected[3]['mean_e_hz']:.6f} Hz。它接近 SN_H（D=0.506566608），仍属于高率平衡态，不能标成全局振荡起点。
- 主图横轴收窄为 0–0.56；放大窗改看高率支边缘的细小折返，保留标准稳定性符号和已确认鞍结星号。
- 对数色图中零值和约 1e-19 Hz 的舍入负值显示在色标下限，修复空白像素；数值原数组不改动。

新目录独立保留，不覆盖上一版结果；q_IE=1.25、Z 作为冻结空间参数、M 保持动态均未改变，没有新增原生 SNN 仿真。图已由 Agent 目检，用户人工验图待完成。

## 可复现输出

- figures/fig_D_fold_focus_spatial.*：更新主图。
- figures/fig_multiple_folds_spatial.*：SN1/SN5/SN7 的平衡场与临界模态。
- fold_catalogue.csv、fold_spatial_audit.json：16 个旧星号逐点空间检查及两两状态距离。
- q1.25_middle_extension_followup.npz / .json 与 q1.25_high_followup.npz / .json：两端追加延拓；完整稳定性仍未分类。
- focus_figure_data.json、qa.json：选点对应与本轮验算。
'''
    (OUT/'scientific_report.md').write_text(report)
    (OUT/'status.json').write_text(json.dumps(dict(status='EXECUTION_COMPLETE',
        branch_gap='UNRESOLVED_CONNECTION',global_oscillation_onset='NOT_ESTABLISHED',
        prior_16_folds='RECHECKED_DISTINCT',new_continuation_points=358,
        plots='GENERATED_AND_AGENT_INSPECTED',human_visual_acceptance='PENDING',active_processes=[]),indent=2)+'\n')
    print('QA PASS',json.dumps(qa),flush=True)

if __name__=='__main__':main()
