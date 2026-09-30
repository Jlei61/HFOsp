"""Source and numerical checks for the critical-node figure revision."""
from threshold_node_scan import *
import csv,hashlib

def run():
    folds=read(DEST/'mean_fold.json');checks=[]
    for row in [folds[0],folds[len(folds)//2],folds[-1]]:
        mean=row['mean_A_mV'];s=MeanSystem(mean,64)
        q=fold(s,np.r_[np.array(row['r_hz'])/1000,row['g'],row['v']])
        delta=q['g']-row['g'];assert abs(delta)<1e-7
        actual=BASE.original_thresholds+mean-MEAN
        assert abs(actual.mean()-mean)<1e-12 and abs(actual.std()-STD)<1e-12 and actual.max()<=18+1e-12
        checks.append(dict(mean_A_mV=mean,quadrature32_64_g_difference=delta))
    table=[]
    for row in folds:table.append(dict(type='SN',mean_A_mV=row['mean_A_mV'],sigma_A_mV=STD,J_EE_core=row['g'],residual=row['residual'],source='mean_fold.json'))
    coverage={};valid=[]
    for k in ['PD1','PD2','PD3','LP1']:
        p=DEST/f'{k}.json'
        if not p.exists():continue
        d=read(p);coverage[k]=dict(status=d['status'],points=len(d['points']),mean_range=[min(r['mean_A_mV'] for r in d['points']),max(r['mean_A_mV'] for r in d['points'])])
        for r in d['points']:
            assert r['orbit_residual']<5e-9 and r['null_residual']<1e-7
            table.append(dict(type=k,mean_A_mV=r['mean_A_mV'],sigma_A_mV=STD,J_EE_core=r['g'],residual=r['orbit_residual'],source=r['source']))
        vp=DEST/f'{k}_grid_validation.json'
        if vp.exists():valid.append(dict(type=k,**read(vp)))
    with (DEST/'critical_nodes.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['type','mean_A_mV','sigma_A_mV','J_EE_core','residual','source']);w.writeheader();w.writerows(table)
    meta=dict(model='Same frozen six-population DDE; actual empirical A thresholds shifted uniformly',mean_threshold_reference_mV=MEAN,
        threshold_std_mV=STD,mean_shift_equation='theta_Ai(new)=theta_Ai(reference)+mean_new-mean_reference',
        spread_axis_explanation='Previous h=1 denotes the reference SD. Affine h>1 raises some core thresholds above 18 mV and changes the established core setting.',
        mean_domain='16.5 mV to reference for SN; actual periodic-curve ranges below',
        periodic_coverage=coverage,quadrature_checks=checks,periodic_grid_checks=valid,
        onset_switch=read(DEST/'AB_onset_switch.json'),
        interpretation='Critical curves of specific branches; a crossing of projected curves does not establish a codimension-two bifurcation on the same orbit.',
        unsupported=['all attractor basins','complete mean-threshold cycle-fold continuation','native SNN or pathology validation'],
        human_visual_acceptance='PENDING')
    write(DEST/'validation_and_scope.json',meta)
    (DEST/'report.md').write_text('''# 双参数图：围绕临界节点的修订

本次按用户要求集中完善参数平面，并新增平均阈值轴。原先 h=1 只是原始标准差的归一化参考，不是生理上限；但当前仿射放大到 h>1 会使部分 core E 阈值超过 18 mV 的背景值，改变已约定的 core 设置。因此主图改用实际标准差（mV），放大真正敏感的区间，全范围保留在配图；没有为得到弯曲线而改动模型。

## 这次真正新增的计算

1. 精修 A/B 首次低率失稳的切换位置：sigma_A 约 0.713 mV、J_EE,core 约 1.175。全系统在此有两个数值上接近零的模态；尚不将它命名为 Bogdanov–Takens 或 cusp。较低标准差下 B 主导，解释原图长竖线；较高标准差下 A 主导。
2. 固定阈值标准差，给全部 A E 阈值加同一个偏移。平均阈值降低时，SN 明显向较小 EE 移动；25 个平衡 fold 点与代表点的 40/64 阶 DDE 特征根、32/64 节点阈值求积都保留验证。所有新均值条件仍满足 core 阈值不高于背景，没有裁剪。
3. 在平均阈值轴上求周期临界点，直接解周期轨道方程及 PD 的反周期零模，不用时间扫描色块定义分岔。PD1、PD2、PD3 和 LP1 的实际覆盖分别见 validation_and_scope.json；较大参数步长失败的尝试保留，图只画收敛的点。LP1 仍只有局部范围，不能称完整周期状态分类。

## 图与分支如何对应

critical_parameter_planes 的上排是异质性轴，下排是平均阈值轴；左右分开显示静息态 fold 和周期分岔，避免把无关分支的线画成整张图的唯一分区。顶部方框和三角形是原模型的具体临界节点，颜色与 matched_reference_branches 一致；后者复用此前的稳定实线／不稳定虚线分支。

SN 是低率平衡解的 fold，cycle fold 与 LP1 是周期解的 fold。PD1 和 PD2 位于混合周期分支的不同边界，PD3 位于高背景周期分支；三者不是同一轨道连续发生的 1→2→4→8 倍周期序列。临界模态中 PD1 主要涉及 B_I，PD2 主要涉及 A_E，PD3 主要涉及 A_I，这有助于解释它们对 A 阈值变化的不同敏感性。

主图上排故意放大 sigma_A 约 0.70–0.74 mV，全扫描区间仍见 full_spread_range。下排 LP1 空心端点与其他未闭合端点表示本轮数值覆盖的边界，不是已经证实的 cusp、同宿或分支消失。参数平面的线相交不自动代表同一个周期解的余维二分岔。

旧 LP0/PD0 的极窄分支仍未获得可靠的双参数延拓。此次没有将低率 fold 与 cycle fold 之间直接涂成稳定共存区，没有补造状态分界，也没有将降阶模型结果当成原生 SNN 或患者组织结论。

代码：threshold_node_scan.py（均值扫描及切换点）、node_figures.py（绘图）、node_report.py（验证和汇总）。图和 CSV/JSON 均在本目录内；待用户人工目视确认。
''')
    (DEST/'figures/README.md').write_text('''# 关键分岔节点与两类参数轴

### critical_parameter_planes.pdf

上排以实际阈值标准差为纵轴，下排以平均阈值为纵轴；平均阈值扫描保持标准差不变。分别放大 onset 和周期分岔，节点颜色与参考分支配图对应。**关注点**：A/B 首次失稳切换、平均阈值对 SN 的影响，以及各周期分支的实际临界点。

### matched_reference_branches.pdf

复用已验证的六群体模型参考分支，显示临界节点位于哪些稳定／不稳定解上。主图的 PD1–3 不属于单一轨道的连续倍周期级联。**关注点**：分岔对象、分支稳定性与参数平面曲线之间的对应。

### full_spread_range.pdf

保留异质性全范围，用细线展示已求解的临界曲线。主图局部放大不意味着未展示部分没有动力学。**关注点**：低异质性条件下 B 控制的首次静息失稳边界。

各图同时提供 PNG/SVG，仍待用户人工目视检查。
''')
    print('NODE REPORT',coverage,checks,flush=True)

if __name__=='__main__':run()
