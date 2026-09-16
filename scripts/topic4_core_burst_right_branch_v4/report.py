"""Numerical validation and scientific interpretation of the extended branch."""
from common import *
import numpy as np
from scipy.signal import resample
from periodic import Orbit

def main():
    rows=read(OUT/'displayed_periodic_orbits.json');s=System();checks=[]
    for c in rows:
        z=np.load(c['path']);r=z['r'];T=float(z['T']);g=float(z['g']);N=len(r)
        assert float(z['residual'])<1e-8 and c['max_transverse']<1
        assert abs(r[:,0].mean()*1000-c['mean'][0])<1e-10
        assert np.isfinite(r).all()
        if str(OUT) in c['path']:
            rr=resample(r,2*N,axis=0);F=Orbit(s,g,2*N).evaluate(np.r_[(rr/.01).ravel(),np.log(T)],rr,np.zeros_like(rr))
            defect=float(abs(F[:-1]).max()*10);assert defect<1e-4
            checks.append(dict(g=g,family=c['family'],N=N,residual=float(z['residual']),offgrid_defect_hz=defect,phase_error=c['phase_error'],maximum_transverse=c['max_transverse']))
    refinement=[]
    for family,g in [('mixed',1.38),('tonic',1.38),('tonic',1.6)]:
        vals=[]
        for dt in (.1,.05):
            path=OUT/family/'floquet'/f'g{g:.8f}_dt{dt:g}.json'
            if path.exists():
                d=read(path);vals.append(dict(dtmax_ms=dt,phase_error=d['phase_error'],maximum_transverse=d['maximum_transverse_modulus']))
        refinement.append(dict(family=family,g=g,steps=vals))
    z1=np.load(OUT/'periodic/g1.30000000_N2048.npz');z2=np.load(OUT/'periodic/refined/g1.30000000_N4096.npz')
    refine_orbit=dict(g=1.3,N=[2048,4096],period_difference_ms=float(z2['T']-z1['T']),mean_difference_hz=float(abs(z2['r'].mean(0)-z1['r'].mean(0)).max()*1000),residual_N4096=float(z2['residual']))
    jac=[]
    # Smooth directional derivative check includes the free-period variable.
    for g in (1.175,1.17625,1.18,1.3):
        path=(V3 if g==1.175 else OUT)/'periodic'/f'g{g:.8f}_N2048.npz';z=np.load(path);r=z['r'];N=len(r);T=float(z['T']);o=Orbit(s,g,N)
        phase=np.zeros_like(r);y=np.r_[(r/.01).ravel(),np.log(T)]
        F,J,_=o.evaluate(y,r,phase,True);rng=np.random.default_rng(12);theta=np.arange(N)*2*np.pi/N
        dy_r=sum(np.cos(k*theta[:,None])*rng.normal(size=(1,6)) for k in (1,2,3,7));dy=np.r_[dy_r.ravel(),.1];dy/=np.linalg.norm(dy)
        eps=1e-5;fd=(o.evaluate(y+eps*dy,r,phase)-o.evaluate(y-eps*dy,r,phase))/(2*eps);actual=J@dy
        err=float(np.linalg.norm(fd-actual)/np.linalg.norm(fd));assert err<1e-5;jac.append(dict(g=g,relative_directional_error=err))
    summary=dict(n_displayed=len(rows),n_new_verified=len(checks),checks=checks,floquet_refinement=refinement,orbit_refinement=refine_orbit,jacobian_checks=jac,
        max_new_offgrid_defect_hz=max(x['offgrid_defect_hz'] for x in checks),max_new_phase_error=max(x['phase_error'] for x in checks),
        scientific_acceptance='Candidate figures delivered; user visual inspection pending')
    write('numerical_validation.json',summary)
    selected=[]
    for family,g in [('low_burst',1.175),('recruited_burst',1.18),('recruited_burst',1.3),('recruited_burst',1.36),('mixed',1.36),('mixed',1.38),('tonic',1.38),('tonic',1.4),('tonic',1.6)]:
        c=next(x for x in rows if x['family']==family and x['g']==g)
        label={'low_burst':'低背景burst','recruited_burst':'周边参与的burst','mixed':'A burst / B高率','tonic':'两核高率振荡'}[family]
        selected.append(f'| {g:g} | {label} | {c["T"]:.3f} | {c["mean"][0]:.3f} | {max(c["lo"][0],0):.4g} | {c["hi"][0]:.3f} | {c["mean"][1]:.3f} | {c["mean"][2]:.3f} |')
    ranges=[]
    for family in ('low_burst','recruited_burst','mixed','tonic'):
        a=[c for c in rows if c['family']==family];ranges.append(dict(family=family,n=len(a),gmin=min(c['g'] for c in a),gmax=max(c['g'] for c in a)))
    write('displayed_branch_ranges.json',ranges)
    doc=r'''# 核内EE右侧周期分支延拓：v4

用户本轮问题是“把右侧继续向右算，显示振荡的完整周期均值，看看继续增大连接后会怎样”。本版把 **J_EE,core 从上一版的1.175扩展至1.6**，使用原六群体确定性时延率方程和同一个冻结网络投影。没有新增原生SNN仿真；四个原生编号例子继续使用原文件。

## 主要结果

右侧并非只有一条均值单调上升的唯一周期状态。已找到两核都burst、A核burst而B核保持高率、两核都在高率背景上振荡的稳定周期解；同一个参数可有两条稳定周期轨道。每一条轨道中，两核属于同一个联合系统，使用同一个完整网络周期。

| J_EE,core | 周期状态 | 完整周期 ms | A均率 Hz | A谷值 Hz | A峰值 Hz | B均率 Hz | 周边E均率 Hz |
|---:|---|---:|---:|---:|---:|---:|---:|
TABLE

1.3时A核每个完整周期有两个明显峰，但六群体在约366.986 ms后整体重复，Floquet也支持该轨道稳定。“一个周期内多峰”不能单独证明倍周期分岔。

**同参数共存有直接数值证据。** 在1.36，两核burst的周期解与“A burst/B高率”的周期解都稳定；在1.38，“A burst/B高率”与“两核高率振荡”两条周期解都稳定。因此仅给J值不足以唯一确定网络最后落在哪一种状态，初始化历史或扰动也可能影响选择。本版确认了局部吸引轨道共存，没有测量各吸引域的体积，也没有把延拓端点当作切换阈值。

1.4时A核的完整周期均率约354 Hz、谷值约274 Hz；1.6时均率约413 Hz、谷值约394 Hz、峰值约423 Hz。后者仍是非零振幅的稳定周期解，但核心放电已经持续处于高水平，不能继续按“低背景之间的孤立burst”描述。1.4至1.6之间峰谷间距减小，均率升高；本次没有把这一高率模型分支解释为临床发作。

## 橙线和绿色线如何计算

对每个J联立求解六群体的完整周期波形、未知周期T和相位条件，使用Fourier配点周期边值问题。保留v2全部传播时延、AMPA/GABA双指数突触极点、真实逐延迟权重和、平方权重和、每群体阈值经验分布积分，以及同一输入矩闭合。周期轨道内部的单位是spikes/ms，作图转换为Hz。

橙线在每条求解轨道上计算

\[
\langle r_{\mathrm{E,core\ A}}\rangle
=\frac{1}{T}\int_0^T r_{\mathrm{E,core\ A}}(t)\,dt.
\]

绿色上下边界为同一周期轨道的最大率、最小率。先在均匀相位点上求轨道，再通过Fourier重采样读取峰谷；均值直接从整个周期的均匀点计算。有限8秒时间积分仅用于找吸引状态和提供周期初值，其最后4秒均值没有用来替代橙线。例如1.37有限窗A均率约60.60 Hz，而严格周期均率为59.71 Hz，差别来自窗内周期数与相位截断。

周期轨道稳定性来自完整时延历史空间上的变分单周期算子及其Floquet乘子、右特征向量，保存在各 `floquet/*.json` 与同名NPZ中。自治系统的相位乘子应接近1；排除它后，横向乘子模均小于1才画为稳定周期实线。它与左侧平衡支的特征根分析是不同的稳定性问题；平衡支仍沿用v2的伪弧长延拓和完整时延特征谱。

图上各点之间用直线相连，没有对原生SNN状态点做拟合，也没有跨不同轨道族强行补线。空心周期端点表示本次已核验的计算范围。左侧低率Fold=1.1254164133及SNIC型起始证据沿用v2报告，本轮没有改变该结果。

## 数值核查及未定边界

本版共展示DISPLAYCOUNT个已核验稳定的周期解，其中NEWCOUNT个来自本轮扩展；原生四状态的文件哈希、720个A核E归一化及2–20秒均值通过原producer核对。本轮展示解的周期边值残差均低于1e-8，双倍配点网格的最大率缺陷为DEFECT Hz；最大相位乘子误差为PHASE。

1.3从2048点加密至4096点后，周期及群体均值保持一致，4096点残差仍满足阈值。周期Jacobian的平滑方向导数经独立中心差分核对。1.38两种共存解和1.6高率解的Floquet步长从0.1 ms细化至0.05 ms，横向稳定性判断保持；逐点数据见 `numerical_validation.json`。

在约1.1763的极窄参数区尝试了35步周期伪弧长延拓，确实获得残差很小的候选轨道及参数方向折返。但是该区的Floquet相位检查不通过：一处候选的最近相位乘子误差从0.1 ms下约0.494降至0.025 ms下仍约0.165。因此这些候选没有纳入稳定实线，也没有据此宣布新的周期折叠、倍周期或混沌。这里仍需更适合强条件数问题的周期稳定性计算。

另外，从两核burst支向1.365、从两核高率支反向到1.37、从混合支向1.39的固定参数Newton尝试没有收敛。这些失败只标记求解器本次未跟过的区域，不证明轨道不存在或分支在该处终止。已确认的共存与高率分支不依赖这些未收敛点；新分支之间如何连接及具体分岔类型仍未确定。

## 模型层与两核含义

J_EE,core为无量纲核内EE权重倍率，同时缩放A→A、B→B的EE通路；其他连接固定。两核不严格镜像，A/B分别有720/742个E细胞和197/200个I细胞，具体连接与阈值样本不同。原图没有直接A↔B连接，耦合通过周边群体发生。两核可能不同时进入高率态，本轮共存解清楚显示这种非对称性；要将原因归给连接或阈值中的某一项，还需要对应消融。

该确定性率闭合没有原生SNN的共享OU随机驱动，且率时间常数尚未由原SNN动态响应校准。它提供周期轨道、平衡点及其稳定性的可计算结构；它没有把原生irregular burst变成一个已证明的确定性分岔类型。四个原生状态均值编号仍保留在其真实J值位置，不能用本轮右侧周期支重新定义原生状态边界。

## 文件与复现

- [单坐标轴分岔图](figures/00_extended_single_axis_bifurcation.png)、[下配四状态版](figures/01_extended_bifurcation_four_native_states.png)。
- [线性均值与峰谷图](figures/02_period_mean_linear_rate.png)、[周期和Floquet](figures/03_period_and_floquet_stability.png)。
- [联合周期波形](figures/04_joint_core_periodic_waveforms.png)、[两核与周边均率](figures/05_joint_core_period_means.png)、[六页PDF](figures/extended_core_bifurcation.pdf)。
- [逐轨道数值CSV](periodic_observables.csv)、[完整显示轨道及来源](displayed_periodic_orbits.json)、[数值验证](numerical_validation.json)。
- 方程：[v2 model.py](../../../scripts/topic4_core_bifurcation_v2/model.py)；历史平衡与起始报告：[v2 scientific_report.md](../core_burst_bifurcation_v2_20260915/scientific_report.md)。
- 本轮代码目录：`scripts/topic4_core_burst_right_branch_v4/`。`explore.py`为全时延Heun积分；`orbits.py`调用原周期求解器；`arc_cycles.py`为周期伪弧长探索；`gap_explore.py`为1.365/1.37/1.375的有界补查；`stability.py`为Floquet核验；`figures.py`只读解重画；`report.py`验证数值并生成本说明。
- 使用 `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python scripts/topic4_core_burst_right_branch_v4/figures.py` 重画，再运行同目录 `report.py` 更新核查。新旧结果目录分开保留。

本版已完成数值核查与图形自查，仍待用户目视检查；不宣称已通过人工验收。
'''
    doc=doc.replace('TABLE','\n'.join(selected)).replace('DISPLAYCOUNT',str(len(rows))).replace('NEWCOUNT',str(len(checks))).replace('DEFECT',f'{summary["max_new_offgrid_defect_hz"]:.3g}').replace('PHASE',f'{summary["max_new_phase_error"]:.3g}')
    (OUT/'scientific_report.md').write_text(doc.lstrip())
    print('VALIDATED',json.dumps({k:v for k,v in summary.items() if k!='checks'}),flush=True)

if __name__=='__main__':main()
