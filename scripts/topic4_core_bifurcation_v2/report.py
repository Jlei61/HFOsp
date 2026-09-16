"""Numerical acceptance checks and self-contained scientific delivery."""
from model import System,OUT
from branches import fold
from periodic import Orbit
from plot import orbits,read
import json,csv,numpy as np
from scipy.signal import resample

def main():
    s=System();f=read('fold.json');ev=read('eigen_validation.json');rows=[]
    for g,z in orbits():
        r=z['r'];T=float(z['T']);N=len(r);rr=resample(r,2*N,axis=0);o=Orbit(s,g,2*N)
        y=np.r_[(rr/.01).ravel(),np.log(T)];off=float(abs(o.evaluate(y,rr,np.zeros_like(rr))).max())
        q=read(f'floquet/g{g:.8f}_dt0.1.json');mm=np.array([complex(*v) for v in q['multipliers']]);idx=np.argmin(abs(mm-1));rho=float(max(abs(np.delete(mm,idx))))
        row=dict(g=g,N=N,period_ms=T,mean_core_a_e_hz=float(r[:,0].mean()*1000),min_core_a_e_hz=float(r[:,0].min()*1000),max_core_a_e_hz=float(r[:,0].max()*1000),orbit_residual=float(z['residual']),off_grid_defect_hz=off*10,
            min_all_populations_hz=float(r.min()*1000),neutral_error=float(abs(mm[idx]-1)),transverse_modulus=rho,floquet_N=q['N'])
        rows.append(row);print('CHECK ORBIT',row,flush=True)
    p=OUT/'periodic_branch.csv'
    with p.open('w') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'periodic_branch.json').write_text(json.dumps(rows,indent=2)+'\n')
    # Independent integration quadrature and threshold quadrature refinement.
    ss=System(groups=48);ss.x,ss.qw=np.polynomial.legendre.leggauss(48)
    ff=fold(ss,np.r_[np.array(f['r_hz'])/1000,f['g'],f['v']])
    selected=ev['selected'];specdiff=max(abs(complex(*q['spectra']['24'][0])-complex(*q['spectra']['64'][0])) for q in selected)
    assert max(x['orbit_residual'] for x in rows)<1e-8
    assert max(x['off_grid_defect_hz'] for x in rows)<.01
    assert max(x['neutral_error'] for x in rows)<.005 and max(x['transverse_modulus'] for x in rows)<1
    assert ev['right_residual']<1e-8 and ev['left_residual']<1e-8
    assert abs(ff['g']-f['g'])<1e-8 and specdiff<1e-6
    checks=dict(status='PASS_FOR_DEFINED_POPULATION_CLOSURE',human_visual_acceptance='PENDING',n_periodic_solutions=len(rows),
        fold_g=f['g'],fold_rate_hz=f['r_hz'][0],fold_extended_residual=f['residual'],right_eigenvector_residual=ev['right_residual'],left_eigenvector_residual=ev['left_residual'],
        eigenvalue_collocation_24_vs_64_difference_per_s=float(specdiff),refined_quadrature_fold_g=ff['g'],refined_quadrature_fold_difference=float(abs(ff['g']-f['g'])),
        max_orbit_residual=max(x['orbit_residual'] for x in rows),max_off_grid_defect_hz=max(x['off_grid_defect_hz'] for x in rows),
        min_all_population_rate_hz=min(x['min_all_populations_hz'] for x in rows),max_phase_multiplier_error=max(x['neutral_error'] for x in rows),
        max_transverse_multiplier=max(x['transverse_modulus'] for x in rows),
        native_snn_bifurcation_identification='NOT_VALIDATED',intermediate_third_attractor='NOT_ESTABLISHED',
        local_bifurcation='NONDEGENERATE_SADDLE_NODE_NUMERICALLY_CONFIRMED',global_cycle_onset='SNIC_TYPE_SUPPORTED_BY_PERIOD_LAW_AND_FINITE_TIME_RETURN')
    (OUT/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    spec=dict(model_layer='Six-population deterministic colored-Siegert delay-rate closure',parameter='same-core A and B E-to-E multiplier; other edges unchanged',threshold_depth=1.,groups=s.cfg['names'],counts=s.count.tolist(),
        rate_response_ms=s.tr.tolist(),rate_response_calibration='Inherited E=5 ms / I=2.5 ms approximation; not calibrated on this graph.',
        threshold_quadrature='32-node Gaussian quadrature of each actual empirical distribution; checked against 16/24/48 nodes and explicit per-core thresholds.',
        integral_quadrature=24,synaptic_rise_ms=s.rise.tolist(),synaptic_decay_ms=s.decay.tolist(),delay_bins=len(s.delay),delay_range_ms=[float(s.delay.min()),float(s.delay.max())],
        ext_mean=s.ext_mu.tolist(),ext_variance=s.ext_var.tolist(),external_drive='Private core-E Poisson retained through moments; shared OU removed; surround and I external drive deterministic.',
        recurrent_variance='Instantaneous independent-spike diffusion closure, squared actual edge weights.',
        synaptic_area='Native discrete DC area held fixed; continuous two-pole temporal response.',
        physical_parameters=s.p)
    (OUT/'model_spec.json').write_text(json.dumps(spec,indent=2)+'\n')
    fit=read('period_scaling.json');represent=next(x for x in rows if x['g']==1.15);near=rows[0]
    flo=read('floquet/g1.15000000_dt0.025.json');mu=np.array([complex(*x) for x in flo['multipliers']]);trans=float(np.sort(abs(mu))[-2])
    table='\n'.join(f"| {x['g']:.7f} | {x['period_ms']:.6f} | {x['mean_core_a_e_hz']:.5f} | {x['max_core_a_e_hz']:.3f} | {x['N']} |" for x in rows)
    report=f'''# Core 低率态到 burst：经典分岔分析候选 v2

本版完成平衡态伪弧长延拓、包含全部传播时延和突触滤波的特征谱、左右临界特征向量、自由周期轨道求解及 Floquet 稳定性。分析范围是 core burst 起始，不包含发作或 Z/M。

**在本版明确写出的六群体降阶方程中，低率态通过非退化 saddle-node 消失；稳定的大幅度 burst 周期随接近折点而增长，周期标度和临界回返支持 SNIC 型起始。当前随机 SNN 的中间表型尚不能据此命名分岔。**

## 图和坐标

主图为 [01_classical_bifurcation.png](figures/01_classical_bifurcation.png)：X 是同一个 core 内 E→E 权重倍率 g，同时作用于 A、B 两核；Y 是 core A 内每个 E 神经元的群体平均率，单位 Hz。固定阈值场幅度 1、原拓扑 2511、GABA 衰减 18 ms。

蓝色实线是稳定低率平衡点，红色虚线是延拓到的不稳定平衡点，绿色给出求解所得稳定周期轨道的最大/最小率。主图 Y 轴在 1 Hz 以下线性、以上对数，下面另有折点的线性坐标放大。绿色阴影表示周期振幅范围，不能理解为这一范围内全是平衡点。这里的 resting 是带背景输入方差的低率群体态；单细胞 spike 后的电压 reset 是不同对象。

## 确定的局部分岔

- 临界 g = **{f['g']:.11f}**，core A E 平衡率 **{f['r_hz'][0]:.9f} Hz**；core B E 为 {f['r_hz'][1]:.9f} Hz。
- 延拓分支折返，扩展方程 F=0、Jv=0、vᵀv=1 的最大残差为 {f['residual']:.3g}；参数横截项与二次项都非零。
- 临界特征值是单个实根 λ=0。非临界最靠近零的根约 −22.36 s⁻¹；不是复共轭对先过零的 Hopf 起始。
- 在 g_c−10⁻⁶，两条局部分支的临界特征值分别约 −0.098945、+0.098915 s⁻¹，虚部为零。
- 右特征向量残差 {ev['right_residual']:.3g}，左特征向量残差 {ev['left_residual']:.3g}。按 [A E, B E, Surround E, A I, B I, Surround I] 排列，v=(1,0,0,0,0,0)，w≈(1,0,0.91524,−0.94752,0,−0.69256)，其中 ||v||₂=1、wᵀv=1。

图中左右向量精确定义为 6×6 特征矩阵 M(0) 的左右零空间向量；完整时延模态还包含突触和历史分量。另存 [402 维生成算子的完整左右特征向量](full_generator_critical_mode_N64.npz)，其独立残差检查见 [full_generator_eigenvector_validation.json](full_generator_eigenvector_validation.json)。

右向量说明这个**六群体坐标系统**里最先变慢的活动方向是 A 核 E 群体；左向量是对该模态的输入敏感度，负号不能理解为“负的活动比例”。这不是 40,000 个原生神经元的逐细胞特征向量，核内空间差异已被投影。

另一个较高率分支折点 g≈0.57003456 的系统已有约 +29.87 s⁻¹ 的不稳定根，不能把该折点当作稳定 resting 的起始边界。图中高率平衡支为虚线；g=0.85、1 的复特征根位于右半平面，直接代回完整时延特征方程的残差低于 10⁻¹⁴。

## 周期分支与 SNIC 证据

周期轨道通过 Fourier 配点求解，未知量同时包括整条六群体波形和周期，并增加相位条件。每个谐波保留原始 368 个时延箱及 AMPA/GABA 双指数滤波。时间积分仅提供初值；图上的周期点都满足周期方程。

| g | 周期 ms | A E 平均率 Hz | A E 峰率 Hz | 配点数 |
|---|---:|---:|---:|---:|
{table}

g=1.15 时，周期为 **{represent['period_ms']:.9f} ms**；Floquet 时间步长约 0.025 ms 时，最大非中性乘子为 **{trans:.9f}**，稳定。中性相位乘子偏离 1 的误差随步长 0.1→0.05→0.025 ms 约按四倍缩小，符合二阶积分收敛。所有图示周期解均单独检查了 Floquet 稳定性；小于 10⁻¹⁰ 的横向乘子只作为上界显示。

最靠近折点的已求解周期位于 g={near['g']:.7f}，周期 {near['period_ms']:.6f} ms，峰率仍为 {near['max_core_a_e_hz']:.3f} Hz。趋近临界主要增长的是低率等待时间，burst 振幅没有缩到零。

令 x 为 A 核 E 率相对折点的偏移，单位 Hz，时间单位秒；由左右特征向量及完整延迟特征方程的 λ 导数得到局部正规形：

    dx/dt ≈ 33.72626276 (g − g_c) + 72.54824315 x²

它独立预测瓶颈时间前因子 C={fit['C_theory_ms']:.6f} ms，即 T≈C/√(g−g_c)+常数。周期解拟合得到 C={fit['C_fitted_ms']:.6f} ms，差 {fit['relative_prefactor_difference']*100:.3f}%，R²={fit['r2']:.9f}。这里是确定性数值曲线的一致性检查，R²不是实验统计证据。

在恰好 g=g_c 处，从折点沿活动方向偏移 +0.02 Hz 后，积分出现一次大 burst，并回到同一折点的低率侧；16 s 时仅低于折点 0.0009204 Hz，且持续靠近。**局部 saddle-node 已数值确证；有限周期分支、正规形标度和有限时间全局回返共同支持 SNIC 型起始。无限时间的全局连接没有做区间算术证明。**

## 原 SNN 对应关系与中间态

图的连接来自本次放电率探索实际使用的双核网络：32,000 E、8,000 I；六组细胞数分别为 720、742、30,538、197、200、7,603。逐延迟箱投影实际连接权重及平方权重，保留原始阈值的经验分布；没有拟合 burst 标签或把旧 q 临界值移过来。

但这仍是新建的近似层：每群体只保留共同输入矩，忽略核内连接/阈值与活动的协变和有限规模同步涨落；使用 colored-Siegert 扩散近似、瞬时递归方差闭合；E=5 ms、I=2.5 ms 的率响应时间沿用旧方法，**尚未在这张图的当前网络上校准动态响应**。

实际不吻合已经可见：仅保留 private Poisson 的原 SNN 在 g=0.7 的平均率约 0.890–0.958 Hz，本闭合低率约 0.19–0.21 Hz；原 SNN 在 g=0.85 已有重复 burst，本闭合沿低率支仍稳定。原 SNN 数值来自 [v1 原生噪声干预报告](../core_burst_onset_brunel_v1_20260915/scientific_report.md)。这不是小数精度可以解决的差异。

因此接受本版作为**可复核的降阶系统分岔候选图**，暂不接受它对原随机 SNN 的临界 g 或中间态类型的定论。对本降阶系统而言，接近阈值的低率态与大幅度可激发回返是合理的结构；原 SNN 的不规则/部分招募事件是否由有限涨落访问这个结构造成，仍需匹配低率工作点、小扰动增益与相位后再判断。已有原生恢复过程还存在过深超极化的限制，亦不能用这个率近似消除。

## 数学定义与验证

详见 [model_spec.json](model_spec.json)、[validation.json](validation.json) 和配套脚本。内部率用 spikes/ms、时间用 ms；所有图转换为 Hz 和秒。令 h_j、c_j 为每个源群体的两级突触滤波率：

    τ_r,j dh_j/dt = r_j − h_j
    τ_d,j dc_j/dt = h_j − c_j
    μ_i(t) = μ_ext,i + τ_m,i Σ_j,d W_d,ij A_j sign_j c_j(t−d)
    V_E,i(t) = V_ext,i + τ_m,i Σ_(j∈E) Q_ij r_j(t)
    V_I,i(t) = τ_m,i Σ_(j∈I) Q_ij r_j(t)
    τ_rate,i dr_i/dt = Φ_i(μ_i,V_E,i,V_I,i) − r_i

A_j 保留原生 dt=0.1 ms 更新的 DC 面积；W 为每个接收细胞的实际权重和，Q 为相应平方权重和，g 对同核 EE 的 W 线性缩放、对 Q 平方缩放。Φ 对实际经验阈值分布积分，包含复位、绝对不应期和有色噪声阈值修正；定义以 model.py 为准。

线性化消去 h、c 和历史后，M(λ)=diag(1+λτ_rate)−V−diag(∂Φ/∂μ)C(λ)，C 保留 exp(−λd) 与两个突触极点。求解 det M(λ)=0 并核对左右零空间；静态平衡残差 Jacobian 的非零特征值没有当作动态 λ 使用。为保留被消去的稳定滤波模态，另用 24/40/64 阶 Chebyshev 历史生成算子求谱，主导谱差 {specdiff:.3g} s⁻¹。右半平面 argument-principle 根数在双倍网格下相同：低率支 0、临界附近 saddle 支 1；矩阵行和界把所有可能不稳定根限制在核查矩形内。

阈值经验分布使用 Gaussian quadrature，而非等频分箱均值；16–48 节点的折点一致。进一步把阈值和 Siegert 积分均提高到 48 节点，g 变化 {abs(ff['g']-f['g']):.3g}。早期等频分箱试算留在 pilot_equal_mass_threshold/，其 g≈1.13797 已被修正，不属于本版最终结果。

周期配点最大残差 {checks['max_orbit_residual']:.3g}；在两倍时间网格上重算周期方程，最大率等价缺陷 {checks['max_off_grid_defect_hz']:.3g} Hz。有限 Fourier 截断会在理论非负的极低率段产生极小负值，最大量级 {abs(checks['min_all_population_rate_hz']):.3g} Hz，文件保留原数值。g=1.15 的 1024→2048、最近折点的 4096→8192 配点加密均保持周期一致。此处数值精度不代表原 SNN 的机制误差或参数置信区间。

## 文献依据

[Brunel & Hakim 1999](https://webhome.phy.duke.edu/~nb170/pdfs/brunel99.pdf)启发先求群体平衡态、谱和非线性分支，再命名振荡起始。[Brunel & Hakim 2008](https://www.phys.ens.psl.eu/~hakim/08chaosnbvh.pdf)与 [Ledoux & Brunel 2011](https://webhome.phy.duke.edu/~nb170/pdfs/ledoux11.pdf)强调动态响应和突触相位，正是本版保留完整时延/滤波、同时承认 τ_rate 还未匹配原 SNN 的原因。SNIC 的局部折点、全局回返与低频起始应联合检查，参见 [Izhikevich 2000](https://www.izhikevich.org/publications/nesb.pdf)；文献里的单细胞 spike 起始结构在这里仅作为数学分类参考，本版的周期事件是群体 burst。

## 文件入口

- [六页完整图册](figures/core_burst_classical_bifurcation_booklet.pdf)；[逐图说明](figures/README.md)。
- [特征值和左右特征向量](figures/02_eigenvalues_and_eigenvectors.png)；[周期标度](figures/03_period_divergence.png)；[Floquet 稳定性](figures/05_floquet_stability.png)。
- [平衡分支及谱](equilibrium_spectrum.json)、[折点细扫描](fine_fold_branch.json)、[周期分支 CSV](periodic_branch.csv)、[验证记录](validation.json)。
- Producer：`/home/honglab/leijiaxin/HFOsp/scripts/topic4_core_bifurcation_v2/`。完整系数在 projected_graph.npz；源码和 README 保留复现入口。

本版图需用户目视检查，未宣称人工验收通过。
'''
    (OUT/'scientific_report.md').write_text(report)
    descriptions={
      '01_classical_bifurcation':'展示 core A E 的稳定低率支、不稳定平衡支和通过周期方程求解的 burst 最大/最小值，下方放大折点和临界实特征值。X 为核内 EE 倍率，阈值场固定；主 Y 轴在 1 Hz 以下线性、以上对数。**关注点**：折点和动态实根过零必须同时出现，绿色范围不是固定点区域。',
      '02_eigenvalues_and_eigenvectors':'展示折点处完整时延系统的主导谱、临界附近稳定支与 saddle 支的谱，以及左右临界向量。向量仅有六个群体坐标，右向量表示活动方向，左向量表示模态输入敏感度。**关注点**：这里没有逐神经元或核内空间分辨率。',
      '03_period_divergence':'展示自由周期轨道靠近折点时周期增长，并把数值周期与局部正规形独立预测的逆平方根前因子比较。幅度仍有限，等待时间增长。**关注点**：该一致性支持 SNIC 型起始，但不替代无限时间全局连接证明。',
      '04_global_return':'展示恰好在临界 g 处沿活动方向偏移后的一次大 burst，以及其向同一折点低率侧的返回。右侧对数轴显示差距持续缩小。**关注点**：这是有限时间回返诊断，与已求解的周期分支互补。',
      '05_floquet_stability':'展示各周期点最大的非中性 Floquet 乘子，以及 g=1.15 的中性相位乘子随步长加密趋近 1。小于 10⁻¹⁰ 的横向乘子用下三角表示上界。**关注点**：自主系统的相位乘子理论上为 1，不能将积分误差造成的略大于 1 误判为周期失稳。',
      '06_continued_burst_waveforms':'从已求解周期轨道重建三个参数条件的连续波形，并只做时间原点平移以完整显示两次 burst。蓝、红分别为 core A 内 E、I 每细胞平均率。**关注点**：靠近折点主要增长 burst 之间的等待时间，群体率不等于 HFO 载频。'}
    text='# 分岔分析候选图 v2\n\n所有图均为已定义六群体降阶系统的数值分析，原 SNN 对应性尚未验收。每图均有同名 PNG/PDF；用户目视验收待定。\n\n'
    for name,desc in descriptions.items():text+=f'### {name}.png\n\n{desc}\n\n'
    text+='### core_burst_classical_bifurcation_booklet.pdf\n\n将上述六张图按平衡分支、临界谱、周期标度、回返、稳定性和波形顺序汇总。每页保留完整坐标和层级说明。**关注点**：先看数学确证，再看原 SNN 对应的限制。\n'
    (OUT/'figures'/'README.md').write_text(text)
    (OUT/'README.md').write_text('''# Core burst 经典分岔候选 v2

先读 [scientific_report.md](scientific_report.md)；本版为降阶系统分析，原 SNN 分岔未确证。

固定系数：`projected_graph.npz`、`model_spec.json`。Python 为 `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python`，producer 为 `/home/honglab/leijiaxin/HFOsp/scripts/topic4_core_bifurcation_v2`。

从已有系数复现的顺序：`branches.py` → `spectral.py` → `validate_eigen.py`；`dynamics.py` 提供周期初值，`periodic.py --g 1.15 --N 1024` 求首个周期点，`continue_cycles.py` 延拓到 1.1263。极近折点的首次猜测若不收敛，检查保存的 residual 后从更好的连续轨迹初值重新求解，失败点不画入图。已交付的最近折点通过重新积分取初值恢复，并加密至 8192 点；周期 seed 和全部最终轨道均已保留。

对每个最终周期文件运行 `floquet.py <orbit.npz> --dt 0.1`；代表点 g=1.15 加密到 0.05、0.025 ms。最后运行 `plot.py` 和 `report.py`。`fold_global_return.npz` 为在扩展方程临界点、沿右向量偏移 0.02 Hz、积分 16 秒的一次回返。

旧 SNN 代码、旧结果和其他工作的文件均保留。`pilot_equal_mass_threshold/` 是本轮被积分精度检查替代的早期试算，不属于最终数值。
''')
    print('DELIVERY VALIDATION',json.dumps(checks),flush=True)

if __name__=='__main__':main()
