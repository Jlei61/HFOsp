# 空间 Z/M saddle organizer：延迟、动态投影与网格收敛审计（2026-09-04）

## 审阅结论

### 1. 一句话判断

**现在有一条可以进入 paper-facing Fig. 5 的结果链**：持续 OU 下的 finite SNN 以
prospective 3/3 seeds 进入 near-saturated tonic global runaway；患者匹配的 reduced
frozen-q fast subsystem 在三种保守空间粗化中都有 generic saddle-node，且三颗 SNN
轨迹都先穿过该 fold、再达到操作性 runaway onset。

最大边界也同样清楚：包含实际 recurrent conduction delays 后，high equilibrium 是
**线性不稳定的 high-rate skeleton**，不是稳定高 fixed point；fold rate 对网格尚未收敛。
因此这是一条 **saddle-node organizer** 证据链，不是 finite-SNN thermodynamic phase
transition 的证明。

### 2. 完成程度

> **完成度：92/100**

已完成：1 mm generic fold 数值验证、实际延迟算子与 branch stability、1841–1843 三颗
SNN 动态轨迹对同一 critical manifold 的投影、2/1.33/1 mm 保守粗化的 branch identity
与 fold comparison、审计图、严格 A–D paper-facing Fig. 5、metadata 与专项测试。

扣分来自不能越过的科学边界：stationary colored OU 尚未进入 reduced transfer closure；
fold rate 未通过网格收敛；finite stochastic SNN 没有定义可用于严格分叉定理的无限时间、
无限系统极限。这些不阻断当前 tonic-runaway Fig. 5，但阻断“精确临界点/严格相变”的强表述。

## 1. 这次到底验证了什么

### 1.1 两个 q 不能混成一个量

- reduced model 的 `q`：把空间场冻结为一个统一控制坐标，只缩放 I→E；它是 fast
  subsystem 的 continuation parameter。
- SNN 的 `q_core(t)`：病灶核内抑制资源的空间平均。
- SNN 的 `q_mean(t)`：整张 sheet 的空间平均。
- SNN 的 `q_min(t)`：最耗竭位置的极值，只作范围/高支率比较，不能替代空间平均。

所以 `q_fold=0.8908` 不是说 finite SNN 在 `q=0.8908` 精确起病。正确问题是：空间动态
轨迹的不同 q 摘要是否先穿过 reduced manifold 的 fold，并随后靠近同一 high-rate
skeleton。三颗 seed 的答案都是“是”。

### 1.2 1 mm generic saddle-node 仍成立

`eta_m=0.02` 时：

- tangent fold：`q=0.8908259261275`；
- Jacobian real-zero fold：`q=0.8908259261280`；
- 两者差 `4.9×10^-13`；
- fold mean E rate：`127.399 Hz`；
- 同一 `q=0.890700` 有 `130.572 Hz` 与 `124.459 Hz` 两个相邻 fixed points；
- simple isolated real zero mode、transversality 与 quadratic nondegeneracy 均通过。

这比“曲线看起来折了”强：它验证的是 generic saddle-node 的局部数值条件。

## 2. 包含实际突触延迟后的稳定性

### 2.1 方法学修正

初版 delay stability 只固定 operating variance，导致零频增益与 saddle-node 使用的完整
fixed-point Jacobian 不一致。正式版改为：

1. 每条 E→E、E→I、I→E、I→I pathway 在空间聚合时同时守恒一阶与二阶 weight moment；
2. 用 explicit rate-history shift register 保留实际 delay distribution；
3. 主分析使用 `self_consistent_variance` stationary-diffusion closure，使零频 gain 与
   saddle-node fixed-point Jacobian 一致；
4. `frozen_variance` 只保留为旧 M3B convention 的 sensitivity，不参与 fold mode 判定。

delay operator 一阶/二阶 moment 最大守恒误差分别不超过 `3.20×10^-14` 与
`2.84×10^-14`。`dt=0.5/0.25/0.1 ms` 三档中，fold 的 stationary null direction 都提升为
delay-map unit mode；主 closure 最大相对残差 `4.76×10^-9`。

### 2.2 结果

| branch | q range | max Re(lambda) range, dt=0.5 ms | 判断 |
|---|---:|---:|---|
| near-silent | 0.775–0.890 | −0.03589 to −0.03538 ms^-1 | 稳定 |
| high-rate | 0.775–0.890 | +0.00829 to +0.05944 ms^-1 | 不稳定 |

在 `q=0.775/0.800`，不稳定符号一直保持到 native `dt=0.1 ms`。对 history-grid 外推后：

| q | Re(lambda), dt→0 | frequency, dt→0 |
|---:|---:|---:|
| 0.775 | +0.00376 ms^-1 | 27.74 Hz |
| 0.800 | +0.00570 ms^-1 | 27.44 Hz |

因此旧“零延迟下 high fixed point 稳定/随后 Hopf”的口径撤回。更安全的解释是：
stationary branch geometry 仍有 saddle-node，但真实延迟另加一个 complex unstable mode；
有限 SNN 的 tonic plateau 可能围绕 high-rate skeleton 形成 nonlinear oscillatory/fluctuating
attractor。这里没有做 nonlinear limit-cycle continuation，所以不能把“可能”写成已经证明。
27 Hz reduced mode 也不等同于虚拟触点的 40–52 Hz ripple。

## 3. 三颗 SNN 轨迹是否被这个 fold 组织

输入严格限定为 frozen prospective family `tonic_b0_v2`，full realized edges、stationary
spatial OU、seeds 1841/1842/1843；不是从图上挑 seed。代表图使用 onset 位于三颗中位的
seed 1842。

| seed | onset (ms) | q_core 穿 fold→onset (ms) | q_mean 穿 fold→onset (ms) | late rE (Hz) | 对 q_min=0.775 high skeleton 的相对差 | late M 对 M* 的相对差 |
|---:|---:|---:|---:|---:|---:|---:|
| 1841 | 500 | 166.7 | 201.2 | 394.43 | 6.98% | 0.49% |
| 1842 | 480 | 283.9 | 242.3 | 393.87 | 6.83% | 0.28% |
| 1843 | 420 | 199.1 | 175.5 | 340.08 | 7.76% | 1.07% |

三颗 seed 都满足：`q_core` 和 `q_mean` 先穿过 reduced fold，操作性 onset 后到达与
high skeleton 同量级的平台；M 也紧贴 self-consistent `M*=tau_M*r_E`。这支持
**temporal ordering + state-scale agreement**。

它不证明因果：finite-size spiking fluctuation、stationary OU 和空间异质性都没有进入
reduced manifold；`q_core/q_mean` 也不是 uniform q。图上必须把 fold 写成 organizer，
不能写成 observed onset threshold。

## 4. 三种空间 reduction 的 branch identity 与 fold convergence

三种模型都从同一 realized patient graph 保守聚合，严格保存 E/I 总细胞数、四类 pathway
的一阶与二阶 moment、阈值人口加权均值。最大 conservative-total 相对误差为
`2.64×10^-15`。

### 4.1 同一条 high branch 可以跨网格认出来

在共享 anchors `q=0.775, 0.800, 0.840, 0.860`：

- mean E rate 跨网格最大相对差 `0.151%`；
- 映射到共同 60×60 grid 后的场 RMS 最大相对差 `1.071%`；
- 1.33 与 1 mm 最低 centered spatial correlation `0.804`，通过预设 `>=0.80` 门。

这支持三种 reduction 追踪的是同一 high branch，而不是碰巧找到三个无关的空间根。

### 4.2 fold 存在性稳健，fold state/rate 不稳健

| cell width | grid | q_fold | fold E rate |
|---:|---:|---:|---:|
| 2.00 mm | 10×10 | 0.869244 | 227.72 Hz |
| 1.33 mm | 15×15 | 0.893641 | 91.41 Hz |
| 1.00 mm | 20×20 | 0.890826 | 127.40 Hz |

三种网格都同时出现 pseudo-arclength tangent reversal 与 fixed-point Jacobian real-zero
crossing，且各自 `q` 定位差小于 `8.1×10^-12`。全部 `q_fold` span 为 `0.0244`；最细两
网格差 `0.00282`，通过预设 `0.005` 门。

但最细两网格的 fold rate 相差 `32.9%`，明显超过预设 `15%` 门；三网格 rate span 为
`136.31 Hz`。因此聚合状态必须保留为
`PATIENT_ZM_GRID_CONVERGENCE_HAS_FAILED_GATES`。不得只挑 q 的收敛而隐藏 rate 的失败。

## 5. 新 Fig. 5 的 panel 语义

当前 paper-facing candidate 严格按作者给出的静态 A–D 排版，不使用 GIF：

- **A**：seed 1842 的一条连续轨迹；上方是 global recruitment，下方是 15 个同时播放的
  virtual-SEEG tonic-level readout。回答“是否从低态进入持续全局高平台”。
- **B**：同一 seed 的 `q_core/q_mean/M/rE` 投到 1 mm patient-matched critical manifold。
  回答“动态轨迹是否先穿 fold 并靠近 high skeleton”。紫色高支必须标为 delay-unstable。
- **C-left**：同一轨迹中按规则选出的低态事件逐 E 神经元 first-spike order；
  **C-right**：固定 onset 后 100 ms 的 firing-energy 空间图。回答低态传播与 runaway 初段
  能量在哪里，而不是证明两个独立 anatomical cores。
- **D**：同一 seed、同一 weak probe、同一分层随机位置，比较低态与 early-runaway 的
  exact-resume probe-minus-sham mean response。因为 early-runaway sham 已在 tonic firing，
  高态 susceptibility 本身不可评价；右图只作 saturation response 描述。

Panel B 另有无 panel 字母的放大版，供正文缩图时核对折点和两条 q 轨迹。

## 6. P0 / P1 关键问题

### P0：如果主张 finite-SNN phase transition，当前证据仍不足

**为什么严重**：reduced fold 和 finite stochastic spatial SNN 不是同一个数学对象；动态
轨迹投影只能建立 organizer-level consistency。

**怎么处理**：正文固定用 “reduced saddle-node organizes the tonic transition” 或更弱；
不写 “the SNN undergoes a saddle-node phase transition”。

### P1：fold rate 未网格收敛

**为什么严重**：精确 fold state 依赖空间离散；只报 `q_fold` 会掩盖 91–228 Hz 的状态差。

**怎么处理**：图与 metadata 同时报告 q agreement 和 rate failure；若以后要把 fold state
当定量生物结果，需更细网格或基于全网络谱/非均匀 continuation 的独立验证。

### P1：high equilibrium 不是稳定吸引子

**为什么严重**：把延迟不稳定高支画成 stable branch 会直接错误解释 B 图的回折。

**怎么处理**：统一命名 `delay-unstable high-rate skeleton`；tonic plateau 只称 finite-SNN
model state，不把其吸引子类型写死。

## 7. 科学性与工程性审查

### 科学性

- 科学目的与实现一致：现在的问题是“tonic runaway 是否有 reduced dynamical organizer”，
  不是深调制 30–80 Hz 是否成立。
- 代表 seed 的选择规则冻结：三颗都过门，1842 仅因 onset 为中位数而用于展示。
- 延迟 closure 与 saddle Jacobian 在零频一致，避免用不同线性化定义拼图。
- 多网格 audit 预先保留失败门，没有把不收敛结果包装成成功。

### 工程性

- delay operator 逐 pathway 守恒一阶/二阶 moment，并记录源文件 hash；
- trajectory projection 绑定三颗原始 JSON/NPZ hash；
- paper figure metadata 绑定 trajectory、static exact-replay、projection、delay audit、grid
  audit、producer 与 renderer；
- PNG/PDF/SVG 同时输出，`figures/README.md` 逐图说明；
- 专项测试覆盖 delay moment/zero mode、dynamic ordering、grid gates、B 图双 q 语义和 Fig. 5
  source lineage。

## 8. 产出与复现入口

核心 artifacts：

- saddle validation：
  `/data/hfosp_topic4_fig45_artifacts/fig5/data_driven_zm_phase_diagram/deterministic_meanfield/patient_zm_saddle_node_validation_ngrid20.{json,npz}`；
- delay operator/stability/audit：同目录 `patient_coarse_delay_ngrid20.{json,npz}`、
  `patient_zm_delay_stability_selfvar_branchmatched_dt*.json`、
  `patient_zm_delay_stability_audit.json`；
- dynamic projection：
  `/data/hfosp_topic4_fig45_artifacts/fig5/data_driven_zm_phase_diagram/dynamic_projection/patient_zm_snn_manifold_projection.{json,npz}`；
- grid audit：deterministic 目录 `patient_zm_grid_convergence.{json,npz}`；
- 四联审计图：
  `/data/hfosp_topic4_fig45_artifacts/fig5/data_driven_zm_phase_diagram/figures/patient_zm_saddle_organizer_audit.{png,pdf,svg}`；
- paper-facing Fig. 5：
  `results/paper-ready-figure/fig5_spatial_zm_ou_tonic/figures/fig5-spatial-zm-ou-tonic-main-v2.{png,pdf,svg}`；
- Fig. 5B 放大版：同目录 `fig5-panel-b-zm-critical-manifold.{png,pdf,svg}`。

复现脚本：

- `scripts/build_topic4_patient_zm_delay_operator.py`；
- `scripts/run_topic4_patient_zm_delay_stability.py`；
- `scripts/audit_topic4_patient_zm_delay_stability.py`；
- `scripts/audit_topic4_patient_zm_dynamic_projection.py`；
- `scripts/audit_topic4_patient_zm_grid_convergence.py`；
- `scripts/plot_topic4_patient_zm_saddle_organizer.py`；
- `scripts/paper_figures/plot_fig5_spatial_zm_ou_tonic_paper_ready.py`。

## 9. 允许与禁止口径

**允许**：patient-matched reduced frozen-q fast subsystem 在 2、1.33、1 mm conservative
reductions 中均有 generic saddle-node；它在时间次序和状态尺度上组织了三颗 prospective
OU-SNN tonic-runaway trajectories。

**禁止**：finite SNN 已被证明发生 saddle-node/thermodynamic phase transition；
`q_fold=0.8908` 是精确 SNN onset；high branch 是 stable high fixed point；27 Hz reduced
unstable mode 就是触点 40–52 Hz；fold state/rate 已完成网格收敛；该 synthetic plateau
复现了临床发作或识别了患者机制。
