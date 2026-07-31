# Z/M Phase D fast-carrier repair：baseline-gate NO-GO

**日期：** 2026-07-31

**分支：** `codex/topic4-m4-snn-native-exit`

**科学状态：** `NO_GO_baseline_calibration_failed_zero_spike_dominance`

## 1. 一句话判断

Phase D 的新膜机制已经在正确的 per-neuron Z/M 空间 SNN 上实现并通过
状态迁移与默认路径一致性验证，但它没有保住原来的间期生成器：在注册范围内
最有利于 E 放电的 conductance 条件仍为严格零放电。因此本轮在 baseline
preservation 门终止，**没有进入 fast carrier，更没有建立可恢复 ictal
lifecycle**。

## 2. 这轮为什么做、加了什么

Phase C 已经证明，仅在既有 frozen `z/m/S_G` 坐标上移动工作点，得到的仍是
高率、低调制的 tonic continuation，而不是有界非-tonic carrier。Phase D
保持原始二维各向异性 SNN、E→E 图、核、外源输入、虚拟电极和真实 Z/M
不变，只在 fast inhibitory/membrane 一侧加入：

1. E 细胞的 conductance membrane；
2. `z_i` 对 local/weak-global GABA conductance 的一次性缩放；
3. `m_i` 作为 `E_K` reversal 的 sAHP conductance；
4. 只作用于 E 细胞的 60--160 ms spike-history dynamic threshold `phi_i`；
5. 与 conductance 方程一致的 synaptic-current observation proxy。

conductance E 膜的核心量为

\[
g_{\Sigma,i}=g_L+g^E_i+g^{I,\mathrm{eff}}_i+g_Mm_i,
\]

\[
V_{\infty,i}=\frac{g_LE_L+g^E_iE_E+
g^{I,\mathrm{eff}}_iE_I+g_Mm_iE_K}{g_{\Sigma,i}},
\qquad \tau_{\mathrm{eff},i}=C/g_{\Sigma,i}.
\]

它和旧 M4 分母不同：这里抑制同时改变膜固定点与有效膜时间常数，不是再给
recurrent E 乘一个经验除法器。E→E 在本线完全冻结，因此也与并行的 E→E
机制线独立。

## 3. 工程地基是否可信

### 3.1 正确衬底与状态迁移

五个真实 Phase-C checkpoint（pre-entry natural、bounded-mid
rising/peak、bounded-late rising/peak）均完成兼容状态迁移。迁移前后 Arm-A
继续运行 500 ms，E/I spikes、LFP、外源噪声、slow traces 和 final state
全部逐位一致。

- artifact：`results/topic4_sef_hfo/zm_fast_carrier_repair/armA_migration_equivalence.json`
- manifest SHA：`b777ca15a22b1f05f04af9e667f75a34eceaec49392ac5e83804cc3311bb56c0`
- `all_rows_exact=true`

所以本轮不是旧 `q_I/g_K` sandbox，也不是脱离空间 SNN 的纯 rate 模型；它是
在正确 Z/M 主线上做的 counterfactual fast-membrane fork。

### 3.2 默认路径保护

新路径全部 off-by-default。conductance 关闭时，旧 engine 分支保持原样；
历史 `BASELINE_SHA=da5fc18...` 的 byte-parity/engine guard 与 Phase-D 专项
测试均通过。`z`、`m` 和 `S_G` 不会被重复施加，conductance arm 也不会再用
旧 `S_G` 同时除 recurrent E。

## 4. 为什么 calibration contract 需要三次预结果修正

这些修订都发生在 B/C/D carrier outcome 打开之前，并且只让 baseline gate
更诚实，没有根据 carrier 结果调参。

1. **原 signed anchor 数学上不可用。** 锁定 pre-entry free-E 的 median
   `V=9.816 mV`，低于 `E_I=11 mV`，会给出负的 `kappa_I`；40.29% 细胞在
   `E_I` 以上，单一 signed operating point 不能代表宽电压分布。
2. **absolute-force anchor 漏掉旧 Arm-A 的 `S_G` effective E divisor。** 它虽
   给出正系数，却不能匹配真实有效 E/I charge budget。
3. **冻结 pre-entry slow state 不是间期 baseline。** 4 s frozen 诊断只有
   sub-1.5 Hz 小波动，并非源轨迹中的返回式 IED。最终行为 gate 因此改为从
   canonical `t=0`、动态 Z/M、完全相同噪声跑到 8.5 s；只有系数 anchor
   使用锁定 pre-entry snapshot。

最终 effective-charge anchor 的运行时系数为
`kappa_E=0.02917206399`、`kappa_I=0.01687781464`、
`g_M=1.76213078e-5`。反转电位、scale lattice
`{0.8,1.0,1.2}^3` 和 baseline acceptance 均未放宽。

## 5. 最终 baseline 实验

### 5.1 动态 reference

current-based Z/M reference 从 canonical `t=0` 跑 8.5 s，准确复现源锚点：

| 指标 | 结果 |
|---|---:|
| returning IED 数 | 15 |
| median duration | 75 ms |
| median core peak | 68.58 Hz |
| median all-E rate | 0.1031 Hz |
| mean all-E rate | 2.9781 Hz |
| peak active fraction | 0.2695 |
| runaway | 无 |
| median `V_inf` | 11.969 mV |
| effective I/E charge ratio | 0.6726 |

artifact：
`results/topic4_sef_hfo/zm_fast_carrier_repair/calibration/dynamic_preentry/reference__noise_replay.json`
（manifest `9694130022243585b4f1b880fe4f37f34b62992614047602905b20b0d7d6a474`）。

### 5.2 注册的 first-spike dominance stop

先完整运行最有利于放电的
`(s_E,s_I,s_M)=(1.2,{0.8,1.0,1.2},1.0)` 三格，每格 8.5 s：

| `s_I` | E spikes | returning events | peak active fraction | median `V_inf` | median `tau_eff` |
|---:|---:|---:|---:|---:|---:|
| 0.8 | 0 | 0 | 0 | 7.648 mV | 13.817 ms |
| 1.0 | 0 | 0 | 0 | 7.653 mV | 13.797 ms |
| 1.2 | 0 | 0 | 0 | 7.658 mV | 13.777 ms |

dominance 证明只覆盖“第一个 E spike 是否可达”：在第一个 E spike 前
`m=0`，所以 `s_M` 无效；固定 `s_I` 时 raw I input 与 z-sensor 轨迹相同；
在 `V<E_E` 时 `s_E=1.2` 是允许范围内最强兴奋；三个注册 `s_I` 已全部列举。
因此其余 24 格不可能先产生 E spike，完整 lattice 不再需要运行。

- machine verdict：
  `results/topic4_sef_hfo/zm_fast_carrier_repair/calibration/calibration_dominance_verdict.json`
- verdict SHA：`8ff95ff3247e148c261cee3cf610bce83fbf41e2c70b52fef4265d7cd6af8d20`
- final input manifest SHA：
  `1b389f9c9281a9c2c16a2682e042a26682229fba9c09793211b73ac14bcced46`

## 6. 核心图怎么读

`results/topic4_sef_hfo/zm_fast_carrier_repair/figures/phaseD_baseline_calibration_no_go.png`

- a：原始动态 Z/M reference 中的 15 个返回式间期事件；
- b：同一时间窗叠加三个 maximum-excitation conductance 臂，全部贴在零线上；
- c：reference 有 810,045 个 E spikes，而三个 conductance 臂均为 0；
- d：新膜方程把 median `V_inf` 从约 11.97 mV 拉到约 7.65 mV，同时把
  `tau_eff` 从 20 ms 缩短到约 13.8 ms。

这张图只表达 baseline prevention，不是 Figure 5，也不是 ictal carrier 或
lifecycle 图。

## 7. 当前能写与不能写

### 能写

- 正确 Z/M 空间 SNN 上的 off-by-default conductance/dynamic-threshold 地基已建成；
- 状态迁移与旧 current-based Arm-A 逐位一致；
- 在锁定的 reversals、effective-charge anchor 与 `[0.8,1.2]` scale 范围内，
  conductance replacement 无法保留原生间期事件，且由 zero-spike dominance
  在 baseline gate 得到 NO-GO；
- 主要失败是 fast membrane operating point 被压至不可放电，不是“载体形态
  不够漂亮”。

### 不能写

- 不能写 conductance inhibition 普遍失败；
- 不能写 B/C/D 没有 fast carrier，因为它们未获准运行；
- 不能写局部/全局 GABA 空间结构已被否证；
- 不能写 entry、offset、recovery 或 lifecycle；
- 不能把零放电称为 termination，也不能把诊断图作为 paper-ready Figure 5。

## 8. 对核心科学目标的意义

本轮最重要的反思是：**不能从 `t=0` 全局替换原生 Z/M 的间期膜定律，再指望
同一组参数同时保留返回式 IED 并生成新的 ictal carrier。** effective-charge
只在一个晚期 snapshot 上匹配，并不保证整条动态 pre-entry 轨迹等价；
reversal-based conductance 还改变了固定点与时间常数，直接消除了原模型赖以
产生 IED 的电压 excursions。

因此下一版不应在看过结果后扩大 scale lattice，也不应直接改 E→E。本线最快、
最可证伪的下一节点应是一个新的 **baseline-preserving state-dependent
homotopy**：

\[
C\dot V_i=(1-\lambda_i)I^{\mathrm{native}}_i
+\lambda_i I^{\mathrm{cond}}_i,
\qquad \lambda_i=\Lambda(1-z_i),
\]

其中 `lambda_i≈0` 覆盖已经验证的高-z irregular interictal 区域，使原生 Z/M
间期生成器逐位保留；只有当局部 `z_i` 穿过跑前锁死的耗竭窗口时，才平滑开启
conductance/local-global/dynamic-threshold fast feedback。这样新增机制只负责
“进入后形成有界非-tonic carrier”，而不是从基线开始重写整个膜方程。

新节点必须另写 Phase-E spec，并在看 carrier 前锁死：`Lambda` 的 z-window、
连续性、局部性、baseline exact-parity gate、无 E→E 改动，以及 carrier/return
判据。当前 Phase-D 结果不能被该新设计追认成成功。

## 9. 验证与资源边界

- 跑前 spec/plan 已被 input manifest 绑定，结果后保持逐位不改；执行状态以本
  archive 为准，不能回写锁定文档；
- Phase-D focused suite：75 passed；
- default-path/engine byte-parity suite：52 passed；
- adjudicator 与 plotter 可离线重建同一 verdict/figure；
- 本轮没有启动 B/C/D carrier worker；
- 其他 worktree 的运行进程仅做了只读 inventory，未终止、未改写；
- 结果目录受 gitignore 管理，tracked archive、spec/plan status、代码与测试共同
  提供可追踪入口。
