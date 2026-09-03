# 空间 Z/M 经验相图与患者匹配分叉分析（2026-09-03）

## 1. 审阅结论

**一句话判断**：现在找到了一个真正满足“分支折返 + 固定点 Jacobian 实零模穿零”的
**saddle-node bifurcation，但它属于 1 mm coarse patient-matched deterministic fast
subsystem**；持续 OU 打开的有限 SNN 仍只能报告经验状态边缘，不能叫热力学相变。

**完成度：78/100。** 已闭合真实邻接/阈值场的确定性 reduction、pseudo-arclength、零模和静态图；
仍缺 coarse-grid 收敛、delay-aware Hopf、OU colored-noise closure，以及经验 SNN 的三 seed 长驻留
双初值确认。因此这张图是机制分析图，不直接替换 paper Fig. 5。

## 2. 为什么之前一直没找到

1. 旧 frozen-q 网格从 `1.0` 跳到 `0.75`，没有测 `0.80–0.825` 的 OU-on SNN 经验边缘；
2. 旧验收要求深调制 30–80 Hz，主动排除了作者本轮接受的 near-saturated tonic plateau；
3. 只看去趋势后的触点频谱会抹掉高平台，只留下浅 ripple；
4. 旧 Fig. 5B 是单条时间轨迹中的 `(1-q, M)`，不是控制参数下的稳态分支；
5. 原 M3B 用理想 Gaussian kernel、统一阈值，并把 M 当外加场，不能为患者空间 SNN 指认 fold；
6. OU-on 工作点的低态本身由 colored noise 支撑。删除 OU 后，deterministic low root 只有约
   `0.03 Hz`，不能拿它与 SNN 的 `30–80 Hz` 低态做数值拟合。

## 3. 患者匹配 deterministic bridge

- frozen substrate：20 × 20 mm，32,000 E + 8,000 I，seed 1842；
- 使用缓存内**实际实现的** 25.6 M E→E、6.4 M E→I 及全部 inhibitory edges；
- 把 rise-jump 矩阵还原为 physical per-spike weight 后聚合到 `20 × 20` 网格（1 mm cell）；
- 每个网格的 E 阈值用 8 个经验分位块积分，不假设 Gaussian threshold distribution；
- row-sum 守恒复核：E→E `126.0`、I→E `121.338`、E→I `210.0`、I→I `189.0`；
- LIF transfer 用 16 点 Gauss–Legendre Siegert integral；单元测试相对 canonical quadrature
  误差在数值精度内；
- q 是 frozen fast-subsystem coordinate，只缩放 I→E；
- self-consistent M 使用 `m*=tau_m*r_E`，E input 减去 `eta_m*m*`；
- fixed-point Jacobian 包含均值与方差对 rate 的导数；pseudo-arclength 可穿过 fold。

模型归档：
`/data/hfosp_topic4_fig45_artifacts/fig5/data_driven_zm_phase_diagram/deterministic_meanfield/patient_coarse_ngrid20.npz`
（SHA256 `8060c9f75c8a1c926e2b2fc9ba0925ac00e84dd5c17778e8a4cc7ad9a33c27d5`）。

## 4. 分叉结果

| eta_m | q_fold | fold E rate | Jacobian 实零模 bracket |
|---:|---:|---:|---:|
| 0.00 | 0.890893859 | 127.542 Hz | +0.002580 → −0.002284 |
| 0.02 | 0.890825926 | 127.399 Hz | +0.000167 → −0.000320 |
| 0.04 | 0.890757823 | 127.255 Hz | +0.002926 → −0.001925 |
| 0.08 | 0.890622353 | 126.966 Hz | +0.003159 → −0.001679 |

`eta_m=0.02` 的 micro-continuation 与 generic-fold 审计满足：

- arclength tangent `dq/ds` 由 `+1.76e-4` 变为 `−3.37e-4`；
- fixed-point Jacobian 最近实特征值由 `+1.67e-4` 变为 `−3.20e-4`；
- 两种零点插值得到 `q=0.8908259261275` 与 `0.8908259261280`，仅差 `4.9e-13`；
- 最近的 corrected fixed point 残差为 `1.49e-14`，最近零模 `|lambda_1|=1.67e-4`，
  第二近模 `|lambda_2|=0.250`，谱间隔比约 `1499`，所以是 simple isolated zero mode；
- 规范化左右零模给出 `|w^T F_q|=26.80`、
  `|0.5 w^T F_xx[v,v]|=0.343`，后者跨五个方向差分步长的相对极差仅 `4.9e-7`；
- 在同一 `q=0.890700`，折点两侧 warm start 分别收敛到 `130.572 Hz` 与
  `124.459 Hz` 两个 fixed points；它们在 fold 合并。另有 near-silent low root，不属于这对合并根。

因此 **1 mm reduced model 的 generic saddle-node 已得到数值验证**，不是只凭曲线外观命名。
M 从 0 增到 0.08 只移动
`Δq_fold≈2.7×10^-4`，说明当前线性 M 不是 fold 位置的主要杠杆。

## 5. 稳定性与经验 SNN 的关系

零延迟、operating variance frozen 的 6/7-field Jacobian sensitivity 显示：

- low branch 在已测 `q=0.775–0.850` 上 `max Re(lambda)<0`；
- high branch 在 `q<=0.840` 为负，在 `q=0.845` 转正；线性插值的 Hopf locator 约
  `q=0.8440`（eta=0）/ `0.8439`（eta=0.02）；
- 这个复特征值变号发生在 saddle-node 之前。

这支持 reduced model 内存在低/高双稳定窗口的**敏感性证据**，但不能正式命名 Hopf，因为
传导 delay 未进入谱问题，且方差在动态 Jacobian 中冻结。

更重要的是，OU-on SNN 的经验边缘约在 `q=0.800–0.825`，与 OU-mean deterministic fold
`q≈0.891` 不重合。这个差距意味着 colored OU、有限网络、checkpoint basin 和亚稳驻留是实质部分，
不能把 deterministic fold 直接当作 Fig. 5 的 observed onset threshold。

## 6. P0 / P1 剩余问题

- **P0 coarse-grid 未收敛**：2 mm diagnostic 也出现零模 fold，但约在 `q=0.8693`、228 Hz；
  1.33 mm reduction 又出现额外 spatial branches。分叉存在性有重复迹象，但精确临界位置和分支
  身份未对网格收敛。下一轮应做 conservative operator coarsening 与 branch matching，而不是只比
  群体平均率。
- **P0 OU closure 缺失**：固定点只取 OU 的零均值；真实低态受 tau=20/150 ms colored noise
  支撑。必须加 colored-noise transfer 或 stochastic continuation sensitivity。
- **P1 delay-aware stability 缺失**：fold 的 lambda=0 条件不受 delay phase 影响，但 Hopf 类型受
  delay 强烈影响。
- **P1 empirical denominator 不足**：每个双初值点目前一个 future-noise seed，没有 3/3 robust
  `LOW/HIGH` 同点共存；`q=0.80625` 的长寿命 intermediate 仍可能是亚稳态。
- **P1 q 是 frozen coordinate**：这里证明的是 fast-subsystem critical manifold 的 fold，不是完整
  dynamic q/M system 以 `q_min` 为参数的自治分叉。

## 7. 最小下一步

1. 在 `q=0.80–0.825` 做三个 matched future-noise seeds、2.5 s 双初值驻留；只确认经验 basin，
   不再跑 348 条矩形盲网格；
2. 对 1/1.33/2 mm operator 做 branch identity matching，报告 fold 是否随网格收敛；
3. 把 homogeneous tau=150 ms OU 与 E-only spatial tau=20 ms OU 分别加入 transfer sensitivity；
4. 用 delay-aware characteristic equation复核 `q≈0.844` 的复模；
5. 只有 empirical `L/H` 三 seed 共存与 reduction 分支方向一致后，才把分叉 panel 升为 paper-facing。

## 8. 产出

- continuation JSON/NPZ：
  `/data/hfosp_topic4_fig45_artifacts/fig5/data_driven_zm_phase_diagram/deterministic_meanfield/patient_zm_bifurcation_ngrid20.{json,npz}`；
- 分析图 PNG/PDF/SVG + metadata：
  `/data/hfosp_topic4_fig45_artifacts/fig5/data_driven_zm_phase_diagram/figures/spatial_zm_phase_bifurcation_analysis.*`；
- 专项 saddle-node 验证 JSON/NPZ：
  `/data/hfosp_topic4_fig45_artifacts/fig5/data_driven_zm_phase_diagram/deterministic_meanfield/patient_zm_saddle_node_validation_ngrid20.{json,npz}`；
- 专项验证图 PNG/PDF/SVG + metadata：
  `/data/hfosp_topic4_fig45_artifacts/fig5/data_driven_zm_phase_diagram/figures/patient_zm_saddle_node_validation.*`；
- 图义：A=OU-on finite SNN endpoints；B=OU-mean deterministic fixed-point skeleton；
  C=`q × eta_m` fold locus；D=fold Jacobian real zero mode；
- 相关测试：60 项通过；PNG 目视通过；PDF 1 页、592.675 × 384.03 pt。

**允许口径**：患者匹配的 1 mm deterministic reduction 存在 saddle-node；OU-on SNN 存在
threshold-like tonic-runaway edge。

**禁止口径**：已经证明有限 SNN 的 phase transition；`q=0.8908` 就是 SNN onset；旧 Fig. 5B
的回折就是 saddle-node；M 是这个 fold 的主要驱动。
