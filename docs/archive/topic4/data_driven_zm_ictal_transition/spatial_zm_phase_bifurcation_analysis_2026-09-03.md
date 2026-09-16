# 空间 Z/M 经验相图与患者匹配分叉分析（2026-09-03）

## 结论

现在找到了满足 **pseudo-arclength 折返 + fixed-point Jacobian 实零模穿零** 的
saddle-node，但该结论严格属于 1 mm patient-matched deterministic frozen-q fast subsystem。
持续 OU 打开的有限 SNN 仍只支持 threshold-like tonic-runaway edge，不支持热力学 phase
transition。

## 两层证据必须分开

### OU-on finite SNN

matched future-noise 双初值 pilot（每点一个 seed）在 `eta_m=0.02` 得到：

- `q=0.860`: L/L；
- `q=0.840, 0.820`: L/I；
- `q=0.805`: I/I；
- `q=0.790, 0.770`: I/H。

q-only 的 `q=0.825` 为 L/I；high-start 在 `q=0.8125/0.80625` 仍为 intermediate。
因此经验形态是 low → broad intermediate/metastable band → tonic runaway，尚无 3-seed
robust L/H 同点共存。

### Patient-matched deterministic reduction

- frozen substrate：20 × 20 mm、32,000 E + 8,000 I、seed 1842；
- 使用缓存内实际 realized graph，恢复 physical weights 后聚合到 `20 × 20` 网格；
- 每个 1 mm cell 的 E 阈值以 8 个经验分位块积分；
- q 只缩放 I→E；M 自洽满足 `m*=tau_m*r_E`；
- fixed-point Jacobian 同时包含输入均值和方差对 rate 的导数。

| eta_m | q_fold | fold E rate | 实零模 bracket |
|---:|---:|---:|---:|
| 0.00 | 0.890893859 | 127.542 Hz | +0.002580 → −0.002284 |
| 0.02 | 0.890825926 | 127.399 Hz | +0.000167 → −0.000320 |
| 0.04 | 0.890757823 | 127.255 Hz | +0.002926 → −0.001925 |
| 0.08 | 0.890622353 | 126.966 Hz | +0.003159 → −0.001679 |

`eta_m=0.02` 的专项 generic-fold 审计进一步确认：`dq/ds` 与最近实特征值在同一 micro
bracket 反号；两种零点插值得到的 q 仅差 `4.9e-13`；最近 fixed-point residual 为
`1.49e-14`，`|lambda_1|=1.67e-4`、`|lambda_2|=0.250`；
`|w^T F_q|=26.80`、`|0.5 w^T F_xx[v,v]|=0.343`。在 `q=0.890700`，折点两侧
warm start 还分别得到 `130.572 Hz` 与 `124.459 Hz` 两个参与合并的根。因此这里的
saddle-node 不是只凭曲线外观命名。

M 从 0 增至 0.08 只移动 `Δq_fold≈2.7e-4`，说明当前 M 不是 fold 位置的主要杠杆。
零延迟/frozen-variance 动态 Jacobian sensitivity 中，low branch 保持负实部；high branch
在 `q<=0.840` 为负、`q=0.845` 转正，给出 `q≈0.844` 的 Hopf locator。由于传导 delay 未纳入，
Hopf 不作正式类型结论。

## 为什么 reduced fold 不能直接解释 SNN edge

真实 SNN 低态由 colored OU 支撑；OU 只取零均值后，deterministic low root 约 0.03 Hz，
而 SNN 低态为 30–80 Hz。SNN 经验边缘约 `q=0.800–0.825`，与 reduced fold `q≈0.891`
明显不重合。因此 colored noise、有限网络、checkpoint basin 和亚稳驻留是实质部分。

## 剩余 P0/P1

- P0：coarse-grid 未收敛。2 mm diagnostic fold 约在 `q=0.8693`、228 Hz；1.33 mm
  reduction 出现额外 spatial branches，不能报精确临界 q 已收敛。
- P0：缺 colored-OU transfer / stochastic continuation closure。
- P1：缺 delay-aware characteristic equation，不能确认 Hopf。
- P1：经验双初值仍缺 3 seeds × 2.5 s 长驻留。
- P1：q 是 frozen fast-subsystem coordinate，不是完整 dynamic q/M 系统的 q_min bifurcation。

## 产出

- bridge：`/data/hfosp_topic4_fig45_artifacts/fig5/data_driven_zm_phase_diagram/deterministic_meanfield/patient_coarse_ngrid20.{json,npz}`；
- continuation：同目录 `patient_zm_bifurcation_ngrid20.{json,npz}`；
- 综合图：`/data/hfosp_topic4_fig45_artifacts/fig5/data_driven_zm_phase_diagram/figures/spatial_zm_phase_bifurcation_analysis.{png,pdf,svg}`；
- 专项 saddle-node 审计：同目录上级 `deterministic_meanfield/patient_zm_saddle_node_validation_ngrid20.{json,npz}`；
- 专项验证图：同一 figures 目录 `patient_zm_saddle_node_validation.{png,pdf,svg}`；
- 图义：A=OU-on SNN endpoints；B=OU-mean fixed-point skeleton；C=`q × eta_m` fold locus；
  D=fold Jacobian real zero mode。

**允许口径**：1 mm patient-matched reduced fast subsystem has a saddle-node；OU-on SNN has a
threshold-like tonic-runaway edge。

**禁止口径**：finite SNN phase transition；`q=0.8908` 就是 SNN onset；旧 Fig. 5B 的回折
就是 saddle-node；M 是主要分叉杠杆。
