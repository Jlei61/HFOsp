# FCXR-LC6A：患者轴 inhibitory-surround capability probe

日期：2026-08-14

版本：rev2

状态：**DESIGN LOCK CANDIDATE — REVISION APPLIED；执行仍需明确授权**

对应 implementation plan：
`docs/superpowers/plans/2026-08-14-topic4-fcxr-lc6a-patient-axis-surround.md`

## 0. 一句话目标

在 E1146 几何、两个低阈值 core、既有患者轴 E→E 拓扑和全部细胞/慢变量参数不变时，只改变生物学
E→I（代码 `IE`）在患者轴方向的 reach，检验患者轴 inhibitory surround 是否能把 `Z/H` 自然进入后
的 escalating saturation 改造成非饱和、可停留、仍可响应的高态 carrier。

LC6A 是 **fast-substrate/carrying-capacity experiment**。它不要求 termination 或 recovery，不接入
`U/M/global tail`，也不要求高态必须空间移动。一个空间稳定但非 refractory、仍有扰动响应的高态，
是合法 carrier candidate。

## 1. 科学假设、备择机制与结论边界

当前 legacy substrate 的经验几何近似为：

```text
interictal state  <->  refractory-limited saturation
```

LC5v2.1 的 cell-local `U_i` 主要在“继续升级”和“进入被阻断”之间移动边界，没有创造可停留中间高支。
LC6A 的主假设是：沿患者传播轴缺少足够宽的 effective disynaptic inhibition，使 low-k/whole-sheet 和
local-rate runaway 一起升级；拓宽 E→I reach 可能打开有限增益的中间高支。

同时预注册相反的慢效应：更宽 E→I 会更早招募远处 I，可能在 firing wavefront 前造成更宽的抑制使用
与 `D=1-Z` 耗竭 halo，反而提前 onset 或加速 recruitment。因此本轮不预设“q 越大越稳定”，而同时
测量：

- 即时 surround/containment；
- onset latency 与 `N_IED_to_onset`；
- D halo 的宽度及其领先 firing front 的距离；
- recruitment speed；
- global-area 与 local-rate saturation。

本轮允许的结论是 canonical E1146 graph/noise 下的连接能力结果。即使五臂全阴性，也只能写：

```text
CANONICAL_SEED_AXIAL_REACH_FAMILY_NO_CARRIER
```

不得写成 Mexican hat、U、inhibitory surround 或患者机制的普遍 no-go。

## 2. LC5v2.1 右删失裁决：与 graph engineering 并行

执行前，上一轮 `tau_U=15 s, Gamma_U=0.003` 在约 23 s 才 onset，25 s 只观察高态约 2 s，属于
`RIGHT_CENSORED_CONTAINMENT_CANDIDATE`。执行后已按本节原合同从 exact state 续跑并在 27 s 达到
405.86 Hz，触发注册饱和线，终局改判 `ESCALATING_SATURATION`；无 offset。该结果授权 LC6A 五臂
自然轨迹，但不改变下列预注册权限映射。

本 spec 获得执行授权后，只允许从 25 s exact state 原参数继续到总时长 40 s：

- 膜/突触/delay ring、Z/H/U、RNG、future input、连接和参数全部连续；
- saturation 连续 1 s，或 autonomous offset 后完成至少 2 s 低态观察时可早停；
- 不补 `tau/Gamma`，不改 `p0/Phi/Imax`。

该续跑可与 LC6A graph module、tests 和 graph-only 构建并行，但它控制昂贵自然轨迹的权限：

| LC5 续跑结果 | LC6A 权限 |
|---|---|
| saturation、继续正漂移或仍无 stable carrier | 冻结 substrate-conditional negative；允许启动五臂自然轨迹 |
| stable non-saturated carrier | 暂停 LC6A 40k trajectory block；优先裁决 LC5 carrier/gain/更长结局 |
| autonomous offset | 暂停 LC6A 40k trajectory block；优先裁决 postictal/Z/returning IED |
| state/input/hash/numerical failure | `INSTRUMENT_BLOCKED` |

LC5 positive 不要求删除已经完成的 graph engineering，但暂停 LC6A 的全部 40k dynamics（short probe 与
自然轨迹）；只允许 graph-only audit 收尾。

## 3. 代码语义与冻结对象

### 3.1 target-first 命名

`C_ab` 表示 population b onto target population a：

| 代码名 | 生物学方向 | 当前 target in-degree |
|---|---|---:|
| `EE` | E→E | 800 |
| `IE` | E→I | 800 |
| `EI` | I→E | 200 |
| `II` | I→I | 200 |

LC6A 唯一允许改变代码 `IE`，即生物学 E→I。source/target 写反属于 graph-legality 硬失败。

### 3.2 冻结

- E1146 positions、sheet registration、source/sink centroid、患者轴；
- 两个 low-threshold core 和 Vth field；
- E→E adjacency/weight、患者轴 angle 与 AR=2；
- I→E、I→I adjacency/weight；
- 每类突触的 weight law、kinetics、`tau0+d/v`、delay quantization；
- 每个 target 的 in-degree；
- RC1 saturation、Z/H 方程和参数；
- `U=0, M=0, X=1`；
- tonic drive、dt、connection seed 1、noise seed 401；
- no kick、reset、parameter step、population sensor 或 runtime seizure detector。

距离变化会依照同一 conduction rule 改变 E→I delay。因此安全因果口径是“E→I reach 及其物理一致
delay 的联合影响”，不是脱离 delay 的纯距离效应。

### 3.3 blessed engine boundary

以下六文件逐字节冻结：

```text
src/snn_engine/kick_probe.py
src/snn_engine/params.py
src/snn_engine/model.py
src/snn_engine/connectivity.py
src/snn_engine/connectivity_rot.py
src/snn_engine/lfp.py
```

graph builder、two-hop audit、probe 和 runner 放在新非 blessed 模块。off/default path 必须还原 C0 graph。

## 4. 固定五图家族

当前 kernel 是 elliptical exponential。文档中的 `sigma_emp` 是实际 edge-displacement covariance width，
不是 sampler `l`。

对 source-target 位移定义：

\[
d_\parallel=(x_s-x_t)\cdot e_\parallel,
\qquad
d_\perp=(x_s-x_t)\cdot e_\perp.
\]

construction coordinate：

\[
q_\parallel^{marginal}=
\frac{
\sqrt{(\sigma_{E\to I,\parallel}^{emp})^2+
(\sigma_{I\to E,\parallel}^{emp})^2}
}{\sigma_{EE,\parallel}^{emp}}.
\]

它只用于构图，不等于实际 inhibitory loop。五个固定条件为：

| 条件 | 目标 `q_parallel^marginal` | 角色 |
|---|---:|---|
| C0 | legacy 实测值，预期约 0.66 | canonical graph |
| C1 | C0 值 ±0.05 | inhibitory-microstate control |
| Q1 | 1.00 ±0.05 | effective-width matching |
| Q2 | 1.25 ±0.05 | mild surround |
| Q3 | 1.50 ±0.05 | strong capability probe |

不再设置条件性 `q=1.375`。`0.623/0.766 mm` 仅可作为 Q2/Q3 proposal 初值；所有 graph target 在任何
SNN trajectory 揭盲前由 graph-only 结果冻结。

垂轴 kernel 不主动调参，relative width 变化作为 confound readout。若构图达到轴向目标时垂轴严重
漂移，必须显式报告，不能把结果包装成纯 axial effect。

## 5. Graph builder 与唯一合法性合同

### 5.1 Primary builder

从同一 C0 E→I graph 出发，在每个 I target 内做 source replacement/MCMC：

- 每个 I target 精确保留 800 个 E source；
- 无 duplicate edge；
- graph RNG 与 runtime external-input RNG 完全隔离；
- C1/Q1/Q2/Q3 使用同一 RNG family 与同一 maximum proposal budget；
- 每个 condition 依据自己的 q/energy/Hamming/edge-interior 稳定性收敛，不强制相同 accepted swap 数；
- proposal 必须证明对称；若不是，使用完整 Hastings correction；
- exact source-outdegree-preserving 2-edge swap 只可作为 sensitivity builder，不是唯一合法 builder。

### 5.2 Weight 与 delay 的唯一规则

对每个 I target，保留原 incoming weight array：

1. 若该 target 的 E→I weights 全相同，新 source list 直接复用该值；
2. 若不全相同，在 graph target 冻结前生成一个只依赖 `(graph_seed,target_id)` 的固定 permutation，把
   原 weight multiset 分配给按 canonical source-id 排序的新 source list；
3. weight 不依赖新距离，不重新拟合；
4. delay 只按新距离和冻结 `tau0+d/v` 重算。

因此 per-target incoming-weight multiset exact，reach 与 physical delay 改变。

### 5.3 Source out-degree distribution

原模型未固定每个 E source out-degree，本轮也不要求逐细胞 exact。相对 C0 要报告并尽量匹配：

- mean（由总边数构造保持）；
- CV、q95、q99：相对差各 `≤10%`；
- interior/edge mean ratio：相对差 `≤10%`；
- out-degree 对轴/垂轴位置的 Spearman：绝对差 `≤0.10`。

若 primary builder 超出这些容差，先改 graph builder；不得用 SNN 结果决定接受哪张图。

### 5.4 三类 hard gate 中的 graph legality

以下任一项失败才阻断对应 graph：

- population direction 错；
- target in-degree、weight assignment 或 delay rule 错；
- duplicate/corrupt edge；
- target q 在 common maximum budget 内不可达；
- graph/runtime RNG 串扰；
- graph artifact/hash/schema 不完整。

C1 baseline 改变、Q3 无 impulse zero crossing、entry blocked、静态 pattern 或低 gain 都不是 graph
legality failure。

## 6. Graph readout：marginal 与 actual two-hop geometry

实际两跳抑制算子是 target-first 记法下的：

\[
W_{E\leftarrow E}^{(2)}=W_{EI}\,W_{IE}.
\]

不构造 dense `N_E x N_E` 矩阵。对 interior E sources 抽样，或在冻结 coarse bins 上做 sparse product，
按 source-target 位移与两跳 delay 分箱：

\[
K^{(2)}(\Delta x_\parallel,\Delta x_\perp,\tau)=
\sum_{i\in I}w_{E_t\leftarrow i}w_{i\leftarrow E_s}
\delta(\Delta x-x_t+x_s)
\delta(\tau-\tau_{si}-\tau_{it}).
\]

五图全部报告：

1. marginal widths 与 `q_parallel^marginal`；
2. two-hop center/tail widths 与 `q_parallel^2hop`；
3. local mass（`|d_parallel|≤sigma_EE,parallel`）；
4. surround mass（`sigma_EE,parallel<|d_parallel|≤3 sigma_EE,parallel`）；
5. center/surround ratio；
6. two-hop latency median/q95；
7. forward/backward symmetry；
8. interior/edge difference；
9. shared-interneuron convergence/divergence statistics。

two-hop mass 使用 inhibitory contribution 的绝对幅值；带符号 sparse operator 另存，不能把负号在
center/surround ratio 中抵消掉。

`q_marginal` 是 construction coordinate；`q_2hop` 与 impulse response 才是 functional geometry readout。

## 7. Short functional probes：描述性 assay，不是准入门

在 `Z=1,H=0,U=M=0,X=1` baseline，对两个 E patch 做 paired sham/probe：

1. primary-core adjacent；
2. neutral patient-axis interior。

C0/Q2/Q3 做两位置；C1/Q1 至少做 neutral interior。probe 幅度只在 C0 上预锁到亚阈值、无 population
event，随后全部条件复用。按实际 two-hop latency q95 校验观察窗，并至少报告：

```text
0--50 ms
50--150 ms
150--300 ms
```

符号合同：从实际膜方程 RHS 分离 excitatory/inhibitory contribution，定义幅值
`F_E≥0, F_I≥0`，主净响应为：

\[
W_{net}=\Delta F_E-\Delta F_I.
\]

同时直接报告带符号的：

\[
\Delta I_{syn}=\Delta\{g_E(E_E-V)+g_I(E_I-V)\},
\]

以及 `Delta gE/Delta gI/Delta rE`，避免 inhibitory current 符号二次取反。

zero crossing、响应延迟、forward/backward asymmetry 均为 readout。baseline 没有 zero crossing 不能删除
自然轨迹，只能写“baseline functional Mexican-hat signature 不明显”。

LC6A 只做 response kernel、activity structure factor 和 descriptive low-k/finite-k spectrum；正式
operator/eigenmode identification 需要多 perturbation basis，移到 positive candidate 的独立分析。

## 8. 五臂固定自然轨迹

### 8.1 固定 block

只要五张 graph 都通过 legality，且 LC5 continuation 未产生 carrier/offset，必须完整运行：

```text
C0 / C1 / Q1 / Q2 / Q3
```

所有 arm：fresh `t=0`、graph 从第一步生效、Z/H dynamic、`U=M=0,X=1`、no kick/reset/step，使用同一
counter-based/pre-generated external input。不得因 C1 baseline 偏移、某档无 zero crossing、entry blocked
或上一档 saturation 删除其余 primary arm。

### 8.2 时间与右删失

- base horizon：50 s；
- onset 后必须继续 12 s；
- 若 50 s 无 onset，但 IED exposure 不足，继续到 hard cap 65 s；
- hard cap：65 s；cap 截断 onset+12 s 时标 `RIGHT_CENSORED_HIGH_STATE`；
- saturation 持续 1 s 可结束该 arm 并归类，但不停止其他 arm；
- NaN/Inf、OOM 或不可恢复 checkpoint 属 hard failure。

`ENTRY_BLOCKED_WITH_IED` 必须同时满足：

\[
N_{IED,arm}\ge\lceil1.5\,N_{IED,C0\ to\ onset}\rceil.
\]

若未 onset 且 exposure 不足，只能标 `ENTRY_UNRESOLVED_LOW_EXPOSURE`，不能写 entry blocked。

### 8.3 C0 parity

C0 graph、initial state 和 external input exact。若 runner 完全复用原 engine path，trajectory 要求 bitwise
parity；若 instrumentation/chunking 改变 reduction ordering，则要求事件、rate 和 state numerical parity，
并记录原因。不能用一个浮点 hash 取代分层 parity。

## 9. Global/local classifier 与连续 readout

LC5 global classifier 保留。为避免局部 carrier 被全局均值漏检，在 C0 揭盲后、Q1--Q3 结果聚合前，
按冻结 LC3/LC5 coarse grid 注册 companion classifier：

- 每 bin 100 ms rate threshold：C0 pre-onset returning-IED 分布的 q99.5；
- active component threshold：C0 IED 最大连通面积分布的 q99；
- local-high onset：超过上述面积且 q95 local rate 越线持续至少 500 ms；
- threshold、grid、connectivity rule 和 C0 source hash 写入 manifest addendum 后冻结。

五臂同时报告：

- global/local onset latency 与 occupancy；
- `N_IED_to_onset`、IED rate/IEI/duration/participation；
- D 每事件增量、D halo width、halo lead distance；
- global mean、local q95/q99 rate；
- active area、first-passage、recruitment speed；
- rate/H/D late Theil--Sen slope；
- E/I current decomposition；
- map persistence、centroid motion、envelope CV。

baseline 相对 C0 的变化是结果向量，不是 primary-arm stop gate。

## 10. Boundedness、stationarity 与 responsiveness 分层

### 10.1 Bounded carrier candidate

候选必须：

1. global 或 local classifier 给出 no-kick onset；
2. onset 后 high-state episode 持续至少 5 s；
3. 所有完整 1 s global mean rate `<250 Hz`；
4. high-state 全时段 `max_t f_ref(t)<5%`，并报告 `time_fraction[f_ref>5%]`；
5. 最后 2 s mean rate、H、D 的归一化 Theil--Sen slope 95% CI 上界均 `≤+0.05 s^-1`；
6. 不是 `<5 s` short after-discharge。

gain 不嵌入 boundedness 定义。空间图稳定也不自动失败。

### 10.2 Spatial phenotype

用 map correlation、centroid RMS 和 envelope CV 将候选标为 dynamic 或 stationary。只有：

```text
stationary + near-refractory/inert + no measurable fork response
```

才标 `INERT_STATIC_BUMP`。stationary 但非 refractory、可响应者标
`BOUNDED_STATIONARY_RESPONSIVE_CARRIER`，可以进入 LC6B 候选池。

### 10.3 Local saturation

local-rate runaway 不能只看末 2 s。报告整个 high-state observation 的：

- `max_t f_ref(t)`；
- `time_fraction[f_ref(t)>5%]`；
- local q95/q99 rate 的最大值和末段值；
- core 与非 core 的分别统计。

area bounded 但 local saturation/H-D 正漂移，标 `AREA_CONTAINED_LOCAL_RATE_RUNAWAY`。

## 11. 两个 phenotype 的 paired forks

完成五臂后，按结果向量选择最多两个不同 phenotype：

1. boundedness margin 最大者；
2. 最接近 saturation/block boundary 或 spatial dynamics 最不同者。

若只有一个有信息量 phenotype，只 fork 一个。每个 phenotype 在 onset+2 s 与 onset+6 s（存在且未
saturation 时）做 50 ms weak-patch paired fork：

- future external input exact shared；
- 2--3 个 exact duplicate checkpoint 验证 roundtrip determinism；duplicate 不为数值零先修 replay；
- 报告 500 ms susceptibility、relaxation time、rate/area deviation、是否 saturation/offset/diverge；
- 不用 20-checkpoint q99 numerical null；
- responsiveness 是独立字段，不覆盖 boundedness 标签。

## 12. 条件性最小确认

若 canonical graph/noise 出现 bounded responsive carrier，参数与 graph construction rule 冻结后，优先
生成第二个 graph realization B：

```text
same q target / weights / delay rule
different graph seed
noise seed 401
```

该确认通过才标 `CLEAN_CARRIER_POSITIVE`。若资源允许，再做 graph A/noise 402；它是次级确认，不允许
回调 q。单 realization 阳性保留为 `PROVISIONAL_SINGLE_GRAPH_CARRIER`，不是机制终局。

## 13. 唯一三类 hard gate

### H1 artifact/state integrity

state、RNG/input counter、source hash、schema 或 exact continuation 不一致。

### H2 graph legality

population direction、target degree、duplicate、weight/delay assignment、q target、graph hash 或 RNG
isolation 失败。

### H3 numerical/resource integrity

NaN/Inf、不可恢复 checkpoint、OOM 后状态不完整或注册数值稳定性失败。

除此之外，C1 偏移、无 zero crossing、entry blocked、static pattern、低 gain、baseline tradeoff、onset
提前或 D halo 加宽，全部是实验结果，不是删除其他 primary arm 的 gate。

## 14. 结果标签与下一分支

结果以向量和 headline label 同时输出：

| 结果 | 解释/后续 |
|---|---|
| clean bounded responsive carrier | `CLEAN_CARRIER_POSITIVE`；另立 LC6B，在新 carrier 上重标 U |
| single-graph carrier | `PROVISIONAL_SINGLE_GRAPH_CARRIER`；先完成 graph confirmation |
| carrier + baseline tradeoff | `CARRIER_WITH_BASELINE_TRADEOFF`；可科学阳性，但不能称 baseline preserved |
| stationary responsive carrier | 合法 LC6B candidate |
| area bounded/local rate runaway | 转 H source/transfer，不继续扩 q |
| broader reach accelerates D halo/onset | `SURROUND_ACCELERATES_D_DEPLETION`；重要机制阳性但非 carrier |
| five-arm canonical negative | `CANONICAL_SEED_AXIAL_REACH_FAMILY_NO_CARRIER` |

若固定宽核家族全阴性，第一 follow-up 是 center-preserving two-component E→I kernel：70--75% legacy
local + 25--30% wide axial。它优先于 uniform/global tail，但不在本 spec 授权范围内。

后续顺序锁定为：

```text
area-contained + local runaway -> H audit/redesign
sharp saturation/block across q -> small heterogeneity probe
carrier positive -> LC6B U recalibration
LC6B U still cannot terminate -> weak recruited-area/global E->I feedback
```

## 15. 产物与不授权事项

结果根：

```text
results/topic4_sef_hfo/mz_full_conductance_spatial_relay/
  lc6a_patient_axis_surround/
```

机器唯一参数源：

```text
config/topic4_fcxr_lc6a_patient_axis_surround.json
```

至少生成：

```text
graph_audit.json
two_hop_kernel_audit.json
impulse_response_audit.json
trajectory_summary.json
phenotype_map.json
gain_forks.json                     # 条件性
confirmation_summary.json           # 条件性
run_manifest.json
STATUS.md
figures/lc6a_graph_and_twohop.png
figures/lc6a_functional_response.png
figures/lc6a_trajectory_phenotypes.png
figures/lc6a_gain_forks.png          # 条件性
figures/README.md
```

本 spec 不授权 LC6B/U、M、global tail、center-preserving follow-up、H 修改、heterogeneity、全五臂
multi-seed 或正式 operator/eigenmode identification。termination 与 lifecycle 在最终报告中固定为
`NOT_TESTED`。

## 16. 2026-08-14 等价采样实现澄清

E→I 重连的非成员 proposal 定义始终是同一 perpendicular bin 内按核权重条件于“当前未选”抽样。
实现允许先做至多 64 次等价拒绝；若未命中，必须从显式条件分布直接抽样，防止 legacy 窄核 C1 因
已选边占据高概率质量而饥饿。该切换只改变计算路径和 RNG 消耗，不改变 proposal 分布或科学合同；
`conditional_fallback_draws`、抽样模式和 cap 必须写入 graph chain audit。

## 17. 2026-08-15 后续编号前向说明（LC6B 改名为 LC6D）

本 spec 的 §14 / §15 里出现的 "LC6B"（"carrier positive -> LC6B U recalibration"、
"另立 LC6B，在新 carrier 上重标 U"、"本 spec 不授权 LC6B/U"）指的是**在新 carrier 上重新标定 U**
这条条件性路线。该路线自 2026-08-15 起**改名为 `LC6D`**，内容、条件性与未授权状态都不变。

`LC6B` 这个编号已重新指派给 frozen-slow fast-subsystem causal atlas，见
`docs/superpowers/specs/2026-08-15-topic4-fcxr-lc6b-frozen-slow-causal-atlas-design.md`。
上文旧措辞保留不改，读到时按本节换算。
