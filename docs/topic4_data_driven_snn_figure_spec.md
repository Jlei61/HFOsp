# Topic 4 data-driven SNN 双图规范

> 状态：v1.1，2026-08-14。本文是 data-driven SNN 结果展示与验收的规范入口。
> 它锁定图型、语义和证据边界，不把当前候选升级为患者模式复现成功。
>
> v1.1 增补（见 §6.1）：图必须把该次运行的**已接受科学裁定**画在画布上，
> 不能只写进 metadata；`direction purity` 等不利限定符不得从图上撤下。

## 1. 规范对象

data-driven SNN 每一轮正式结果必须同时产出以下两张图，不能只给场、代表事件或综合距离：

1. `field + MTA/MTB spatial modes + same-network electrode readout`；
2. `KMeans event heatmap + rank distributions + cluster profiles + model-patient matrix`。

当前规范 producer：

```text
scripts/paper_figures/plot_fig4_spatial_edge_flow_validation.py
```

当前规范输入与候选：

```text
config/topic4_rev10_d5_2_spatial_ou_confirmation.json
candidate_id = spou_local_s010_ell038
```

当前规范输出：

```text
results/topic4_sef_hfo/data_driven_core_field_rev10_d/
  spatial_ou_accessibility_d5_2_confirmation/figures/
    fig4a_spatial_ou_direct_readout.{png,pdf}
    fig4a_spatial_ou_direct_readout_metadata.json
    fig4b_spatial_ou_kmeans_consistency.{png,pdf}
    fig4b_spatial_ou_kmeans_consistency_metadata.json
    README.md
```

后续候选可以更换输入和结果目录，但必须保持这两个输出语义，并在 metadata 中完整记录新输入、模式映射和文件哈希。

## 2. Figure A 合同

Figure A 回答三个问题：数据驱动场在哪里、两种冻结传播语义在场上怎样分布、同一网络中能否读出两种事件。

- 左侧只画一个连续 `h(x,y)` landscape，不加长标题；视角和植入几何必须复用 Figure 2A 合同。
- Figure 2A 的 20 个局部触点全部显示。进入模型 readout 的 15 个触点使用 ICL/SCL 语义色；SCL1-SCL5 等上下文触点用灰色，并贴合投影平面，不能悬浮成第三根虚构电极。
- 场使用 `plasma`；相机允许轻微倾角，但二维投影方向必须和 Figure 2A 一致。
- 中间只显示 `Model TA`、`Model TB`，内容为全部 formal clean events 的 onset density 和平均传播方向；不以单个星号代替事件分布。
- 右侧不写 `direct electrode readout` 标题。必须显示 Figure 1E 固定顺序下的全部 15 个 readout contacts，并从同一网络选择时间上分离的一次 MTA 和一次 MTB 事件。
- readout 为 contact firing-density envelope 的 30--80 Hz 带通结果，不是 current-LFP，也不是临床 SEEG 电压。事件 onset 可以逐触点标点，但不得用折线把两次或多个 onset 连成一条虚构波形。

## 3. Figure B 合同

Figure B 是 Figure A 的必需核验图，不是可选补图。布局固定为：

```text
KMeans event heatmap | rank distribution | cluster rank profile | model vs patient
```

- heatmap、masked cell、簇分隔、rank colorbar、固定电极顺序和 rank summary 必须调用 Figure 1E 已接受 painter；不得另写一套近似 renderer。
- KMeans 输入必须是 masked normalized event ranks。灰格表示该事件未招募触点，不能把 phantom rank 当成观测值。
- 三个 rank panel 必须共享同一个 y 轴顺序和位置。当前固定顺序由 patient-training rank 合同产生。
- 模型模式统一写 `MTA`、`MTB`；患者模板统一写 `TA`、`TB`。模型用实线，患者用虚线；TA 语义用红色，TB 语义用蓝色。
- cluster profile legend 放在 panel 内右上角。最右矩阵行固定为 `MTA, MTB`，列固定为 `TA, TB`；colorbar 与方形矩阵等高。
- 图和 metadata 必须同时报告 cluster counts、direction purity、KMeans stability、within-cluster consistency、KMeans-to-patient matrix 和 supervised-mode-to-patient matrix。

## 4. 模式语义合同

原始数值标签和 KMeans cluster id 都不是论文语义。每次生成图前必须先对 Figure 2 冻结 TA/TB template 做语义审计，再命名 MTA/MTB。

当前确认数据的审计结果为：

```text
numeric label 1 -> MTA
numeric label 0 -> MTB
```

这一映射不能硬编码成永久生物学方向。若新结果的 Figure 2 对照改变，producer 必须更新映射、metadata 和两张图；不能沿用旧名称掩盖模式翻转。

## 5. 当前 SNN 方程与机制边界

这批 data-driven 图使用的是 40,000-neuron spatial E-I current-based LIF SNN，而不是 Z/M 版本：

```text
N_E = 32,000
N_I = 8,000
tau_m dV_i/dt = -V_i + I_E,i - I_I,i
V_i >= V_th,i 时放电并 reset
V_th,i 使用 data-driven node field 提供的 per-neuron threshold
```

当前候选还加入局部、平移不变的外部 E-rate OU drive：

```text
sigma_rate = 0.1 / ms
tau = 20 ms
ell = 0.38 mm
```

本轮 edge coefficients 全为零，因此 edge mapper 是 exact no-op；topology、delays 和 E->E weights 均未改变。以下机制全部关闭：

```text
adaptation = off
inhibitory resource = off
E->E short-term depression = off
slow = None
```

仓库中的 Z/M 接口位于独立 slow-variable / MZ 机制分支，可实现 `I_net = I_E - z I_I - g_K` 等慢变量动力学；它们没有进入本轮 worker 调用。因此当前图、结果和说明不得标为 `Z/M SNN`。未来若接入 Z/M，必须生成独立 manifest、metadata 和结果目录，不能覆盖或重命名当前 LIF+field+spatial-OU 结果。

## 6. 双层验收

### 图形与工程验收

- 同一次 producer 调用生成 PNG、单页 PDF、metadata 和 `figures/README.md`；
- PNG/PDF 有 SHA256，内容在画布内，无文字遮挡；
- Figure 2A 几何、Figure 1E painter、contact order 和 MTA/MTB 映射都有可追溯来源；
- 绘图测试通过。

### 科学验收

图形通过不等于患者模式复现通过。必须同时检查：

1. supervised MTA/MTB 是否分别匹配 TA/TB；
2. natural KMeans 是否恢复相同的两类；
3. 两类是否在同一网络出现；
4. patient-matched direction purity、event support 和 network-level variability 是否达标。

当前 `spou_local_s010_ell038` 的 supervised matrix 具有正对角、负交叉，但 natural KMeans matrix 未恢复 TA/TB，尤其 MTB 对 TB 为负。因此当前安全结论是：

> 连续 data-driven node field 加局部 spatial OU drive 可以在同一网络产生可监督区分的 MTA/MTB 事件，但无监督事件结构尚未复现患者 TA/TB 双模式。

不能写成“data-driven SNN 已复现患者间期活动”“已恢复真实 core”“已证明传播机制”或“Z/M 模型复现”。

## 6.1 裁定必须上画布（v1.1，2026-08-14）

producer 打印的 `*_COMPLETE` 只说明绘图输入齐全，不是科学结论。因此：

- 两张图都必须在画布顶部打印该次运行 `confirmation_verdict.json` 的
  `fig4_acceptance` / `status`；若 `replication_pass=false`，必须同时写明
  “network-level replication rule NOT met”。metadata 增设 `science_status` 字段
  记录 verdict 路径、SHA256 与上述三项。
- Figure B 图下方必须常驻一行：`direction purity`（有 patient-matched q05 时并列）、
  `seed AMI`、`within-cluster tau`、supervised 矩阵对角、pooled 描述性矩阵对角，
  以及有 per-network 复制臂时的 equal-network 区间。**禁止**把不利限定符
  从图上移除只留 metadata（历史违规字段名 `visible_qualifier_removed`）。
- Figure B 最右矩阵是多种口径中最有利的一种；图注和 `figures/README.md`
  必须写明口径，并在 pooled 描述性矩阵第二模式与之不同号时点出。
- Figure A 的模式事件计数是跨网络 pooled 值，面板内必须标注 pooled 网络数。
- `figures/README.md` 的网络数、口径与科学状态一律由 verdict / config 生成，
  不得硬编码。

## 7. 本线当前状态（2026-08-14）

- §1 的 D5.2 目录仍是规范双图源，但科学状态是
  `REV10D5_2_DIRECTION_PROTOTYPES_RECOVERED_KMEANS_BELOW_PATIENT_BENCHMARK`
  （direction purity `0.674` < patient-matched q05 `0.884`）。
- D6.3 复制目录
  `results/topic4_sef_hfo/data_driven_core_field_rev10_d/continuous_field_kmeans_d6_3_fresh_replication/figures/`
  是同规格的第二套图，状态 `DIAGNOSTIC_ONLY` /
  `REV10D6_3_JOINT_CONTINUOUS_FIELD_NOT_REPLICATED`，12 张新网络未复制，
  **不得替换主文 Fig.4**。
- 整条 D4.1→D7 线的裁定见
  `docs/archive/topic4/sef_hfo/rev10_d4_1_to_d7_fig4_goal_closeout_2026-08-14.md`。

