# Figure 5 候选：E1146 SNN state-dependent readout

> **状态（2026-07-19）**：Figure 5 candidate，尚未 LOCKED。
> **核心论点**：同一个患者特异性二维 SNN scaffold 在间期单次群体事件中给出触点先后次序，并在抑制资源耗竭进入 operational runaway 的早期读出同向空间能量梯度。
> **证据层级**：单模型、单 seed、单连续轨迹的机制可行性与 observation-layer bridge；不是临床发作复现、cohort 统计或完整发作机制证明。

## 1. 这张图在 Figure 5 中承担什么

Figure 4 已回答基础问题：患者特异性固定 scaffold 能否产生并被虚拟 SEEG 读出为稳定的正反间期传播。Figure 5 候选向前推进一步，但只推进到可被当前仿真诚实支持的层级：

1. 同一条连续轨迹中，runaway 前仍可见局部、短时、约 50 Hz 的间期样群体 burst；
2. 其中一段明确的单次事件可在完整 E1146 montage 上定义触点 recruitment order；
3. operational runaway onset 后、下一次外源 pulse 前的早期能量场，与该事件的 early-to-late 空间次序同向；
4. 因而模型支持 `same scaffold, different state` / `stable spatial axis + state-dependent recruitment` 这一动力学 bridge。

这张图不承担终止、恢复或完整 seizure cycle；也不把 `q_I` 宣称为患者真实发作的唯一细胞机制。

## 2. 三个显示块各自的 argument

### 上方：同一条连续 virtual-SEEG 轨迹

- 只画一条未经拼接的 0–1500 ms 连续记录。
- 显示 signed 30–80 Hz component，使约 50 Hz burst cycles 可直接看见。
- 每个 contact 按自身 runaway 前 absolute amplitude 的第 95 百分位定标；runaway 不参与该尺度，允许冲出纵轴。
- 只保留两个与下图一一对应的时间标记：左下单次事件的蓝灰色 window，以及右下 early-runaway energy 的浅红色 window。
- operational runaway onset 用红色虚线标出；不画 pre-runaway peak 点、zigzag 或传播连线。
- legend 单独占一行，不压在 trace 上。

**Argument**：下方两个场来自同一条连续动力学轨迹的两个明确状态窗，不是把两次独立仿真拼成故事。

### 左下：单次间期事件的 contact recruitment order

- 使用 transition-side source 在 runaway 前最后一个 qualifying local event；当前为 `TB`、535–620 ms。
- 这是单次事件，不是 TA/TB 多事件模板，也不是 contact variance。
- 对该 exact window 内每个 virtual contact 的 30–80 Hz burst-envelope peak latency 排序，得到 `1..N` recruitment rank；colorbar 最大值是参与 contact 数，不是毫秒。
- contact field 用 `viridis`，深色表示早、黄色表示晚。
- 彩色神经元颗粒来自同一 event 中真实 `E_spk_bool` 的 first-spike order；禁止把 contact 插值反采样成神经元活动。

**Argument**：一个具体的间期样群体事件在患者 montage 上给出可审计的 early-to-late 空间顺序。

### 右下：early-runaway energy field

- 时间窗从 operational runaway onset 开始，到 `onset+100 ms`、下一次 pulse 或记录末尾三者的最早者结束；当前为 1109.8–1209.8 ms，下一 pulse 为 1210.0 ms。
- 使用全部 finite contacts 的 mean-squared positive excess virtual-LFP energy。
- 这是静态时间窗平均，不画 rank、peak connector 或 zigzag。
- contact field 用 `Blues`，深蓝表示能量高；colorbar 保留 raw model energy，而不是只显示归一化 0–1。
- 彩色神经元颗粒来自同一 window 内真实 spike 计算的逐神经元 firing rate。

**Argument**：runaway 刚开始时，能量增强首先在此前间期事件较早读出的空间端更强。

## 3. 固定几何与视觉合同

- 下方两图完整复用 accepted E1146 registered plane、extent 和 15-contact montage：ICL 11 个、SCL 4 个；两个 panel 的触点位置与顺序必须完全一致，不按 TA/TB 翻转。
- 所有电极统一黑色外边框；每个 colorbar 紧贴自己的 field，两个 field 等大、等高。
- 模型薄片为 20 mm，display kernel 固定为 3.0 mm；不得机械照搬数据 Fig3-B 的 6 mm。
- Gaussian wash 只表示 contact readout，并使用连续 confidence fade；不得用硬 support 边界填满整张图。
- 两个 field 背景必须显示同一 run 的全部 8,000 个 E-neuron 真实位置，以保留真实模拟神经元的颗粒感；不得使用装饰性随机点。
- 不重复 mechanism panel，不标 A/B/C；总标题只写 `E1146`。标题简洁，字体按 paper-ready 标准放大。
- 当前静态候选不产 GIF；若以后需要动态版本，必须沿用同一事件窗、几何、contact order 和 field provenance，另立动态图合同。

## 4. 当前数值与选择纪律

- 15/15 contacts 在左图有 finite rank；4/4 SCL contacts 有 virtual-contact burst peak。
- all-contact earliness–early-runaway-energy Spearman = 0.814；ICL source-distance–rank Spearman = 0.764。两者只作单轨迹描述，不上升为 cohort inference。
- 左图同一事件中 1,954 个 E neurons 发放；early-runaway window 中 3,888 个 E neurons 发放。
- SCL 的 contact-level readout 为 4/4，但“contact 周围 1.5 mm 内至少 5% E neurons 发放”的 local-tissue gate 为 0/4。因此只允许写 `upper contacts participate in the group readout`，禁止写“SCL 下方局部组织直接被招募”。
- 低阈值区圆半径、椭圆横向尺度、`k_q`、AR 和 pulse phase 的审计 sweep 没有得到同时满足 SCL-local、方向保持、rank–energy 对应和 delayed runaway 的替代点。Figure 5 候选保留 1.5-mm baseline；不得为了让上方电极看起来“被覆盖”而事后挑参数。

## 5. Claim boundary

### 允许

- 同一固定 scaffold 可在不同动力学状态下产生不同 readout；
- 间期单事件较早的触点，在一条 q_I-depletion 轨迹的 early-runaway window 中能量更强；
- 该结果为数据端“间期传播场与发作早期能量梯度一致”提供模型侧可行性 bridge。

### 禁止

- 把 operational runaway 写成真实 seizure；
- 声称已复现发作终止、恢复或完整 seizure cycle；
- 把 sustained-rate onset 写成解析 separatrix `q_I*` crossing；
- 把 virtual-LFP excess energy 等同于 clinical broadband SEEG power；
- 把 contact peak 等同于其下方局部神经组织已经被直接招募；
- 把单 seed 的 Spearman 写成独立统计验证或机制因果证明。

## 6. Producer、artifact 与 canonical output

- computation producer：`scripts/run_topic4_m3_runaway_readout.py`
- computation artifact：`results/topic4_sef_hfo/early_recruitment_readout/m3_runaway_readout.{npz,json}`
- plotting-only producer：`scripts/paper_figures/plot_fig_topic4_early_recruitment_readout.py`
- Figure 5 candidate：`results/paper-ready-figure/fig5_snn_state_readout/figures/fig5_candidate_E1146_snn_state_readout.{png,pdf}`
- 同目录必须同时保存 metadata JSON 与中文 `README.md`；PNG/PDF 必须由同一代码和 artifact 状态生成。
- Topic 4 兼容诊断副本可保留在 `results/topic4_sef_hfo/early_recruitment_readout/figures/`，但不作为 paper-ready canonical path。

Figure 5 是否最终 LOCKED，后续由整张 Figure 5 的信息增量、跨 seed 稳健性和机制 ablation 共同裁决；候选身份本身不等于完整机制已验收。
