# Group-Event State v0.3.2 — 模型侧设计（residual marked-history predictive state）

**日期：** 2026-09-02
**分支：** `codex/topic5-group-event-state-v032-model`（起点 `4c2ed958`）
**状态：** `DESIGN_LOCKED_FOR_IMPLEMENTATION`
**Goal：** 完成 v0.3.2 模型侧工作包（12 维受约束 marked leaky bank + 安全 residual adapter + 30 min NB residual trainer + 状态路径诊断 + synthetic 正/空对照 + 架构分诊 + 三患者 × 3 seeds 开发实验 + frozen state registry）。不改 v0.3.1 历史产物，不打开 sealed/formal 分区。

## 0. 一句话

在 Agent 2 给定的显式历史基线 `H_strong`（每个固定物理时间 anchor 的 `log μ_H`）之上，只学一个**残差**：`log μ_{H+S} = log μ_H + α·wᵀS`，其中 `S` 是由事件 token 写入、按真实时间指数衰减的 12 维状态。三条承重比较全部在同一批 anchor 上配对进行：
`H+S_correct` vs `H`；`H+S_correct` vs `H+S_shifted`；`H+S_dynamic` vs `H+mean(S_train)`。
状态只称 **residual marked-history predictive state**。

## 1. 数据与分区（沿用，不改）

- 事件流：`/data/hfosp_group_event_state_v0_1/dataset/<subject>/`（`SubjectSequence`）。
- 时间骨架：`load_subject_timeline` → seizure/gap 切出的 coverage segments、5 min anchor grid、horizons `(300, 1800, 7200)` s、`B_multiscale` 特征（仅用于 provisional H 与 R² 诊断）。
- nested 分区：`nested_time_partition` → calibration 0–20% / state_train 20–70% / dev_val 70–80% / dev_test 80–100%（累计有效记录时间）。dev_test 只作 development scoring，**不参与任何选择**。
- 三位冻结患者：`epilepsiae_1146`（27 segments，state_train 235 个 30 min anchors）、`yuquan_pengzihang`（5 segments，93）、`yuquan_zhangkexuan`（4 segments，106）。dev_val 30 min anchors 分别 50 / 16 / 19 —— 选择噪声大，是设计固有限制，如实报告 effective independent windows。

## 2. 事件 token `X_e`（primary 不含 raw waveform / 背景 SEEG / seizure label）

每个群体事件一行，全部来自事件时刻及之前可得的信息：

| 家族 | 列 |
|---|---|
| participation / tied-group | participation 向量 (C)；leader 向量 = tied_group_id==0 (C)；extent = n_part / n_valid；log1p(n_part)；n_groups/n_part；首组占比；最大组占比；平均组占比 |
| 精确 delay | 参与触点 relative delay 的 span / mean / std / median（秒） |
| 空间离散度 | 参与触点 MNI 坐标的平均两两距离、RMS 半径、首组质心到全体质心距离（mm）；单触点事件置 0 并加 flag |
| multiband | 每个 band：参与触点的 mean log_integrated_energy、mean log_peak_amplitude、max log_peak_amplitude、mean peak_time；跨频带 lag 的参与触点均值 (P=10) |
| detector confidence 代理 | ied_low 的 max log_peak_amplitude、mean log_integrated_energy；core_seconds；has_waveform |
| coverage / reference | 有效触点比例、有效触点数 |

- **Δt 不进入 `X_e`**，只用于衰减。
- 标准化：仅用 state_train 事件的均值/尺度（冻结为 buffer）；TRAIN 内零方差列置零并记录；NaN 在标准化后置 0。
- 特征矩阵按患者缓存到 `/data/hfosp_group_event_state_v0_3_2/model/features/<subject>.npz`，带 dataset `index.json` 与 session inventory 的 sha256 指纹。

## 3. 状态骨干

### 3.1 primary：12 维 marked leaky bank
- `φ_e = f_θ(X_e) ∈ R^4`，`f_θ` = `Linear(D,32) → GELU → Linear(32,4)`。
- `u_e = tanh(φ_e − mean_train(φ))`；`mean_train(φ)` 每个 optimizer step 由**当前 θ 在 state_train 事件上**重算并 detach，checkpoint 保存时冻结为 buffer，replay/导出均用该 buffer。
- τ ∈ {300, 1800, 7200} s，每个 τ 4 维，同一个 `u_e` 写入三个 τ 组：
  `S⁻_{τ,e} = exp(−Δt_e/τ)·S⁺_{τ,e−1}`，`S⁺_{τ,e} = S⁻_{τ,e} + u_e`。
- segment 起点 `S = 0`；不跨 gap/seizure/postictal 携带。
- **没有** state-to-state 混合、没有 LayerNorm、没有可学习 τ、状态模块 0 个可训练参数（审计项）。
- 轨迹用 chunked-cumsum 闭式精确计算（float64 内部、float32 输出），对 `u` 的梯度**不截断**（120 min 尺度的 credit assignment 完整）。TBPTT 的 chunk-detach 只作为审计开关存在。
- anchor 状态：`S_a = exp(−(t_a − t_last)/τ)·S⁺_last`（同 segment 内最后一个严格早于 anchor 的事件；没有则 0）。
- **读出前的 TRAIN-only 固定均值/尺度（实现后补记，2026-09-02 晚）**：τ=2 h 的积分器持有 rate×τ ≈ 10²–10³ 次事件写入，原始 `S` 量级悬殊，未标准化的 `α·wᵀS` 读出条件极差（toy 上训练损失在 70 步后单调回升）。因此读出使用 `S̃ = (S − mean_train)/scale_train`，统计量取自 state_train anchors。训练前向中该统计量（以及 φ 的 TRAIN 均值）是当前全批次 TRAIN 集的**可微**函数（等价于对整个 TRAIN 集做一次确定性 BatchNorm；detach 版本会让梯度与目标不一致，toy 上表现为损失回升），评估/回放/导出一律使用冻结进 checkpoint 的 TRAIN buffer。它不是 per-time LayerNorm：不按样本归一化，不改变各 τ 的名义时间常数。

### 3.2 triage 对照：repaired RNN
同一 `X_e`、同一标准化、同一 τ 衰减、同一 residual adapter、同一 NB loss 与优化器分组，仅把写入换成旧 v0.3 的 gated 更新：`[g, c] = U_θ([S⁻, e_e])`，`S⁺ = S⁻ + σ(g)·frac(τ)·(tanh(c) − S⁻)`，`e_e ∈ R^16` 来自同族 MLP。修复项：去掉所有 LayerNorm、TRAIN-only 固定标准化、H 上的 residual 读出、α 非零初始化。逐事件顺序计算（segments 并行 padding），梯度同样不截断。

## 4. 读出与似然

- `log μ_{H+S,a} = log μ_{H,a} + α·wᵀS_a`。`w`: `nn.Linear(12,1,bias=False)` 默认随机初始化；`α` 初值 0.03，前 50 个 optimizer steps 冻结；**无自由截距**（截距效应由 `H+mean(S_train)` 臂显式对照）。
- NB：`y ~ NB(μ, r)`，`log r` 可学习标量（矩估计初始化），FP32 计算。
- `H` 单独臂：`log μ_H` 原样 + 在 state_train anchors 上 1 维 MLE 拟合的 `r_H`（若 Agent 2 registry 自带 NB 参数则优先用其值）。
- 优化器分组（AdamW）：`encoder_weights`(wd) / `encoder_bias`(no wd) / `state`(RNN 才非空; leaky bank 为空) / `state_bias`(no wd) / `adapter_w`(wd) / `adapter_gate_alpha`(no wd) / `adapter_dispersion`(no wd)。标准化统计为 buffer，不进任何组。
- 全批次训练（每步覆盖所有 state_train anchors），global grad-clip 1.0（记录裁剪前后），每 10 步在 dev_val 上评估，patience 10 次评估，最多 600 步，最少 100 步；记录 `selected_step`、`selected_first_validation`、`selected_at_budget_edge`。
- AMP：只允许包在 encoder MLP 上（bf16 autocast）；state 衰减、NB、reduction 全 FP32。默认关闭，诊断中做 AMP vs FP32 梯度比较。

## 5. 评估臂（dev_val 选择、dev_test 只评一次）

同一批 anchor、同一 horizon（primary 1800 s；300 s 诊断；7200 s 仅当 `endpoint_eligibility.json` 事前判 eligible）：
1. `H`；2. `H+S_correct`；3. `H+S_shifted`（segment 内 block-circular 半段移位，donor 与本 anchor 至少相隔一个 horizon；副本：1/4、3/4 移位）；4. `H+mean(S_train)`；5. `H+S_random`（同结构、encoder 随机冻结、只训 adapter 的 random reservoir）。
对比量为逐 anchor 配对 NB NLL 差（nats/anchor），报告均值、segment 内 moving-block bootstrap CI、favourable segments 数、seeds 中位数、effective independent windows。

**臂定义的两点后果（实现后补记）**：(i) 因为读出使用 TRAIN 中心化的 `S̃` 且没有自由截距，`H+mean(S_train)` 的调制恒为 0，该臂 = `log μ_H` 配以 H+S 模型学到的 dispersion；因此 `mean − correct` 度量的是"固定 dispersion 下动态状态的贡献"，`H − mean` 度量的是 dispersion 差异。(ii) 为了仍能量化"只做截距重校准能赚多少"，增加辅助臂 `H + c`（`c` 在 state_train anchors 上用 `r_H` 做一维 NB MLE），它对应非中心化设计里会被 `α·wᵀmean(S)` 吸收的份额。

## 6. Synthetic assays（真实时刻 + coverage + event token）

- 隐藏分量：固定随机 `g(X_e) = tanh(W_g X_e + b_g) ∈ R^4`，τ_g = 1800 s 的 leaky 轨迹，anchor 处随机线性组合后按 TRAIN 标准化得 `z_a`。
- positive：`log μ = log μ_H + 0.35·z_a`；null：`log μ = log μ_H`；`y ~ NB(μ, r=5)`。
- 预设通过标准：positive 在 ≥2/3 replicates 上 dev_test `Δ(H − H+S_correct)` 的 CI95 下界 > 0 且 `Δ(shifted − correct)` 均值 > 0；null 6 个 replicates 中 dev_test 上 CI95 下界 > 0 的不超过 1 个，且中位 Δ < 0.01 nats/anchor。
- synthetic 未过 → 修实现/优化，不扩人、不作生物学解释。

## 7. 诊断（写入三份机器 JSON）

`state_gradient_audit.json`：optimizer 参数覆盖；单 loss backward（30 min NB / continue-size probe / subset probe，后两者为线性梯度路径探针，不是 H2a 结果）；module-wise grad norm；clip 前后；AMP vs FP32。
`detach_replay_audit.json`：primary 路径无 detach、∂S_a/∂u_j = exp(−Δ/τ) 数值验证、TBPTT-detach 对照差异；checkpoint-specific replay 与训练期轨迹一致；stale trajectory 结构性排除（每步重算 + 参数 hash）。
`state_functional_diagnostics.json`：single-segment overfit；dynamic / TRAIN-mean / random S 输出差异；adapter logit RMS；‖α w‖ Jacobian；temporal variance + effective rank；event-write RMS vs autonomous-decay RMS；trained vs init 轨迹变化；S → H 线性重建 R²。

## 8. 与 Agent 2 的接口

**实际接口（2026-09-02 17:23 registry 出现后确认）**：Agent 2 的分区是 base_fit 0–60% / inner_val 60–70% / dev_val 70–80% / dev_test 80–100%（0.7、0.8 两个边界与本侧 nested 分区完全重合）；`H_strong` 是 NB2 ridge GLM（base_fit 拟合、inner_val 选 ridge、0–70% refit），registry 为 `patients[subject].horizons["1800"/"300"/"7200"].arrays` → npz(`anchor_time`, `log_mu_h`, `count`, `eligible`, `anchor_phase`)，另有 `nb_log_dispersion = log r`。三位患者的 anchor 网格与本侧逐点相同（673/221/248）。Agent 2 会把本侧 `anchor_state` 作为特征列并入其 GLM 重新拟合各臂，因此它消费的是冻结状态数组而不是本侧的 `α,w`。本侧 state_train（20–70%）与其 inner_val（60–70%）重叠：只影响其对 S 臂 ridge 的选择（偏保守方向），不触及 dev_val/dev_test 评分；registry 元数据中显式写出 `state_train_recorded_fraction=[0.2,0.7]`。

**事前 eligibility（Agent 2 冻结）**：30 min 主终点仅 `epilepsiae_1146` 合格；`yuquan_pengzihang`、`yuquan_zhangkexuan` 不合格（block 数不足）；120 min 三人均不合格；5 min 三人只能称短程。三人仍全部运行（用户要求），所有汇总标注资格。

- 读：`/data/hfosp_group_event_state_v0_3_2/shared/history_baseline_registry.json`、`endpoint_eligibility.json`。读取器只做对齐校验（anchor 时刻/分区一致、无 dev_test 拟合声明），不改 H 与 eligibility 定义。
- 缺席时：先完成模型、synthetic、单测、schema；后台定时轮询。synthetic/单测/single-segment overfit 使用 **provisional local H**（`B_multiscale` 去 seizure 列后的 NB ridge，state_train 拟合、dev_val 选 ridge），所有产物打 `h_source=provisional_local` 标记；三患者正式开发实验要求 `h_source=agent2_registry`，若最终仍缺席则以 provisional 运行并在报告里明确“待 Agent 2 H 替换”。
- 写：`/data/hfosp_group_event_state_v0_3_2/shared/frozen_state_registry.json`（原子写），每条含 subject、seed、architecture、checkpoint/config hash、per-event `state_pre/state_post`、per-anchor state、train-mean state、split/segment/session/physical epoch、coverage、source artifact fingerprint、`h_source`。

## 9. 运行与产物

- python：`/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python`（torch 2.5.1+cu124）；`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`。
- 输出根：`/data/hfosp_group_event_state_v0_3_2/model/`（features/ synthetic/ runs/<arch>/<subject>/seed_<seed>/ diagnostics/ triage/ summary/ logs/ task_manifest.json STATUS.json）。
- 任务队列：`task_manifest.json` + 每 GPU 一个 worker（`setsid nohup`），hash 一致的完成任务跳过，checkpoint/manifest 原子写，STATUS.json ≤10 min 刷新，断点续跑。
- 顺序：features → synthetic（leaky bank 为主，RNN 各 1 个 replicate）→ triage（1146 + zhangkexuan × 1 seed × 2 架构）→ 锁架构 → 三患者 × 3 seeds → diagnostics → export registry → summary → 报告。

## 10. 明确假设（用户未指定处）

1. segment 起点状态为 0；leaky bank 无可学参数。
2. `mean_train(φ)` 每步重算（TRAIN-only、detach），checkpoint 冻结。
3. 无自由截距；截距效应由 `H+mean(S_train)` 臂承载。
4. random reservoir = 随机冻结 encoder + 训练 adapter。
5. continue/size、subset 探针 = 线性梯度路径探针（Poisson extent、Bernoulli participation），仅证明梯度可达。
6. Agent 2 registry schema 未知，读取器容错并在到达后按实际 schema 适配；不自建 H。
