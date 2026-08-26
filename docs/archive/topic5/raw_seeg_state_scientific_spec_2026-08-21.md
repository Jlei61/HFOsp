# Raw-SEEG 可演化预测状态模型 — 科学合同 (R0.1)

**建立日期**：2026-08-21
**修订号**：R0.1
**结果根目录**：`results/epi_prssm/raw_seeg_state/r0_1/`
**冻结常量**：`src/topic5_raw_seeg_state/contract.py`（主 agent 独占，worker 不得修改）
**执行计划**：`docs/archive/topic5/raw_seeg_state_execution_plan_2026-08-21.md`

---

## 0. 一段朴素话：我们在测什么

我们把颅内电极连续录到的原始电信号（不做任何事件挑选、不告诉模型哪里是病灶、
哪里发过作）喂给一个模型，让它自己压缩出一小串数字（32 个），我们叫它"状态"。

然后我们**切断输入**：从某一刻起不再让模型看任何新的脑电，只让它拿着那一刻的
32 个数字，靠一条固定的衰减+旋转规则往前推，去猜 1 分钟、5 分钟、10 分钟、
100 分钟之后**每一根触点、每一个频段的能量会是多少**。

如果它猜得比三种"什么都不懂"的对照更好，就说明连续脑电里确实存在一个可以
**脱离输入自行往前走**的低维信息。三种对照分别是：

1. 就说"未来跟这个人平时的平均水平一样"（均值）；
2. 就说"未来跟此刻一模一样"（持续）；
3. 用一个极小的线性模型，只看过去 10 分钟的频谱数字直接外推（低容量频谱自回归）。

这一版**只**建立这个"状态骨架"。它不学棘波、不学发作、不学致痫区，也不学
物理量更新。那些是后面 E0.4–E1.0 的事。

---

## 1. 科学问题

从连续预处理 SEEG 中推断低维状态 $z_t \in \mathbb{R}^{32}$，并检验：**在停止读取
未来 SEEG 之后**，$z_t$ 能否开放环（open-loop）预测未来 1 / 5 / 10 / 100 分钟的
多触点频谱场。

## 2. R0.1 允许与不允许的结论

### 2.1 R0.1 能支持的最强结论

> 连续 raw SEEG 中存在可开放环预测未来多触点频谱场的低维信息，其可预测时间尺度
> 由 1、5、10、100 分钟的结果决定。

### 2.2 R0.1 不得声称（写作红线）

- $z_t$ 是癫痫易感性；
- $z_t$ 能预测发作；
- IED 塑造了 $z_t$；
- 某个 latent 维度是兴奋性 / 抑制性；
- 长 horizon 增益必然来自癫痫机制。

### 2.3 分层解释规则（预注册）

| 观察到的情形 | 允许的措辞 |
|---|---|
| 仅 1 分钟胜过全部基线 | "主要是局部（分钟级）可预测性" |
| 5–10 分钟仍胜 | "支持中等时间尺度状态" |
| 100 分钟仍胜过 persistence + mean + feature-AR | "支持长时间尺度预测状态" |
| 100 分钟失败 | **不否定** 1–10 分钟状态 |
| 100 分钟成功 | **不得**直接称癫痫易感状态 |

**不设置"一项失败则全部停止"的总 gate。** 每个 horizon、每位患者独立报告。

### 2.4 forecast 与 consistency 的表述分层（§CLAUDE.md 6.3 pronoun discipline）

两层必须分开写，禁止合并成一句 "R0.1 PASS"：

- **预测层**：解码出来的未来频谱场误差是否低于基线。
- **一致性层**：编码器在 $t{+}1$ 独立编出来的状态，是否等于从 $t$ 推一步得到的状态
  （$E_{\mathrm{cons}}$，定义见 §7）。

**若预测层为正而一致性层失败，只能称 "forecastable latent code"，不得称
"统一的可演化状态"。**

---

## 3. 队列与分区

34 位患者 = 18 Epilepsiae + 16 Yuquan，主体名单继承 Epi-PRSSM v0.1
`SPLIT_MANIFEST.json`（34 subjects，chronological 60/20/20，test 已 SEALED）。

R0.1 **完全继承**该 split 的墙钟时间边界：

- `train`  : $t <$ `boundaries.train.last_epoch`
- `validation` : `train.last_epoch` $\le t <$ `boundaries.validation.last_epoch`
- `sealed` : $t \ge$ `boundaries.validation.last_epoch` — **本修订不得读取**

唯一的读取入口是 `contract.dev_end_epoch(subject)`；任何时间戳数组在写入产物前
必须过 `contract.assert_not_sealed()`。取 `validation.last_epoch` 而不是
`test.first_epoch` 作为封条，是刻意保守（封条比上游 test 分区真正起点更早）。

所有 normalization（每触点每频段的均值/标准差、artifact 阈值、robust scale）
**只允许**用 train 时段估计。

---

## 4. 数据合同

### 4.1 采样率审计与频率上限冻结

队列内实测原生采样率：

| 数据集 | 采样率 | 主体 |
|---|---|---|
| Yuquan | 2000 Hz | 全部 16 位 |
| Epilepsiae | 1024 Hz | 14 位 |
| Epilepsiae | 512 Hz | 253（及 139 的一部分 recording） |
| Epilepsiae | 256 Hz | 384、583（部分 recording）、139（部分 recording） |

队列 Nyquist 下限 = **128 Hz**（由 256 Hz recording 决定）。为保持全队列可比，
冻结：

- **公共分析采样率 `ANALYSIS_RATE_HZ = 256`**
- **目标频段 1–100 Hz，12 个 log 等分 bin**（上限比 128 Hz 留 22 % 抗混叠余量）

> **明确边界**：R0.1 对 >100 Hz（ripple / HFO 带）的可预测性**没有任何发言权**。
>
> 逐 block 审计后修正（2026-08-21）：384 与 583 确有 256 Hz recording，但那些 block
> **全部落在封条之后的 sealed 尾段**，其 dev 分区只有 1024 Hz。因此
> **dev 分区内真正受采样率限制的只有 epilepsiae_139 一位**（256 与 512 Hz 混合），
> 在 `eligibility_summary.csv` 中以 `nyquist_limited=True` 标出；最高 bin
> （68–100 Hz）在该患者上可能被采集端抗混叠滤波额外衰减，解释时须并列这一点。
> 频段上限**仍然锁在 100 Hz**：队列可比性要求单一口径，且 139 本身就需要它。
> 各 block 的完整原生采样率保留在 `data_audit.json::native_rates_all_blocks`，
> 供未来 R0.2 对 ≥512 Hz 子集扩展频段时使用。

### 4.2 参考与蒙太奇

**Bipolar within shaft**（同一根杆相邻触点相减），触点坐标取两端中点。

理由：两个数据集的原始信号都记录在共同参考（scalp / Ref）之下。共同参考会给
所有触点注入同一个全局分量，使"预测整个触点场"变得平凡地容易，而这跟脑状态无关。
Bipolar 去掉这个混淆。**代价**：损失真正的全局同相成分；作为后续敏感性可以补
common-average reference 版本，本轮不做。

### 4.3 时间轴与 eligibility

- 时间轴真值：Epilepsiae 用 SQL 导出的 `block.begin/end`（已冻结在
  `results/epilepsiae_block_inventory.csv`）；Yuquan 用
  `results/dataset_inventory/yuquan_block_inventory.csv`。
  **9 位 Yuquan 患者在 inventory 中无行**（chengshuai / chenziyang / hanyuxuan /
  liyouran / songzishuo / wangyiyang / zhangbichen / zhangjiaqi / zhaochenxi），
  其 block 区间必须由 EDF 固定头重建（沿用 Epi-PRSSM v0.1 的
  `recorded_coverage_rule`），并在 manifest 中记 `source_kind=edf_header`。
- **分钟网格**：以每位患者第一个 block 起点为原点，每 60 s 一格。
- **覆盖**：一分钟内 ≥95 % 时间落在 block 内才算 `covered`；否则该分钟不可用。
  未记录时间**不得**当作无事件或稳定背景。
- **session**：gap ≤ 300 s 不开新 session（沿用 Epi-PRSSM v0.1
  `session_join_seconds=300`）；gap > 300 s 断开 session。
- **发作护栏**：EEG onset 前 3600 s、EEG offset 后 3600 s 及发作本身，全部从
  backbone 训练池移除。这同时满足"不得跨越 seizure onset"和"held-out seizure 的
  preictal raw windows 不得进入 backbone 训练"。
- **artifact**：每触点每分钟，宽带 log 功率偏离该触点 train 中位数 > 6 robust SD
  （median / 1.4826·MAD），或 >1 % 采样点触到量化上下轨，判为 artifact，该
  触点-分钟在损失中被 mask；一分钟内存活触点 <70 % 则整分钟不可用。

### 4.4 一个 (context, target) 对的合法性（全部条件同时成立）

1. context 的 10 个分钟全部 `minute_usable`；
2. target 分钟 $t{+}h$ `minute_usable`；
3. context 起点与 target 同属一个 session；
4. 区间 $[t-10\min,\; t+h]$ 内**没有任何**发作护栏分钟；
5. context 起点与 target 同属一个 split 分区（train 或 validation）；
6. 全部时间 $<$ `dev_end_epoch`。

**允许**：$[t, t+h]$ 中间的分钟可以未被记录（≤300 s 的微 gap），因为开放环预测
用的是绝对墙钟时间 $h$，中间是否观测到与主张无关。这一放宽必须在报告里写明。

### 4.4b 坐标可用性与两条独立的有效性轴（2026-08-21 裁定）

全盘核查结果：Yuquan 只有 15 位患者在 `patients_elecs_reGen/` 下有导出的电极坐标
（`chnXyzDict.npy`）；**chenziyang / gaolan / hanyuxuan / sunyuanxin / wangyiyang
这 5 位在整个挂载盘上没有任何导出坐标**，只有原始 `yuquan_images/MRIandCT/`。
从 MRI/CT 反解电极位置是另一条流水线，不在 R0.1 范围内。Epilepsiae 侧则有若干
患者部分覆盖（最差 1073：60 条 bipolar 只有 23 条有坐标）。

**裁定：`contact_valid` 与 `coord_valid` 是两条独立的轴。**

- `contact_valid` = "这里有一路成形的 bipolar 信号"（两端都在、都是颅内、native
  索引一致）。缺坐标**不**使其为 False。
- `coord_valid` = "我们知道它在脑内哪里"。
- 每位患者记 `coord_mode ∈ {mm, shaft_index_only}`。

如果把两者合并，会直接丢掉 5 位完好患者、以及 1073 的 40 % 通道——为了一个
**位置先验**丢掉数据是错误的取舍。代价是：`shaft_index_only` 的患者不携带任何
解剖距离信息，**不能支持任何空间主张**，队列统计必须把两组分开报。

### 4.4c 发作护栏的来源必须取并集（2026-08-21 裁定）

冻结的 `yuquan_seizure_inventory.csv` 会丢掉零时长标注（onset == offset 过不了
`has_complete_eeg_interval`）。审计发现这样**静默漏掉两个真实 onset**：
zhangbichen（落在 train）与 chenziyang（落在 validation）。

**裁定**：Yuquan 护栏取冻结 inventory 与 `results/seizure_detection/pr1_seizure_<subject>.json`
标注扫描的**并集**，按 onset 1 s 内去重，无可用 offset 时用 `onset + 120 s`。
每人训练少 ~2 h 是便宜的；让一次发作转变进入 backbone 不是。

**同时必须写明的局限**：16 位 Yuquan 患者中有 7 位在两个来源里都查不到任何发作
标注（`seizure_guard_source = none_found`）。**"没有标注"不等于"没有发作"**——
对这些患者我们无法保证 ictal 排除，这条必须进最终报告的局限段，不得省略。

### 4.5 缓存预算（工程约束，非科学约束）—— 2026-08-21 已撤销上限

原先按 zstd 对 int16 脑电约 1.4× 的假设估算，全量缓存需 150–200 GB，因此设了
`train ≤ 36 h` / `validation ≤ 12 h` 的截断。**第一位真实患者建完后实测压缩比是
4.68×**（yuquan_huanghanwen，1289 分钟 × 87 触点，3.45 GB → 0.737 GB），全队列
无上限也只需 **约 68 GB**，而盘上有 834 GB 空闲。

因此**两个上限都撤掉**（`CACHE_*_HOURS_CAP = None`），缓存全部 dev 已覆盖分钟。
一个只为省盘、却让 epilepsiae_620 丢掉 213 小时里四分之三的工程限制，在它不再
换来任何东西时没有保留价值。代价是建缓存的源读取从 ~1056 h 升到 ~2415 h，
8–10 路并行约 1.5 小时。逐患者实际缓存时长仍写入
`eligibility_summary.csv::cached_train_hours / cached_val_hours`。

---

## 5. 模型合同 (R0.1)

### 5.1 输入（且仅输入）

| 键 | 形状 | 说明 |
|---|---|---|
| `raw` | (C, 10·15360) | 过去 10 分钟 256 Hz 解码信号，train 统计量归一 |
| `coords_mm` | (C, 3) | 触点坐标（bipolar 中点），患者自身空间，均值中心化；无坐标处写 0 |
| `coord_valid` | (C,) | 该触点**是否有解剖坐标** |
| `shaft_id` | (C,) | 杆编号 |
| `shaft_index` | (C,) | 沿杆的 0-based 序号 |
| `contact_valid` | (C,) | 该触点**电学上**是否可用（与有无坐标无关） |
| `minute_valid` | (C, 10) | 每触点每分钟 artifact mask |

位置编码始终有下限：
`pos = shaft_emb(shaft_id) + shaft_index_emb(shaft_index) + coord_proj(coords_mm)·coord_valid`。
`coord_proj` 在 `coord_valid=False` 处必须**整体门控为零**（含 bias），否则会给无坐标
触点泄漏一个常数"幻影位置"。

**禁止输入**（`contract.FORBIDDEN_INPUT_KEYS`，出现即硬报错）：IED label / IEI /
event rate / seizure label / seizure onset / SOZ / contact rank / lagPatRank /
template rank / 现有 Epi-PRSSM latent / swap_class / vigilance / day-night。

### 5.2 Encoder

1. 共享 Conv1D raw patch projection：250 ms = 64 sample → $d=128$；
2. 2 层触点内 Temporal Transformer（序列 = 一个 5 s 窗内的 20 个 patch）；
3. 2 层触点间 masked Spatial Transformer（序列 = C 个触点，坐标+杆 ID 作为
   位置编码，`contact_valid` 作为 key padding mask）；
4. 一分钟 attention pooling（12 个 5 s token → 1 个 minute token）；
5. 3 层 causal context Transformer（序列 = 10 个 minute token）；
6. 线性头 → $z_t \in \mathbb{R}^{32}$。

### 5.3 稳定动力学

**不使用**把 $A$ 限制在近单位阵小邻域的旧参数化，也**不**在第一版上 full
continuous-time stable generator 或大规模 matrix exponential。

16 个二维阻尼旋转 mode，$h$ 单位为分钟：

$$
B_j(h) = e^{-h/\tau_j}
\begin{pmatrix} \cos(\omega_j h) & -\sin(\omega_j h) \\ \sin(\omega_j h) & \cos(\omega_j h)\end{pmatrix},
\qquad
z_{t+h} = \mu + B(h)\,(z_t - \mu)
$$

- $B(h)$ 是 16 个 2×2 块的 block diagonal；
- $\tau_j$ 在 log 空间学习，硬夹在 **[1 min, 48 h]**，log 空间均匀初始化覆盖全程；
- $\omega_j$ 有界（$|\omega_j| \le 2\pi/1\text{min}$），允许 $\omega_j=0$ 的纯衰减；
- 每个 mode 严格稳定（$e^{-h/\tau_j} < 1$）；
- **100 分钟预测直接算 $B(100)$**，不保存 100 步递归计算图。

> 只有当这个简单模型出现明确表达能力不足的证据时，才把 full continuous-time
> stable generator 列为后续探索；本轮不提前复杂化。

### 5.4 Decoder

单个线性层 $\hat{y} = W z + b$，$W \in \mathbb{R}^{(C \cdot 12) \times 32}$。
所有 horizon **共用同一个 decoder**——horizon 的差别只体现在 $B(h)$ 上。

### 5.5 目标

一分钟 contact × frequency log-power 场：256 Hz 信号上 Welch
（`nperseg=2048` = 8 s，50 % overlap，hann，per-minute 14 段平均），积分到 12 个
log bin（band-power 内剔除 50/100 Hz ±1 Hz），取 $\log_{10}$，再用
**train 分区的每触点每频段均值/标准差**归一。

归一后 "patient mean" 基线在 train 上恰好 = 0，其归一化 MSE ≡ 1.0，因此
**主指标（归一化 MSE）可以直接读作"相对患者均值基线的剩余方差比"**。

---

## 6. 损失

$$
\mathcal{L} = \mathcal{L}_{\text{forecast}} + \lambda_{\text{cons}}\,\mathcal{L}_{\text{cons}}
$$

- $\mathcal{L}_{\text{forecast}}$：1 / 5 / 10 / 100 分钟四个 horizon **等权**，
  各自对有效 (contact, freq) 取 masked MSE。
- $\mathcal{L}_{\text{cons}} = \mathrm{Huber}\big(z_{\text{enc}}(t{+}1),\ \Phi_1(z_{\text{enc}}(t))\big)$，
  $\delta = 1.0$。$\Phi_1$ 即 $h=1$ 的稳定动力学映射。

**只比较两档**：$\lambda_{\text{cons}} = 0.1$（默认）与 $\lambda_{\text{cons}} = 0$。
第一轮不做大规模 consistency 超参搜索。

---

## 7. 必须报告的一致性量

$$
E_{\text{cons}} = \frac{\lVert z_{\text{enc}}(t{+}1) - \Phi_1(z_{\text{enc}}(t))\rVert}
{\lVert z_{\text{enc}}(t{+}1) - z_{\text{enc}}(t)\rVert + \varepsilon}
$$

在 validation 上按患者报中位数与四分位。$E_{\text{cons}} \ll 1$ 表示"一步动力学
解释了状态的大部分变化"；$E_{\text{cons}} \gtrsim 1$ 表示编码器每分钟各编各的、
不在同一条轨迹上（此时适用 §2.4 的降级措辞）。

---

## 8. 最小探索实验集

用户明确要求第一阶段保持探索性，不堆防御性实验。

### 8.1 基线（仅此 5 个）

| # | 名称 | 定义 |
|---|---|---|
| 1 | patient/session mean | 预测 train 均值（归一化后即 0） |
| 2 | persistence | 预测 = context 最后一分钟的观测场 |
| 3 | spectral feature-AR（低容量） | 每频段共享一组 ridge 系数，特征 = 本触点该频段过去 10 分钟 + 全触点均值过去 10 分钟（21 维），train 上内部 CV 选 alpha |
| 4 | raw encoder + identity dynamics | 同一 encoder，$B(h) \equiv I$ |
| 5 | 完整模型 | encoder + damped rotation + consistency |

### 8.2 核心分析（仅此 5 项）

1. 1 / 5 / 10 / 100 分钟 horizon curve（模型 vs 5 个基线）；
2. observed vs open-loop spatial-frequency trajectory（代表患者）；
3. matched state-swap；
4. encoded / generated state consistency（$E_{\text{cons}}$）；
5. latent mode 的时间常数、频率，以及触点 / 频段 loading。

**matched state-swap**：在同一患者内，找到当前频谱场与 $t$ 近似匹配的另一时刻
$t'$（在归一化场空间中最近邻，且 $|t-t'| >$ 2 h 以避开自相关），把 $z_t$ 换成
$z_{t'}$ 后重新解码未来。**若当前场相似而未来预测显著变差**，说明状态携带了超出
当前快照的历史信息。报告 $\Delta\text{MSE} = \text{MSE}(z_{t'}) - \text{MSE}(z_t)$
按患者中位数 + 符号检验。

### 8.3 本阶段明确不做

大量 time shuffle、大量 patient swap、多套坐标 null、adversarial nuisance
removal、复杂 sleep/day-night 分层、数十个模型容量对照。核心结果为正后再在下一
阶段增加**有针对性**的替代解释实验。

---

## 9. 硬性无效条件（发现即修复并重跑受影响作业，但不阻断其他作业）

1. 时间泄漏（context/target 顺序错误、未来信息进入 encoder）；
2. split 泄漏（normalization 或早停用到 validation / sealed）；
3. context/target 跨 gap（>300 s）或跨 seizure 护栏；
4. 通道顺序错误（raw cache 列序与 contact_metadata 不一致）；
5. normalization 使用 validation/test 数据；
6. 非有限结果（NaN/Inf loss 或 metric）；
7. manifest 与实际运行不一致（code_revision / package_hash / 配置漂移）。

---

## 10. 与既有工作的关系

- **不覆盖、不混入** Epi-PRSSM v0.1（`results/epi_prssm/v0_1/`）的任何产物。
  R0.1 只**读取**其 `SPLIT_MANIFEST.json` 作为封条来源。
- R0.1 与 Epi-PRSSM v0.1 的 latent 完全独立：v0.1 的 latent 在
  `FORBIDDEN_INPUT_KEYS` 中。
- 已知的 v0.1 教训直接继承（见 memory `project_topic5_epi_prssm_v0_1_2026-08-18`）：
  `softplus(log τ)` 会把时间常数压在秒级 → R0.1 用 `exp(clamp(log τ))` 并在
  log 空间跨 1 min–48 h 均匀初始化；容量对照必须逐节点冻结 → R0.1 的 identity
  dynamics 臂与完整模型 encoder 容量完全相同，只换 $B(h)$。

---

## 11. 后续版本（本轮不得混入同一训练作业）

- **R0.2**：10 分钟 vs 2 小时历史；必要时 observation-update filter。
- **R0.3**：shared encoder + patient adapter + patient-specific decoder。
- **E0.4a**：冻结 raw backbone，接 event-mark decoder，检验 H2a。
- **E0.4b**：分离 short-lag renewal/burst，加 arrival intensity。
- **E0.5**：发作前连续状态轨迹、pseudo-onset、seizure subtype sensitivity。
- **E0.6**：T1 observation-only / T2 physical-update / IED innovation /
  多时间尺度核 / time reversal / future placebo。
