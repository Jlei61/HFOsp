# Group-Event State v0.2：共同科学合同（重大修订版）

状态：**development scientific contract；工程修复已通过，科学设计按本版执行，正式分区保持封存**

代码线：`codex/topic5-group-event-state-v0-2`

数据：只读复用 `/data/hfosp_group_event_state_v0_1/dataset`
新结果根：`results/epi_prssm/group_event_state/v0_2/`

本文件只规定科学问题、共同数据边界、共享状态生产者和跨线接口。历史缺陷、资源、原子写入和回归测试移到
`group_event_state_v0_2_engineering_invariants_2026-09-01.md`。

## 1. 唯一科学主线

```text
背景生理过程（睡眠/昼夜/药物/住院进程/记录条件）
                         │
                         ▼
                 慢易感状态 S(t)
                  ╱             ╲
                 ▼               ▼
       群体间期事件 X(t)      发作风险与早期路径 Y(t)
                 │
                 └──── ? ────► S(t+)
                         H3 feedback
```

三条线不是三个互不相干的 benchmark：

- **A：H1/H2a** 建立只依赖间期群体事件历史的 predictive state，并判断它预测 rate 还是 network repertoire。
- **B：H2b** 判断该状态能否跨任务预测距发作的距离和发作早期空间场；这决定状态是否接近“癫痫易感状态”。
- **C：H3** 比较 common-drive 与 explicit event-feedback，判断 IED 数量和内容是否对之后状态具有额外反馈式预测依赖。

A 是地基，B 决定核心科学价值，C 是高风险机制扩展。三线可以并行探索；任何一线的阴性都不阻断另外两线，C 的首轮算力不得超过 B。

## 2. 一个时间步和完整事件输入

模型的一步对应一次完整间期群体事件，不是单触点尖峰、rank step 或固定一分钟窗。事件输入保留：

- participation mask、size/STOP、10 ms tied groups；
- participant-masked `lagPatRaw` 频谱质心相对延迟；
- bipolar/CAR 后的原生事件窗 waveform；
- 多频带能量、包络、峰时和 cross-band lag；
- 触点几何、shaft、坏道、gap 和 coverage mask。

连续背景 SEEG 是辅助 observation；manifest 必须明确某个 producer 是否真的使用它。每次预测必须是 causal prefix：先预测，再读取当前真实事件更新 observer/state。一步 teacher-forced 结果称 **filtering**；从同一 anchor 不读取中间真实事件的预测才称 **forecasting**。

## 3. 三种必须并列生产的候选状态

共享目录不再只登记一个“最佳 checkpoint”，而是维护：

`results/epi_prssm/group_event_state/v0_2/shared/checkpoint_registry.json`

至少包含三种 producer；B/C 全部读取，不能由 A 的阳性/阴性筛选。

### 3.1 `B_multiscale`：可解释多尺度基线

在固定物理时刻构造 1/5/30/120 min EWMA，至少覆盖：

- event rate、time since last event；
- size/STOP、participation burden；
- repertoire occupancy；
- band-energy summary；
- clock time、session position、coverage；
- time since previous seizure、postictal/cluster indicator；
- sleep/wake、ASM/临床干预只在数据确实存在且通过元数据合同后加入。

使用线性 GLM 和一个低容量 MLP。若它追平 recurrent state，这本身是有价值的科学结果。

### 3.2 `P_local`：局部事件状态

当前 next-event recurrent model，优化：

\[
\mathcal L_{local}=\mathcal L_{timing}+\mathcal L_{mark}.
\]

它回答近期事件历史是否影响下一事件，不承担小时级慢状态的阴性结论。

### 3.3 `P_slow`：真正的 multi-horizon predictive-state producer

保留 local loss，同时直接训练未来物理时间块：

\[
\mathcal L=\mathcal L_{local}+\lambda_5\mathcal L_{5m}
+\lambda_{30}\mathcal L_{30m}+\lambda_{120}\mathcal L_{120m}.
\]

future-block target 至少包含：event count、conditional mark/repertoire、participation field、extent/STOP、multiband expression。长 horizon 主要读取候选慢状态；第一版不加正交约束，不靠 latent 命名证明快慢分离。

## 4. 什么才叫“慢预测状态”

`z_fast` 和 `z_slow` 仍保留为架构模块和诊断量，但不是科学定义。它们都读取事件、时间常数重叠且潜空间可重参数化。

承重对象是冻结功能读出：

\[
S_{func}(t)=\big[p(N_{t:t+\Delta}),\ p(mark\mid N>0),\
p(participation),\ p(extent/STOP),\ p(multiband)\big].
\]

一个状态只有在固定真实时间上持续预测未见 future block，并且相对 `B_multiscale` 有增量、正确时刻优于 block-shift 时，才称 time-specific predictive state。`fast-only`、`slow-only`、reset 和 raw latent trajectory 都是诊断，不是必要判据。

## 5. 两类 anchor，不再用事件率给慢分析加权

### 5.1 Event anchor

只用于 H2a：下一事件、same-prefix continuation、event-to-event mark/propagation。

### 5.2 Fixed physical-time anchor

慢状态、future-block、H2b risk trajectory 的主分析使用每 5 min 一个 anchor；30/120 min target，覆盖足够再探索 6 h。在 grid 时刻，把最后事件后的状态按真实 `dt` 传播到 grid：

\[
S(t_{grid})=propagate\{S(t_{last\ event}),t_{grid}-t_{last\ event}\}.
\]

每段真实时间只按网格贡献，不因 IED 多而自动重复加权。

## 6. 时间 null、历史尺度和主比较

承重比较是同一批 anchors 上的嵌套增量：

\[
B_t\quad\text{vs}\quad B_t+S_t.
\]

主要 time-specific null 是同患者、同 recorded session 的 **block circular shift**：平移量严格大于 target horizon，保留状态自相关和 session/coarse-clock 结构，破坏与当前未来块的精确对应。

matched wrong-time donor 降为敏感性，只粗匹配 session、time-of-day bin、coverage、recent-rate bin；每 anchor 5–10 个 donor，报告可匹配比例。不要匹配 size、participation 等可能本身构成慢状态的信号。

reset 网格缩为：

- event count：1、100、1,000、full；
- physical time：5、30、120 min、full；
- fast-only/slow-only reset 只在少数固定患者做机制诊断。

历史尺度由 future-block score 随真实 horizon 的曲线决定，不用“哪个 reset 首次不显著”定义。

## 7. 数据切分、session 和发作边界

1. state model 按累计 **recorded physical time** chronological 切 TRAIN/inner-validation/development-test，不再按事件数切；所有 future target 不跨 split。
2. batch 可以包含不同 recorded sessions；同一 session 内 chunk 严格按序、carry hidden state，边界只 `detach` 不 reset；只 shuffle sessions，不 shuffle session 内 chunks。
3. 不跨未记录 gap、记录段或不允许的 coverage boundary 传播状态。
4. H1/H3 exposure 和 target window 不跨 seizure onset。
5. H2b 只读取 seizure 前 trajectory；seizure onset 立即终止该条 trajectory。发作后不静默桥接：从 seizure offset 后 60 min 起新 segment；60 min 为首轮 primary postictal exclusion，其他长度只作敏感性。
6. ictal-overlap 事件排除；preictal 间期事件保留。以后若要学习 seizure 对状态的更新，另开显式 ictal-token 版本。
7. formal/sealed 分区继续关闭，所有结果均为 development。

## 8. Repertoire 和 target 语义

future block 必须分开：

- `count/rate`：未来有多少事件；
- `conditional mark`：给定发生事件后，它们属于什么空间—频带表达。

TRAIN-only patient-specific clusters 只作解释性输出；主稳健输出使用 continuous event embedding distribution（如均值/协方差、energy/probabilistic score）。这样结论不依赖 KMeans 的 K 和初始化。

future targets 使用稀疏 anchor、cumulative sums、prefix counts 和 sparse participation arrays；只保存 anchor index、区间和必要统计，不完整复制“事件×horizon×触点”张量。

## 9. H3 的两种解释必须在模型内分开

普通 RNN 的 event update 首先是 observer 看见新观测后的 belief update，不能自动解释为生理反馈。H3 必须显式比较：

\[
M_0:\ S_{e+1}=G(S_e,\Delta t_e,B_e),\quad X_e\sim p(X_e\mid S_e)
\]

\[
M_1:\ S_{e+1}=G(\cdot)+A_{count/rate}(X_e)
\]

\[
M_2:\ S_{e+1}=G(\cdot)+A_{mark}(participation,extent,waveform,multiband).
\]

最高允许措辞是 **event-feedback-like predictive dependence**。人体观察数据不直接证明 IED 生理因果塑形。

## 10. 共同评估和 checkpoint registry

每个 registry 条目必须显式包含：

```json
{
  "producer_id": "P_slow",
  "model_family": "group_event_recurrent",
  "uses_waveform": true,
  "uses_multiband": true,
  "uses_background": false,
  "event_update": true,
  "feedback_model": "observer_only",
  "physical_dt": true,
  "training_objective": ["next_event", "future_5m", "future_30m", "future_120m"],
  "anchor_grid_minutes": 5,
  "source_commit": "...",
  "config_hash": "...",
  "checkpoint_hash": "..."
}
```

checkpoint 只能依据各自的间期 TRAIN/inner-validation objective 选择，不能看 seizure 或 H3 结果。B/C 读取 registry 的全部合格 producer；缺失 producer 报 `not_available`，不得静默 fallback。

共同统计纪律：patient-first；H2b 以 held-out seizure 为基本分母；H3 以不重叠 physical block 为基本分母；seed 是重复拟合，不是样本量；工程 PASS、assay sensitivity 和生物学结论分开报告。

## 11. 执行顺序和三张承重图

### Phase 0：共同地基

验收 warm-up/session carry，落实发作边界、fixed-time grid、physical-time split、checkpoint registry 和稀疏 future target。

### Phase 1：三种 producer

并列训练 `B_multiscale`、`P_local`、`P_slow`，不按结果筛 producer 或患者。

### Phase 2：A 与 B 并行

- A：future-block、count/conditional-mark 分解、same-prefix、block shift。
- B：fixed-grid seizure survival 与 early ictal spatial field/path。

### Phase 3：C

先 functional innovation，再比较 `M0/M1/M2`，最后做最小 perturbation。无需等待 B 阳性，也不把 synthetic recovery 作为科学 gate。

首轮主文只需要三张承重图：

1. A：相对 `B_multiscale` 的 future-block gain 随 5/30/120 min horizon 变化，拆 count 与 conditional mark，并显示 local/slow/correct-time/block-shift。
2. B：survival/Brier 和 early ictal spatial-field gain 随 lead time 变化，分母为 held-out seizures。
3. C：M0/M1/M2 的未见 future-block score，加 event-type-specific signed impulse response。

其余 latent、tau、reset、matched donor、cluster transition、update norm 和额外 perturbation 全部放辅助图或技术报告。

## 12. 当前允许和禁止的结论

- `P_local` 只在一步上赢：短程 predictive filtering。
- `P_slow` 在 fixed-time future block 超过 baseline 且 correct-time 超过 shift：time-specific predictive state。
- conditional mark、same-prefix 或 early ictal field 有增量：network-expression/susceptibility state。
- M1/M2 在未见 future block 超过 M0：event-feedback-like predictive dependence。
- 只有 latent update 或 post-hoc ablation：observer sensitivity，不称 IED 塑形。
- 住院 SEEG 首轮主目标是几十分钟至数小时；6 h 可探索，跨天不作必要 gate。
