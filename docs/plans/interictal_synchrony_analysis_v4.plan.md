---
name: Interictal Synchrony Analysis
overview: "Three-phase plan + optional sleep staging. Phase 1 (PR1–PR2.5): seizure/interval truth + Epilepsiae baseline — done. Phase 2 (PR4–PR6): event-level synchrony + interval annotation + stats — Epilepsiae done, **population-level null result confirmed**. Phase 3 (PR7–PR8): lagPat backfill. Science pivot: individual-subject dynamics, not cohort-level resynchronization."
todos:
  - id: p1-pr1-edf-parser
    content: "PR1: fast_read_edf_annotations + parse_seizure + zoneinfo + recording timeline (see PR1 supplement 2026-04-01)"
    status: completed
  - id: p1-pr1-5-epilepsiae-survey
    content: "PR1.5: Epilepsiae data structure / seizure timeline / legacy helper survey + report doc + normalized export contract"
    status: completed
  - id: p1-pr2-ll-rms-streaming
    content: "PR2: streaming LL+RMS detector + litengsheng metrics + onset/offset visualization vs normalized intervals (infrastructure complete; channel-mean detector retained only as baseline)"
    status: completed
  - id: p1-pr2-5-spatial-extent
    content: "PR2.5: per-channel LL + spatial participation detector (first-principles: progressive recruitment, high-frequency high-amplitude oscillation, self-limiting dynamics)"
    status: pending
  - id: p1-pr3-yuquan-detect
    content: "PR3: full Yuquan outputs -> seizure_onsets JSON + interval inventory + yuquan_tier_assignment.csv (tier supplements interval truth)"
    status: pending
  - id: p2-pr4-sync-metrics
    content: "PR4: event-level metric contract (interictal_synchrony event rows + block compat view); Yuquan+Epilepsiae isomorphic CSV; 34 tests pass"
    status: completed
  - id: p2-pr5-period-slicer
    content: "PR5: event-level interval annotation + fixed-window vs normalized-trajectory + gap-aware exclusions (Epilepsiae aggregation pattern; Yuquan timeline)"
    status: completed
  - id: p2-pr6-analysis-script
    content: "PR6: interictal_sync_analysis + Figures A–E + stats — Epilepsiae done (null result); Yuquan pending PR3 interval inventory"
    status: completed
  - id: p2-pr6-science-pivot
    content: "PR6 follow-up: science pivot — subject stratification, SOZ-label core analysis, n_participating covariate, prediction framing"
    status: pending
  - id: p3-pr7-pipeline-validate
    content: "PR7: GPU smoke + run_pipeline EDF->lagPat vs legacy lagPat"
    status: pending
  - id: p3-pr8-pipeline-t4
    content: "PR8: Tier C lagPat backfill + low-concurrency GPU batch + fold into analysis"
    status: pending
  - id: opt-pr9-sleep-staging
    content: "PR9 (optional): SEEG delta/sigma/alpha sleep proxy streaming + circadian control"
    status: pending
isProject: false
---

# Inter-ictal HFO 同步性分析 — 定稿计划 (v4)

## 核心科学问题

验证假设：间期 HFO 群体事件的通道间同步性是否在两次癫痫发作之间呈现「发作后 reset → resynchronize → 发作前峰值」的趋势。

---

## 当前科学结论（2026-04-03 实证更新）

### 核心结论：队列水平 null，个体水平异质

在 Epilepsiae `ready_full_artifacts` 16 subjects / ~1,280,824 event rows / 232 intervals 上完成了 event-level PR4→PR5→PR6 全链分析。**原始假设（发作后 reset → 再同步 → 发作前峰值）在队列水平不成立**。

**定量证据**

| 检验 | 指标 | n | 统计量 | p | 判定 |
|---|---|---|---|---|---|
| 固定窗口 Post vs Pre (paired Wilcoxon) | legacy | 128 pairs | r=0.064 | 0.529 | null |
| | phase | 128 pairs | r=0.089 | 0.380 | null |
| | span | 128 pairs | r=0.007 | 0.947 | null |
| Within-interval trajectory (median ρ → one-sample Wilcoxon) | legacy | 232 intervals | median_ρ=−0.003 | 0.290 | null |
| | phase | 232 intervals | median_ρ=+0.001 | 0.933 | null |
| | span | 232 intervals | median_ρ=−0.008 | 0.053 | 边缘，方向与假设相反 |

Subject-level direction counts（phase pre−post）：11 负 / 5 正 → 无一致方向。

**关键个体发现**

- **Subject 548**（= 论文 E14）：within-interval legacy ρ = +0.133。按论文同口径（event-level, 发作前 1h, Pearson）复现 r=0.147, p=3.2e−14, n_seizures=14 — **数值完全一致**。证明论文图 7B/7C 是真实的单 subject 效应，不是管线 bug。
- **Subject 916**（52 seizures, 435 blocks）：within-interval legacy ρ = +0.175, p=0.001 — **本队列最强正信号**。
- **Subject 1073**（>216h, 194k events）：legacy ρ = −0.674 — **强负趋势**，与假设完全相反。
- 仅 3/16 subjects 弱满足 adaptive-Kuramoto 双判据（phase pre>post AND positive within-interval ρ）。

**方法学教训**

1. **Pooled Spearman 不可用**：~1M event-level pooled 相关给出 p≈0（legacy ρ=−0.019），但效应量 <0.04，且 Simpson 悖论使正/负 subject 混为一谈。
2. **Legacy 指标 "0.6 wall"**：n_participating=3 时理论极限 ≈ 0.5918，非生物学。
3. **Core/Global 当前不可区分**：Epilepsiae lagPat 通道 ≡ legacy high-event 通道，非临床 SOZ；`n_core == n_channels` 在 100% event rows 上成立。

### 第一性原理：为什么 null，以及可能的破局点

**为什么 null 是合理的**

原始假设假定"所有 subject 的发作间同步性遵循同一动力学"。但从第一性原理看：

1. **发作起源区异质性**：不同 subject 的 SOZ 位置、范围、传播模式完全不同。用"高事件率通道"作为通道宇宙（legacy `avgPickChns`）并不等价于 SOZ-internal 通道。一个 subject 的"高事件率通道"可能主要是传播区，另一个可能确实是起源区。**通道选择语义不统一时，全队列均值化没有物理意义。**

2. **Interval 时长分布极不均匀**：3h 到 >50h 的 interval 被归一化到 [0,1] 后，"早期"和"晚期"的物理含义完全不同。对短 interval，Post 1h 可能就是整个 interictal 段；对长 interval，Pre 1h 只是末端一小段。**归一化时间轴假设了标度不变性，但没有物理理由。**

3. **1h block 时间分辨率限制**：如果再同步化的特征时间尺度是分钟级（如发作后 10-30 分钟的快速 reset），1h 粒度看不到它。如果是 >24h 的慢趋势，短 interval 根本装不下。**block 粒度对快/慢过程都不利。**

**可能的破局方向（按可行性排序）**

1. **消费 SQL `focus_rel` 构建真实 SOZ mask**：`focus_rel = 'i'`（in-focus）的电极子集与 `'e'`（extra-focal）分开分析。如果 SOZ-internal 通道的同步性确实在发作前上升，而 propagation 区不变或下降，当前 all-channel 分析会把两个方向平均成 null。
   - **可行性**：高。SQL 已解析；只需在 PR4 的 event export 中增加 `focus_rel` 标注，按通道子集重算 phase metric。
   - **风险**：`focus_rel` 语义未完全确认（`l` 可能是边界不确定标签）；部分 subject 的 `focus_rel` 混合（如 1073 有 `i`+`l`，1077 有 `e`+`i`）。

2. **`n_participating` 条件分析**：legacy 0.6 wall 证明低通道事件的指标是离散化噪声。只分析 n_participating ≥ 某阈值（如 ≥5 或 ≥ median）的事件，或将 n_participating 作为协变量进入混合效应模型。
   - **可行性**：高。数据已有 `n_participating` 列。
   - **科学意义**：大事件（高 participation）更可能代表"网络级"同步化，而非局部孤立事件。

3. **Subject 分层描述而非 cohort 检验**：放弃"全队列单一假设"的框架。对每个 subject 给出 within-interval trend 方向与强度，然后问"什么样的 subject 表现出正趋势？"——按 SOZ location、seizure frequency、seizure type、recording duration 等临床特征分组。
   - **可行性**：中。需要更多临床元数据（部分可从 SQL 获取）。
   - **科学意义**：最诚实的描述；可以写成"异质性本身就是发现"。

4. **Event-level temporal resolution（非 block-epoch）**：当前 event 的时间戳是 block start epoch + event 在 block 内的偏移。如果直接用 event timestamp（秒级精度）做 trajectory，而不是先塌缩到 1h block center，可以捕捉更细的时间结构。
   - **可行性**：中。需修改 `build_event_rows_from_result()` 以输出精确 event timestamp（目前仅 block_start_epoch + event_idx 推算）。
   - **风险**：事件间距不均匀；需要 GAM/样条而非简单 Spearman。

5. **预发作 prediction framing**：不问"trajectory shape"，而是问"发作前 1h 的事件特征是否系统性区别于其余时段？"——本质上是 seizure forecasting / pre-ictal state 的分类问题，而非 trajectory 的回归问题。
   - **可行性**：低-中。需要多 subject 的发作标注（Epilepsiae 已有）。
   - **科学意义**：与 seizure prediction 文献对接更自然。

6. **Yuquan Tier A subjects（手工标注 ≥2 seizures）**：gaolan(4), litengsheng(6), sunyuanxin(5), xuxinyi(3) 共 4 subjects。这些有最可靠的发作标注。如果方向 1-2 在这些 subject 上出现信号，可以作为 Epilepsiae null 的对照。
   - **依赖**：PR3（Yuquan interval inventory）必须先完成。

---

## 分析主语与队列（v4 修订）

**主分析单元是 `seizure_interval`（相邻两次 seizure 之间的间期），不是「先给患者分级再全队列卡死」。**

- **Subject eligibility**：存在可排序的相邻 seizure 序列；至少部分 interval 可对齐到 `lagPat`/block 时间轴；**不要求**该 subject 必须先有一段 `>3h` 的间隔才允许进入研究。
- **Interval eligibility**：每个 interval 单独判断：是否完整 EEG/clinical 区间、是否 gap-aware、是否有 `lagPat`+`packedTimes` 支撑、固定窗是否能完整落下。

**三个分析队列（参数化，非神圣常数）**

| 队列 | 含义 | 典型用途 |
| --- | --- | --- |
| `allEligibleIntervals` | 满足时间轴与 artifact 条件的相邻 seizure interval | 归一化轨迹、队列汇总、最大覆盖 |
| `legacyComparableLongIntervals` | 相邻 onset 间隔 `>= min_sync_interval_sec`（默认与 legacy 对齐，如 3h） | 固定 Post/Mid/Pre 1h 窗、与老图可比 |
| `shortIntervals` | 间隔 `<` 上述阈值 | 仅进入不跨边界的 interval-aware 分析；固定窗装不下则标记 `not_applicable_fixed_window`，**不**因此剔除整个 subject |

**短间期不是坏数据**：禁止半块归属或伪造子块标签；1h `lagPat` block 粒度上限见 `docs/epilepsiae_dataset_structure.md` 与 `AGENTS.md`。

---

## 数据现状快照

### Yuquan（21 subjects，`/mnt/yuquan_data/yuquan_24h_edf/`）

- **T1**（lagPat + 手动标注 ≥2 sz）：gaolan(4), litengsheng(6), sunyuanxin(5), xuxinyi(3) — 4 subjects
- **T2**（lagPat + 手动标注 =1 sz）：chenziyang, hanyuxuan, huanghanwen, zhangjinhan — 4 subjects
- **T3**（lagPat + 无标注）：chengshuai, dongyiming, huangwanling, liyouran, wangyiyang — 5 subjects
- **T4**（无 lagPat）：pengzihang, songzishuo, zhangbichen, zhangjiaqi, zhangkexuan, zhaochenxi, zhaojinrui, zhourongxuan — 8 subjects
  - zhangjiaqi 有 13×`_gpu.npz` + 1×`_refineGpu.npz`（最易先跑 pipeline）
  - 其余 7 人通常无中间产物，需全链重跑

### Epilepsiae（挂载全量 **27** subject SQL；**20** 个有 `all_data_lns` 间期中间产物 cohort）

- 已挂载：`/mnt/epilepsia_data`
- 原始数据不是 EDF，而是 `*.data + *.head`；临床/发作信息在 `all_data_sqls/*.sql`
- 间期间期文章相关中间产物也在挂载树里，老代码直接消费 `all_data_lns/*/all_recs` 风格目录
- **时区**：法/德/葡采集地不全是同一时区；**禁止**用中国服务器 OS 默认时区给欧洲数据打昼夜标签。使用 `timezone_default` + **per-subject / per-record override**。
- 记录连续性不能只看文件名；优先使用 SQL 里的 `recording.begin/end/duration` 与 `block.gap`，`.head.start_ts` 仅作为块级校验

### 昼夜标注与老代码 Bug

- 老代码 `plotting_fig3_lagPat.py` 使用 `utcfromtimestamp(start_t).hour + 1` ≈ **CET (UTC+1)**，对 **Epilepsiae** 大致合理，对 **Yuquan（CST）** 会偏约 **7h**。
- 新代码：`zoneinfo` + **显式** `timezone_default` / `overrides`，**不依赖 OS 本地时区**。

### SEEG 通道命名（Core/Penumbra）

- 例：litengsheng `A2, A5, …, H1` — 字母≈靶点/电极柄，数字≈深度；用于 top 参与频率的 Core vs Penumbra 分层。

---

## 执行环境与并行策略

### GPU / CPU 分工

- **Phase 1（PR1–PR3）**：CPU + I/O；不依赖 GPU。
- **Phase 2（PR4–PR6）**：CPU；lagPat 特征、切窗、统计、作图均非 GPU 瓶颈。
- **Phase 3（PR7–PR8）**：需认真管 GPU
  - `analysis.use_gpu_envelope: true` → `cupy` / `cupyx`
  - HFO detection 主要为 CPU + joblib；**全链** EDF→lagPat 才是重活。

### 推荐环境

- Python 3.11 + `zoneinfo`
- **推荐**：1× NVIDIA CUDA GPU，≥16 CPU cores，≥64GB RAM，SSD 数据盘
- **最小**：无 GPU 可完成 Phase 1/2；Phase 3 变慢但逻辑仍成立

### 并行原则

- **横向并行**：按 subject / EDF 文件并行跑 detection 与分析；**避免**「多文件并行 + 单文件内 `n_jobs=-1`」双层打满。
- `HFODetector`：chunked 模式下代码注释倾向 `n_jobs=1`（joblib 开销）；大 chunk 再提高 `n_jobs`。
- **GPU job**：单卡默认 **1** 个 pipeline；显存充足再试 2 并发。
- **先关重负载**：`network_analysis.enabled: false`；按需关 TF tile cache / `compute_tf_centroids`。

### GPU 准入（PR7 前）

- 跑 `config/smoke_gpu_chengshuai.yaml`（短 crop）确认 `cupy`、GPU envelope 可用。
- 再跑 `verify_cuda_fresh` / `verify_recompute_gpu_window` 类配置验证路径。
- Smoke 失败：Phase 1/2 继续；Phase 3 标记 CPU-only 高风险路径。

---

## Phase 1：发作真值、interval 与队列（Yuquan + Epilepsiae 真实验证）

**目标**：产出可靠的 `subject -> seizures -> seizure_intervals -> eligible analyses`，而不是仅「重分层」。Epilepsiae 自 PR1.5 起即为**全阶段真实验证基线**（inventory/manifest/aggregation 已落地），不在此 Phase 仅做「调研」。

**Phase 1 交付（v4）**

- `interval inventory` 语义说明（与 SQL/EDF 对齐规则一致）。
- 区分 `subject eligibility` 与 `interval eligibility` 的准入定义。
- 明确 `legacyComparableLongIntervals` 与 `shortIntervals` 的 cohort 说明及各自允许的分析类型。

### PR1：EDF+ 标注解析 + 时区 + 记录时间轴（已完成，见下文补充）

**落点**：`[src/preprocessing.py](file:///home/honglab/leijiaxin/HFOsp/src/preprocessing.py)`

- `fast_read_edf_annotations` — 二进制 TAL；NFS 场景可用 `pread` 优化
- `parse_seizure_onsets_from_annotations` — label 匹配、onset–END 配对、归一化区间
- `epoch_to_local_hour` — **只用** `ZoneInfo`，不用 OS 隐式时区
- `read_edf_record_info` / `build_recording_timeline` — **绝对时间分析**以 EDF header 为准，禁止 `12 files == 24h` 硬编码

**Config 示例**：

```yaml
dataset:
  timezone_default: 'Asia/Shanghai'
  timezone_overrides: {}   # Epilepsiae: per-subject e.g. Europe/Paris vs Europe/Lisbon
```

### PR1 补充：真验收与 Yuquan 审计（2026-04-01）

**核心判断**

- PR1 **可以作为真正验收**，但边界必须限定在：EDF+ 解析、发作区间抽取、时区转换、**header 驱动记录时间轴**。
- PR1 **不能**被夸大成「临床 gold-standard seizure inventory」。EDF 手工标注含重复 onset、孤立 onset、缺失 END 等脏数据。

**已落地修正（摘要）**

1. `fast_read_edf_annotations()`：NFS 下 annotation-tail 线程化 `pread`，冷文件约 `113s/edf` → 约 `10–14s/edf`（受 NFS 波动影响）。
2. `parse_seizure_onsets_from_annotations()`：精确 label 匹配；onset–END 配对；orphan 零时长 onset 不再当区间返回；同文件重叠/重复区间合并。
3. `read_edf_record_info()` / `build_recording_timeline()`：后续 PR 的 day/night、post-ictal、interictal 必须基于**时间轴 helper**，不得以固定 24h 假设。

**全量 Yuquan 审计（摘要）**

- 21 subjects / 260 EDF
- 原始命中：32 seizure-bearing EDF / 54 raw intervals
- 归一化后：25 valid interval-bearing EDF / 30 normalized intervals
- 另保留 16 个 orphan onset markers
- offset：有效 offset 来自 END 配对；重复 onset 共享同一 END 等需在 PR2/PR3 用**归一化区间**作 truth
- 连续性：多数 subject 连续分段；总时长非固定 24h（如 22h / 24h / 26h / 30.9h）；`litengsheng`、`zhangjiaqi` 存在真实缺口

**对后续 PR 的约束**

- PR2 / PR3 的 manual truth **必须**消费归一化后的 interval，不能 raw hit 计数。
- PR5 切窗必须基于 subject timeline helper。
- 文档基线：`docs/yuquan_24h_dataset_structure.md`、`docs/DEVELOP_PLAN.md` 的 2026-04-01 更新。

---

### PR1.5：Epilepsiae 数据结构与老代码可复用性调研

**核心判断**

- 这是个真问题，而且比补丁式乱接脚本更值得先做。`Epilepsiae` 不是 Yuquan 的 EDF 变体，而是另一套数据契约：`*.data + *.head + SQL`。
- 真正该信的时间真值不是老脚本里遍历 `.head` 拼出来的近似时间轴，而是 SQL 里的 `recording` / `block` / `seizure` 表；老脚本只能当线索，不配当真相。

**调研落点**

- 新文档：`[docs/epilepsiae_dataset_structure.md](file:///home/honglab/leijiaxin/HFOsp/docs/epilepsiae_dataset_structure.md)`
- 若需要统一导出：补一份 machine-readable survey 结果，例如 `results/epilepsiae_dataset_inventory.csv` / `results/epilepsiae_seizure_inventory.csv`

**调研目标**

1. 钉死原始数据结构
  - 原始根：`/mnt/epilepsia_data/inv`*, `/mnt/epilepsia_data/epilepsiae_3patient`
  - 每个 block 的最小合同：`.head.start_ts`, `sample_freq`, `duration_in_sec`, `num_channels`, `elec_names`
  - 明确是否存在 EEG、ECoG、SEEG 混合；老代码现状只是用 `eeg_chns` 把 scalp EEG 排除，剩下都当 intracranial，**并没有真正区分 SEEG vs ECoG**
2. 钉死临床/发作/连续性真值
  - SQL `seizure`：`eeg_onset`, `clin_onset`, `eeg_offset`, `clin_offset`, `pattern`, `classification`, `vigilance`
  - SQL `recording`：长程 `begin/end/duration/blocks/sample_rate/commentary`
  - SQL `block`：块级 `begin/end/gap`
  - 输出需要回答：
    - 每个 subject 总记录时长
    - 是否连续记录
    - 块间 gap 分布
    - seizure 标注是否完整
    - seizure-to-seizure interval
    - 是否能可靠做 day/night 标签
3. 钉死中间产物覆盖率
  - 是否已有 `*_gpu.npz`, `*_packedTimes.npy`, `*_lagPat.npz`, `*_lagPat_withFreqCent.npz`, `*refineGpu*.npz`
  - 哪些 subject 只有原始块，哪些已有完整间期中间件
  - `lagPat` 资产是否足够支撑后续同步性分析，还是必须先补 pipeline
4. 钉死老代码可复用入口，而不是搬垃圾
  - 可复用候选：
    - `ReplayIED/inter_events/epilepsiae_interictal/epilepsiae_utils.py::epilepsiae_block`
    - `ReplayIED/inter_events/epilepsiae_interictal/epilepsiae_cutIctal.py::return_timeStamp_fromDatetime`
    - `ReplayIED/inter_events/epilepsiae_interictal/epilepsiae_cutIctal.py::find_corrFileAndT`
  - 需要明确淘汰的坏味道：
    - 大量硬编码绝对路径
    - 同一 helper 在多个脚本里重复拷贝
    - 直接假设已有 `lagPat` / `packedTimes`
    - 把 `eeg_chns` 黑名单当成 modality 真值

**PR1.5 交付**

- 一份类似 Yuquan 的结构报告，至少包含：
  - 目录结构
  - 原始数据合同
  - SQL 表到分析字段的映射
  - 连续性与 gap 审计
  - seizure 标注完整性审计
  - modality 审计（EEG / intracranial，能否细分 SEEG）
  - 中间产物覆盖率
  - 老代码可复用函数表
- 一份“我们需要的统一格式”定义，最小字段建议：
  - `subject`
  - `recording_id`
  - `block_id`
  - `block_start_epoch`
  - `block_end_epoch`
  - `gap_to_prev_sec`
  - `sample_rate`
  - `n_channels`
  - `intracranial_channels`
  - `has_eeg`
  - `has_seizure_annotation`
  - `seizure_eeg_onset_epoch`
  - `seizure_eeg_offset_epoch`
  - `seizure_clin_onset_epoch`
  - `seizure_clin_offset_epoch`
  - `vigilance`
  - `has_gpu`
  - `has_packed_times`
  - `has_lagpat`
  - `has_lagpat_freq`
  - `has_refine_gpu`

**验收标准**

- 至少完成 1 个代表性 subject 的全链路核对，证明 SQL `block.gap` 与 `.head.start_ts` 能对上
- 至少完成全量 subject 的 inventory 统计
- 明确回答下面这些问题，而不是写空话：
  - 记录时间是否连续
  - 间隔时间怎么定义
  - seizure 标注是否全面
  - 有没有 lagPat 中间文件
  - 有没有 EEG，是否大部分为 SEEG
  - 昼夜标签能否可靠推导
  - seizure 间 interval 如何计算

**验收结论（2026-04-02）**

- ✅ **PR1.5 已验收通过**
- 已完成：数据契约钉死、`src.epilepsiae_dataset` 正式接口、四张 inventory、manifest、时区/day-night 规则、`ready_full_artifacts` block-level synchrony 实跑、以及严格整块归属的 interval/window 聚合
- 明确保留到后续 PR：PR6 统计建模、跨 subject 终稿图与正式统计结论

**对后续 PR 的影响**

- PR5 的昼夜与间期切窗，Epilepsiae 侧必须消费 PR1.5 输出的统一时间轴与 seizure inventory
  - PR6 若要把 Epilepsiae 纳入同步性分析，前提不是“能读文件”，而是 PR1.5 已经证明其标注与 lagPat 资产足够可信
  - PR7/PR8 若考虑补跑 Epilepsiae，必须先决定是复用现有 `all_recs` 资产，还是从 `.data/.head` 重建，不允许两条语义混着用

**Epilepsiae 在本 Phase 的验收锚点（真实数据）**

- `docs/epilepsiae_dataset_structure.md`；`results/epilepsiae_*_inventory.csv`；`results/epilepsiae_sync_subject_manifest.csv`
- 已跑通示例：`results/interictal_synchrony/epilepsiae_ready_full_artifacts/` 及 `aggregated/*.csv`

---

### PR2：Streaming LL+RMS 检测器 + **可视化验证（必做）**

**落点**：`[src/preprocessing.py](file:///home/honglab/leijiaxin/HFOsp/src/preprocessing.py)`

- `detect_seizure_streaming(edf_path, ...)` — 通道均值 + 现有 `detect_seizure_onsets_from_data` 逻辑；控制峰值内存。

**可视化验收**

1. **单 EDF**：X=文件内秒；竖线标手动 onset/offset（归一化区间）与算法 onset/offset；叠加 LL、RMS robust-z 与阈值。
2. **24h 拼接总览**（如 litengsheng）：绝对时间轴上人工区间 vs 算法区间。
3. **误差图**：onset/offset error 散点；标 FP/FN。
4. **审计表** CSV：每 EDF `n_manual`, `n_detected`, TP/FP/FN, median errors。

**数值验收**：recall ≥80%（可调）、onset 中位误差 <30s（小时级分析可放宽但需记录）。

**假阴性预案**：降阈值宁多不漏；rate spike 辅助；间期 sync 曲线异常崩塌标记；置信度分级。

**复盘结论（2026-04-02）**

- PR2 的工程基础设施已完成并可复用：二进制 EDF 流式读取、并行验证脚本、4类可视化产物、审计 CSV、缓存加速。
- 但 channel-mean LL+RMS 检测器不作为最终方案：跨 subject 泛化不足，参数在低 FP 与高 recall 之间不可同时满足。
- 根因是数据结构顺序错误：先做通道均值再提特征，抹掉发作空间招募信息。

### PR2.5：空间招募检测器（第一性原理）

**第一性原理约束**

1. 发作具有通道逐步招募（progressive recruitment）特征。
2. 发作期存在大幅高频振荡（LL 对该特征敏感）。
3. 发作具有自限性（参与比例回落形成 offset）。

**核心顺序修正**

- 旧：`channels -> mean -> LL/RMS -> threshold`
- 新：`channels -> per-channel LL -> per-channel z -> active-channel fraction -> threshold`

**实现落点**

- `src/preprocessing.py`
  - `_stream_edf_channel_ll()`：单次顺序扫描 EDF，输出 `(n_channels, n_records)` 的 LL 矩阵
  - `detect_seizure_by_spatial_extent()`：基于 `per_channel_k` + `min_active_frac` 的发作区间检测
- `scripts/pr2_seizure_validation.py`
  - 缓存改为 per-channel LL z 特征
  - 图形改为 participation 轨迹（替代 LL/RMS 双阈值图）
  - CLI 改为 `--per-channel-k`, `--min-active-frac`
- `tests/test_seizure_streaming.py`
  - 增加“全通道招募能检出 / 单通道伪迹不触发”的合成测试

**验收标准（两名患者同时达标）**

- `litengsheng` 与 `sunyuanxin` 同参数 `recall >= 80%`
- `FP` 压到与手工标注同量级（目标不超过 2x）
- onset 中位误差 `< 30s`
- 无需 `ignore_initial` 这类补丁参数

---

### PR3：全 Yuquan 输出 + `seizure_onsets.json` + interval 表 + Tier（artifact）

- T1+T2：标注/归一化区间驱动；T3+T4：算法驱动。
- 输出：统一 JSON；`results/yuquan_tier_assignment.csv`（A/B/C/D）**作为 lagPat/检测资产分层**，**不**替代 `seizure_interval` 准入。
- 必含：`subject -> ordered seizures -> seizure_intervals`（与 PR5/PR6 对齐），及每 interval 的 `interval_sec`、`eligible_for_fixed_window`（由固定窗是否可完整落下决定）。

---

## Phase 2：已有 lagPat 上的同步性与统计（Yuquan + Epilepsiae 同构交付）

**状态：Epilepsiae 侧 PR4–PR6 已完成，结论为 population-level null。Yuquan 侧 event export 完成，interval 分析待 PR3 interval inventory。**

**实现主链（已落地）**

- `src/interictal_synchrony.py`：`build_event_rows_from_result()` → `event_sync_v1` schema；`build_interictal_synchrony_from_legacy_lagpat()`；`run_epilepsiae_interictal_synchrony_from_manifest()`
- `src/interictal_synchrony_aggregation.py`：`_annotate_sync_events_against_intervals()`；`aggregate_epilepsiae_sync_rows()`；`run_epilepsiae_sync_aggregation()`
- `src/interictal_synchrony_analysis.py`：`assign_fixed_window_positions()`；`compute_normalized_trajectory()`；`paired_window_test()`；`within_interval_trend_test()`；`run_pr6_analysis()`；Figures A–E
- `scripts/run_epilepsiae_interictal_synchrony.py`、`scripts/aggregate_epilepsiae_interictal_synchrony.py`、`scripts/pr6_interictal_sync_figures.py`
- Yuquan：`scripts/run_yuquan_interictal_synchrony.py`、`scripts/aggregate_yuquan_interictal_synchrony.py`、`scripts/interictal_sync_analysis.py`
- `tests/test_interictal_synchrony_analysis.py`：34 tests pass

### PR4：Event-level synchrony metric contract（✅ 已完成）

**原计划 block-level → 实施时重构为 event-level**：每个 HFO 群体事件输出一行三指标（`sync_legacy_global`, `sync_phase_global`, `sync_span_global`），加 `n_participating`、`n_channels`、`block_stem`、`block_start_epoch` 等元数据。Block mean 保留为兼容派生视图（`block_sync_compat_v1`），但不再是分析主语。

**落点**

- 低层事件函数：`src/group_event_analysis.py` — `sync_legacy_pdelay`、`sync_exp_pairwise`、`sync_pairwise_coincidence`
- 对外契约与批量：`src/interictal_synchrony.py` — `build_event_rows_from_result()` → `event_sync_v1` schema
- 元数据：`subject`、`recording_id`（如有）、`block_stem`、`block_start_epoch`、`block_end_epoch`、`event_idx`、`n_participating`

**数据规模**

- Epilepsiae：16 subjects / 2962 blocks → **~1,280,824** event rows（CSV ~数百 MB）
- Yuquan：156 blocks → event CSV ~72.6 MB
- Subject 1073 单独贡献 ~194,521 event rows

**验收**

- 指标契约固定；Yuquan 与 Epilepsiae 输出同构可 join
- 单测：`tests/test_interictal_synchrony.py` + `tests/test_interictal_synchrony_analysis.py` — 合计 34 tests pass

### PR5：Event-level interval annotation（✅ 已完成）

**重构**：annotation 对象从 block 变为 event row。Day/night 直接由事件 epoch 推导，不继承 block label。

**窗型 1 — fixed-window analysis**

- 仅对 `clean_between_seizures_sec ≥ 3h` 的 interval 执行；Post/Mid/Pre 各 1h
- 事件按 `block_center` 落入判定；优先级 Post > Pre > Mid
- Epilepsiae 实跑：128 Post–Pre pairs / 15 subjects

**窗型 2 — normalized-trajectory**

- `norm_t = (event_epoch − clean_start) / span`，clamp 到 [0,1]
- Within-interval trend 要求 ≥3 events per interval → 232 intervals 可用

**Gap-aware 排除**

- 跨 seizure / post-ictal / day-night / gap 边界的 event 直接排除
- 排除统计（Epilepsiae）：outside_intervals 796, overlaps_seizure 263, phase_boundary 161, day_night_transition 238, nontrivial_gap 194

### PR6：统计 + 同步性变化图 + cohort summary（✅ Epilepsiae 已完成；Yuquan 待 PR3）

**脚本**：`scripts/pr6_interictal_sync_figures.py`（CLI）、`src/interictal_synchrony_analysis.py`（统计与图核心）

**落盘（Epilepsiae）**

- `results/pr6_analysis/stats/pr6_analysis_stats.json`
- `results/pr6_analysis/figures/subjects/<subject>/figure_a_*.png`（16 subjects，Subject 1073 产出 57 张 = 3×(1+18)×12h panels）
- `results/pr6_analysis/figures/figure_b_*.png`、`figure_c_*.png`、`figure_d_*.png`、`figure_e_*.png`

**统计结果摘要（Epilepsiae `ready_full_artifacts`）**

见本文档"当前科学结论"章节。核心判定：

- 固定窗口 Post vs Pre：三指标均 p > 0.35 → **null**
- Within-interval trajectory：三指标均 p > 0.05（span p=0.053 方向反） → **null**
- Subject-level：巨大异质性，3/16 弱满足 Kuramoto 双判据

**图单产出状态**

- Figure A：✅ 16 subjects，12h facets，三指标 + seizure shading + window markers
- Figure B：✅ normalized trajectory ribbon（pooled + within-interval 两级汇总）
- Figure C：✅ fixed-window Post/Mid/Pre paired
- Figure D：✅ robustness（三指标并列）
- Figure E：✅ coverage/exclusion audit

**Yuquan 侧 PR6 现状**

- Event export（PR4）：已完成，CSV 在 `results/interictal_synchrony/yuquan_blocks/`
- Interval annotation（PR5）：部分完成；需 PR3 产出 Yuquan seizure interval inventory 后才能做完整 interval 标注
- 统计（PR6）：待 PR5 Yuquan 完成后才能跑

---

## Phase 3：EDF / `.data`→lagPat 全链补全（Yuquan + Epilepsiae 真实验证）

**Epilepsiae**：不允许仅在 Yuquan 上 smoke 通过后宣称泛化；须对 Epilepsiae 至少一条真实路径（复用 `all_recs` 资产 **或** 自 `.data/.head`+SQL 重建）做对齐或补跑验收，与 PR1.5「两条语义不混用」一致。

### PR7：GPU smoke + pipeline 对齐 legacy

- 执行顺序（不得跳步）：
  1. `smoke_gpu_chengshuai.yaml`（短 crop）
  2. `verify_cuda_fresh.yaml`
  3. `verify_recompute_gpu_window.yaml`
  4. full-record 对齐验证
- 对齐指标：
  - `lagPatRaw` 相关系数
  - `eventsBool` overlap
  - 事件数偏差
- 统一降负载：
  - `network_analysis.enabled: false`
  - 非必要 `tf_centroids` / tile cache 关闭
  - full-record 下 `hfo_detection.config.n_jobs` 默认 `1`
- PR7 验收硬标准：
  - GPU smoke 全通过（无 OOM / 无 cupy 错误）
  - 与 legacy 对齐达到预设阈值（阈值写入报告）
  - 记录 step 级耗时：preprocess / detections / windows / centroid / lag_rank
  - Epilepsiae：在选定路径上记录与 legacy `lagPat`/事件数对齐结果（或明确记录为何不可比）

### PR8：Tier C 批量补 lagPat

- 批跑顺序：
  - 第一批：已有 `_gpu` / `_refineGpu` 的 subject（先 zhangjiaqi）
  - 第二批：无中间件 subject，全链重跑
- 调度策略：
  - 单卡默认 1 并发；仅在 smoke 稳定后提到 2 并发
  - CPU 侧并行做 seizure json、质控、汇总
- 每个 subject 必须输出运行摘要：
  - 是否使用 GPU、总耗时、最慢步骤、是否复用 `_gpu`、最终事件数
- PR8 验收硬标准：
  - Tier C 应纳入者全部产出 lagPat
  - 更新后的跨 subject 统计自动重跑并落盘
  - 失败 subject 必须给 machine-readable 错误原因
  - Epilepsiae：`missing_interictal_artifacts` 队列按选定策略补跑后，更新 `epilepsiae_sync_subject_manifest.csv` 并重跑 PR4–PR6 聚合（或等价脚本）

---

## Optional PR9：SEEG 睡眠代理

- Streaming Welch：delta / alpha / sigma；连续 sleep-depth proxy；与墙钟 day/night 对照。

---

## 风险摘要（2026-04-03 更新）


| 风险 | 当前状态 | 应对 |
|---|---|---|
| 时区错（Epilepsiae 多国） | ✅ 已解决 | `timezone_default` + overrides；当前全量 UKLFR |
| N 与网络扩张混淆 | ⚠️ 未解决 | Core/Global 不可区分（无真实 SOZ label）；需消费 `focus_rel` |
| Legacy 指标离散化伪影 | ⚠️ 已识别 | n_participating=3 时 legacy ≈ 0.5918；phase 无此问题 |
| 空间漂移 | 未验证 | Jaccard/余弦（PR4 已输出 `adjacent_jaccard`） |
| 假阴性 seizure | PR2.5 进行中 | spatial detector v3 candidate；LOSO 待做 |
| 昼夜节律 | 部分控制 | day/night 已标注；PR9 sleep proxy 仍 optional |
| 短间期硬切固定 1h 窗 | ✅ 已解决 | trajectory 覆盖全 interval；fixed-window 仅限 ≥3h |
| Pooled 统计 Simpson 悖论 | ✅ 已解决 | within-interval + paired 两级检验；pooled 仅附录 |
| 队列水平假设本身不合理 | ⚠️ 核心发现 | **需科学 pivot**：subject stratification 或条件分析 |


---

## PR 与验收总览（2026-04-03 更新）


| PR | Phase | 状态 | 核心交付 | 验收要点 |
|---|---|---|---|---|
| PR1 | P1 | ✅ | EDF 解析 + 时区 + timeline | 归一化区间 |
| PR1.5 | P1 | ✅ | Epilepsiae 契约 + manifest + 聚合层 | 已验收；全阶段验证基线 |
| PR2 | P1 | ✅ | streaming 检测 + 可视化 | 基础设施通过；channel-mean 方案降级 |
| PR2.5 | P1 | 🔄 进行中 | 空间招募检测器 | v3 候选待 LOSO + Epilepsiae 验证 |
| PR3 | P1 | ⏳ 待做 | Yuquan JSON + interval 表 + tier | interval-first + 队列标签 |
| PR4 | P2 | ✅ | **event-level** metric 契约 + 同构 CSV | 34 tests pass；~1.28M event rows |
| PR5 | P2 | ✅ | event-level interval 标注 + 排除 | 128 pairs + 232 intervals |
| PR6 | P2 | ✅ Epi / ⏳ Yuquan | 统计 + Figures A–E | **Epilepsiae null confirmed**；Yuquan 待 PR3 |
| PR6+ | P2 | ⏳ 待做 | 科学 pivot：SOZ label + n_participating + stratification | 破局方向 |
| PR7 | P3 | ⏳ 待做 | GPU smoke + legacy 对齐 | corr + smoke |
| PR8 | P3 | ⏳ 待做 | Tier C lagPat + Epilepsiae 补跑 | manifest 更新 |
| PR9 | Opt | ⏳ 待做 | sleep proxy | 曲线合理 |


---

## Git 与文件位置说明

- 本文件位于 `**~/.cursor/plans/`**，**默认不在** `HFOsp` git 仓库内；丢失通常不是 `git revert` 能解释。
- 若需版本控制，可复制到 `HFOsp/docs/plans/` 并 `git add`。

