# Topic 5 — Stage 2 Early-Ictal Recruitment-Time Instrument Design Spec (2026-06-10)

> **状态**：设计稿 **v2**（v1 brainstorm 2026-06-10 + review patch：①montage/channel-identity 硬合同 §3.4 [P0]；②global-onset 去单调空条件 §4.2；③λ 单位=per-hour + pooled baseline + calibration_unstable §5.3；④Null D 按坐标空间分 epi-MNI-NN / yuquan-region §7.3；⑤spectral-edge 完全移出 feature_agreement 硬门 §6.2-6.3）。plan 待写。
> **v1 已锁的 3 点收紧**（仍有效）：窗口分层（extraction vs recruitment）/ 特征族融合（amplitude family gate + spectral corroboration）/ narrow=Main-A·broad=Main-B。
> **v2.1 montage trace（2026-06-10 Phase 2 trace-montage 步骤实测）**：§3.4 的 montage 工作假设被推翻——**按数据集不同**：yuquan = bipolar aliased-left；epilepsiae = **car**（非 bipolar）。ICTAL_REFERENCE 改为 per-dataset。见 §3.4。
> **Topic**：搭一个**真正的 early-ictal 招募时序仪器**——从原始发作 EEG 上，用多个独立检测器测"每个触点真正开始改变的时刻"，得到逐触点招募顺序，再用 Stage 1 已验证的 echo 统计量重测它与间期传播模板的对齐。**主线 = 真实招募仪器；pre-ictal 状态层是另一个 secondary layer，单独立 spec，不混进本 spec 的结论。**
> **定位**：Stage 1（proxy triage）的结论是"现成 ER/atlas 代理里的'像'主要来自共享粗锚（病灶距离 / 早晚优先级），不是具体路径复用；且 ER-derived rank 不是传播路径仪器"。Stage 1 同时留下两个**结构上闭不了**的缺口：(i) construct-validity（ER 最早 vs 真特征最早是否一致）pending；(ii) Null D 跨病人特异性对照在按通道名对齐时跑不起来。**Stage 2 的真仪器把这两个缺口变成可闭环的：cross-feature agreement = construct validity；coordinate/region-matched Null D = 特异性。**
> **Owner（user-locked 2026-06-08）**：topic5 拥有"真正发作 EEG 招募层 + 间期↔发作桥接"。与 Topic 4 H5（间期高频事件端点在发作邻近的招募）不重复——本层碰的是真正发作 EEG 信号本身的顺序。
> **前身**：Stage 1 `docs/archive/topic5/echo_gate/stage1_proxy_triage_2026-06-08.md`（spec `2026-06-08-topic5-ictal-template-echo-gate-design.md` v4）。Stage 1 的纯数学 echo 核 `src/topic5_echo_gate.py`（36 tests green）在 Stage 2 **直接复用**，不重造统计量。

---

## 0. 一个真正测什么的朴素话

每个病人有好几次发作。间期（没发作时）我们已经知道每个病人有一条形状固定、被高频事件反复扫到的传播通路（topic1 的"模板"）。Stage 2 问：

> **当一次发作真正开始时，各触点被"点着"的先后顺序，像不像这个病人间期那条固定通路的顺序？**

**和 Stage 1 的本质区别**：Stage 1 用的是一个**现成的、可能很烂的代理**（发作能量上升排名 ER-rank）当作"谁先点着"。它甚至没被验证过是不是真的在测传播。Stage 2 不再借代理——它**直接从原始发作脑电上量"每个触点什么时候真正开始改变"**。怎么量"开始改变"？不只看一种迹象：线长突变、宽带能量上升、高频活动上升、能量比上升、还有频率向快活动跳变（不靠幅度）。每种迹象各自给一个"开始时刻"。

**为什么用好几种迹象**：(a) 单一迹象会漏——低压快活动几乎不抬幅度，只有频率在变，所以加一个纯频率轴的检测器；(b) **几种独立迹象如果都说"同一批触点先点着"，这本身就回答了 Stage 1 没闭上的那个问题**——"代理测的到底是不是真招募"。几种真特征互相印证 = construct validity，是仪器的副产品，不是额外工作。

**一个必须写死的窗口纪律（user P1）**：发作前后我们会取一段宽的脑电（[-30, +30]s 量级）当**背景上下文**，但**"招募顺序"绝不能用整段窗里随便哪个漂移来定**。否则发作前几十秒的慢漂移会被当成"招募"。规矩是：先在宽窗里找**网络级最早的持续改变**（data-driven global onset），再**只在这个 global onset 附近的窄带里**为每个触点找它自己的招募时刻。落在标注 onset 很久之前（如 < −10s）的触点，标成"pre-onset change"，**不进**发作招募排名。

> **写进 spec 的一句话（user-verbatim lock）**：*Stage 2 uses a generous raw extraction window for context, but ictal recruitment rank is defined from sustained per-contact feature changes around a data-driven global onset, not from arbitrary pre-ictal drift inside the full window.*

---

## 1. 一句话主张（locked framing）

在 Stage 1 的扩展 epilepsiae + yuquan 探索性 cohort 内，**每次发作的多特征招募顺序（real recruitment-time rank，非 ER 代理）与该病人间期传播模板的相似度，跨病人合并后系统性高于 within-seizure 通道身份打乱的零假设**（单侧），并在**按惯常早晚分箱打乱（anchor-matched null）下是否仍成立**这一点上给出 Stage 1 无法给出的、基于真仪器的判读。

- 判定语言只允许 "**像 / 不像 / 没看清**"，**禁止** "predicts seizure" / "causes" / 任何 within-subject α-claim 升格为机制因果。
- "像"= ictal recruitment ordering 与 interictal template ordering 的 rank 相似度，**相对 within-seizure channel-shuffle null** 的超出量（dimensionless，沿用 Stage 1 `e_k`）。
- **Stage 2 相对 Stage 1 的增量**：(a) 用真仪器替换 ER 代理；(b) cross-feature agreement 闭合 construct-validity；(c) coordinate/region-matched Null D 闭合特异性。三者都做到 → 才有资格把"具体路径 vs 共享粗锚"这一层判读写成基于真仪器的结论。

---

## 2. 假设结构

### 2.1 H1（primary）— 真招募回声

**H1**：per-seizure recruitment echo strength `e_k`（fused recruitment rank vs template rank，§7）跨 cohort 合并的均值 > 0（单侧），且 bad-data regression 下零假设变平。

- 复用 Stage 1 `compute_echo_strength` / `pool_echo_subject_level`（subject-level primary）。
- H1 是 **inclusive claim**（具体通路重现 ∪ 共享病灶锚）。**anchor-matched null + LOO 去锚是必跑的解释层**，把 PASS 分成"含具体通路"vs"稳定锚为主"——它们决定结论强度，但**主统计只锁回声**（沿用 Stage 1 "只锁回声"纪律）。
- 每个 seizure 必须带 `feature_agreement_flag`（§6）；不达标的 seizure **只进 sensitivity，不进 primary**。

### 2.2 construct validity — 升级为 cross-feature agreement（不再人工 sentinel）

Stage 1 的 construct-validity 是"人工抽 ≥5 seizure 核对 ER 最早通道与 LL/broadband/HFA 是否同向"，且 pending。Stage 2 把它**机械化为 cross-feature agreement**（§6 / §9）：5 个特征各自给招募顺序，pairwise rank correlation + early-K overlap + feature-family consistency 作为每个 seizure 的 construct-validity 量。**这是仪器的内生产物**，不再依赖人工裁定（但 sentinel 阶段仍人工目视核对，§13）。

### 2.3 显式不在本 spec（secondary layer 单独立 spec）

- **真·pre-ictal 状态层（[-30,-1]min 的 rate / sync / 状态时序）= 另一个 secondary layer，单独立 spec**（user-locked：不与招募仪器混成一个结论）。它复用 onset-feature menu（`2026-05-10-topic5-subtype-onset-features-menu.md` B 类）。**本 spec 只做 early-ictal 招募仪器。**
- 有向连接（DTF/PDC/PTE）、贝叶斯 / 神经场反演 = 更重的独立子项目，不在本 spec。
- **EI（Epileptogenicity Index, Bartolomei 2008）= 文献定位，不作为独立 feature**：它是 CUSUM-on-spectral-ratio 的 composite，与 HFA + ER + spectral-edge 高度重叠，纳入会和这三者**双计数**。spec 引用它作为"本仪器在 SEEG 招募排名文献里的定位"，不进 5-feature set。

---

## 3. 队列 gating

### 3.1 入选规则（沿用 Stage 1 cohort，按真仪器可行性收）

subject 入选 ⇔ **同时**满足：

1. **有稳定间期模板（phantom-safe，硬合同）**：见 §8 narrow（Main-A）/ broad（Main-B）两套来源。模板 rank 必须 masked（per-cluster `valid_mask`），从 `results/interictal_propagation_masked/`（narrow）或 broad 树读，沿用 Stage 1 §3.6 硬合同。
2. **原始发作 EEG 可加载**：`extract_seizure_window(subject, seizure_idx)` 能取出不跨 block 边界的窗（epilepsiae + yuquan，见 §5），且 `n_seizures_eligible ≥ 2`。
3. **baseline 可解**：每特征在该 seizure 上能解出合法 baseline 窗（§5.3），否则该 (seizure, feature) drop。

> **B0 recruitment audit（必先跑）**：每 subject 枚举：`subject_id, dataset, fs, n_seizures_total, n_seizures_loadable, n_seizures_eligible, n_channels_template_{narrow,broad}, n_channels_recruited_{min,median,max}, per_feature_available (LL/broadband/HFA/ER/spectral_edge), global_onset_resolved_fraction, feature_agreement_flag_fraction, template_k_{narrow,broad}, swap_class, template_montage, ictal_montage, channel_identity_contract, n_channels_montage_matched, calibration_unstable_per_feature, pooled_baseline_sec, no_onset_rate_per_feature, null_d_mode, coord_available, channel_name_normalization_status, alignment_guard_pass, n_preonset_change_contacts, MIN_CH_pass`。
> **门槛锁死，audit 只报 drop**：`MIN_CH=8`、各特征 baseline / λ / detection-window 参数（§5）、global-onset 判据（§4）都**在跑数据前锁死写进本 spec**，audit **只报告按锁定门槛各掉多少**，**禁止**看了 audit 再回调门槛。**audit 必须在 cohort inference 之前跑且人工看过。**

### 3.2 per-seizure 资格门

- `n_channels_recruited(seizure) ≥ MIN_CH` 才进 echo；`MIN_CH=8`（沿用 Stage 1，跑数据前锁死）。`n_channels_recruited` = 在 §4 per-contact recruitment search 窗内解出合法招募时刻、且模板 `valid_mask==True`、且不是 pre-onset-change 的触点交集。
- **HFA 特征受 Nyquist 限制**：HFA 上界 150 Hz 需 `fs > 300 Hz`；fs 不够的 subject **HFA 特征不可用（feature drop，flagged `per_feature_available[HFA]=False`），不 drop seizure**——其余 4 特征照常。
- channel-order：recruitment rank 向量与 template rank 向量必须对齐到同一 channel 顺序后再求 ρ；**mismatch 必须 hard fail，不得 silent 截断**（沿用 Stage 1 alignment guard）。

### 3.3 数据集分层（mandatory sensitivity）

- **primary** = 合并，dataset 作 covariate / stratum。
- **必报 sensitivity** = epi-only、yuquan-only 各自合并估计。
- yuquan 只有单一 onset 标注（eeg_onset 即 clinical），故 §4 "annotation anchor vs data-driven anchor" 的双锚 sensitivity 在 yuquan 上退化为"data-driven anchor only"；epilepsiae 两锚都有。这一退化进 B0 audit。

### 3.4 montage / channel-identity 硬合同（P0 — 桥接的承重点；**2026-06-10 traced，per-dataset**）

> **核心风险**：interictal 模板的 `channel_names` 是单触点标签，若 ictal feature 用错 reference，名字相同但**信号对象不同**，echo 就变成"名字像"而非"同一路径比较"。
>
> **trace 结果（v2 已 trace，推翻了 v1 "两数据集都 bipolar-aliased-left" 的工作假设）**：montage **按数据集不同**：
> - **yuquan**：`config/default.yaml` `reference: bipolar` + `alias_bipolar_to_left: true`，`config/subject_params.json /yuquan/_defaults reference='bipolar'` → 模板通道（如 `D13`）= **bipolar pair `D13-D14` 别名取左**。
> - **epilepsiae**：`config/subject_params.json /epilepsiae/_defaults reference='car'`（`run_hfo_detection.py` epilepsiae 路径默认 car）→ 模板通道（如 `FLA2`）= **CAR 单触点**。佐证：epilepsiae 模板里相邻触点 `FLA2,FLA3,FLA4,FLA5` 同时出现，是 CAR 单触点特征，不是相邻 bipolar pairing。

- **Main-A primary 硬合同 — ictal reference 按数据集匹配检测 reference**：
  - yuquan ictal：`extract_seizure_window(reference="bipolar")`，再**按 alias-left 约定**（pair `D13-D14` → 标签 `D13`）对齐；`template_montage = ictal_montage = "bipolar_aliased_left"`。
  - epilepsiae ictal：`extract_seizure_window(reference="car")`，单触点同名对齐；`template_montage = ictal_montage = "car"`。
  - **不匹配（如 epilepsiae 用 bipolar、或 yuquan 用 car）→ `assert_channel_identity` hard fail**。CAR-vs-bipolar 的**跨数据集对比**不存在（各自 within-dataset 匹配；dataset 作 stratum，§3.3）。
- **`channel_identity_contract` 不是名字相等**：runner 按数据集取 `ICTAL_REFERENCE[dataset]`，断言 `template_montage == ictal_montage`（语义层）；**名字相同但 montage 语义不同 → hard fail，禁止 silent 通过**。
- B0 audit 加列：`template_montage`（per-dataset traced 值）、`ictal_montage`（生成时用的 reference）、`channel_identity_contract`（`matched_bipolar_aliased_left` / `matched_car` / `MISMATCH`）、`n_channels_montage_matched`。
- TDD 必含：构造"同名单触点、但一个是 CAR-monopolar 一个是 bipolar-aliased"的两路 → 断言 contract 检查 **hard raise**（不得因名字相等而通过）。

---

## 4. 窗口合同（user P1 收紧 — 三层窗，招募 ≠ 漂移）

### 4.1 三层窗定义（锁死）

| 层 | 名字 | 范围 | 用途 |
|---|---|---|---|
| 1 | **extraction / baseline-load window** | `[-PRE_SEC, +30]s` rel 标注 onset；`PRE_SEC` 默认 **300s**（沿用 atlas，给 λ 校准足够 baseline），block 不够时退到可用值，低于 baseline 下限则该 seizure 走 pooled 校准 / 标 `calibration_unstable`（§5.3）| 提供长 baseline（z + λ 校准）+ global-onset search 上下文 |
| 2 | **data-driven global onset search** | 在 extraction 窗内 `[-30, +30]s` 内带 | 找**网络级最早持续改变** `t_global`（§4.2）|
| 3 | **per-contact recruitment search** | `[t_global - 2s, t_global + RECRUIT_POST_SEC]`，`RECRUIT_POST_SEC=15s`（lock；sensitivity 10/20s）| 每触点只在此窄带内找自己的招募时刻 |

- **baseline 窗 ≠ recruitment 窗**：baseline（z + λ 校准）取层 1 的长 pre-onset 段（`resolve_baseline_window` EEG-onset-aware），recruitment rank 只从层 3 窄带定（§0 verbatim lock）。
- **pre-onset-change flag**：若某触点的 per-contact onset 落在 `标注 onset − 10s` 之前（即 `< -10s`），标 `pre_onset_change=True`，**不进**发作招募排名（进 audit 计数 `n_preonset_change_contacts`）。该 flag 在 §5.2 pass 1 上判。

### 4.2 data-driven global onset `t_global` 判据（锁死）

`t_global` = **在 global-onset search 窗内，跨触点的"已招募比例"首次达到阈值的时刻**。具体：

1. 对每个特征、每个触点，先按 §5（pass 1）z-score + CUSUM 得 per-contact onset（在 search 窗内，未约束到 recruitment 窄带）。**sustained-ness 在这一层强制**：CUSUM 越阈 + §5.4 ambiguous-drop 规则（越阈后 1s 内回落到 < λ/2 → 判 ambiguous，该 onset 作废）——所以进入 fraction 的每个 onset 已是"持续改变"，不是瞬时假警。
2. 在 fused 层（§6）得每触点一个 provisional onset（仅 non-ambiguous）。
3. `recruited_fraction(t)` = provisional onset ≤ t 的触点比例（分母 = 该 seizure 合法触点数）。
4. `t_global` = `recruited_fraction(t)` 首次 ≥ `GLOBAL_ONSET_FRAC`（lock=0.15）的最早 t。
5. 解不出（fraction 全程 < 阈值）→ `global_onset_resolved=False` → seizure drop（进 audit）。

> **去掉了原 v1 的 `GLOBAL_PERSIST_SEC` 单调空条件**（reviewer P1）：`recruited_fraction(t)` 按定义单调不降，"其后 1s 内不回落"天然成立、防不了瞬时假警。瞬时假警由 **per-contact 层的 ambiguous-drop 规则（§5.4）** 拦掉——sustained 约束放在它该在的层。global onset 只要求 ≥15% 触点（已通过持续性过滤）点着。

### 4.3 双锚 sensitivity（user：两锚互为 sensitivity）

- **annotation anchor**：t=0 = 标注 onset（epilepsiae clin_onset / eeg_onset；yuquan eeg_onset）。
- **data-driven anchor**：t=0 = `t_global`（§4.2）。
- recruitment rank **主口径用 data-driven anchor**（对标注误差稳健，回应"标注不准"）；annotation anchor 作 sensitivity。
- 每 seizure 报 `|t_global − 0|`（global onset 相对标注的偏移）；偏移过大（如 > 15s）→ `onset_anchor_discrepancy_flag`，进 audit + 降 confidence。

---

## 5. 特征定义（5 features — 每个特征自带完整 change-point 合同）

> **工程纪律（user P1）**：`compute_cusum_n_d_with_time` 是一个**通用 change-point wrapper**，不是 ER 专用魔法。每个特征都必须独立定义：trace shape / hop / window / baseline / normalization / λ calibration / detection window / no-onset·tie·ambiguous 规则。**禁止**把 ER 的默认参数照搬到别的特征。

### 5.0 共用骨架（所有特征走同一条路）

```
raw signal (n_ch, n_samp) @ fs
  → feature trace builder f_i: (n_ch, n_frames)  on a COMMON hop grid (HOP=0.1s)
  → baseline z-score against baseline window (resolve_baseline_window)  [robust variant §5.3]
  → per-(subject,feature) λ calibration on POOLED baseline (fpr_target_per_hour, §5.3)
  → clamped CUSUM (compute_cusum_n_d_with_time) → per-contact onset frame → onset_sec
  → no-onset / tie / ambiguous rules (§5.4)
```

- **COMMON hop grid**：所有特征用 `HOP=0.1s`，使 5 个 detector 的 onset 时间可直接比较 / fuse（ER 比较但不进 fuse）。各特征的分析 `WIN` 可不同（见下），但落到同一 hop 轴。
- λ calibration 复用 `src.ictal_er_rank.calibrate_lambda_per_subject`，**每特征独立**校准，**单位 = `fpr_target_per_hour`（不是 per-window）**，pooled baseline（§5.3）。
- z-score 复用 `baseline_zscore_er` 的 baseline-window 机制；**normalization 默认改 robust**（median / MAD）以抗 baseline 内偶发尖峰，见 §5.3。

### 5.1 五个检测器（fused instrument = amplitude ×3 + spectral ×1；ER = held-out 代理参照）

> **设计决定（user "或者不使用 ER" latitude 下取的判断）**：ER 是 Stage 1 的代理本体，把它放进 fused instrument 会让"真仪器 echo 模板"这个 headline 主张**被它要验证的那个代理污染**（循环）。故 **ER 不进 fused instrument**——它作为 held-out 代理参照单独算 `er_vs_fused_consistency`（§6.4），直接回答"Stage 1 代理是否同意真仪器"。**fused recruitment rank 只由 F1/F2/F3/F5 四个真特征定。**

| # | feature | role | trace builder | WIN | 物理含义 | Nyquist 约束 |
|---|---|---|---|---|---|---|
| **F1** | `line_length` | fused · amplitude | 滑窗内 `Σ|diff(signal)|`（时域，每 WIN 窗一帧）| 1.0s | 形态突变（幅度+频率合体）| 无（时域）|
| **F2** | `broadband_power` | fused · amplitude | spectrogram PSD 在 (1,45)Hz 求和 → `log` | 1.0s | 总体能量上升 | hi=45 需 fs>90（恒满足）|
| **F3** | `hfa_power` | fused · amplitude | spectrogram PSD 在 (80,150)Hz 求和 → `log` | 0.5s | fast activity / HFO 边缘 | hi=150 需 **fs>300**（不够则 feature drop）|
| **F5** | `spectral_edge` | fused · **spectral** | 每帧 90% spectral-edge frequency（PSD 累积到 90% 的频率）→ 上行跳变 | 1.0s | **非幅度轴**：频率向快活动跳变，捕捉低压快起始 | 用 (1, min(127, nyq))Hz 估 SEF |
| **ER** | `er_gamma` | **held-out 代理参照（不进 fused）** | `compute_er` fast=(60,100) slow=(4,20)（沿用现有 ER）| 1.0s | Stage 1 代理本体；只用于 `er_vs_fused_consistency`（§6.4）| hi=100 需 fs>200 |

- F1 line-length 在时域算（不经 spectrogram），但落到同一 hop 轴：帧 j 覆盖 `[j*HOP, j*HOP+WIN]`。
- F2/F3/F5/ER 复用 `compute_er` 的 spectrogram 骨架（`nperseg=WIN*fs`, `noverlap` 使 hop=0.1s）；F2/F3 用单 band power（不是 ratio），F5 用 SEF，ER 用现有 fast/slow ratio。
- **ER 走完整一样的 baseline z + λ + CUSUM 管线**（得 `er_rank`），只是**不进 fused median**；它的招募顺序 vs 真仪器顺序的一致性是 Stage 1 留下的 construct-validity 问题的直接答案（§6.4）。

### 5.2 两遍检测（two-pass — 先定 global onset，再定 recruitment）

为消除"global onset 需要 per-contact onset、per-contact onset 又需要 detection window（围绕 global onset）"的循环，检测分**两遍**，CUSUM 内核与 §5.0 完全相同，只是 search window 不同：

- **Pass 1（provisional，定 `t_global`）**：detection window = §4.2 的 global-onset search 窗（整个 `[-30,+30]s` 内带）。对每特征每触点跑 CUSUM 得 provisional onset → fuse（§6.1）→ `recruited_fraction(t)` → `t_global`（§4.2）。
- **Pass 2（final recruitment rank）**：detection window = `[t_global-2s, t_global+RECRUIT_POST_SEC]`（§4.1 第 3 层窄带）。**重跑**每特征每触点 CUSUM（从该窄带起点累积，沿用 `compute_cusum_n_d` 的 `search_start` 语义）→ pass-2 onset → fuse → **final `recruitment_rank`（§6.1 用的是 pass-2 onset）**。
- pre-onset-change flag（§4.1）在 **pass 1** 上判（onset < 标注−10s 的触点），这些触点**不进 pass 2 / 不进 recruitment rank**。

### 5.3 baseline + normalization + λ 校准（锁死 — reviewer P1）

**baseline + normalization**：
- baseline 窗：`resolve_baseline_window`，EEG-onset-aware，取层 1 长 pre-onset 段（§4.1），`MIN_BASELINE_SEC=60`（lock，per-seizure z-score 下限）。
- normalization：**robust z = (x − median_baseline) / (1.4826 · MAD_baseline)**（替换 `baseline_zscore_er` 的 mean/std；新写 `baseline_robust_z`，沿用其 baseline-frame mask + min-valid 逻辑）。MAD=0 的触点 → NaN → drop（不回退全局统计）。mean/std 版作 §10 sensitivity。

**λ 校准（单位明确 + 短 baseline 不可校准的处理）**：
- `calibrate_lambda_per_subject` 的语义是 **`fpr_target_per_hour`（每小时假警数），不是 per-window**。lock `fpr_target_per_hour=1.0`（沿用 atlas）。
- **单条 seizure 的 60–120s baseline 在 per-hour 口径下分辨率不足**（reviewer P1）：故 λ 在 **(subject, feature)** 层用**跨该 subject 所有合格 seizure 的 pooled baseline z-frames** 校准（pool 后帧数足以分辨 per-hour FPR）。
- `MIN_POOLED_BASELINE_SEC=600`（lock）：pooled baseline 总时长 < 600s → 该 (subject, feature) 标 `calibration_unstable=True` → 该 subject 在该特征上**只进 sensitivity，不进 primary**（不 silent 用一个推不准的 λ）。
- **sentinel 阶段（§13）必输出每特征**：`lambda`、`baseline_alarm_count`、`pooled_baseline_sec`、`no_onset_rate`（detection 窗内未触发触点比例）。no-onset rate 过高（λ 被推太高）在 sentinel 抓出来，回炉前不跑 cohort。

### 5.4 no-onset / tie / ambiguous 规则（锁死，每特征一致）

- **no-onset（unreached）**：CUSUM 在 detection window 内未越 λ → `detected=False`，该 (contact, feature) onset = NaN（"未招募"，**不**伪造时刻）。
- **tie**：同一帧多触点同时越阈 → fractional rank ties=`'average'`（沿用 Stage 1）。
- **ambiguous**：CUSUM 越阈后 1s 内回落到 < λ/2（瞬时假警）→ 标 ambiguous，该 (contact, feature) onset = NaN。
- per-contact 在某特征 NaN ≠ 该触点出局：fused rank（§6）按 available features 的 median 算；available features < 2 的触点 → 该触点 fused onset = NaN（出 echo）。

---

## 6. 融合规则（user P1 收紧 — 特征族结构，不是简单多数投票）

### 6.1 fused recruitment rank（主口径 — 用 pass-2 onset，4 个真特征）

- 每触点 c：收集其在 available **fused 特征（F1/F2/F3/F5）**上的 **pass-2** onset 时刻 `{onset_{c,i}}`；`fused_onset(c) = median_i onset_{c,i}`（feature-wise median，对单特征噪声稳健）。**ER 不进 median。**
- `recruitment_rank(c) = rankdata(fused_onset, ties='average')`（NaN 触点不进）。
- **family 不简单投票**：median 在 4 个 fused 特征上取，其中 F1/F2/F3 是 amplitude family、F5 是 spectral family——所以**同时**报下面的 agreement 量，让 fused rank 的可信度可被审视：

### 6.2 必报的 agreement 量（每 seizure）

| 量 | 定义 | 用途 |
|---|---|---|
| `pairwise_rank_corr` | F1/F2/F3/F5 四者两两 recruitment-rank Spearman 矩阵（ER 作单列对照行，不进 fused 统计）| 特征是否一致测同一招募 |
| `amplitude_family_agreement` | F1/F2/F3（LL/broadband/HFA）三者两两 ρ 的 median | amplitude family 内部一致性 |
| `spectral_support` | F5（spectral-edge）rank vs amplitude-family-median（F1/F2/F3）rank 的 ρ | spectral 轴是支持还是冲突 |
| `early_K_overlap` | **F1/F2/F3 amplitude family** "最早 K=3 触点"集合的 Jaccard（pairwise median）；F5 不进此量 | 端点（最先点着）是否一致 |
| `early_K_overlap_with_spectral` | 含 F5 的同一量（**仅 diagnostic，不进硬门**）| spectral 是否改变端点判读 |
| `feature_agreement_flag` | `amplitude_family_agreement ≥ 0.5` **AND** `early_K_overlap ≥ 0.3`（**两项都只用 amplitude family F1/F2/F3，F5 完全不进硬门**）| **进 primary 的门**（§3.2）|

### 6.3 spectral-edge 的特殊处理（user：不要求每次必过，冲突时降级）

- spectral-edge **不进 `feature_agreement_flag` 的硬门**（它可能噪声大）。
- 但 `spectral_support < 0`（spectral 轴与 amplitude family **冲突**）→ 标 `spectral_conflict_flag=True`，该 seizure `confidence=low`，进 sensitivity 单列；**不**因此 drop。
- `spectral_support > 0`（支持）→ confidence 加强（写进 per-seizure 记录，cohort 层报 "spectral-corroborated 子集"方向是否一致）。

### 6.4 ER held-out 一致性（construct-validity 核心）

- ER 走完整 baseline z + λ + CUSUM 管线得 `er_rank`，但**不进 fused median**（§5.1 决定）。
- `er_vs_fused_consistency = Spearman(er_rank, fused_rank)`（fused = F1/F2/F3/F5）。
- 这是 Stage 1 留下的 construct-validity 问题的**直接答案**：现成 ER 代理的招募顺序，和真仪器（4 真特征 fused）一致吗？cohort 层报其分布。**这是 Stage 2 对 Stage 1 的关键交代**——若 H1（真仪器）站住而 `er_vs_fused_consistency` 低，则写"Stage 1 的 ER 代理低估 / 错估了招募，真仪器才看到 echo"（§7.4 "代理 vs 真仪器分歧"档）。

---

## 7. 统计 contract（复用 Stage 1，不重造）

### 7.1 H1 primary — echo strength

- 对每次合格 seizure：`r_obs = max_m Spearman(recruitment_rank, template_m_rank)`（max over 模板条数 k，null 同样 max；复用 `src.topic5_echo_gate.echo_r_obs`）。
- null（within-seizure channel-label shuffle）→ `e_k`（standardized exceedance）+ `p_k`（复用 `compute_echo_strength`）。
- **合并：subject-level primary**（复用 `pool_echo_subject_level`）：`E_s = mean_k e_k` → one-sided Wilcoxon signed-rank + sign + bootstrap；per-seizure cluster-robust OLS 作 sensitivity，方向须一致。

### 7.2 更强 null（复用 Stage 1 §4.6，retained）

| Null | 构造 | 复用 |
|---|---|---|
| **channel** | 全通道身份打乱 | `shuffle_null(null_mode='channel')` |
| **within-shaft** | 同 shaft 内打乱 | `null_mode='within_shaft'`，`blocks=shafts`（`parse_shaft`）|
| **anchor-matched** | 按惯常早晚 / 病灶距离分箱内打乱 | `null_mode='anchor_matched'`，`blocks=anchor_bins` |
| **Null D（特异性）** | **改坐标 / region matched，不再按 channel name** | 见 §7.3 |
| **bad-data regression** | `e_k_baddata` 真 null 抽样合并须变平 | `bad_data_regression` |

- LOO 去锚（`compute_deanchor_echo`，`n_seizures≥4` primary）retained：把"稳定 earliness 锚"剥掉后是否仍 echo。

### 7.3 Null D 改坐标 / region matched（闭合 Stage 1 缺口）

Stage 1 Null D（别人模板）按通道名对齐 → epi 0 overlap、yuquan within-patient names → 跑不起来。Stage 2 改坐标 / region，但**两数据集坐标空间不同，不能一刀切**（reviewer P1 + Topic 0）：

`src.seeg_coord_loader.load_subject_coords`：**epilepsiae = `mni152_1mm`（跨病人可比）；yuquan = `fs_native_ras_mm`（subject-native，源码注释明确"No cross-subject registration; per-subject native space only"）**。所以：

- **Epilepsiae → MNI coordinate Null D**：把"别人模板"按 **MNI 坐标最近邻**映射到本病人触点（跨病人合法），再算 echo；合并方向应 ≈ 中性。
- **Yuquan → region / shaft / clinical-network matched Null D**：yuquan subject-native 点云**禁止**跨病人 pooled nearest-neighbor（坐标不可比）。改用 **解剖 region / shaft / 临床网络标签**匹配的 Null D（`results/lagpat_broad/yuquan_clinical_networks.json` 等 region 标签）。
- coord / region 标签缺失的 subject → Null D 标 `null_d_inapplicable`，进 sensitivity（不进 primary 硬约束），**不** silent 跳过。
- B0 audit 加列 `null_d_mode`（`mni_nn` / `region_matched` / `inapplicable`）。
- **§7.4 verdict 里 "Null D 中性" 的硬约束按 `null_d_mode` 分别判**：epi 用 mni_nn 结果，yuquan 用 region_matched 结果，不混。

### 7.4 判决合同（把结论写进数字；no-veto 原则不适用 — Stage 2 是真仪器本体）

> Stage 2 不是 proxy triage，它就是真仪器；其结论**可**进主文档（在 sensitivity battery + 更强 null + 坐标 Null D + user 视觉巡视全过后）。

| Verdict | 条件 |
|---|---|
| **站住·含具体通路** | subject-level H1 单侧 p<0.05 **AND** median E_s>0 **AND** sign/bootstrap 同向 **AND** per-seizure cluster-robust 同向 **AND** bad-data 变平 **AND** ≥10 subjects **AND** `feature_agreement_flag` primary 子集 **AND** anchor-matched **或** within-shaft 仍同向显著 **AND** LOO 去锚同向显著 **AND** Null D（坐标）中性 |
| **站住·稳定锚为主** | inclusive echo 成立 **但** anchor-matched + LOO 去锚变平 → shared ictal/interictal channel-priority anchor（非 specific path replay）|
| **代理 vs 真仪器分歧** | H1（真仪器）站住但 `er_vs_fused_consistency` 低 → 写"Stage 1 的 ER 代理低估 / 错估了招募，真仪器看到 echo" |
| **真仪器阴性** | H1 p≥0.15，≥10 subjects，feature-agreement + 坐标 Null D 干净 → "真仪器在干净情况下也没看到具体路径 echo"（这是 Stage 1 粗锚结论的真仪器确认）|
| **没看清** | <6 subjects eligible，或 B0 audit 显示结构不可解释，或 `feature_agreement_flag` primary 子集 < 6 |

---

## 8. narrow / broad 证据层级（user P1：不平级，Main-A / Main-B）

> **不是弱化 broad，而是避免 broad 未封板时污染主结论。**

### 8.1 Main-A：canonical narrow masked template（承接 Topic 1 cohort 发现）

- 来源：`results/interictal_propagation_masked/`（Stage 1 用的 canonical），per-cluster `valid_mask`，phantom-safe 硬合同。
- 角色：**主结论**。"发作真招募顺序是否 echo topic1 cohort 主发现里那条窄模板"。
- 所有 §7 verdict 以 Main-A 为准。

### 8.2 Main-B：broad-template extension analysis（更大临床网络，带 caveat）

- 来源：broad lagPat（`results/lagpat_broad/` Yuquan top_n=20 已封板 17/17 stable k=2；`results/lagpat_broad_dyn/` dynamic）。
- 角色：**扩展分析**，回答"在**更大的临床网络**里（发作招募的通道比窄模板多），是否仍 echo"——这正是 user 关心的"发作招募通道 > 高 HI 窄核心"。
- **强制 caveat（写进 verdict 文字）**：(i) Yuquan dynamic broad 已较强但 **Epilepsiae broad re-pack 未 sealed**（多日 running）→ Main-B 的 Epilepsiae 部分标 `broad_unsealed`，只进 sensitivity，**不进 Main-B 主估计**直到封板；(ii) broad **variable-k**（dynamic 出现过 litengsheng k=3）→ k 作 covariate + 出 `effect vs k` 图，k>2 子集单列。
- Main-B **不**覆盖 Main-A；两者并排报，但**主文档结论句以 Main-A 为承重**，Main-B 作"更大网络里的稳健性扩展"。

---

## 9. construct validity = cross-feature agreement（闭合 Stage 1 pending）

- Stage 1 construct-validity gate（人工 sentinel）→ Stage 2 机械化为 §6.2 的 agreement 量 + §6.4 的 `er_vs_fused_consistency`。
- cohort 层报：`feature_agreement_flag` 通过率、`amplitude_family_agreement` 分布、`spectral_support` 分布、`er_vs_fused_consistency` 分布。
- **sentinel 阶段（§13）仍人工目视**：抽 3–5 个 sentinel seizure，把 raw trace + 5 个特征 trace + 各自 onset + fused onset 画在一起，人工确认"最早点着的那批触点"在 5 个特征下大体同向——这是仪器**上线前**的一次性人工 construct-validity，cohort 层之后靠机械 agreement 量。

---

## 10. sensitivity battery（必报）

- normalization：robust(MAD) primary vs mean/std。
- `RECRUIT_POST_SEC`：15s primary vs 10/20s。
- anchor：data-driven primary vs annotation（epi 双锚；yuquan 单锚退化）。
- MIN_CH：8 primary vs 10。
- B（shuffle 次数）：1000 vs 2000。
- 含 / 不含 spectral-edge 的 fused rank（spectral 是否改变结论）。
- template k：k=2 primary vs 含 k=1 / k>2（Main-A + Main-B 各报）。
- Main-A（narrow）vs Main-B（broad）方向是否一致。

---

## 11. 代码 / 数据架构

### 11.1 复用（不重造）

| 需要 | 复用来源 | 匹配？|
|---|---|---|
| raw ictal EEG 窗 | `src.ictal_onset_extraction.extract_seizure_window`（epi+yuquan）| ✅ 双数据集；**Main-A 的 ictal reference 按数据集匹配检测 reference：yuquan=`bipolar`(alias-left)、epilepsiae=`car`（§3.4 traced）** |
| ER trace | `src.ictal_onset_extraction.compute_er` | ✅ ER held-out 参照；F2/F3/F5 仿其 spectrogram 骨架 |
| baseline 窗 | `resolve_baseline_window` | ✅ EEG-onset-aware |
| baseline z-score 骨架 | `baseline_zscore_er` | ⚠️ 新写 `baseline_robust_z`（MAD）沿用其 mask 逻辑 |
| λ calibration | `src.ictal_er_rank.calibrate_lambda_per_subject` | ✅ 每特征独立校准 |
| change-point onset | `compute_cusum_n_d_with_time` | ✅ **通用 wrapper，非 ER 专用** |
| echo 统计 + null | `src.topic5_echo_gate.*`（Stage 1，36 tests）| ✅ **直接复用，不重造统计量** |
| shaft 解析 | `src.propagation_skeleton_geometry.parse_shaft` | ✅ |
| 坐标 / region（Null D）| `src.seeg_coord_loader.load_subject_coords` | ✅ 闭合 Stage 1 缺口；**epi=mni152_1mm（NN 合法）/ yuquan=fs_native_ras_mm（只能 region matched）**（§7.3）|
| masked 模板加载 | Stage 1 runner 的 masked loader 逻辑（`results/interictal_propagation_masked/`）| ✅ Main-A；Main-B 加 broad 树 |

### 11.2 新代码（focused 新模块）

- **新建 `src/topic5_ictal_recruitment.py`（pure，no I/O）**：
  - `line_length_trace(signal, fs, *, win_sec, hop_sec)` → (n_ch, n_frames)。
  - `band_power_trace(signal, fs, band, *, win_sec, hop_sec)` → F2/F3。
  - `spectral_edge_trace(signal, fs, *, edge=0.9, win_sec, hop_sec)` → F5。
  - `baseline_robust_z(trace, baseline_idx_window, *, hop_sec, min_baseline_valid_sec)` → MAD z。
  - `detect_contact_onset(z_trace_1d, *, lam, search_start, ambiguous_drop_sec)` → onset frame（包 no-onset/ambiguous 规则；内部调 `compute_cusum_n_d_with_time`）。
  - `resolve_global_onset(provisional_onsets, n_valid, *, frac, persist_sec, hop_sec)` → `t_global`（§4.2）。
  - `fuse_recruitment_rank(per_feature_onsets, families)` → `(fused_rank, agreement_dict)`（§6）。
  - `feature_agreement(per_feature_ranks, families, *, early_k)` → §6.2 量 + `feature_agreement_flag`。
- **新建 `scripts/run_topic5_ictal_recruitment.py`**：`audit / sentinel / per-subject / cohort / figures` 子命令。masked 模板（Main-A narrow + Main-B broad）；raw EEG via `extract_seizure_window`；echo via `src.topic5_echo_gate`。`audit` 与 `sentinel` 先跑、人工看过再 `per-subject`/`cohort`。
- **新建 `scripts/plot_topic5_ictal_recruitment.py`** + `figures/README.md`。
- TDD（`tests/test_topic5_ictal_recruitment.py`）：合成已知招募波（注入已知 onset 顺序）→ 4 fused 特征 trace + onset 应复原顺序；spectral-edge 在"低压快活动无幅度上升"合成信号上仍触发而 amplitude 特征不触发（证明 F5 的独立轴价值）；global-onset 把"发作前慢漂移"判 not-global；**单个瞬时假警触点不足以触发 global onset（≥15% 且 ambiguous-drop 过滤生效）**；pre-onset-change flag 把 < −10s 的 onset 排除；fuse median 对单特征噪声稳健；feature_agreement_flag 在一致 / 冲突合成上分别 True/False；HFA 在低 fs 上 feature-drop 而非 seizure-drop。
- **reviewer-mandated TDD（P0/P1）**：
  - **montage hard fail**：同名单触点、一路 CAR-monopolar 一路 bipolar-aliased → `channel_identity_contract` 检查 **hard raise**，名字相等不得通过（§3.4）。
  - **λ 短 baseline**：pooled baseline < `MIN_POOLED_BASELINE_SEC` → 返回 `calibration_unstable=True`，该 (subject,feature) 不进 primary（§5.3）。
  - **global onset 非空条件**：构造"招募比例先到 0.15 又因 ambiguous 撤回"的合成 → 验证撤回的 onset 不计入 fraction（§4.2）。
  - **Null D 分数据集**：epi 走 `mni_nn`，yuquan 走 `region_matched`；断言 yuquan **不**调用跨病人坐标 NN（§7.3）。
  - **spectral conflict 不 drop**：`spectral_support < 0` 的 seizure → `spectral_conflict_flag=True` + `confidence=low`，但**仍进 primary**（不被 drop），且 `feature_agreement_flag` 只看 amplitude family（§6.2/§6.3）。

### 11.3 输出

```
results/topic5_ictal_recruitment/
├── b0_recruitment_audit.csv         # §3.1 全列（先跑先看）
├── sentinel/                        # §13：3–5 seizure 的 raw+5特征+onset 叠图 + 人工核对记录
├── per_subject/<ds>_<sid>.json      # per-seizure: 5特征 onset + fused rank + agreement量 + e_k(每null) + 双锚 + Main-A/B
├── cohort_recruitment_summary.json  # subject-level verdict（Main-A 主 + Main-B 扩展）+ 每null + 去锚 + 坐标Null D + sensitivity + construct-validity分布
└── figures/
    ├── README.md                    # 中文逐图（图生成后写）
    ├── recruitment_echo_forest.png  # subject-level E_s（Main-A）+ pooled + 坐标Null D + bad-data
    ├── construct_validity.png       # er_vs_fused + amplitude-family agreement + spectral_support 分布
    ├── null_mode_panel.png          # channel/within-shaft/anchor-matched 合并方向
    └── narrow_vs_broad.png          # Main-A vs Main-B 方向对比（broad caveat 标注）
```

**工程 locks**：RNG seed 固定且落盘；HOP=0.1s 全特征统一；每特征 λ 落盘；CUSUM no-onset=NaN 不伪造；channel-order mismatch hard fail；坐标缺失 / HFA 低 fs / global-onset 解不出 / spectral conflict 全部 flag 不 silent；每 subject forest + sentinel 叠图人工巡视后才进 cohort 结论。

---

## 12. Caveats & 显式 NOT-DO

### Caveats
1. **exploratory**：sensitivity battery（§10）+ 更强 null（§7.2/7.3）+ user 视觉巡视全过前，不写 paper-level cohort claim。
2. **pre-ictal 状态层是另一个 spec**：本 spec 只做 early-ictal 招募仪器；不在此写 pre-ictal 状态结论（user-locked）。
3. **broad 未封板**：Main-B 的 Epilepsiae 部分在 broad re-pack sealed 前只进 sensitivity（§8.2）。
4. **招募 ≠ 漂移**：recruitment rank 只从 §4 windowed 数据驱动 global onset 附近的持续改变定；full extraction 窗里的 pre-ictal 漂移不进 rank（§0 verbatim lock）。
5. **spectral-edge 可能噪声大**：不进硬门，冲突降 confidence（§6.3）。
6. **HFA 受 fs 限**：低 fs subject HFA feature drop（非 seizure drop）。
7. **annotation 不准**：故主口径用 data-driven anchor；annotation 双锚 sensitivity（epi）/ 退化（yuquan）。

### 显式 NOT-DO
- 不搭 pre-ictal 状态层（另 spec）。
- 不做 directed connectivity / Bayesian / neural-field。
- **不把 EI 当独立 feature**（文献定位，双计数风险）。
- **不把 ER 默认参数照搬到别的特征**（每特征独立 baseline/λ/window）。
- **不在 trace 出模板真实 montage 前就当 ictal montage 对**（§3.4）；**不按名字相等当 channel identity**（必须 montage 语义匹配，否则 hard fail）。
- **不用单条 seizure 短 baseline 的 per-hour λ**（必须 pooled；< MIN_POOLED 标 calibration_unstable，§5.3）。
- 不按 channel name 做 Null D（改坐标 / region）；**不在 yuquan subject-native 坐标上做跨病人 NN**（改 region matched，§7.3）。
- **不因 spectral conflict drop primary seizure**（只降 confidence，§6.3）；**不让 F5 进 feature_agreement 硬门**（§6.2）。
- 不让 broad 未封板部分进 Main-B 主估计。
- 不在 within-subject 写 α-claim 当机制因果。
- 不跑 sentinel 通过前就跑 cohort（§13 staged gate）。

---

## 13. Staged execution（sentinel-first gate，user 最小修改路线）

1. **写 spec**（本文件）→ user review。
2. **写 implementation plan**（TDD task-by-task）→ user review。
3. **实现 instrument**（`src/topic5_ictal_recruitment.py` + runner，TDD 绿）。
4. **montage trace（sentinel 前置，P0）**：先 trace interictal 模板的真实 montage（monopolar vs bipolar-aliased-left），落 `template_montage`；据此定 ictal `reference`。**montage 没 trace 清楚 → 不进 sentinel。**
5. **B0 audit + sentinel**：跑 audit；抽 **3–5 个 sentinel seizure**（覆盖 epi + yuquan，至少 1 个低压快起始候选），生成 §9 叠图 + **每特征 `lambda`/`pooled_baseline_sec`/`no_onset_rate` 表**，**人工目视核对**：5 detector 是否同向、global-onset 是否合理、pre-onset-change flag 是否生效、montage 对齐后通道是否对得上、λ 下 no-onset rate 是否过高。**sentinel 不过 → 回炉特征 / 窗口 / λ 参数，不跑 cohort。**
6. **sentinel 通过后** → 跑 `per-subject` + `cohort`（Main-A 主 + Main-B 扩展）。
7. **figures + 人工巡视** → 写 archive doc（`docs/archive/topic5/ictal_recruitment/`）→ 仅在全 sensitivity 过后回链主文档。

---

## 14. 来源 / 关联文档

- `docs/superpowers/specs/2026-06-08-topic5-ictal-template-echo-gate-design.md` — Stage 1 spec（echo 统计 + null 来源）
- `docs/archive/topic5/echo_gate/stage1_proxy_triage_2026-06-08.md` — Stage 1 结论（粗锚 vs 路径；construct-validity + Null D 缺口来源）
- `docs/superpowers/specs/2026-05-10-topic5-subtype-onset-features-menu.md` — onset-feature menu（pre-ictal 状态层 secondary spec 的来源）
- `src/ictal_onset_extraction.py` / `src/ictal_er_rank.py` — raw EEG 窗 + ER + baseline + CUSUM + λ calibration（复用骨架）
- `src/topic5_echo_gate.py` — Stage 1 echo 核（36 tests，直接复用）
- `src/seeg_coord_loader.py` — 坐标 / region（坐标 Null D）
- `results/lagpat_broad/COHORT_SUMMARY.md` — broad 模板（Main-B 来源 + 封板状态）
- AGENTS.md Cross-PR：lagPatRank phantom（masked 必经）、`channel_names` ordering、Topic 4 H2 input source order（rank-displacement swap-k 与本 spec 无关，勿混）
