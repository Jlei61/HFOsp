# Topic 5 V3p — 发作前非轴向轨迹（preictal-only）：完整硬门阴性

date 2026-07-05 · 前身 V3a（P3→I1 onset 对比，脆弱候选信号）· 姊妹 V2a（restricted-axial，偏阴性）
原执行分支 `topic5-v3p-preictal-trajectory`（off V3a HEAD `ac042f3`，只加新文件、read-only 继承 V3a）；2026-07-12 已整合入文章工作分支并同步 canonical docs
结果：**tier 0 / `pre_registered_negative=true` / `state_v3p_supported=false`**

---

## 0. 摘要（朴素话）

病人两次发作之间，短暂高频异常放电按一条走熟的小路在电极间传开（间期传播轴）。有人猜发作快启动时，系统会把"沿小路的有序传播"松掉、转到小路**之外**去。V3p 换最干净的窗口再问一次：**只看发作真正开始前的两分钟**（−120~−10 秒，锚在每次发作的电生理 onset、完全不碰发作本身），问小路之外的连锁流和最易放大方向是不是随发作临近**逐渐爬升**、而且**专门压在真非轴向触点**上。

**没看到。** 全 `n_perm=1000` 校正后：narrow 主队列 7 人、broad 复制队列 13 人（含策展核心 9），两个承重量都没有队列方向，没有一个被试过完整机制硬门。broad 里有**少数**被试在单条 label-null 上碰巧显著（`253/620/139` 的流、`1084/916` 的模态），但它们散在不同被试/不同腿、方向不一致，且**全被更严格的 rate / lag1-specific / phase / block / 双 span 门筛掉**——正是这些硬门要滤掉的"激活多 / 同步共激活 / 频谱-平滑假象"。所以这是一次**完整硬门阴性**（不是"发作前完全没有非轴向变化"），加固 Topic 5 主线：间期 HFO 轴是患者内共享的**粗骨架**，但没有证据表明它在发作前被逐渐重放或搬迁。

---

## 1. 为什么在 V3a 之后做 V3p

V3a（`docs/archive/topic5/v3a_mode_transition_2026-07-04.md`）用"发作前 P3（−30~−10s）对比发作后 I1（+10~+30s）"的同发作配对差，非轴向净流腿在主/复制队列机械上达到 tier 4；但原始流多在下降、lag1≈lag0、个体完整稳健门几乎全塌，模态腿也全阴，所以只能算**跨 onset 的脆弱候选信号**。其命门是 **I1 落在发作已点着之后**——信噪比塌 / 饱和 / 全场淹没都可能污染终点。V3p 不再用发作后终点，只看纯 preictal 段的斜率，专门检验这个候选信号是否在 onset 前已经表现为渐进式轴向→非轴向搬迁。

## 2. 测了什么（设计）

- **时间**：只用 preictal 相位 `P0..P3`（−120~−90 / −90~−60 / −60~−30 / −30~−10s），eeg-onset 锚，完全不碰 O/I1/I2/I3/Post。每发作切 10s/5s 滑窗，实测 ~17-18 窗/发作。
- **双 span**：headline `full=[−120,−10]` + jitter-safe `guard=[−120,−20]`；强支持须两轨同过（只 full 过→`near_onset_dependent` 降级）。
- **co-primary（承重）**：H3p-b 非轴向连锁流 `net_offaxis_flux_surplus_slope`（avalanche `i≠j`，source_mean，lag1）+ H3p-c 最易放大方向 `mode_shift_density_surplus_slope`（low-rank VAR 主奇异向量映回、density-归一 P_N−P_A）。`surplus = obs_slope − label-null 中位`；Theil-Sen 斜率、被试=median over seizures。
- **裁判 = 同杆 label-null**（主）：把 axis/非-axis 标签在杆内打乱、重算整条斜率——控"发作前全场升温"。回归残差（vs global+axial energy）作保守 sensitivity。
- **硬门（rev1）**：流须 `direction ∧ p_label ∧ p_rate（逐窗保率）∧ lag1_specific>0（lag1−lag0，剔同步共驱动）∧ 两 span`；模态须 `direction ∧ p_label ∧ p_phase ∧ p_block（时间 surrogate）∧ 两 span`（strong vs weak grade）。
- **H3p-a 轴向减弱** supportive-only（`module_support_flag_a` 恒 False）；**H3p-d** 负担/自持(lag1-specific)/相对增益(gain_shift) secondary。
- **纪律**：EXPLORATORY、无 forecasting、预注册阴性可接受、不救 1125、几何/非轴向定义对 ictal+preictal 全盲（防循环）。

## 3. 预注册的 support 定义

- **subject_support** = `(H3p-b 硬门 OR H3p-c 硬门) ∧ onset_jitter_pass ∧ ¬single_contact_driven ∧ leave_one_contact_pass ∧ axis_only_control_pass ∧ ¬near_onset_dependent(支持腿) ∧ ¬label_null_underpowered`。H3p-a 只加强，永不单独。
- **cohort**：被试 `slope_label_z` 做单侧 Wilcoxon，H3p-b/H3p-c **Holm** 校正。
- **tier 0–5**：0 无 / 1 仅方向 / 2 ≥1 被试 support 无队列方向 / 3 narrow 队列 co-primary Holm 过 + subject_support 计数 ≥2 / 4 narrow tier-3 且 broad 同向复制（broad_core 也须同向）/ 5 模型侧。`state_v3p_supported=tier≥3`，V3p 最高 tier 4。narrow+broad 永不 pool。

## 4. 工程 / QC

- **实现**：17 个实现 commit（`84ba936`..`45c6233`，11 files，+3382）+ 2 个验收硬化 commit（`dbe72b4` admission/degenerate、`1caa656` per-cohort README）。read-only 复用 V3a 全部机制，**只加 V3p 新文件、未碰任何 V3a/V2**。整合到当前 V3a API 后测试合同为 11 个纯单测 + 16 个 integration，共 27 个。
- **队列**：narrow 7（1096 1125 1146 253 384 442 958）；broad_core 9（139 253 1077 1096 1125 1150 620 635 916）；候选 4（1084 583 590 922，**全 admit**）；broad_expanded=13。yuquan 2（xuxinyi zhangkexuan）**pre-gate 丢弃**（间期 HFO 参与度在 V3p mount 加载不了——Yuquan 数据不在 epilepsia LAGPAT_ROOT）。narrow/broad 不受影响。
- **门校准（gate-lock）**：broad `axis_participation_gap_min` **0.15→0.0**——broad 几何轴（`propagation_geometry_broad` 插值场）含 0-参与度触点，参与度分离判据对每个 broad 病人塌成 0.0；门改由 `axis_rank_distinct≥5`（broad_core 12-20、候选 13-18）+ geometry + rank_displacement_json 裁。narrow 保完整门（gap 0.23-0.59、7/7 过）。**broad 门与 narrow 不同构 → broad 只作复制/敏感性，不反向定义主结论**（broad_core / broad_expanded 分开报）。
- **admission artifact**：`results/topic5_ictal_recruitment/v3p_preictal_trajectory/admission.json` 落盘为权威名册（broad_core / candidates / admitted / excluded+reason / broad_expanded / gate 阈值）；trajectory 从它读 broad_expanded（config fallback + warn）；summary JSON `provenance` 段记录。
- **退化-null 显式 guard**：`flux/mode_label_null_mad_zero` = label-null p 有限但 z 非有限（零 dispersion / MAD=0）。实测 `n_flux_label_null_mad_zero=1`（narrow `1146`：obs surplus=0、p=0.523、**非阳性**；其 z 被 nan-filter 排出队列 Wilcoxon）。summary 记 count + `degenerate_null_note`。

## 5. 主结果（`n_perm=1000`）

### 5.1 队列层（全阴）

| 队列 | subject_support | H3p-b 流 median-z / Wilcoxon p / **Holm p** | H3p-c 模态 median-z / Wilcoxon p / **Holm p** |
|---|---|---|---|
| **narrow (7)** 主 | **0/7** | 0.000 / 0.642 / **1.000** | −0.097 / 0.656 / **1.000** |
| **broad_expanded (13)** 复制 | **0/13** | +0.141 / 0.580 / **0.685** | +0.148 / 0.342 / **0.685** |
| **broad_core (9)** | **0/9** | +0.296 / 0.326 / **0.652** | +0.090 / 0.545 / **0.652** |

所有 median |z| ≤ 0.30，Holm 后全部 p ≥ 0.65，无一方向/队列过 α=0.05。tier=0。这不是 power 差一点没过——是 **effect size 本身近零、正负混杂**（不是"方向一致但 n 小"）。

### 5.2 逐被试（narrow，全 supp=F）

| 被试 | nsz | 流 surplus / z / p_b | 模态 surplus / z / p_c |
|---|---|---|---|
| 1096 | 8 | +1.0e-4 / +0.12 / 0.451 | +1.7e-5 / +0.58 / 0.309 |
| 1125 | 13 | 0.0 / 0.00 / 0.549 | −1.4e-5 / −0.51 / 0.706 |
| 1146 | 23 | 0.0 / **nan(MAD=0)** / 0.523 | −4.0e-7 / −0.10 / 0.596 |
| 253 | 6 | 0.0 / 0.00 / 0.698 | +5.6e-5 / +0.95 / 0.166 |
| 384 | 10 | −9.4e-4 / **−2.12** / 0.946 | −8.1e-6 / −0.64 / 0.730 |
| 442 | 20 | −2.3e-4 / −0.59 / 0.724 | +2.1e-6 / +0.16 / 0.438 |
| 958 | 12 | +9.5e-5 / +0.67 / 0.383 | −1.2e-5 / −1.26 / 0.917 |

narrow 无一被试正向-显著；384 的流甚至显著**下降**。

### 5.3 broad 单项 label-null 命中（scattered，全被硬门筛掉，全 supp=F）

- **H3p-b 流 label 显著**：`253`(z=+3.56, p_b=0.021)、`620`(+1.63, 0.039)、`139`(+1.53, 0.038)。
- **H3p-c 模态 label 显著**：`1084`(z=+2.38, p_c=0.006)、`916`(+1.95, 0.022)。
- **反向的大 z**：`1077` 流 z=−3.35（强下降）、`922` 流 z=−1.91、`139` 模态 z=−1.99。

这些是 **single-null nominal hits, filtered by prespecified hard gates**——分散在不同被试/不同腿、方向不一致，无一过 label∧rate∧lag1_specific∧两span（流）或 label∧phase∧block∧两span（模态）的完整机制门。正是硬门设计要滤掉的假阳性类型（激活多 / 同步共驱动 / 频谱-平滑）。**不能写成"潜在阳性"。**

## 6. 能说什么

> 在当前 SEEG 覆盖、1–45 Hz 宽带包络、−120~−10 秒 preictal-only、HFO-defined non-axis、label/rate/phase/block null 的 V3p 设计下，**没有证据支持发作前非轴向流或非轴向最易放大模态随 onset 临近逐渐爬升**；队列层面与个体完整硬门支持均为阴性。少数单-null nominal 命中被预设硬门滤除。

## 7. 不能说什么

- 不能说"没有临界性 / 发作前没有任何 state 变化 / HFO 轴与发作无关 / 所有病人非轴向机制都不存在"。
- 不能说"发作前完全没有非轴向变化"——是**完整机制门下没有稳定一致的非轴向 preictal ramp**。
- `geometry_insufficient` / 短记录 ≠ negative（本队列几何全充裕、无此情形）。broad 门与 narrow 不同构 → broad 不反向定义主结论。

## 8. 与 Topic 5 主线的关系

三层已较完整：V2a（restricted-axial preictal criticality，偏阴）、V3a（P3→I1 跨 onset 非轴向净流，机械 tier 4 但科学上脆弱）、V3p（preictal-only 非轴向爬升，**tier 0 干净阴性**）。三者共同把“发作前渐进式非轴向搬迁/重放”从主叙事降级，但不把 V3a 的 onset-crossing 候选变化直接判死。2026-07-12 已接受的主线口径：

> **共享粗骨架存在；渐进式 preictal relocation 未获支持；onset-crossing 变化仍是脆弱候选。** 间期 HFO 传播轴可作为患者内稳定骨架读出，但当前数据不支持它在发作前最后两分钟内逐渐转成非轴向流或模态。

## 9. QC 明细（验收前透明化）

1. **NaN-z 处理（1146 流）**：label-null dispersion=0 → z 不可定义、但 p 可算（0.523）；obs surplus=0，**不构成阳性**。cohort Wilcoxon 用 `slope_label_z`，对 non-finite z 做 nan-filter 排除（不置 0、不 fallback）。已显式记 `n_flux_label_null_mad_zero=1` + `degenerate_null_note`。
2. **broad 门不同构**：见 §4（gap_min 0.0）；broad_core / broad_expanded 并列汇报，broad 只作复制/敏感性。
3. **yuquan exclusion**：工程路径问题（参与度加载不了），非"效应失败"；admission.json `excluded` 记原因。
4. **axis-only control**：Task 7 审阅已确认——**流腿的 axis-only control 近乎 trivial**（非轴集清空→flux 恒 0→塌成 direction），**模态腿的 axis-only control 非平凡**。V3p 阴性不依赖它过滤（无阳性需过滤）；subject_support 已不把流腿 axis-only 当独立证据。
5. **safe-span**：两轨（full `−120~-10` / guard `−120~-20`）都要过；输出含 `*_guard` 列 + `near_onset_dependent`。阴性对两 span 一致（即便避开 −10~0 也无 preictal ramp）。

## 10. 下一步（建议，非本 archive 承诺）

- **不建议**继续为 preictal ramp 加窗/换阈值/换指标（易成 fishing）；V3p 已对这条具体假设形成完整硬门阴性。V3a 的跨-onset 候选信号应作为不同时间边界的问题单独保留。
- **建议**：(A) 只做少量阴性稳健性敏感性（z_threshold 1.5/2.0/2.5、non-axis participation 阈、full vs safe span、global/axial positive-control slope），目的是让阴性更可信而非救主假设；(B) 把 V2a/V3a/V3p 合成 bounded synthesis（shared scaffold ≠ gradual preictal relocation）；(C) V3b 定位为**模型 observability / falsification**（M3B Jacobian 在哪些参数预测非轴向 transient amplification、投影到 SEEG 覆盖后是否可见），而非"救 V3p"。

---

**产物**：`results/topic5_ictal_recruitment/v3p_preictal_trajectory/{admission.json, narrow/, broad/}`（trajectory CSV + `v3p_cohort_tier.json` + `v3p_summary_subject.csv` + `figures/`）。正式跑批采用 80 核 / 20 job、总时长约 7h35m；可复现入口是已提交的 `scripts/run_topic5_v3p_{feasibility,trajectory,summary}.py` 与 `scripts/plot_topic5_v3p_summary.py`，不保留硬编码旧 worktree 路径的临时 `.superpowers/sdd` launcher。原分支最终图版为 `topic5-v3p-preictal-trajectory`@`ea0d7be`；2026-07-12 已完成用户签核后的文档整合与合并。
