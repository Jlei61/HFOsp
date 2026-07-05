# 发作内 field 动力学 pilot — 轴向走廊 vs 非轴向随发作进程（2026-06-28, exploratory）

## 摘要（朴素话）

**测什么**：一次发作从头（onset）到尾（offset），脑电活动的"空间形状"怎么随时间变。把电极触点分块：
两个间期传播模板（A/B）各自**最早响应**的小核（假设的着火点 / 端点）、这两个核**之间**的"走廊"中段、
以及离走廊远的横向触点。问：随发作进程，走廊（轴向）的相对活动是不是减弱、横向（非轴向）是不是增强、
整体是不是更同步、场的主方向有没有漂移。

**怎么测**：每隔几秒取一个 10s 窗，算每个触点相对**发作前安静期**的 robust-z（z-ER，和发作早期 A 线同一把尺子），
按上面分块比相对占比；窗按发作进程（0→100%）排列；终止前后另按 offset 对齐看。**统计**：每次发作算
progress 与"轴向中段正质量占比"的 Spearman ρ（期望<0）、与"非轴向占比"的 ρ（期望>0）；每个被试再对它各次
发作的 ρ 做 Wilcoxon（发作=重复单位，避免一次发作内相邻窗自相关灌水）。

**揭示什么（描述性，pilot，不是结论）**：发作场随时间**确实在变**（GIF 可见）、整体**大致仍落在间期轴上**
（maxAB 多数中后期不降反升）；但"轴向走廊变弱 / 非轴向变强"这个**具体方向假设不稳健**——**broad 队列有暗示**
（轴向 median ρ<0 在 5/8、非轴向 ρ>0 在 **8/8**），**但 narrow 扩队列不复现甚至反向**（轴向 ρ<0 仅 **3/7**、
非轴向 ρ>0 仅 **2/7**；E1146 轴向 ρ=+0.52、E442 +0.37 明确反向）；两队列各只有 1 个被试 Wilcoxon 显著。
**= 方向不是稳健现象，依队列/substrate 而变（扩队列调查把 broad 的暗示证否了一半）。** E916（非 swap）证明
走廊几何上确可在非 swap 被试构建（n_axial_mid=4），但其发作太短（中位 8s、仅 1 个≥40s）→ 无法贡献趋势。

## 1. 队列与为什么是这 8 个

**队列 = 8 个 swap-positive broad ECoG**：`epilepsiae_{139,253,1077,1096,1125,1150,620,635}`
（= `field_vs_ictal_swap` 参考图那批，swap_class strict/candidate；与 `ictal_direction_clustering_2026-06-27`
的 6 个"几何干净 ECoG"是不同选法）。

**为什么需要 swap-positive**：本分析的"走廊"= 间期模板 A 的早端核 ↔ 模板 B 的早端核**之间**的中段。要有
"中段"，两个核必须在轴的**两端分开**。swap（A、B 互为反向）天然把两核摆到一根轴两端 → 走廊可定义。
原来挑的 6 个 narrow 几何干净 ECoG 大多 `swap=none`（两模板非反向）→ 两核共位 / 轴退化 → 没走廊
（实测只有 442 勉强有），故**对"走廊"这个问题选错了队列**，已降为 control（cache 留盘）。

**走廊 still 受电极采样限制**：即使 swap、即使用紧凑核，走廊只在"电极采到两核中间"时有触点。**7/8 可测**；
**253 双侧植入、两核分跨左右半球、中段无电极 → 走廊不可测**（如实标 `n_axial_mid=0` / not measurable，不强测）。

## 2. 口径（全部沿用现有，未新发明）

- **高质量发作**：`t0_eligibility_audit.csv` `analysis_eligible` ∧ inventory `has_complete_eeg_interval`。
- **baseline z-ER**：现有自适应 `[-pre,-60]`(pre≥120)、逐通道 median/MAD，复用 `extract_seizure_window`
  + `baseline_robust_z`（与发作早期 A 线**完全一致**，数值 parity 证毕，见 §4）。band：bb 1–45Hz primary，
  hfa 60–100Hz secondary（都算，CSV 各一行）。
- **长窗 cache**（新建，现有 v2_windows 只到 onset+20s 不够到 offset）：每发作抽 onset−130s →
  `offset+90s`（`post_sec=ceil(max(eeg_offset_rel,duration))+90`；`span>600s` 疑似 status → drop）。
- **间期 A/B 场**：几何 record `typical_rank`（t_a/t_b），与 `run_topic5_axis_alignment` 同源。
- **source 核 = compact**：每模板最早 top-2（间距<15mm 取双点否则单点 fallback）。**关键教训**：
  - 不能用 full swap 集合（k=10 的整串 src_a/src_b）作核 → 核太大吃光中段 → axial_mid=0（139/253 实测）。
  - 也不能用 narrow 单点（轴退化）。compact top-2-3 是对的（139 → axial_mid=3）。
  - `decision_k` 只作 provenance，不定核大小。
  - **图上红蓝圈仍画 full swap 集合（分布式，匹配参考图，display 用）；轴/走廊数学用 compact 核（analysis 用），两者口径分开。**
- **四分区 MECE**：`source_core`（两核）/ `axis_end_noncore`（贴轴靠端）/ `axial_mid`（中段贴轴=走廊，**检验对象**）
  / `non_axial`（离轴远=对照）。退化轴（两核共位，L<0.15·bbox_diag）→ 标记不进结论。
- **占比 = `positive_mass_share`** = `Σmax(z,0)_组 / Σmax(z,0)_全场`（有界 [0,1]，四组和=1；空组返回 **NaN**
  = not measurable，非 0）。比绝对 z 稳，对 z-ER 中后期退化不那么敏感。
- **其它指标**：场-轴对齐 `maxAB |corr_pair_mirror_invariant|`（窗活动场 vs 间期 A/B 场）；同步 = 窗内触点
  robust-z trace 两两 Pearson median；梯度方向漂移（fold [0,90]）。
- **窗**：onset 滑窗 10s/step5s 到 offset；progress 0–100%；终止窗 `[-60,-30,-10,0,+30]s` rel offset
  （左缘<onset 标 `pre_onset_overlap` 排除）；onset 窗加 `ictal_fraction`（窗内落 [0,offset] 比例），
  ictal 轨迹默认 `ictal_fraction≥0.5`（短发作 [0,10] 跨 offset 不当 ictal）。

## 3. 趋势统计结果（`results/topic5_ictal_recruitment/field_dynamics/trend_stats.json`）

每次发作 Spearman ρ(progress, 占比) → 每被试对各发作 ρ 做 Wilcoxon（单边）：

| subject | 走廊 n_sz | 轴向中段 median ρ (期望<0) | frac ρ<0 | Wilcoxon p | 非轴向 median ρ (期望>0) | frac ρ>0 | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| 1077 | 7 | −0.37 | 0.71 | 0.188 | +0.49 | 0.57 | 0.109 |
| 1096 | 8 | −0.30 | 0.88 | 0.098 | +0.76 | 1.00 | **0.004** |
| 1125 | 13 | −0.68 | 0.69 | **0.009** | +0.54 | 0.54 | 0.137 |
| 1150 | 7 | **+0.50** | 0.29 | 0.961 | +0.12 | 0.62 | 0.422 |
| 139 | 4 | −0.51 | 0.75 | (n<6) | +0.09 | 0.50 | (n<6) |
| 620 | 4 | **+0.23** | 0.25 | (n<6) | +0.06 | 0.75 | (n<6) |
| 635 | 15 | −0.12 | 0.53 | 0.640 | +0.34 | 0.53 | 0.681 |
| 253 | 0 (NA) | — | — | — | +0.01 | 0.50 | 0.656 |

（broad 还含 E916：44 sz 但中位 8s、仅 1 个 ≥40s → 趋势 n=2 不可用，证明非 swap 走廊几何上可建但短发作无趋势。）

**broad cohort（9，pilot 描述）**：可测走廊 8 个；轴向 median ρ<0 在 **5/8**、非轴向 median ρ>0 在 **8/8**；
per-subject Wilcoxon p<0.05：轴向 **1/8**（1125）、非轴向 **1/8**（1096）。1150/620 轴向反向。

### 3b. narrow 扩队列（平行批，`results/topic5_ictal_recruitment/field_dynamics_narrow/trend_stats.json`）

用每模板**端点 compact core** 在 narrow substrate 跑 7 个（1096/1125/1146/253/384/442/958，含非 swap 442 + 长发作 E1146 23sz）：

| subject | n_sz | 轴向 median ρ (期望<0) | Wilcoxon p | 非轴向 median ρ (期望>0) | Wilcoxon p |
|---|---|---|---|---|---|
| 1096 | 8 | +0.19 | 0.727 | −0.06 | 0.727 |
| 1125 | 13 | −0.00 | 0.580 | −0.53 | 0.998 |
| 1146 | 21 | **+0.52** | 0.962 | +0.20 | 0.237 |
| 253 | 6 | +0.40 | 0.891 | −0.68 | 1.000 |
| 384 | 9 | −0.07 | 0.102 | −0.65 | 0.992 |
| 442 | 19 | **+0.37** | 0.992 | +0.22 | 0.445 |
| 958 | 8 | −0.75 | **0.008** | −0.59 | 0.770 |

**narrow cohort**：轴向 median ρ<0 仅 **3/7**、非轴向 ρ>0 仅 **2/7**；只有 958 轴向显著。**→ broad 的方向暗示在
narrow 不复现、甚至多数反向（1146/442/253 轴向 ρ>0）。**

**两队列合起来的诚实结论**：发作场在变（GIF 可见）、整体仍贴间期轴（maxAB 不降反升）；**但"轴向走廊变弱/非轴向
变强"不是稳健现象——broad 暗示、narrow 证否**。依队列/substrate 而变；narrow 走廊多 n_axial_mid=2（薄、噪）。
不写成机制/cohort 规律。`align_maxab` 多数被试中后期反而升高（轴共享在时间上保持，与 A 线一致）。

## 4. 机制无关健康检查（全过）

- **baseline parity**：长 cache 的 `bb_auc[0,10]` 与现有 `t0_feature_cache_v2_windows` **逐通道数值一致**
  （`max|Δ|<1e-3`，全 0 fail）→ z-ER 口径证毕一致，非口头声称。
- **覆盖**：8 队列 70 次发作全覆盖到 offset+60（含 eeg_onset 晚于 clin_onset 的个案，`span=max(...)` 修复证毕）。
- **空走廊 = NaN**：548/583/1084（narrow control）、253（swap）走廊空 → `pms_axial_mid=NaN`，图标 not measurable。
- **too-long 守门**：一条 29741s（≈8h）的 status/标注异常被 `MAX_ICTAL_SEC=600` 自动剔除。

## 5. z-ER 中后期 caveat（重要）

所有 ictal 场是**示意性**的：z-ER 相对发作前安静期归一，越往发作中后期归一越不可靠。图/动画画的是**窗内
rank01**（相对空间形状，比绝对 z 稳）；`positive_mass_share` 是占比（compositional，相对稳）；但**绝对幅度
中后期不要当定量**，轨迹中后期偏示意。结论一律 subject-level exploratory，不写 cohort/机制规律。

## 6. 图（paper-ready 候选 = supplementary）

复用 `field_vs_ictal_swap` 渲染（`_field_panel`），锚到该 subject **发作前** frame；viridis 场 + 触点按值着色
+ **红圈=source in A / 蓝圈=source in B（分布式 swap 集合）** + SOZ；最左 = 间期 **A | B** 两场对比。

- subject 各 4 PNG：间期 A|B 锚 / 平均早期 ictal 场 / progress summary（spaghetti+median）/ offset summary。
- **field 演化 GIF 各 1**（最长发作，onset→offset+30s，step1.5s/fps8）：**直观看发作场的传播变化**——
  这是 paper supplementary 的主推可视化。
- per-seizure 各 3 PNG（dur≥40s）：演化行 [间期A|间期B|ictal 0/25/50/75/100%] / 指标轨迹 / 终止行。

**paper 定位**：本图模式（间期 A|B 锚 + ictal 场演化 + GIF）保留为 **supplementary**，直观展示"发作场相对间期轴
怎么演化"。主文 claim 仍是 A 线"轴共享、极性不重放"；本 pilot 是其**时间维**的探索性补充（轴结构随发作是否
减弱），方向有暗示但未达稳健。

## 7. 文件

- 代码：`scripts/build_topic5_ictal_field_long_cache.py`、`scripts/run_topic5_ictal_field_dynamics.py`、
  `scripts/plot_topic5_ictal_field_dynamics.py`、`scripts/analyze_topic5_field_dynamics_trend.py`、
  `src/topic5_ictal_field_dynamics.py`（+ `tests/test_topic5_ictal_field_dynamics.py`，14 pass）。
- 数据/图：`results/topic5_ictal_recruitment/field_dynamics/`（`per_seizure_metrics.csv`、`per_subject/*.json`、
  `trend_stats.json`、`figures/`（含 README + 8 GIF + 32 subject PNG + per_seizure/））；
  长 cache `results/topic5_ictal_recruitment/ictal_field_long_cache/`（gitignore）。
- spec/plan：`docs/superpowers/specs/2026-06-28-topic5-ictal-field-dynamics-design.md`、
  `docs/superpowers/plans/2026-06-28-topic5-ictal-field-dynamics.md`。

## 8. 扩队列调查（已完成 2026-06-28）

**问题**：走廊/轴的数学只需每模板**最早 2-3 个端点**作核（compact core），不必 swap-positive——非 swap 被试
也能用间期 A/B 端点构轴 + 统计 + 画 GIF。调查：全队列能扩到多少 + 方向暗示扩了之后还在不在。

**可行性扫描**（间期几何，无需 cache）：瓶颈不是 swap，而是"两模板 + 非退化轴 + 中段有电极 + 够发作"。
broad 12 个有两模板+几何里 8 可用（非 swap 只多 E916）；narrow 18 里 7 可用（多 1146/384/442/958）。
**非 swap 确实能用**（E916/E442 swap=none 有真走廊），证实"不必 swap"。已加 ungated loader（`_load_subject`：
swap→swap 集合作圈、非 swap→template-earliest top-K 作圈；轴核都用 compact）+ `--substrate {broad,narrow}`。

**关键结果（见 §3b）**：broad 的方向暗示（轴向降/非轴向升）**在 narrow 扩队列不复现、甚至反向**。
→ **扩队列把 broad 的暗示证否了**：该方向不是稳健现象，依队列/substrate/走廊厚度而变。E916 证明非 swap 走廊
几何可建但发作太短（中位 8s）无趋势贡献；真正长发作的 E1146（21sz）轴向 ρ=+0.52（反向）。

**结论**：作为**探索性 + 阴性偏向**收口——发作场会变、整体贴间期轴（时间上轴共享保持，呼应 A 线），但
"轴向走廊随发作减弱"不成立为稳健现象。图/GIF 模式保留为 supplementary（直观看场演化），统计不进 claim。
若将来要追，需要**长发作 + 厚走廊 + 同 substrate** 的更大队列，且先解决 z-ER 中后期 reliability。
