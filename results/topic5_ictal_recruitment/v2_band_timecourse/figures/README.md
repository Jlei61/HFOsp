# Topic 5 · 多频带 peri-onset field-similarity 时程图（v2 band-scan cache 扩展）

判读 tier = **exploratory candidate scaffold（探索性候选骨架；非 formal gate、非机制）**。这是已验收 Fig3-Sup1（多频带发作早期能量场 ↔ 间期 HFO 几何 alignment，单窗 onset+0–20s）的**时间分辨扩展**：把单窗换成 `[-120,+20]s`、10s 窗 / 2s 步的滑窗，看**每个频带**的相似性轨迹。数据源 = 已提交的 `v2_band_scan/cache`（masked band-power baseline-robust-z），度量 = 与 Fig3-B 完全相同的 formal normalized plane 上 mirror-invariant signed-corr（逐字复用，未重写）。

回答三个描述性问题（**只描述，不下 formal 判决**）：
1. **发作前就在吗** — 相似性在 onset 之前是否已经偏高；
2. **onset 附近上抬吗** — 越过 0s 是否有小幅抬升；
3. **band-generic 还是少数被试 band-leaning** — 抬升是全频段一致还是个别频带 / 个别被试。

**当前观察**（描述性，n=20 cached）：pre-onset 段全频段 cohort 中位 `maxAB|r|` 已经 ~0.70–0.76（**band-generic、preictal-present**）；跨 0s 的 early−pre 增量对绝大多数频带 ≈ ±0.02，**只有 δ 1–4（+低频 1–13）有一个小幅正抬升**（narrow δ +0.071、13/20 被试为正；broad δ +0.080、10/17），无频带出现陡峭 onset 跳变、且正向被试只 ~7–13/20 → **偏 band-generic preictal-present + δ/低频小上抬，而非 onset 处频带分化**；与已验收 Phase-1-v2 "when" 口径一致。

**band 命名**：δ 1–4 · θ 4–8 · α 8–13 · β 13–30 · γ 30–80 · **R 80–150（ripple）** · **FR 150–250（fast ripple）**；composites 1–13 / 13–80 / 80–220 / 80–250。maxAB `|r| = max(|r_A|,|r_B|)` 是 **sign-free scaffold readout**；signed A/B 只解释 polarity。

**禁止措辞**：formal Gate A/B/C passed · HFO-/LVFA-/ripple-/FR-specific · timing-order / propagation replay · 振荡 / criticality / 机制证明 · 过任何空间随机场。承重口径仍是 **间期传播 ↔ 发作早期宽带能量相关 = cohort 承重**；**频带偏向 = 个别患者现象**。正式 cohort shift 仍是 Fig3-A Data-vs-Null。

生成：`python scripts/plot_topic5_multiband_field_similarity_timecourse.py --subjects all --bands all --axis-set both`（fail-closed 逐 subject/逐 band，OMP_NUM_THREADS=1）。索引 `../multiband_timecourse_subject_index.{csv,json}`。每图各答一个独立问题（CLAUDE.md §7）。

### {subject}_..._band_time_heatmap.png
每被试一张 band×time 热图（左：maxAB `|r|` sign-free scaffold，viridis 0–1；中/右：signed r · 模板 A / B sidecar，RdBu 蓝白红 −1..1）。y=频带（primary 7 + composite 4，低→高频），x=窗心时间，黑虚线=0s（clinical onset）。颜色=跨发作中位数。左图看"哪些 band×time 格子相似性高"；signed A/B 看"该模板极性在发作间是否稳定"（A/B 近镜像=forward/reverse 模板，非信号）。
**关注点**：先看左图 pre-onset 是不是全频段普遍偏高（band-generic 底高）、onset 附近有没有变亮；signed A/B 近反相是正常（模板互为反向）；颜色是描述幅度非显著性。

### {subject}_..._primary_band_lines.png
每被试一张 primary-7 频带轨迹（按 low / LVFA / ripple 三组分面板，避免 7 条 IQR 叠一起）：粗线=跨发作 median maxAB `|r|`，阴影=IQR，黑虚线=0s，灰虚线=**1–150 Hz Fig3-B 参考**（raw-block 路径，hop / 发作纳入口径不同，仅作对照、**不属 v2 频带家族**）。图例出框不遮线。
**关注点**：看每个频带 median 在 pre 段的水平 + onset 处是否小抬升（who/when 的可视化）；跨频带比较是否 band-generic；参考曲线只作 broadband 对照，legacy `legacy_bb_1_45`（1–45Hz）不在这里、也不是 1–150 参考。

### cohort_band_time_heatmap_{broad,narrow}.png
队列 band×time 热图（对每被试 aggregate 的 median maxAB `|r|` 再取**被试中位数**），viridis 0–1，0s 黑虚线。**broad(17) / narrow(20) 各一张独立文件，永不 pool**（config `never_pool_axis_sets`；成员=`v2_band_scan/{broad,narrow}/**/*.marker.json`）。配套 `../cohort_pre_vs_early_delta_{broad,narrow}.csv`：每频带 pre(−120,0) vs early(0,20) 的**描述性** delta（被试级差再取队列中位 + Δ>0 的被试数）。
**关注点**：这是**描述性汇总非 formal cohort 统计**；图看似整体均匀偏绿正是"band-generic + 高而平"的诚实呈现（小的 onset 增量在 delta 表 + per-subject 线图里，不靠拉伸色标制造结构）；delta 表只描述方向/幅度，不报 p 值、不做检验。
