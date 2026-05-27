# SEF-ITP Phase 3 v2 Trajectory + Per-Event Endpoint Geometry — 2026-05-24 (v2.3)

> **v2.3 状态**: full PR-2 cohort, per-event 几何指标 + 收紧 k 准入 + swap mechanics probes A/B/E. NOT a hypothesis test (no baseline comparison, no BH-FDR). Exploratory descriptive only.

---

## v2.2 → v2.3 ERRATA (2026-05-25, 用户工程合同 catch)

**v2.2 的 k 准入 `n_eligible >= max(6, k + 1)` 在 k=5/6 时会让 source top-k 和 sink last-k 在同一个 6-channel 池里大幅重叠 → "端点紧凑" 测的其实是 "参与场整体紧凑"。**

工程错误形式: 准入仅要求 "至少 6 个 eligible channel", 但 k=5 要从这 6 个里取 top-5 earliest 和 last-5 latest, 必然有 4 个 channel 同时属于两端 — centroid RMS 就退化成参与子集的 spread, 不是端点几何对比。

**v2.3 修复** (`src/sef_itp_phase3_trajectory.py:compute_per_event_endpoint_geometry`):
```
# v2.2 (buggy)
min_part = max(6, k + 1)
# v2.3 (fixed)
min_part = max(6, 2 * k)
```

效果: k=4 要求 n_eligible >= 8, k=5 要求 ≥10, k=6 要求 ≥12 — 保证 source / sink 不重叠。

**Trusted k 范围** (用户口径):
- **k=2/3 = trusted endpoint geometry** (端点 < 半参与场, 几何意义干净)
- **k=4/5/6 = sensitivity sweep only** — 报告时不能跟 k=2/3 一起当 "稳定端点" 读, 仅 "随 k 增大紧凑度怎么变" 描述

**重要 distinction**: v2.3 的 per-event endpoint sweep **仍是描述性 trajectory**, 不替代 formal H5 检验。正式 H5 v1.1 (matched-baseline peri-ictal recruitment test) 仍是绿的 (45 passed), 其 NULL verdict 不被 v2.x 任何变体推翻。

**重跑 cohort 输出**:
- per-subject JSON: `results/topic4_sef_itp/phase3_ictal_adjacent/v2_trajectory/per_subject/*.json` (34 subjects, 2026-05-26 重生成)
- CSV 摘要: `per_event_per_subject_summary.csv` (2026-05-26 重生成, k=4/5/6 大量 NaN = 准入收紧导致)
- timeline figures (25 张) + cohort_rms_vs_k.png 已重画

---

## v2.1 → v2.2 ERRATA (2026-05-24, 用户科学合同 catch)

**v2.1 timeline panel 2/3 用 swap-k decision_k + source_radius/sink_radius 当 per-event SOZ 边界读 — 错的语义滑移。**

科学错误形式: swap-k / decision_k 是 **template-pair 统计量** — 定义对象是 cluster A 整体 vs cluster B 整体 (或 window 内重估的 local template pair) 之间的 rank-displacement / role-swap statistic, **不是单事件的 source/sink 几何**。把它解读成"每次发作前后病人 SOZ 怎么变" 是把模板比较统计量误当 per-event 病理边界读数。

**正确的三层分清** (用户 catch 全文落地):

| 层 | 定义对象 | 能回答 | 不能回答 |
|---|---|---|---|
| **swap-k / decision_k** | 模板对 (cluster A vs B 整体) | "这病人有没有两个互为镜像的稳定传播模板" | 单事件几何 / SOZ 边界 |
| **template-of-window** | 窗内所有事件平均的 template | 这一窗 "平均传播模式" 的端点 | 单事件起始/终止 |
| **per-event top-N early / last-N late** | **每事件单独** rank 里取 top-N 早火 + last-N 晚火 | **每次事件传播起始 seed / 终止端点的空间紧凑度** | "这 N 个 = 完整 SOZ" |

**重要 caveat (升级到 H5 verdict 框架)**: 即使 per-event source RMS 紧凑也**不证明 "SOZ = 这 N 个 channel"**, 只支持 "每次事件传播 seed 在它发射的 channel 上空间紧凑"。一个大 SOZ 也可能每次只从其中一个小 seed 触发; SEEG 稀疏采样也可能让 seed 看起来比实际紧凑。**能说的最强口径是 "传播起始/终止端点采样上稳定, rate 在变",不是 "SOZ 边界精确证明稳定"**。

v2.2 修复:
- 数据层加 `compute_per_event_endpoint_geometry`: per-event, k=2..6, source/sink 各自 centroid RMS, window 内 median + IQR (准入: per-event 参与 channel >= max(6, k+1); per-window 至少 10 events used)
- **timeline figure panel 2 = source (5 k 渐变), panel 3 = sink (5 k 渐变)** (panel 2 swap-k decision_k 删了 — 因为它就是被 catch 降级的 template-pair 统计)
- Cohort 扩到**全 PR-2 cohort (34 subjects, stable_k=2)** — 因为 per-event 指标独立于 swap evidence; v2.1 只跑 swap-positive 是不必要约束

---

> **module**: `src/sef_itp_phase3_trajectory.py` v2.0.0 + per-event geometry (PER_EVENT_K_VALUES = (2,3,4,5,6))
> **runner**: `scripts/run_sef_itp_phase3_trajectory.py --all --full-sweep --full-pr2-cohort --skip-per-seizure`
> **plotter**: `scripts/plot_sef_itp_phase3_v2_timeline.py --all` (3-panel: rate / source-k-gradient / sink-k-gradient)
> **summarizer**: `scripts/summarize_sef_itp_phase3_per_event.py` (per-subject CSV + cohort RMS-vs-k)
> **per-subject 输出**: `results/topic4_sef_itp/phase3_ictal_adjacent/v2_trajectory/per_subject/<dataset>_<sid>.json::full_sweep_trajectory[*].per_event_endpoint_geometry`
> **CSV 摘要**: `results/topic4_sef_itp/phase3_ictal_adjacent/v2_trajectory/per_event_per_subject_summary.csv`
> **figures**: `figures/timeline_<dataset>_<sid>.png` (per subject) + `figures/cohort_rms_vs_k.png`
> **跟 v1.1 关系**: v1.1 是 hypothesis test, 现在视角下 H5 verdict 应说 "**per-event endpoint 不随 seizure state 进展;rate 在变;能否解读为 SOZ 不扩张依赖 SOZ-vs-seed identity 假设**"

---

## 0. 一句话朴素话承诺 (CLAUDE.md §8, v2.2 重写)

**测了什么**: 全 PR-2 cohort 34 个有稳定 2 模板的病人。对每个病人的每次发作间期 HFO 事件, 单独算"这次事件里最早火的 N 个 channel 在脑子里成多紧凑一个团" (source) 和"最晚火的 N 个 channel 成多紧凑一个团" (sink), N = 2/3/4/5/6 都算。然后跨整段录制 (-24h 到 +24h around 每个发作) 看这个 per-event 紧凑度有没有随时间漂动。

**怎么测的**: 每个 subject, sliding 55-min/30-min stride window 全程扫描整段录制 (不锚 seizure, 因为我们想看整段 trajectory 不只是发作邻近)。每个 window 内, 对每个有 ≥6 个 channel 参与的 HFO 事件, 取 rank 里 top-N earliest = source, last-N latest = sink, 算这 N 个 channel 坐标的 centroid RMS (= 距几何中心的均方距离 = "团有多紧"); window 内对所有合格事件取 median (要求至少 10 个事件)。最后 per-subject 看跨整段 windows 的 median 是不是稳, range (max - min) 是多少。

**揭示了什么** (按 strict / candidate / none 分层, 重要 caveat 在 §0.5):

1. **Strict 组 (3 个有 coords 的: epi_1146 / 139 / 958)** — source RMS 跨整段 window range:
   - 1146: k=3 range = **1.0 mm**, k=6 range = **1.0 mm** (几乎完全水平)
   - 139: k=3 range = 3.0 mm, k=6 range = 2.7 mm (稳)
   - 958: k=3 range = 6.5 mm, k=6 range = 7.9 mm (中等稳)
   - → **strict 组传播起始 seed 在整段录制内空间紧凑且稳定**, 不被发作状态打乱

2. **Candidate 组 (5 个有 coords 的)** — 表现分化:
   - epi_384: k=3 range = 4.5 mm (稳)
   - epi_253 / 620 / 635: k=3 range = 18 / 11 / 22 mm (zigzag)
   - epi_1073: NaN (events 不够或 coords 缺)
   - → candidate 内部异质, 不能一句话总结

3. **None 组 (≥10 个有 coords 的)** — 大多 zigzag (range 3-15 mm), 个别极端 (epi_922 source k=3 range = 57 mm — 异常 outlier 需诊断)
   - → 没真实病变 seed 结构, per-event endpoint 注定 unstable, 跟 H5 假设无关

**rate 是唯一明显被发作状态调制的指标** — 大多数 subject 在 seizure 簇时段 HFO rate 从基线飙升 (epi_1146 baseline ~500/h 跳到 ~5000/h = 10×; epi_139 baseline ~50/h 跳到 ~700/h)。但**这个 rate 飙升不伴随 source/sink 几何变化** — rate-vs-geometry decoupling 在 ictal-adjacent timescale 也成立 (跟 H4 v1.1 结论一致)。

## 0.5 重要 caveat (口径定义)

> "Per-event source/sink 空间紧凑度稳定" **不等于** "SOZ 边界稳定"。

理由:
- 一个大 SOZ 也可能**每次只从其中一个小 seed 触发** → per-event 看到的是 seed 集中, 但 SOZ 整体很大
- SEEG **稀疏采样**也可能让 seed 看起来人为紧凑 — 没插针的 channel 不可能出现在 top-N
- top-N 是采样,不是边界。增大 N 看 RMS 增长率能部分缓解 (k=2..6 渐变), 但仍是采样不是完整边界

最强能说的口径: **传播起始/终止端点在 SEEG 采样到的 channel 上空间紧凑且稳定, rate 在 seizure 邻近变化。结论"SOZ 不扩张"需额外 SOZ-vs-seed identity 假设**。

---

## 1. 数据 / 准入标准

- **Cohort**: 全 PR-2 cohort (`results/interictal_propagation_masked/per_subject/*.json` 里 `adaptive_cluster.stable_k == 2`) = 34 subjects
- **可用 subset** (有 seizure inventory + coords + 足够 events): ~20 subjects 完整出 per-event 数据
- **掉队 10 subject**: 没 seizure inventory (大多 yuquan strict-but-no-seizure 那批)
- **NaN 4 subject** (epi_1073, yuquan_gaolan / huanghanwen / litengsheng / sunyuanxin / xuxinyi): coords 缺失或 per-event 准入不够

**Per-event 准入**:
- 该事件参与 channel 数 (bools=True) >= max(6, k+1) — 防止 k=6 时几乎选光所有 channel
- 该 channel 必须有 finite rank AND mapped coords
- per-window 至少 10 events 通过准入才报 median, 否则 None

**Window 准入** (沿用 v1.1):
- effective_seconds / nominal >= 0.75
- events_total >= 30

---

## 2. Per-subject within-subject summary 表

`per_event_per_subject_summary.csv` 完整版; 这里挑 strict + 几个有意思 candidate / none:

| subject | swap_class | n_sz | source_rms<br>k=2 / range | k=3 / range | k=5 / range | k=6 / range | sink_rms<br>k=3 / range |
|---|---|---|---|---|---|---|---|
| **epi_1146** | strict | 26 | 3.8 / 6.4 | 10.7 / **1.0** | 11.4 / **1.0** | 11.6 / **1.0** | (sink stable) |
| **epi_139** | strict | 6 | 3.4 / 3.4 | 8.2 / **3.0** | 11.3 / 3.7 | 11.5 / 2.7 | (sink stable) |
| **epi_958** | strict | 16 | 7.3 / 5.7 | 11.0 / 6.5 | 14.7 / 9.6 | 15.7 / 7.9 | (modest range) |
| epi_384 | candidate | 15 | 15.6 / 6.5 | 16.3 / 4.5 | 17.5 / 2.0 | 18.1 / 2.5 | (stable but large) |
| epi_253 | candidate | 7 | 5.8 / 13.8 | 8.3 / **18.3** | 9.5 / 15.1 | 9.9 / 12.9 | (zigzag) |
| epi_620 | candidate | 7 | 4.1 / **17.6** | 6.9 / 11.4 | 10.4 / 6.6 | 11.4 / 5.5 | (zigzag) |
| epi_635 | candidate | 21 | 10.0 / **23.3** | 11.9 / 22.5 | 18.4 / 15.6 | NaN | (very unstable) |
| epi_1077 | none | 9 | 3.9 / 10.9 | 4.5 / 14.4 | 18.7 / 0.7 | NaN | (mixed) |
| epi_1084 | none | 92 | 7.8 / 5.1 | 10.6 / 3.6 | 12.4 / 2.5 | 13.1 / 2.1 | (relatively stable, none class!) |
| epi_922 | none | 29 | 7.6 / **59.8** | 59.4 / 57.1 | 52.5 / 4.3 | 49.3 / 2.5 | (extreme outlier) |
| epi_590 | none | 13 | 3.1 / 6.6 | 6.8 / 8.3 | 19.3 / 18.6 | 22.2 / 11.6 | (variable) |

**简单 take-aways**:
- **Strict cohort range 全部 < 10 mm** (大多 < 5 mm) — within-subject 几何随时间不动
- **Candidate cohort 内部分化** — epi_384 像 strict 稳定, epi_253/620/635 显著 zigzag (range 11-23 mm)
- **None cohort 多 zigzag** — 跟 "没 swap 结构 → per-event 端点不稳" 一致
- **epi_922** 异常: source k=2 range = 60 mm 离群, 单独诊断 (可能特定窗口 weird events)
- **epi_1084 (none, 92 seizures)** 出乎意料 stable (k=3 range = 3.6 mm) — none class 里特例, 可能 swap evidence 阈值边界附近

---

## 3. Cohort RMS-vs-k 图 (visual overview)

`figures/cohort_rms_vs_k.png`: 每个 subject 一条曲线, X = k=2..6, Y = within-subject median RMS。蓝 = strict, 绿 = candidate, 灰 = none。

**关注点**:
- 大多数 subject monotonic 增长 (k 增大 RMS 增大 — 正常)
- strict 蓝色聚成 tight 一团 (4-16 mm range)
- candidate 绿色更分散
- none 灰色 distribution 最宽, 一条 outlier (epi_922) 在 50-60 mm
- sink 曲线增长更陡 (k=2 ~2 mm, k=6 ~12 mm) — 因为 sink (晚火) 是 sparse network, k 增大显著 spread

---

## 4. 跟 v1.1 + framework H5 verdict 的关系 (v2.2 corrected reframing)

**v1.1 (peri-vs-baseline test, swap-k metric) NULL** — 在新视角下:
- v1.1 NULL 本身仍 valid (该 cohort, 该指标, 真实 no-effect)
- 但 v1.1 的科学解读应限定: "swap-k template-pair 不被 seizure state 调制" — **不直接证明 "SOZ 不扩张"**
- v2.2 per-event endpoint 是补充证据: strict cohort 内部 per-event seed/terminus 也稳定 → 即使切换到 event-level 指标, 仍看不到 ictal-adjacent geometric drift

**Framework H5 verdict** (建议 §3.5 修改):
- 当前 v1.0.7 H5 verdict mapping: SUPPORTED / NULL / FAIL / UNDERPOWERED
- v2.2 视角下应说: "**H5 is NULL with the qualifier that "no per-event seed/terminus geometric drift" was tested; "SOZ extent stable" is a STRICTER claim requiring SOZ-vs-seed identity assumption**"
- Default 建议: framework §3.5 保持 NULL 桶, 但在脚注加 "what was tested: per-event endpoint stability + template-pair stability; what was NOT proven: SOZ extent stability — requires SOZ-vs-seed identity assumption out of scope"

---

## 5. Pending user decisions (Phase 3 v2.2 之后)

1. **Framework §3.5 H5 verdict 措辞**: 是否按上述加 caveat 脚注?
2. **epi_922 outlier 诊断**: source k=2 range = 60 mm 是真实 biology 还是 quirky windows / event detection 残留? 值得单看 timeline + raw events
3. **epi_1084 (none, 92 seizures, k=3 range = 3.6 mm) 特例**: 为什么 none class 反而稳? 是否 swap-evidence 评级偏严 — 这个 subject 应该是 candidate?
4. **Per-subject 回归**: 看不看每个 subject 内部 `source_rms ~ rate` 的相关性? 如果两者正相关 → "rate 涨时 seed 越散" (recruitment 弱信号); 不相关 → 完全 decoupled
5. **跟 PR-T3-1 Layer A ictal early channel cross-cite**: PR-T3-1 报道 ictal recruitment 是否跟 per-event source channel 重叠? 这能直接 test "seizure 招的是同一批 source 还是另一批"

---

## 6. 内部归档代号映射

- **per-event endpoint geometry**: 每事件 rank 取 top-k earliest + last-k latest, 这 k 个 channel 的坐标 centroid RMS
- **window median**: 该窗内所有合格事件的 per-event RMS 的 median (要求 n_used >= 10)
- **within-subject range**: max(window_median) - min(window_median) 跨该 subject 整段 windows
- **full-sweep**: 整段录制 sliding 55-min/30-min stride, 不锚 seizure (v2.1 加, v2.2 默认)
- **--skip-per-seizure**: 跳过 per-seizure-anchored trajectory (v2.2 加, 速度提 ~10x; 不影响 full sweep)
- **K_VALUES = (2,3,4,5,6)**: per-event top-k / last-k 的 k 范围
- **PER_EVENT_MIN_EVENTS_USED = 10**: window 准入: 至少 10 个事件通过准入才报 median

---

## 7. Figures

- `figures/timeline_<dataset>_<sid>.png` (~20 张) — 每个 subject 3 panel: rate + source RMS (5 k 渐变蓝) + sink RMS (5 k 渐变橙); 红线标 seizure onset
- `figures/cohort_rms_vs_k.png` (1 张) — cohort overview, 每个 subject 一条 source / sink RMS vs k 曲线, 按 swap class 着色

**Deprecated (v2.2 删):**
- v2.1 swap-k timeline (panel 2/3 用 swap_k source_radius/sink_radius, panel 1 = decision_k scatter)
- v2.1 per-subject 5×3 grid + cohort overlay (time-bin 太宽 + 用 template-pair 指标)
- v2.0 cohort-pooled time-bin tables

实际目录现状: 仅保留 v2.2 timeline + cohort overview。

---

## 8. Swap mechanics probes A / B / E (2026-05-25 用户深入探索请求)

**用户问题**: "我们的 swap 究竟在测什么? 单一 source 的场效应 side-effect, 还是 ≥2 个真实空间分离的 generator?"

三个独立 probe 各从不同角度回答这个问题, 设计原则: 每个 probe 是一个**独立的物理判别**, 不重复同一个量。

### 8.1 Probe A — cluster A vs B events 的时间交错性

**测了什么**: 把整段 recording 里所有 HFO 事件按时间顺序排, 给每个事件贴上 cluster A 或 B 标签 (取自 PR-2 KMeans 的 dominant cluster), 然后统计 "连续同 cluster 事件" 的 run length 分布。

**怎么测的**: 实测 run length 均值 vs "如果 A/B 完全随机交错的期望均值" (random expected = `(N_A + N_B) / (N_runs_random)`, 这里用 simplified geometric expectation)。比 = `obs / expected`。
- ≈ 1.0 → 随机交错 (两 generator stochastic 共活动)
- 显著 > 1.0 (e.g., 2.0+) → 时间聚成块 (mode switching, 病人在两个 brain state 间切换, 每个 state 各自 dominant 一个 cluster)
- 显著 < 1.0 → 强迫交替 (一个事件后必然换 cluster, 类似 oscillator)

**揭示了什么**: cohort 共 34 个 subject, runs_ratio 分布:
- **strict 组 (n=7 有 ratio)**: 0.90 – 1.12, 中位 ~1.00
- **candidate 组 (n=5)**: 0.91 – 1.11, 中位 ~1.00
- **none 组 (n=22)**: 0.91 – 1.23, 中位 ~1.00 (一个 yuquan_zhangkexuan = 1.23 略高但仍温和)

**几乎全 cohort 的 runs_ratio 都贴在 1.0 附近 (没有 mode switching)**。这强烈支持 "两 generator 在整段 recording 内**持续共活动**, 每次事件随机选一个 generator 触发", **排除** "病人在两个脑状态间切换、每个状态独占一个 cluster" 这种解释。

### 8.2 Probe B — cluster A 与 cluster B 各自 top-k source channel 的解剖距离

**测了什么**: PR-2 给每个 cluster 一个 template (10-channel rank vector)。取 cluster A 的 top-k earliest channels 和 cluster B 的 top-k earliest channels, 算这两组 channel 之间的 **min** 和 **mean** pairwise Euclidean distance (mm, MNI coord)。
- min ≈ 0 (或重合) → 至少一个 channel 共享 (single source 在两个 cluster 都最早火)
- mean 大 (>10 mm) → 两组 channel 整体上空间分离

**怎么测的**: 对每个 subject k=2..6 各算一次。如果 A 和 B 的 top-k Jaccard = 1.0 → 完全相同的 k 个 channel (re-entry 单 circuit); Jaccard < 1 且 mean_dist 大 → 真实空间分离两组 source。

**揭示了什么** (k=3 横切, n=24 有 coord):
- **min_dist = 0 mm** 在 16/24 subject 出现 — 说明 **A 和 B 的 top-3 通常有 1 个 channel 重合** (e.g., 共享一个"主 source")
- **但 mean_dist** 分布广: 4 mm (yuquan_zhangkexuan) – 68 mm (yuquan_songzishuo). strict 组 mean_dist 集中 17–27 mm, candidate 组 8–32 mm, none 组 4–68 mm (含 epi_922 outlier)
- **典型 strict 例 (epi_1146)**: k=3 min = 0, mean = **26.7 mm**, Jaccard = 0.20 — 1 个 channel 共享, 剩下的明显分开
- **典型 strict 例 (epi_958)**: k=3 min = 15.3, mean = **31.4 mm**, Jaccard = 0.0 — 完全分开

**核心解读**: 即使 A 和 B 共享 1 个 top channel, 剩下的 source 在解剖上**明显分离** (mean > 17 mm, 远超 SEEG adjacent contact 3.5 mm)。这**排除** "单一 source 的场反向 side-effect" 解释 — 如果只有一个真 source, A 和 B 的 top-k 应该高度重叠 (Jaccard 接近 1)。

### 8.3 Probe E — cluster A 与 B 的 event count dominance

**测了什么**: `dom_frac = max(N_A, N_B) / (N_A + N_B)`. 
- ≈ 0.5 → balanced (两 generator 真实共活动)
- ≈ 0.9+ → 一个 generator 主导, 另一个稀少 (可能 "main + noise reverse template", 弱信号 cluster 是 KMeans 凑出来的, 不是真 generator)

**揭示了什么**: cohort dom_frac 分布:
- **strict 组**: 0.50 – 0.66, 中位 0.55
- **candidate 组**: 0.54 – 0.66, 中位 0.61
- **none 组**: 0.50 – 0.72, 中位 0.62
- **没有任何 subject 达到 0.85+** (最高 yuquan_zhangkexuan 0.72)

**核心解读**: cohort 一致表现为 "balanced two-generator system", **排除** "main + weak noise reverse" 解释。strict 组尤其接近 50/50 (3/4 subject < 0.6) — 两 cluster 在 event count 上势均力敌, 不是统计噪声。

### 8.4 三个 probe 合起来对 "swap 在测什么" 的物理解读

把三个 probe 当独立约束放在一起, 三个机制候选只剩一个生还者:

| 机制候选 | Probe A (interleaving) | Probe B (source distance) | Probe E (dominance) | 评判 |
|---|---|---|---|---|
| **(M1) 单一 source 的场反向 side-effect** (e.g., dipolar reversal artifact, oscillator) | ≈ 1 (✓) | small min, **small mean** (✗) | balanced (✓) | **排除** (B 反对) |
| **(M2) 时间分块的两脑状态 mode switching** | > 1 (✗) | large mean (✓) | imbalance OK (~) | **排除** (A 反对) |
| **(M3) 两个空间分离的真实 generator 在整段共活动, 每事件随机选一个触发** | ≈ 1 (✓) | small min (有时共享 1 channel) + **large mean** (✓) | balanced (✓) | **survives** |

**Cohort 一致支持 M3**: swap-positive 病人有 **≥2 个空间分离的真实 generator**, 共享 1 个 "桥接 channel" 是常见现象, 但 source 整体上明显分离。两 generator 在整段 recording 持续共活动, 每次 HFO 事件 stochastic 决定哪个 generator 触发。

**对 "swap 的科学意义" 用户问题的直接回答**:
- swap **不是**单 source 的场反向 side-effect (probe B 否决) — 真有两个 source
- swap **不是**两脑状态切换 (probe A 否决) — 没 mode 切换, 是稳态共活动
- swap 的"两个模板"是真实两个生成 circuit (而非 KMeans 强切两半的伪结构) — 因为 candidate / strict / none 在 dom_frac / runs_ratio / mean_dist 三轴上都明显分层 (strict mean_dist 中位 27 mm vs none 中位 18 mm 但 spread 大)

### 8.5 Outputs

- `results/topic4_sef_itp/swap_mechanics/cohort_probe_summary.csv` — 完整数据 (34 subject × 13 列)
- `figures/probe_A_time_interleaving.png` — runs_ratio scatter by swap_class
- `figures/probe_B_anatomical_distance.png` — min / mean dist (k=3) by swap_class
- `figures/probe_E_dominance_ratio.png` — dom_frac scatter by swap_class
- `figures/triple_panel_overview.png` — 三 probe 横向并排 (注: 2065 px 宽, 多图比较时可能超 API 显示宽度限制)

### 8.6 Pending follow-ups (执行进展见 §9)

- ~~**Probe F** (event-level earliest seed switching)~~ → 已执行 (§9.1)
- ~~**Probe G** (rank-distance slope comparability)~~ → 已执行 (§9.2)
- ~~**Probe H** (implant coverage / shaft audit)~~ → 已执行 (§9.3)
- **Probe C / D** (event-level direction vector / PR-T3-1 cross-cite) — 尚未执行, 留给后续 PR
- **epi_922 outlier engineering 诊断** → §9.3 部分确认 (max pairwise dist = 142 mm = brain-spanning, 坐标 unit/labeling 问题), 仍需手工核查 raw coord file

---

## 9. Axis-sanity probes F / G / H (2026-05-26 用户深入要求)

**用户原话**: "event-level earliest seed 是否在两端真实切换; 同轴双源 vs 单源回传的 per-event latency/rank-distance slope; implant coverage audit"

§8 的 probe A/B/E 否决了 (1) 大脑两状态切换 (mode switching) 和 (2) 主+弱噪声两 cluster 这两种解释, 但**没有排除 "单 SEEG axis 上的双向 wave"** — 这是 M1 的子变种。下面三个 probe 直接攻击这个 confound。

### 9.1 Probe F — per-event modal seed 双对称性

**测了什么**: 对每个 valid event, 取该事件 mask 后 rank 最小的 channel = earliest, rank 最大的 channel = latest。按 cluster A / B 分组算 modal earliest / latest channel, 然后看几何对偶: dist(A_earliest, B_latest) 和 dist(B_earliest, A_latest)。

**怎么测的**: 取这两个距离的最大值 `dual_symmetry_mm`。
- **小** (< 10 mm) → A 的 earliest channel 跟 B 的 latest channel 大致重合, 同时反向也对 → 整个 swap 是同一根轴上的双向 wave (M1 单 axis bidirectional)
- **大** (> 20 mm) → A 和 B 的端点几何不对偶 → 两套独立 source 各自有自己的端点 (M3)

**揭示了什么** (strict 组 5 个有 coord):
- **dual_sym 小 (M1-like) 的 strict**: yuquan_zhangjiaqi = **3.5 mm** (A_e=H7, A_l=H3, B_e=H4, **B_l=H7**, H7 同时担任 A 起点和 B 终点 = 经典单 axis 双向), epi_139 = **6.8 mm** (HL6/HL4/HL4/HL7 = 全在 HL 这一根 shaft 上), yuquan_zhaochenxi = 7.1 mm
- **dual_sym 大 (M3) 的 strict**: epi_1146 = **23.4 mm** (SCL9/ICL2/ICL1/ICL9 = 跨 SCL 和 ICL 两根 shaft), epi_958 = **28.7 mm** (GF6/GH7/GH7/GD8 = 跨 GD/GF/GH 多根 shaft)
- **candidate 组**: 0–47 mm 高度分散
- **none 组**: 22–131 mm (epi_922 outlier 131 mm)

**核心新发现**: **strict cohort 内部并非同质 M3。至少 2/5 (epi_139, yuquan_zhangjiaqi) 表现为 single bidirectional axis (M1), 不是真正的两 source**。这翻转了 §8 "swap 全 cohort 一致支持 M3" 的过强解读。

### 9.2 Probe G — per-event rank-distance slope 可比性

**测了什么**: 对每个 valid event, 取 earliest channel 当 reference, 算每个参与 channel 的 (rank, distance to reference), 做 OLS slope = mm per rank step, 同时算 Pearson r(rank, distance)。按 cluster 分组算 per-event 中位数。

**怎么测的**: 比较 |median_slope_A| / |median_slope_B|。
- ≈ 1 (0.5–2 之内) → A 和 B 是同一根 axis 上的反向 wave (axis length 相同)
- 远离 1 → 两 cluster 的传播 axis 长度本就不同 → 真两源

**揭示了什么**:
- **strict 组 5 个全部 ratio 在 0.51–1.71 范围** (同一 order of magnitude, 跨 1.0 两侧)
- candidate / none 组分散 (0.008 – 38.7)
- 但 r(rank, dist) 散点显示 strict 组的 Pearson r 多在 0.25–0.55, **不在 ≈ 1**, 说明每事件传播 ≠ 干净 1D wave (channel 几何并非完全沿 rank 顺序排列)。slope 本身较弱, 这个 probe 对噪声敏感, 主要作为辅助证据

**核心解读**: strict 组 slope ratio ≈ 1 的现象**跟 probe F 一致支持 M1 解读**: 同 axis 上的双向 wave 自然有相同的 axis 长度; 真两源不会强制 axis 长度相等。但因为 G 单独太弱, 主结论仍以 F + H 为准。

### 9.3 Probe H — implant coverage / shaft audit (engineering 关卡)

**测了什么**: 把每个 channel name 的 alpha 前缀当 SEEG shaft id (e.g. `HL6` → shaft `HL`, `GD8` → shaft `GD`)。每个 subject 算: shaft 总数, max inter-shaft distance, cluster A top-3 source 来自哪些 shaft, cluster B top-3 来自哪些 shaft, shaft Jaccard, max pairwise coord distance。

**怎么测的**: 
- `h_AB_same_single_shaft = True` 表示 A 和 B 的 top-3 source 完全都在 **一根 shaft 上** → single-axis warning, implant coverage 不足以判别 M3 vs M1
- `h_n_shafts_total = 1` 表示该 subject 整个 implant 只有 1 根 SEEG 电极链 → 完全无法测真双 source
- `h_max_pairwise_dist_mm` 异常大 (> 100 mm) 跟脑长接近, 警示 coord 单位/标签错误

**揭示了什么** — 这是 cohort 解读的硬约束:

| Subject | swap_class | shafts | A top-3 shaft | B top-3 shaft | dual_sym | 判别 |
|---|---|---|---|---|---|---|
| **epi_139** | strict | **1 (HL 唯一)** | HL HL HL | HL HL HL | 6.8 mm | **M1 (coverage 不允许测 M3)** |
| **yuquan_zhangjiaqi** | strict | **1 (H 唯一)** | H H H | H H H | 3.5 mm | **M1 (coverage 不允许测 M3)** |
| **yuquan_zhaochenxi** | strict | 5 | E E E | C C C | 7.1 mm | M3 弱 (跨 shaft 但 dual_sym 仍小) |
| **epi_1146** | strict | 2 | SCL ICL ICL | SCL ICL ICL | 23.4 mm | **M3 (真两源)** |
| **epi_958** | strict | 6 | GD GF GE | OPL GH GG | 28.7 mm | **M3 (真两源)** |
| epi_620 | candidate | 2 | HR HR HR | HR HR HR | 7.1 mm | M1 (coverage 限制) |
| epi_1150 | none | 4 | RFB RFB RFB | RFB RFB RFB | 48.5 mm | M1 (即使有多 shaft, A/B 都集中 RFB) |
| chengshuai | none | 2 | K K K | K K K | 26 mm | M1 (coverage 限制) |
| **epi_922** | none | 3 | GC GC GB | GC GC GC | 131 mm | **engineering ✗** (max pairwise 142 mm 接近脑长, coord 单位/半球标签问题) |

**单 shaft 病人有 5 个** (epi_139, zhangjiaqi 是 strict; epi_620 cand; epi_1150 + chengshuai none) — 这些 subject 的 swap_class 结论**不能升级为 M3**, 只能说 "实测到双向 wave, 但 implant coverage 不允许排除单 SEEG axis 双向解释"。

**epi_922 engineering 诊断结果**: max pairwise 距离 142 mm, max inter-shaft 也 142 mm — 这跟成人脑长接近, 几乎不可能是单 subject 的合理 SEEG implant extent。**几乎确定是 coord 单位错误 (cm/mm 混) 或半球混入 (GC 和 GB shaft 实际不在同一脑半球但被同一坐标系误标)**。原始 coord 文件需手工核查, 不能当 biology 解读。

### 9.4 三 probe 合起来对 §8 结论的修正

§8 原结论: "swap-positive 病人都是 ≥2 空间分离 generator (M3)" — **过强, 被 §9 部分推翻**。

修正后的分层解读:

| 病人组 | §8 probe ABE | §9 probe FGH | 最终解读 |
|---|---|---|---|
| **multi-shaft + dual_sym 大** (epi_1146, 958) | M3 ✓ | M3 ✓ | **可靠 M3 真两源** |
| **multi-shaft + dual_sym 小** (zhaochenxi) | M3 (mean dist 大) | M1/M3 边界 | **M3 弱信号**, 需更多 channel 验证 |
| **single-shaft strict** (epi_139, zhangjiaqi) | "支持 M3" 但其实是同 shaft contact 间距 | **M1 ✓** | **M1 (单 axis 双向), §8 的 M3 解读被否决** |
| **single-shaft non-strict** (epi_620, 1150, chengshuai) | mixed | M1 ✓ | **M1 (单 axis 双向)** |
| **epi_922 outlier** | mean dist 12 mm (k=3) | 142 mm 离群 | **engineering 错误**, 不可解读 |

**对用户原问题"swap 究竟测的是什么"的最终回答**:
- 当 implant 有**多根分散 SEEG shaft** 且 cluster A/B 各自的 top source 分散在不同 shaft 时 (epi_1146, epi_958), swap **确实是 ≥2 空间分离 generator** 的证据 (M3 站立)
- 当 implant **只覆盖 1 根 shaft** 时 (epi_139, zhangjiaqi), swap 是**同一根 SEEG axis 上的双向 wave** 的证据 (M1) — 不能升级为 M3, coverage 不允许测
- 这是**纯 engineering 约束**, 不是 swap_class 标签错误 — strict 标签本身有效 (它准确捕获了"两个互逆模板"这件事), 但**"两个模板 = 两个解剖独立的 source" 这个解读需要 multi-shaft 支撑才成立**

### 9.5 对 §3.5 framework H5 verdict 措辞的进一步建议

§8.4 已建议 "v2 视角下 H5 verdict 保持 NULL 但加 caveat"; §9 进一步建议:
- 任何"strict cohort 一致支持 M3" 的统计应**按 multi-shaft / single-shaft 子集分别报告**
- Topic 4 framework §3.2 "spatial layer = primary cohort claim" 是 n=23 binomial test — 该统计**没有把 single-shaft coverage limitation 当协变量**, 这是 framework 层面遗留的 caveat (但 framework v1.0.7 banner 已 lock, 不要在这里改; 留作 v1.0.8 future)

### 9.6 Outputs (§9)

- `results/topic4_sef_itp/swap_mechanics/cohort_axis_sanity_summary.csv` — 34 subject × 22 列
- `figures/probe_F_earliest_swap.png` — dual_sym scatter
- `figures/probe_G_rank_distance_slope.png` — wave-like r 散点 + slope ratio
- `figures/probe_H_shaft_audit.png` — 3-panel: shaft Jaccard, max inter-shaft dist, mapped channel count
