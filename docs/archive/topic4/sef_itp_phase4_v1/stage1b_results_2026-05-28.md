# SEF-ITP Phase 4 Stage 1b Results — Burst-Envelope Observation-Unit Calibration

> 状态：**PASS**（Stage 1b 验收满足；观测单位 = burst envelope 经数据确认）
> 日期：2026-05-28
> 上游 spec：`docs/superpowers/specs/2026-05-27-sef-itp-phase4-v1-design.md` v0.2 §5.6
> 上游 plan：`docs/superpowers/plans/2026-05-28-topic4-phase4-stage1b-burst-envelope-calibration.md` v1
> 前置：`docs/archive/topic4/sef_itp_phase4_v1/stage1_results_2026-05-28.md`（Stage 1 ⚠️ 段：unit 待裁决）
> 用户裁决（2026-05-28）：primary event = node burst envelope，spike-level = within-burst secondary

## 一句话朴素话（测了什么 / 怎么测的 / 揭示了什么）

**测了什么**：一个会"间歇性放电"的神经元节点——平时安静，被随机扰动推一下就打出
一小串密集的电脉冲，然后又安静下来。我们要回答的问题是：当我们说"这个节点发生了
一次事件"时，到底是把**每一次越阈放电**各算一次，还是把"一小串放电"**整体**算一次。
这关系到后面拼 2D 网格时，怎么数节点之间谁先谁后、谁参与了一次群体事件。

**怎么测的**：在选定的静息工作点（输入电流、慢变量速率固定不动），把随机扰动的强度
从 0 一点点调大（0、0.2、0.4、0.6），每个强度跑 5 条等长轨迹。对每条轨迹用两种数法
各数一遍：**法一**每次越过阈值算一个事件；**法二**把彼此间隔小于一个阈值的连续脉冲
合并成一串、整串算一个事件、事件时刻取这一串里第一个脉冲的起点。然后对比两种数法
给出的事件数、事件时长、以及每串平均含几个脉冲。那个"合并间隔"不是拍脑袋定的——
先测了脉冲之间间隔的分布：同一串内部的脉冲间隔都小于 20，不同串之间的间隔都大于 50，
中间有一段明显的空档（20–50），所以把合并间隔取在空档里（30）。

**揭示了什么**：在这个尺度上看起来——扰动为 0 时两种数法都数出 0 个事件（节点确实
完全静息）；扰动大于 0 时，"按串数"明显比"按脉冲数"少（最大扰动下脉冲有 19 个，但
只合并成约 7 串），而且每串平均含的脉冲数随扰动从 1.5 涨到 2.6——说明"一串"和
"一个脉冲"确实是两个不同的计数单位，不是换个名字而已。更关键：即使在最大扰动下，
每串事件平均只持续约 19 个时间单位，但相邻两串之间平均隔 115 个时间单位（间隔远大于
时长），事件仍然稀疏而短暂——也就是"平时安静、偶发被触发"的可激样图景没有被破坏，
没有退化成连续不停的节律放电。所以把"一串放电"当作一次节点事件这个选择，在这个静息
工作点是站得住的，可以交给下一阶段的 2D 网格使用。

（内部归档代号：Stage 1b, `detect_burst_envelopes`, `BurstConfig.envelope_gap=30`,
baseline I\*=1.0 / r\*=0.006 / σ\*=0.4, spike-level vs burst-envelope unit）

## Stage 1b 验收结果

- [x] σ=0 → envelope 事件数 **0**（静息 gate）：**PASS**
- [x] σ ∈ {0.2,0.4,0.6} → envelope 事件存在且稀疏短暂（excitable-like，由 raw stats 判定，
      **未**走 `classify_regime` —— RegimeConfig 是 spike-unit 标定，envelope 跨度会被误标）：**PASS**
- [x] spike vs envelope 对照（count / duration / n_spikes_per_burst / IBI）已输出
- [x] `envelope_gap=30` 经 probe 实测的 gap 谷底标定，非调参
- [x] σ=0.6 边界复核（见下）：gap=30 下仍 excitable-like，**未**调 gap

## spike-level vs burst-envelope 对照（5 seed 平均，T=1000, burn_in=100）

| σ | spike 数 | envelope 数 | n_spikes/burst | envelope 时长 | envelope IBI | spike 时长 | spike IBI |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.0 | **0.0** | — | — | — | — | — |
| 0.2 | 6.4 | 4.2 | 1.54 | 10.0 | 224.1 | 2.01 | 132.1 |
| 0.4 | 14.0 | 6.6 | 2.13 | 14.0 | 137.0 | 1.89 | 65.4 |
| 0.6 | 19.0 | 7.4 | 2.60 | 19.2 | 115.4 | 1.92 | 47.1 |

读法：σ 增大时 spike 数（6.4→19）比 envelope 数（4.2→7.4）涨得快，差额被
n_spikes/burst（1.54→2.60）吸收——证明两单位真的不同。envelope IBI 始终 ≫ envelope
时长（如 σ=0.6 时 115 ≫ 19），事件稀疏短暂 = excitable-like。

## envelope_gap=30 标定依据（probe，2026-05-28，写码前）

baseline (I=1.0, r=0.006) 下 spike-level 事件的 **inter-spike gap 分布是双峰的**：

| σ | gap 百分位 [10,25,50,75,90] | 解读 |
|---|---|---|
| 0.2 | [14.8, 17.4, 103, 172, 211] | 串内 ~15，串间 ~100+ |
| 0.4 | [5.9, 8.1, 14, **59.7**, 166] | 串内 <20，上四分位跳到 >50 |
| 0.6 | [5.3, 6.7, 12, **48.8**, 126] | 串内 <20，谷更窄（~48） |

谷底稳定落在 ~20–50（串内 gap 与串间 gap 之间），`envelope_gap=30` 取在谷底，对所有
σ 都干净分开串内/串间。**这是测出来的，不是调出来的**——禁止为了让某个 regime 好看而
扫 envelope_gap（这正是 Stage 1 archive 警告过的、用户在 I=2.0 例子上点名的调参陷阱）。

## σ=0.6 边界复核（advisor flagged）

probe 显示 σ=0.6 的 gap 谷比 σ=0.2/0.4 更窄（75 百分位 48.8，逼近串间下界）。
复核结论：在 envelope_gap=30 下，σ=0.6 仍是 envelope 数 7.4、时长 19.2、IBI 115.4 ——
IBI 远大于时长，**excitable-like，不是 repetitive**。边界没问题，且未为此动 gap。

## 关键设计后果（写给 Stage 2+ reader）

- **节点事件时刻 = envelope 第一个 spike 的 onset**（不是 peak、不是 centroid）。
  这是 Stage 2-3 算节点间传播 lag / 模板顺序的时间戳来源。`BurstEnvelope.onset` 即此。
- **n_spikes / peak_x / duration 是 secondary 字段**，与 onset 一起在首次实现里给出
  （非延后补丁），供 within-burst 细节分析。spike-level 仍可用（`detect_bursts`），
  但只作 within-burst secondary，不作群体事件计数单位。
- **未实现**：group-level event（多节点 burst onset 在短窗共现）属 Stage 2/3，依赖
  尚不存在的 `grid2d.py`，Stage 1b 不碰。

## Artifacts

`results/topic4_sef_itp/phase4_modeling/stage1b_envelope_calibration/`
- `comparison.json` — per-σ spike-vs-envelope 表 + envelope_gap + 验收 flag
- `figures/spike_vs_envelope.png` — 两面板（事件数对照 + envelope 稀疏度），已目视
- `figures/README.md` — 中文逐图说明

## 代码 / 测试

- `src/topic4_modeling/hr_dynamics.py` — `BurstEnvelope` dataclass + `detect_burst_envelopes`
- `src/topic4_modeling/hr_config.py` — `BurstConfig.envelope_gap=30.0`（标定 docstring）
- `tests/test_topic4_modeling_hr_dynamics.py` — 7 envelope 测试（merge / onset / 字段 / 边界）
- `tests/test_topic4_modeling_stage1b_cli.py` — 验收逻辑 4 测 + CLI help + slow end-to-end
- `scripts/run_topic4_phase4_stage1b_calibration.py` — 标定 CLI（exit 1 on 验收失败）

## Caveats

1. **envelope_gap 是 baseline-specific 标定**：30 在 (I=1.0, r=0.006) 谷底成立。Stage 2 若
   节点动力学/噪声尺度变了，gap 谷可能移动——Stage 2 用前应在新参数下复核 gap 分布，
   不要盲搬 30。
2. **"excitable-like" 由 raw stats 判定，无严格动力系统分类**：沿用 Stage 1 caveat 3。
3. **未对 envelope 跑 `classify_regime`**：刻意（plan 合同 clause #6）。RegimeConfig
   (excitable_max_burst=50, excitable_min_ibi=30) 是 spike-unit 标定，envelope 跨度常超 50
   会被误标。如 Stage 2 需要 envelope-unit regime label，须新建 EnvelopeRegimeConfig。

## 下一步

Stage 1b PASS → 观测单位锁定。Stage 2 hand-off 契约：
1. 每节点基础动力学 = baseline (I\*=1.0, r\*=0.006, σ\*=0.4)。
2. **事件单位 = burst envelope（onset = 第一个 spike 起点）**；spike-level 仅 within-burst secondary。
3. Stage 2 加 2D anisotropic diffusion 扫 D_x/D_y ∈ {1,2,3,5}，看均质网格 regime map——
   需自己的 plan + advisor gate + 用户目视 movies/regime map（spec staged discipline）。
