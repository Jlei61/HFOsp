# M4-MZ discovery + P3 最终报告（2026-07-18）

> tier = **mechanism screen（探索性）**。所有 phenotype 是**检出标签**，不是发作主张。
> 分支 `codex/topic4-mz-slowvars` @ `.worktrees/topic4-mz-slowvars`（从 main `2d01634`）。**不 push、不动 main、不删旧 M3/M4 结果、不改论文 Methods。**
> 设计文件：`docs/superpowers/specs/2026-07-18-topic4-mz-per-neuron-slowvars-design.md`；阶段 handoff：`mz_slowvars_stage_handoff_2026-07-18.md`。

## 摘要（朴素话）

同行提了一个"最小两慢变量"假说：只给兴奋神经元加两个慢开关——`z`（被持续强抑制轰炸→有效抑制变弱→事件沿病人轴摊大，**推**）和 `m`（放电越多→自身刹车越强→限制高招募、可能恢复，**拉**），想解释"间期短事件如何被推大变长、又能恢复"。我们在同一 E1146 衬底上跑四个对照臂（其它慢变量全关），先便宜地扫一遍（seed=1），再对关键格跨三个 seed 复核。

**本轮门槛（对标 M2）= 间期基线 + runoff（失控），不要求"有界可恢复临界态"。结论：达标，且跨 3 seed 稳健。**
- 安静间期基线在三个 seed 都有（38/40/39 个可自恢复事件）；
- 去抑制（z）能把网络推进持续失控，且失控起始随去抑制强度可调（强 ~4.8s、弱 ~9.5s），三个 seed 一致。

**超出门槛那部分**：z 是推、m 是纯刹车、z+m 在测到的"强推×强刹"角落被 m 一路刹回基线；唯一一格"有界扩大"只在 seed=1 出现（三 seed 里 1 个）——脆弱，不作 bounded 主张。**关键局限**：预注册选格塌进一个角落，z+m 的中间地带没测到，所以**不能下 clean no-go**。

## 1. 测什么 / 怎么测（详见设计文件 §0–§8）

只对 E 神经元加 `z_i`（抑制效能，`τ_z ż=H(I_th_EI−I_EI)−z`，clip[0,1]）+ `m_i`（放电适应，每 E spike +1、`τ_adp` 衰减，减性电流 `−η_m m`）。四臂：slow-off / z-only / m-only / z+m，同 E1146 衬底（narrow/template_source/L20/dens100/AR2/双低阈值核/真实电极注册），自发无 kick，同 seed 同噪声。判读用 slow-off 同 seed 事件分布做基线，7 类标签（interictal_like / expanded_bounded / expanded_returned / fragment / suppress / runaway / insufficient）。标定只用 slow-off baseline（anti-circularity），预注册写入 `calibration.json`。

## 2. 结果数据

### 2.1 discovery（seed=1，T=12s，27 格）

phenotype counts：`{runaway:5, expanded_bounded:1, interictal_like:15, suppress:6}`（0 `expanded_returned`）。

| 臂 | 分解 |
|---|---|
| **A（z-only，9 格）** | 5 runaway + 1 expanded_bounded + 3 interictal_like |
| **B（m-only，9 格）** | 6 suppress + 3 interictal_like |
| **C（z+m，9 格）** | 9 interictal_like |

arm A 失控起始随去抑制强度/速度延后：q50_tz2500 1740ms、q50_tz5000 3042ms、q50_tz10000 4937ms、q75_tz2500 5144ms、q75_tz5000 9294ms；唯一有界扩大 = `zA_q75_tz10000`（48 events、峰 61Hz、无失控）。arm B suppress 把峰率压到 ~1Hz、participation≈0。

### 2.2 P3 多 seed（seeds 1/3/4，T=15s，并行）

baseline returning events：38 / 40 / 39。

| 格 | seed 1 | seed 3 | seed 4 | 稳健性 |
|---|---|---|---|---|
| `zA_q75_tz10000`（bounded 候选）| expanded_bounded | interictal_like | interictal_like | **1/3 seed-脆弱** |
| `zA_q75_tz5000`（runaway）| runaway 9294ms | runaway 9499ms | runaway 9758ms | **3/3 稳健** |
| `zA_q50_tz10000`（runaway）| runaway 4937ms | runaway 4707ms | runaway 4862ms | **3/3 稳健** |
| `zA_q90_tz10000`（interictal）| interictal | interictal | interictal | **3/3 稳健** |

## 3. 逐条回答设计 §14 的 8 问

1. **slow-off 间期基线是否保留？** 是。seed 1/3/4 各 38/40/39 个可自恢复间期事件（远超 gate=3），非均质 sheet 的全或无 R0。
2. **z-only / m-only / z+m 各产生什么 phenotype？** z-only=多失控（5/9）+ 1 有界扩大 + 3 间期；m-only=纯刹车（6/9 suppress，从不扩大）；z+m=在测到的强 z×强 m 角落全 interictal_like（m 过刹回基线）。
3. **是否存在非 runaway 的 expanded recruitment？** 存在但脆弱：仅 `zA_q75_tz10000` 在 seed=1 有界扩大（z 单独、最弱最慢去抑制），P3 三 seed 里只 1/3。
4. **是否存在恢复（returned）？** 否。全程 0 `expanded_returned`。
5. **结果是否跨 seed？** 间期基线 3/3 稳健、runoff（两失控格）3/3 稳健且起始一致；**bounded 格 1/3 = 不稳健**。
6. **当前能否进入 field readout？** 不作为承重结论：无跨 seed 稳健的 bounded candidate。只为 seed=1 那格保存了 `readout_ready/readout_zA_q75_tz10000_seed1.npz` 作 artifact，**不作 field concordance 主张**（本轮明确不做）。
7. **当前能否称为 seizure-like event？** 否。mechanism screen、检出标签；runaway ≠ seizure，且无有界可恢复事件。
8. **最大科学缺口是什么？** **arm-C 预注册选格塌角**：规则本想按实测 z 耗竭取弱/中/强三档，但实测去抑制没覆盖"弱"档（多数深度耗竭），三档全落强 z（q50）、m 侧全落 ta500_f20，arm C 实际只测了"强 z×强 m"一个角落。⇒ z+m 的中间地带（弱 z 自限 + 分级 m）**未测**，无法判定 z+m 假说成立与否。

## 4. 图（`results/topic4_sef_hfo/mz_slowvars/figures/`）

- `mz_phenotype_map.png` —— 参数网格上的 7 类表型分布（arm A 的 I_th×τ_z、arm B 的 τ_adp×η_m、arm C 的 z×m），一眼看"去抑制→失控、适应→压制、z+m→回间期"。**关注点**：arm A 从弱去抑制的间期/有界过渡到强去抑制的失控；arm B 全在压制侧；arm C 全间期。
- `mz_mechanism_traces.png` —— slow-off + 每臂代表格的 1D 时程（群体率、active fraction、`z`、`m`、适应电流）。**关注点**：z-only 代表格里 `z` 单调耗竭、率随之爬升到失控；m-only 里 `m`/适应电流累积、率被压下去；z+m 里两者相消、率回到间期带。
- `mz_spatial_recruitment.png` —— 代表格的空间招募快照/演化（E-active 分数 24×24）。**关注点**：失控是不是沿病人轴的相干招募 vs 同步全场（描述性，本分支不要求破轴）。

**眼检确认（2026-07-19；出图前修了 plotter `_tags` 的 off-by-one bug——`k[:-7]` 应为 `k[:-6]`，否则 arm 代表列被丢、只画 slow-off 一列）**：
- mechanism_traces 四列机制分解清晰——z-only 代表格 `z` 缓降到 ~0.87、事件变密变高但仍 <120Hz（有界，是 seed=1 那格；arm A 多数其实失控）；m-only 一次事件触发适应电流尖峰(~6mV)把后续率压到 ~0（刹车）；z+m `z` 轻降 + 适应尖峰 → 稀疏间期（m 抵消 z）。
- spatial_recruitment 因"时间取峰"饱和，四格都是宽场高招募、**不区分表型**——只说明招募在所有 regime 都是全场而非紧贴轴的行波（描述性，与均质衬底预期一致）。
- phenotype_map 的 arm C "z weak/mid/strong" 是**目标档位标签**、非实测（塌角，见 §3 Q8）。详见 `figures/README.md`。

## 5. 三选一建议

- ❌ **进 40s acceptance + formal field readout** —— 不建议：无跨 seed 稳健的 bounded/returned candidate，进 acceptance 没有承重对象。
- ✅ **只保留为 early recruited-state mechanism screen（本轮采纳）** —— M2 层级门槛（间期 + runoff）达标且 seed-robust；z 推 / m 拉的机制分解清晰。这是本轮可写的口径。
- ❌ **clean no-go（停 M4-MZ）** —— 不建议：z+m 假说因 arm-C 塌角**尚未在正确 regime 测过**，判 no-go 会是假阴。

**下一轮关键补测（新一轮、明确 post-hoc、不在本轮预注册内）**：修正版 z+m 扫描——以自限的弱 z（`zA_q75_tz10000` 档）为推、叠**分级 m**，看 m 能否把"z 单独的有界扩大"稳成可再触发的招募态；并对强 z（失控档）叠**弱→强 m**，看 m 能否把失控**刹成有界**而非一刀刹回基线。这才是回答同行 z+m 假说的正位测试。

## 6. Claim boundary

**可报告**：E1146 上 M4-MZ 复现"间期基线 + runoff"两 regime（对标 M2 达标，跨 3 seed）；z=push（去抑制→失控/弱档有界扩大）、m=pull（纯刹车）；z+m 在强×强角落 m 过刹回基线；bounded/returned 未稳健出现；z+m 中间地带未测。

**禁止报告**：❌ 称任何格为 seizure；❌ 把 runaway 早停截断当发作/expanded；❌ 称 z+m clean no-go；❌ 从 field 分数反挑参数、把 activation proxy 称真实 1–150 Hz 能量；❌ 把单例与 Fig3-B best-case 配对称独立验证；❌ 把"轴向保持可接受"当成改 topic4 主文档"必须破轴"框架（分支内口径）。

## 7. 文件 / 复现

- commits（分支未 push）：`16fb03f`（代码+测试+标定）、`41f54c1`（discovery+并行化 multiseed）、`c6751fb`（handoff）、`d3615c1`（P3）、（本报告+图 commit 待补）。
- 复现：`python scripts/run_topic4_mz_slowvars.py {calibrate,discovery,multiseed,capture-figures} --confirm-run`（discovery/multiseed 加 `--workers N`；multiseed 加 `--candidates results/topic4_sef_hfo/mz_slowvars/p3_candidates.json --seeds 1,3,4`）；再 `python scripts/plot_topic4_mz_slowvars.py`。
- 数据：`results/topic4_sef_hfo/mz_slowvars/{calibration.json, discovery_summary.json, per_run.jsonl/csv, p3_candidates.json, per_seed/multiseed_summary.json, readout_ready/, figure_capture.npz, figures/}`。
- 关联：[[project_topic4_m4_divisive_shared_inhibition_2026-07-05]]（S_G 池 bound-not-terminate）、[[project_topic4_m4_3a_discovery_2026-07-10]]（shunt clean no-go）——M4-MZ 是同行提的第三条终止/恢复机制候选。
