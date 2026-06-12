# Topic 5 — Stage 2b Early-Ictal Dynamic Pattern Echo Design Spec (2026-06-11)

> **状态**：设计稿 **v2.1**（v2 4-patch + 第二轮 review patch：术语硬区分 early-ictal main vs preictal secondary §1；新增 §6b preictal state-template alignment 层[三档窗 + 状态量非强度峰 + matched-interictal control，不阻塞 main]；§4 operational 主判据写明）。plan 待写。v2 patch：①max-over-time/feature null §2.2[最重要统计门]；②align_score 符号表 spec-locked+TDD §2.1；③cohort cache-building §2.3/§8；④latency eligibility + MIN_CH/MIN_GROUP §3.2/§3.4；+ Savitzky-Golay / region-label null / EEG-onset 窗。
> **Topic**：把"间期那条固定的高频传播通路（topic1 模板）"与"发作头 0–10s 的**动态激活模式**（不是首次越阈的逐触点 onset rank）"对齐——问"发作早期的激活/上升模式是否系统性地贴合间期模板的通道优先级"。
> **为什么 pivot（Stage 2 first-onset 仪器失败的诚实记录）**：Stage 2 用 band-power 变点（CUSUM）测逐触点首次招募时刻，detector repair 全扫（bias 0.5–2.0 × detrend{none,median,quantile} × {cusum,zcross}）**都不过门**。诊断确认 blocker 不是阈值标定，而是**最早被招募的触点在触点层面近乎同步、并列**——几十个触点在招募窗起点同帧越阈，"最早 3 个"在并列里随机挑，各特征挑各的（early_K_overlap 恒 0）；换 pass-1 onset 也救不了。**结论：发作早期在触点层面近乎同步，不能稳定定义"最早几个触点"。** 这**支持** Stage 1 谨慎口径（最稳的是病灶距离/早晚粗锚，不是具体路径重放），但**不是**"发作动态与间期模板无关"的负结论——只是"first-onset rank"这个量错了。Stage 2b 换量。详见 [[project_topic5_stage2_recruitment_2026-06-11]] + `results/topic5_ictal_recruitment/detector_sweep.csv`。
> **Owner**：topic5 拥有"真正发作 EEG 招募/动态层 + 间期↔发作桥接"（与 Topic 4 H5 不重复）。
> **前身**：Stage 2 spec `docs/superpowers/specs/2026-06-10-topic5-ictal-recruitment-stage2-design.md`（v2.1，仪器链路 + per-dataset montage + 缓存 + echo 复用，**全部继承**；只换"招募顺序量"这一层）。

---

## 0. 一个真正测什么的朴素话

每个病人有好几次发作。间期我们已知每个病人有一条形状固定、被高频事件反复扫到的传播通路（topic1"模板"——每个通道一个优先级排名）。Stage 2 想问"发作时谁先点亮、顺序像不像模板"，但发现**发作头一两秒里很多触点几乎同时亮**——"谁第一"测不准。

Stage 2b 换个问法，绕开"谁第一"：

> **发作头 0–10s 里，各通道的激活强弱 / 上升快慢这个整体模式，是不是和间期模板的通道优先级对得上？模板里排在前的通道，在发作早期是不是也更强、上升更快？**

具体三种朴素问法（互相印证）：
1. **模板对齐曲线**：每个时刻，把"此刻各通道的激活程度"和"间期模板的通道排名"算一个相关；得到一条 echo(t) 曲线，看 0–10s 里有没有系统性的正峰。——不依赖"谁第一"，依赖"整体哪些通道此刻更活跃"。
2. **上升快慢的潜伏期**：不用首次越阈，改用每个通道**上升最快的时刻**（或升到一半/八成的时刻）。同步起跳之后，传播差异体现在"谁升得更快/更早到峰"，比"谁先越阈"更能拆开。
3. **早期强度**：每个通道在 0–2s / 2–5s / 5–10s 的激活面积或斜率。问"模板里排在前的通道，是不是在早期窗里更强、更快"，而不是"是不是第一个"。

并列太多时，把同一杆 / 解剖区的触点聚合，先问"**区域**招募模式像不像模板"，再决定要不要回触点层面。

**揭示了什么（叙事）**：若这些动态量里有多条都显示"模板靠前的通道在发作早期系统性更强/更快"，就支持"间期刻板模板=被发作早期复用的网络骨架"；若都打平，则进一步坐实 Stage 1 的"共享粗锚为主"。

---

## 1. 一句话主张（locked framing）

在扩展的 epilepsiae(CAR) + yuquan(bipolar) 探索性 cohort 内，**发作头 0–10s 的动态激活模式（模板对齐曲线 echo(t) / 上升斜率潜伏期 / 早期强度）与该病人间期传播模板的相似度，系统性高于 within-seizure 通道身份打乱的零假设**（单侧），并在 within-shaft / anchor-matched null 下不完全消失。

- 判定语言只允许 "**像 / 不像 / 没看清**"；禁止 "predicts seizure" / "causes" / α-claim 升机制因果。
- **不再用 first-onset contact rank**（Stage 2 已证其在触点层面并列、测不稳）。改用 §3 的动态量。
- echo 统计复用 Stage 1 `src/topic5_echo_gate.py`；模板 = masked narrow（Main-A）为主、broad 为 §6 secondary。

> **术语硬区分（P1，三层不混结论）**：
> - **Stage 2b-main = early-ictal（发作早期），窗 `[0,+10]s`**——本 spec §2–§5 的主线。**不是 preictal**。文献支撑早期 spread 临床时间窗常落 ~3–10s（JAMA Neurol 2019 前 10s fast beta / rapid spread；early-spread 综述 1–14s 较强证据集中 3–10s），且经典 SEEG 量化是"快活动 + 相对 onset 延迟"（Bartolomei EI, Brain 2008）、空间快活动图（60–100Hz epileptogenicity maps）——都与 echo_curve/ramp/latency 一脉，而非单一 onset time。
> - **Stage 2b-secondary = preictal state（发作前状态），窗 `[-300,-100]/[-60,-10]/[-10,0]s`**——§6b，**另一层、另一套量**（状态漂移而非传播峰），**不阻塞 main sentinel**，单独 sub-plan。
> - **coarse 层 = region/shaft（§3.4）+ broad lagPat（§6）**——并列太多时退路。
> 三层各自结论分开写，不合并。

---

## 2. 核心对象：Z(contact, time, feature) — 已建/已缓存

Stage 2 已经把每个发作的每个特征算成了逐触点逐帧的 robust-z 轨迹 `Z[feature][contact, frame]`（line-length / broadband / HFA / spectral-edge，+ ER held-out），并缓存到 `results/topic5_ictal_recruitment/sentinel_cache/`（raw feature traces + fs/channels/pre_sec/eeg_rel/montage/baseline-pool）。**Stage 2b 直接消费这个 Z**，不重读 EDF。

- 时间轴：帧中心相对临床 onset 的秒（`t_center = frame*hop + win/2 - pre`），hop=0.1s，per-feature win（HFA=0.5s 其余 1.0s）。**echo_curve 必须把各特征插值到同一 0.1s 时间网格后再算**（不同 win 的 t_center 偏移半窗，否则 echo(t) 跨特征错位）。
- `dZdt` 平滑方法**锁死**：Savitzky-Golay（window=0.5s=5 frames, polyorder=2）对 robust-z 求一阶导；**不在执行期随手调**。
- 分析窗（**pre-registered 分层，不 post-hoc 加宽**）：early-ictal primary = `[0, +10]s` rel **临床** onset。sensitivity（仅稳健性，不进主判据）：临床 onset `[-5,+15]s`；EEG-onset-anchored `[-5,+15]s`（epi 有 `eeg_onset_epoch`，用它作 t=0 的 anchor 重算一遍；yuquan 只有单一 onset 退化）；compact confirmatory broadband `echo_mean[0,+5]s`。**只有 primary `[0,+10]s` 临床锚结果能承载 Stage 2b 主判据；sensitivity 窗只能支持 robustness，不能靠事后加宽窗口救回 failed primary。** **不再需要 data-driven global onset 的 -2s 重测窄带**（那正是制造并列的根源）；Stage 2b 直接在 early-ictal 窗上算动态量。
- **cache manifest（每 seizure）记录**：`fs/channels/pre_sec/eeg_rel/montage/template_id/feature_win/hop`，供 cohort 复算时上下文不丢。

### 2.1 alignment_score convention（**spec-locked，TDD 必锁——P1**）

> 把"对齐符号"从"实现期核对"升级成硬合同。所有量统一成 `align_score`，`align_score > 0` 永远表示**模板靠前（rank 小=源）的通道在发作早期更早 / 更强 / 更快**。

```text
template_rank: small = early / source.
- activation_z, dZdt, AUC_w, slope_w  (larger = stronger/faster):
      align_score = -Spearman(template_rank, value)
- latency rank (t_max_slope / t50_rise / t80_rise / t_peak; smaller = earlier):
      align_score = +Spearman(template_rank, latency_rank)
=> align_score > 0  <=>  template-early contacts are earlier / stronger / faster.
```

TDD：构造"模板靠前通道升更快/更强"的合成 Z → 所有族 `align_score > 0`；翻转 → `< 0`。符号写反 = 整个结论反号，必须被测试拦下。

### 2.2 max-over-time / max-over-feature null（**Stage 2b 最重要统计合同——P1**）

`echo_peak = max_t align_score(t)`（A 族）和"跨特征取最大"都是**时间 / 特征上的挑选**，普通点位 channel-shuffle null 会让 p 偏乐观。硬合同：

- **每次 shuffle 重算整条 `align_score_null(t)` 曲线**，null statistic 取 `max_t align_score_null(t)`（与观测同口径）。
- 若主张跨特征取最大，**null 里也跨特征取最大**（同 max-over-feature）。
- `p / e_k` 对这个 **max-null 分布**算（不是某个固定 t 的点位 null）。
- **预注册兜底**：除 max-over-time 主口径外，**预注册一个主特征 + 主时间窗**（建议 broadband + `[0,+5]s` 均值 `echo_mean`）作为无挑选的 confirmatory 量，与 max-over-time 主口径方向必须一致。
- 同一 max-null 机制套用到 within-shaft / anchor-matched null（§4）。

### 2.3 cache 两层（sentinel cache ≠ cohort cache——P1）

"不重读 EDF"只对 sentinel 成立（现有 `sentinel_cache/` 只覆盖 epi 1146:2/5 + yuquan litengsheng:0）。cohort 前必须建 cohort cache：

1. **sentinel**：直接用现有 `sentinel_cache/`。
2. sentinel 过门后 **`build-cache`**：每个合格 seizure 的 EDF **只读一次**，写 raw feature traces + §2 manifest 到 `results/topic5_dynamic_echo/cache/`。
3. **`per-subject` / `cohort` 只读 cache**，不读 EDF。

---

## 3. 三族动态量（minimal sentinel set；互相印证）

> 所有"模板对齐"统一成 §2.1 的 `align_score`（符号已锁，`>0`=模板靠前通道更早/更强/更快），over **common 通道**：`template valid_mask==True` ∩ 该量非 NaN ∩ montage-aligned（per-dataset alias，§5），且每 seizure/feature common ≥ `MIN_CH=8`（region-level ≥ `MIN_GROUP=4`）才进 echo。

### 3.1 A — 模板对齐曲线 `echo_curve`（主量，绕开并列）

对每个特征 f、每个时刻 t（early-ictal 窗内逐帧或逐 0.5s 窗，统一 0.1s 网格）：
- `activation_z[f][:, t]` = 此刻各通道的 robust-z。
- `dZdt[f][:, t]` = Savitzky-Golay 一阶导（§2 锁）。
- `echo_act[f](t) = -Spearman(template_rank, activation_z[f][:, t])`（§2.1 符号锁；>0=模板靠前更活跃）。
- `echo_slope[f](t) = -Spearman(template_rank, dZdt[f][:, t])`（>0=模板靠前升更快）。

**判读量（per seizure, per feature）**：`echo_peak = max_t align_score(t)`（**max-over-time null，§2.2**）、峰值时间 `t_peak_echo`、预注册主窗均值 `echo_mean`（broadband `[0,+5]s`，无挑选 confirmatory）。主张 = 多特征 echo(t) 在合理 early-ictal 时间（`t_peak_echo ∈ [0,~6]s`）出现**系统性正峰**且**经 max-null 仍显著**（不是挑了一个时间点/特征）。

### 3.2 B — 上升斜率潜伏期 `latency`（拆同步起跳后的传播差异）

**eligibility 硬条件（P1——否则平坦噪声通道也会产出"合法"latency）**：通道在 early-ictal 窗内须有真实上升，`peak_z ≥ Z_MIN`（lock=2.0）**且** `peak_z − z(at T0) ≥ DELTA_MIN`（lock=1.0）；不满足 → 该通道所有 latency = NaN（不进 rank）。每 seizure/feature 合格通道 ≥ `MIN_CH=8` 才进 echo。

每个合格通道、每个特征，在 early-ictal 窗内：
- `t_max_slope` = `dZdt` 最大的时刻（上升最快时刻）。
- `t50_rise` / `t80_rise` = robust-z 首次升到该通道窗内峰值 50% / 80% 的时刻。
- `t_peak` = robust-z 峰值时刻。

→ 每个量给一个逐触点的"潜伏期 rank"（小=早）→ `align_score = +Spearman(template_rank, latency_rank)`（§2.1 符号锁；>0=模板靠前更早到峰/半高）。这比 first-crossing 更能在同步起跳后拆出传播顺序（峰/半高时刻不并列在窗起点）。

### 3.3 C — 早期强度 `ramp_strength`

每个通道、每个特征，分窗 `0–2s / 2–5s / 5–10s`：
- `AUC_w` = robust-z 在窗 w 的积分（早期累计激活）。
- `slope_w` = robust-z 在窗 w 的线性斜率。

→ `align_score = -Spearman(template_rank, AUC_w)` 等（§2.1 符号锁；>0=模板靠前在最早窗里更强/更陡）。

### 3.4 D — region / shaft 聚合（并列太多时的退路）

把 §3.1–3.3 的逐触点量按 **shaft（`parse_shaft`）/ 解剖 region** 聚合（median over contacts in group；**group size ≥ 2 才成 region**），得 region-level 量，对 region-level 模板优先级（模板 rank 的 region 聚合）做同样的 `align_score`。**region-level null 必须在 region 单位上打乱（shuffle region labels），不能复用 contact-level channel-shuffle**（否则零假设的单位错了）；region-level 须 ≥ `MIN_GROUP=4` 个 region 才进 echo。先报 region-level echo（更稳），再回触点层面。

---

## 4. 判决合同（revised gate — 不再要求 early_K）

> Stage 2 的硬门 `feature_agreement_flag`（含 early_K_overlap≥0.3）**作废**（已证其卡在并列）。Stage 2b 改用以下门：

| 门 | 条件 |
|---|---|
| **方向一致** | §3 的多个动态量（A echo_act/echo_slope、B latency、C ramp）的 cohort 合并方向**同向为正**（模板对齐为正），不是单个量孤证 |
| **时间合理** | echo(t) 的系统性正峰落在 early-ictal 窗（`t_peak_echo ∈ [0, ~6]s`），不是窗尾偶发 |
| **null 不全消（max-over-time，§2.2）** | within-seizure channel-shuffle 的 **max-null** 下显著；且 **within-shaft 或 anchor-matched max-null 下不完全消失**（否则只是病灶距离粗锚，写"shared anchor"）。所有 null 都重算整条 echo(t) 取 max_t，与观测同口径 |
| **construct validity** | 多个**独立**特征（line-length / broadband / HFA / spectral-edge）+ ER held-out 的动态量方向一致（替代失败的 early_K：现在看"整体动态模式"跨特征是否一致，而非"最早 3 个"） |
| **双数据集** | epi(CAR) + yuquan(bipolar) **都**能画出可解释的 echo(t) 曲线且方向一致（cross-dataset bridge 必须，sentinel 上先看） |

- **站住·动态 echo（含路径）**：上述全过 **AND** within-shaft 或 anchor-matched 仍正 → 发作早期动态系统性贴合模板具体通道优先级。
- **站住·稳定锚为主**：channel-shuffle 下正但 within-shaft/anchor-matched 打平 → shared ictal/interictal channel-priority anchor。
- **没看清 / 阴性**：动态量方向不一致或全打平 → 当前动态量也没看到 echo（与 Stage 1 粗锚口径一致，不升级为"无关"负结论）。
- 合并仍 subject-level 优先（Stage 1 `pool_echo_subject_level`），dataset 分层 sensitivity。

**operational 主判据（写进 plan）**：
```text
Primary (early-ictal main):
  pre-registered broadband echo_mean [0,+5]s direction > 0
  AND max-over-time echo_peak survives channel max-null (§2.2)
  AND at least one of within-shaft / anchor-matched max-null remains positive
Construct validity (replaces failed early_K):
  activation_z, dZdt, ramp-AUC align_score mostly same direction across features
  ER held-out reports er_vs_fused consistency but does NOT vote
Secondary (preictal, §6b — separate conclusion):
  template-alignment trajectory monotone-rising toward onset across
  far/near/immediate windows AND exceeds matched-interictal control + max-over-window null
```

---

## 5. 复用 + 新代码

### 5.1 复用（不重造）

| 需要 | 复用 | 说明 |
|---|---|---|
| Z(contact,time,feature) 轨迹 + 缓存 | Stage 2 `_raw_traces`/`_z_from_traces` + `sentinel_cache/` | ✅ 直接消费，不重读 EDF |
| per-dataset montage + 对齐 | Stage 2 `ICTAL_REFERENCE`/`bipolar_alias_label`/`assert_channel_identity` | ✅ epi=CAR / yuquan=bipolar；模板 rank 方向核对 |
| masked narrow 模板（Main-A）| Stage 2 `_load_masked_template`（`rank_displacement` pairs[0]）| ✅ |
| echo 统计 + null（channel/within-shaft/anchor-matched）+ subject pooling | `src/topic5_echo_gate.py`（Stage 1）| ✅ 喂动态量 rank 而非 onset rank |
| shaft 解析（region 聚合）| `src.propagation_skeleton_geometry.parse_shaft` | ✅ §3.4 |

### 5.2 新代码

- **新建 `src/topic5_dynamic_echo.py`（纯数学，no I/O）**：
  - `activation_and_slope(z_trace, *, hop)` → `(activation_z, dZdt)`（**Savitzky-Golay** win=0.5s polyorder=2，§2 锁）。
  - `align_score(template_rank, value, *, kind)` → `kind∈{intensity,latency}` 的符号锁 score（§2.1）：intensity=`-Spearman`，latency=`+Spearman`；common-channel mask（valid ∩ 非 NaN），< MIN_CH → NaN。
  - `echo_curve(template_rank, z_by_t, *, mode)` → `align_score(t)` 数组（mode∈{activation,slope}），+ `echo_peak=max_t`、`t_peak`、`echo_mean`（预注册主窗）。
  - `echo_curve_null(template_rank, z_by_t, *, null_mode, blocks, B, rng)` → **每 draw 重算整条曲线取 max_t** → max-null 分布（§2.2；null_mode channel/within_shaft/anchor_matched）。
  - `slope_latencies(z_trace, *, window, hop, z_min, delta_min)` → eligibility-gated `{t_max_slope,t50,t80,t_peak}` per contact（不合格 NaN，§3.2）。
  - `ramp_strength(z_trace, *, windows, hop)` → `{AUC_w, slope_w}` per contact。
  - `region_aggregate(per_contact, shafts, *, min_group=2)` + region-label shuffle null（§3.4，不复用 contact-shuffle）。
- **新建 `scripts/run_topic5_dynamic_echo.py`**：`sentinel`（读 `sentinel_cache/`）/ **`build-cache`（§2.3，EDF 只读一次）** / `per-subject` / `cohort`（只读 cache）。**先 sentinel**，过门再 build-cache→cohort。
- **新建 `scripts/plot_topic5_dynamic_echo.py`**：echo(t) 曲线（每特征一条 + **max-null 带**）、latency/ramp 模板对齐散点、region-level 对照；paper-grade 自洽 + 中文 README。
- TDD（`tests/test_topic5_dynamic_echo.py`）：**符号锁回归**（"模板靠前升更快/更强"合成 → 所有族 align_score>0；翻转→<0）；echo(t) 在该合成上出正峰、在"同步并列无优先级"合成上平；**max-null 回归**（随机 Z 的 max_t observed 不超过 max-null 分布 → p 不假阳）；slope-latency eligibility（平坦噪声通道 latency=NaN）；slope-latency 在同步起跳+不同上升率下拆出顺序；region 聚合 + region-shuffle null 正确；Savitzky-Golay 导数符号正确。

### 5.3 输出

```
results/topic5_dynamic_echo/
├── cache/                    # cohort cache (build-cache, §2.3): raw traces + manifest
├── sentinel/                 # 每 sentinel seizure: echo_curve.json + echo(t) 曲线 PNG（A/B/C/region）
├── per_subject/<ds>_<sid>.json
├── cohort_dynamic_echo_summary.json   # subject-level verdict（§4）+ 每 max-null + dataset 分层 + 方向一致性
└── figures/ (README.md 中文)
```

---

## 6. Secondary line — broad / large lagPat（Stage 2b-2）

user 次线：发作早期参与的网络比 topic1 高-HI-index 窄模板大。所以 **secondary**：
- 模板换 broad lagPat（`results/lagpat_broad/` Yuquan 已封板；epi broad 未封板 → 只 sensitivity）。
- 在 early-ictal 窗内提取峰 / 尖波 / HFA burst / ramp peak，构 early-ictal lagPat-like pattern，与 interictal **broad** 模板做 §3 的动态相似度。
- 角色：扩展分析（更大网络是否更回声），**不替代** narrow Main-A；带 broad-unsealed caveat。

---

## 6b. Secondary — preictal state-template alignment（**另一层，不阻塞 main，单独 sub-plan**）

> **与 main 的硬区分**：main 问"发作早期传播动态像不像模板"；本层问"**发作前**，间期模板靠前的通道/区域是不是更早进入高风险**状态**"。preictal 是**状态漂移**，不是传播峰——**禁止复用 §3 的"强度峰值/上升斜率"口径**。本层**不阻塞 main sentinel**，main 过门后再单独立 sub-plan / 缓存。

- **窗（onset 前，三档）**：`far_preictal [-300,-100]s`、`near_preictal [-60,-10]s`、`immediate_preictal [-10,0]s`（文献 preictal 多在分钟级；Sci Rep 2022 报 SOZ degree/centrality 在 onset 前 ~37s 升、全通道 ~8s 升；Sci Rep 2019 PAC 用 300s vs 10s 窗区分 involved 电极）。
- **状态量（非 onset rank、非 §3 强度峰）**：
  - **primary**：HFA/HFO **rate 或 power 的 template-alignment 轨迹**（模板靠前通道的 HFA power/rate 是否随接近 onset 逐渐相对升高）。
  - **optional（更重，按需）**：PAC / PLV；directed connectivity centrality（degree / betweenness / inflow-outflow）；synchrony。这些是独立子工作量，按价值决定是否做。
- **对照（关键，与 main 不同）**：**必须与 matched interictal control windows 比**（同长度、远离发作/post-ictal/睡眠期边界的间期窗），不能只和 channel-shuffle 比——因为 preictal 漂移可能是全局状态变化而非模板特异。再叠 **max-over-window null**（跨三档窗取 max，null 同口径）。
- **判读**：模板对齐轨迹是否随接近 onset 单调升高 **且** 超过 matched-interictal control 的同口径分布。
- **角色**：探索性 secondary；与 main 的 early-ictal 结论**分开写**，不合并成一个 "Stage 2b PASS"。

---

## 7. Caveats & NOT-DO

1. **Stage 2 first-onset 失败已记录**：发作早期触点层面近乎同步 → 不能稳定定义"最早几个触点"。Stage 2b 换动态量。
2. **不是负结论**：Stage 2 失败 ≠ 发作动态与模板无关；只是 onset-rank 量错。Stage 1 谨慎口径（粗锚为主）仍是最稳。
3. **方向约定是 spec-locked 合同（§2.1）**：`align_score` 符号已在 §2.1 锁死并 TDD 必锁（否则 echo 符号反）；不再是"实现期核对"。
4. **exploratory**：sensitivity（窗 / 特征 / null）+ user 视觉巡视 echo(t) 曲线全过前不写 paper-level claim。
5. **双数据集必须都过**（epi+yuquan），montage per-dataset。
6. **不进 cohort 直到 sentinel echo(t) 曲线 epi+yuquan 都可解释且方向一致**（staged gate）。
7. **early-ictal main 与 preictal secondary 结论分开（§1 术语锁）**：不合并成单一 "Stage 2b PASS"；preictal 不复用强度峰口径、必须对 matched-interictal control。
- NOT-DO：不用 first-onset contact rank；不重读 EDF（用缓存）；不在 within-subject 写 α-claim；broad 未封板不进 Main 估计；**不把 preictal 状态漂移当 early-ictal 传播 echo**；**preictal 不只对 channel-shuffle（要 matched-interictal）**。

---

## 8. Staged execution

1. 写 spec（本文件）→ user review。
2. 写 plan（TDD）→ user review。
3. 实现 `src/topic5_dynamic_echo.py` + runner（TDD 绿）。
4. **sentinel**（从已有 `sentinel_cache/` 的 epi 1146:2/5 + yuquan litengsheng:0，必要时多缓存几个）：算 §3 三族量 + echo(t) + **max-over-time null（§2.2）**，出曲线图，**人工目视** echo(t) 是否有系统正峰、epi+yuquan 是否方向一致。
5. sentinel 过门 → **`build-cache`（§2.3）**：每合格 seizure EDF 只读一次写 cohort cache。
6. **`per-subject` / `cohort` 只读 cache**（§4 verdict）。
7. figures + archive doc（`docs/archive/topic5/dynamic_echo/`）→ sensitivity 全过后回链主文档。

---

## 9. 来源 / 关联

- `docs/superpowers/specs/2026-06-10-topic5-ictal-recruitment-stage2-design.md`（Stage 2 仪器链路 + montage + 缓存，继承）
- `results/topic5_ictal_recruitment/detector_sweep.csv`（Stage 2 onset-rank 失败证据）
- `src/topic5_echo_gate.py`（echo 统计 + null，复用）
- `results/topic5_ictal_recruitment/sentinel_cache/`（Z 缓存，复用）
- `results/lagpat_broad/`（broad 模板，secondary §6）
- memory [[project_topic5_stage2_recruitment_2026-06-11]]、[[project_topic5_echo_gate_2026-06-08]]

**文献锚（窗口 / 量化口径支撑，review 提供）**：
- Bartolomei et al., *Brain* 2008 — Epileptogenicity Index = 快活动出现 + 相对 onset 延迟（"快活动+延迟"，非单一 onset time；支撑 latency/ramp 口径）。
- Perucca et al., *Brain* 2014 — intracranial onset pattern 异质（LVFA / 周期 spike / sharp / spike-wave 等，且部分 onset pattern 也现于 spread 区）→ 不强压统一 first-onset detector。
- JAMA Neurology 2019 — 前 10s fast-beta power / rapid spread 与术后失败相关；early-spread 综述较强证据集中 3–10s → 支撑 `[0,+10]s` 主窗。
- Epileptogenicity Maps (Frontiers Neurol 2019) — 60–100Hz 快活动空间统计图定位 onset/propagation，baseline 选择敏感 → 近 echo_curve/AUC。
- Esteller 2001 — line-length 是 seizure **detection**（均延迟 ~4.1s），不是稳定的早期触点排序工具 → line-length 只作 detection-family 特征之一。
- Sci Rep 2022 — directed network centrality：SOZ degree/betweenness onset 前 ~37s 升、全通道 ~8s 升 → preictal 可定义为 centrality trajectory（§6b optional）。
- Sci Rep 2019 — preictal PAC（300s vs 10s 窗）区分 involved 电极 → §6b 状态量 + 分钟级窗。
