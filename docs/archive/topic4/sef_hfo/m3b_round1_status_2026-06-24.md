# M3B Round-1 STATUS — model↔SEEG **instrument-probe** bridge (Task 1–4, 2026-06-24/25, P1-reviewed)

> Plan-of-record: `docs/superpowers/plans/2026-06-21-...-plan.md` **at commit `19c3398`**
> （"model↔SEEG bridge" 重写版；⚠️ worktree 工作区里的同名文件还是旧的 a1213ee 版，hash 不符——
> 读 plan 必须 `git show 19c3398:...`，不要读工作区副本）。

## 朴素三段式（面向用户，外部读者也能复述）—— 全 Round-1（Task 1–4）

1. **测了什么** — 把模型（已验证的"戳一下"驱动速率场）的传播场透过"假电极"读出，问两件事：它像不像真实病人的**间期 /
   发作**传播图样，而且这个"像"靠的是模型的**传播结构**还是只靠**电极几何**摆位。

2. **怎么测的** — 四步：① **仪器门**：假电极能不能回收模型本有的 45° 主轴（误差 3.3°）；② **落点**：模型场和真实队列的
   场相似度 + 它落在队列内部相似度分布的百分位；③ **胜过几何**：把模型每触点的先后排序随机打乱、电极几何不动，看真实排布
   是不是比随机更像（"如果只是电极摆位，打乱也该一样像；实测真排布明显更像"）；④ **增益**：把网络兴奋性调大，看事件长大时
   形状 / 轴变不变。发作那一腿把真实场换成**发作早期激活场**重做 ②③。

3. **揭示了什么** —
   - **间期 = 结构层**：模型落进真实队列（相关 0.84、上四分位；参数稳健；对更干净的纯二维队列是中位、仍在内），**且胜过几何**
     （打乱零分布 channel p=0.001；对纯二维队列仍胜 p=0.014）→ 像靠的是模型 **45° 传播结构**、不只是电极几何 = **共享间期 scaffold**。
   - **发作 = 只到粗/几何层**：落进队列（0.42）但**不胜几何**（p=0.13）→ 和数据侧 A 线"间期↔发作只共享粗骨架、细对齐弱"**完全同型**。
   - **增益 = 够不着量程**：kick 幅度全或无、兴奋性把事件只放大 ×1.11 → 事件**~固定大小**，"两个增益"在这个速率场仪器里测不出来
     （需 SNN / 自发层 = M3A）。
   - **全程口径 = kick 速率场 instrument-probe**，**不主张"自发机制已复现真实 scaffold"**；模型 record 自标 1D → 轴可读、2D 跨杆宽度待加强。

（内部代号补注：M3B / increment3a 四对照 C1 / `compare_model_to_cohort` / 模型 rank-置换几何 null / A 线四层 null /
`bb_auc` / `mean_field(ratio)` / cm-SNN / M3A `s_slow` / θ_EE=45° / rho_EE=0.6。）

---

## 1. Task 0 — Round-1 scope freeze（plan §5 Task 0）

- **Round 1 = the bridge（PRIMARY）**：模型传播场透过虚拟 SEEG 读出后，落不落进真实 SEEG 间期队列自身的场相似度分布里，
  并且过同一套 geometry/anatomy null（A 线四层 null）。
- **"同一场、两个增益" readout** 折进同一套度量（secondary，Task 3）。
- **降级/删除**：模型侧 W 三法估计 / `W_event` estimator → optional appendix；**分辨率扫 `n_bins∈{5,9,11}` 直接删**
  （要做就一次跑满分辨率，不做梯度）。
- **Lane B2（`W_eff(s)`）** 保持 gated 在 M3A 的 `s_slow` 上，本轮不碰。

## 2. 两个 §0 审计发现（写死，免得再踩）

- **模型轴是精确已知的 45°，不用估**：`src/snn_engine/params.py` `rho_EE=0.6` 椭圆指数核，长轴沿 (1,1) 对角 = 45°。
- **之前反复出现的"W ≈ 距离 / 没解出轴"是分辨率假象，不是科学结论**：bin=4mm vs 事件 r95≈5mm，读出被 4 个正交近邻
  主导，物理上分辨不出 45° 对角。磨更细的 W estimator 是在追假象。

## 3. Task 1 — 模型→虚拟 SEEG record 适配器 + 45° sanity gate ✅ SANITY-PASS

- 虚拟 SEEG 观测层模块**定位完毕** = `src/sef_hfo_observation.py`（成熟、phantom-masked）。
- pilot：`scripts/run_m3b_task1_bridge_pilot.py` → `results/topic4_sef_hfo/m3b_bridge/task1_pilot/`
  （`task1_pilot_summary.json` + `model_record_lif_rate_45deg.json` + `figures/task1_axis_recovery.png` + README）。
- **结果**：observation 轴误差 **3.3°**（readability 0.977）；record 源/汇主轴 **41.7°**（误差 3.3°）；
  `compare_model_to_cohort` schema 合法（status ok、12 通道、六个 scalar 齐全：axis_length 11.57mm、
  rank_vs_xnorm_spearman 0.972）。**门 3.3° ≪ 25° → PASS。桥不是分辨率不可测，可进 Task 2。**
- 复用链确认可用：`_read`/`_integrate`（increment3a）出 LagPatArtifact → `build_record_from_events` 出标准 record。

## 3.6 Task 2 — the bridge（Parts 1/2/3 完成；**INSTRUMENT-PROBE**，P1 review 已修）

**口径锁（P1-1）**：模型场 = **kick 驱动的 LIF 速率场**（labeled instrument probe），**不是**自发 M3 事件。
下面每条 verdict 都是 **"instrument-probe bridge"** 陈述——**不主张"自发机制已复现真实 scaffold"**。
复用 driver `run_real_vs_model_comparison.py`（main 仓库 71 条真实 record）。输出 `.../m3b_bridge/task2_bridge/`。

**Part 1 — placement（落点）**：模型归一化传播-rank 场与真实间期队列中位场相关 **0.84**、落在真实"队列内部相似度
分布"的 **74 百分位**（z=+0.40，subject-first n=27）→ **落进真实间期队列、略高于中位**。方向轴 scalar 高端
（rank-vs-位置单调 pct 100、早晚分离 pct 81）；尺度 scalar 低（axis_length pct 7、transverse pct 15）= 片小、已归一化抵掉。
⚠️ **P1-4 caveat**：模型 record 自标 `one_dimensional_sampling=true`（招募触点沿 45° 轴近 1D）→ **45° 轴可读，但
"完整二维 SEEG 几何桥"（跨杆 transverse 宽度）仍是 caveat**，不写"完整 2D 几何桥"。

**Part 2 — geometry null（胜过几何；P1-2 subject-first 重跑）**：`run_m3b_task2_geometry_null.py`（B=2000，
**subject-first 折叠**，与正式 placement 分母一致——不再对 54 条 t_a/t_b 直接取 median）。
- ⚠️ **§6.1 修正**：A 线四层 null 是"间期轴↔发作激活、同被试逐通道按名配对再置换"，模型虚拟触点**进不了**；
  按 plan §3.1/§4 **意图**做了**模型专属 rank-置换 null**（固定触点几何/support、只打乱招募 rank）。
- **channel 层**：实测 0.84 > null p95 0.73，**p=0.0010 → 胜过**（plan 要求的最低几何门）。
- **within_shaft 层**（A/B/C 三杆）：0.84 > p95 0.77，**p=0.0030 → 胜过**。
- **anchor_matched / joint**：**N/A**（按发作活动分箱，模型没有）——诚实标注。

**Part 1+2 合起来（§6.3，不收成 "bridge PASS"）**：**instrument-probe** 下，模型传播场**落进真实间期队列**
（74 pct、相关 0.84）**且胜过几何剥离 null**（channel p=0.001、within_shaft p=0.003）→ 匹配靠模型 **45° 传播结构**、
非仅触点几何 = **共享间期 scaffold（结构层）**。与 A 线**数据侧**同型（粗骨架稳赢 channel null）。

**Part 3 — model 轴 ↔ 发作早期轴（PLACEMENT-ONLY；P1-3 主队列=Epilepsiae）**：复用 `make_field_record` + 发作
`t0_feature_cache`（broadband `bb_auc` = A 线 PRIMARY，每被试 over 合格发作取中位、铺在该被试间期轴框架上）。
**主口径 = Epilepsiae n=18**（Topic 5 主队列；Yuquan 仅 1 例**另列 descriptive、不并入主**）。
- **落点**：模型↔发作场中位相关 **0.42**、落在 Epilepsiae 发作场分布的 **72 百分位**（落进队列）。
- **几何 null**：**不胜过**（channel p=**0.126**、within_shaft p=**0.119**）→ **发作腿 = placement-only / 几何层**。
- 这与**数据侧 A 线本身**一致：间期↔发作只共享**粗骨架**、细对齐弱（发作场互相也只 ~0.4，远低于间期 0.84）。
- **HFA 敏感性**（换数据侧"细对齐"的 60–100Hz `hfa_auc` 重做，A 线里唯一稳赢 joint null 的指标）：**仍 placement-only**
  （corr 0.46、67 百分位，channel p=**0.16** 不胜几何）→ 模型连数据侧**更细的** HFA 发作梯度也只匹配到几何层；发作腿结论对
  broadband / HFA **都稳健**（`task2_part3_ictal_axis_hfa_auc.json`）。

### Round-1 桥 — 分层口径（§6.3 + 全部 instrument-probe）
- **模型(kick-rate-field) ↔ 真实间期 scaffold**：落进队列（74 pct/0.84）**且胜过几何**（channel p=0.001）→ **结构层共享间期 scaffold**（instrument-probe）。
- **模型 ↔ 真实发作早期梯度（Epi n=18）**：落进队列（72 pct/0.42）但**只 placement-only/几何层**（p≈0.12）→ **粗/几何层**。
- 合起来 = **kick-rate-field instrument-probe bridge**：复现的是**间期传播 scaffold（结构层，胜过几何）**；与**发作早期梯度**
  只到**粗骨架/几何层**——和数据侧 A 线"间期↔发作共享粗骨架、细对齐弱"**同型**。plan §4 严格 "B-PASS field bridge"
  （落点+胜几何+发作轴对齐）**未全满足**；诚实归类 = 间期 **B-PASS field bridge (instrument-probe)** + 发作 **B-PASS placement-only**。
  **不主张机制相变 / 自发复现**；1D-sampling caveat 限定"轴可读、2D 跨杆宽度待加强"。

### 3.6.1 Robustness（间期桥稳健性，autonomous 加跑）

- **参数敏感性**（S_THRESH×OVERLAP_MIN 9 格，`run_real_vs_model_comparison --sensitivity`）：落点稳健——
  median_corr 0.82–0.86、field_pct 63–74，overlap_min 无影响。
- **纯二维真实队列**（`--real-2d-only`，剔除 1D 真实 record，n=47 record / 24 subj；针对 P1-4 公平性）：
  - **落点** = median_corr **0.76 / 50 百分位**（仍**落进**队列，但从上四分位降到**中位**）→ 原来"上四分位"部分是被 1D 真实
    record 抬高的；对更干净的纯二维队列模型是中位-典型（仍在内、非离群）。
  - **几何 null 仍胜**：0.76 > null p95 0.71，channel **p=0.014**、within_shaft **p=0.039**（比全队列 0.001/0.003 弱但仍显著）。
- **小结**：载重的"**胜过几何 = 结构层共享**"对 1D/2D 公平性**稳健**（全队列 + 纯二维都胜）；只有"落点上四分位"软化为"中位、仍在内"。
  artifacts：`task2_bridge/{sensitivity/sensitivity.json, real2d_only/, task2_geometry_null_2d_only.json}`。

## 3.7 Task 3 — same field two gains（SECONDARY instrument readout = **INCONCLUSIVE / 仪器量程限制**）

P1-1 锁：instrument-probe，**不写机制相变**。问：把"招募增益"调大，传播**形状/轴**变不变（同一 scaffold 两个增益）。
- **gain knob 选择（plan §3 pilot 决定并 log）**：① kick 幅度 sweep（amp 5→16）**全或无**——事件逐字节相同
  （轴 3.3°、n_part 12、support 3869px、落点 0.844），kick 幅度**根本不是增益旋钮**（过阈即定，传播由工作点决定）。
  ② 换**兴奋性**旋钮（operating-point ratio 0.45→0.75，kick 固定过阈）——事件 extent 只变 **×1.11**（support 3490→3869px），
  **没有梯度增益量程**。
- **结论 = INCONCLUSIVE（仪器量程限制，NOT 反对 scaffold）**：该速率场 instrument 在当前 regime 事件**~固定大小**
  （印证 static-μ flat-event caveat，现两个独立旋钮都证实）。"两个增益"在这个仪器里**够不着**。
- **唯一正向**：那点小变化里，轴稳在 ~45°（range 1.1°）、形状几乎不变（consec corr ≥0.989）→ 有变化的部分 scaffold 稳；
  但量程太小，不足以下 "two gains" 强结论。
- **去哪找增益量程**：真正的招募增益需要 SNN/自发层（事件大小由网络动力学涨落决定）= **M3A 领域，本轮不碰**。
- `scripts/run_m3b_task3_gain_sweep.py` → `task3_gain_sweep/task3_gain_sweep.json`。

## 3.8 Task 4 — figures（每图一问，§7 纪律）

- Fig 1 `task1_pilot/figures/task1_axis_recovery.png`：仪器门——读出回收已知 45° 轴。
- Fig 2 `task2_bridge/figures/bridge_interictal.png`：A 模型落进真实间期队列 / B 胜过几何 null（channel）。
- Fig 3 `task2_bridge/figures/bridge_ictal_and_gain.png`：A 发作腿 = placement-only（不胜几何）/ B 事件大小~固定
  （增益量程够不着）、轴稳 45°。
- 中文逐图说明：各 `figures/README.md`。

## 4. 资产确认（plan §3 "confirm before build"）

- **桥度量** `compare_model_to_cohort(model_record, real_records, X, Y, ...)` — `src/propagation_contact_plane_readout.py:370`；
  record 需要 `channels[]`(每通道 `x_norm/y_norm/typical_rank/support/uncertainty_rank/signed_transverse_mm`) +
  `scalars{}`(六项) + `axis_length_mm/norm_scale_mm`；六个 scalar 由 `compute_cohort_scalars` 预算（compare 不自己算）。
- **轴框架** `compute_axis_frame(coords, source_idx, sink_idx)` :75；source/sink 由 `build_endpoint_cores` 取 rank 两端 k=3。
- **A 线四层 null**（channel/within_shaft/anchor_matched/joint）= `src/topic5_axis_alignment.py` 纯函数。
- **真实数据在主仓库**（worktree 的 results/ 是 gitignored、不含这些）：
  - 间期轴：`/home/honglab/leijiaxin/HFOsp/results/spatial_modulation/propagation_geometry_broad/components/path_axis/per_subject/<subject>.json`（n_ok=9）+ `cohort_summary.json`。
  - 发作早期轴：`/home/honglab/leijiaxin/HFOsp/results/topic5_ictal_recruitment/axis_alignment/axis_alignment_FINAL.json`（n=18）。
- **Topic 5 A 线轴 contract = 已接受/锁定，不 stale**（plan §8 前置条件满足）：粗骨架稳赢 channel null（FDR q≈0.02、
  LOSO p≈0.015），细对齐仅 60–100Hz HFA 稳赢 joint null（q≈0.029，split-half 不稳 → sensitivity 档）；符号自由共线 ≠ 逐点重放；
  无 phantom 污染（用发作激活，不用 lagPatRank）。来源 `docs/topic5_seizure_subtyping.md` §3.0 + `docs/archive/topic5/axis_alignment_AB_result_2026-06-14.md`。

## 5. 两个岔口（已按"严格 plan" + P1-1 review 锁定 = kick-rate-field instrument-probe）

> **已定（P1-1）**：Round 1 桥 = **已验证 kick 速率场**，全程 **instrument-probe** 口径；自发-cm-SNN 读出**另立后续 validation**，
> 不在 Round 1。所有 verdict 写 "instrument-probe bridge"，**不写"自发机制已复现真实 scaffold"**；Task 3 只按 secondary
> instrument readout 开，**不写机制相变**。下表保留为决策依据。


**计划 Task 1 原话 = "collect spontaneous events" on "one accepted Stage-3/M3 (cm-SNN) substrate"。但审计发现：**

| 维度 | 计划原话 | 已验证可用的 | 现状 |
|---|---|---|---|
| 事件来源 | 自发事件 | **kick 驱动** | 自发读出**任何尺度都没验证过** |
| 底物 | cm 脉冲网络 (SNN) | **LIF 速率场** (L=24,n=96) | cm-SNN 在 density=100 **点不着**（kick 盘内神经元太少） |

- **§4 sanity gate 的本质是"仪器能不能分辨 45° 轴"**（分辨率问题，与事件来源无关）→ 用已验证的 kick-速率场是正当的，
  这一锤已 PASS。
- **Task 2 搭桥用哪种模型场是真正的科学决定**，本 STATUS **不替用户拍板**（CLAUDE.md §1/§5）。推荐口径：
  Round 1 桥用**已验证的 kick 驱动速率场**，明确标注为 instrument probe（读"传播出去的形状"，不读 kick 的径向种子——
  正是 plan Task 3 允许的 fallback）；把**自发-cm-SNN 读出另立为后续 validation**，不在 Round 1 阻塞桥。

## 6. 分支状态（重要工程现实）

- 重写版 plan（`19c3398`）在分支 **`topic4-event-extent-audit`**；本 worktree 在 **`topic4-snn-m3-hub`**（HEAD `0942e9f`）。
- worktree 工作区的 plan 文件副本仍是**旧版**（hash `91c41b8` ≠ `19c3398` 的 `75d1f11`）——执行以 `git show 19c3398:...` 为准。
- 本轮所有产物（pilot 脚本、record、figure、本 STATUS）落在 worktree 分支；results/ gitignored。

## 7. 下一步（PILOT-FIRST，等用户定 §5 岔口后）

1. **定 §5 岔口**：桥的模型场来源（推荐 = 已验证 kick 速率场，instrument-probe 口径）。
2. **Task 2 桥（PRIMARY）**：`compare_model_to_cohort(model_record, 真实间期队列)` 的 scalar + field 落点；
   过 A 线四层 null（至少 channel/几何剥离层，对齐数据侧门槛）；再比发作早期轴。**先报 model-vs-cohort 落点，不收一句 "bridge PASS"**（plan §6.3 pronoun 纪律）。
   工程前置：真实数据在主仓库 results/，需从主仓库路径读（或桥脚本指向主仓库）。
3. **Task 3 同场两增益**（折进 Task 2 同一度量，secondary）：gain source pilot 决定（内生事件大小 / kick 强度 / μ 旋钮）。

---
verdict: **Task-1 SANITY-PASS + Task-2（Parts 1/2/3）done = kick-rate-field INSTRUMENT-PROBE bridge**（P1 review 已修）。
**模型(kick-rate-field) ↔ 间期 scaffold = 结构层共享**（落点 74 pct/相关 0.84 + 胜过几何 channel **p=0.001** / within_shaft **p=0.003**，subject-first n=27）；
**模型 ↔ 发作早期梯度（Epi n=18）= placement-only/几何层**（落点 72 pct/相关 0.42，不胜过几何 p≈0.12）——与数据侧 A 线
"间期↔发作共享粗骨架、细对齐弱"同型。诚实归类 = 间期 **B-PASS field bridge (instrument-probe)** + 发作 **B-PASS placement-only**；
**不主张机制相变/自发复现**；P1-4 caveat = 模型 record 自标 1D，轴可读但 2D 跨杆宽度待加强。
§5 岔口按"严格 plan"定为 kick 速率场（plan §3 accepted-substrate=已验证资产；Task 3 允许 kick-probe）。
**Task 3（同场两增益，secondary）= INCONCLUSIVE**：kick 幅度全或无、兴奋性旋钮事件 extent 只 ×1.11 → 仪器够不着增益量程
（印证 static-μ flat-event；轴/形状在小变化内仍稳）；真正增益量程需 SNN/自发层 = M3A。Task 4 figures 已出。
**Round 1 完成（Task 1–4），等用户审阅。**
artifacts: `results/topic4_sef_hfo/m3b_bridge/{task1_pilot,task2_bridge,task3_gain_sweep}/`；脚本
`run_m3b_task1_bridge_pilot.py` + `run_m3b_task2_geometry_null.py` + `run_m3b_task2_part3_ictal_axis.py`
+ `run_m3b_task3_gain_sweep.py` + `run_m3b_figures.py` + 复用 `run_real_vs_model_comparison.py`。
prior: [[m3_b1_validation_recap_2026-06-24]]（B1 收口 = B-BOUNDED NEGATIVE）。
followup: [[m3b_spontaneous_bridge_2026-06-25]]（自发场桥探索：自发**能**搭桥但读出**确定性/与种子无关**=几何决定的沿轴模板，
不比 kick 更强；不主张机制自发复现）。
