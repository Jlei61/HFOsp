# SEF-ITP Phase 4 v1 Modeling Track — Design Spec

> 状态：**v0.2 draft 2026-05-27**（v0.1 → v0.2 = user-return strict catch ratified；4 处 surgical edit: H-C reentry framing 拆解 + L4 principal-curve 从 hard gate 改 descriptive classifier + L3 swap_class per-sim layer fix + 3-proxy verdict strict 收紧）
> v0.1 → v0.2 commit history：spec 文件本身有 git log 追溯
> 上游 framework：`docs/topic4_sef_itp_framework.md` v1.0.7（待 banner v1.0.8 amendment）
> 上游 plan：`/home/honglab/leijiaxin/.cursor/plans/phase4-modeling-redesign_81bb208e.plan.md`（用户起草，本 spec 在其上细化）
> Archive 落地：`docs/archive/topic4/sef_itp_phase4_v1/`（spec lock 后建）
> Banner authorization：user-return 2026-05-27 catch ratified，scope = HR / Phase 4 staged roadmap / shaft control / cluster discipline / aniso D in stage 2 / smoke-first grid / 3-proxy sensitivity / adaptive gate / event extraction sensitivity sweep
>
> **v0.2 → v0.3 amendment 2026-05-28 (Stage 1b unit lock, user-ratified)**：节点事件单位 = **burst envelope**（一簇 spike 合并，event_time = 第一个 spike 起点），spike-level 仅 within-burst secondary。Stage 1 ⚠️（spike vs envelope 待裁决）已由单节点 baseline 标定 PASS 关闭。merge 阈值 `envelope_gap=30` 取自实测 inter-spike gap 谷底（非调参）。证据：`docs/archive/topic4/sef_itp_phase4_v1/stage1b_results_2026-05-28.md`；plan：`docs/superpowers/plans/2026-05-28-topic4-phase4-stage1b-burst-envelope-calibration.md`。下方 §3 Stage 2 入口契约 + §5.6 据此锁定。

---

## 0. 一句话承诺（CLAUDE.md §8 朴素话）

我们要做一个**最小可行的脑区尺度 2D 神经场模型**，看一个简单的"局部异常兴奋区 + 各向异性扩散 + 随机扰动"装置，**在哪些条件下**能在杆状采样上自然产生**正反两套时序模板**——也就是真实 SEEG 数据里看到的 PR-2 双簇模板。模型不是用来拟合任何病人，而是用来检验"几何 + 连接是否足以解释双向模板"这一假设。如果**控制条件下**（各向同性几何 + 随机方向 shaft）模型也照样能装出双向模板，那说明 PR-2 的双簇可能是分析 artifact (H-F)；如果**只有**在椭球几何或各向异性连接的特定组合下才出现双向模板，那 SEF-ITP H-A/H-C 假设得到机制支持。

---

## 1. Framework v1.0.7 → v1.0.8 amendment scope

修订 `docs/topic4_sef_itp_framework.md` §6.5 Phase 4 spec，三处：

| 项 | v1.0.7 字面 | v1.0.8 改字 |
|---|---|---|
| Phase 4 prerequisite | "Phase 1+2 H6+H1+H3 PASS + Phase 3 NULL" | "**framework structural prerequisite met**：phantom-rank 修复 done + Phase 1 runner 落地 done 即可启动 Phase 4 modeling track；model 输出作 mechanism exploration 层，**不替代** cohort verdict" |
| 节点动力学 | "FHN，不加 Hindmarsh-Rose / Epileptor" | "**HR (Hindmarsh-Rose) 主**，FHN / 简化 excitable unit 作 sensitivity；不模拟 HFO carrier 80–250Hz" |
| Shaft 采样 / cluster 判据 | 未规定 | "**强制 shaft 控制**：aligned / orthogonal / offset / angled / random / jittered，最少 6 种几何 + isotropic θ negative control；**禁止单用 KMeans k=2 作机制成功**；必须并列 split-half + odd-even + forward/reverse Spearman + rank-displacement swap_sweep + principal-curve audit；**isotropic θ + random shaft 也不能通过这套**" |

Banner: `user-return 2026-05-27 catch ratified`。

---

## 2. 核心科学假设（pre-registered）

H0：**间期群体 HFO 双向模板可以从"局部异常兴奋区 + 局部扩散 + 随机扰动" 三个 minimal 要素自然涌现，不需要预设 attractor / Hebbian 矩阵 / 状态切换**。

这条 H0 可被三种途径**支持**：

| 途径 | model setup | 对应数据假说 |
|---|---|---|
| **G-only**: 几何驱动 | isotropic D + elongated θ (rod) | H-A 狭长病理核两端 seed |
| **C-only**: anatomical-axis-substrate sanity | aniso D (D_x/D_y=3) + isotropic θ (blob) | **NOT** H-C reentry。aniso D 是 directional substrate（方向性传播底物），不是闭环 reentry。真正的 H-C reentry 检验留到 Phase 4+ `K_long` loop sensitivity（已立 issue, 不在本 v1 spec）|
| **G+C**: 几何 + 连接同向 | aniso D + elongated θ | SEF-ITP 整套图景 |

被**证伪**的条件：

- **negative control 通过**：isotropic D + isotropic θ + random shaft 也产生双向模板 → 模型层在装 BHPN/H-F artifact，framework 整个回退
- **proxy 不一致 (取严格版, user-return 2026-05-27 ratified)**：P1 (x) 通过但 P2 (dx/dt) / P3 (envelope) NULL → P1-only exploratory archive 注，**不进 framework v1.0.8 mechanism support 栏**（不是"主引 P1"，而是"三 proxy 一致才进 mechanism support"）

---

## 3. 五阶段路线图

### Stage 1 — 单节点 HR 基础设施

**目标**：证明 HR 单节点在选定参数下处于 excitable / bursting 边界，OU 扰动可触发 brief burst 然后回静默。

**入口契约**：无（首阶段）。

**实现**：

> **口径同步注（2026-05-28，Stage 1 实跑后锁定）**：以下扫描范围 / 检测阈值 / classifier 原为 framework 时拍脑袋估值，**已被 Stage 1 实跑重标定到 HR 实测时间尺度**。下方已改为实际口径；权威来源 `docs/archive/topic4/sef_itp_phase4_v1/stage1_results_2026-05-28.md` + plan v3.3。Stage 2 agent 以本节实际口径为准。

- HR ODE：`dx/dt = y − ax³ + bx² − z + I + η`, `dy/dt = c − dx² − y`, `dz/dt = r(s(x − x_R) − z)`
- 参数 baseline：a=1, b=3, c=1, d=5, r=0.006, s=4, x_R=-1.6
- 数值方案：RK4, Δt = 0.05 HR time unit；evaluate_cell 默认 burn_in=100 丢初始松弛瞬态
- 参数扫描（实跑口径）：`I ∈ [-1.5, 1.0]`, `r ∈ [0.003, 0.012]`, `σ_OU ∈ [0, 0.6]`, 多 seed per cell（旧估值 I∈[-2,0]/σ≤0.5 整个落在 silent 区，已废）
- 可视化：时间序列 x/y/z trace + phase portrait (x-y plane) + nullclines + 事件 overlay

**事件检测（实跑口径）**：spike-atom 检测 `detect_bursts`，spike-level 重标定默认 `x_threshold=0.0, min_burst_duration=0.3, bridge_gap=1.0`（旧 `x>1.0 持续≥5ms` 把每个 HR spike 都滤掉 → 全网格 0 事件，已废）。**primary 事件单位 = burst envelope**（`detect_burst_envelopes`，`envelope_gap=30` 合并 spike 簇，onset=第一个 spike 起点；Stage 1b lock），spike-level 仅 within-burst secondary。

**Regime classifier**：silent / **excitable** / repetitive-burst / unstable，四类自动分类（spike-unit RegimeConfig；excitable = 事件少且间隔大）。注：classifier 用于 Stage 1 选 baseline，**不**作用于 burst envelope（envelope 用 raw stats 判 excitable-like，见 Stage 1b archive）。

**退出契约**：

- 存在参数子带 (I*, r*, σ_OU*)，节点无外驱时静默，OU 推扰可 trigger brief burst 然后回静默
- 该 regime 对 noise amplitude ±50% 不漂
- **实跑锁定 baseline = (I\*=1.0, r\*=0.006, σ\*=0.4)**（I\* 贴网格上界是结构性的，非切早 artifact —— I≥1.5 时 σ=0 已自发点火）
- 输出 `regime_map.png` 3 轴扫描 + `regime_summary.json` (含选定 baseline)
- TDD 测试模块全 GREEN（见 §7）

**失败模式 → fallback**：

- 找不到 excitable regime → 改 x_R 或换 FHN-with-adaptation；不退回前阶段
- excitable regime 对 noise 极敏感 → 调 burst detection 阈值或换 baseline param 重试

---

### Stage 2 — 2D 均质脑区 sheet (anisotropic D sweep)

**目标**：在 θ ≡ 0（纯均质）下看 anisotropic D × OU noise 在 2D sheet 上的 regime map，建立 local-event regime 边界。

**入口契约**：Stage 1 给出的 (I*, r*, σ_OU*) + **事件单位 = node burst envelope**（onset = 第一个 spike 起点；Stage 1b lock，`detect_burst_envelopes` + `BurstConfig.envelope_gap=30`）。"local-event" regime 据此定义为多节点 burst-envelope onset 在短窗共现，不是 spike 共现。

**实现**：

| 项 | 默认 | 备注 |
|---|---|---|
| 网格 (smoke) | 80×60 nodes | **smoke first** — 调试 + regime 探索 |
| 网格 (replay) | 200×150 nodes | 仅在 smoke regime 锁定后跑代表性 cell sanity |
| 物理刻度 | 0.4 mm/node (smoke) 或 0.2 mm/node (replay) | 共 32×24 mm (smoke) 或 40×30 mm (replay)，多 gyrus 范围 |
| 扩散 | `(D∇²x)_aniso = D_x · ∂²x/∂x² + D_y · ∂²x/∂y²` | **作用于 x 变量** |
| **D_x/D_y sweep** | **{1, 2, 3, 5}** | **修订**：stage 2 已扫 aniso，**不**留到 stage 5；解决 confounding |
| K_long | **OFF** | stage 5 sensitivity 才开 |
| OU noise | per-node OU + spatial Gaussian conv λ_η=1.5 grid units | τ_η=10ms, σ_η sweep |
| 边界条件 | Neumann (no-flux) | |
| 数值方案 | semi-implicit Crank-Nicolson on diffusion + RK4 on local | numpy + scipy.sparse |

**Regime parameter sweep**：扫 (D_x/D_y, σ_OU, I)，自动分类每 cell 的 regime：

- **silent**：grid 上几乎无 burst
- **scattered**：node 独立 burst 无空间相关
- **local-event**：偶发短传播波 (< 20% grid 参与) ← target regime
- **synchronous-burst**：整 grid 同时 burst
- **traveling-pattern**：sustained 波 / 螺旋

**可视化**：

- 单 representative cell `x(t)` heatmap movie（5 个 cell）
- node raster（按空间位置排序）
- propagation speed map (local-event regime 内)
- population mean trace

**退出契约**：

- 至少 1 个 connected component 的 local-event regime
- 在 local-event 区域选 1 个 (D_x/D_y*, σ_OU*, I*) 作 stage 3 baseline
- 高分辨率 replay 同 cell：regime 标签**不漂**
- 输出 `regime_map_*.png` + `regime_summary.json` + 5 movies

**失败模式 → fallback**：

- 没有 local-event regime → 调 λ_η 或换 OU τ；如仍不行，HR 不适合 → fallback FHN 重做 stage 1
- local-event regime 太窄 (< 5%) → 该 regime 不 robust，stop and review

---

### Stage 3 — 观测层 (shaft + 3 proxy LFP)

**目标**：把 grid ground truth 转成 SEEG shaft 看得见的 LFP-like 信号；**并列 3 个 proxy 作 robustness sensitivity**。

**入口契约**：Stage 2 给出的 (D_x/D_y*, σ_OU*, I*) + grid baseline。

**Shaft 几何（pre-registered）**：

| 控制 | 数量 | 描述 |
|---|---|---|
| aligned | 1 | 沿 D_x 长轴穿 grid 中心 |
| orthogonal | 1 | 沿 D_y 短轴穿 grid 中心 |
| offset-parallel | 2 | 平行于 D_x 但偏 ±5 mm |
| angled | 2 | 与 D_x 成 30° / 60° |
| random | 5 | 随机 location + 随机 angle, fixed seeds |
| jittered | 1 | aligned 但每 contact 加 σ=1 mm random offset |

每 shaft = **10 contacts spacing 3.5 mm**（SEEG 标准）。

**LFP proxy（3 条并列）**：

| Proxy | 公式 | 物理近似 | tier |
|---|---|---|---|
| **P1** | `LFP_c(t) = Σ_grid w(d) · x_node(t)` | abstract membrane (与 PR-2 lag pattern 对齐) | **primary** |
| **P2** | `LFP_c(t) = Σ_grid w(d) · dx_node/dt` | synaptic current proxy (Buzsaki LFP 经典) | sensitivity |
| **P3** | `LFP_c(t) = envelope(LFP_P1 − local_detrend(LFP_P1, T=1s))` | locally-detrended HFO-like envelope | sensitivity |

Kernel：`w(d) = exp(−d² / (2 σ_LFP²)), σ_LFP = 1.5 mm`（SEEG effective sampling radius literature, Bartolomei 2017）

**Bipolar derivation**：每相邻两 contact `LFP_c − LFP_c+1`，与 Yuquan + Epilepsiae pipeline 对齐。

**退出契约**：

- 同一 grid ground-truth event 在 aligned shaft 上可识别 + orthogonal shaft 上次序几乎压平 + offset 上可识别但 rank 不同
- 3 个 proxy 在 aligned shaft 上 event timing 大致同向
- 输出 `shaft_overlay_2d.png` + `lfp_traces_per_shaft_per_proxy.png` + `event_lag_panel.png`

**关键 caveat**（写在 spec 红线）：

- LFP proxy 是**观测模型**，**不**等同真实 SEEG LFP（HR x 是抽象膜活动变量，不是 80-250 Hz HFO carrier）
- aligned shaft event count 显然 >> orthogonal — 这本身就是 shaft-geometry artifact，stage 5 必须显式报"每 shaft 几何在 isotropic θ + isotropic D 下 produce 多少 events"

---

### Stage 4 — Event extraction + regime / 参数空间分析 (with sensitivity sweep)

**目标**：在加 θ 异质性之前，定义"模型事件"的客观提取 + sensitivity sweep 锁定稳健 extraction regime。

**入口契约**：Stage 3 给出的 3 套 LFP trace per shaft per proxy。

**Event extraction sensitivity sweep（修订：不预 lock）**：

- **windows**：{30, 50, 100, 200} ms
- **thresholds**：{2σ, 3σ, 4σ}（σ = per-contact bipolar envelope baseline std）
- = **12 (window, threshold) cell** sweep
- 合并 close events；剔除多 seed 或全局爆发事件时记录原因

**Event metric** per (window, threshold, proxy)：

- event rate (events/min)
- event duration
- n_participating contacts
- seed count
- propagation speed (mm/ms)
- per-event lag pattern
- KMeans k=2 sanity（report only, **not** pass criterion）

**Regime map（升级版）**：(D_x/D_y, σ_OU, I, window, threshold) 5 轴，标 contact-level event rate / participation。

**退出契约**：

- 稳健 event regime：选 event rate ∈ [0.1, 1] event/s **且**跨 ≥ 6/12 (window, threshold) cell 通过的 (window, threshold) 子空间
- KMeans k=2 在 isotropic D + isotropic θ + random shaft 下 split-half/odd-even **NULL**（如不 NULL → H-F artifact 实证，stop-and-review）
- 输出 `extraction_regime_map.png` + `event_summary_per_cell.json`

---

### Stage 5 — 几何异质性 + 全套对照 (adaptive gate)

**目标**：测 H0 三条路径 (G-only / C-only / G+C) 是否成立；negative control 不应通过。

**Phase 划分（修订：adaptive，不预 lock 540 sims）**：

| Phase | Scope | 入口契约 | 总 sim |
|---|---|---|---|
| **5A smoke** | 4 cell (isotropic/elongated × isotropic D/aniso D=3) × 3 shaft (aligned / orthogonal / random) × 1 seed | Stage 4 regime locked | 12 sim |
| **Gate A → B** | Cell A (双 isotropic) **必须 fail** L2 + 至少 1 cell (B/C/D) **必须 pass** L2 (sub-criterion) | 5A 结果 | — |
| **5B mid-density** | 5A pass 的 cell × 12 shaft × 1 seed | Gate A pass | ~ 36-48 sim |
| **Gate B → C** | shaft control 表现合理（aligned 强 / orthogonal 弱 / random 中等）+ **advisor consult** verdict over-claim 检查 | 5B 结果 | — |
| **5C full predict** | Pass cell × 12 shaft × 5 seed | Gate B pass + framework v1.0.8 advisor pass | ~ 180-240 sim |
| **5D high-res replay** | 200×150 high-res 复跑代表性 cell × 3 shaft × 3 seed | 5C cohort verdict 落定 | ~ 27 sim |

**Stage 5 总 sim：~ 250-330**（不是原估 540）。

**θ 几何（pre-registered）**：

| 类型 | 描述 | 对应假说 | 期望 verdict |
|---|---|---|---|
| **isotropic blob (negative control)** | 单 Gaussian bump，圆形，σ_θ = 4 mm | — | **应 fail** L2/L3/L4 |
| **elongated rod** | 椭圆 Gaussian，长轴 ×3 短轴，σ_long=12mm σ_short=4mm | H-A | aligned shaft 应 pass，random shaft ~30% pass |
| **two-end hotspots** | 两个 ±10 mm separated Gaussian，各 σ=3 mm | H-B | aligned shaft 应 pass 但 principal-curve audit 应识别 bimodal 而非 manifold |

**Connectivity (aniso D)**：

| 类型 | 描述 | 对应假说 |
|---|---|---|
| isotropic (D_x/D_y = 1) | 各向同性扩散 | — |
| aniso (D_x/D_y = 3) | 各向异性扩散 | **anatomical-axis-substrate**（不是 H-C reentry — H-C 留到 K_long loop sensitivity） |

**2×2 factorial 切分**（替代原"3 θ × 12 shaft"线性矩阵）：

|  | isotropic D | aniso D |
|---|---|---|
| **isotropic θ** | **Cell A** negative control | **Cell B** anatomical-axis-substrate sanity（非 H-C reentry） |
| **elongated θ** | **Cell C** G-only test (H-A) | **Cell D** G+C test (SEF-ITP) |

**判据（4 层，禁止单用 KMeans k=2）**：

| 层 | 指标 | 期望（elongated rod + aligned shaft, cell C/D）|
|---|---|---|
| L1 KMeans descriptive | k=2 balance | ≥ 20% / cluster（report only）|
| L2 split-half reproducibility | per-cluster template Spearman r | ≥ 0.7 (mirror PR-2.5 "strong") |
| L2 odd-even reproducibility | 同上 | ≥ 0.7 |
| L3 forward/reverse | inter-cluster Spearman r | < −0.5 |
| L3 rank-displacement | swap_sweep `T_obs` / `decision_k` / `p_fw` per simulation (template-pair-level) | **每 sim 输出 strict / candidate / none label**；**cohort gate（across seeds, per cell condition）**：strict_or_candidate fraction ≥ 50% across seeds (修订：layer fix — swap_class 是 per-sim 不是 per-event) |
| L4 principal-curve audit (**descriptive, NOT a hard gate**) | 1D manifold fit BIC vs k=2 GMM BIC | **分类输出**："discrete bidirectional" / "continuous manifold" / "unclear"；与 L2/L3 verdict **并列报告**；continuous manifold 不等于 fail —— 真实几何传播也可能是连续流形（修订：从二元 gate 改 descriptive classifier） |

**3 proxy verdict 处理（修订）**：

stage 5 同 cell × shaft × seed 在 **P1 / P2 / P3 三 proxy** 上各自跑 L1-L4，**并列报告**：

- **framework v1.0.8 mechanism support 栏只在三 proxy 一致 PASS 时引用**（user-return 2026-05-27 strict catch ratified — 解决早期 spec 自相矛盾）
- 若三 proxy 全 PASS → archive 单独 highlight robust + framework mechanism 栏可引用
- 若 P1 PASS 但 P2/P3 NULL → **archive exploratory only**，标记 "primary observation mapping dependent" caveat，**不进 framework mechanism support**
- P2 / P3 verdict 始终 archive 作 sensitivity sub-report

**Negative control gate**：

Cell A (isotropic D + isotropic θ + random shaft) 必须**全 fail** L2/L3：

- L2 split-half/odd-even reproducibility median < 0.5
- L3 forward/reverse Spearman CI 跨 0
- L3 rank-displacement：cohort strict_or_candidate fraction across seeds < 25% (per-sim label, not per-event — 修订 layer fix)
- L4 principal-curve audit：分类为 "continuous manifold" 或 "unclear"（不期望返回 "discrete bidirectional"）— 注意 L4 是 descriptive 不是 hard gate，但 negative control 下 L4 出 "discrete bidirectional" 是异常信号

**如果 negative control 通过任何 L2/L3 判据 → stop-and-review (不调参 P-hack)**：要么 sim regime 选错了 (a)，要么 KMeans 本身可以闭眼出双 cluster (b)。两个都是 framework-level 意义，advisor consult 后才决定下一步。

**退出契约**：

- Cell A 全 fail L2/L3 (negative control gate pass)；L4 出 "continuous manifold" 或 "unclear" 为预期
- Cell C 或 Cell D **三 proxy 一致 PASS** L2/L3 才进 framework mechanism support；P1-only PASS 仅 archive exploratory
- Cell B 表现作为 **anatomical-axis-substrate 是否足以驱动方向性传播** 的描述性证据（pass / NULL 都报告，**不进 H-C reentry cohort claim** — H-C 真正检验留 Phase 4+ K_long loop sensitivity）
- L4 principal-curve 分类：每 cell × proxy 报告 "discrete bidirectional" / "continuous manifold" / "unclear"，作为对模型 nature 的描述，不入 verdict pass/fail
- Cross-shaft：aligned 强 / orthogonal 弱 / random 中等（shaft sensitivity 合理）
- 3 proxy verdict 三套并列报告（不一致则 archive caveat，**不进 framework mechanism**）
- 输出 `verdict_summary.json` + `figures/` + `README.md`（中文，AGENTS.md 规范）

---

## 4. 整体 discipline / 红线

继承用户 plan 4 条 + 新加 3 条：

1. **HR 主，FHN sensitivity**：避免结论依赖 HR 特定 burst regime
2. **LFP proxy 是观测模型，不等同真实 SEEG LFP**
3. **Abbott/Liou 只作连接思想来源**，不搬 chloride/ictal wavefront/exhaustion
4. **KMeans k=2 是下游摘要，不是机制成功标准**
5. **每个 stage hand-off contract 写死**：stage N 退出契约 = stage N+1 入口契约
6. **negative control gate 也是退出契约的一部分**：stage 5 不只要 cell C/D 通过，cell A 也必须 fail
7. **gate 之间强制 advisor consult**：每个 stage gate (5A→5B, 5B→5C) 之间用 advisor() 检查 verdict over-claim

---

## 5. 数学规格

### 5.1 节点动力学（HR）

```
dx_i/dt = y_i − a x_i³ + b x_i² − z_i + I_i + (D∇²x)_aniso[i] + η_i(t)
dy_i/dt = c − d x_i² − y_i
dz_i/dt = r ( s (x_i − x_R) − z_i )
```

| 参数 | 默认 |
|---|---|
| a, b, c, d | 1, 3, 1, 5 |
| r, s, x_R | 0.006, 4, -1.6 |
| I_i | `I_0 + θ_max · θ_shape(i)`（stage 2: θ ≡ 0；stage 5: per geom）|
| I_0 | **1.0**（stage 1 sweep 实跑 lock 2026-05-28；旧估值 -1.6 已废，见 Stage 1 archive）|
| θ_max | 0.3-0.6（stage 5 sweep）|

### 5.2 Anisotropic Laplacian

```
(D∇²x)_aniso[i,j] = D_x · (x[i+1,j] − 2 x[i,j] + x[i-1,j]) / Δ²
                  + D_y · (x[i,j+1] − 2 x[i,j] + x[i,j-1]) / Δ²
```

D_x/D_y sweep：**{1, 2, 3, 5}** in stage 2.

### 5.3 Spatial-OU noise

```
dη_ij/dt = −η_ij/τ_η + σ_η · ξ_ij(t)
ξ_ij = Gaussian_conv(white_noise, λ_η)
```

| 参数 | 默认 |
|---|---|
| τ_η | 10 ms |
| σ_η | 0.05-0.2 sweep |
| λ_η | 1.5 grid units |

### 5.4 θ 几何 generators

```python
# isotropic blob
theta(x, y) = exp(−((x−x_c)² + (y−y_c)²) / (2σ²))

# elongated rod (long axis aligned to D_x)
theta(x, y) = exp(−((x−x_c)² / (2σ_long²) + (y−y_c)² / (2σ_short²)))

# two-end hotspots
theta(x, y) = exp(−((x−x_c−Δ)² + (y−y_c)²) / (2σ²))
            + exp(−((x−x_c+Δ)² + (y−y_c)²) / (2σ²))
```

### 5.5 LFP proxy

```
P1(c, t) = Σ_node w(d(c, node)) · x_node(t)
P2(c, t) = Σ_node w(d(c, node)) · dx_node(t)/dt
P3(c, t) = |hilbert(P1(c, t) − rolling_mean(P1, T=1s))|
where w(d) = exp(−d² / (2 σ_LFP²)), σ_LFP = 1.5 mm
```

Bipolar：`B_c(t) = LFP_c(t) − LFP_c+1(t)`。

### 5.6 Event extraction

```
envelope_c(t) = |hilbert(B_c(t))|
crossings_c = times where envelope_c > μ_baseline + k·σ_baseline   (k ∈ {2,3,4})
event = {c, ..., c_n} where ≥3 contacts have crossings within window W ms (W ∈ {30, 50, 100, 200})
lag_c = first-crossing time within event
template = argsort(lag_c)
```

> **分辨率层提示（§6.2 教训）**：§5.6 是**观测代理事件**（shaft/bipolar LFP envelope 越阈），属 Stage 3-4，与 Stage 1b 锁定的**节点 ground-truth burst envelope**（`detect_burst_envelopes`，单节点 x 轨迹上 spike 簇合并）是两个不同层。前者从 LFP 信号 Hilbert 包络提事件，后者从节点动力学提事件。两者都叫 "envelope" 但层不同——不要混用。Stage 2 "local-event" regime（多节点 burst-envelope onset 共现）用的是后者。

## 6. 代码 / 测试 / 输出边界

### 6.1 Module

```
src/topic4_modeling/
├── __init__.py
├── hr.py            # Stage 1: HR ODE, RK4 integrator, burst detect, regime classifier
├── grid2d.py        # Stage 2: anisotropic Laplacian, spatial OU, K_long stub (OFF default)
├── observation.py   # Stage 3: shaft geom, 3 LFP proxies, bipolar derivation
├── regime.py        # Stage 2/4: regime classifier, parameter sweep, event extraction
├── heterogeneity.py # Stage 5: theta field generators (isotropic/elongated/two-end)
└── analysis.py      # Stage 5: split-half/odd-even/forward-reverse/swap/principal-curve adapters
                      #         (re-uses src/interictal_propagation.py + src/rank_displacement.py)
```

### 6.2 CLI scripts

```
scripts/
├── run_topic4_phase4_stage1_hr.py             # single-node param sweep
├── run_topic4_phase4_stage2_grid.py           # 2D homogeneous + aniso D sweep
├── run_topic4_phase4_stage3_observation.py    # shaft + 3-proxy LFP demo
├── run_topic4_phase4_stage4_regime_events.py  # event extraction sensitivity sweep
├── run_topic4_phase4_stage5_heterogeneity.py  # adaptive gate 5A→5B→5C→5D
└── summarize_topic4_phase4.py                 # cohort verdict summary
```

### 6.3 Test 模块（TDD）

```
tests/topic4_modeling/
├── test_hr.py
│   ├── test_hr_node_ode_integration_rk4
│   ├── test_hr_node_phase_portrait_nullclines
│   ├── test_burst_detector_hysteresis
│   └── test_regime_classifier
├── test_grid2d.py
│   ├── test_aniso_laplacian_isotropic_recovers_standard
│   ├── test_neumann_boundary
│   ├── test_spatial_ou_correlation_length
│   └── test_regime_classifier_population
├── test_observation.py
│   ├── test_shaft_contact_geometry_aligned_vs_orthogonal
│   ├── test_lfp_kernel_falloff
│   ├── test_lfp_three_proxies_consistency
│   ├── test_bipolar_derivation
│   └── test_event_extraction_min_participating
├── test_regime.py
│   └── test_event_sensitivity_sweep
├── test_heterogeneity.py
│   ├── test_theta_isotropic_blob
│   ├── test_theta_elongated_rod
│   └── test_theta_two_end_hotspots
└── test_analysis.py
    ├── test_split_half_reproducibility_adapter
    ├── test_forward_reverse_spearman_adapter
    ├── test_rank_displacement_swap_sweep_adapter
    └── test_principal_curve_audit_adapter
```

### 6.4 输出目录

```
results/topic4_sef_itp/phase4_modeling/
├── stage1_hr_single/
│   ├── regime_map.png
│   ├── regime_summary.json
│   └── phase_portraits/
├── stage2_2d_homogeneous/
│   ├── regime_map_aniso_sweep.png
│   ├── regime_summary.json
│   ├── representative_movies/
│   └── high_res_replay/
├── stage3_observation/
│   ├── shaft_overlay_2d.png
│   ├── lfp_traces_per_proxy/
│   └── event_lag_panel.png
├── stage4_regime_events/
│   ├── extraction_regime_map.png
│   └── event_summary_per_cell.json
└── stage5_heterogeneity/
    ├── phase_5a_smoke/
    ├── phase_5b_mid_density/
    ├── phase_5c_full_predict/
    ├── phase_5d_high_res_replay/
    ├── per_cell_per_shaft_per_seed/<cell>_<shaft>_<seed>.json
    ├── verdict_summary.json
    ├── figures/
    │   └── README.md   # 中文，AGENTS.md figures README 规范
    └── README.md
```

### 6.5 Archive

```
docs/archive/topic4/sef_itp_phase4_v1/
├── phase4_v1_design_2026-05-27.md          # 本 spec 的 archive copy
├── stage1_results_<date>.md
├── stage2_results_<date>.md
├── stage3_results_<date>.md
├── stage4_results_<date>.md
├── stage5_phase_5a_smoke_results_<date>.md
├── stage5_phase_5b_results_<date>.md
├── stage5_phase_5c_cohort_results_<date>.md
├── stage5_phase_5d_high_res_replay_<date>.md
└── README.md
```

---

## 7. 计算预算

| stage | 网格 | 时长 | param cells | seeds | runs | 单 run | 总（单核）|
|---|---|---|---|---|---|---|---|
| 1 | 0D | 100s | ~3000 | 5 | 15000 | 0.1s | ~25 min |
| 2 smoke | 80×60 | 200s | ~100 | 3 | 300 | 30s | ~2.5 h |
| 2 replay | 200×150 | 200s | 2 (representative) | 3 | 6 | 5 min | ~30 min |
| 3 | 80×60 | 50s | 1 cell × 12 shaft × 3 proxy | 5 | 60 | 30s | ~30 min |
| 4 | 重用 stage 2/3 | — | 12 (window, thresh) sweep | — | — | analysis | ~30 min |
| 5A smoke | 80×60 | 500s | 4 cell × 3 shaft | 1 | 12 | 1 min | ~12 min |
| 5B mid | 80×60 | 500s | pass cell × 12 shaft | 1 | ~40 | 1 min | ~40 min |
| 5C full | 80×60 | 500s | pass cell × 12 shaft | 5 | ~200 | 1 min | ~3.5 h |
| 5D high-res | 200×150 | 500s | representative × 3 shaft | 3 | ~27 | 5 min | ~2.5 h |

**总：~10 单核 hr，可并行到 ~1.5 wall hr on 8-core CPU**。

**Tech stack**：numpy + scipy.sparse + numba（热点 ODE jit）。不需要 GPU。joblib / multiprocessing 并行 parameter sweep 和 seed。

---

## 8. 失败模式 (pre-registered)

| 阶段 | 失败 | 行动 |
|---|---|---|
| Stage 1 | 找不到 excitable regime | 调 x_R / 换 FHN-with-adaptation; 不退回 |
| Stage 2 | 没有 local-event regime | 调 λ_η / OU τ; 仍失败 → FHN sensitivity; 仍失败 → stop and review |
| Stage 3 | 3 proxy 互相矛盾 | archive caveat; framework v1.0.8 mechanism 栏只引 P1; P2/P3 作 sensitivity |
| Stage 4 | event regime 子空间 < 6/12 cell | extraction-parameter dependent; stop and review extraction spec |
| Stage 5 Gate A | negative control 通过 L2 | **stop and review** (a) sim regime (b) KMeans/cluster pipeline; advisor consult 后才决定 |
| Stage 5 Gate B | shaft control 不合理 (aligned 不比 orthogonal 强) | shaft proxy 设计有问题; 退回 stage 3 重审 |
| Stage 5 5C | cell C/D 都 NULL | H0 在 modeled regime 内被弱化; archive 报告 + framework v1.0.8 mechanism 栏改"未在 model 内观察到几何驱动证据" |

---

## 9. Out of scope (Phase 4 v1)

1. ❌ patient-specific θ 拟合（future PR; framework v1.0.7 §7.2 红线）
2. ❌ HFO 80–250 Hz carrier 物理（HR burst ~5-15Hz，**not** HFO carrier）
3. ❌ Ictal scenarios（no Mexican hat, no chloride exhaustion, no sAHP slow K）
4. ❌ Multi-shaft full SEEG implantation simulation
5. ❌ 3D（保留 2D）
6. ❌ Clinical SOZ overlay
7. ❌ Patient-specific DTI / SC integration
8. ❌ Multi-subject 跨 model（model 不是 cohort 的）
9. ❌ Inhibition exhaustion (Mexican hat / chloride / GABA-A reversal)
10. ❌ Spike-timing dependent plasticity (Abbott §5 STDP-induced foci)

---

## 10. 与 framework v1.0.7 cross-doc 关系

- **取代** v1.0.7 §6.5 Phase 4 prerequisite 条款 + FHN-only 条款
- **保留** v1.0.7 §1-§5 framework 主体（H0 单核心 + H1-H6 + §4 失败模式表）
- **保留** v1.0.7 §7.2 Out-of-scope 红线（HFO carrier / E/I / patient fitting / endpoint=SOZ）
- **保留** v1.0.7 §8 红线（HFO 不分 E/I / H3 措辞锁 / sin not cos / k=3 / H1 三层 / H2 两条 reversal index / H3 SUPPORTED naming / H4 normalized instability / H5 statistical contract / H6 shaft-stratified shuffle）
- **新增** stage 1-5 staged roadmap + adaptive gate + 3 proxy sensitivity + shaft control matrix + negative control gate

---

## 11. Open / TBD (pending stage 1 实跑验证)

- HR 具体 baseline 参数 (I*, r*, σ_OU*) — stage 1 sweep 后 lock
- 2D local-event regime 边界 (D_x/D_y*, σ_OU*, I*) — stage 2 smoke 后 lock
- θ_max value — stage 5 5A smoke 后 lock
- 高分辨率 replay 选哪些 cell — stage 5C cohort verdict 落定后选
- K_long stage 5 sensitivity 形式（distance-decay vs sparse small-world）— stage 5C 后再 decide

---

## 12. 自检清单

- [x] §1 framework v1.0.7 → v1.0.8 amendment scope 写死
- [x] §2 三条假设路径 (G-only / C-only / G+C) + negative-control falsifier 明确
- [x] §3 五阶段每阶段有 入口/实现/退出/失败 四段式
- [x] §3 Stage 2 已扫 D_x/D_y aniso（不留到 stage 5；解 confounding）
- [x] §3 Stage 2 smoke first (80×60) + replay (200×150)
- [x] §3 Stage 3 三 proxy 并列 (P1 primary, P2/P3 sensitivity)
- [x] §3 Stage 4 event extraction 12 cell sensitivity sweep
- [x] §3 Stage 5 adaptive gate (5A → 5B → 5C → 5D, advisor consult between gates)
- [x] §3 Stage 5 2×2 factorial (cell A/B/C/D 对应 G-only / C-only / G+C / negative control)
- [x] §3 Stage 5 4 层判据 (L1 KMeans / L2 reproducibility / L3 forward-reverse + swap / L4 principal-curve)
- [x] §3 Stage 5 negative control gate (cell A 必须 fail)
- [x] §5 数学方程完整（HR + aniso Laplacian + spatial OU + 3 theta + 3 LFP proxy + event extraction）
- [x] §6 module / CLI / test / 输出 / archive 目录写明
- [x] §7 计算预算 ~1.5 wall hr on 8-core, numpy + numba, 不需 GPU
- [x] §8 每阶段失败模式 pre-registered 行动
- [x] §9 out of scope 10 条
- [x] §10 与 framework v1.0.7 cross-doc 关系
- [x] CLAUDE.md §8 朴素话 §0 写完
- [x] AGENTS.md figures README 中文规范引用

---

## 13. 下一步

1. **User reviews this spec**（本文档）→ 如有修改 → iterate
2. User approve spec → 写 implementation plan (`superpowers:writing-plans` skill)
3. Implementation plan approved → Stage 1 实施开始
4. 每 stage 完成 → archive results doc → advisor consult → 下一 stage
5. Stage 5 cohort verdict 落定 → framework v1.0.8 amendment 写进 `docs/topic4_sef_itp_framework.md`
