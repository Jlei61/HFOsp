# SEF-HFO Step 1 — noise-driven spontaneous events: locked contract (2026-06-03)

> **状态**：Step 0 已过 mechanism-scale gate（见顶部 framework v2 verdict）。Step 1 已解锁。本文件把用户 2026-06-03 的 6 条方法学加固**锁成数值合同**，开工前先冻结，之后所有结果对照本合同，不允许事后调检测器/挑 seed/调噪声幅度去凑率。
> **口径**：mechanism-scale 粗量级筛查，不是 patient-fitted 拟合（与 Step 0 同口径）。
> **canonical 代码**：`src/sef_hfo_lif.py`（已 w_ee_mult 贯穿场 + bistability-aware mean_field，commit e95af61）。Step 1 不得在 runner 里重新内联动力学（dual-track 已在 e95af61 消除）。

---

## 0. 朴素话：Step 1 在测什么

Step 0 是"我手动戳一下让它点着、看它会不会定向铺开再自己熄灭"。Step 1 **撤掉手指**，只给系统加**随机涨落**（模拟真实输入的抖动），问三件事：

1. 涨落弱时——系统**不会**自发持续响（静息稳）。
2. 涨落到某个范围——系统**偶尔自己点着**，且点着的是**一颗一颗分开、会自己熄灭**的事件（不是一直响 = 持续振荡，也不是越烧越大 = runaway）。
3. 这些自发事件的**铺开方向**仍然跟着连接各向异性轴 θ_EE 走，**不跟电极杆几何走**。

组织视角（促临界 ↔ 稳态回拉拮抗）：噪声 = 促临界推力（偶尔把局部推过点火阈）；recovery（适应）+ 抑制 = 稳态回拉。一颗"事件" = 推力偶发点着、回拉在 runaway/被高态俘获之前赢、自终止。

---

## 1. 锁死的事件检测器合同（用户风险 #2：先锁定义）

在任何用于结果的仿真之前冻结。检测器作用在 canonical `integrate_lif_field` 返回的 **`ext_coh`（coherence 活跃比例 = 把 `rE` 先做空间高斯平滑 `coh_len≈ELL_PAR` 再按 `> op["nuE"] + DETECT` 阈值，DETECT=0.005 kHz）** + `front`（每步活跃区最大 x）+（窗口 B 需要）末态场。`dt=0.25 ms`。

> **v1.1 coherence amendment（2026-06-03，开 smoke 后 + advisor）**：原 v1.0 用 raw 每像素活跃比例 `ext`，但每像素 OU 噪声 speckle 会直接抬高 raw `ext`（smoke：σ=2.0 speckle raw_ext 0.048 / coh_ext 0.034 vs 真实相干脉冲 coh_ext 0.142），raw 测度把 speckle 当事件。**改为 coherence 测度**：先空间平滑（长度 = 模型自身 E→E 尺度 `ELL_PAR`，不是任意旋钮）再阈值，散点 speckle 被平均掉、只有空间相干活动计数。这是"按模型自身定义（相干、定向、传播）才算事件"的修正，不是调参。`EVENT_ON_FRAC` 随之按**新测度的噪声地板 + margin** 重定（见下）。

**锁定阈值（数值固定，改动须改本文件 + 记 changelog）：**

| 量 | 符号 | 值 | 含义 |
|----|------|----|----|
| 相干尺度 | `coh_len` | `ELL_PAR`(0.54mm) | 空间平滑长度 = 模型 E→E 尺度；散点 speckle 被压、相干 patch 留存 |
| 事件 ON 阈 | `EVENT_ON_FRAC` | 0.05 | coherence 活跃比例 > 5% 算"事件进行中"；设在 σ=2.0 speckle 的 coh-floor(0.034) **之上**、真实相干脉冲 coh 峰(0.142) **之下** = 噪声地板+margin（**不是**调到出事件）；S1.0 已端到端验证 σ=2.0 noise → extinction |
| 最短时长 | `MIN_DUR_MS` | 8 | ON 段须 ≥8ms 才算事件（排除单/双帧噪声尖峰；远低于 ~70ms 真事件） |
| 合并间隙 | `MERGE_GAP_MS` | 12 | 两 ON 段间隔 <12ms 视为同一事件闪烁，合并 |
| 自终止比 | `RETURN_FRAC` | 0.2 | 事件 off 后 ext ≤ 0.2×峰值 = 已回落（沿用 step0b） |
| 回低态阈 | `CAPTURE_FRAC` | 0.5 | （窗口 B）末态/事件后稳定窗均值 rE 距低态 < 0.5×(nuE_high−nuE_low) = 回到低 root；否则 captured_high |
| runaway 比 | `RUNAWAY_FRAC` | 0.5 | max_ext ≥0.5 且未回落 = runaway（沿用 step0b） |
| 持续阈 | `SUSTAINED_MS` | 400 | 单 ON 段持续 >400ms 未回落 = 持续振荡/平台（真事件 ~70–150ms） |
| 稳定窗 | `SETTLE_MS` | 50 | 事件 off 后取 50ms 窗判回低态 |

**每次 run 的 regime 标签（优先级从上到下）：**

1. 某 ON 段 `dur > SUSTAINED_MS` 且未回落 → **`sustained`**
2. `max_ext ≥ RUNAWAY_FRAC` 且末态未回落 → **`runaway`**
3. （窗口 B）末态稳定均值距高 root 内 → **`captured_high`**
4. 合并后自终止事件数 `n_events == 0` → **`extinction_only`**
5. `frac_time_on ≥ FRAC_TIME_ON_MAX`(0.30) → **`sustained`**（连续活动/闪烁，**非时间离散**；advisor 2026-06-03 加固：离散事件必须时间上分开、大部分时间静息）
6. 否则（`n_events ≥ 1`，全部自终止、时间分开、无 runaway/sustained/capture）→ **`discrete_events`** ← **目标**

> 时间分离阈 `FRAC_TIME_ON_MAX=0.30`：活跃时间占比 ≥30% = 连续活动，不算"一颗一颗分开的事件"。这是 advisor 命名的判别项（"coherent AND temporally-separated AND self-terminating"）。

**计数单位**：一颗"discrete self-terminating event" = 合并后、dur≥MIN_DUR、回落到 ≤RETURN_FRAC×峰、（B）且回低 root 的 ON 段。这是 fraction（风险#1）与 rate（风险#6）的唯一计数对象。

---

## 2. 噪声模型（OU 外驱涨落）

外部驱动的时空涨落，加在 μE 上：`muE += noise_field(t)`。

- **时间**：每像素 Ornstein–Uhlenbeck，相关时间 `tau_noise`（**默认 100 ms = 慢 afferent；2026-06-03 改：原 5ms≈TAU_AMPA 会双重计数 σ 里的快涨落 → 连续点火 confound，见 §9.1，不可用**），稳态 std = `sigma_noise`（mV，扫描旋钮）。验证带 50–200 ms。
- **空间**：白噪声经高斯平滑，相关长度 `ell_noise`（默认 0.5 mm，与 E→E 核同量级）。
- **种子**：每 cell 一个整数 seed，可复现。
- **实现**：有状态闭包（OU 内部态逐步推进），由 integrate 的 `stim_fn(t)` 按序查询；要求 integrate 单调按步调用（成立）。
- **TDD 锁**：`sigma_noise=0` → 与确定性 run 逐位一致；同 seed 两次调用结果一致；不同 seed 不同。

> 注：突触抑制时间尺度等不可锚（不可锚清单见文末附录 A）；OU 参数 tau_noise/ell_noise 是建模假设，作 sensitivity，不声称数据锚定。

---

## 3. seed × amp 网格 + 比例报告（风险 #1：不准挑幸运 seed）

对每个**候选窗** × 每个 `sigma_noise` × 每个 `seed` 跑一次，给 regime 标签。

- **窗口 A（主线 PRIMARY）**：`w_ee_mult=1.0`，`b_a=0`（recovery off），单一稳定 root（Brunel-like）。
- **窗口 B（敏感性 SENSITIVITY）**：`w_ee_mult=1.4`（双稳，`n_clean_roots=2`），`b_a≈2000, tau_a≈25ms`（recovery on，Pinto–Ermentrout）。
- **窗口 B-ablation（失败对照）**：B 但 `b_a=0` → 预期 `captured_high`/`sustained`（证明 recovery 必需）。

**报告**：对每个 (窗口, sigma_noise) 报 **discrete_events regime 的 seed 比例**（= n_discrete_seeds / n_seeds），以及各 regime 的计数分布。**报全网格，不报"存在某 seed 出事件"**。

首轮规模（本轮）：N=64, L=16, T_RUN=5000ms；A：sigma_noise ∈ {1,1.5,2,2.5,3,4} mV × seeds {0..4} = 30 sims（~12min）。B 随后同规模。

---

## 4. 触发率曲线（风险 #6：曲线不调点；事件率只作量级校准）

噪声幅度本身是自由旋钮——**不准把它调到真实事件率再说"率支持模型"**（循环论证）。正确做法：

- 画 **event_rate (events/s) vs sigma_noise**，把 `extinction`（低端）/ `sustained`+`runaway`+`captured`（高端）区段**阴影标出**。
- 验收看：是否存在一段**宽**的 sigma_noise 区间（≥2 个网格点、理想 ≥~1.5× 幅度跨度），其中 **≥60% seed 是 discrete_events** 且 event_rate 落在数据量级内。
- **数据率锚**（量级，源 exploration 2；完整出处表见文末附录 A）：事件包络 ~100–300ms、通道滞后展布 ~50–178ms、IEI median ~3.3s（floor 0.18s）、rate ~209/h≈0.06/s（IEI 反推 ~0.3/s）→ 取量级带 **[0.01, 1] /s**。
- 报**区间宽度**，不报单点。窄到只有一个点 = 调出来的，不算通过。

---

## 5. 方向读出：先场、后真实 pipeline（风险 #4：不让 pipeline 当唯一闸门）

1. **场层先确认**：在 discrete_events regime，聚合各事件峰值场，量 active 区主轴（covariance 特征向量，同 0d）随 θ_EE 走（err<20°、ratio>1.3）。**事件数须 ≥ `MIN_EVENTS_FOR_READOUT=10`** 才进下一步。
2. **达标后**才把合成 SEEG（沿电极杆采样通道 → 各事件通道激活顺序）喂进**真实 masked PR-2/PR-6 模板 pipeline**（`--masked-features`，遵 AGENTS.md runner 纪律）还原方向。
3. 事件太少时**禁止**强行聚类（会制造假稳定）；记 "insufficient events for pipeline readout"，不出方向结论。

---

## 6. 噪声下的负对照（风险 #5：硬合同进 noisy readout，不只静态 0d tripwire）

承重判别指标（v2 plan 2026-06-02 amendment）：**方向随连接各向异性轴转、不随电极杆转**。Step 1 在**噪声驱动事件**上重跑此对照：

- (a) **isotropic E→E**（ell_par=ell_perp）→ 噪声事件**无优势轴**（ratio<1.3）。
- (b) **isotropic E→E + 电极杆对齐到固定方向** → **必须过不了**：不得产生跟电极杆走的方向模板（方向只能来自连接，不能来自杆几何）。
- (c) anisotropic E→E：θ_prop 随 θ_EE 转、随杆旋转不变。

(b) 是最强对照——线性电极杆两端成核会给出"正反反向模板"假象；必须证明在我们的 readout 里它过不了。

---

## 7. 验收（mechanism-scale，两窗分开报）

> **⚠️ 2026-06-04 amendment（见 §10.1，承重）**：本节 5 条**追加前置 (0) + (7')**——(0) 必须先按 §10.3「模型自身相结构窗形」验证工作点是间期工作点（不照搬 SNN 名义 ratio / 不用 E 率对齐）；(7') 离散事件区须是 (drive×σ) 二维稳健区域（§10.4），非单 drive 一维 σ 切片。原 5 条**在验证过的间期工作点上**才算数。

**窗口 A（PRIMARY）通过** 当且仅当（全部）：
- 无噪声（sigma=0）/ 亚阈噪声 → 无自发持续活动（`extinction_only`，ext 长期 <EVENT_ON_FRAC）；
- 存在**宽** sigma 区间，≥60% seed = `discrete_events`（非 sustained/runaway）；
- 事件自终止；
- 场层主轴随 θ_EE（+ 若事件足够，真实 pipeline 也还原方向）；
- 该区间 event_rate 落 [0.01,1]/s 量级；
- 负对照 (a)(b) 过不了、(c) 过。

**窗口 B（SENSITIVITY）通过**：A 全部 + **recovery-essential**（B-ablation `b_a=0` → `captured_high`/`sustained`，即关掉回拉就被高 root 俘获）+ 事件后回低 root + 包络更长（~110–300ms 量级）。

**纪律**：
- 检测器阈值锁死在 §1，结果不许回头调阈。
- 报全网格比例（§3）+ 触发率曲线区间宽度（§4），不报幸运 seed / 调出的单点。
- 窗口 B 的 capture 控制**不得**反过来绑架主结论（A 是主线）。
- 诚实 null 合法：若不存在"宽、不失控、不持续、率同量级"区间，就是真结果（写明，不抢救）。
- 预登记 tier：**A = primary mechanism-scale 演示；B = sensitivity（recovery-essential 分支）**。
- 遗留 speed-scale 张力（regime b reach ~10cm）作 Step 1/2 sensitivity 带入，不阻断 A。

---

## 8. 交付物

- `src/sef_hfo_events.py`：`make_ou_noise(...)` + `detect_events(ext, front, dt, op, ...)` + `classify_run(...)`（§1 阈值常量在此模块顶部）。
- `tests/test_sef_hfo_events.py`：检测器（单峰/双峰合并/持续/runaway/captured）+ 噪声（amp=0 恒等、seed 复现）TDD。
- `scripts/run_sef_hfo_step1_noise.py`：seed×amp 网格 runner（§3）→ `results/topic4_sef_hfo/step1_noise/` JSON+CSV。
- `scripts/plot_sef_hfo_step1_noise.py`：触发率曲线（§4）+ regime 热图 → `figures/` + `figures/README.md`（中文）。
- archive 续写本文件的 results 段；framework / INDEX / v2-plan 加 cross-link。

---

## 9. Results

### 9.1 Window A (recovery-off) smoke — 诚实 NULL（2026-06-03，coherence 检测，N=64, T_RUN=2000ms, 2 seeds）

| σ (mV) | seed 0 | seed 1 |
|--------|--------|--------|
| 1.5 | extinction | extinction |
| 2.0 | extinction | extinction |
| 2.5 | discrete (frac_on 0.26) | **sustained** (0.61) |
| 3.0 | sustained | sustained |
| 4.0 / 6.0 | sustained | sustained |

**⚠️ 修正（2026-06-03，advisor catch）：初读"window A 无离散窗、离散需 recovery"是错的，被自己的数据推翻 —— 不要把它当结论。** σ=2.5 **两个 seed 都有 `n_ev=11` 个自终止事件**（seed0 frac_on 0.26 标 discrete；seed1 frac_on 0.61 仅因越过 0.30 时间分离阈被标 sustained）。即 **recovery-off 的 window A 确实从噪声里点出了 ~11 个自终止事件**（与 Step-0b 单脉冲自终止一致，现在是噪声触发）。所以：

- **离散性不需要 recovery**（A 没 recovery 却出事件；B 有 recovery 却因被俘获失败——见 §9.2，两者失败原因不同，不可合并）。
- 问题不是"没有离散"，而是**事件太密**（frac_on 高、~5.5/s vs 数据 ~0.3/s），卡在时间分离阈两侧。
- **根因是噪声驱动被污染**：`tau_noise=5ms ≈ TAU_AMPA`——快突触涨落已被 Φ_LIF 的 σ 吸收，再加同速 per-pixel OU = **双重计数** → 持续的点火尝试流 → 任何能点着的 σ 都连续点着 → sustained。这是 OU 项用错了频段（应代表**慢、空间结构化**的输入：afferent burst/drift，几十–几百 ms），是**正确性问题不是调参**。

**所以当前 smoke 既不是 pass 也不是干净 null——是 confounded。** 决定性实验（§9.3）：window A 换**慢噪声**（tau_noise ~50–200ms），预登记假设 = 稀疏越阈 → 孤立自终止脉冲、frac_on 跨 seed 稳健 <0.30。一个变量、有物理依据、先声明。慢噪声若**仍**无稳健窗 = 强且可解释的 null；当前这个不可解释。

### 9.2 Window B (recovery-on) + B_ablate — 也 NULL（noise 把系统踢进高 root 被俘获）

roots(B, wee×1.4) = [0.339, 2.028] Hz（双稳：低静息 + 高饱和）。smoke 同上（N=64, 2000ms, 2 seeds）。

**B（recovery on，b_a=2000, τ_a=25）**：σ≤1 extinction；σ≥1.5 **全部 sustained 且 `captured_high=True`**（coh_max 0.6–0.8，frac_on 高）。无离散事件。
**B_ablate（recovery off）**：σ=0.5 extinction；σ≥1.0 **全部 sustained，`captured_high=True`，coh_max=1.000**（整场饱和到高 root）。

**B 失败的原因和 A 不同——不可合并、不作主结论。** B 是双稳（roots 0.34/2.03Hz），噪声不是"点着一个会熄灭的事件"，而是把系统**踢过 separatrix 落进高 root 盆地并留在那**（`captured_high=True`，advisor 预测命中："noise kicks the system into the high attractor"）。recovery 有可量化回拉（B σ≥1.5 才被俘获、coh 0.6–0.8；B_ablate σ≥1.0、coh=1.0）但**所测 b_a=2000/τ_a=25 不足以失稳那个高不动点**，故仍 captured。

**这是一个结构性发现（B 的高态是稳定吸引子，所测 recovery 不能让它变 transient），不是 Step-1 主结论。** B 的真问题"是否存在某 recovery regime 让高态从稳定吸引子变成瞬态"更深，单独立项；**不与 A 的噪声驱动 probe 合并扫**（慢噪声救不了 B——稀疏 kick 仍落进高盆地）。

**Step-1 主线（window A "噪声→离散事件"）的判定推迟到 §9.3 慢噪声 probe**——当前 fast-noise smoke 是 confounded（见 §9.1 修正），既非 pass 亦非干净 null。

### 9.3 Window A 慢噪声 probe（决定性，预登记假设）

**改 ONE 变量**：`tau_noise` 5ms → 慢（50–200ms），其余全锁（wee=1.0, recovery off, ell_noise=0.5, coherence 检测, N=64）。物理依据：Φ_LIF 的 σ 已含快突触涨落；OU 在 μ 上应代表**慢、afferent 级**输入，不是再加一遍快涨落。

**预登记假设（跑之前写死）**：慢噪声 → 稀疏越阈激发 → **孤立**自终止脉冲 → 存在一段 σ 使 `frac_on` 跨 seed 稳健 <0.30 且 `n_ev≥1` 自终止（= discrete）。**若慢噪声仍无稳健离散带 = 强且可解释的 null（"持续随机驱动无论快慢都给不出离散，离散需要别的结构"）。** 若出现稳健离散带 = window A primary 的 mechanism-scale pass 候选（再验方向/率/对照）。

**⚠️ 重大更正（2026-06-04，CRITICAL detector bug 修正后重跑）**：下表是**修正检测器（all-returned：run 内每个 ON 段都须自终止）+ σ=0 对照**重跑的结果。**之前"宽稳健带 σ∈[1.8,2.2]"是 detector bug 的高估**（旧检测器只要 ≥1 个自终止事件，忽略了同 run 里未回落的段）。

discrete_events 的 seed 比例（/5，修正后）：

| σ (mV) | τ=50 | τ=100 | τ=200 | （旧 buggy） |
|--------|------|-------|-------|------|
| 0.0（无噪对照）| 0.00 | 0.00 | 0.00 | — |
| 1.0 / 1.5 | 0.00 | 0.00 | 0.00 | 0 |
| **1.8** | **1.00** | **0.60** | **0.60** | (1.0/0.6/0.8) |
| 2.0 | 0.00 | 0.20 | 0.20 | (1.0/1.0/1.0) |
| 2.2 | 0.00 | 0.00 | 0.00 | (0.4/1.0/1.0) |
| 2.5 / 3.0 | 0.00 | 0.00 | 0.00 | 0 |

**离散带塌缩成窄窗 ≈σ=1.8**（≥60% seed 只在 σ=1.8 三 τ 都成立；σ=2.0 塌到 0–0.2、σ=2.2 全 0）。**σ=2.0 的塌缩是真的，不是 run 末截断 artifact** —— 诊断显示 σ=2.0 的未回落段**散布全程**（t≈540/1580/3917ms…，最后一个事件 t_off~4934ms 在窗内已回落），即驱动一强，事件频繁但**很多 excursion 不自终止**（中途不回落）→ 正确判为非离散。σ=0 / σ≤1.5 → extinction（**无噪不自发持续，已在网格**）。

**事件率**：σ=1.8 的 discrete seed rate ~0.3–0.5/s，落 [0.01,1]/s、贴 target ~0.3/s（rate-compatible）。

**朴素话 + 机制**：换成物理正确的慢输入后，window A（Brunel-like、无 recovery）**确实能从纯噪声点出离散自终止间期样事件，且就在点火阈附近（σ≈1.8）率与数据同量级**——但**窗很窄**：再加一点驱动（σ≥2.0）事件就频繁到很多不自终止、退化成 bursty/sustained。**离散性不需要 recovery**（§9.1 坐实）成立；但"宽稳健带"不成立。

**判定（修正后，诚实降级）**：window A primary —— **无噪不持续（σ=0/≤1.5 extinction）+ 噪声→离散自终止事件 + 该点率同量级** 在 **σ≈1.8 成立**；但**"宽带"判据（§4：≥2 网格点 / ≥1.5× 跨度）当前网格分辨率下 NOT MET**——干净离散只在 σ=1.8（σ=2.0 已非离散）。是**点火阈附近的窄窗**，不是宽稳健带。**更细 σ（τ=100, 5 seeds）定型**：frac_discrete = 0.0(σ1.6) → 0.2(1.7) → **0.6(1.8)** → 0.4(1.9) → 0.2(2.0) → 0(2.2)，rate 0.4–0.7/s 全程数据兼容。即**一个以 σ=1.8 为峰、两侧迅速衰减的窄峰；≥60% 只在单点 σ=1.8**。**§4 宽带判据（≥2 网格点 ≥60% / ≥1.5× 跨度）确定性 NOT MET ——是刀刃/窄峰，非宽带。** **所以 window A 离散性 = 「点火阈附近能出离散自终止事件、且那里率与数据同量级，但只在窄窗（σ≈1.8）稳健、宽带判据未达」，不是干净 PASS。** **仍待（承重，全部 OPEN）**：
- 方向读出（事件传播方向随 θ_EE，非电极杆几何）—— 现有测度 grid-contaminated，需逐事件隔离 + 0d 偏轴标定（§9.5）；
- 噪声下负对照 **(a) isotropic 无轴**（场层做过、聚合后对照 ratio 1.07 OK，但建立在 grid-contaminated 测度上，需随新测度重做）、**(b) isotropic+aligned-shaft 必须过不了 — 未实现**、**(c) 合成 SEEG → 真实 masked lagPat/PR-2 模板 pipeline — 未实现**。
**未做完这些前，window A 仅坐实"能从噪声出离散自终止事件、低边率同量级"，未坐实"方向由连接定"——不能写"window A 通过 Step 1"。**

### 9.4 Window B（结构性，单独）

B（wee×1.4 双稳）在所有 σ 要么 extinction 要么被高 root 俘获（`captured_high`，§9.2），所测 recovery（b_a=2000,τ_a=25）回拉但不失稳高不动点。B 的高态是否可变 transient 是更深的单独问题，不并入 window A 主线，不与 A 的慢噪声 probe 合并。**主线结论用 window A。**

> **B 测度局限（B 是 sensitivity，不挡 A；但以后认真报 B 必须补）**：`captured_high` 现在看**最终单帧**场均值 `rE_final.mean()`，不是**事件后/末态稳定窗均值**。最后一帧若卡在涨落中可能误判。当前 screen 可接受；正式报 B 前须升级为末态稳定窗均值（积分器需暴露末 ~SETTLE_MS 场平均）。已在 `classify_run` 代码内标 LIMITATION。

### 9.5 方向判别（S1.5/S1.6）—— 第一次测法错（单事件），换聚合测法

**第一次尝试（`scripts/run_sef_hfo_step1_direction.py` v1，σ=2.0 τ=100）= 失败，但测法违反了本合同 §5。** 我用**单个全局峰值场**（一次 run 一帧）测主轴：anisotropic axis_err 散乱（1–89°、不跟 θ_EE），isotropic 对照也出虚假轴（ratio 1.1–1.8）。原因：**噪声触发的单个事件，其形状被那一次触发涨落的随机不对称主导**——单事件主轴无意义。静态 0d 能 <0.1° 是因为只有一个干净确定性事件。合同 §5 本就要求**聚合 ≥10 事件**，我违反了自己的合同。

**修法 v2（按 §5）**：canonical `integrate_lif_field` 加 `axis_accum`——对所有"事件进行中"帧累加**居中二阶矩张量**（每帧减自身质心，活动加权），principal axis = 跨多事件系统拉长方向。结果（σ=2.0 τ=100, 3 seeds, T=5s, n_onframes~4万）：

| 条件 | θ_prop | axis_err | ratio |
|------|--------|----------|-------|
| aniso θ_EE=0 | 170.0 | 10.0° | 1.21 |
| aniso θ_EE=60 | 104.9 | 44.9° | 1.08 |
| aniso θ_EE=90 | 101.2 | 11.2° | 1.10 |
| **isotropic 对照** | 148.4 | — | **1.07** |

**部分进步、仍不达判据**：isotropic 对照现在**正确给出无轴（ratio 1.07）**——聚合修掉了 v1 的虚假轴。但 anisotropic 拉长**很弱**（ratio 1.08–1.21，低于 1.3 阈），角度只松散跟随（0°/90° 约 10° 内，60° 差 45°）。

**诊断（两层，指向换测度）**：(1) 近阈噪声事件**小**（coh_max ~0.07–0.12），瞬时形状几乎看不出核的 2:1 各向异性（0d 大确定性脉冲 ratio~4.2 是因为事件大、沿轴传播放大）；(2) 更根本——**瞬时形状二阶矩不是对的测度**。数据侧的方向信号 = 传播**模板/通道滞后顺序** = 事件**质心/波前随时间沿 θ_EE 扫**，不是某一帧的形状拉长。我测错了量。**对的测度 = 逐事件质心轨迹方向（波前位移）/ 合成通道激活顺序**（= 真实 pipeline 读出的东西，§5 step2）。

**advisor 复核（关键补漏）**：θ=60（偏离网格轴）差 45°、而 θ=0/90（贴网格轴）约 10° —— **过的两个都是网格轴对齐**，偏轴那个塌到各向同性水平。这说明聚合被**网格/边界结构主导、不是被核主导**；加上"整帧 active 区二阶矩把多事件的**位置间距**和单事件**内部拉长**混在一起" → v2 测度是 **artifact 主导**，换成轨迹测度但仍用整帧聚合会**同样失败**。根因 = **逐事件隔离 + 网格污染**，不是形状 vs 轨迹之争。

**正确下一步（窄、决定性、预登记；advisor 定）**：
1. **逐事件隔离**（时空连通域 / 单事件窗），测**每个事件自己的**方向，再聚合方向估计——无论用质心轨迹/波前位移/通道滞后都先做这步。
2. **先在 ground truth 上标定仪器，且用偏轴角**：0d 确定性脉冲（已知沿 θ_EE 传播、波前 ~7mm）在 **θ=30 与 60**（不是 0/90，网格 artifact 会假过）跑新测度——若连干净传播脉冲偏轴都还原不出 θ_EE，测度就是坏的，**在这儿修、绝不在噪声上修**（我对形状测度跳过了这步 = 出错根源）。
3. **预登记 + 设停止条件**：跑噪声前写死 pass/fail、两种结果都报。**若逐事件轨迹测度（标定好后）在噪声事件上也弱，那就是答案——不准发明测度 #4**；升级到真实数据仪器（合成 SEEG → 真实 lagPat/PR-2 模板，§5 step2）或把方向记为 OPEN。已换测度两次，第三次是最后一次仍属方法学，第四次是 fishing。

**诚实报弱形状结果**："近阈噪声事件瞬时形状各向异性很弱（ratio ~1.1）"可能是个真事实；方向主张则落在传播**顺序**上（= 数据用的量）。

**当前判定（分开写，不可合并）**：
- window A **离散事件存在**（离散/自终止/阈附近率同量级），**但宽带判据未达**（§9.3 修正：≥60% 只在 σ≈1.8 单点窄峰）——**不是干净 PASS**，独立于方向测度成立。
- **方向承重判别项 = OPEN / UNRESOLVED**（测度待按上面 1–3 重做）。**因此现在不能写"window A 通过 Step 1"**——方向是 SEF-HFO vs 几何 artifact 的承重判据，未决。方向阻断的是 window A 的**完整**主张，不阻断"离散事件存在（窄窗）"这个子结论。

### 9.6 Step 1 还差什么 + 收紧的下一步顺序（2026-06-04 user lock）

**Step 1 当前缺口（4 项，全未过）**：
1. **宽窗没过**：修正检测器后只有 σ≈1.8 单点 ≥60% seed（1.7/1.9/2.0 都掉）= 窄阈值峰，非预注册的稳健宽区间。
2. **方向承重判据没过**：尚未证明噪声事件传播方向由连接轴定、而非网格/电极几何假象。
3. **isotropic + aligned-shaft 硬负对照没做**：区分"真连接方向" vs "电极杆采样假象"的关键，未做。
4. **合成 SEEG → 真实 masked lagPat/PR-2 pipeline 没做**：仍停在场层事件，没走到数据侧"通道先后顺序"读出。

**收紧的执行顺序（按此序、不跳步）**：
1. **先做方向测度标定**：逐事件隔离 + 0d 偏轴 **θ=30/60** ground-truth 标定。**最先做**——若干净确定性脉冲都读不出方向，噪声上继续测 = 垃圾进垃圾出。（注：旧 0d 单事件**形状**主轴在 θ=30/60 已 PASS，commit c183ed6；但**新的逐事件隔离+聚合 pipeline 是新代码，必须在 ground truth 上重新验证**，不能直接信。）
2. **标定通过后再上噪声事件**：只用预注册的第三种测法；若仍弱 → 记 OPEN 或升级真实 pipeline，**不再发明第 4 种测度**。
3. **再做 isotropic + aligned-shaft + 真实 masked pipeline**：Step 1 最硬科学判据，不做不能说"方向由连接定"。
4. **最后讨论窄窗是否可接受**：这是**科学判据问题不是代码问题**。按原合同"宽区间防调参"当前**不合格**；若认为近阈窄窗机制上可接受，**必须显式改合同**——不能偷偷把失败改成通过。

### 9.7 工作点重定位 1.0→0.6（SNN N-scaling 强制）+ drive-grid 预登记（2026-06-04）

**这是 §9.6 第 4 条要求的"显式改合同"，不是偷偷把失败改成通过。** 依据是**独立先验证据**：coworker Zou 的 spiking ground-truth（`spiking_gt_validation_2026-06-03.md`）用 **N-scaling** 判定 **drive ratio=1.0 是真·确定性 Hopf / 自持振荡 = 趋发作态**（密度 ×4 振荡 prominence 不降、频率稳 26.7Hz），间期样安静可激静息在 **drive ≈0.6**（0.5 dead → 0.6–0.65 quiet → 0.70+ oscillation）。**我之前 §9.3 的 window-A 网格全跑在 ratio=1.0 = 趋发作点**——所以离散窗窄/塌缩是"在错的（趋发作）工作点测"的产物。**§9.3 的 1.0 结果保留为 SUPERSEDED（在 seizure-ward 点测得），不删除。** 速率字段 drive 结构定性吻合 SNN（mean_field：0.5 nuE~0.0001Hz dead、0.6 ~0.023Hz quiet、0.7 ~0.10Hz、1.0 ~0.22Hz）。

**预登记 drive-grid（跑之前写死，用户给的 bar）**：drive ∈ {0.5,0.6,0.7} × sigma_noise，修正检测器（all-returned）+ σ=0 对照，tau_noise 先 100ms smoke 再扩 50/200。**验收 = 在 drive≈0.6 是否出现 ≥2 个相邻 σ 点、≥60% seeds、且 rate 落 [0.01,1]/s 同量级。**
- 若 drive=0.6 让窗变宽（≥2 相邻 σ）= **有价值新证据**，离散性在正确工作点成立（宽窗判据达成）。
- 若 0.6 仍是单点窄峰 = **诚实结论"离散事件需要更精细的临界调节"**（reportable，不抢救，不算 PASS）。
- σ=0 / 亚阈 在 drive 0.6 必须仍 extinction（无噪不自发持续，新工作点也要守）。

**onset-front 方向测度（采用 SNN 原样）**：方向不用整事件形状/整帧聚合（已废，grid-contaminated），改用 SNN 验证过的 **onset-front**——踢后**最早 ~8ms（饱和前）**点亮像素的主轴，**逐事件**（不跨事件聚合）。先在 0d 确定性脉冲偏轴 θ=30/60 标定，再上 drive-0.6 噪声事件，带 θ 旋转 + isotropic 对照。**继承 SNN 的诚实 caveat**：isotropic 对照是近-fizzle（弱无向招募），精确表述 = "各向异性 E→E reach 是强传播 AND 定向前沿两者的**必要条件**"，**不**声称"从强各向同性波里干净读出方向"。

**drive-grid 结果（2026-06-04）—— 预登记假设 NOT SUPPORTED：relocation 不让离散窗变宽。** （仅报实测事实，不下架构结论——见末尾"不下的结论"。）
- **实测事实**：
  - rate-field 静息 loop gain（G_E·C_EE·W_EE）随 drive 单调上升、全程亚临界 (<1)：0.001(d0.5)→0.123(d0.6)→0.404(d0.7)→0.538(d0.8)→0.581(d1.0)。这是**静息线性增益**；事件期的**超阈增益**才是驱动传播的（所以 step0b 能点燃）。亚临界稳定正是"可激介质"的前提。（**注：rate field 全程稳定无 Hopf 是早已 ratified 的设计**——见顶部数学路线更正"稳健稳定但可激、非 near-critical、色散诊断/finite-pulse 闸门"，**不是这里新发现的**。）
  - drive=1.0：σ≈1.8 单点窄峰（≥60% seed，离散/自终止/rate ~0.3–0.5/s 同量级）—— rate field **确实产生过**离散定向率对的事件，只是窗窄。
  - drive=0.6（3 seeds, τ=100）：σ=0/0.5 extinction → σ=1.0 fragile discrete (1/3 seed, 0.6/s) → σ≥1.5 标为 sustained。σ=0 对照仍 extinction ✓。
- **⚠️ drive-0.6 "sustained" 未坐实（检测器 confound）**：检测器（DETECT 绝对 ~5Hz、FRAC_TIME_ON_MAX=0.30）是在 drive=1.0（rest 0.22Hz）标定的；drive=0.6 rest ~0.02Hz 近零 + 慢噪声 τ=100ms 可能把 coherence 测度在事件**之间**也顶在阈上 → frac_time_on≥0.3 触发"sustained" ≠ 真·网络自持。3 σ 点 × 3 seed 经异工作点检测器 = **太薄，不能据此说 0.6 不行**。**现在不重调检测器**（exhausted context 里第三次换测度 = 陷阱）；只是停止把"0.6 sustained"当已确立。
- **实测判定**：**rate field 在所测 drive 上，离散自终止事件只在窄近阈窗出现；预登记的"≥2 相邻 σ ≥60%"宽窗判据 NOT MET**（= 预登记 null 口径"离散需要更精细的临界调节"）。**到此为止——narrow，不是 can't。**

**不下的结论（不是我能定的）**：窄近阈窗**是否可接受**、以及"噪声自发离散 event-train"**是否应改到 spiking 底物**做——这是 §9.6 第 4 条我上一轮按 user 口述写下的**保留判断**（"不能偷偷把失败改成通过"）。把它扩写成"rate field 整体不忠实、降级路线"是同一个保留判断的更大版本，**同样不是我能 commit 的**。**上交 user 定，见 §9.8 fork。**

### 9.8 Step-1 当前真实状态 + 上交 user 的 fork（2026-06-04，autonomous push 在此停）

**Step 1 已确立的（durable）**：
- **离散性**：rate field 能从慢噪声自发点出**离散、自终止、率同量级（~0.3–0.5/s）的事件**，但只在**窄近阈窗**（drive 1.0 σ≈1.8；drive 0.6 更窄更脆）。预登记**宽窗判据 NOT MET**。= narrow，**不是 can't**。
- **方向**：0d 确定性脉冲形状主轴随 θ_EE（commit c183ed6）+ spiking GT onset-front 随连接轴转、isotropic 无轴（`spiking_gt_validation`）——**承重判据在 0d + spiking 两处 PASS**。
- **无噪不持续**：σ=0 / 亚阈 extinction（两 drive 都守）。
- **窗口 B**：双稳被高 root 俘获（结构性 sensitivity，单独）。

**未完成 / stalled（不是 failed）**：
- **噪声驱动事件的 onset-front 方向**：rate field 慢（τ_m=20ms），SNN 的 8ms onset 窗太早（n_active=0）；需逐事件隔离 + 适配 rate-field 时标（~20–40ms pre-saturation）。**stalled on 测度适配，不是模型错。**
- **isotropic+aligned-shaft 硬对照 + 合成 SEEG→真实 masked lagPat/PR-2 pipeline**：未做（Step-1 最硬判据）。

**Fork（user 定，autonomous 到此为止）**：
- **(a)** 接受"窄近阈窗"为 rate-field Step-1 离散性结果（**显式改合同**，把宽窗判据放宽或重定义）；方向继续用 spiking GT 已 PASS 的 onset-front 背书。
- **(b)** 把"噪声自发离散 event-train + 方向"测试**搬到 spiking 底物**（那里 Hopf/工作点结构忠实、onset-front 已验证）；rate field 保留为 Step-0 机制草图 + 线性稳定性工具。
- **(c)** 在 rate field 把 drive=0.6 做对——**先解检测器 confound**（DETECT 绝对值/FRAC_TIME_ON_MAX 在近零 rest 下重标定），再判 0.6 有无离散窗；**不是从当前 confounded 数据下结论**。

**我（agent）不替 user 选**；当前 durable 记录已是诚实的 Step-1 现状。

### 9.9 Step-1 expectation 的重新提问（user 2026-06-04 conceptual catch）—— 提案，不 enact

**user 的问题**：把"一次传播出去的事件"当一次间期事件，那么"宽/窄激发窗口"其实在探索的是不是"间期事件由 SOZ 内**多源离散**产生还是**单源**"？这是不是我们该研究的问题？对后续 step 的必要条件是什么？

**分析（advisor pressure-tested，措辞按"compatible-with 而非 evidence-for"收紧）**：
1. **数据已经约束了答案，不是模型去发现**：实测间期事件高度**刻板**——cohort `stable_k≈2`（一条主传播轴 + 其正/反向）、participation 中位 10、bias_fraction~0.71（固定骨架占主导）。即**模板多样性低（~1 轴 + 反向，不是一束）**。所以单源/多源在数据侧已基本回答 = **接近单一主源/主轴**；模型的任务是**匹配这个低多样性**，不是去发现它。
2. **宽/窄窗 ≠ 单源/多源（两条轴，别混）**：我量的"窄窗"是在**全局噪声幅度**这条轴上；源数目是**空间生成结构**那条轴。**窄窗 compatible-with 单源，但不是单源的 evidence**（也可能是同质 patch 被连接几何漏斗进同一模板、或检测器/工作点 artifact）。**判别量是模型产生的"模板多样性"，这个还没在模型里测。**
3. **真正的张力 + 真正该测的**：**同质 patch + 全局噪声**会让事件在等价的各点随机成核 → **多样模板** → 与数据（~2 模板）矛盾。**异质核（一个低阈值热点）→ 一致成核 → 低多样性 → 匹配数据**（framework 本就 lean "异质 patch + surround"）。**所以"窄窗失败"底下藏的真发现是：宽窗判据问错了问题；正确的 Step-1 验收应是"在间期工作点能否复现 stable_k≈2 的模板多样性 + IEI + 时长"，而同质 patch 很可能无论窗宽窄都过不了这个。** ——这是**可检验的断言**，不是结论。
4. **提案的验收判据替换（属 §9.6-item-4 user 保留判断，我只提案不 enact）**：把 (a) 的"宽 σ 带"**换成"在间期工作点复现 stable_k≈2 模板多样性 + IEI(~3s) + 时长(~100–300ms)"**作为科学上正确的 Step-1 离散性验收。
5. **新增 fork 选项 (d)（决定性实验，替代 σ-窗 hunt）**：**往 patch 里放一个异质核 → 噪声触发事件 → 数出现了几个不同模板 → 跟 ~2 比。** 这一刀直接测"单源 vs 多源"+"模型能否匹配数据的低多样性"，比扫噪声窗有意义得多。**（不在本轮 autonomous 跑——提案，等 user 选。）**

**与发作的关系 + 对后续 step 的必要条件（advisor 认为这是最 load-bearing 的一条）**：
- **间期 = 促临界但被稳态回拉约束住的核，被噪声偶发点出的自限瞬态；趋发作 = 同一个核，工作点沿 drive/gain 轴漂上去（回拉失守 → 持续招募）**。SNN 已给出这条轴（0.6 安静可激间期 / 1.0 自持振荡趋发作）。间期与发作是**同一基质的两个工作点**。
- **必要条件（锁死后续所有 step）**：模型必须把**间期与发作呈现为同一条 drive/gain 轴上的连续两段**（间期事件在转变点之下），即 **Step-1 的工作点和发作 step 的工作点必须是"同一个模型沿 drive 移动"，不是两套各自调参的模型**。否则发作桥接（H5/Phase 3）站不住。

### 9.10 Step-1 判定（division-of-labor 纠正，user 2026-06-04 ratify）—— PASS 判定已被 §9.11 降级，见 banner

> **⚠️ 2026-06-04 降级（user catch + advisor）**：本节"5 条满足 ⇒ PASS"已被 **§9.11 降级为 PENDING**——判据 2/4 是在**未被验证为间期工作点**的点上评出来的（双模型 drive 轴不可公度 + 仅一维 σ 切片 + 检测器单点标定）。**本节的 division-of-labor 纠正（方向=Step2、异质核/模板多样性=Step3、发作桥接=Step4–5、率场无 Hopf）仍成立**；只有"PASS"那句被推翻。PASS 表格保留作 audit trail。**权威口径见 §9.11 + §10 联合分析合同。**

回到 v2 plan §4 Step 1 的**本职 5 条验收**（**不是**我之前自创的"宽窗"门槛——plan 从未要求窗宽）：

| plan Step-1 验收 | 实测（σ≈1.8, τ=100, 修正检测器） |
|---|---|
| 1 无噪声不持续 | ✓ σ=0 / 亚阈 extinction |
| 2 加噪声出现离散群体事件 | ✓ discrete（all-returned 检测器确认） |
| 3 落自限区、不持续扩散 | ✓ 自终止、非 runaway |
| 4 率可调到数据量级 | ✓ ~0.3–0.5/s ≈ 实测 ~0.3/s |
| 5 时长/空间范围合理 | ✓ ~100ms（实测包络低端）、次全局 |

**5 条全满足 ⇒ Step 1 按 plan 本职 PASS**（在窄近阈窗 σ≈1.8；窄窗如实报 = 满足"报比例不报单点"纪律）。

**纠正（§9.3/9.5/9.8 中"window A 不通过 Step 1"是错位，以本节为准）**：那是把 **Step 2** 的承重判据和 **Step 3** 的内容错当成 Step-1 门槛。按 plan 分工：
- **Step 1**（PASS）：噪声→离散自限事件存在 + 工作点 + 率量级。
- **Step 2**（未做）：方向随连接轴转/不随电极杆转 + 全套几何负对照（isotropic+aligned-shaft 必须过不了、转杆、抖动…）+ k=2/identity-bias 描述。【0d 已做确定性脉冲方向；噪声版 onset-front + 硬对照 + 合成 SEEG→真实 masked pipeline 在此】
- **Step 3**（未做）：局部异质核 → source/端点与病理位置挂钩、移核 source 随动、转轴 forward/reverse 随转。【**单源 vs 多源 / 模板多样性匹配 ~2（§9.9 选项 d）在此**】
- **Step 4** = LIF SNN（coworker 已做：机制 + onset-front 方向 + 间期点 0.6 验过）；**Step 5** = 慢变量→发作桥接；**Step 6** = held-out 一致性印证。
- **发作轴连续性必要条件** 由 **SNN / Step 4–5** 承接：简化率场无自持振荡（已 ratified 设计），本就不负责发作桥接，只负责 Step 0/1 机制 + 线性稳定性。

**遗留 sensitivity（不阻断 Step 1，带入后续）**：时长 ~100ms 在实测包络低端；窄窗是否对应"低模板多样性单源"留 Step 3 异质核处判（§9.9 选项 d）。

### 9.11 §9.10 PASS → PENDING（2026-06-04 user catch：工作点未验证 + 双模型 drive 轴不可公度 + 仅一维 σ 切片）—— 本节为权威口径

**§9.10 的 division-of-labor 纠正仍然成立**（方向=Step 2、异质核/模板多样性=Step 3、发作桥接=Step 4–5、率场无 Hopf 只负责机制+线性稳定性）。**被推翻的只是"5 条 plan 验收已满足 ⇒ Step 1 PASS"这一句**——那 5 条（尤其判据 2「噪声→离散事件」、判据 4「率到数据量级」）是**在一个从未被验证为间期工作点的点上**评出来的。**验收门漏写了承重前置：「在一个被验证过的间期工作点上」**（= 我自己存的教训"acceptance gate 必须编码结论本身，不只编码现象存在"）。

user 2026-06-04 catch（advisor pressure-tested）三条，现有一维数据都排除不了：

1. **两个模型的 drive 轴不可公度（同名不同刻度）**：rate field 与 SNN 共用名义外驱比 ν_ext/ν_θ，但同一名义 drive 下决定点火/传播的 **E 群发放率 rate field 比 SNN 低 3–4×**（rate 场名义 0.6 ν_E~0.023Hz 近死寂；SNN 间期 0.6 是 0.10–0.26Hz）。**且这 3–4× 被 `spiking_gt_validation` 文档自己标为"大部分是比较假象"**（稳态 FP vs 振荡网络、低率凸尾 Jensen），不是干净传递读数 → 连"两轴差多少"都没干净标定。**§9.7 把 SNN 名义 0.6 直接搬给 rate 场，违反了同一份文档"按机制+窗形对、不按 ratio-for-ratio 对"的告诫。**
2. **(drive × σ) 联合空间从未扫过**——只在两条固定 drive（旧跑 1.0 写死 / 新跑 0.6 默认）上各扫了一条 σ 一维线（`run_sef_hfo_step1_noise.py`：drive 是命令行单值，σ 内层循环）。"窗很窄"是两条竖线上的结论，**不是那张面上的**；离散事件区是 (drive×σ) 上的二维楔形，固定 drive 上 σ 窄推不出楔形窄。
3. **0.6 上"脆弱/sustained"是未解混淆，两个互斥解释一维数据分不开**：
   - **甲（drive 太低）**：0.6 真接近死寂，该往上挪工作点。
   - **乙（检测器没重标）**：3–4× 率差既"大部分假象"，则 0.6 在两模型是同一对的工作点，"sustained" 是检测器（DETECT 绝对 ~5Hz / FRAC_TIME_ON_MAX 在 drive=1.0 标定）在近零静息下被慢噪声顶上阈的假象（§9.7 已标）。
   - **若率差真"大部分假象"，靠挪 drive 去补 = 重复计数**（把读数偏移当工作点偏移修）→ 文档自身逻辑偏向乙。**现有数据不能裁决，两者都 open。**（**不下"1.0 才是真间期点"这个 confirmation-bias 形状的结论**——advisor catch。）

**判定：Step 1 PASS → PENDING。** 不依赖工作点验证的 durable 子结论：**rate field 能在其参数空间某处（名义 drive=1.0, σ≈1.8）从慢噪声点出离散、自终止、率同量级（~0.3–0.5/s）的事件**——但"那处是不是间期工作点"未验证、(drive×σ) 面未扫、检测器单点标定。**解锁 Step-1 PASS 须先做 §10 联合分析。**

> **PENDING 已判定（2026-06-04，§11）**：(drive×σ) 联合分析 + 逐点重标检测器跑完 → **诚实 NULL**：同质 window-A 率场**无稳健离散区**（0 accepted 格、robust_2d_block=False，σ=0 每 drive extinction），离散只是 silent↔continuous(sustained) 间一道种子脆弱窄缝；旧"σ≈1.8 窄窗"经查部分是单点偏低阈高估（重标后 0.6→0.2）。**Step-1 的"噪声自发离散事件存在"在简化同质率场 = NULL（不是 PASS）= 稳健结论。** 但**下一步去向（Step 3 异质核 vs window-B recovery）取决于 sustained 的失败模式子类**（too_frequent vs non_returning/long_plateau），原始 run 未记录 reason → 由 `joint_A_failmode` 重跑坐实（§11 失败模式 bullet）。详见 §11。

---

## 10. (drive × σ) 联合分析 + 逐工作点检测器标定合同（pre-registration，2026-06-04 lock）

> **朴素话**：上一轮发现我们只在两个"猜来的"驱动档位上各扫了一条噪声强度线，而且检测器"算不算活动"那把尺子是在其中一个档位上刻的、搬到另一档会读错。这一节在动手前先锁死三件事：(1) 怎么用模型**自己的脾气**找到"间期那个驱动档位"（不照搬另一个模型的数字）；(2) 检测器那把尺子怎么**按每个驱动档位重新刻**、且不许刻成"刚好出事件"；(3) 然后在"驱动 × 噪声"这张**二维面**上系统扫，看离散事件区是一整块还是一条缝。开工前冻结，跑完对照本节。

> **范围**：本节处理 **(drive × σ) 标定轴**（噪声仍 homogeneous，user 2026-06-04 明确搁置异质性）。**不**处理模板多样性 / 单源-多源（= §9.9 选项 d / Step 3，正交的另一条轴）。本节是 §9.8 fork 选项 (c) 的**升级版**——光修 0.6 检测器分不开 §9.11 的甲/乙，必须连"工作点定位+逐点重标+二维面"一起做。

### 10.1 补回缺失的 Step-1 验收承重条件（§7 amendment）

§7 原 5 条验收**追加前置**（= §9.11 暴露的缺口）：
- **(0) 工作点必须先被验证为间期工作点**：判据 2/4（噪声→离散、率到量级）只有在**按 §10.3 用模型自身相结构定出的间期工作点**上成立才算数；任意猜来的名义 drive 上成立**不算**。
- **(7') 离散事件区必须是 (drive×σ) 上的稳健二维区域**，不是单 drive 上的一维 σ 切片；报二维 map（§10.4）的区域宽度/形状，不报单点。

### 10.2 逐工作点检测器标定合同（= user 要的"对检测器的合同"）

**问题**（§9.7 坐实）：§1 检测器把"算不算活动"的幅度尺（per-pixel bar = `op["nuE"]+DETECT`，DETECT=5Hz 绝对）+ 时间分离阈 FRAC_TIME_ON_MAX 都在 drive=1.0（rest 0.22Hz）刻好；搬到近零静息（drive 0.6 rest~0.02Hz）会因慢噪声顶阈而把事件之间也判 ON → 假 sustained。

**修法（pre-registered，逐工作点重刻「幅度尺」，shape/time 常数不动）**：

**(A) 每工作点重刻的只有「幅度尺」，且夹在两个无循环参照之间**：
- **下界 = 静默地板 floor**：该 drive 跑 **σ=0（确定性无噪）+ 指定亚阈参照 σ_ref**，量 coherence 活跃测度的**静默上包络**。**σ_ref 必须跨整条 drive 带都亚阈（不点火）**，否则 floor 被真事件污染=循环。**σ_ref=0.5（pre-registered）**：σ=1.0/1.5 在 drive 0.6 已点火（1D 实测），不能当那里 floor 参照；σ=0.5 在 0.6–1.0 全 extinction。（v1.3 初稿误写"网格最低噪声点"，已更正：最低非零网格点 1.0 在间期端会点火。）
- **上界 = 真事件幅度 event-peak**：该 drive 跑 **step0b 确定性有限脉冲**（已知会点出自限事件），量其 coherence **峰值**。
- **ON 幅度尺锁在 (floor, event-peak) 之间**（pre-registered 比例，如几何均值；并验证 `floor < bar < event-peak`，否则该工作点判 **`undetectable` loud-fail**——不偷偷过、也即该 drive 非可激/非间期候选）。
- **反循环**：幅度尺**只**由 (i) 无噪/亚阈参照 + (ii) 确定性脉冲事件参照算出，**绝不**由噪声驱动事件网格回算；每工作点冻结后才跑噪声网格。

**(B) 工作点不变量（跨 drive 锁死、不重刻）**：`MIN_DUR_MS=8`、`MERGE_GAP_MS=12`、`RETURN_FRAC=0.2`（相对自身峰）、`SUSTAINED_MS=400`、`SETTLE_MS=50`、`FRAC_TIME_ON_MAX=0.30`、all-returned 规则、`RUNAWAY_FRAC`/`CAPTURE_FRAC`（相对 roots/峰）—— 无量纲/时间常数，重刻幅度尺后即可跨工作点转移。

**(C) 不变量回归测试（每工作点重刻后必过；= 反调参闸门）**：
- σ=0 → extinction（**每个 drive 都必须**；若重刻让 σ=0 出"事件" = 重刻坏了）。
- σ_ref（亚阈）→ extinction。
- 确定性脉冲 → 仍判 step0b 的 discrete/self-limited（确认幅度尺没把真事件刷掉）。

### 10.3 间期工作点定位：按模型自身相结构「窗形」，不按名义 ratio / 不按 E 率

- **禁止**：照搬 SNN 名义 0.6（轴不可公度，§9.11-1）；用 E 率对齐（E 率差被标"大部分假象"，用它对齐 = §9.11-3 的重复计数陷阱）。
- **做法**：用 rate field **自己**的 drive 轴相结构（dead：脉冲 fizzle → excitable：脉冲→自限事件 → bursty/captured/elevated-rest），**按窗形**对 SNN 的 drive 轴结构（0.5 dead → 0.6–0.65 quiet-excitable → 0.7 osc-onset）。**间期候选 drive 带 = rate field 的 step0b 有限脉冲自限传播窗**（excitable 且 sub-runaway；sub-Hopf 率场恒成立）—— 这是模型**自身**的可激结构，与名义数字无关；落在 0.6/0.7/1.0 哪都行、由结构定，不预设。
- **与 §10.2 自洽**：dead drive 上 kick fizzle → 无 event-peak → §10.2(A) 判 `undetectable` → 自动排除出间期带。即 §10.2 标定本身就实现了"按可激性定位"。

### 10.4 (drive × σ) 二维联合扫 + 区域验收

- **网格**：drive ∈ {§10.3 的 step0b excitable 带，≥3–5 点} × σ ∈ {含 0 与 σ_ref}，每格 ≥5 seeds，τ_noise 先 100ms（再 50/200 sensitivity）；每格用 §10.2 逐点重刻的检测器 → regime 标签。
- **报**：discrete-events seed 比例的**二维 heatmap**（frac_discrete over drive×σ）+ 每格 rate/IEI/包络。
- **验收（取代单点 σ 判据）**：存在一块**稳健二维区域**（非单点/单缝），其中 ≥60% seeds = discrete、rate ∈ [0.01,1]/s、IEI/包络兼容。报**区域面积/形状**。
- **σ=0 在每个 drive 必须 extinction**（无噪不自发持续，跨 drive 守）。
- **诚实 null 合法**：正确定位工作点 + 逐点重刻检测器后，离散事件**仍只是细缝/单点** → 真结果（"离散性需要更精细临界调节 / 需异质核"，指向 §9.9 选项 d / Step 3），写明、不抢救。

### 10.5 交付物 + 锁

- `src/sef_hfo_events.py`：检测器幅度尺改为按 op 标定（`calibrate_detector(op, ref_runs, kick_run) → bar`，§10.2 A）；shape/time 常数保留为模块顶常量（§10.2 B）。
- `tests/`：§10.2 C 三条回归（σ=0/σ_ref→extinction 每 drive；kick→discrete）+ 反循环（bar 不依赖噪声网格）TDD（先红后绿，§6 deep-contract-verify）。
- `scripts/`：drive×σ 二维 runner（§10.4）→ `results/topic4_sef_hfo/step1_noise/` 二维 JSON+CSV；heatmap plot + `figures/README.md`（中文）。
- 跑完续写 §11 results；§9.11 PENDING 据 §11 二维结果再判 PASS / 诚实 null。

---

## 11. (drive × σ) 联合分析结果（2026-06-04）—— 诚实 NULL：同质 window-A 率场无稳健离散区

> **朴素话**：把"撤掉手指、只给随机抖动"在"驱动强弱 × 抖动强弱"二维面上系统扫了一遍（驱动 0.5–1.3 × σ 9 档 × 5 种子），每个驱动档位都用"无噪静默 + 手动戳一下的真事件"两端重刻了检测器的尺。结果：**在同质组织、不开恢复机制（window A）这设定下，找不到一整块"随机抖动稳定自发点出一颗颗分开、自己灭、节奏对的事件"的区域**——只有零星几个、换种子就没的格子。这是 §10.4 预登记的诚实 null 分支。

**机器判定（runner `region_summary`，不靠人眼看图）**：
- **accepted 格子 = 0**（验收 = ≥60% 种子离散 且 离散事件率 ∈ [0.01,1]/s）；**robust_2d_block = False**。
- **σ=0 在每个驱动档位都熄灭** ✓（无噪不自发持续，新坐标系也守）。
- **drive 1.3 被脉冲检验判非可激**（kick=local_bump 非 self_limited_propagation）自动剔除——§10.2 C 脉冲门生效。

**结构（图 b 是要害，`results/topic4_sef_hfo/step1_noise/figures/step1_joint_drive_sigma.png`）**：
- **低驱动 0.5–0.7**：σ 一够强直接进 continuous（活动太频繁连成片，noise max_ext 中位 ~0.48 贴近 runaway），**无离散带**（silent 直接跳 too-frequent）。
- **高驱动 1.0–1.1**：σ≈1.8–2.2 出现一道**极窄、换种子就塌**的离散缝（frac 峰值仅 0.4 = 2/5 种子）；σ 更低=silent，更高=continuous。
- **失败模式（待坐实，证据不足，重跑中）——user review #1 catch**：原始 run 只存了 label/n_events/max_ext/frac_time_on/rate，**没存 all_returned/longest_on_ms/reason**，无法把 `sustained` 拆成 long_plateau(rule1)/too_frequent(rule5)/non_returning(rule6)。已记录的（n_events 3–8/run >0、frac_time_on 0.32–0.80）**compatible-with"太频繁"**，但 **n_events>0 不能排除**同一 run 里还有未回落段(rule6)或长平台(rule1)——即不能据此断定"事件都自终止"。**故先前"失败=太频繁、所以不需要 window B"是欠证据推断，降为待定。** 已 (1) 给 `classify_run` 加 `reason` + runner 记 reason/all_returned/longest_on_ms/n_events_total（commit 9f46312），(2) 跑 drives 0.6/1.0 的 failure-mode 重跑（`joint_A_failmode`，reason 记录）坐实。**坐实前 window B（recovery-on）保持 OPEN、不排除。**

**与旧 1D 的关系**：旧 1D（固定阈 0.05）drive 1.0 σ=1.8 报 frac 0.6；逐点重标后阈升到 0.071，同格**掉到 0.2** → 旧"0.6 窄窗"部分是**单点偏低阈的高估**。逐点重标 + 多种子地板 + 脉冲验证后，离散信号比旧分析更弱更碎。

**口径限定（不可外推）**：限于 **window A（recovery off, w_ee_mult=1.0）+ 同质噪声 + 中点阈（frac=0.5）**，**不是"率场整体不行"**。
**Caveats**：
- (a) **阈值比例敏感**：边界 σ（drive 0.9–1.1, σ≈1.8–2.0）的 extinction 格子 max_ext 贴在阈下方（gap 0.000–0.007）→ discrete/extinction 边界对中点阈位置敏感；未做 frac<0.5 阈敏感性（序列未缓存，需重跑）。但 no-region 对**验收阈值**稳健（advisor：frac≥0.4 也无 2×2 块），因绑定约束是**种子脆弱性 ≤2/5**（独立于阈值）。
- (b) **n=5 种子**：accepted(3/5) vs fragile(2/5) 只差一个种子 → 脆弱性主张有 power 限制。
- (c) **window B（recovery-on）未二维扫 + 失败模式未坐实（见上）→ recovery 是否是缺的那一环 = OPEN**：待 `joint_A_failmode` 重跑的 reason 分布判——若 sustained 主要 too_frequent → recovery 帮助有限、指向 Step3/Step5；若含大量 non_returning/long_plateau → window B 必须二维扫才能下"同质率场不行"的结论。framework 原 window B 是 wee×1.4 双稳（旧 smoke→captured_high），与单稳 window-A substrate 上加 recovery 不同，需另定义。

**指向**：同质 window-A 率场给不出稳健噪声自发离散区 → 需 **异质核（低阈值热点 → 一致成核、空间受限、抑制全局再点火）和/或结构化/慢输入**（= §9.9 选项 d / Step 3），和/或把噪声自发 event-train 搬 **spiking 底物**（§9.8 fork b，工作点结构忠实 + onset-front 已验证）。

**§9.11 PENDING 的判定（据本节）**：Step-1 "噪声→离散自限事件存在 + 工作点" 在**同质 window-A 率场 = 诚实 NULL（不是 PASS）= 稳健**。离散事件只是 silent↔continuous(sustained) 间一道种子脆弱的窄缝，无稳健二维区。**去向待定**：若失败模式坐实为 too_frequent（事件都自终止、只是太密）→ recovery 帮助有限，存在性**移交 Step 3（异质核）/ Step 4（spiking 底物）**；若含 non_returning/long_plateau → 须先把 **window-B（recovery-on）二维扫**才能下"同质率场不行"。NULL 与数据本身要"低模板多样性 / 近单源"（§9.9）一致——同质 patch 本就被预期可能过不了，但"为什么过不了"由 failmode 重跑定。**代码**：`scripts/run_sef_hfo_step1_joint.py`（runner）+ `src/sef_hfo_events.py::{event_on_frac_from_refs,calibrate_detector,accepted_cell}`（TDD）+ `scripts/plot_sef_hfo_step1_joint.py`（图）。

---

## 附录 A：数据锚定出处表（原 exploration 2，2026-06-03 整理合入）

> 此表 + 不可锚清单原在 `step1_unlock_feasibility_2026-06-03.md`（已于 2026-06-03 整理删除，git commit `7addc71` 保留）。§2 / §4 引用的"源 exploration 2"出处映射在此。原 Exploration 1（率层色散，commit `eedd1ce`）+ Exploration 2（Topic 1/2 artifact archaeology）的机制结论已分别由 `lif_rate_field_theory_2026-06-03.md` §9（慢抑制相位滞后）与 `lif_transfer_route_2026-06-03.md` 吸收。

| 量 | 数据值 | 锚到模型的哪个 | 出处 |
|---|---|---|---|
| 事件包络时长 | ~100–300ms | 振铃时长 / 离临界距离（recovery 时间常数若开） | `pr4b_lag_validation` + 检测器 50–200ms |
| 通道激活展布 | 中位 178ms（典型 ~50–150ms） | 传播速度（延迟 + 核 + 网格） | `pr4b_lag_validation::relative_lag_max` |
| 网格尺度 L | cm 级（= SOZ 电极覆盖范围，不是 1–2mm） | 空间域大小 | 见下注 2 |
| 不应期 / dead-time | ~0.18–0.5s | 触发不应期 | `iei_fit.iei_min` |
| 事件率 | ~209 次/h（基线） | 噪声 + 亚临界设定的触发率 | `event_periodicity::n_events/dur` |
| 慢漂移 | 数小时，调制率 + 参与数（xcorr 0.74） | Step-5 慢变量 S(t) | `event_periodicity_analysis §5.6/5.8` |
| 参与通道数 | 中位 10（IQR 7–16） | 事件空间范围 | `pr1_subject_summary` |
| 结构 vs 动态占比 | bias_fraction ~0.71 | 模板"固定骨架"占主导 | `pr1_cohort_summary::bias_fraction_median` |

**不可锚清单（数据测不出、必须扫或假设的）**：

- E-I 比、离阈距离、绝对突触权重、单通道基线发放率 —— 数据链里没有，必须扫（= framework 要求的 operating-point family）。
- 突触抑制衰减时间常数本身 —— exploration 2 测不到；"慢抑制"只能作 Brunel 锚定假设带进去 + 扫一个范围，不能说成"数据已确认"。

**注**：

1. **慢抑制是假设不是测量** —— 数据**不反对**（事件级时间尺度兼容），但也**没法直接确认**它；带着扫。
2. **空间尺度用电极覆盖范围（cm）**，不能照搬 coworker1 的 1–2mm 小网格 —— 1–2mm 网格上 Brunel 速度给 ~38ms 传播，比实测 ~50–178ms 快；换 cm 级网格（真实电极覆盖）后 ~33–100ms 对上。L 和传导速度要**联合锚**。

## Changelog
- 2026-06-03 v1.0：开工前冻结（用户 6 条加固落数值）。
- 2026-06-03 v1.1：**coherence 测度 amendment**（开 smoke 验证噪声地板时触发 + advisor）。raw 每像素活跃比例被 OU speckle 污染（σ=2.0 raw_ext 0.048 vs coh_ext 0.034 vs 真脉冲 coh 0.142）→ 检测改用 coherence 活跃比例（空间平滑 `coh_len=ELL_PAR` 后阈值），`EVENT_ON_FRAC` 0.01→0.05（按新测度噪声地板+margin 重定），新增时间分离判别 `FRAC_TIME_ON_MAX=0.30`（离散事件必须时间分开）。`coh_len` 作为可选项加进 canonical `integrate_lif_field`（commit 见下）。端到端 TDD：σ=2.0 noise → extinction（speckle 被拒）。**这是先验证再冻结的修正，不是看结果调参——新阈值不为 window A 制造离散窗（A 的诚实结局可能就是无离散窗 = 离散需要 recovery）。**
- 2026-06-03 v1.2：**文档整理**——把原 `step1_unlock_feasibility_2026-06-03.md` 的 Exploration-2 数据锚定出处表 + 不可锚清单合入本文件附录 A（§2/§4 的"源 exploration 2"引用改指附录 A），随后删除该中间探索文件（git commit `7addc71` 保留）。同批删除 `exploration1_rate_lif_dispersion_2026-06-03.md`（结论被 fsolve 更正推翻；commit `eedd1ce`）与 `datalocked_step0b_exploration_2026-06-03.md`（sigmoid 失败已由 `lif_rate_field_theory` §3 自洽复述；commit `58ed445`）。
- 2026-06-04 v1.3：**§9.10 PASS → PENDING（§9.11）+ §10 (drive×σ) 联合分析 + 逐工作点检测器标定合同 + §7 gate amendment (0)/(7')**。user 2026-06-04 catch（advisor pressure-tested）：§9.10 的"5 条 plan 满足 ⇒ PASS"是在**未验证为间期工作点**的点上评的——双模型 drive 轴不可公度（E 率 3–4× 被标"大部分假象"，§9.7 照搬名义 0.6 违反"按窗形不按 ratio"告诫）+ (drive×σ) 面从未扫（只两条一维 σ 切片）+ 检测器在 drive=1.0 单点标定。§9.10 的 division-of-labor 纠正仍成立，仅 PASS 降级。§10 锁死：检测器幅度尺逐工作点夹在 (σ=0/σ_ref floor, 确定性脉冲 event-peak) 之间、反循环、shape/time 常数不变、σ=0 每 drive 必 extinction 回归；工作点按模型自身相结构窗形定位（非名义 ratio/非 E 率）；二维区域验收取代单点 σ 判据。
