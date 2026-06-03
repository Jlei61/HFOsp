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
