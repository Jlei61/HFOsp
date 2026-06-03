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

- **时间**：每像素 Ornstein–Uhlenbeck，相关时间 `tau_noise`（默认 5 ms），稳态 std = `sigma_noise`（mV，扫描旋钮）。
- **空间**：白噪声经高斯平滑，相关长度 `ell_noise`（默认 0.5 mm，与 E→E 核同量级）。
- **种子**：每 cell 一个整数 seed，可复现。
- **实现**：有状态闭包（OU 内部态逐步推进），由 integrate 的 `stim_fn(t)` 按序查询；要求 integrate 单调按步调用（成立）。
- **TDD 锁**：`sigma_noise=0` → 与确定性 run 逐位一致；同 seed 两次调用结果一致；不同 seed 不同。

> 注：突触抑制时间尺度等不可锚（见 exploration 2）；OU 参数 tau_noise/ell_noise 是建模假设，作 sensitivity，不声称数据锚定。

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
- **数据率锚**（量级，源 exploration 2）：事件包络 ~100–300ms、通道滞后展布 ~50–178ms、IEI median ~3.3s（floor 0.18s）、rate ~209/h≈0.06/s（IEI 反推 ~0.3/s）→ 取量级带 **[0.01, 1] /s**。
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

## Changelog
- 2026-06-03 v1.0：开工前冻结（用户 6 条加固落数值）。
- 2026-06-03 v1.1：**coherence 测度 amendment**（开 smoke 验证噪声地板时触发 + advisor）。raw 每像素活跃比例被 OU speckle 污染（σ=2.0 raw_ext 0.048 vs coh_ext 0.034 vs 真脉冲 coh 0.142）→ 检测改用 coherence 活跃比例（空间平滑 `coh_len=ELL_PAR` 后阈值），`EVENT_ON_FRAC` 0.01→0.05（按新测度噪声地板+margin 重定），新增时间分离判别 `FRAC_TIME_ON_MAX=0.30`（离散事件必须时间分开）。`coh_len` 作为可选项加进 canonical `integrate_lif_field`（commit 见下）。端到端 TDD：σ=2.0 noise → extinction（speckle 被拒）。**这是先验证再冻结的修正，不是看结果调参——新阈值不为 window A 制造离散窗（A 的诚实结局可能就是无离散窗 = 离散需要 recovery）。**
