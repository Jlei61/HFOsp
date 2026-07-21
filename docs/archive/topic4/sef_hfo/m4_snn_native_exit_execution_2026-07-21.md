# SNN-native M4 containment-to-exit lifecycle — execution log (2026-07-21)

> **Status: 开环+对称退不出；不对称 slow-release = 待确认候选（GO 窄确认）.** Stage 0–2 done; P0（`d_sweep` 漏传 `tau_p_down`）
> 已修 + 真·不对称重跑（`arms_asymfix`）。当前科学标签=**slow-release suppression–rebound bursting candidate**，非确证可恢复 lifecycle。
> 最大风险：候选参数（`p50=0.15`, η80/150）可能压制普通 IED（prevention）→ 正跑候选参数匹配控制 `arms_prevctl_eta*`。Verdict §8。
> Mechanism SCREEN, not a seizure claim. Branch `codex/topic4-m4-snn-native-exit`, base `4d40b03`.
> Spec: `docs/superpowers/specs/2026-07-21-topic4-m4-snn-native-exit-design.md`.

## 摘要（朴素话，CLAUDE.md §8）

**测了什么** —— 有一个二维兴奋/抑制脉冲网络，它已经会两件事：(1) 背景噪声下自发点出一串**短促、会自己停**的间期事件；
(2) 当"抑制油箱" `q_I` 慢慢漏空、再叠一个"除法式全局抑制池" `S_G` 兜住时，掉进一个**不失控但也不结束**的持续放电态。
我们要问：能不能给它加一个**只在持续态建立后才慢慢积累**的局部恢复电流，让这个持续态自己停下来、回到能重新产生间期事件的状态——
全程不用人工 reset。

**怎么测** —— 先不建全套动态机制，先用最省的**开环探针**：把持续态里的发放用"抬阈值"硬压住一段时间再松手，
看压多久之后它才不弹回去。然后才建**真正的动态机制**（一个持续时间门控的局部场 `p(x,t)`，只在活动持续够久的地方积累，
积累起来就给一个局部外向电流把灶压住，活动停了它再慢慢衰减）。

**揭示了什么（到目前为止）** ——
- **持续态是个稳健吸引子**：开环抬阈值把发放压住、让 `q_I` 回灌，压得越久越不容易失控（压 0.5s 松手失控 293Hz → 压 14s 松手不失控但仍弹回持续态），
  **但即使把 `q_I` 加到接近基线（0.87），低阈值双灶 + 递归骨架仍会把网络重新点着成持续态**。单靠"压住 + 回灌 q_I"退不出去。
  这跟历史上所有终止器失败、以及 R4 必须用离散 latch 硬切**同向**——退出的真正障碍是灶的自发再点火，不是 q_I 不够。
- **动态机制的传感器工作正常**：持续态上 `p` 平滑积累（12s 到 p_max≈0.55），而间期短事件几乎不充它（持续时间选择性成立）。
- **对称动态电流退不出**（被负反馈压到更低持续水平、或饿死 `S_G` 失控）；**但真·不对称 slow-release（快充 τ_p=3000/慢放 τ_p_down=12000）质变**：
  `no_runaway`、活动被压到 0 → `q_I` 回灌 0.6–0.8 → 之后离散短促自终止 burst，不再回宽持续态——**更像生命周期、是有希望的候选**（`arms_asymfix`）。
  （⚠️ 原"不对称=周期性失控脉冲"因 `d_sweep` 漏传 `tau_p_down` 的 P0 bug 实为对称、已作废。）
  **但未确证**：`τ_p=3000` 快充可能在成形期就压住（prevention 而非 termination，须加大 τ_up 复验）、后段 burst 未验真假、seed1 单例、退出 basin（`q_core×S_G`）未映射。
  一个**候选**观察：`q_I` 与 `S_G` 对活动响应**符号相反**、负相关（不是"绝无重叠"——实测有 ~403ms 重叠、均值 `q_I` 可能掩盖 core）。
  **结论=开环+对称退不出；不对称 slow-release=待验证候选，非确证 lifecycle、也非 bounded-negative**。

（内部代号：M4 `k_q=0.10 alpha_G=16` 有界态、`q_I` 场、`S_G` 除法池、persistence 场 `p`、Hill Φ、R4 latch。）

## 1. 方法 / 复现

- 衬底：E1146 narrow twoend_equal，`L=20, N=40000 (NE=32000)`，`AR=2`，两低阈值灶在 E→E 长轴两端。`run_m4_phaseplane.build_substrate`。
- 引擎复用（off-by-default byte-parity，`slow_field.py` 非 guarded，无需 re-bless）：`simulate_kick`（KICK_BOOST=0 自发）、
  `SpatialSlowField`（`q_I` + `S_G` + 新增 `p` 场）、`LFPRecorder`。
- 新增 persistence 场（spec §5）：`τ_p ∂_t p = Ψ(K_p*r_E − θ_p) − p`；恢复电流（仅 E）`I_net −= η_r·Φ(p)`，
  `Φ` 线性或 Hill；`clamp_persist` 冻结 `p`（开环/消融）。8 个契约测试全绿，`BASELINE_SHA` 未变。
- Runner `scripts/run_m4_snn_native_exit.py`（`exit_atlas` 开环探针 + `arms` 动态臂 + `d_sweep`）；诊断图 `scripts/plot_m4_snn_native_exit.py`。
- 资源：canary 单跑 peak RSS 7.12GB，40s 量级慢；worker 预算 `min(18, 36, 8)=8`，首批 ≤2；`OMP=1`；`topic4_resource_monitor.py` 记 `resource_log.jsonl`。

## 2. Stage-0 canary

`(k_q=0.10, alpha_G=16)` seed1 T=2000：`no_runaway`，`q_final=[0.05,0.27]`，`S_G_max=0.41`，peak 64Hz，peak RSS 7.12GB，
wall 364s（build 107s + sim 257s）。**复现 M4 有界态**。

## 3. Stage-1a 退出边界开环探针（seed1；bounded-negative for threshold actuator）

`inhibitory_pulse`（+15mV，全 E）在 t0=3000 起压住不同时长，松手后看是否弹回。

| hold (ms) | q_I@release | 松手判决 | maxHz |
|---|---|---|---|
| 500 | 0.22 | rebound_runaway @3535ms | 293 |
| 3000 | 0.51 | rebound_runaway @6021ms | 241 |
| 6000 | 0.70 | rebound_bounded (fragment) | 130 |
| 10000 | 0.82 | rebound_bounded | 90 |
| 14000 | 0.87 | rebound_bounded | 85 |

**单调**：压得越久 `q_I` 越满、弹回越弱（失控→有界，293→85Hz），但**没有一档退到间期**。机制：压制期 `S_G` 排空，
松手时灶自发再点火（高 q_I 时是 ~2s 缓升而非过冲），把 `q_I` 重新耗回 ~0.10 → 有界态重建。
→ **开环抬阈值终止器：clean no-go（任何时长，q_I 回灌至 0.87 仍不退出）**。图 + README：`results/.../stage1_exit_atlas/figures/`。

## 4. Stage-2 校准（seed1，T=12000，arms A/B/C）

| arm | 判决 | n_ev | maxHz | q_min | S_G_max | p_peak |
|---|---|---|---|---|---|---|
| A_slow_off（间期基线）| no_runaway | 34 | 40.6 | 1.00 | 0.00 | – |
| B_m4_anchor（有界）| persist | 7 | 97.2 | 0.05 | 0.45 | – |
| C_sensor_on（有界+传感器 η_r=0）| persist | 7 | 97.2 | 0.05 | 0.45 | 0.55 |

- A = 34 个短促自终止 IED（q_I=1、S_G=0）= **恢复目标**；B/C 相同（η_r=0 parity 确认）；B/C 有界态是**振荡/bursting**（非平台）。
- C 的 `p`（τ_p=5000, θ_p=0, a50_p=0.3）：p_max 0→0.26(3s)→0.41(6s)→0.55(12s)，p_mean→0.33，慢升；间期短事件不充它。
- **arm D 校准**：Hill Φ `p50_r=0.25, n_r=4`（p<0.15 关、p>0.35 开），`η_r` 从 p_max≈0.55 → 灶上 ~38mV（η_r=40）；扫 `τ_p∈{5000,8000}`, `η_r∈{40,80}`。

## 5. Stage-2 dynamic arm D（seed1，T=20000，Hill Φ p50_r=0.25）

对称 `p`（单 τ_p）扫描：

| arm | cls | maxHz | q_mean_fin | area_tail | p_peak | 解读 |
|---|---|---|---|---|---|---|
| B_m4_anchor | persist | 116 | 0.09 | 0.67 | – | 有界参照（宽） |
| D τ5000 η40 | fragment | 67 | 0.17 | 0.18 | 0.22 | 压到更低的持续水平（~25Hz），不终止 |
| D τ8000 η40 | **runaway** | 375 | 0.12 | 0.18 | 0.20 | 弱电流饿死 S_G → 失控（同 M4-3A shunt 失效模式）|
| D τ8000 η80 | fragment | 126 | 0.21 | 0.14 | 0.22 | 同上，压到更低持续水平 |

- **机制判读**：对称 persistence 电流是一个**负反馈控制器**（把活动调到一个更低的设定点），**不是终止器**。
  一压活动就少 → `p` 的驱动就少 → 电流就弱 → 稳到一个自洽的低持续态（~25Hz、`q_I` 只回灌到 ~0.18）。
  太弱/太慢的电流反而饿死 `S_G` 除法池 → 失控。**都不给 clean terminate+recover**——与 STD/shunt 的 fragment/suppress 撞同一堵墙。
- **反馈自限**：`p` 只涨到 ~0.22（远低于 sensor-only 的 0.55），因为压制切掉了自己的传感器输入（闭环自限），有效电流比标定弱。
- **传感器选择性（⚠️ 控制参数不匹配，结论待补）**：`A_sensor_on`（slow-off + sensor η_r=0）测得真 34 个间期事件 `p_max` 只涨到 **0.084**（p_mean 0.027），
  远低于有界态的 0.55——这是好迹象。**但此控制用 `p50_r=0.25`（`Φ(0.084)≈0.013`≈0），而候选用 `p50_r=0.15`**：`Φ(0.084)≈0.090` → 电流 **7.2mV(η80) / 13.4mV(η150)**，
  **不可忽略**（review P0-science 抓到）。且候选的慢衰减会跨事件累积。所以**不能**据 `A_sensor_on` 说"候选只作用持续态、不误伤 IED"——
  必须用**候选参数匹配**的 slow-off + actuator 控制（`A_persist_act`：τ3000/τ_down12000/p50=0.15/η80,150，`arms_prevctl_eta*` 运行中）看真 IED 是否存活。这是判"prevention vs termination"的关键门。
- **空间**：源空间帧显示电流把宽持续态推成一个**大面积游走活动**（压这里→那里冒→漂移），非局灶起始、非轴向招募、非终止波前
  （空间验收失败；6 帧不足以严格区分"连续游走"与"不同区域交替点燃"，下一版需 kymograph）。图 `results/.../stage2_arms/figures/{arms_s2d_seed1.png, spatial_D_tau5000_eta40.png}`。
- **⚠️ asymmetric `p`（见 §7）**：原 `s2dasym` 因 `d_sweep` 漏传 `tau_p_down` 实为对称、已作废；**真·不对称（τ_up3000/τ_down12000）已修复重跑 `s2dasymFIX`，结果折入 §7**。

## 6. Stage-2 ablations（seed1，符号版 arm-D 参数 τ5000 η40）

| ablation | cls | n_ev | maxHz | q_min | S_G_max | p_peak | 揭示 |
|---|---|---|---|---|---|---|---|
| E1_no_qI（k_q=0）| suppress | 13 | 25.7 | 1.00 | 0.06 | 0.04 | 无 q_I 耗竭=无 entry → 不成有界态、`p` 不涨、电流无对象 → q_I 是 entry driver |
| E4_clamp_p（p≡0.8 恒定）| suppress | 0 | 0.0 | 0.89 | 0.00 | 0.80 | 恒定电流=纯 prevention（0 事件、q_I 不耗）→ **动态延迟是必需的**（arm D 让发作先成形再压）|

（E2_no_SG 的 no-pool→runaway 已由 M4 pass1 机制对照确立，未重跑省时。）

## 7. Stage-2 asymmetric arm-D —— ⚠️ 原运行作废（P0 bug），已修+重跑

**⚠️ 撤回**：标为 "asymmetric"（`arms_s2dasym_seed1`, `arms_s2dasym_s3_seed3`）的运行**实际是对称 `τ_p=3000`**——
`_build_arms` 的 `d_sweep` 分支逐项拼 `_persist_cfg` 时**漏传 `tau_p_down`**（默认 `None`=对称），尽管 `argv` 里写着 `--tau-p-down 12000`
（review 2026-07-21 P0 抓到）。故"不对称→周期性失控脉冲列"这个结论**无效、撤回**，主结论图第四列作废。

**已修**：`d_sweep` 改为从 `P`（唯一 persist 参数源，含 `tau_p_down`）构建、`{**P,"tau_p":tp}` 覆盖，label 加 `_dn{tau_p_down}`；
每 row 落 `cfg_effective`（不再只靠 argv）；加 `test_d_sweep_propagates_tau_p_down` 回归测试（修复前会失败）。

**真·不对称结果（`arms_asymfix_seed1`，τ_up=3000 / τ_down=12000, η_r=80/150, `cfg_effective` 确认 tau_p_down=12000, T=20000, early-stop on）**：

| arm | cls | maxHz | q_mean_fin | area_tail | runaway | 轨迹 rate@[3,6,10,14,18]s |
|---|---|---|---|---|---|---|
| D τ3000 η80 dn12000 | fragment | 67.6 | **0.61** | 0.0 | None | 18.8 → 0 → 0 → 8.1 → 9.8 |
| D τ3000 η150 dn12000 | fragment | 126 | **0.60** | 0.0 | None | 16.2 → 12.5 → 0 → 6.6 → 8.3 |

**关键：跟错误的"runaway train"结论相反——真·不对称 hold `no_runaway`**，且行为**质变、更像生命周期**：初段活动（~20–50Hz）→ 电流把它**压到 0**
→ 一段 ~8–10s **安静期 `q_I` 回灌到 0.7–0.8** → 之后出现**离散、短促、自终止的 burst**（~14.5s、18.5s，每个伴 `q_I` 下凹 + `S_G` 瞬起）。
即"活动 → 压住 → `q_I` 恢复 → 返回式短事件"，**不再回到宽持续态、不失控**。这正是 review 猜的"慢释放让 `q_I` 恢复 + 缓慢放开网络"的机制。

**但两条诚实 caveat（别再过度解读）**：
1. **可能是 prevention 而非 termination**：`τ_p=3000` 快充 + 强 `η_r` → 电流约 3s 就介入，此时有界态还在爬升（~20–30Hz、没到满态 ~80Hz）→ 更像"在成形期就压住"而非"让发作充分成形后再终止"。要判"form-then-terminate"须**加大 τ_up**（如 5000–8000）让有界态先立起来。
2. **后段离散 burst 未验证**是真返回间期事件还是 `p` 衰减到阈下的击穿；`cls=fragment`（非 `terminate_clean`）；seed1 单例；退出 basin 未映射。

**定性**：真·不对称是**有希望的候选**（避开了 open-loop/对称的 rebound/lower-persistent/runaway 失效），**不是**确证的 clean lifecycle。下一步见 §8。

## 8. Verdict —— open-loop+对称 actuator 退不出；不对称 slow-release = 候选（未确证）

**一句话**：开环抬阈值 hold + 对称闭环电流都**没能**把 M4 有界态干净退回间期（rebound / 更低率持续态 / runaway）。
**但真·不对称 slow-release（快充 τ_p=3000 / 慢放 τ_p_down=12000）质变**：`no_runaway`、把活动压到 0 后 `q_I` 回灌到 0.6–0.8、
之后出现离散短促自终止 burst，不再回宽持续态——**更像生命周期，是有希望的候选**。这**不是**确证的 clean lifecycle
（可能是 prevention 而非 termination、后段 burst 未验、seed1 单例、退出 basin 未映射），也**不是**原先误报的 bounded-negative
（那个"不对称→runaway"因 P0 bug 实为对称、已作废）。

**候选共同机制（不是 impossibility proof）**：`q_I` 与 `S_G` 对活动 `R` 的响应**符号相反**——
`∂q̇_I/∂R < 0`（活动耗 `q_I`，局部去抑制助点燃），`∂Ṡ_G/∂R > 0`（活动建 `S_G`，除法压 recurrent 完成 containment）。
两者随活动周期**负相关**：压住活动去灌 `q_I` 往往让 `S_G` 衰减，已测轨迹没把网络带进低活动间期 basin。
**但注意**：(a) `q_I` 有**始终存在**的恢复项 `(1−q_I)/τ_q`、`S_G` 有**有限记忆** `τ_S`——不是"安静即零/活动才有"的二值；
实测 14s-hold release 后 `q_I≥0.5 ∧ S_G≥0.2` **确曾同时成立 ~403ms**，只是不足以掉进低 basin。(b) 用的是**空间均值** `q_I`，
可能掩盖 core 仍低于 surround。**因此"退出 corridor 是否存在、需要多高的 `q_core`/`S_G`/维持多久"未测**——要一张 `q_core × S_G`
冻结-态 exit atlas 才能定。R4 用"固定 bath（跟活动解耦的 `q` 储库）+ latch"闭环，可解读为**绕过**这个负相关，但这不等于 SNN 里绝无退出。

**完成度（分层，诚实）**：
- engineering green：✅（persistence 场 off-by-default byte-parity，8+ 契约测试 + `BASELINE_SHA` 全绿；无引擎 re-bless）。
- fast-state existence：✅（复现 M4 有界态 persist；复现间期 34 IED 基线）。
- exit / dynamic accessibility：开环 hold + 对称电流 ❌；**不对称 slow-release = 🟡 候选**（no_runaway + `q_I` 回灌 0.6–0.8 + 离散短事件；未确证）。
- termination / recovery：对称 ❌；不对称 🟡（活动被压到 0 + `q_I` 恢复 + 返回式短 burst，但 `cls=fragment` 非 `terminate_clean`、可能 prevention）。
- spatial pattern：❌（电流把宽态推成大面积游走活动，非局灶起始/终止波前）。
- cross-seed：seed1（open-loop 5 + 对称 3 + 消融 2）+ seed3 对称复核（D_tau5000_eta40=fragment、D_tau8000_eta80=runaway）。
  **两 seed 每个已测对称动态臂都落 {fragment, runaway}、无一 clean terminate**；fragment↔runaway 边界随 seed 微移（S_G-饿死阈值 seed-敏感）。
  （原"不对称 D_tau3000"两 seed 因 P0 bug 实为对称 `τ_p=3000`，已作废；真·不对称 cross-seed 视重跑结果补。）

**能写 / 不能写**（review 2026-07-21 收紧后）：
- 能写：M4 有界态在**已测窗口内**是 reproducible persistent state；**已测的**开环 hold + 对称恢复电流不能 clean-exit（2 seed）；
  `q_I` 与 `S_G` 对活动响应符号相反、随周期负相关，是这些失败的一个**候选共同解释**。
- 不能写：❌"证明不存在退出轨迹 / `q_I`-`S_G` 绝无重叠"（未做 basin/nullcline atlas，且实测有 ~403ms 重叠）；❌"稳健吸引子"
  （只测了单 actuator/固定 ΔV_th/5 hold/seed1 的开环 + 有限对称扫描）；❌"穷举/任意时长"；❌"解释 lineage 所有失败"（只能说"为多个失败模式提供一个候选共同解释"）；
  ❌"不对称 hold 也失败"（真·不对称重跑中）；❌"造出 seizure lifecycle / 证明发作机制 / 任何 clinical/SOZ"；❌把 fragment/游走当"发作样招募"。

**与旧工作的关系（真正新增了什么）**：不是把 STD/shunt/m/g_K 换名重跑——本线加的是**持续时间门控 + 空间局部**的恢复坐标
（lineage 从没测过的组合），并**在 M4 有界态上**测；新增的正面视角是 **`q_I` 与 `S_G` 对活动反号**这个可能统一多个失败模式的
**候选**诊断（非证明），以及一个可复用的、测试齐的 `p` 场基础设施。

**下一步（按 review 2026-07-21 §7 优先级，交用户定向）**：
1. **[做中]** 修 P0 后重跑真·不对称 `p`（τ_up=3000/τ_down=12000，η_r=80/150；有 candidate 才上 seeds 3/4，无则停）。
2. **`q_core × S_G` frozen exit atlas**（P1，关键）：从同一 M4 persistent checkpoint 分叉，独立 clamp `q_core∈{0.2..1.0}×S_G∈{0..0.4}`，
   短跑 fast 子系统分类 low/rebound/bounded/runaway，再叠真实动态轨迹——直接回答"退出 basin 是否存在、需要什么组合、当前轨迹是错过/穿太快/无低支"。
3. 若 basin 存在但轨迹错过 → 试**持续性全局 containment memory** `H`（`τ_H Ḣ = P_G(p) − H`，`I_EE^eff = I_EE/(1+α_f S_G + α_H H)`，
   慢建慢衰、活动暂降时仍保 recurrent-E containment、`q_I` 恢复后再衰减），直接作用 M4 分母、不改 E→E、与并行 EE 线独立。
4. 补齐 review 列的对照：open-loop **current** actuator atlas、matched-instantaneous sensor、E2 no-`S_G` 同批、core/surround/`q_min` 慢变量 traces、
   连接长轴+横轴 space–time kymograph（真 mm 坐标）。若 frozen atlas 显示即使 `q_core=1、S_G/H` 足够高也无可保持低 basin → 本线正式停、瓶颈交 fast substrate。

## 9. 产物、Stage-4 空间、cross-seed、资源、git

**关键图**（`results/topic4_sef_hfo/m4_snn_native_exit/`，各 `figures/` 有中文 README）：
- `figures/fig5_exit_attempts_diagnostic.png` —— 四列结局（persist / open-loop rebound / symmetric lower-rate-persistent / **真·不对称 candidate=suppress→refill→discrete bursts**）。
- `stage1_exit_atlas/figures/exit_atlas_s1_seed1.png`（短压 500/3000/6000）+ `exit_atlas_s1long_seed1.png`（长压 10000/14000）。
- `stage2_arms/figures/{arms_s2cal_seed1.png（A/B/C 校准）, arms_s2d_seed1.png（对称扫描）, arms_asymfix_seed1.png（真·不对称 candidate）, spatial_D_tau5000_eta40.png（对称态游走）}`。
- **作废**（P0-bug，移入 `stage2_arms/invalidated_pre_p0_fix/`）：`arms_s2dasym_seed1.*`, `arms_s2dasym_s3_seed3.*`（实为对称 τ_p=3000）。

**Summary JSON**：`stage1_exit_atlas/exit_atlas_s1{,long}_seed1.json`、`stage2_arms/arms_s2{cal,d,abl}_seed1.json` + `arms_asymfix_seed1.json`（真·不对称）+ `arms_sensor_/prevctl_eta*`（selectivity/prevention 控制）（raw `.npz` gitignored）。

**脚本**：`scripts/{run_m4_snn_native_exit.py（exit_atlas 探针 + arms 动态臂 + d_sweep + A_persist_act）, plot_m4_snn_native_exit{,_spatial,_summary}.py, topic4_resource_monitor.py}`；引擎 `src/snn_engine/slow_field.py`（`p` 场，含非对称 `tau_p_down`）；测试 `tests/test_m4_snn_native_exit_persist.py`（10 绿含 `test_d_sweep_propagates_tau_p_down` + `BASELINE_SHA` 绿）。

**spec / archive**：`docs/superpowers/specs/2026-07-21-topic4-m4-snn-native-exit-design.md`；本文件。

**Stage-4 空间**：对称 fragment 态（`spatial_D_tau5000_eta40.png`）是大面积游走活动（非局灶）。真·不对称 candidate 的后段 burst **空间上仍是宽条带/半平面切换、非紧凑双核事件**——即使时间上孤立、空间上没恢复成原始 IED 形态，故**不能**写"回到同一种间期事件"。**未跑** Stage-3 自发长轨迹 / paper-ready Figure 5（候选未确证，按 cheap-first 先做 §8 窄确认）。

**cross-seed**：seed1（open-loop 5 holds + 对称 3 + 消融 2 + 真·不对称 2）+ seed3 对称复核 `arms_s2d_s3b_seed3.json`（D_tau5000_eta40=fragment、D_tau8000_eta80=runaway）。
**已测对称动态臂两 seed 都落 {fragment, runaway}、无 clean terminate**（fragment↔runaway 边界随 seed 微移）。真·不对称 candidate 目前只有 seed1，多 seed 待窄确认后再上。
（⚠️ 原 §此处引用的"不对称 D_tau3000_eta150=train_then_runaway seed3"是 P0-bug 的对称跑、已作废。）
（注：`s2d_s3` 首次 detached 跑因 shell 退出 SIGHUP 死于写 JSON 前 → 已 harness-tracked 重跑为 `s2d_s3b`。）

**资源**：worker 峰值 ≤8（多为 2–4），canary peak RSS 7.12GB，全程 `min mem_avail_frac=0.85`、**swap 增长 0**、未触发 protective stop、
不干扰并行 FCXR 线；`OMP=1`；`resource_log.jsonl` + `topic4_resource_monitor.py`。

**git**：worktree `.worktrees/topic4-m4-snn-native-exit`，branch `codex/topic4-m4-snn-native-exit`，base `4d40b03`（clean，未混入 R4/FCXR）；
commits `8109ccd`(spec) `d79d9f9`(persist 场+runner+tests) `69e431a`(Stage-1a 开环+diag) `c85e2ad`(asymmetric-p+machinery) `bb292e1`(verdict+diagnostics)（未 push、未合 main）。

## 7. Provenance

commits: `8109ccd`(spec) `d79d9f9`(persist field+runner+tests) `69e431a`(Stage-1a screen+diag)；
tests `tests/test_m4_snn_native_exit_persist.py` 8 绿 + `test_snn_gates.py` BASELINE_SHA 绿。
