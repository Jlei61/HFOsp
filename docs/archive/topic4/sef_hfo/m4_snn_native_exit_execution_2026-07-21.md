# SNN-native M4 containment-to-exit lifecycle — execution log (2026-07-21)

> **最终验收入口（2026-07-26）**：旧 q_I 沙盒继续保持隔离；正确 Z/M 重建、
> \(S_G/H\) 结果、carrier gate 与局部抑制场的统一判决见
> `zm_carrier_exit_line_acceptance_2026-07-26.md`。当前安全结论是“entry 到 runaway
> 成立；containment 产物不是持续 carrier；完整 lifecycle 未建立”。

> ## 🔴 WRONG-SUBSTRATE — ARCHIVE-ONLY q_I SANDBOX (review 2026-07-22)
> **This entire line ran on the OLD field-based `q_I + S_G + p/H` model** (`src/snn_engine/slow_field.py`,
> `use_qI=True, use_gK=False`), **NOT the current locked `Z/M` per-neuron model** (`src/snn_engine/slow_vars.py`,
> `use_z/use_m`; lockpoint `zA_q75_tz5000__mA0p001_tau500`, τ_Z=5000/τ_M=500/A_M=0.001). The 2026-07-21 spec
> handed to this line explicitly defined the substrate as q_I+S_G (§5 line 120) and said z is "not used here"
> (§3 E2); the agent followed the spec but **failed to surface that the task brief said "preserve Z/M slow
> variables"** (CLAUDE.md §1 miss). **Every scientific conclusion below is q_I-M4-sandbox only and is NOT
> validated on the current Z/M SNN** — the H "terminates", "no recovery attractor", t_form=1600, onset=2300,
> the frozen `q_core×S_G×J_exit` atlas all describe the q_I sandbox. **Reusable = engineering + method** (Phase-0
> crash-safe resume/provenance; formed-state detection; core/surround + kymograph + virtual-SEEG readouts;
> frozen dual-IC atlas; termination/recovery classifiers) **and H as a CANDIDATE divisive slow-memory mechanism
> to re-test on Z/M.** Note: `slow_vars.py` (Z/M) is per-neuron and has NO S_G / no divisive term, so "Z/M + S_G
> + α_H H" is a NEW combination to build; the Z/M lines live in the peer MZ worktrees this line's spec listed
> READ-ONLY. **Correct口径**: *the old q_I-M4 sandbox suggests a slow divisive memory MAY terminate its bounded
> persistent state, but this is NOT validated on the current Z/M SNN; the recoverable Z/M lifecycle is UNSOLVED.*
> At this historical checkpoint the branch was unmerged/unpushed and did not touch any Z/M worktree;
> current delivery status is recorded in the 2026-07-26 unified acceptance linked above.

---

## 🟢 Z/M-native rebuild (session-3, 2026-07-22 → 07-23) — CORRECT substrate

按 WRONG-SUBSTRATE 复审，在**正确的 Z/M 衬底**上重建这条线（用户选 "port z+m into slow_field.py" + "Z/M+S_G+H as written"）。
代码见 commits `538c8ae`(移植)/`c144ef7`(harness)/`4bf3828`(active H 传感器)/`4c7af2e`(p 追踪+json 累积)。

**测了什么（朴素话）** —— 换成当前锁定的模型：每个兴奋神经元自己带两个慢变量——`z`（抑制效能，长时间挨强抑制就"疲劳"、把抑制卸掉→去抑制）、`m`（放电后适应）。在真实 E1146 双灶衬底（L=20, N=40000）上，先标定那个疲劳阈值 `I_th_EI = 静息态兴奋细胞所受抑制电流的 q75 = 1.28`（只有挨最强 25% 抑制的细胞才疲劳→去抑制→点火，这就是慢场→发作的机制入口），再看能不能走完整生命周期：间期 IED → 起始 → 有界发作 → 终止 → 回间期。

**怎么测的（梯级 + 逐位一致地基）** —— 先证明移植没走样：新搬进 `slow_field` 的 z+m，和它的规范母本 `mz_slow_vars.py`，同一网络同一噪声跑出来的**每个神经元每步放电栅格逐位相同**，且和"关掉慢变量"的基线明显不同（`tests/test_zm_slow_field_parity.py`）。然后三臂梯级（自发协议，两个低阈核自己点火，无外部 kick）：
- **裸 Z/M**：只有 z+m。
- **+S_G**：加原来那套"除法式全局抑制池"当容纳器。
- **+S_G+H**：再加"慢记忆" H（不随活动塌、慢建慢衰的除法兜底），公平地试了 3 种 H 传感器（全局空间均值 / active-focus 均值 / active + 非对称 p 累积）。

**揭示了什么（seed 1 pilot 判决）：**
- **裸 Z/M = 间期→起始→失控**：核区先冒一串离散 IED（≈350/750/1100/1550/1900ms），随 z 疲劳（z_core→0.75）越来越密、招募 surround，冲到 175Hz 失控（~2.5s 截断）。链路端到端验证通过，且和独立的 mz-onset-dynamics 线对上口径。
- **+S_G = 按成"持续 bursty IED 串"**：S_G 把平滑失控整形成一串越来越猛的离散爆发（核峰 100→325Hz），但**全场平均率只有 ~2.2Hz（=间期级！）**——低占空比。z_core 疲劳到准稳态 ~0.31–0.33，S_G 随每次爆发涨落、事件间就塌（S_G_max~0.13）。有界，但 25s 内**不自终止**。（和 q_I sandbox 一个教训：S_G 只塑形/封顶，不终止。）
- **+S_G+H = 终止阻塞，H 建不起来**：3 种传感器 H_max 全 ≤0.035（0.011/0.029/0.035）。存了 p 迹象后**诊断清楚**根因：被 S_G 容纳的 Z/M 发作态是**低均值率的 bursty 串**（全场均值 2.2Hz、p_max 才 0.09），持续场 p 一时间平均就被稀释到 ~0.01–0.09 → H 没有可积累的"持续活动"信号。**这不是传感器工艺问题，是本质错配**：H 当年在 q_I sandbox 能终止，只因为那个有界态是**持续高率平台**；Z/M 的有界态是**bursty 低均值串**，天生喂不饱一个靠"活动持续度"的终止器。（连 H_max=0.035×α_H=16=0.56 都把峰从 72 削到 40Hz，但离终止很远；m 在锁点 η_m=0.001 可忽略。）

**判决**：**Z/M 容纳-退出生命周期的终止+恢复两条腿仍未解，但瓶颈和 q_I 不同**——q_I 卡在**恢复**（没有能回去的稳定间期吸引子）；Z/M 更早一步卡在**终止**（S_G 容纳出的 bursty 低均值串既不自终止、又喂不饱 persistence-based H）。当初赌的"Z/M 的 z 会自愈→1、事件后重建可兴奋态、所以恢复也许能成"——**测不到**，因为根本走不到终止那一步，z 一直被活动压在 0.31 附近、拿不到安静窗去自愈。这是**公平测了 H（3 传感器、诊断非瞎猜）之后的诚实负结果**。下一杠杆（待用户）：按爆发**计数**（而非时间平均）的终止器；把 m 适应调强（脱离锁点）让衬底自己适应出来；或接受 Z/M+S_G bursty 串就是模型终点。**仅 seed 1，多 seed 稳健性待跑。**
可复用工程：Z/M-native harness `scripts/run_zm_snn_native_exit.py` + 标定 + active-focus H 传感器（opt-in，默认 `global` 保持 q_I byte-parity）。图 `results/topic4_sef_hfo/zm_snn_native_exit/figures/`。

---

> **Status: 开环+对称退不出；不对称 slow-release = 待确认候选（GO 窄确认）.** Stage 0–2 done; P0（`d_sweep` 漏传 `tau_p_down`）
> 已修 + 真·不对称重跑（`arms_asymfix`）。当前科学标签=**slow-release suppression–rebound bursting candidate**，非确证可恢复 lifecycle。
> **候选参数匹配控制已证实电流压制普通 IED（34→15/12）→ 相当程度 prevention、选择性不足**（`arms_prevctl_eta*`）；能否终止**已成形**态由 established-state fork 判（running）。Verdict §8。
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
  **⚠️ 但候选参数匹配 prevention 控制证实电流压制普通 IED：34 个 → 15（η80）/12（η150）**——所以上面"候选"有一半是普通 IED 被压掉、只剩击穿，**相当程度是 prevention**。仍未确证：能否终止**已成形**态（established-state fork running）、后段 burst 真假未验、seed1、basin 未映射。选择性不足（`θ_p=0`），修法=按活动幅度门控。
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
  必须用**候选参数匹配**的 slow-off + actuator 控制（`A_persist_act`：τ3000/τ_down12000/p50=0.15/η80,150）看真 IED 是否存活。
  **结果（`arms_prevctl_eta{80,150}`）：候选电流确实压制普通间期事件——34 个 IED → 15（η80）/ 12（η150），峰值 40.6→32Hz**。
  机制：`τ_down=12000` 慢衰减让 `p` 跨 IED 串累积（正是 review 警告的），逐步压掉后面的事件。**所以候选是相当程度的 prevention、不是干净的选择性终止**——
  "候选 lifecycle"里的"发作压住 + 之后离散 burst"有一半是"普通 IED 被压掉、只剩一部分击穿"，不是真恢复。**修法=更好的选择性**（`θ_p` 按活动**幅度**门控：让 40Hz IED 不充 `p`、只让 ~80Hz 有界态充），这是下一轮的必测项，不是当前能声称的。
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
真·不对称 slow-release（快充 τ_p=3000 / 慢放 τ_p_down=12000）**不失控、且出现"压住→`q_I` 回灌→离散 burst"轨迹**，一度看着像候选；
**但候选参数匹配的 prevention 控制证实：该电流把 34 个普通 IED 压到 15/12——相当程度是 prevention、不是干净选择性终止**（慢衰减让 `p` 跨事件累积）。
所以当前科学标签只能是 **slow-release suppression–rebound bursting candidate**，且**选择性不足**（`θ_p=0`，需按活动幅度门控）；
**不是**确证 lifecycle、**也不是** bounded-negative（原"不对称→runaway"是 P0-bug 的对称跑、已作废）。"能否终止**已成形**有界态（vs 仅 prevention）由 established-state fork（`persist_onset_ms=2500`）判，running。

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
commits `8109ccd`(spec) `d79d9f9`(persist 场+runner+tests) `69e431a`(Stage-1a 开环+diag) `c85e2ad`(asymmetric-p+machinery) `bb292e1`(verdict+diagnostics)（当时未 push、未合 main）。

## 7. Provenance

commits: `8109ccd`(spec) `d79d9f9`(persist field+runner+tests) `69e431a`(Stage-1a screen+diag)；
tests `tests/test_m4_snn_native_exit_persist.py` 8 绿 + `test_snn_gates.py` BASELINE_SHA 绿。

---

## 10. Session-2（2026-07-21 续）：Phase 0/1/2 re-plan 执行

**执行决策——杀掉 estfork**：上个会话遗留一个 `--persist-onset-ms 2500` 的 established-state fork 进程仍
存活（本 doc §8 曾记"running"）。它把恢复电流**固定在 2500ms 介入**——正是 re-plan Phase 1 明确禁止的
"默认 2500ms 已成形"假设，且用未修的旧 runner（无断点续跑/provenance）。已杀，Phase 1 用"先测成形时间、
再介入"的干净版本重做。`invalidated_pre_p0_fix/` 旧结果仍作废、不复用。

**Phase 0（工程，commit `4d7a4d6`）——runner 崩溃可恢复 + 全 provenance**：旧 `_run_arms` 用 `pool.map`
一次性等全部 arm 跑完才写一个合并 JSON——中断丢掉全部、无续跑、JSON 无 `base_sha`/`engine_versions`、
`cfg_effective` 漏了 `persist_onset_ms`。改为 `_orchestrate_arms`：每个 arm 一完成立刻落 per-arm JSON+NPZ、
每次完成重写 `run_manifest`（pending/running/complete/error）、`--resume` 跳过已完成 arm；每份输出记
`base_sha`+`engine_versions`(guarded 引擎 sha256)+完整 `cfg_effective`；`_engine_guard()` 对 guarded 引擎漂移
loud-fail。**本会话网络中断恰好实测了它**：anchor 进程被上个会话退出杀掉、只留 `_running_` 标记无完成产物，
`--resume` 干净重跑。7 新 runner 测试 + BASELINE_SHA 门全绿、无 re-bless。

**Phase 1/2 基础设施（commit `2230fad`）**：
- `slow_field.py`（unguarded、只读 trace→字节一致性保住）：借已有 `core_mask_E` 钩子加 core/surround 场
  分裂 → 每步 `q_core`/`q_surround`(+`p_core`/`p_surround`)。3 新测试 + parity 门绿。
- runner：`_run_persist_arm` 现输出 Phase-1 清单（core/surround 慢变量 trace、沿轴+横轴 kymograph、
  核/周活动率）；新增 `frozen_atlas` 模式（见下），复用 `_orchestrate_arms`。
- `analyze_m4_snn_native_exit.py`：`formed_state_time`——数据驱动 t_form（非假设 2500ms）：率/`S_G`/`q_I`/
  active-area **全部**连续落在有界带 ≥ window_ms 的最早时刻；末态非有界则返回 None。

**Phase 2（commit `735abce`；审阅纠正 07-22）——冻结退出相图：低放电/静默落脚点存在（非已证明回到原有间期事件的间期态）**（朴素话）：
- **测了什么**：那个"停不下来的持续放电态"里，网络到底有没有一个**低放电/静默的落脚点**能退回去；要多满的
  抑制油箱 / 多强的除法刹车 / 多大的恢复电流才够得着。
- **怎么测的**：把三个慢变量冻死（抑制油箱 `q_core`=**整片均匀**冻结 q_I、除法刹车 `S_G`、**全场恒定**外向电流
  `J_exit`=p≡1 产生），让真脉冲网络从冷（不踢）/热（狠踢核区=**焦点** kick 代理"已在发作"）两起点各跑。
  seed 1、T=2500、27 格×2 起点=54 跑、无引擎改动。**四类判决**：only-低 / bistability-consistent(冷低热高) /
  only-高 / reverse-discordant(冷高热低,非双稳)。
- **揭示了什么（安全口径）**：**低放电落脚点广泛存在**（17/27 冷热都归低；`q_core=0.9` 处处低放电）。高支=
  **2 only-高**（`S_G`=0+耗竭 q+无电流）+**7 bistability-consistent（全是 低-vs-失控，非 低-vs-有界发作态→
  只是"双稳兼容双探针结果"、非已证明 间期↔发作 双稳）**+**1 reverse-discordant**（q0.9/S_G0/J0: 冷高8Hz振荡/
  热低,判不清）。**抬高 `S_G`+`J_exit` 能把热起点从失控转成低放电**，即便 q_core=0.05[(S_G0.4,J≥8)/(S_G0.2,J20)]。
  **分工（限定在 抑制耗竭+被 kick 激活的已测区域）**：除法刹车 `S_G` 是必需刹车（=0 则踢就失控 at q≤0.4）；
  **但 `S_G` 非普遍必需**——`q_core=0.9` 或 `J_exit` 够强时无 `S_G` 也归低。**→ 排除"测试范围内处处高态/失控"**；
  瓶颈从"有没有低落点"收缩成**纯动态问题**：轨迹能否进入、停住、**释放后恢复原有 IED**（活动降则 `S_G` 衰减
  →可能需"刹车慢记忆" H）。
- **⚠️不能写**：低格多是**近0Hz静默态、非已证明"回到产生原有间期事件的间期吸引子"**；只能说"冻结慢坐标下
  存在低/静默 fast-state 结果"。caveat：冻结≠动态；冻结 `S_G` 低估动态刹车；`q_core` **均匀**冻结(非核空周满真实
  空间场)、`J_exit` **均匀**恒流(非动态局部 p(x,t))；热IC=焦点 kick 代理(kick_probe guarded 无真 V-checkpoint)；
  未探 wavefront stall/annihilation；seed 1。产物：`phase2_exit_atlas/{figures/exit_atlas_coarse_seed1.png+README,
  arms_coarse_seed1.json, exit_atlas_analysis.json}`。

**Phase 1（进行中；审阅 07-22 纠正后重排）**：anchor `B_m4_anchor` T=12000 seed1 因会话边界/内存压力**死过**
（机器有 OOM 历史；单长 arm 未完成仍需整臂重跑）→ `--resume` 续跑 + 持续监控。审阅要求的**硬合同**（在启动
干预臂前必须满足）：
1. **formed-state 用 core/surround 非空间均值**：`formed_state_time` 现主要用 `trace_qI_mean`；须升级为核率成形+稳、
   周招募达 bounded plateau、`q_core` 耗竭、`q_surround`−core 差已成形；且 t_form 对 window{1.0,1.5,2.0}s + 阈值小扰动
   稳定才可启动（先出 formed-state 诊断图 + 敏感性）。
2. **干预臂两严格合同**：(a) intervention 与 anchor 在 t_form 前 spike/rate/slow trace **逐字一致**；(b) 每个 intervention
   自己在介入前也要**重新过 formed-state gate**（不只继承 anchor 的 t_form）。
3. **真 recovery matcher**：终止后不能只看"又冒 burst"；须把 post-offset 事件与原 slow-off IED 比 duration/IEI/peak-rate/
   active-area/core-surround-ratio/轴向顺序/spatial-mode/虚拟电极顺序。没有这个 producer，`lifecycle-candidate` 只是未接线
   的标签（`classify_phase1_verdict` 现只收外传 `recovered_events` 计数）。
4. **不急着上 H**：只有 Phase 1 结果=fragment/rebound（活动降 `S_G` 消失太快）才测 containment-memory H；
   termination-only→调释放时间+sensor selectivity；prevention→改 persistence sensor 防跨事件累积；termination+matched
   IED recovery→做 seed 3/4+空间确认不加机制。H 是现有 M4 分母的慢记忆、不改 `W_EE`/各向异性。
5. **不做 paper-ready lifecycle 作图**，直到自主长轨迹（去掉所有定时开关）跑通。

判决：invalid / termination-no-go / termination-only / lifecycle-candidate。

Session-2 commits：`4d7a4d6`(Phase0) `2230fad`(Phase1/2 infra) `735abce`(Phase2 atlas)；41 测试绿。审阅 07-22 后：
plot 4 类 taxonomy 修正(17 low-only/7 bistability-consistent/2 high-only/1 reverse-discordant)+全文档"低=近0Hz静默≠
interictal"改口径。

---

## 11. Phase-1 + Phase-3 结果（2026-07-22，seed 1）——一句话：**能进、能兜、能终止；回不去**

**Phase-1 form-then-terminate 判决 = termination-no-go / fragment**（commit `2099320`）：无-p anchor（T=12000, 73min）
形成真宽有界态，数据驱动 t_form=1600ms（敏感性稳），85% plateau ~2150ms → onset=2300ms（τ_up=3000 使电流
~3370ms 才起效、过成形，MEASURED 非盲设）。两干预臂（η=80/150）都 fragment：电流**能把成形态压到 0**，但活动一停
S_G 塌 → 兜底没了 → 灶复燃成 11.8/13/14.5s 的稀疏击穿（非恢复 IED）；赖着的慢衰 p 又过度压制。匹配 prevention 对照
证实选择性差（门控电流仍把 onset 后 IED 29→11）。两严格合同都过（介入前逐字一致、onset 处确成形）。**顺带纠错**：
M4 有界态是宽条带（q_core≈q_surround≈地板、无核-周梯度，spec §9），我的成形判据"梯度硬门"错杀真态、已改判宽招募+宽耗竭。

**Phase-3 vNext = 除法兜底慢记忆 H（commits `10d036f` 建 + `3100695` 判决）**：`τ_H Ḣ=<Φ(p)>−H`，进
`I_EE/(1+α_G S_G+α_H H)`，off-by-default 字节一致、无 re-bless、4 TDD 绿。三消融 + 快放诊断：
- **H-only**（C_sensor_on + H、无减法电流）：H 建到 **0.31**（活动没被自限）→ 除法兜底 **~5.2s 终止**有界态 + 兜过回灌
  （无击穿、q_I→0.82）。因果：无 H = anchor persist 永不终止；加 H 才终止。**→ H 是干净的新终止器、比自限的减法 p 干净。**
- **p+H**（同 fragment 干预臂参 + H）：H 几乎没建起来（0.05）——减法电流把活动压死、切断 p 传感器、`<Φ(p)>` 被稀释；
  仍终止 + **静默**。
- **快放**（τ_down=2000）：放太快、抑制油箱没灌满 → 灶复燃成 ~200–250Hz 反复爆发（**runaway**）。
- **→ Recovery 腿处处失败：保住(H)=静默、放快=复燃，中间没有间期事件那一档。** 跟 Phase-2 相图一致（low=静默）。
  **机制定位：间期 IED 是"抑制油箱边耗边发"的进入瞬态、不是稳定吸引子——没有可回的间期态。瓶颈=衬底（缺稳定间期
  吸引子），不是退出机制（H 已是干净终止器）。**

**最终分层完成度（seed 1）**：Entry ✓ / Containment ✓ / **Termination ✓（H 干净终止器=本线正面新增）** / **Recovery ✗
（根本瓶颈=衬底无稳定间期吸引子）** / spatial+cross-seed 未做（recovery 先卡）。**能写**：H 是 divisive containment
memory、是比减法电流更干净的 M4 终止器；recovery 腿在减法/H/快放全设定下都不落间期（静默或复燃）。**不能写**：lifecycle
candidate；"证明衬底绝无 recovery"（只 seed 1、未穷举 θ_p/独立节律器）；任何 clinical/发作机制。**下一杠杆（待用户）**：
θ_p 持续时间门控（但静默是真 rate=0、可能不够）/ 给衬底独立间期节律器（让 recovery 有对象）/ 非对称 τ_H 让 H 在 p+H
也建得起来。产物 `phase3_containment_memory/{figures/{arms_pH,arms_Honly,arms_pfast}+README, phase3_verdict.json}`。
