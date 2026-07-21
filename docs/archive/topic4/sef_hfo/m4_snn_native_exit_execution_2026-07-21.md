# SNN-native M4 containment-to-exit lifecycle — execution log (2026-07-21)

> **Status: CLOSED — BOUNDED-NEGATIVE (temporal lifecycle).** Stage 0–2 done (open-loop exit atlas + dynamic
> arms symmetric/asymmetric + ablations, seed1 exhaustive; seed3 cross-seed confirmation running). Verdict §8.
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
- **已测的动态电流也退不出去（负结果）**：对称版被负反馈压到一个更低的持续水平（不终止）、或饿死 `S_G` 失控。
  （⚠️ 原"不对称版=周期性失控脉冲"因 P0 bug 实为对称 `τ_p=3000`、已作废；真·不对称重跑中，见 §7。）
  **候选共同解释（非 impossibility proof）**：`q_I` 与 `S_G` 对活动响应**符号相反**、随周期负相关——压住活动灌 `q_I` 往往让 `S_G` 衰减，
  已测轨迹没进低活动间期 basin。但 `q_I` 有始终存在的恢复项、`S_G` 有有限记忆（实测有 ~403ms 重叠），且用的是空间均值 `q_I`（可能掩盖 core）——
  **退出 corridor 是否存在未测**（需 `q_core×S_G` 冻结 atlas）。**结论=已测 actuator 下的 BOUNDED-NEGATIVE，带一个待验证的候选机制**。

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
- **空间**：源空间帧显示电流把宽持续态推成一个**大团四处游走**（压这里→那里冒→漂移），非局灶起始、非轴向招募、非终止波前。
  即"退出的障碍是灶/衬底的自发再点火 + 活动可空间搬家"，不是恢复坐标不够。图 `results/.../stage2_arms/figures/{arms_s2d_seed1.png, spatial_D_tau5000_eta40.png}`。
- **修正尝试（asymmetric `p`，running）**：快充 τ_p=3000 + 慢放 τ_p_down=12000 —— R4 active-low latch 的连续版：终止后 `p` 慢衰 → 长 hold 让 `q_I` 灌满 → 缓释放。判决待填。

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
每 row 落 `cfg_effective` 全量有效配置（不再只靠 argv）；加 `test_d_sweep_propagates_tau_p_down` 回归测试（修复前会失败）。
**真·不对称（τ_up=3000 / τ_down=12000, η_r=80/150）已用修复代码重跑（`s2dasymFIX`），结果待折入。**

注：那批（实为对称 `τ_p=3000` 强 `η_r`）确实给出 runaway（train_then_runaway / one_shot_burst），可作为"快对称+强电流→runaway"的数据点，
但**不是**不对称 hold 的检验。

## 8. Verdict —— BOUNDED-NEGATIVE（已测 actuator 下 temporal lifecycle 未达成；候选机制待验证）

**一句话**：在**已测的**恢复 actuator（开环抬阈值 hold + 对称闭环电流；真·不对称电流重跑中）下，都**没能**把 M4 有界持续态干净地
终止并恢复回间期——得到 rebound / 更低率持续态 / runaway pulse train。这是一个可信的**负结果**；一个候选的共同机制解释见下，
但**尚未证明"不存在退出轨迹"**。

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
- exit / dynamic accessibility：❌（已测：开环 hold + 对称电流；真·不对称重跑中）。
- termination / recovery：❌（无 clean terminate；无回间期）。
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
- `figures/fig5_no_go_diagnostic.png` —— 结论图：四列结局（persist / open-loop rebound / symmetric fragment / asymmetric runaway）+ `q_I`/`S_G` 反相。
- `stage1_exit_atlas/figures/exit_atlas_s1_seed1.png`（短压 500/3000/6000）+ `exit_atlas_s1long_seed1.png`（长压 10000/14000）。
- `stage2_arms/figures/{arms_s2cal_seed1.png（A/B/C 校准）, arms_s2d_seed1.png（对称扫描）, arms_s2dasym_seed1.png（不对称）, spatial_D_tau5000_eta40.png（游走）}`。

**Summary JSON**：`stage1_exit_atlas/exit_atlas_s1{,long}_seed1.json`、`stage2_arms/arms_s2{cal,d,dasym,abl}_seed1.json`（raw traces `.npz` gitignored）。

**脚本**：`scripts/{run_m4_snn_native_exit.py（exit_atlas 探针 + arms 动态臂 + d_sweep）, plot_m4_snn_native_exit.py, plot_m4_snn_native_exit_spatial.py, plot_m4_snn_native_exit_summary.py, topic4_resource_monitor.py}`；引擎 `src/snn_engine/slow_field.py`（`p` 场）；测试 `tests/test_m4_snn_native_exit_persist.py`（9 绿 + `BASELINE_SHA` 绿）。

**spec / archive**：`docs/superpowers/specs/2026-07-21-topic4-m4-snn-native-exit-design.md`；本文件。

**Stage-4 空间**：源空间活动帧（`spatial_D_tau5000_eta40.png`）显示恢复电流把宽持续态推成一个**四处游走的大团**（压这里→那里冒→漂移），
非局灶起始、非渐进招募、非终止波前——即"电流把活动空间搬家，不是熄灭它"。因无 lifecycle candidate，**未跑昂贵的 Stage-3 自发长轨迹**（cheap-first 纪律），
也无 paper-ready Figure 5；Figure 5 落为诚实的 NO-GO 诊断图。

**cross-seed（2-seed 一致）**：seed1 穷举（open-loop 5 holds + 对称 3 + 不对称 2 + 消融 2）；seed3 复核 `arms_s2d_s3b_seed3.json`（对称：D_tau5000_eta40=fragment、
D_tau8000_eta80=runaway）+ `arms_s2dasym_s3_seed3.json`（不对称 D_tau3000_eta150=train_then_runaway）。**两 seed 每个动态臂都落 {fragment, runaway}、无一 clean terminate**；
fragment↔runaway 边界随 seed 微移（S_G-饿死阈值 seed-敏感）。失败根因（`q_I`↔`S_G` 反相）是 M4 设计的**结构属性、与 seed 无关**，seed3 作稳健性复核确认。
（注：`s2d_s3` 首次 detached 跑因 shell 退出 SIGHUP 死于写 JSON 前 → 已 harness-tracked 重跑为 `s2d_s3b`。）

**资源**：worker 峰值 ≤8（多为 2–4），canary peak RSS 7.12GB，全程 `min mem_avail_frac=0.85`、**swap 增长 0**、未触发 protective stop、
不干扰并行 FCXR 线；`OMP=1`；`resource_log.jsonl` + `topic4_resource_monitor.py`。

**git**：worktree `.worktrees/topic4-m4-snn-native-exit`，branch `codex/topic4-m4-snn-native-exit`，base `4d40b03`（clean，未混入 R4/FCXR）；
commits `8109ccd`(spec) `d79d9f9`(persist 场+runner+tests) `69e431a`(Stage-1a 开环+diag) `c85e2ad`(asymmetric-p+machinery) `bb292e1`(verdict+diagnostics)（未 push、未合 main）。

## 7. Provenance

commits: `8109ccd`(spec) `d79d9f9`(persist field+runner+tests) `69e431a`(Stage-1a screen+diag)；
tests `tests/test_m4_snn_native_exit_persist.py` 8 绿 + `test_snn_gates.py` BASELINE_SHA 绿。
