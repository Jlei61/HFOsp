# Z/M ictal-carrier gate — archive (2026-07-24)

Branch `codex/topic4-m4-snn-native-exit`. Pre-registered design:
`docs/superpowers/specs/2026-07-24-topic4-zm-ictal-carrier-gate-design.md`.
This is the **Z/M-only** archive. The old q_I+S_G+p/H sandbox is quarantined in §7 as ARCHIVE-ONLY.

---

## 0. 朴素话摘要（测了什么 / 怎么测 / 揭示了什么）

**测了什么.** 我们有一张 2 万个兴奋细胞 + 8 千个抑制细胞的二维皮层片子（各向异性连接，来自病人
E1146 的电极几何）。每个兴奋细胞带两个"慢变量"：一个记录"抑制还剩多少劲"（z，反复放电会把它耗掉→
去抑制），一个记录"自己放电后有多累"（m，适应）。片子会自己冒出零星的"发作间期"放电事件；随着 z 被耗
掉，事件越来越密，最后冲进失控高频（runaway）。我们再加一个"全局分裂式抑制池"S_G，想把失控刹住，看它
会不会变成一段真正的"发作"——而且这段发作在虚拟电极（SEEG）上应该是**持续的高频能量**，而不是一串彼此
分开的尖峰。

**怎么测.** 三条臂，都在同一张片子、同一个随机种子（seed 1）上：只有 z/m（bare）、z/m + S_G（sg）、
z/m + S_G + 记忆刹车 H（sgh）。我们把每个细胞的电流按病人真实电极位置加权，读出 15 路虚拟 SEEG（就是
论文里 |I_E|+|I_I| 的 LFP 代理，10 kHz 采样）。然后问一个很具体的问题：核心区放电率在微爆发之间是**掉回
静息**还是**一直保持在高位**；电极上的 30–80 Hz 能量是**持续抬高**还是**一阵一阵回到基线**。判据在跑之前
就锁死（见 spec 的两层门 A 源空间 / B 电极观测）。

**揭示了什么（截至本文，Phase 0 完成，Phase 1 判决待填）.** 目前只有 seed 1 的旧诊断跑（还没上正式高频读
出）能说的是：sg 把失控刹成了一段**长时间反复、逐渐增强的核心爆发串**——全片平均放电率只有 ~4.79 Hz（像
发作间期水平），但核心峰值到 403.6 Hz，核心平均只有 30.8 Hz（即核心大部分时间是安静的、~7% 占空的爆发）。
**这既不是终止，也不能仅凭"全场平均低"就叫发作间期。** 它到底是"持续发作载体"还是"一串 HFO 样爆发"，
必须用锁死的两层门在正式高频读出上判，不能拍脑袋。（内部代号：Z/M lockpoint
`zA_q75_tz5000__mA0p001_tau500`，S_G α_G=16，arm `sg`。）

---

## 1. 底座（当前正确 substrate）

- E1146 `twoend_equal`，`PP.build_substrate(seed)`，L=20 mm，N=40000（NE=32000，NI=8000）。
- 慢变量：`use_z=True, use_m=True, use_qI=False, use_gK=False`（q_I 冻结在 1 → `z·q_I·I_I == z·I_I`，与
  canonical `mz_slow_vars.py` 逐位一致，`tests/test_zm_slow_field_parity.py`）。
- lockpoint `zA_q75_tz5000__mA0p001_tau500`：τ_z=5000, τ_adp=500, η_m=0.001, I_th_EI=q75(slow-off 间期
  E-cell I_I)=1.280（in-run 标定）。
- **不回 q_I / g_K。E→E 拓扑不动。** 本线独立问题：不改 EE 的前提下，抑制侧的空间反馈能否先形成一段持续的
  ictal carrier，再由慢变量退出。

## 2. seed-1 三臂实测（25 ms bin，来自 `results/topic4_sef_hfo/zm_snn_native_exit/*_seed1.npz`）

| arm | 时长 | 分类(旧 termination cls) | all-E mean | all-E peak | core mean | core peak | z_core终 | S_G max | H max |
|---|---|---|---|---|---|---|---|---|---|
| bare | 2.9 s (runaway 截断@2871.8ms) | runaway | 44.8 Hz | 270.6 | 113.8 | 431.2 | 0.702 | – | – |
| sg | 15.0 s | fragment | **4.79 Hz** | 86.6 | 30.8 | **403.6** | 0.343 | 0.159 | – |
| sgh | 25.0 s | fragment | **2.17 Hz** | 39.8 | 18.1 | 319.7 | 0.372 | 0.102 | **0.0349** |

**共性数字纠正（本文锁定口径，纠正早期 memory 混淆）：**
- **SG 是 15 s，不是 25 s**；25 s 是 SGH。
- **SG all-E mean 是 ~4.79 Hz**（不是 ~2.2 Hz）；~2.17 Hz 是 SGH。
- 全片低均值（4.79 / 2.17 Hz）**不能掩盖核心 100–400 Hz 的爆发**：core peak 403.6（sg）/ 319.7（sgh）。
  "全场平均像间期水平" ≠ "这是间期状态"。
- bare 的 runaway 由引擎 Hz 阈值触发，写进 `runaway_early_stop_ms=2871.8`（**不是** `runaway_ms`——后者
  在引擎里根本不存在；harness 读对了字段）。

## 3. 概念纠正（跑正式门之前先把话说清楚）

1. **sensor amplitude ≠ actuator gain ≠ effective load.** H 有三个不同的量：H 感到的输入 `phi_drive`
   （sensor 读数）、耦合强度 `alpha_H`（actuator gain）、真正进膜的负载 `alpha_H · H`（effective load）。
   sgh 的 `H_max=0.0349`、`alpha_H=16` → effective load ≤ 0.56。**"H_max 小" 不等于 "H 无效"**——要看
   effective load 相对分母 `1+alpha_G·S_G+alpha_H·H` 的占比，而不是 H 的绝对值。
2. **不是每个 burst 都叫 IED.** sg 的核心爆发是"S_G 反复把核心同步 reset → 再点火"的产物，是一段候选的
   ictal 内循环（candidate clonic-like inner cycle），不是发作间期离散事件。措辞用 "persistent focal
   recurrent burst train" / "candidate inner cycle"，不用 "IED train"。
3. **不声称旧 q_I substrate 证明了"不存在稳定间期吸引子".** 那是 q_I 沙盒的结论，与当前 Z/M 无关（§7）。
4. **H 的真实输入是 phi_drive，不是 p_max.** 旧诊断图把 `p_max`（p 场的空间最大值标量）标成 "H sensor in"
   是错的。真实驱动是 `phi_drive = mean(Phi(p) over cells where phi > 0.2·pmax)`（active-focus 均值）。
   Phase 0.2 已把 `phi_drive` 与 active-focus 占比正式 trace（`slow_field.py`，off-by-default byte-parity，
   `tests/test_zm_hdrive_diagnostics.py`）。**本线 Phase 1 H 冻结关闭**——H 的建立/终止是 Phase 2 的问题，
   不在此处下"H 建不起来"的结论。

## 4. 两个分类器分离（task §5.4）

旧 M4-2 termination 分类器的 `fragment` 标签描述"活动曲线形状"，**不能**承担"是否存在 ictal carrier"的判
定。新增两个互相独立、词表不重叠的 verdict（`src/topic4_zm_carrier_verdict.py`，合成 fixture 测试
`tests/test_topic4_zm_carrier_verdict.py`）：

- `ictal_carrier_verdict` ∈ {`fail_runaway`, `fail_plateau`, `fail_hfo_like_train`,
  `candidate_source_only`, `candidate_observed_carrier`}。
- `lifecycle_verdict` ∈ {`carrier_not_established`, `no_onset`, `prevention`, `persistent`,
  `terminate_to_silence`, `terminate_then_reignite`, `terminate_and_recover`}——**只有 carrier 通过才允许
  输出 lifecycle candidate**；否则返回 `carrier_not_established` 哨兵。

## 5. Phase 0 工程加固（commit 1）

- **H-drive 观测**：`slow_field.py` 新增 `trace_phi_drive`（真实 H 输入）、`trace_active_frac`（active-focus
  网格占比）、`trace_m_core_mean/surround`；均为观测，spike 输出不变（BASELINE_SHA=`da5fc18c27d5340a` 不动，
  Z/M parity 不破）。
- **verdict 分离** + **pre-registered carrier gate spec**（阈值锁死，跑前冻结）。

## 6. 下一步：Phase 1 carrier 门（正式高频读出）

复用引擎自带 `lfp_recorder=` 钩子（`kick_probe.py:291`，观测-only，不改动力学）+ E1146 15 触点 montage
`S["reg"]["montage_sheet"]`，把 LFP（10 kHz 采样、存 2 kHz、Nyquist 1 kHz > 150 Hz）读回。跑 `bare` / `sg` /
`interictal_ctrl`（H 全关，seed 1 先判），用锁死的两层门判 `ictal_carrier_verdict`。判决填入 §8。

## 7. 🔴 LEGACY q_I + S_G + p/H 沙盒 —— WRONG-SUBSTRATE / ARCHIVE-ONLY（不代表当前 Z/M 模型）

> 以下全部来自 **旧的 field-based q_I + S_G + p/H 模型**（`use_qI=True`），**不是** 当前锁定的 per-neuron
> Z/M。所有科学结论（"H 是干净终止器"、"没有可恢复的间期吸引子"、frozen atlas、t_form=1600、onset=2300）
> **只在 q_I 沙盒内成立**，不得当作 Z/M 结果读。细节见
> `docs/archive/topic4/sef_hfo/m4_snn_native_exit_execution_2026-07-21.md`（§1–§11）与
> memory `project_topic4_m4_snn_native_exit_2026-07-21`。可复用的只有：工程/方法（crash-safe resume、
> provenance、active-focus H sensor），以及 H 作为一个**候选**机制留待在 Z/M 上重测。

## 8. 判决（Phase 1 跑完后填 — commit 3）

_待填：seed-1 `bare` / `sg` 的 `ictal_carrier_verdict`（源空间门 A + 电极门 B 的具体数值）+ carrier 是
"持续高频能量" 还是 "HFO-like burst train" + Phase 2 分支选择（Path A exit atlas / Path B spatial
inhibitory carrier）。_
