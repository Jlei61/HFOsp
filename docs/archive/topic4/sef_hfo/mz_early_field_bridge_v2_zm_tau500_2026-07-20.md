# MZ early-field bridge V2（z+m，τ_adp=500 ms）— 2026-07-20

分支：`codex/topic4-mz-early-bridge-v2`（local-only；从 onset-dynamics `3d5bc48` 起，cherry-pick V1 桥四提交后开发）
设计合同：`docs/superpowers/specs/2026-07-19-topic4-mz-early-field-bridge-design.md`（读出/统计合同，与 V1 共用）
结果根：`results/topic4_sef_hfo/mz_early_field_bridge_v2_zm_tau500/`
完成层级：**observational bridge — supported**（工程 + 数值复现 + 观测桥都到位；因果层未完成，见 §7）

## 摘要（朴素话）

**测了什么。** 一块固定的模型脑组织，安静期自发冒出一串小事件；把每个事件里各虚拟电极触点"谁先亮、谁后亮"排出来，得到一条双向的先后顺序轴。问题是：给每个兴奋神经元加上一个"刚发放完就自我压一下、然后较快松开"的快速适应机制之后，这条**用安静期排出的先后顺序**，还能不能预测这块组织滑向失控性爆发之前那一小段时间里、各触点的能量高低分布。

**怎么测的。** 复用**同一块组织、同一批安静期模板**，只把中间那段自然演化从"只有去抑制"换成"去抑制＋快速适应"。先确认新版轨迹能**精确复现**之前单独记录的失控时刻——三个噪声实现都差 **0 毫秒**，说明快速适应确实进了仿真、加装虚拟电极也没改变动力学。再取失控前那段窗口（识别到的招募起点后 0–50 ms）的触点能量，跟安静期先后顺序做**方向无关**的相关（A→B、B→A 两个方向都试、取较强的那个，叫 maxAB），然后把触点在**各自电极杆内部**随机重排一万次，看真实相关是不是明显超过随机。

**揭示了什么。** 三个噪声实现都一致：加了快速适应之后，失控时刻从约 9.3–9.8 秒**推后到约 11–13 秒**；但那条安静期先后顺序**仍然预测**失控前的能量分布——相关中位数约 **0.90**（三个分别 0.89 / 0.97 / 0.90），杆内随机重排里都只有 0.01%–0.3% 能达到这么高（**三个都过随机线**），胜出方向一致。看起来这条轴在加了快速适应后被**完整保留**了，甚至比"只有去抑制"时更一致——"只有去抑制"那版里偏弱、没过随机线的那个实现，在加了快速适应后反而是最强的。安全结论只能到这一步：**加入快速适应变量、并推后了点火时刻之后，安静期时序轴对失控前早期能量场的预测仍然存在。** 这是一块组织 × 三个噪声实现的**观测层**结果，不是队列级结论；因果需要逐位可续跑的状态分叉证据，本轮没有。

（内部归档代号：candidate `zA_q75_tz5000__mA0p001_tau500`；`t120` / `t_recruit`；`rho_maxab` / within-shaft null；source-grid toroidal null；design §14 level-4 "bridge supported"。）

## 1. 冻结候选与 provenance

- label：`zA_q75_tz5000__mA0p001_tau500`
- `candidate.cfg`（只放引擎字段，`**cfg` 直接进 `MZSlowVarsConfig`）：`use_z=true, use_m=true, I_th_EI=95.19851312666987, tau_z=5000, tau_adp=500, eta_m=0.007451594355587098`
- `A_target=0.001` 是**推导标签**，不是引擎字段（不进 cfg，避免 `**cfg` splat TypeError）。
- `eta_m = A_target × I_EE_scale / peak_m_tau2000 = 0.001 × 272.75518960107513 / 36.6036014019694 = 0.007451594355587098`，从提交的 `results/topic4_sef_hfo/mz_slowvars/calibration.json` 逐位推导（**非手抄**；preflight 断言逐位相等）。注意：已提交的 tau500 onset run 用的是 **tau2000 归一化的 peak**（不是 `peak_m.tau500=13.1957`）——τ 敏感性刻意固定 eta_m、只变 tau_adp；若改用 tau500 峰值会得 0.02067、破坏 onset 复现＝改动冻结候选（禁止）。
- native `T=20000`（task/committed run 一致）；`slowoff_T=15000`（与 V1 相同 → 间期模板逐位相同 → 干净配对）。其余 detector/timing/split/onset/windows/nulls 与 V1 逐块相同。
- 引擎 6 个 guarded 文件与 blessed `engine_versions.json` 逐位相同；`mz_slow_vars.py` 未 guarded 且未改 → **无需 re-bless**。

## 2. Preflight（design §16 / task §6）

`preflight.json`：`use_m=True, eta_m=0.007451594355587098（==calibration-derived）, tau_adp=500`；适应轨迹 `adaptation_trace_absmax=0.0238`（非零）；`lfp_recorder_is_noop_on_rate=True`（带/不带虚拟电极记录器 `rate_E` 逐位相同）。
**per-seed t120 gate：三个 seed 全 PASS，delta=0.0 ms**（native `t120` = 12956.2 / 11008.6 / 12676.6，与提交 onset run 完全一致）。这同时证明 z+m 参数进入仿真 + 记录器不扰动动力学。

## 3. 复用（design §8 / task §8）— 本轮实际=fresh

新分支是 fresh git checkout，V1 的 `*.npz`（LFP/raster）**未被 git 跟踪、故不在本 worktree**，无法实际复用。因此本轮 slow-off **全部重新跑**（z/m-off，确定性 → 与 V1 逐位相同）。交叉核对：V2 slow-off 的 `n_returning`（seed1=38，与 V1 `templates.json` 一致）+ 端点方向计数 + held-out 模板 eligibility 复现 V1 → 间期模板确实相同，配对干净。
`--reuse-slowoff-root` + `verify_slowoff_reuse()` 仍作为 fail-closed 合同交付并测试（缺 artifact / 引擎 SHA / dt / 触点名+顺序+坐标 / 模板 eligibility 任一不符即 raise，无静默 fallback；见 `tests/test_topic4_mz_early_field_bridge_v2.py`）。

## 4. Per-seed 结果（primary window `early_0_50_ms`）

| seed | gate | t120 (ms) | contact maxAB | winner | within-shaft p | source maxAB | toroidal p | recruited | core-excl n_kept | LP median | n_ret |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PASS Δ0.0 | 12956.2 | 0.8915 | B_to_A | 0.0027 | 0.573 | 0.069 | 5 | 15 | 0.287 | 38 |
| 3 | PASS Δ0.0 | 11008.6 | 0.9691 | B_to_A | 0.0001 | 0.498 | 0.045 | 5 | 15 | 0.171 | 40 |
| 4 | PASS Δ0.0 | 12676.6 | 0.9043 | B_to_A | 0.0017 | 0.535 | 0.047 | 4 | 15 | 0.355 | 39 |

- **contact maxAB 三个 seed 全 eligible + 全过 within-shaft null**（median 0.9043，range [0.891, 0.969]，3/3 positive，胜出方向都是 B_to_A）。这满足 design §14 level-4 "bridge supported" 的诊断判据（≥2/3 held-out eligible + positive contact maxAB，source 不矛盾，无结果只靠 core loading）。**按 task 口径只报到 observational（level 3–4 诊断判据），不作队列级统计主张（n=3，一块组织）。**
- **source-grid（supplementary）**：2/3 过 toroidal null（seed1 marginal p=0.069），方向不矛盾（无相反方向）。比 V1（三个都过）略弱；仍只作方向无关轴调用的补充诊断，不与 contact 合并成"跨尺度同方向"。
- **core-exclusion**：三个 seed `n_kept=15`（一个触点都没删）→ **uninformative**；**不能**下"不依赖核心区域"的结论（与 V1 相同）。
- **local participation（V2-only，真值）**：median 0.287 / 0.171 / 0.355，per-contact 0–1 混合（部分触点附近神经元几乎都参与、部分几乎不参与）。所以高 contact energy **部分**对应局部招募、部分是远场读出——诚实的"混合"，不是纯局部也不是纯远场。V1 因为是 readout-only patch、没持久化 raster，**没有可配对的基线值**（V2-only）。

## 5. V1 z-only ↔ V2 z+m 配对（按 seed，`v1_vs_v2_comparison.{json,csv}`）

| seed | V1 maxAB (p) | V2 maxAB (p) | Δ maxAB | Δ t120 | same winner |
|---|---|---|---|---|---|
| 1 | 0.945 (0.0004) | 0.891 (0.0027) | −0.054 | +3663 ms | 是 (B_to_A) |
| 3 | 0.735 (0.086 **未过线**) | 0.969 (0.0001) | +0.234 | +1509 ms | 是 (B_to_A) |
| 4 | 0.924 (0.0010) | 0.904 (0.0017) | −0.020 | +2919 ms | 是 (B_to_A) |

判读（对齐 task §10 的问题）：
- **加了 m 后双向轴的可预测性是否保留？** 是——3/3 seed contact maxAB 仍显著，胜出方向不变。
- **m 改变了什么？** 主要是**点火时刻**（三个 seed 都推后 +1.5～+3.7 s）；早期场**强度**（recruited 4–5、dynamic range 有限）与**轴调用本身**大体保留，个别 seed 略弱、个别更强。
- **seed3 的弱 contact result 是否仍在？** **不在**——V1 里 seed3 是唯一未过线的（0.735，p=0.086），V2 里 seed3 反而最强（0.969，p=0.0001）。
- **contact 与 local participation 是否一致？** 部分一致（高能量触点里有一部分局部参与度高），但 median 中等，属混合。
- 统计单位只有 3 个噪声实现；V1+V2 六次运行**不是**六个独立样本，9 格/6 次不做队列 p 值。

## 6. 图

- 主图（Figure-5 语法，seed1）：`results/paper-ready-figure/fig_mz_early_bridge_v2_zm_tau500/figures/fig_mz_early_bridge_v2_zm_tau500.{png,pdf}` + metadata + 中文 README（已目检）。一条连续 z+m native Virtual-SEEG 轨迹 + 蓝(TB event) + 粉(pre-t120 早窗) + 红虚线(t120)；下排两张场（event-order viridis + early-energy Blues）沿 E1146 长轴同向。只读 V2 artifact；灰点=固定 E-neuron 几何、不表示局部招募。
- 三 seed 配对诊断图（**非主图**）：`.../fig_mz_v1_v2_paired_diagnostic.{png,pdf}`（左 maxAB V1 vs V2 + within-shaft 星号；右 t120 V1 vs V2）。已目检。

## 7. 完成层级（分开报告）

- **engineering complete**：fixed-bar detector 复用、reuse fail-closed 合同 + 测试、`--output-dir` 隔离、resumable artifacts、V1 不被覆盖。✔
- **numerically eligible**：三个 seed held-out 双向模板 eligible、pre-t120 窗完整非退化、t120 gate Δ0.0。✔
- **scientific observation（本轮上限）**：方向/效应量/nulls/seed 一致性都按合同报告（不论正负）。✔ 结论：观测层桥**supported**。
- **causal mechanism：未完成。** CRN replay 不是 checkpoint 后的真实状态分叉，无法区分"整体去抑制增益"与"局部 z 图案"，也不能说间期事件因果触发失控。需逐位可续跑的 snapshot/resume + native/uniform/shuffle/reset z 对照（design §11.2）——本轮明确未做。

## 8. 声明边界（design §15）

可以写：固定患者布局支架上，held-out 双向间期时序轴在三个噪声实现里预测了失控前虚拟触点能量的空间分布（加入快速适应 m、并推后点火后仍然成立）——观测层"同一支架、状态依赖读出"的可行性桥。
**不可以写**：临床发作 / 临床宽带功率 / 完整发作循环 / m 稳住了发作或产生恢复态 / `z_i` 是唯一生物机制 / 某一端是固定发作灶 / 间期事件因果触发失控 / 局部 z 图案有因果作用 / contact 热点＝局部神经元优先招募 / 结果不依赖 core（core-exclusion uninformative）。

## 9. Provenance / 复现

runner：`scripts/run_topic4_mz_early_field_bridge.py --confirm-run --config config/topic4_mz_early_field_bridge_v2_zm.yaml --output-dir results/topic4_sef_hfo/mz_early_field_bridge_v2_zm_tau500 --seeds 1,3,4`
per-seed `bridge_metrics.json` 携带 git_sha + 6 engine_shas + candidate + T；`provenance.json`（cohort）同。大 `*.npz`（LFP/raster，各 ~8–10 MB）不进 git，路径与内容可由上面命令确定性重生。
