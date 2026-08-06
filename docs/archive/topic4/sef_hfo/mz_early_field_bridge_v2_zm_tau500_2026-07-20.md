# MZ early-field bridge V2（z+m，τ_adp=500 ms）— 2026-07-20

分支：`codex/topic4-mz-early-bridge-v2`（local-only；从 onset-dynamics `3d5bc48` 起，cherry-pick V1 桥四提交后开发）
设计合同：`docs/superpowers/specs/2026-07-19-topic4-mz-early-field-bridge-design.md`（读出/统计合同，与 V1 共用）
结果根：`results/topic4_sef_hfo/mz_early_field_bridge_v2_zm_tau500/`
验收状态：**ACCEPTED / FROZEN（2026-07-20，用户签核）**——观测层桥，冻结于两层结论：**(A) 固定长轴在失控前被调用 = supported（3/3、LOO 稳健）；(B) 间期模板超越几何轴 = 未确立（LOO 后只 1/3 seed 稳健）**。完成层级：工程 ✔ + 数值复现 ✔ + 观测桥 ✔（(A) supported / (B) 未确立）；因果层**未完成**（§7）。验收合同见 §11。

## 摘要（朴素话）

**测了什么。** 一块固定的模型脑组织，安静期自发冒出一串小事件；把每个事件里各虚拟电极触点"谁先亮、谁后亮"排出来，得到一条双向的先后顺序轴。问题是：给每个兴奋神经元加上一个"刚发放完就自我压一下、然后较快松开"的快速适应机制之后，这条**用安静期排出的先后顺序**，还能不能预测这块组织滑向失控性爆发之前那一小段时间里、各触点的能量高低分布。

**怎么测的。** 复用**同一块组织、同一批安静期模板**，只把中间那段自然演化从"只有去抑制"换成"去抑制＋快速适应"。先确认新版轨迹能**精确复现**之前单独记录的失控时刻——三个噪声实现都差 **0 毫秒**，说明快速适应确实进了仿真、加装虚拟电极也没改变动力学。再取失控前那段窗口（识别到的招募起点后 0–50 ms）的触点能量，跟安静期先后顺序做**方向无关**的相关（A→B、B→A 两个方向都试、取较强的那个，叫 maxAB），然后把触点在**各自电极杆内部**随机重排一万次，看真实相关是不是明显超过随机。

**揭示了什么（分两层，见 §10 审阅修订）。** 三个噪声实现都：加了快速适应之后，失控时刻从约 9.3–9.8 秒
**推后到约 11–13 秒**；失控前 0–50 ms 的早期能量场沿 E1146 长轴排布。关键对照是：**单凭触点的固定长轴坐标
本身，就已经能显著预测这个早期能量场（3/3，maxAB 0.79/0.84/0.78，p<0.01）**。安静期先后顺序模板**几乎就是
这条长轴本身**（与长轴坐标相关 −0.95～−0.98），只在长轴之上多贡献一点点（maxAB +0.10～0.13）；控制长轴之后，
模板对能量的残余关联全样本看 2/3 显著，但 **leave-one-contact-out（掉任一触点）后只 1/3 稳健**（seed3 最坏
p=0.022；seed1 的显著完全由单个触点撑着、掉一个即到 p=0.203；seed4 本就边缘）。而且这条读出是**早期现象**：
0–50 ms 强，到 50–100 ms 减弱（seed1 失去显著、seed3 边缘）。所以最安全的说法要分两层：
**(A) 失控前早期能量场沿固定病理长轴组织、并在失控前被调用——这一层 3/3 成立、LOO 稳健；(B) 间期事件的细粒度
时序是否在几何长轴之上再提供预测信息——LOO 后只 1/3 seed（seed3）稳健、其余脆弱，(B) 未确立、至多单-seed 线索。**
加入快速适应**主要推后了点火时刻，但不只
改变点火时刻**：它同时改变了早期场的动态范围、招募规模、以及跨尺度（source-grid）表达（见 §5/§10）。这是一块
组织 × 三个噪声实现的**观测层**结果，不是队列级、也不是因果结论（因果需逐位可续跑的状态分叉证据，本轮没有）。

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
- **source-grid（supplementary）**：2/3 过 toroidal null（seed1 marginal p=0.069）。**不写成"方向不矛盾"**：seed1 的 source 名义 winner 是 A_to_B，与 contact 的 B_to_A **不同**；且 source maxAB 三个 seed 相对 V1 **全部下降**（0.651→0.573、0.546→0.498、0.585→0.535）。准确口径：source-grid 在 2/3 seed 出现方向无关的正轴匹配，但 seed1 不显著、名义方向与 contact 不同——**尚无稳定的跨尺度方向一致性**。只作补充诊断，不与 contact 合并。
- **core-exclusion**：三个 seed `n_kept=15`（一个触点都没删）→ **uninformative**；**不能**下"不依赖核心区域"的结论（与 V1 相同）。
- **local participation（V2-only，真值）**：**不是简单的"混合"**。逐 contact 看，energy 与局部参与度的 Spearman = **0.903 / 0.914 / 0.941**；每 seed 最热三触点附近参与度约 0.89–1.00、最冷三触点约 0（median 0.287/0.171/0.355 只因触点在轴上冷热各半被拉到中间）。所以**高能量触点确实伴随明显的局部神经元招募**；但 energy 与 participation **都沿长轴变化**，**不能证明 contact 信号由局部组织独占产生**——既不写"contact 热点=局部源"，也不写成"局部参与混合/不支持"。V1 是 readout-only patch、无 raster、无可配对基线（V2-only）。

## 5. V1 z-only ↔ V2 z+m 配对（按 seed，`v1_vs_v2_comparison.{json,csv}`）

| seed | V1 maxAB (p) | V2 maxAB (p) | Δ maxAB | Δ t120 | same winner |
|---|---|---|---|---|---|
| 1 | 0.945 (0.0004) | 0.891 (0.0027) | −0.054 | +3663 ms | 是 (B_to_A) |
| 3 | 0.735 (0.086 **未过线**) | 0.969 (0.0001) | +0.234 | +1509 ms | 是 (B_to_A) |
| 4 | 0.924 (0.0010) | 0.904 (0.0017) | −0.020 | +2919 ms | 是 (B_to_A) |

判读（对齐 task §10 的问题）：
- **加了 m 后 contact 级轴匹配是否保留？** 是——3/3 seed contact maxAB 仍显著。但注意 winner 都是 B→A、A→B 模板 held-out 稳定性弱（§10），**winner 一致不作为"双向轴被稳定使用"的证据**。
- **m 改变了什么？** **不只推后点火时刻。** 它推后了点火（+1.5～+3.7 s）、并保留最早 0–50 ms 的 contact 级轴匹配；但它**也**改变了早期场的**动态范围**（seed3 dynamic_range 3.88→22.06、seed4 5.15→29.88）、**招募规模**（seed3 recruited 0→5）、以及**跨尺度表达**（source maxAB 三个 seed **全部下降**）。所以"m 主要只改变点火时间"是**不准确的**。
- **seed3 的弱 contact result 是否仍在？** **不在**——V1 里 seed3 是唯一未过线的（0.735，p=0.086），V2 里 seed3 反而最强（0.969，p=0.0001）；这与 seed3 招募规模/动态范围的大幅上升同步。
- **contact 与 local participation 是否一致？** 一致方向上是——高能量触点局部参与度明显（energy-participation Spearman 0.90–0.94，§10/§4）；但两者都沿长轴变化，不能据此说 contact 信号由局部独占产生。
- 统计单位只有 3 个噪声实现；V1+V2 六次运行**不是**六个独立样本，9 格/6 次不做队列 p 值。

## 6. 图

- 主图（Figure-5 语法，seed1）：`results/paper-ready-figure/fig_mz_early_bridge_v2_zm_tau500/figures/fig_mz_early_bridge_v2_zm_tau500.{png,pdf}` + metadata + 中文 README（已目检）。一条连续 z+m native Virtual-SEEG 轨迹 + 蓝(TB event) + 粉(pre-t120 早窗) + 红虚线(t120)；下排两张场（event-order viridis + early-energy Blues）沿 E1146 长轴同向。只读 V2 artifact；灰点=固定 E-neuron 几何、不表示局部招募。
- 三 seed 配对诊断图（**非主图**）：`.../fig_mz_v1_v2_paired_diagnostic.{png,pdf}`（左 maxAB V1 vs V2 + within-shaft 星号；右 t120 V1 vs V2）。已目检。

## 7. 完成层级（分开报告）

- **engineering complete**：fixed-bar detector 复用、reuse fail-closed 合同 + 测试、`--output-dir` 隔离、resumable artifacts、V1 不被覆盖。✔
- **numerically eligible**：三个 seed held-out 双向模板 eligible、pre-t120 窗完整非退化、t120 gate Δ0.0。✔
- **scientific observation（本轮上限）**：方向/效应量/nulls/seed 一致性都按合同报告（不论正负）。✔ 结论：观测层桥**supported**。
- **causal mechanism：未完成。** CRN replay 不是 checkpoint 后的真实状态分叉，无法区分"整体去抑制增益"与"局部 z 图案"，也不能说间期事件因果触发失控。需逐位可续跑的 snapshot/resume + native/uniform/shuffle/reset z 对照（design §11.2）——本轮明确未做。

## 8. 声明边界（design §15 + 2026-07-20 审阅）

可以写：固定患者布局支架上，失控前早期虚拟触点能量场**沿固定长轴组织、并在失控前被调用**（**单凭触点长轴坐标即可预测，3/3**；加入快速适应 m、推后点火后仍成立）——观测层"同一支架、状态依赖读出"的可行性桥。
**不可以写**：临床发作 / 临床宽带功率 / 完整发作循环 / m 稳住发作或产生恢复态 / **m "只改变点火时间"**（它也改招募/动态范围/跨尺度，§5/§10）/ `z_i` 唯一生物机制 / 某端固定发作灶 / 间期事件因果触发失控 / 局部 z 图案有因果作用 / **contact 热点＝局部独占源**（§10.1）/ 结果不依赖 core（uninformative）/ **间期时序模板在几何长轴之上提供了预测信息**（LOO 后只 1/3 seed 稳健、未确立，§10.1）/ **双向轴都被稳定使用**（三个都 B→A、A→B held-out 弱，§10.3）/ source 跨尺度方向一致（seed1 名义方向与 contact 不同、且全部下降，§4）。

## 9. Provenance / 复现

runner：`scripts/run_topic4_mz_early_field_bridge.py --confirm-run --config config/topic4_mz_early_field_bridge_v2_zm.yaml --output-dir results/topic4_sef_hfo/mz_early_field_bridge_v2_zm_tau500 --seeds 1,3,4`
per-seed `bridge_metrics.json` 携带 git_sha + 6 engine_shas + candidate + T；`provenance.json`（cohort）记录 per-seed producer SHA。大 `*.npz`（LFP/raster，各 ~8–10 MB）不进 git（清单见 `LARGE_ARTIFACTS_MANIFEST.txt`），路径与内容可由上面命令确定性重生。

## 10. 2026-07-20 审阅修订（长轴几何对照 / 时间局限 / 口径更正）

审阅指出并经复核确认（均基于已提交 artifact，**未重跑 SNN**；分析 `scripts/paper_figures/analyze_mz_v2_axis_and_temporal.py`，结果 `axis_and_temporal_control.json`，补充图 `fig_mz_v2_axis_temporal_supp.{png,pdf}`）：

### 10.1 长轴几何对照（最重要）
把触点的**固定长轴坐标本身**当模板，做同样的双向 maxAB + within-shaft null：

| seed | 间期模板 maxAB | **长轴-only maxAB (p)** | 模板−长轴 | corr(模板,长轴) | 残余 partial(模板,E\|轴) | **LOO 最坏 p** |
|---|---|---|---|---|---|---|
| 1 | 0.891 | **0.789 (p=0.0048)** | +0.102 | −0.954 | r=−0.554, p=0.032 | **0.203（脆弱）** |
| 3 | 0.969 | **0.839 (p=0.0009)** | +0.130 | −0.965 | r=−0.780, p=0.001 | **0.022（稳健）** |
| 4 | 0.904 | **0.779 (p=0.0063)** | +0.126 | −0.977 | r=−0.486, p=0.078 | 0.209 |

- 单凭长轴几何就 **3/3 显著**预测早期能量场；间期模板**几乎就是长轴**（|corr|>0.95），只多贡献 +0.10～0.13。
- 控制长轴后，残余关联**全样本**看 2/3 显著；但 **leave-one-contact-out（LOO）后只 1/3 稳健**：seed3 掉任一触点仍显著（最坏 p=0.022），**seed1 的显著完全由单个触点撑着**（掉一个就到 p=0.203），seed4 本就边缘。
- 结论分两层：**(A) 固定病理长轴在 runaway 前被调用 → 3/3 成立、LOO 稳健**（主结论就是这一层）；**(B) 间期细粒度时序超越几何轴的额外信息 → LOO 后只 1/3 seed（seed3）稳健，其余单触点驱动或边缘 → (B) 未确立，至多单-seed 线索。** 最小补法已做齐（axis-only / 增量 / 残余 partial / LOO）。

### 10.2 时间局限
contact maxAB 在 0–25 / 25–50 ms 都强，到 **50–100 ms 减弱**：seed1 p=0.133（失去显著）、seed3 p=0.054（边缘）、seed4 p=0.013；同时 local participation 继续上升。更符合数据的说法：**最初 50 ms 先出现刻板的 contact 级轴读出，随后局部招募扩大、但间期模板与能量场的精确排序关系减弱**。这条桥是**早期（0–50 ms）现象**。

### 10.3 双向模板 held-out 不对称
contact held-out median：seed1 A→B **0.361**（含负分）vs B→A 0.995；seed4 A→B 0.743 vs B→A 1.0；seed3 两向都强但 A→B 含负分。当前 eligibility 只查事件数/共享触点/非退化，**不查 held-out 中位数或符号一致性**，故"两向都够数据进 maxAB"≠"两向同样稳定"。三个 V2 早期场都由 **B→A** 胜出——本轮**只**验证了 B→A 分支；**winner 一致不列为成功证据**。

### 10.4 工程口径同批修复
- **t120 gate 现 fail-closed**：gate 失败 → seed 状态 `preflight_gate_failed`、被 aggregate 排除（本轮 3/3 pass，不影响结果，但符合 fail-closed 声明）。
- **--resume fingerprint** 去掉整仓 git HEAD，改为 engine SHA + candidate + T + slowoff_T + seed + schema：提交 results/docs 不再让相同模型/config 的 seed 被判不可 resume。
- **cohort provenance** 记录 per-seed producer SHA（seed1=`7951052`，seed3/4=`1719a67`）。
- **`rotated_90`** 更名 `_rotate90_coarse_field`；docstring 说明它是**粗场旋转**（每格均值→旋转→映射回，格内神经元数不等时不保持神经元级 z 分布），**不是严格 state-matched 因果对照**。
- **主图标签**：顶排 "Virtual-LFP (30–80 Hz)"、两张场加 "(contact readout)"，杜绝误读为真实 SEEG / 局部源。

## 11. 验收（ACCEPTED / FROZEN 2026-07-20）

本条 MZ early-field bridge V2 经 2026-07-20 审阅（长轴几何对照 + 时间局限 + 工程口径）修订后**正式验收、冻结**。验收基于以下承重主张各有数值门（gate）+ 坏数据回归（测试），审阅项全部落实。

**冻结结论（就是这两层，不多不少）**
- **(A) 固定长轴在失控前被调用 = SUPPORTED。** gate：axis-only maxAB > within-shaft null，3/3（0.789 / 0.839 / 0.779，p<0.01）、LOO 稳健。
- **(B) 间期模板超越几何长轴的额外信息 = 未确立。** gate：控制长轴后残余 partial，全样本 2/3 显著、**LOO 后只 1/3 稳健**（seed3）；seed1 单触点驱动、seed4 边缘 → 至多单-seed 线索，**不写成"确立 / 部分成立"**。
- **数值复现**：native `t120` delta = 0.0 ms，3/3（gate：|t120 − committed onset| ≤ 1 ms，fail-closed）。
- **时间范围**：早期 0–50 ms 现象；50–100 ms 减弱（gate：window within-shaft p）。
- **方向**：只 B→A 分支被验证（A→B held-out 弱）；winner 一致不作证据。
- **m 效应**：推后点火 + 保留 0–50 ms 轴匹配，同时改招募 / 动态范围 / 跨尺度——非"只改点火"。

**验收清单（审阅项 → 状态）**
- 长轴几何对照（axis-only / 增量 / 残余 partial / LOO）✔（§10.1）
- 时间演化补充图 ✔（§10.2 / `fig_mz_v2_axis_temporal_supp`）
- local participation 正确口径（Spearman 0.90–0.94、轴共变、不宣称独占）✔（§4）
- t120 gate fail-closed ✔ ｜ resume fingerprint 去 git-HEAD ✔ ｜ cohort per-seed producer SHA ✔（§10.4）
- rotated_90 → coarse-field 标注 ✔ ｜ 主图 virtual-LFP / contact-readout 标签 ✔（§10.4）
- source / held-out / m 文档口径更正 ✔（§4 / §5 / §10.3）
- 测试 53 绿、`git diff --check` 干净、worktree 干净、并行 worktree 未扰。

**冻结纪律**
- 本 V2 结论**不被后续覆盖**；结果图 / 指标 / 文档为 frozen 记录。
- **上限 = observational bridge**。要推进到 (B) 干净确立或 causal 需**新工作**：(B) 更大 seed / 轴保持 null / 真实解剖轴独立性；causal 需引擎 bit-identical 全状态 snapshot/resume + native/uniform/shuffle/reset-z 状态匹配对照（design §11.2 / §17）——不在本冻结内。
