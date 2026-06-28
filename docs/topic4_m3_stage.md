# Topic 4 · M3 阶段主文档（SEF-HFO 机制建模）

> 这是 Topic 4 建模程序里 **M3 阶段**的 stage 主文档（一个主文档 + A/B 两条分文档）。
> Topic 4 的 topic 级主文档仍是 `docs/topic4_sef_hfo.md`；本 doc 是它下面 M3 阶段的收口，
> 由 §13 历史索引回链。两条分文档：
> - **A 线（慢变量机制）**：`docs/archive/topic4/sef_hfo/m3a_stage_conclusion_2026-06-27.md`
> - **B 线（谱相图 / W-场）**：`docs/archive/topic4/sef_hfo/m3b_stage_conclusion_2026-06-28.md`
>
> 这是机制 **screen**，不是癫痫发作机制 **validation**。下面所有结论受 §"红线"约束。

## 0. 一句话判断

> 在一片带 E→E 各向异性 scaffold、中间埋了易激核的皮层薄片上：**慢变量（局部抑制资源耗竭 + 恢复项）
> 能把小的局部事件推成更大的、沿轴的相干招募波**（A 线，源空间 onset 梯度证实）；**而把同一块组织
> 线性化，核扰动也确实会被非正规瞬态放大、沿同一 E→E 轴铺开再自限**（B 线，§5 主读法）。两条线在
> "轴向自限招募/传播"这一点上互相印证。但**"大范围仍沿轴" 只能算 expanded axial recruitment，
> 还没破开轴向 scaffold 成为离轴/全局发作样招募**——所以 M3 整体停在 **机制 screen 通过、发作机制
> 未 validate**。

## 1. M3 是怎么来的：M0→M1→M2 的一致性

M3 不是凭空转向。前面三个子阶段指向同一件事——**在当前读出口径、现有连接强度、均质（flat-threshold）
衬底下，"空间自限"很难不靠"把活动压死"来实现**：

- **M0**（边界审计）：减小兴奋连接的空间核 / 连接数，没有打开"可读 + 空间自限"的窗口；mean-field
  lever 被耗尽（lever-exhausted hard block）。
- **M1**（E→E 短期抑制 + 恢复）：给每个兴奋神经元加"用进废退"恢复变量。扫了 60 组参数（U×τ×2 seed）。
  结果**时间上自限成立**（事件自己平息、不持续烧、占空比~5%、runaway=0），**但空间上不成立**：凡是
  能读出方向的事件，放电范围都铺满整片（L=20 可读 r95~13mm reach_axis~24mm 溢出边界；L=32 也铺满，
  margin −4.5mm）。两个诊断探针排除了"小片逼出来"和"读不出来"的廉价解释 → NULL 是真的。
  口径锁（**不要在 M3 doc 里违反**）：scope 到"当前 Stage-3 工作点 + 4mm-pitch ≥7-触点读出 + 扫的
  U/τ/l_EE/C_EE"，**禁** "E→E STD 只给时间不给空间自限"这种绝对话（montage 没改）。
  归档：`m1_recovery_stage2_NULL_2026-06-18.md`（committed on branch `topic4-snn-m1-recovery`）。
- **M2**（前沿抑制刹车）：给网络装"波前方的抑制先点亮、提前布防"的刹车（癫痫文献成熟的 surround
  inhibition 概念），并把抑制做成把电压钳向静息。**一维平均率模型里有效**（波传一小段就停、尺度固定
  L-invariant）；**搬到放电网络就失败**——刹车只压**事件率**不压**传播 reach**（gate 0→0.6：率峰掉
  50×，但 max_reach 几乎不变）。机制诊断：ahead-recruit 腿装上了（抑制在 80% 轴向 bin 里领先 E 前沿、
  中位领先 6.55ms），但**钳制腿基本缺席**（front-cell 抑制电流太小，conductance 钳不住，
  frac_axial_gated_by_shunt 仅 2–3%）。
  关键框定：**M2 ≠ Liou & Abbott 2020 完整机制**——M2 只借了"抑制约束"的概念、并加了更强的**均质**刹车，
  方向恰恰**相反**于 Liou/Abbott（他们的传播机制是抑制在某处**耗竭/失效**，usage-dependent inhibition
  exhaustion）。口径锁：scope-limited NULL（g=0.25 shunting、固定宽核、drive 0.6、L=20）；**禁**
  "前沿抑制不可能 / Liou-Abbott 被证伪 / 空间自限被证伪"。
  归档：`m2_shunting_gate_result_2026-06-19.md`（committed on branch `topic4-snn-m1-recovery`）；
  数据侧 Task-0 事件展布审计（`src/topic4_event_extent_audit.py`，n=23 真 SEEG）已在主线当前分支。

**这条一致性正是 M3 转向"抑制约束的空间异质性失效"（Liou-direction）的依据**：既然均质衬底 + 当前
lever 给不出"可读 + 空间自限"，就让**慢变量在异质核上造出许可度差**（A 线），并**用线性算子谱直接
问这块异质组织天生倾向哪种空间模式**（B 线）。

## 2. A 线 · 慢变量机制（→ 分文档 m3a_stage_conclusion）

**测了什么** — 给组织装一个"慢慢变"的旋钮（抑制变弱 / GABA 反转电位去极化 / 阈值平移 / 钾电流外流 /
区域抑制油箱随放电耗竭），网络**自己（不外部戳）**会不会从"零星小放电、扩一点就回静"（间期样）滑进
"大范围发作样"再滑回来。全程不用 W、不用 h(W)（旧的 V_th 随 W 降只提高事件率、不改事件大小/时长/类别，
被当反面证据）。

**骨架（各级一句话）**：

- **A1a 准静态·均质衬底 = 有界负面**：均质组织根本没有"间期基线"可锚（旋钮放正常位时一个事件都没有、
  纯 R0、全程无 R4a）。这本身印证 M3 初衷：同质组织产生不了"间期 vs 发作"的区别。
- **A1b 静态·双焦核衬底 = 状态地形图**：发作样"大但回静"态落在 **局部回路强度 × 全局抑制强度** 对角中段
  （全局抑制是主轴）。口径收紧：`seizure_like` 是旧 screen 标签，若源空间仍沿轴推进应叫
  `expanded axial recruitment`，不是发作态成立的证据。
- **A1c 动态全局反馈·pilot = 均匀刹车治不了核集中型失控**：没有一个增益既终止强核失控、又不把弱工作态
  压成静默（终止是时序特异的，但毛病出在空间均匀）。
- **A2 Abbott 局部+全局动态资源耗竭**：单靠"抑制油箱"耗竭只给"要么停要么失控"的**尖锐双稳**（无中间区，
  只进不退的正反馈）；**加一个钾电流 sAHP 刹车**在一个很窄窗口给出慢-快**脉冲/弛豫振荡候选**（种子脆弱）。

**A 线的决定性纠正（A2-P，2026-06-26 重开）** — **源空间逐细胞 onset 梯度推翻了"同步爆发不传播"的
判读**：把每个 E 细胞的首发放电时间画在源空间，高许可度大态**不是同步共燃、而是一条相干单源招募波**，
沿两核轴平滑推进（**onset~位置回归 R²≈0.87、梯度沿轴 align≈1.0、方向可读率 1.0**；40000 神经元、
L=20mm SNN）。之前两次判 FAIL 用错了仪器：(1) 触点空间方向可读性~0.12 是因为 12 触点里 <7 个活跃（太稀，
读不出方向 ≠ 没方向）；(2) 双核 collision=0 对**单源有向扫掠**本就是正确值（collision 只数两核各自点燃
相撞），不是"不传播"的证据。

> **A 线当前能写**：慢变量提供状态迁移机制假说——局部资源耗竭提高有效许可度、恢复项给回落力，使事件能从
> 小局部轴向传播变成更大的轴向相干招募波、并可能回落。**不能写**："已证明慢变量导致发作 / A2 已过间期-发作
> 两态 / 大范围沿轴 recruitment = 发作 / 连接 scaffold 从各向异性变各向同性"。Gate A（可投到 B 线相图的
> 慢状态轨迹）可以先过，**不能自动升级成 Gate B（seizure-like phenotype）**。

**→ M3A-v2（空间慢变量场，2026-06-28 计划锁定）** — 把 v1 的两个**标量**油箱
（`q_core`·`q_global`）升级成**空间场** $q_I(x,t)$（抑制资源）+ $g_K(x,t)$（疲劳/恢复）。动机：两个全局标量
没有**空间历史**，"轴向疲劳的同时周边许可度上升 → 破轴" 这件事结构上承载不了（标量去抑制只会继续**加强**轴向）；
空间场给每个位置自己的慢状态，让破轴**可被表示、可被检出**。公式见 `docs/snn_core_model_equations.md §B5`，
实现计划 + red-TDD 骨架见 §6。仍是 **screen**：破轴是否真发生是经验问题，移交延后的 ablation。

## 3. B 线 · 谱相图 / W-场（→ 分文档 m3b_stage_conclusion）

**测了什么** — 把这块带核薄片**线性化**，算出它天生最先放大哪些空间本征模式，扫"核兴奋度 × 全局去抑制"
相图，并按计划 §5 用**非正规瞬态**正确读轴向。

**结论 = SPM-PASS frozen map**（线性算子层面）：

- **主导本征模式是全局的**（齐次色散最高在 k=0、没有有限-k 峰）——所以**只看"最先持续长大的模式"会误判
  成没有轴向**。
- **§5 主读法（非正规瞬态）才是对的**：给核一个扰动，瞬态增益在 ~10ms 冲到约 2 倍，沿 E→E 轴的拉伸峰更靠后
  （~30ms、max≈0.45），随后增益与轴向都衰减（**自限**）。在 8/8 个未饱和相图点全出现，且**方向骨架特异**
  （AR1 各向同性对照轴向≈0；放大本身通用、轴向才挑骨架）。
- **谱增长率 α₁ ≠ 非线性饱和**：相图里 runaway 格全是工作点非线性饱和（op saturated，α₁ 实际负≈−0.05），
  不是 α₁>0 线性失稳。
- 工程闸 fail-closed：frozen-map 需 `controls_pass AND non_normal_axial_pass`；三道桥（SNN/M3A overlay/
  几何零模型）各有各的闸，缺一封顶。M3A overlay = `refused`（5 产物缺失）；读出 = `projection_only`
  （几何零模型 not_run，非 failed）。

**B 线的 SNN 口径（已纠正）** — B 线内置的 tiny-SNN 抽查（L=0.5mm、~500 神经元 + 放电空间拉伸）是
**错仪器、不作数**，它的"招募轴向≈0"是小片伪影，**绝不能读成"轴向不在 spiking 层复现"**。轴向在 spiking
层的正确验证就是 **A 线 A2-P 的源空间 onset 梯度（40000 神经元）= 轴向招募波**。所以 §5 线性轴向与 SNN
**一致**；B 线停在 frozen-map 是因为**它本身是线性算子结果**，不是因为 spiking 失败。

## 4. 方法学锁（M3 全程，承重）

> **判"慢变量/算子有没有改传播"的正确仪器 = 源空间逐细胞 onset 梯度**（per-cell 首发放电时间对位置回归的
> R² + 梯度沿轴 alignment），**不是**触点空间方向可读性、**不是**双核 collision、**不是**放电空间拉伸、
> **不是**只看主导本征模式。错仪器（小片 + 接触空间 + collision + elongation）会把一条有向招募波误判成
> "同步爆发 / 无传播 / 无轴向"。

## 5. M3 整体能写 / 不能写

**能写**：

- A+B 两条独立路线在"E→E scaffold 上存在轴向自限招募/传播"这一点上互相印证（A=源空间 onset 梯度的招募波；
  B=线性算子非正规瞬态的自限轴向放大，骨架特异）。
- 慢变量给出状态迁移机制假说（局部耗竭推、恢复项回落）；线性谱给出对应的机制地图（核扰动优先激发轴向模式）。

**不能写**（M3 红线）：

- "M3 / Abbott 已证明癫痫发作机制" / "已完成 model-to-patient bridge"。
- "大范围仍沿轴的 recruitment wave = 发作"（这是 expanded axial recruitment，还没破开轴向 scaffold）。
- "§5 轴向不在 spiking 层复现"（这是 B 线错仪器的假阴；A2-P 正仪器为正）。
- "α₁>0 = 发作" / "一个本征值就是发作"。

## 6. 下一步最小路线

1. **冻结 phenotype gate（四类分开）**：小局部轴向事件 / 大轴向招募波 / 离轴或全局发作样候选 / runaway。
2. **A2-P canonical readout**：源空间 onset gradient + axis score + perpendicular spread/isotropy + low-k/
   globality，多 seed 复核低 vs 高许可度事件；加 rate-/mass-matched control（同放电量下高许可度是否更广、
   更**离轴**，而不只是更大/更轴向）。
3. **两个 gate 分开**：Gate A（可投到 B 线相图的慢状态轨迹）先过；Gate B（seizure-like = 破开轴向 / 离轴
   或全局招募）才是发作。
4. **A→B overlay** 只在 A2 的 trajectory/export schema 过测试、且 M3A→M3B 接口合同（normalized phase
   coords，见 `src/sef_hfo_m3_interface.py`）满足后才做；当前 B 线相图是 raw-knob 原子图，
   `m3a_overlay_consumable=False`，不能被直接 overlay。

> **M3A-v2 落地状态（2026-06-28）** — 上面第 1–2 步（四类 phenotype gate + 源空间 onset canonical
> readout）已收敛为可执行 spec + 计划：
> - 公式：`docs/snn_core_model_equations.md §B5`（空间场 $q_I/g_K$、四类判据、proxy/spectral 相图、红线）。
> - 计划：`docs/superpowers/plans/2026-06-28-sef-hfo-m3a-v2-spatial-slowvar-field-plan.md`（10 任务，逐任务 TDD）。
> - red-TDD 骨架：`tests/test_m3a_v2_spatial_slowvars.py`（40 红 + 1 `@slow` 红，含 2026-06-28 review 加固的
>   `k_K` bounded build / `area_large` size gate / `Y=P_global` / `aq_drive` `eta_I` 加权 4 条合同）；stub
>   `src/snn_engine/slow_field.py::SpatialSlowField` + `src/topic4_m3a_v2_phenotype.py`。
> - 延后（plan 本轮不建）：$D_{EE}$ 场、ablation A/B/C/D 机制证明。**破轴主张受 ablation gate，未解锁。**
> - **实现后首轮 pilot（2026-06-28，descriptive screen）**：见 `docs/archive/topic4/m3a_v2_field_pilot_2026-06-28.md`；
>   可复现 runner `scripts/run_m3a_v2_field_pilot.py` → `results/topic4_m3a_v2_field_pilot/pilot_results.json`。
>   一句话（锁定口径）：**field-only mechanism sanity positive**（载体在地图层面 σ_q>σ_K 因果地造出"旁边追上主轴"的离轴易激性优势，剂量可控）；
>   **但闭环 ictal-like broken-axis transition NOT established**——这块衬底踢出的是全场招募事件（R_area~0.65，非局部沿轴行波），
>   q_I 被均匀抽干（主轴=旁边 gap≈0），强耗竭只 runaway。瓶颈在 **substrate regime**（substrate enters full-field recruitment before
>   localized axial propagation），不在 M3A-v2 慢变量。proxy β_K=0.3 相对膜 η_K=1.0 低估疲劳、滞后报 off-axis 追上。
> - **Step 1 substrate qualification（slow OFF, single core, 2026-06-28）**：见 `docs/archive/topic4/m3a_v2_substrate_qualification_2026-06-28.md`；
>   runner `scripts/run_m3a_v2_substrate_qualification.py` → `results/topic4_m3a_v2_substrate_qual/qualification_results.json`。
>   **YES——局部沿轴自限间期事件存在且可达（8/192 全 5 判据过，AR 是 localize 杠杆：AR=2 全场→AR=4–6 局部 R~0.4/F_off~0.1/沿轴传播）。** 解开了 closed-loop 死结。
>   **broad sweep（864 runs multi-seed +Lever 2）找到 4/4 稳健区。primary canonical（默认 I→E 结构、无 Lever 2 confound）= `AR=4, g=8, l_EI=0.25, C_EI=200, nu=0.46, kick=3.0`**（mean R~0.40/S~0.99/F_off~0.14/peak~42Hz, **seed 1–8 全 8/8**）。
>   自限稳健由 **AR↓ + nu↓** 主导（AR 是 localize 梯度非硬阈，AR=2 高 g 下也能贴界过——见 AR=2 boundary probe）；**Lever 2（surround 抑制 l_EI/C_EI）边际平、不灵**（诚实负，合 memory M2 无 containment window）。机器可审计：sweep_results.json（raw_rows 带 gate flags + fresh + ar2 probe + canonical per-seed）+ multiseed_results.json。
> - **Step 2 q_I-only（g_K=0, D_EE=1, 2026-06-28）= NEGATIVE（informative）**：见 `docs/archive/topic4/m3a_v2_step2_qI_2026-06-28.md`；runner `scripts/run_m3a_v2_step2_qI.py` → `results/topic4_m3a_v2_step2_qI/step2_results.json`。
>   3 衬底 × 4 σ_q × 3 q_min × 5 Δq_axis × 4 seed（k_q 用 baseline 重放标定，K_q mass-normalized）。**576 q_I run：532 A_no_effect、44 E_runaway、B_expanded_axial=0。**
>   **q_I 单独给不出 expanded axial**——returned run 全 R 不变（max dR=+0.016）轴向完好；会长大的全 runaway + 轴读出崩（headroom 最少的 sensitivity 衬底；**F_off 仍低~0.14，未证 off-axis recruitment**）。
>   **scope-limited（非"结构性"满话）**：当前单事件 + q_init=1 + dq≤0.30 + 3 衬底；几何半边可审计（`axis_reach_frac=1.0` 于 L=10&16，见 `L16_control.json` + raw_rows）。**未测低-q 初始态。** 结论=支持把 g_K 作下一步必要测试（q_I-only 缺终止机制），**但 g_K 能否把 runaway 变 returned recruitment 仍未证**——Step 3 判决。
> - **Step 3 q_I+g_K rescue scout（2026-06-28）= 机制不闭合（NEGATIVE, informative）**：见 `docs/archive/topic4/m3a_v2_step3_qI_gK_2026-06-28.md`；runner `scripts/run_m3a_v2_step3_qI_gK.py` → `results/topic4_m3a_v2_step3_qIgK/step3_results.json`。boundary 点 × σ_q/σ_K × **Γ_K∈{0,0.5,1,1.5,2}**（η_K 标定）× 4 seed。**432 g_K cell：298 B_oversuppress、127 C2_still_axial、7 C1_runaway、A(off-axis recruitment)=0、RESCUED=0。g_K 是刹车**（疲劳正确压轴 gap +0.118），**不重定向离轴**：**F_off 没达 recruitment gate**（max dF +0.08；4 个小升全 still-axial）、**q_off 轻–中降但从未<0.7（min 0.735）= 没进低-q permissive regime**（55ms 事件来不及耗深离轴——field-probe 持续活动有、closed-loop 单事件无的差距）。g_K 把 runaway 转 returned 但成 **suppressed/axis-dominant（R 0.05–0.50）非 off-axis**。**Γ_K target vs achieved**：target 2.0→achieved median 0.08（事件被压短 g_K 来不及积累，须同时报）。**Step 3 不能证伪整体机制——fork A 低-q 未测。****fork（用户 C2 路）**：最该试=**低-q 初始态/持续事件 regime**（补 timescale 缺口）；次选 D_EE。**M3A-v2 closed-loop 机制当前 regime 未闭合；field-only sanity 仍正。**

## 7. 合并与 worktree

- A 线证据来自 `topic4-m3a-a2` worktree 的 A1/A1b/A1c/A2 screen + `results/topic4_sef_hfo/m3a_slowvars/`。
- B 线代码 + 测试 + 生成脚本已在主线（`src/topic4_m3b_spectral_phase.py`、
  `scripts/build_m3b_spectral_outputs.py`、`tests/test_topic4_m3b_spectral_phase.py`，M3B commit
  `32ba62d`→当前）；artifacts/figures 在 ignored 的 `results/topic4_sef_hfo/m3b_spectral_phase_map/`，
  由脚本再生。
- worktree `topic4-m3a-a2`、`topic4-m1` 阶段性收口后关闭（clean，`git worktree remove` 不丢提交，分支留存）；
  `topic4-m3`（m3-hub）保留。
