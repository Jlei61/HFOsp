# SEF-HFO Step 0 结果 + go/no-go 闸门（2026-06-02，带延迟版）

> 状态：**Step 0 机器搭建完成 + scaffold 跑通**。本次是 scaffold（占位参数）跑，正式科学结论须等数据锚定的工作点与单位（见 §go/no-go 闸门 formal tier）。计划见 `docs/superpowers/plans/2026-06-02-topic4-sef-hfo-step0-stability-pulse-plan.md`。

## 测了什么

跑大网络前先用最便宜的方式问两件事：

1. **把一块组织的神经元阈值"调整齐"（降低异质性），系统会不会更接近"一点就着"的边界？** 我们在一族背景工作点上算，而且报"这一族里有多大**比例**真的更靠近边界"，而不是"找到一个能用的点就行"。
2. **给一束有限大小的脉冲，活动会怎样？** 是"传一段又自己熄灭"（这才是我们要的间期事件），还是"点不着就熄灭 / 一点就失控 / 只是原地一起亮一下（全局同步闪光，不是传播）"。

这一版还把**突触的快慢**和**信号沿轴传导的延迟**也放进了稳定性计算里（不是事后近似）。

## 怎么测的

- **0a（线性稳定性地图）**：每个工作点先解一个自洽稳态（群体活动自己喂自己达到的平衡点），再把突触动力学和传导延迟当成"附加的线性状态"，算这个系统对各种空间波长扰动的增长率，得到一个"离失稳边界有多远"的读数（越大越稳，接近 0 是候选可激窗，负的是失稳），以及"最容易先失稳的是哪种空间花样"。还验证了把延迟用有限级链近似时，级数够多、结果已经收敛。
- **0b（有限脉冲响应）**：对 0a 给出的候选工作点逐个打脉冲，扫半径×时长×幅度。只有活动表现为**会移动的紧凑波前**（不是原地长大的同步闪光）才算"传播"；而且要求"失控的幅度"明显高于"刚点出事件的幅度"，中间留正的安全余量，才算有真正的工作窗。恢复变量（让活动用一会儿就累、自己压下去）**关掉和打开两套都跑、都报，不自动选**。

## 揭示了什么（本次 scaffold 跑）

**一句话：机器搭好了、测试齐了、端到端跑通了；在占位参数下系统深度稳定、主模在 k=0（全局），所以 0a/0b 全退化。但单独的能力体检证明这套机器有能力算出框架要求的三区结构和有限 k 空间轴——退化是 scaffold 参数坐在稳定角落，不是机器算不出。唯一真正悬而未决的是"会移动的有限 k 行波（finite-k Hopf）可不可达"（见下"能力体检"）。**

具体数字（都是 scaffold 占位参数下的事实，不是正式科学结论）：

- **低异质性筛选**：把阈值异质性 `sigma_phi` 从 1.0 降到 0.5，全部 5 个工作点的"离边界距离"都下降（`fraction_closer = 1.0`）——方向上"调整齐神经元确实让系统更靠近临界"成立。但即便一路降到 0.05，这个距离也只是从 ~0.28 掉到 ~0.157 的地板，**始终没跨过 0**：占位参数下根本到不了失稳/可激区。
- **候选可激工作点**：**0 个**。数据锁定的 5 个工作点"离边界距离"都在 0.27–0.30，全落在深度稳定区（相图整片是稳定的蓝色，没有黄色候选带、没有红色失稳区）。
- **0b 有限脉冲**：所有工作点、所有脉冲（半径×时长×幅度 0.2–3.0）的响应**全部是 extinction（点不着就熄灭）**；`fraction_with_window = 0`，恢复关/开都一样；半 dt、小 L 的敏感性检查也都是 0 → 这个"退化"结论在数值上是稳定的，不是 dt 或边界假象。因为没有任何"自限传播/全局同步/失控"事件，例子快照图本次没生成（4 张图里出了 3 张）。

**额外做的"能力体检"（advisor 在 Step 0a 关口提醒、final review 收紧后补的，只动了被允许调的连接核 / 抑制强度探针，没动锁定参数；可复现产物 `results/topic4_sef_hfo/linear_stability/capability_probe.json`，脚本 `scripts/run_sef_hfo_step0a_capability_probe.py`）**：

体检扫了 3 个连接 regime（每个 15×8=120 个背景工作点）：

| regime | 收敛点 | `eta_lin<0`（失稳区可达？） | 有限 k 主模 | 有限 k 且失稳 | 有限 k 且行波(Hopf, ω*>0) |
|---|---|---|---|---|---|
| `scaffold_default`（锁定默认） | 120 | 0 | 0 | 0 | 0 |
| `moderate_hat`（中等短兴奋/宽抑制） | 120 | 9 | 1 | 0 | 0 |
| `aggressive_hat`（强短兴奋/宽强抑制） | 120 | 40 | 42 | **40** | **0** |

- **scaffold 默认参数**：深度稳定（无 `eta_lin<0`）、主模全在 k=0。这是主结论。
- **但这套机器结构上做得到三区相结构 + 有限 k 空间轴**：把连接推到"短程兴奋 + 长程强抑制"（Mexican-hat），`aggressive_hat` 有 40 个工作点出现**有限波长（沿轴）的失稳主模**（例：`|k*|≈0.20, max Re≈0.279 @(I_E=1.4,I_I=0.25)`，静态 Turing）。所以"各向异性连接挑出有限 k 传播轴"**不是机器算不出来**——之前一版写"始终 k=0、连 Mexican-hat 都只给 k=0"是探针太窄导致的过度断言，现已更正。
- **真正还没拿到的是"行波"那一档**：本次扫到的有限 k 失稳全是**静态 Turing 花样（ω*=0）**，没有一个是**有限 k 行波（finite-k Hopf, ω*>0）**——而 SEF-HFO 要的是会**移动**的自限瞬态（对应 finite-k Hopf）。本扫描是有限的（3 regime、固定 k 网格），"finite-k Hopf 在这套 rate-field + 延迟/恢复结构里到底可不可达"**没有定论**（本次 0 个，但不能据此判死）。

**这意味着什么（诚实表述，不修饰）**：(1) 机器搭好、测齐、跑通，且**经体检证明有能力**产出框架要求的三区相结构与有限 k 空间模；(2) **scaffold 默认参数坐在深度稳定、k=0 的角落**，所以本次 0a/0b 全退化是参数位置问题，不是机器算不出结构；(3) **唯一真正悬而未决的科学问题**是"会移动的有限 k 行波（finite-k Hopf）可不可达"——这要和数据锚定一起、并很可能要认真看延迟/恢复时间结构（它能把静态 Turing 变成行波）才能回答，而不是只换数值。

## go/no-go 闸门

- **smoke tier（结构性检查，scaffold 跑就该过）：通过**。`erlang_n_convergence.converged = True`（延迟级数收敛，色散可信）；低异质性筛选输出是合法比例（0–1）；0b 的 `fraction_with_window` 在半 dt、小 L 下一致（数值稳定）。
- **formal tier（解锁 Step 1 的正式闸门）：SKIP（scaffold 跑，按设计不解锁）**。formal tier 要求 `provenance.source == "data_locked"` + 数据锚定单位；当前是 `"scaffold"`。即便跑了 formal，本次 `fraction_with_window = 0` 也会判定 **Step 1 不启动**——计划早就锁定"`fraction_with_window == 0` 也是一个发现，不是失败"。

**结论**：Step 0 的机器（模块 + 测试 + 两个 runner + 闸门）已就绪、全测试通过；scaffold 默认参数下深度稳定、无自限工作窗、主模在 k=0（但能力体检已证明这套机器能算出三区结构 + 有限 k 空间模，见上）。**Step 1 保持锁定**，等：(a) 数据锚定的工作点族 + 单位（Brunel Table 1 突触/膜时间常数、实测 HFO 群体事件时长定 `tau_a`、Hz 量纲）；(b) 对"会移动的有限 k 行波（finite-k Hopf）在这套 rate-field 里可不可达"这一悬而未决问题的处理（很可能要看延迟/恢复时间结构）。

## 与 Brunel 2026 的关系

方法（自洽 2×2 色散、外驱相图、各向异性 E→E 选轴、`|I^E|+|I^I|` LFP 代理）与 Bachschmid-Romano/Hatsopoulos/Brunel 2026 一致；区别是他们的事件是近全局 Turing–Hopf 行波，我们要的是局部自限瞬态——所以 0b 的 recovery off/on 正好对应"亚临界近 Hopf"与"恢复变量局部脉冲"两条机制。能力体检显示：scaffold 默认坐在 k=0 角落，但把连接推到 Mexican-hat 能拿到**有限 k 的静态 Turing 失稳**；**还没拿到的恰恰是他们模型核心的那一档——有限 k 的 Hopf（行波）**。换句话说与 Brunel 的差距不在"能不能有限 k"，而在"有限 k 能不能动起来"，这正是与数据锚定一起要解决的下一步。

## Step 0 构建审计链（实现中发现并修复的计划代码缺陷）

本次按 TDD 逐任务实现，红→绿→commit。计划给的是逐字代码，但执行中 TDD 暴露出 **4 处真实的计划代码缺陷**（都在 commit message 里详记，均经独立验证后修复，未放松任何容差/锁定参数）：

1. **transcendental 交叉校验（Task 5）**：计划的"角点变号"复平面盒扫描在实轴上有盲区（`Im(D)≡0` 使底排判据永不触发），返回 `-inf` → 改为同一盒内多起点复 Newton（`fsolve`）。仍是独立校验（D 由解析式独立构造）；4 工作点×4 k 切片上与矩阵特征值吻合 <0.0004。
2. **场线性化↔特征值一致性（Task 9）**：计划把场的 rE 模衰减率与矩阵"最大 Re 特征值"比——错配，因为 rE-only 种子激发的是模式混合（此工作点最大 Re 模在 rE 上投影极小）→ 改为"同一可观测量"的对照：场 rE 模 vs 矩阵自身预测的 rE 模（`expm(M t)@ic`）。更强的一致性检验，|diff| 0.004–0.007。
3. **分类器测试辅助 `_act`（Task 10）**：分类器本身正确，但计划的 `_act` 造数据辅助无法实现自己的两个用例（width 下限为 1 → frac=0 不熄灭；块向右长 → 静止闪光的质心漂移）→ 重写为 frac=0 出空帧 + 块以 cx 为中心。
4. **runner 缺 `sys.path` bootstrap（0a/0b）**：计划逐字 runner 漏了本仓库 `scripts/` 既有 runner 的惯例 `sys.path.insert(0, parents[1])`（如 `scripts/run_rank_displacement.py`），导致 `python scripts/X.py` 报 `No module named 'src'` → 按仓库惯例补上。

## 留给 formal run 的已知问题（final review 标记，不影响本次 scaffold 结论）

- **`safety_margin` 在"有自限窗但量程内无 runaway"时被低估**（`src/sef_hfo_pulse.py` + `scripts/run_sef_hfo_step0b_pulse.py` 的 `scan_point`）：此时 `safety_margin = A_runaway(=inf) − A_self_limited = inf`，而 `np.isfinite(safety_margin)` 把这个"最大安全"的窗判成 `has_window=False` → **漏计**。本次全 extinction 用不到，formal run 前要修（例如把 `A_runaway=inf` 视为"余量无上限即安全"）。
- **单位锚定不是独立闸门**（`tests/test_sef_hfo_step0_gate.py` formal tier）：计划 review #5 要求"单位锚定 gate formal run"，但当前 formal test 只查 `source=="data_locked"` / `locked_before_sweep` / `hash` / 窗>0，没有独立的单位字段与断言（隐式并进 `data_locked` 标志）。formal run 前补一个显式单位锚定检查。

（内部归档代号：Step 0a/0b、delayed dispersion、erlang_n、eta_lin、operating-point family、fraction_closer、finite-pulse、A_event/A_self_limited/A_runaway/safety_margin、global_synchronous、recovery off/on、k_star、Mexican-hat、provenance scaffold vs data_locked、Brunel 2026）
