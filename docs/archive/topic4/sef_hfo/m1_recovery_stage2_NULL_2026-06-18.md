# M1 (E→E short-term depression+recovery) — Stage 2 parameter screen: NULL

**Date:** 2026-06-18
**Status:** archive result (Stage 2 scientific stop per ralph-loop plan §8 / 06-17 Task 6). NOT a main-doc conclusion.
**Branch:** `topic4-snn-m1-recovery` (worktree `.worktrees/topic4-m1`, off stage4 `369cba4`).
**Plan:** `docs/superpowers/plans/2026-06-18-sef-hfo-snn-m1-recovery-ralph-loop.md`

## 0. 朴素话摘要（测了什么 / 怎么测的 / 揭示了什么）

**测了什么。** 给一片二维兴奋-抑制神经元网格加了**一个"用进废退"恢复变量**（每个兴奋神经元每放一次电，它往外传的劲暂时变弱，再慢慢恢复），扫了 60 组参数（每次掉多少劲 `U` ∈6 档 × 恢复多快 `tau` ∈5 档 × 2 个随机种子）。问题：加了这一个变量，事件能不能"**既读得出传播方向、又在碰到网格边界之前就把放电范围收住（离墙还剩 >1mm）**"。

**怎么测的。** 每个事件分两件事看：(1) 活动是不是自己平息了（**时间上停**：不持续烧、不失控）；(2) 放电的空间范围有没有在碰墙前就收住（**空间上停**：放电云的 95% 半径离最近的墙还有 >1mm）。一个事件要算"能读方向"，得点亮电极阵里 ≥7 个触点。

**揭示了什么。**
- **时间上：成立。** 加恢复变量后事件确实自己平息——不持续烧、不失控、占空比中位只有 ~5%。这正是 M1 该解决的"有限事件为什么会停"。
- **空间上（能读方向的事件）：不成立，而且不是小网格逼出来的。** 凡是"能读出方向"的事件，放电云都铺满整张网格：L=20（半宽 10mm）上 95% 半径 ~13mm、沿轴铺 ~24mm，前沿摸到墙（离墙余量 ≈ −5mm）；换到更大的 L=32（半宽 16mm）上，那个能读方向的事件半径反而长到 ~20mm，**又铺满了大网格**。换句话说，**能读方向的事件会扩散到"填满当前这张网格"为止，没有一个比网格更小的固有自限尺度**。只有那些**小事件**（半径 ~6–8mm）才空间上收得住——但它们点不亮 7 个触点，读不出方向。
- 加大"每次掉劲"`U`（0.05→0.6）只**减少事件数量、不缩小能读方向事件的尺寸**。

**一句话结论（注意 scope，措辞已收窄）：** **在当前 Stage 3 工作点（twoend_equal m17.5 sep0.7 L20 drive0.6）、当前虚拟电极 readout（4mm pitch / ≥7 触点 floor）、当前 `U × tau` 网格下**，E→E STD/recovery 单独**能给时间自限，但没有打开"可读方向 + 空间自限"的事件窗口**——能读方向的事件在 L=20 与 L=32 上都扩散到填满网格。**这不证明所有 M1 参数 / 所有 montage / 所有 `l_EE`、`C_EE` 都不可能**：本轮**没有扫 `l_EE`/`C_EE`、没有改 montage**，所以不能写成"E→E STD 只能给时间自限、给不了空间自限"这种绝对话。这是 plan 预先写明的 NULL 情形之一。按 plan：归档 NULL、停，**不在没有新 plan 的情况下加第二个机制或改电极**。（两个诊断 probe 把"廉价解释"都排除了——采样太粗 §5.1、灶离墙太近 §5.2；关键事实是 **M1 给时间自限、但事件空间尺度随几何放大**；唯一没扫的杠杆是 E→E 空间核 `l_EE`/`C_EE`。）

## 1. 工作点与网格

**前情（M0，为什么试 M1）：** 不加恢复变量的 M0 在 step0 边界审计里，**减小 `l_EE` 或 `C_EE` 没有产生"可读 + 空间自限"的窗口**（主网格 `self_limited_window=0`，唯一一个 `selflimit_but_feeble` 只有 2 个触点、读不出方向；见 `stage3_m0_boundary_audit_2026-06-17.md`）。所以才有理由加一个活动依赖恢复变量，只解决"有限事件如何自停"。

- 固定工作点：`twoend_equal --core-mean 17.5 --core-std 1.0 --sep-frac 0.7 --L 20 --drive 0.6`，事件阈值 `prefix_peak --cal-prefix-ms 3000`，`T=8000`。
- 网格：`U ∈ {0.05,0.10,0.20,0.30,0.40,0.60} × tau_rec_ms ∈ {50,100,200,500,1000} × seed ∈ {1,2}` = **60 runs**，`PAR=8`（L=20 建连 ~13GB/进程的 OOM 上限）。
- 执行：**60/60 干净完成，0 失败、0 NaN/inf/OOM**。
- 引擎护栏：`ee_std_u=0` 与 M0 逐比特一致（`test_u0_is_bit_identical_to_M0`，sha `da5fc18c27d5340a`），5/5 机制单测 + 33/33 引擎回归全过；引擎已 re-bless。

## 2. 验收口径（plan §2 / §3.2）

逐事件：candidate = `n_fired_E/NE ≥ 0.02`；readable = `n_part ≥ 7 且 axis_err 非空`；axis_following = readable 且 `axis_err ≤ 25°`；returned = 检测器的 returned；self_limited_before_boundary = returned 且 readable 且（前沿 `T_stop+5ms < T_edge` 或前沿从不碰墙）且事件期间最小离墙余量 ≥ 1.0mm；primary_event_pass = readable 且 axis_following 且 returned 且 self_limited。
候选门（plan §3.2，两个种子都要满足）：`n_primary≥3, pass_frac≥0.30, axis_following_frac≥0.70, boundary_contact_rate≤0.10, runaway==0, frac_time_on≤0.30`。

分析器：`scripts/analyze_m1_recovery.py`（12/12 TDD 测试；含真实数据上发现并修复的 `sign=None` 崩溃回归）。

## 3. 结果数表

### 3.1 全网格（60 cell）聚合分布

| 量 | min | median | max |
|---|---|---|---|
| n_candidate | 1 | 14 | 41 |
| n_readable | 0 | 2 | 11 |
| **n_primary_event_pass** | **0** | **0** | **0** |
| primary_event_pass_fraction | 0 | 0 | 0 |
| axis_following_fraction (readable 中) | 0 | 1.0 | 1.0 |
| **margin_fail_rate** (readable r95 余量 <1mm) | 0 | **1.0** | 1.0 |
| **near_boundary_rate** (readable r95 bulk 越墙 ≤0) | 0 | **1.0** | 1.0 |
| **median_edge_margin_r95 (mm)** | −8.37 | **−5.23** | −2.42 |
| boundary_contact_rate (r95-margin 口径) | 0 | 1.0 | 1.0 |
| runaway_event_count | 0 | 0 | 0 |
| frac_time_on | 0.005 | 0.048 | 0.142 |

- **`hard_no_go = True`**：没有任何 cell 有 ≥3 个 primary_event_pass。45/60 cell 有 ≥1 个能读方向事件；这些 cell 里能读方向事件**几乎全部越墙**（中位 `margin_fail_rate=1.0`、`near_boundary_rate=1.0`）。
- **指标口径修正（reviewer P1）：** 旧版 `boundary_contact_rate` 用的是"前沿包围盒严格越墙 `edge_dist≤0`"，它触底在 ~0.0002mm、几乎从不严格 ≤0，所以报成 0 —— **会误导成"没碰边界"**。失败其实来自 **r95 余量**：能读方向事件 95% 半径 `r95≈13mm`、沿轴 `reach_axis≈24mm`、`edge_margin_r95 ≈ −5mm`（r95 放电云 overflow 墙）。现已把 self-limitation 与边界口径统一到 plan 的 **`edge_margin_r95(t) = 中心离最近墙 − r95(t)`**，并新增 `margin_fail_rate`/`near_boundary_rate`/`median_edge_margin_r95` 列（上表已是修正后数字）。trajectory 现保存时间分辨的 `margin_r95(t)` + `min_edge_margin_r95`（plan §2.5 口径）；本 60-cell 网格的边界判定用**事件级 fullfield `edge_margin_mm`（同为 r95 口径，已存在，无需重跑）**，时间分辨版用于 readout-escape probe 与将来重跑。

### 3.2 能读方向事件尺寸不随 `U` 缩小

L=20、各 `U` 下（pool tau+2 种子）的候选事件几何中位：`r95 ≈ 5.8–6.3mm`（候选含大量小事件），但**能读方向的子集 `r95≈13mm`**；随 `U` 0.05→0.6，候选事件**数量** 171→102、能读方向**数量** 34→14 下降，但能读方向事件**尺寸不变**。

### 3.3 L=32 诊断（决定性：尺度是否网格无关）

`L=32 --prune-radius 4.3 --sep-frac 0.5`（灶 ~8mm 离墙），U=0.2 tau=200，T=8000：

| | 候选 r95 (min/med/max) | 能读方向 r95 | 能读方向 edge_margin | 能读方向 traj_min_edge | self-limited |
|---|---|---|---|---|---|
| L=32 | 6.6 / 8.2 / 20.1 | 20.1mm (n=1) | −4.5mm | ~0.0mm | 0/1 |

- 能读方向的事件在 L=32 上 `r95≈20mm`，**又铺满了大网格**（半宽 16mm，余量 −4.5mm）。
- 小候选事件中位 `r95≈8mm`，**仍然收得住**——但读不出方向。
- ⇒ **能读方向事件没有比网格更小的固有空间自限尺度**；放大网格本身不能制造"能读方向 + 空间自限"的事件。（注意 L=32 能读方向样本 n=1，薄；但 L=20 侧有数百个能读方向事件、全部铺满网格，是结论主干。）

## 4. Verdict

> **Stage 2 = NULL（fail），scope 已收窄。** 在 `twoend_equal m17.5 sep0.7 L20 drive0.6 prefix_peak` 工作点、**当前 4mm pitch / ≥7 触点 readout**、`U × tau` 网格下，单独加一个 E→E 用进废退（short-term depression+recovery）变量，**足以让事件在时间上自限（returned、不 runaway、占空比 ~5%），但没有打开"既能读出传播方向、又在碰墙前空间自限"的事件窗口**——能读方向的事件在 L=20（`r95≈13mm`）与 L=32（`r95≈20mm`）上都铺满网格（r95 bulk 越墙 ~5mm），只有读不出方向的小事件（`r95≈6–8mm`）才空间收得住。结论对 `U`(0.05–0.6)、`tau`(50–1000) 与网格尺度（L=20/L=32）稳健。
>
> **不可外推（本轮未测）：** `l_EE`/`C_EE` 没扫、montage 没变、未试其它 readout 定义——所以**不能**写成"E→E STD 在任何空间耦合 / 任何电极下都给不了空间自限"。

**只能写到这一步的措辞约束（plan §10 + CLAUDE.md §5）：**
- ✅ 可写："在当前 Stage 3 工作点、当前 4mm pitch / ≥7 contacts readout、当前 `U×tau` 网格下，E→E STD/recovery 单独能给时间自限，但没有打开'可读方向 + 空间自限'的事件窗口。"
- ❌ 不可写："E→E STD 只能给时间自限、给不了空间自限"（太绝对——未扫 `l_EE`/`C_EE`、未改 montage、未试其它 readout）；"恢复机制无效"（它实现了时间自限）；"模型复现/无法复现发作"；"加第二个机制就能解决"（未测）。

## 5.1 Readout-escape probe（reviewer P1：区分"不可读" vs "不能成方向模板"）

**做了什么。** 同一个 M1 工作点（twoend_equal m17.5 sep0.7 L20 U=0.2/0.4 tau200），把虚拟电极 readout 改细：`pitch 2mm`（原 4）、`nc 9`、`k_dir 2`（floor `part_min=5`，原 7），看那些 `r95≈6–8mm` 的小事件到底是"读不出方向"还是"读得出但成不了稳定模板"。

**揭示了什么（两层、都重要）：**
1. **小事件不是"不可读"——它们读得出方向。** 细 readout 下，`r95≈6mm` 的小事件就能点亮 ≥5 触点、`axis_err 0–18°`、并给出干净的正/反方向（三个 run 的 clean fwd/rev = 7/1、**2/3（平衡）**、4/1）。**所以排除了"小事件本质上不能形成方向模板"这个机制性解释。**
2. **但小事件仍然不空间自限。** 即使是 `r95≈6mm` 的小事件，`edge_margin_r95` 中位仍是 **−1.4 / −1.7 / −1.5mm**（self-limited 0/12、0/9、0/11）——因为**灶本身离墙只有 ~5mm**（sep0.7/L20），一个 6mm 的事件就越墙 ~1mm。

**收窄后的诊断（§5.1 单独看）：** 障碍**既不是 readout pitch（可越过）、也不是机制（小事件读得出方向）**，而看起来是**灶到墙的距离**（sep0.7/L20 → 灶离墙 ~5mm < 事件 6mm）。这指向"灶离墙更远 + 细 readout 一起上"或许能打开窗口——但下面的组合 probe（§5.2）把这个希望也排除了。

## 5.2 组合 probe（L=32 灶离墙更远 + 细 readout）—— 把"灶到墙几何"也排除

**做了什么。** L=32（prune_radius 4.3）、sep0.7（灶离墙 ~8mm、且两灶分得开）、细 readout（pitch 2、nc 14、floor 5）、U=0.2 tau200。

**揭示了什么（关键，强化 NULL）：**
- **方向模板很好。** clean fwd/rev = **3/3（平衡）**，隐藏源 neg/pos = 3/3、0 碰撞——scaffold 的方向读出本身没问题。
- **但仍然 0 个空间自限（0/11）。** 因为**事件尺寸随可用空间一起放大**：L=20 灶离墙 5mm 时 `r95` 中位 ~6.8mm；L=32 灶离墙 8mm 时 `r95` 中位涨到 **10.1mm**（单个事件 `r95` 最大到 22mm）。事件总是铺到比"灶到墙距离"还大 ~30%，所以**越墙 ~2mm（`edge_margin_r95` 中位 −2.1mm）依旧**。
- ⇒ **不存在一个"灶位置 / 网格大小 / readout 粗细"的组合能让事件在墙前 1mm 收住**：给多大空间，事件就铺多大。M1 的恢复变量给了**时间自限**（活动会停），但**没有给一个比几何更小的固定空间自限尺度**——空间上事件实质是被墙挡住的（wall-limited），不是自己在墙前收住的。

**仍未测的杠杆（reviewer 锁，必须留口子）：** 事件铺多远，主要由 **E→E 空间核 `l_EE` / 连接度 `C_EE`** 决定（本轮全程用 paper 默认 `l_EE=0.38`/`C_EE=800`，**没扫**）。**降低 `l_EE` 会直接缩短扩散距离**，理论上可能给出一个固定的小尺度。M0 step0 审计扫过 `l_EE`/`C_EE`（M0 下 `self_limited_window=0`），但 **M1 + 低 `l_EE`** 这个组合本轮没跑。所以**不能**写成"M1 在任何 E→E 耦合下都给不了空间自限"——只能写"在 paper 默认 `l_EE`/`C_EE` + 测过的几何 / readout / `U×tau` 下，没打开窗口，且事件空间尺度随几何放大"。

## 5. 为什么停（plan §8）+ 移交人审的设计问题

按 plan §8 scientific stop（"No Stage 2 candidate after full parameter scout"）与 06-17 Task 6（"archive that NULL and stop, do NOT add more slow variables without a new plan"）：**loop 在此停止，Stage 3–6 不进行**（它们都依赖不存在的 Stage 2 候选）。**未擅自**：(a) 加第二个机制（如 V_th 适应 / chloride / 电极几何改动），(b) 把整条流水线改到更大网格重跑——这些都需要新 plan / 你的设计决定。（注：readout-escape + 组合 probe 是**小诊断**，用来给你的决定提供证据，不是把工作点改了重跑全流水线。）

**留给你拍板的设计问题（已被 §5.1 + §5.2 大幅收窄）：** 两个诊断 probe 排除了两个"廉价"解释——**不是 readout 太粗**（细了小事件就能读方向）、**也不是灶离墙太近**（灶移远事件就跟着长大、照样越墙）。现在的核心事实是 **M1 给时间自限、但不给固定的空间自限尺度（事件随几何放大）**。可能的方向（**都需要新 plan**，本 loop 不动）：
1. **扫 `l_EE` / `C_EE`（最该先做的）：** 这是直接控制"事件铺多远"的旋钮，本轮没动。降低 `l_EE` 可能把事件钉在一个固定小尺度上，配合细 readout 才有机会同时拿到"可读方向 + 空间自限"。**这是把 Stage 2 NULL 判成"机制结论"还是"参数没扫到"的分水岭。**
2. **重新定义 readout**（不靠固定触点数，靠空间连续场）。
3. **接受"M1（在默认 `l_EE`/`C_EE` 下）只给时间自限"**，空间自限交给别的机制 / 别的问题（plan 明确不在无新 plan 时加机制）。

## 6. Stage2b connectivity rescue screen（扫 `l_EE`/`C_EE`，user spec 2026-06-18）

§5 把"最该先做的"定为扫 E→E 空间耦合。这一节执行它，作为**最后一个 bounded screen**：先扫 `l_EE` 线，再（仅当出现过渡带时）局部扫 `C_EE`。固定最好读条件：`U=0.2 tau=200 pitch=2 k_dir=2`（floor=5）、`twoend_equal m17.5 std1.0 sep0.7`、主判定 `L=32`、`L=20` 算 `r95` 放大比；逐 `l_EE` 用 tail-bounded `prune_radius=8·l_EE·√2`。

### 6.1 `l_EE` 线（C_EE 固定 800）—— 缩短 E→E 空间核

驱动 `run_m1_stage2b_lee_screen.sh`，评估 `eval_m1_stage2b.py`。`l_EE ∈ {0.38…0.075}`，28 run 全干净。

| `l_EE` | verdict | L=32 可读事件数 | `r95_L32`(mm) | margin_r95(mm) | 隐藏的小自限事件 (`n_subreadable_self_limited`) |
|---|---|---|---|---|---|
| 0.38 | no | 13 | 10.1 | −1.8 | 0 |
| 0.30 | no | 13 | 10.9 | −2.6 | 0 |
| 0.25 | no | 7 | 12.9 | −2.9 | 0 |
| 0.19 | no | 6.5 | 10.2 | −2.1 | 0 |
| 0.14 | silent_local | 0 | — | — | **1 (seed1)** |
| 0.10 | silent_local | 0 | — | — | 0 |
| 0.075 | silent_local | 0 | — | — | 0 |

**逐 `l_EE` 没有 candidate。** 形态是：高 `l_EE`（0.19–0.38）= 事件**能读方向但越墙**（注意 verdict "no" 指"可读但越墙"，**不是"不会停"**——这些事件 `returned` 11/18、15/16，时间上是停的）；`l_EE≈0.14` = 出现**极个别小自限事件**（`r95=3mm`、`returned`、`margin_r95 +5.5mm`，真自限）但**低于读出门槛**（`n_part=3 < 5`）且稀疏（T=8000 只有 1 个、只在 seed1）；`l_EE ≤ 0.10` = 传播基本关掉（0 候选）。

**两个关键点：**
- **缩短 `l_EE` 没有缩小可读事件**（`r95` 一直 ~10–13mm，到 0.19 都没小）——因为 `C_EE=800` 把递归增益钉高，事件一直铺到墙、直到 0.14 附近传播突然塌掉。**所以真正控制"事件多大"的旋钮是 `C_EE`（增益），不是 `l_EE`（核宽）。**
- 诚实度指标 `n_subreadable_self_limited`（新加）把 `l_EE=0.14` 那个被 `silent_local` 标签盖住的真自限小事件露了出来——但 1 个/T8000、单 seed，**符合你预设的"silent/local 小事件 = 不是 rescue，是把传播关掉了"**。

### 6.2 `C_EE` 局部扫（`l_EE ∈ {0.25,0.19}` × `C_EE ∈ {800…400}`）—— 降增益缩事件

因为 6.1 指出 `C_EE` 才是增益/事件大小的旋钮，在仍频繁可读的 `l_EE=0.19/0.25` 上降 `C_EE`。20 run（L=32）全干净。

| `l_EE` | `C_EE` | L=32 可读事件数 (s1,s2) | `r95`(mm) | margin_r95(mm) | n_primary |
|---|---|---|---|---|---|
| 0.25 | 800 | 9, 5 | 12.9 | −2.9 | 0, 0 |
| 0.25 | 700 | 15, 10 | 10.4 | −2.4 | 0, 0 |
| 0.25 | 600 | 6, 0 | 14.2 | −6.6 | 0, 0 |
| 0.25 | 500/400 | 0, 0 | — | — | 0, 0 |
| 0.19 | 800 | 9, 4 | 10.2 | −2.1 | 1*, 0 |
| 0.19 | 700 | 6, 2 | 13.7 | −4.0 | 0, 0 |
| 0.19 | 600 | 3, 0 | 16.8 | −8.8 | 0, 0 |
| 0.19 | 500/400 | 0, 0 | — | — | 0, 0 |

(*同 §3.1 那个孤例，非新 candidate。)

**降 `C_EE` 没救活，而且方向相反：** 把连接数从 800 往下降，残存的可读事件**不但没变小、反而更大**（`r95` 10→14→17mm，`margin_r95` −2→−9mm），再往下（500/400）传播直接关掉。整个 C_EE 扫 **`n_primary_event_pass` 合计 = 1（那个孤例）、`n_subreadable_self_limited` = 0**。（注：低 C_EE 的"更大"是少数残存事件的选择效应，样本小；但**没变小 + 中间没甜区**是稳的。）

### 6.3 Stage2b 总判定：E→E 耦合不救 M1

> **沿 E→E 空间耦合两条轴（缩短核宽 `l_EE` / 削弱连接数 `C_EE`）都没有 candidate。** 事件没有一个比几何更小的固定空间尺度：要么**填满/越墙**（高耦合，可读但 `margin_r95` −2~−9mm），要么**传播关掉**（低耦合，0 候选），中间**没有"小而可读又空间自限"的甜区**。`l_EE≈0.14` 那 1 个孤立小自限事件（sub-readable、单 seed、T=8000 仅 1 个）符合预设的"silent/local 小事件 = 不是 rescue"。

**按 user spec 的解释规则 → "没扫出候选"分支：** M1 NULL 显著强化——**在恢复变量(`U×tau`) + 读出(pitch/floor) + 几何(L/灶位置) + E→E 空间耦合(`l_EE×C_EE`) 四个维度都扫过后，仍没有 "readable spatial self-limit" 窗口。** M1 给的是时间自限、不是空间自限。**到此，转向 M2（换/加一个机制）在科学上站得住**——这是需要新 plan / 你拍板的下一步（本 loop 不擅自加机制）。**仍未触的边角**：`l_EE`/`C_EE` 联合大网格、`drive`、非 twoend 灶——但单轴各自已 close，联合扫属 M2-之前的可选稳健性，非必需。

## 7. 产物

- 网格读出：`results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/m1_recovery/stage2_param_screen/`（`readout/fullfield/af/trajectory_*` ×60，gitignored）。
- 聚合：同目录 `m1_recovery_param_screen.csv` + `m1_recovery_param_screen_summary.json`（`hard_no_go=true`）。
- L=32 诊断：`.../m1_recovery/stage2_diag_L32/`。
- 引擎/runner/分析器/网格驱动：branch `topic4-snn-m1-recovery` 提交 `69a19dd → 1e0daa4`。
- 断点状态：`.../m1_recovery/loop_state.json`（stage2 verdict=fail，loop STOPPED）。
