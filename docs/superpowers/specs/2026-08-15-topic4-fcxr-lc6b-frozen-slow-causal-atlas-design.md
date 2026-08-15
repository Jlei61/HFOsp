# FCXR-LC6B：frozen-slow fast-subsystem causal atlas

日期：2026-08-15
上游：`docs/archive/topic4/fcxr_lc6a_patient_axis_surround_no_carrier_2026-08-15.md`
结果根：`results/topic4_sef_hfo/fcxr_lc6b_frozen_slow_atlas/`

## 0. 命名前向说明（必读）

`LC6B` 这个编号被**重新指派**。

- **旧含义**（LC6A spec §14 / §15 里出现的 "carrier positive -> LC6B U recalibration"）：在新 carrier 上重新标定 U 的条件性后续。该路线**改名为 `LC6D`**，内容不变、仍未授权、仍是条件性的。
- **新含义**（本文档）：`LC6B` = frozen-slow fast-subsystem causal atlas。

任何引用 "LC6B" 的旧文本，若语境是 "recalibrate U on a carrier"，指的是现在的 LC6D。本轮不执行 LC6D。

## 1. 一句话目标

在**把 D_i(x) 和 H_i(x) 按住不动**的条件下，当前 fast subsystem 到底有没有一个有界的高分支？

朴素话：那片组织进入发作以后一路冲到顶。可能有两种原因。一种是快回路本身就没有中间平台 ——
只要点着了就一定冲顶。另一种是本来有平台，但两个慢变量（突触疲劳 `D`、循环增益记忆 `H`）
一直在动，把系统从平台上推了过去。这两种情况在自然轨迹里长得一模一样，因为慢变量永远在动。
唯一能分开它们的办法是：把慢变量按住，只让快回路跑，看它停不停得住。

## 2. 上游已成立、本轮继承的结论

LC6A 的正式标签是 `CANONICAL_SEED_AXIAL_REACH_FAMILY_NO_CARRIER`：把患者轴方向的两跳 E→I
抑制周边从 q≈0.935 加宽到 q≈1.490，五条自然轨迹全部在进入后升级到注册饱和，没有打开有界载体。
LC6A 的 `termination` 和 `lifecycle` 都是 `NOT_TESTED`，本轮**也不测**。

LC6A 明确禁止的说法在本轮继续有效，尤其是：不得说 Mexican-hat 普遍无效、不得说 U 被否定、
不得说 gain fork 测到了高态增益。

## 3. 被测系统与冻结对象

### 3.1 fast subsystem 的定义

"fast" = 除 `z`（`D = 1 − z`）与 `h_lc2_E` 之外的一切前向状态：膜电位 `V`、不应期计数 `ref`、
四个突触滤波器 `s_E/I_E/s_I/I_I`、循环对 `s_E_rec/I_E_rec`、两个延迟环 `ring_sE/ring_sI`、
OU 噪声 `xi`、以及网络 RNG。

`X`（`x_relay`）在 LC6A 的锁定配置里就已经冻结在 1.0（`x_relay_frozen_E = ones(NE)`），
本轮实测两个源快照的 `x_relay` 逐元素恰好等于 1.0，因此 X 不是本轮的自由慢变量。
`U`（pump）与 `M`（adaptation）在 LC6A 配置里关闭，本轮**保持关闭**。

### 3.2 D 的冻结路径（已存在）

引擎已有 `MZSlowVarsConfig.z_frozen_E`（FCXR Stage D）：`membrane_terms` 在
`use_z or z_frozen_E is not None` 时都用 `self.z[:NE]` 调制受到的 GABA，而 `step()` 只在
`use_z` 时更新 `z`。所以 `use_z=False` + `z_frozen_E = <当前 z 场>` 就是逐步逐位不变的
D 冻结。`_validate_config` 已经强制 "`z_frozen_E` 要求 `use_z=False`"。

### 3.3 H 的冻结路径（本轮新增）

`use_h_lc2` 同时门控两件事：`membrane_terms` 里的 H **输出**通路
（`gH = rho_h_lc2 · S̃(h)`，加在 tanh 饱和之前）与 `step()` 里的 H **状态**更新。
把 `use_h_lc2` 关掉会同时关掉输出通路并落到 HEO1 cooperative 分支 —— 那是另一个机制，不是冻结。

因此新增 `MZSlowVarsConfig.h_lc2_frozen_E`，语义与 `x_relay_frozen_E` 完全对齐：
`use_h_lc2` 保持 `True`，膜项通路一个字节都不改，只跳过 `step()` 里那两行状态更新。
`_h_source_lc2_E`（诊断源，逐帧仍写）继续记录，但不再喂进 `h_lc2_E` —— 这满足
"source trace 仍可记录，但不能写 frozen state"，且因为完全**跳过**写入而非写完再覆盖，
不存在一帧泄漏。

`h_lc2_E` 本身已在 `state_hash` 的哈希字段里，所以冻结场的内容进入 exact-state hash。

### 3.4 off-by-default

`h_lc2_frozen_E` 默认 `None`。`None` 时 `step()` 的条件式与改动前逐字符等价，
必须有 byte/numerical parity 回归测试证明既有轨迹一个 bit 都不变。

## 4. 两个源快照：先修 checkpoint 语义

LC6A 的 `checkpoint_onset_detected.npz` 与 `checkpoint_onset_plus_1s.npz` **文件逐位相同**
（两者 `t = 240000`，即 12.0 s，`state_hash` 前缀 `da550ff474e6efd1`）。原因在 LC6A 的 manifest
里已有留痕：`onset_detected` 的 `timing_error_ms = 1000.0` —— onset 在 11.0 s，但检测在 1 s 的
chunk 边界上才确认，所以两个名字落到同一步。**不得把它们当作两个时刻。**

本轮第一批直接用两个互相独立、时间已从状态与 manifest 双向核对过的快照：

| 快照 | 文件 | `t`(steps) | 绝对时间 | 相对 onset | fork 前 1 s 全局率 |
|---|---|---:|---:|---:|---:|
| S2 | `trajectories/C0/checkpoint_onset_plus_2s.npz` | 260000 | 13.0 s | +2.0 s | 34.27 Hz |
| S4 | `trajectories/C0/checkpoint_onset_plus_4s.npz` | 300000 | 15.0 s | +4.0 s | 58.57 Hz |

C0 的 onset 在 11.0 s，`dt = 0.05 ms`，`260000 × 0.05 = 13000 ms`、`300000 × 0.05 = 15000 ms`，
与 manifest 的 `actual_ms` 一致，`timing_error_ms` 均为 0。两个快照都远低于 250 Hz 的注册全局
饱和线，也远低于 450 Hz 的近不应期线，所以"会不会冲顶"在这两个点上都还是真问题。

### 4.1 新 checkpoint 的元数据合同

本轮写出的每个 checkpoint 必须携带：`snapshot_time_ms`、`onset_time_ms`、
`relative_to_onset_ms`、`full_state_sha256`、`external_input_counter/state`、
`graph_sha256`、`config_sha256`。**不得再靠文件名表示时间。**

## 5. 事前预期：两个慢变量的剩余量程差得很远

这一节是**预注册的先验**，写在这里是为了让结果可以推翻它，而不是事后追认。

**H 侧几乎没有量程了。** H 的膜端输出是 `gH = rho_h_lc2 · S̃(h)`，`rho_h_lc2 = 0.54` 是硬上限，
而 `S̃` 是 [0,1] 的平滑门。实测门占用：

| 快照 | H 均值 | 门均值 `S̃` | `gH` 均值 | `gH` 上限 |
|---|---:|---:|---:|---:|
| +2 s (13 s) | 2.107 | 0.5559 | 0.300 | 0.540 |
| +4 s (15 s) | 3.790 | 0.9586 | 0.518 | 0.540 |
| +6 s (17 s) | 13.306 | 1.0000 | 0.540 | 0.540 |

也就是说，在 S4 上冻结 H 最多只拿掉 H 执行端 4% 的剩余行程；到 +6 s 门已经完全打开，
H 状态继续涨对膜项**完全没有**额外作用。所以如果 H_CLAMP 什么都没改变，那是**预期之内**的，
不能读成 "H 与升级无关"——只能读成 "在这两个采样点之后，H 的执行端已经接近饱和"。

**D 侧量程很大。** `D = 1 − z` 直接缩放每个 E 细胞收到的 GABA：

| 快照 | D 均值 | D 中位 | D 最小 | D 最大 |
|---|---:|---:|---:|---:|
| +2 s | 0.1364 | 0.1458 | 0.0000 | 0.2635 |
| +4 s | 0.2752 | 0.2950 | 0.0841 | 0.3857 |
| +6 s | 0.4254 | 0.4404 | 0.2820 | 0.5080 |

在 S2 冻结 D，就是把抑制效能按住在 0.864 而不让它掉到 0.575。这是一个大干预。

**因此本轮的实际判别力主要来自 D_CLAMP 与 DH_CLAMP。** 这个先验不改变任何一条预注册主臂
——四条臂在两个快照上全部执行，H_CLAMP 的"零效应"本身就是要记录的结果。

## 6. 第一批实验：8 条 clamp

两个源快照 × 四条臂：

| arm | D / Z | H |
|---|---|---|
| `NAT` | dynamic | dynamic |
| `H_CLAMP` | dynamic | frozen |
| `D_CLAMP` | frozen | dynamic |
| `DH_CLAMP` | frozen | frozen |

每条从对应快照的 exact state 继续 **6000 ms**。

固定不变的：C0 graph、同一快照内完全相同的 full fast state、相同的未来外部输入、
`X = 1`、`U = 0`、`M = 0`、无 kick、无 reset、无参数阶跃、所有原始 cell/synapse 参数。

`NAT` 是**对应源快照的配对对照**，不是从 t=0 重跑的自然轨迹。

### 6.1 未来外部输入的配对性

`run_fcxr_loop` 每步消耗固定的两笔随机数：一次 `standard_normal()`（OU）和一次
`poisson(nu·dt, size=N)`。这两笔的**数量与形状都不依赖 D 或 H**，而 RNG 状态由 checkpoint
逐位恢复，所以同一快照下的四条臂天然收到逐位相同的未来外部输入。这一点必须用
`ExactInputHasher` 的 `external_input_sha256` 在四条臂之间**实测相等**来证明，不能只靠论证。

### 6.2 单次注册延长

如果某条臂在 6 s 窗口结束时仍然处于"高于间期带、未越注册饱和线、且末 2 s 仍在正漂移"，
它按 §8 判为 `RIGHT_CENSORED`，并**注册允许从它自己的 exact 末状态再续 4000 ms 一次**。
这是让实验回答问题，不是加一道门。第二次仍未解决则保持 `RIGHT_CENSORED`。

## 7. 每条臂的输出

- 10–20 ms 分辨率的 population rate（本轮取 **20 ms**）
- per-cell 1 s 窗 rate 的 q50 / q75 / q90 / q95 / q99
- 每个 1 s 窗中 rate > 250 / 300 / 350 / 400 / 450 Hz 的细胞比例
- 实测 refractory occupancy（≥ 0.9 × 500 Hz = 450 Hz 的细胞比例）
- active area（mm²，100 ms 空间图 + LC6A 锁定的 local rate 阈值）
- local q95 / q99 rate
- `H` 状态、H 门占用比例（`S̃ > 0.5`）、`gH`
- `gErec_raw`（tanh 之前的 raw 循环电导，也就是 H 的源）
- RC1 tanh 之后的有效循环电导 `gErec`
- `D`（`1 − z`）
- E / I signed current decomposition
- late rate slope
- classifier label
- exact future-input hash

## 8. 分类语义

只使用这八个标签：

`ESCALATING_SATURATION`、`BOUNDED_STATIONARY`、`BOUNDED_OSCILLATORY`、`LOW_STATE`、
`SILENCE`、`AFTER_DISCHARGE`、`NUMERICAL_FAIL`、`RIGHT_CENSORED`。

### 8.1 不得复用 LC6A 的 `classify_high_state`

LC6A 的 `classify_high_state` 要求 rate / D / H **三条** drift 同时平坦才判 bounded。
在本轮的 clamp 下 D 和 H 逐位恒定，它们的 Theil–Sen 斜率恒等于 0，于是这两个判据被
**钳制动作本身**自动满足 —— 直接复用会把干预制造成结论。另外它以"进入时刻"为起点判读，
而本轮所有臂**从高态内部起跑**，根本没有窗口内的 onset。两条理由都指向：本轮必须写新的分类器，
**判据里不出现任何 D 或 H 的斜率**。

### 8.2 注册阈值（沿用 LC6A 的同一批线，便于两轮对读）

- 全局饱和：任一完整 1 s 窗均值 ≥ `250.0 Hz`（`SAT_CEILING_HZ`）
- 局部饱和：任一 1 s 窗中 ≥ `450 Hz`（0.9 × 500 Hz refractory ceiling）的细胞比例 ≥ `0.05`
- 间期带上沿：`roll_hi = 9.7382291667 Hz`（LC1 seed-1 classifier snapshot 的注册值）
- rate 漂移门：末 2 s 归一化 Theil–Sen 斜率 CI 上界 ≤ `0.05 s⁻¹`
- 静默比例门：末 2 s 中 population rate ≤ `roll_hi` 的 20 ms bin 比例 ≥ `0.25` 记为 bursty

### 8.3 判决树（逐条求值，先命中先返回）

1. 出现非有限值 / 引擎抛数值异常 → `NUMERICAL_FAIL`
2. 未跑完注册时长（资源停机 / bundle 未完成）→ `RIGHT_CENSORED`（`INCOMPLETE_REGISTERED_WINDOW`）
3. 越过全局或局部饱和线 → `ESCALATING_SATURATION`
4. 末 500 ms 一个 E spike 都没有 → `SILENCE`
5. 末 2 s 均值 ≤ `roll_hi` → 若全程高于 `roll_hi` 的累计时长 < 2000 ms 则 `AFTER_DISCHARGE`，否则 `LOW_STATE`
6. 末 2 s rate 漂移门未过（仍在正漂移）→ `RIGHT_CENSORED`（`STILL_ESCALATING_AT_WINDOW_END`），
   适用 §6.2 的单次注册延长
7. 其余（末 2 s 全程高于间期带、未越饱和线、漂移门通过）：
   静默 bin 比例 ≥ 0.25 → `BOUNDED_OSCILLATORY`；否则 `BOUNDED_STATIONARY`

求值顺序里，**饱和与"掉回间期带"都是已解决的结局，必须排在"仍在升级"这个删失判据之前**；
否则一个正在缓慢恢复的低态会因为斜率为正而被误判成删失，而不是被如实报成 `LOW_STATE`。

### 8.4 bounded 的边界

`bounded_candidate = label ∈ {BOUNDED_STATIONARY, BOUNDED_OSCILLATORY}`。

每条臂**无条件**记录 `perturbation_return_tested: false`。§8 的判决只回答
"这个分支在这段窗口里存在吗"，不回答"弱扰动之后还回得来吗"。后者是第二阶段的事，
gain fork 不是第一阶段的必要条件。

### 8.5 为什么要单列 `BOUNDED_OSCILLATORY` 和静默比例

FCXR-LC3 有过一次教训：一个看起来"持续的发作载体"，用 300 ms 滚动均值读是稳的，
实际结构是每 86 ms 从**完全静默**重新点火的爆发串，57% 的时间三万两千个细胞零放电。
所以本轮的 population rate 必须在 20 ms 分辨率上读，且静默 bin 比例与最长静默段长度
必须与标签一起报出来 —— 一个真正的稳态高分支不应该反复回到间期水平。

## 9. 三类硬 gate（只有这三类）

**G1 exact-state / input / hash integrity** —— checkpoint 的 `state_hash` 载入后必须复现；
同一快照四条臂的 `external_input_sha256` 必须逐位相等；mechanism source hash 在跑的全程不得漂移；
graph 与 config 的 sha256 必须记录且不得漂移。

**G2 numerical integrity** —— 非有限值、负电导、`fail_on_clip` 触发都立即失败。

**G3 resource / checkpoint integrity** —— MemAvailable 不足以支撑当前并发数时不再提交新 worker；
swap 增长触线时停止提交；已完成产物不删。滚动 exact checkpoint 必须可恢复。

**科学结果不是 gate。** saturation、silence、low state、bounded state 都是结果，
任何一个都不得中止其余的预注册主臂。

## 10. 首轮结果的自动决策

**A. `DH_CLAMP` 仍然 saturation** —— 解释为：在这个被冻结的慢状态下，fast substrate 没有有界分支，
或者源快照已经越过了所有有界区。动作：**不跑** natural-path atlas；出阶段报告；
落 H-EFF / H-CAP 的独立 spec/plan（只写不执行），等待对核心结果的审阅。

**B. `DH_CLAMP` bounded** —— 说明有界态至少可以存在。继续比较：
`H_CLAMP` bounded 而 `D_CLAMP` runaway → 主驱动是 H 的持续招募；
`D_CLAMP` bounded 而 `H_CLAMP` runaway → 主驱动是 D/Z 的持续耗竭；
两个单 clamp 都 runaway 而 `DH_CLAMP` bounded → D/H 慢流协同；
两个单 clamp 都 bounded → 任一慢变量继续演化都足以穿过载体区。
此时自动进入 §11 的 natural-path atlas。

**C. `DH_CLAMP` silence / low** —— 说明源状态依赖慢变量维持，但**不能**据此证明有界分支不存在。
需要在相邻较早快照（onset+1 s，即 `t = 240000` 那个状态）补一个点，不得直接判 no-go。

## 11. 条件性 natural-path atlas（只有首轮证明冻结慢态可以避免 runaway 才跑）

先生成或重跑六个带明确时间戳的 D/H 场：onset−1 s、onset、onset+1 s、onset+2 s、onset+3 s、onset+4 s。

每个场用三种 initialization：
1. **path-native** —— 该时刻自己的完整 fast state
2. **locked-low** —— 所有 atlas 点复用**同一个** interictal fast state，只替换目标 D/H 场
3. **locked-high** —— 所有 atlas 点复用**同一个** high fast state，只替换目标 D/H 场

low / high 的 graph、配置与未来输入必须一致；**不得**让每个 atlas 点用不同的 low/high 初始化。
path-native 只作自然轨迹可达性读数。单 seed 时输出 classification，**不写 probability**。

第一轮只做一维 natural-path atlas，不直接做 5×5 或 7×7。只有出现 low/high 初始化收敛到不同状态、
或出现有界窗口时，才在该局部补 3×3 二维 D/H grid；3×3 仍有信息才允许扩 5×5。

## 12. H-EFF / H-CAP 的授权边界

第一批 8 条 clamp **不得**混入 H-EFF。若 `DH_CLAMP` 仍 runaway，只允许写下一轮 spec/plan，
不得直接偷偷改变 H。两者必须分开：

- **H-EFF** —— 只改变 H 的 **source**；检验 raw recurrent source 是否造成过度招募。
- **H-CAP** —— 只改变 H 的 **output transfer**；检验早期进入支持与晚期 runaway 能否分离。

不得把 source 和 transfer 同时改掉。任何 H-EFF 定义必须明确：source 是**不含** `gH` 的
RC1-effective recurrent drive，还是**含** `gH` 后的总 effective drive；必须避免同一步的代数循环；
必须保持一帧因果延迟；必须 off-by-default parity。

## 13. 本轮不授权

CP-S / CP-L center-preserving surround、threshold heterogeneity、global E→I tail、`U`、`M`、
full lifecycle、大型 D/H grid、spatial stimulation atlas、termination pulse、multi-mechanism combination。

正确顺序：clamp/atlas → 判断分支是否存在 → H-EFF/H-CAP 或 CP/HET → bounded carrier →
重新标定 U（即 LC6D） → lifecycle → spatial controllability atlas。

## 14. 产物

```
results/topic4_sef_hfo/fcxr_lc6b_frozen_slow_atlas/
  STATUS.md
  run_manifest.json
  clamp_fork_summary.json
  resource_log.jsonl
  figures/
    README.md
    lc6b_clamp_forks.png
    lc6b_clamp_forks.pdf
```

条件性（仅 §10-B）：`natural_path_atlas.json`、`figures/lc6b_natural_path_atlas.{png,pdf}`。

归档：`docs/archive/topic4/fcxr_lc6b_frozen_slow_causal_atlas_<date>.md`。

图一必须直接展示：两个源快照；四条 clamp 臂对齐到 fork 时刻的 rate；rate 分布；
H 门 / D；以及最终动力学标签。**不要先画大而复杂的论文图。**

## 15. 报告口径

阶段报告必须先用白话回答：固定 D/H 后还会不会继续冲顶？若不会，是固定 H、固定 D、还是两者都固定才有效？
中间高态是真的存在，还是只得到低态/静默？结果说明下一步该改 H、D slow flow，还是 fast transfer？
哪些结论只属于 canonical graph/noise？

并明确写出：`engineering status`、`scientific status`、`termination tested?`、`lifecycle tested?`、
`remaining largest uncertainty`、`next authorized action`。
