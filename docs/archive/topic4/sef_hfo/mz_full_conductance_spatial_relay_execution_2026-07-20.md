# MZ-FCXR full-conductance + spatial-relay 执行报告

日期：2026-07-20
分支：`codex/topic4-mz-conductance`（工作树 `.worktrees/topic4-mz-conductance`）
设计合同：`docs/superpowers/specs/2026-07-20-topic4-mz-full-conductance-spatial-relay-design.md`
状态入口：`results/topic4_sef_hfo/mz_full_conductance_spatial_relay/STATUS.md`

## 0. 朴素话摘要（测了什么 / 怎么测的 / 到哪步）

**测了什么。** 现有 E1146 40k 网络里，把兴奋神经元收到的兴奋输入（AMPA）从"直接加一个电流"改成"真正的电导"——
它像真实突触一样，膜电位越接近兴奋反转电位 `E_E`，同样的输入能推的劲就越小（会饱和）。目的：看这个更真实的
快系统本身，在固定的慢状态下会不会出现一个"有限的高活动支"（不是要么静息、要么冲到天花板，而是中间存在一个能停住
的高活动态）。只有这个成立，才加第二味料：一个"只认长时间持续招募、基本不理会普通短尖波"的 E→E 接力资源 `x_j`，
让传播前沿走过的地方留下一段"暂时接不动力"的空间尾迹。

**怎么测的（工程 + gate 阶梯）。** 先把引擎改干净、且默认关掉时和旧版逐比特一样（re-bless）；再定 full-conductance 的
力匹配系数 `c_E`（只允许 0.85 / 1.0 / 1.15 三档），要求它还能保留旧的间期放电工作点；然后冻结慢状态扫快系统找高支；
再上持续度传感器与接力资源。任何一个最早的 gate 干净 no-go 都是合格交付，不为了跑出漂亮结果加旋钮。

**到哪步（一句话结论）。** 工程层（§1 Stage 0A）已验收；但 full-conductance 本身在锁定参数下**保留不了正常间期节律**
（§2 Stage 0B workpoint **NO-GO**）——偏弱档静息、偏强档过热+触顶，中间没有甜区，所以按 stop rule 停在第一关，
"接力资源"没有加（§3 层判决、§4 结论与下一步）。

## 1. Stage 0A — engineering green（ACCEPT）

- 改动文件：`src/snn_engine/kick_probe.py`（guarded）+ `src/snn_engine/mz_slow_vars.py`（非 guarded）。全部 off-by-default。
- full-conductance 膜（`membrane_mode="full_conductance"`）：E 细胞 AMPA 折成朝 `E_E=58` 的 reversal-aware 电导，`V_match=18`
  处由 `c_E` 力匹配；`I_E_rec`（recurrent AMPA accumulator，复用 M4 的 `track_rec`）拆 feedforward/recurrent 仅供诊断，
  x 调制在 scatter 源端故 `I_E` 已含 → `g_E_ff+g_E_rec == c_E·I_E/(E_E−V_match)`。I 细胞保持 literal current。
- persistence-gated E→E relay（`use_x`）：Hz 单位 sensor `y_j`（exact decay + 每 E spike 跳 `1000/tau_y`）经 Hill(n=4) gate 驱动
  `x_inf`；`slow.step` 在更新 y/x **之前**快照 `x_j(t−)`，scatter 用该值只缩放 E→E 边（复用 `ee_std_apply`），与 M1 `ee_std` 互斥。
- **parity 证据**：pre-edit fixture 下 slow=None / partial conductance / M1 三旧路逐比特一致；全回归 **115 passed**（含 a1c
  frozen-baseline T1/T2/T9/T10、scatter、ee_std、shunting、onset-dynamics、mz-conductance），FCXR 专项 **19 passed**（force-match、
  x(t−) 因果、x 只作用 E→E、mutex、determinism、relay-off identity、bless）。
- **re-bless**：仅 `kick_probe.py` sha → `febba30005cb…`，其余 5 engine hash 未动。
- commits：`f73fcfe`(design-lock) → `40246a8`(engine) → `cd5f483`(tests)。
- Stage 0A smoke（L=20 seed1 c_E=1.0 spontaneous 1000ms）：RSS peak **6.79 GiB**（预算内），finite。

## 2. Stage 0B — workpoint（NO-GO）

一句话：饱和电导把兴奋性窗口挪了位，允许的 `c_E` 三档跨"静息↔过热"，没有一档同时数值安全且保留 accepted 间期节律
（详见 §2.1 诊断 / §2.2 seed1 正式判决 / §2.3 seed3 确认）。

### 2.1 full-conductance 兴奋性诊断（探索性，非正式 gate）

把 AMPA 从"加性电流"改成"朝 E_E=58 饱和的电导"后，兴奋性对 `c_E` **极其敏感**，而且是在"静息"与"过热"之间
陡切换。诊断：L=20 seed1 spontaneous、per-250ms bin（`scratchpad/fc_clip_timing.py`）：

- **c_E=0.85**：**几乎完全静息** —— peak 0.3–0.6 Hz、mean ~0.00 Hz（2500ms 内基本无 spike）、clip=0、tau_eff_min=10.2ms。
  远达不到间期工作点（需要 returning events、峰值 ~45Hz、participation ~0.037）。原因：饱和电导使 driving force
  `E_E−V` 在接近阈值时变小，recurrent 累积不起来，网络点不着。
- **c_E=1.0**：**过热/持续** —— smoke 1000ms peak 131.9Hz；2500ms 直跑因持续放电而极慢（9+ min，vs 0.85 的 103s），
  且启动即有 ~0.6% cell 触 conductance cap（tau_eff 落到 2dt 下限）。不是稀疏的间期事件。
- **c_E=1.15**：更热（未直测完）。

即 **允许的 `c_E∈{0.85,1.0,1.15}` 恰好跨越"静息 ↔ 过热"，中间没有落在间期工作点上的档**。正式判决用 T=8s +
事件检测器的 workpoint（§2.2），不因诊断提前下结论。

**机制归因（reasoning，已被 parity/force-match 测试锁住不是 bug；但"谁是主因"待 §2.4 通路拆分实验判）**：
full conductance 是一个**状态依赖变换**，不是简单"把外源输入削弱"——在 `V_match=18` 力匹配处两者相等，但在别处
`g_E(E_E−V)/I_E = (E_E−V)/(E_E−V_match)`：在静息附近（`V=0`）该比值 = `58/40 = 1.45`，即电导化在低膜电位区反而
**比原电流更强**（更容易点着 → 更热）；接近阈值时才变小（饱和）；同时额外 `g_E` 进入 `tau_eff = tau_m/(1+g_E+g_I)`
分母 → 膜时间常数变短 → 对涨落响应更快。这三点合起来把兴奋性窗口整体挪位，accepted 间期区间落进了测试的 `c_E`
三档的"静息↔过热"之间。（注：数值层面 clip 与 `tau_eff=0.2ms` **不是两条独立证据**——总电导被 clip 到 cap=99 后
`tau_eff = 20/(1+99) = 0.2ms`，二者是同一个 cap 事件的两种读法。）

### 2.2 workpoint 正式判决（L=20 seed1 T=8s，Z/M/global off）

reference（同 run 的 current-model slow-off，已复现 accepted pilot）：n_returning=**20**、duration_median=**28.0ms**、
participation_median=**0.0358**、peak_rate_median=**42.2Hz**（pilot：20 / 28 / 0.0375 / 45.3 —— 基本一致，参照可信）。

c_E 判决（seed1，Z/M/global off，`fail_on_clip=False` 让每格跑完再判，numerical 判 settled window）：

| c_E | phenotype | n_returning（参照 20） | settled clip | settled_safe | all_bands | 判读 |
|---|---|---:|---:|---|---|---|
| 0.85 | suppress | **0** | 0.0% | True | False | 太弱，无间期事件 |
| 1.0  | interictal_like | **58**（2.9×） | 0.93% | **False** | False | 过活跃 **+** 持续触顶 |
| 1.15 | interictal_like | 45（2.3×） | **26.7%** | **False** | False | 过活跃 **+** 严重触顶 |

**没有任何 `c_E` 同时"数值安全（settled clip=0 且 tau_eff≥2dt）"且"事件画像落进参照 band"**：`c_E=0.85` 抑制到 0
事件；`c_E=1.0` 58 个事件（超参照 2.9 倍）且 ~0.9% cell 持续触 conductance cap；`c_E=1.15` 45 个事件且 26.7% cell
持续触顶。图：`results/topic4_sef_hfo/mz_full_conductance_spatial_relay/figures/stage0_workpoint.png`。

**判决：workpoint gate NO-GO（`no_go_no_safe_workpoint_c_E`）**。drive=0.6 与 `c_E∈{0.85,1.0,1.15}` 均锁定、
不扩 bracket，按 Stage 0B / 设计 §11 stop rule，full-conductance route 在 workpoint 关就停。seed3 确认见 §2.3。

### 2.3 seed3 确认（seed-independence）

seed3 workpoint（c_E=0.85,1.0；reference current-model slow-off n_returning=22，复现 pilot seed3=22）**同向 NO-GO**：

| c_E | phenotype | n_returning（参照 22） | settled clip | settled_safe | all_bands |
|---|---|---:|---:|---|---|
| 0.85 | suppress | **0** | 0.0% | True | False |
| 1.0  | expanded_bounded | **56**（2.5×） | 0.88% | **False** | False |

与 seed1 完全同一模式：`c_E=0.85` 静息 0 事件、`c_E=1.0` 过活跃（56 vs 58）+持续触顶。**silent↔hot 非 seed1 特异，
两个 primary seed 一致 NO-GO**（seed3 verdict = `no_go_no_safe_workpoint_c_E`）。机制（tau_eff 变短 + 饱和）本就 seed-独立，
数据佐证。

### 2.4 通路归因 2×2（feedforward vs recurrent conductance，seed1，c_E=1）

审阅意见（2026-07-20）：当前只跑了外源+recurrent AMPA **都**改成电导（arm D），"外源饱和是主因"缺独立对照。
补一个固定 `c_E=1` 的 2×2（在 workpoint 关闭 Z/M/X/global）：

| arm | feedforward(外源) AMPA | recurrent E→E AMPA | 作用 |
|---|---|---|---|
| A | additive | additive | accepted reference（partial conductance） |
| B | conductance | additive | 只改外源 |
| C | additive | conductance | 只改 recurrent（推荐下一版主候选） |
| D | conductance | conductance | 复现当前 NO-GO |

四臂共享 `V_match` 力锚（`ampa_drive + gE·(E_E−V_match) == c_E·I_E`，已单测），只在 off-`V_match` 的状态依赖上不同。

**结果（seed1，c_E=1，参照 current-model slow-off n_ret=23 / participation 3.75%）：**

| arm | 改动 | n_ret | participation% | settled clip | safe | all_bands | preserves |
|---|---|---:|---:|---:|---|---|---|
| A add/add | 无（参照） | 23 | 3.75 | 0 | ✓ | ✓ | **YES** |
| B ff-cond | 只外源 | **52**(2.3×) | 6.52 | 0 | ✓ | ✗ | over-active |
| C rec-cond | 只 recurrent | 30 | 4.35 | **0.48%** | ✗ | **✓** | clips |
| D cond/cond | 两路（=NO-GO） | 58 | 7.47 | 0.93% | ✗ | ✗ | both |

**判读（干净的因果拆分，且是可加的——D ≈ B 的过活跃 + C 的 clip，不需要交互项）：**

- **A 复现 accepted workpoint**（23 事件、bands、safe）——验证 2×2 setup 可信。
- **feedforward 电导 → 过活跃**（arm B：52 事件 2.3×、off-band，但 clip=0/数值安全）。事件数暴涨来自外源电导——正是
  状态依赖 1.45×-at-rest 效应：外源在静息附近推得更狠 → 更多点火。
- **recurrent 电导 → clip，但事件画像基本保住**（arm C：**bands=True**、30 事件/participation 4.35% 都近参照，但 0.48%
  cell 触顶 → settled_safe=False）。clip 来自 recurrent 电导 `g_E_rec = c_E·I_E_rec/40` 随本地 recurrent 驱动**无界增长**
  ——事件峰值时少数 cell 的 recurrent 电导过冲 cap（driving force 饱和的是"劲"，电导本身不饱和）。

**对审阅推荐（arm C = 只 recurrent 电导）的判决：部分验证。** arm C **保住了间期 workpoint 的事件画像**（bands=True），
证明"外源保持 additive"这一方向是对的（外源电导才是压垮 baseline 兴奋性的那半）；但它暴露一个**新的、和原 NO-GO
不同的障碍**——recurrent 电导过冲 clip（0.48%，比 D 的 0.93% 小）。按 §6 step 3，arm C **未完整通过**（数值不安全），
故**不自动跑 seed3**；但它是目前最接近的一档（画像已保住，只差 clip）。剩余问题从"静息↔过热"变成了"recurrent 电导
本地过冲"——更局部、更可能可治（例如给 recurrent 电导本身一个饱和上界，或核查这 0.48% 是否可容忍），由用户拍板。

## 3. 分层判决（更新）

| 层 | 判决 | 备注 |
|---|---|---|
| engineering green | **ACCEPT** | §1；parity 115+19 tests、re-bless |
| workpoint pass | **NO-GO（seed 1+3 一致）** | §2.2/§2.3；两个 primary seed 都无 c_E 同时数值安全+匹配 band |
| fast-topology gate | **不执行** | 被 workpoint NO-GO 门控（Stage 1 前置未过） |
| sensor gate | 不执行 | 同上 |
| temporal lifecycle | 不执行 | 同上 |
| spatial lifecycle | 不执行 | 同上 |

## 4. 当前安全科学结论 + 下一步

**安全结论（措辞受限，边界收窄）：** 在锁定的 E1146 / L=20 / drive=0.6 衬底上，把 E 细胞兴奋输入从 additive current
改成朝 `E_E=58` 饱和的 full conductance 后，**测试的三个离散档 `c_E∈{0.85,1.0,1.15}` 都不能复现 accepted 间期工作点**
（seed 1+3 一致）：`c_E=0.85` 静息（0 事件）、`c_E≥1.0` 过活跃（事件数 2–3× 参照，participation seed1 c_E=1.0 约 7.5% /
c_E=1.15 约 23%——过热但**不是全片同步**）且触 conductance cap。**只能说这三个注册点失败，不能说 `0.85–1.0` 连续区间
里不存在很窄的临界窗口**（当前设计禁止插值）——这不影响预注册 gate 失败，但限定了机制解释的范围。

**禁止写成**：已得到发作态 / 极限环 / Hopf / 双稳态 / 有限高支 / 完整 seizure lifecycle（Stage 1 未执行）。

**2×2 通路归因已跑完（§2.4），答案是分工而非单一主因**：**feedforward 电导 → 过活跃（事件数暴涨）**、**recurrent
电导 → 本地过冲 clip（但事件画像保住）**，两者可加地凑成 arm D 的 NO-GO。所以"feedforward 是唯一主因"不成立；准确说是
**外源电导负责过活跃、recurrent 电导负责 clip**。

**下一步（由用户拍板，一次只放开一个轴、重新预注册）：** 审阅推荐的 arm C（只 recurrent 电导、外源保持 additive）
**部分验证成功**——它保住了间期 workpoint 的事件画像（bands=True），证明"外源保持 additive"方向正确；剩下唯一障碍是
recurrent 电导过冲的 0.48% clip（比 D 的 0.93% 小、且更局部）。按 §6 step 3 arm C 未完整过（数值不安全），故未自动跑
seed3。建议用户从以下里选一（只选一个）：**(a)** 给 recurrent 电导本身一个饱和/软上界（治 clip，最贴合"要有限高支"的
初衷）；或 **(b)** 核查这 0.48% clip 是否可容忍（若只是峰值瞬时少数 cell，放宽 numerical gate 后 arm C 可能直接过）；
或 **(c)** 先不动 clip、用 arm C 进 Stage 1 看有没有有限高支（clip 只在事件峰值、可能不影响 topology 判读）。**不建议**
去调 drive 或扩 c_E——2×2 已把问题从"静息↔过热"收窄到"recurrent 电导本地过冲"这一个更小的点。

## 5. 最小执行路线（审阅 §6）

1. **（已做）** 修 workpoint 候选 gate 漏 `all_bands`（+回归测试 `tests/test_fcxr_workpoint_gate.py`），并让
   full_conductance 在 X 关闭时也记录 `gEff/gErec`（Stage 0 才能归因 ff vs rec）。commit `2a97311`。
2. **（已做）** `seed1, L=20, T=8s, c_E=1` 跑 A/B/C/D 四臂（runner `pathway`）——见 §2.4：ff→过活跃、rec→clip、可加。
3. **（不触发）** arm C bands=True 但 settled_safe=False（0.48% clip），未同时满足两条 → **不跑 seed3**。
4. **（待用户）** arm C 若把 clip 治掉再过 workpoint，方程锁为 `tau_m V̇ = −V + I_E^{ff} + g_E^{rec}(E_E−V) + g_I(E_I−V)`，
   再重进 Stage 1 查有限高支；高支存在后才做 `y/x`。
5. **（待用户）** 现在的岔口不是"调 drive/扩 c_E"（2×2 已否掉那个方向），而是"怎么治 recurrent 电导过冲"——见 §4 的
   (a)/(b)/(c) 三选一，一次只开放一个轴、重新预注册。

这个 2×2 比"直接相信 feedforward 是主因"多跑了 3 个 arm，但把 NO-GO 从"full-conductance 不行"精确拆成"外源电导过活跃 +
recurrent 电导 clip"，并把审阅推荐的 recurrent-only 方向从"合理假设"变成"画像已验证、只差 clip"。
