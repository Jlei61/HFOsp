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

**判读（因果分工，但"可加"只对事件画像成立、对 clip 不成立——审阅 2026-07-20 修正）：**

- **A 复现 accepted workpoint**（23 事件、bands、safe，score 0.27）——验证 2×2 setup 可信。
- **feedforward 电导 → 过活跃**（arm B：52 事件 2.3×、off-band，但 clip=0/数值安全）。事件数暴涨来自外源电导——正是
  状态依赖 1.45×-at-rest 效应：外源在静息附近推得更狠 → 更多点火。
- **recurrent 电导 → clip，事件画像近参照**（arm C：**bands=True**、30 事件/participation 4.35%，但 0.48% cell 触顶 →
  settled_safe=False）。clip 来自 `g_E_rec = c_E·I_E_rec/40` 随本地 recurrent 驱动无界增长。
- **可加性只在事件画像层成立，clip 层有明显活动依赖交互**：`Δ_int = D−B−C+A`，事件数 Δ_int=−1（近似可加），但
  **clip Δ_int = +0.447 个百分点**（D=0.928% ≈ 2×C=0.481%）。即 feedforward 自己不 clip，但它把活动抬高后**显著放大**
  了 recurrent 过冲。准确表述：**工作点画像近似可加的通路分工，但高电导尾部是活动依赖交互，不是简单叠加。**

**对审阅推荐（arm C = 只 recurrent 电导）的判决：方向对，但 arm C 只是"边界候选"，不是"只差一个可忽略 clip 的工作点"。**
arm C 保住四维 bands（方向验证："外源保持 additive"是对的），但它**不在工作点中心**：baseline-distance score **1.04**（arm A
只 0.27）、population mean rate **7.81Hz**（arm A 5.01，高 56%）、事件数 30 卡在 1.5× 上限。且 **0.48% clip 很可能是主 recurrent
空间模态在少数高增益节点的局部失稳**（7/4000 帧、~154 cell、gErec 群体均值 P95 仅 2.51 vs cap 99 = 重尾，不是全网逼近 99），
不是全局数值瑕疵。**硬 clip 本身又可能人为制造我们正想找的"有限高支"**（不可微 fast negative feedback）——所以 §4 原来的
"容忍 clip"/"带 clip 进 Stage 1" 两个选项**撤回**。下一步见 §6 FCXR-RC1：先做 mode-resolved clip audit + dt 收敛，再决定
recurrent-only smooth saturation。

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

**2×2 通路归因已跑完（§2.4）：分工，事件画像近似可加、但高电导尾部有活动依赖交互**：**feedforward 电导 → 过活跃**、
**recurrent 电导 → 本地过冲 clip（事件画像保住）**；事件数 `Δ_int=−1`（近可加），但 **clip `Δ_int=+0.447pp`（D≈2×C）**——
外源自己不 clip，却把活动抬高后放大了 recurrent 过冲。所以"feedforward 是唯一主因"和"两路完全可加"都不成立。

**下一步（审阅 2026-07-20 修正，不是"三选一"）：** 主线方向确定为 **external additive + recurrent E→E conductance**（arm C
方程）。但 arm C 只是**边界候选**（score 1.04 vs A 0.27、mean rate 高 56%），不能直接进 Stage 1——0.48% clip 很可能是主
recurrent 空间模态在少数高增益节点的局部失稳，且硬 clip 本身会人为制造"看似有界"的高态（不可微 fast negative feedback）。
**原来的 (b) 容忍 clip、(c) 带 clip 进 Stage 1 两个选项撤回。** 关键科学问题变成：**这 0.48% 是不是控制整个空间动力学的
dominant localized mode？** 执行序列见 §6（FCXR-RC1）：**mode-resolved clip audit + dt 收敛 → 确认稳定局部模态后做
recurrent-only smooth saturation → 重过 seed1/3 workpoint → 带 eigenvalue/eigenvector readout 进 Stage 1 → 有限高支成立前不加 X。**
**不调 drive、不扩 c_E**（2×2 已否掉那条路）。

## 5. FCXR-RC1 执行线（审阅 2026-07-20 §6/§7）

已做：workpoint all_bands gate 修复（`2a97311`）、gEff/gErec always-on trace（`2a97311`）、2×2 通路归因（§2.4）。
下面是审阅指定的、进 Stage 1 之前必须先做的 clip 身份+收敛审计（本报告 §6 记录 Stage A/B 结果，代码在 `run_topic4_mz_fcxr.py`
的 `clip-audit` 命令 + `src/topic4_mz_fcxr_modes.py`）。

- **Stage A（mode-resolved clip audit，不改方程）**：只重跑 arm C，加 O(N) observer（per-cell `clip_count`、`max_raw_gErec`、
  首末 clip 时刻），不存新的 T×N 矩阵；post-hoc 算 `W_EE`（E→E 块）主 left/right 特征向量 + leading singular vectors + IPR，
  以及 recurrent in/out-strength、core/source/sink/axis 标签；核 clip cell 与各模态热点/高中心度/低 V_th core 的 overlap，
  以及 clip 属于哪个 returning event。硬判读：同一批 core/高中心度反复 clip + 高 mode overlap = 真实局部 recurrent 模态；
  clip 身份跨事件随机 + 低 overlap = 随机尾/离散化；沿轴传播 = 可能是要找的空间主模态但增益过强；近全局均匀 = arm C 不适合空间机制。
- **Stage B（数值收敛，不许"直接容忍"）**：arm C 做 dt/cap 诊断——现 `dt=0.1/cap=99`、无缩放高 cap/无限 cap、以及 `dt=0.05`，
  比工作点四维 / clip-raw-g 尾 / leading-mode 位置+IPR / event rate+duration 的 dt 漂移。若减小 dt 不收敛，0.48% 不能算可容忍。
- **Stage C（只有 A 确认稳定局部模态才做）**：recurrent-only smooth saturation `g_rec_eff = g_sat·tanh(g_rec_raw/g_sat)`
  （小输入斜率 1 保工作点、高输入平滑饱和、`S'` 随活动下降真正压高态 recurrent eigenvalue、只作用 recurrent E→E、保空间异质）。
  `g_sat` 只由 arm C slow-off 原始 `gErec` 分布锁（主值 + ±20% sens，不做参数海）。重过：seed1/3 workpoint、零 hard-clip、
  主模态 IPR/位置没被抹成全局均匀、arm A/off parity。
- **Stage D（Stage C 过后才进原 Stage 1）**：冻结 Z/X 沿 D 查 low branch / finite high branch / leading eigenvalue 实部 /
  complex pair / eigenvector-IPR-vs-D / low-high hysteresis。理想：low 态主模态稳定但可激 → Z 耗竭把主模态推近 0/降低有限幅
  ignition 阈 → 高态 smooth saturation 使 `S'` 降、主模态重新有界 → X 用 `W·diag(x)` 进一步降传播模态并促退出。**有限高支成立前不加 X。**

**科学问题（审阅 §7）**：不是"0.48% 能否容忍"，而是"这 0.48% 是不是控制整个空间动力学的 dominant localized mode"。

## 6. FCXR-RC1 Stage A/B 结果

（clip audit + dt 收敛结果填充中——见 `run_topic4_mz_fcxr.py clip-audit`、`results/.../clip_audit_*` 与 `figures/clip_audit.png`。）
