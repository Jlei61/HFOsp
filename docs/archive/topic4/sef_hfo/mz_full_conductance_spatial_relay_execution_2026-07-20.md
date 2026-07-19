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

**机制归因（reasoning，已被 parity/force-match 测试锁住不是 bug）**：full conductance 把兴奋输入折成电导后，
额外的 `g_E` 进入 `tau_eff = tau_m/(1+g_E+g_I)` 的分母 → 膜时间常数变短 → 对输入涨落响应更快、动力学更"敏感"；
同时 driving force `E_E−V` 在接近阈值时变小（饱和）。二者叠加把兴奋性窗口整体挪动：在锁定的 drive=0.6 下，
accepted 间期区间落进了 `c_E` 的"静息↔过热"缝里。drive 与 `c_E` 均锁定、不许扩 bracket，故若 workpoint 确认
无匹配档即 NO-GO。

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

**安全结论（措辞受限）：** 在锁定的 E1146 / L=20 / drive=0.6 衬底上，把 E 细胞兴奋输入从 additive current 改成朝
`E_E=58` 饱和的 full conductance 后，**在允许的 `c_E∈{0.85,1.0,1.15}` 力匹配 bracket 内，没有一个档能复现 accepted
的间期放电工作点**：`c_E=0.85` 把网络压成静息（0 事件）、`c_E≥1.0` 把网络推成过活跃（事件数 2–3× 参照）且开始/持续
触 conductance cap（tau_eff 掉破 2dt）。机制归因（已被 parity/force-match 测试锁住不是 bug）：饱和电导让 `tau_eff`
变短、driving force 接近阈值时变小，兴奋性窗口整体挪动，accepted 间期区间掉进 `c_E` 的"静息↔过热"缝里。

**禁止写成**：已得到发作态 / 极限环 / Hopf / 双稳态 / 有限高支 / 完整 seizure lifecycle（Stage 1 未执行，无从声称）。
本结论只到"full-conductance 快系统在允许 bracket 内不保留间期工作点"。

**下一步（唯一建议，交由用户拍板，非自主继续）：** full-conductance route 在 workpoint 关 NO-GO。若要继续 MZ-FCXR
的科学目标（找有限高支 + persistence-gated 空间尾迹），需要用户先决定放开哪一个当前锁定项之一（且只放开一个、
重新预注册）：例如 (a) 允许更细/更宽的 `c_E`（当前禁止扩 bracket）；或 (b) 只让 recurrent E→E 成电导、feedforward
外源 AMPA 保持 additive（当前设计让二者都成电导，正是它压掉 baseline 兴奋性的主因）；或 (c) 调整 drive 重配工作点
（当前 drive 锁 0.6）。**在没有新的预注册决定前不继续**——一个正确 gate 的 NO-GO 已是合格交付。
