# Topic 4 — MZ 整网状态对齐有限时间空间模式追踪（archive）

Branch `codex/topic4-mz-m-eigenmode-tracking` · base `2c4f82b`（`codex/topic4-mz-direct-spatial-modes`）· 2026-07-21
Spec: `docs/superpowers/specs/2026-07-21-topic4-mz-m-eigenmode-tracking-design.md`
Tier = model-side mechanism analysis（NOT seizure validation；每个表型是 detection label）。

> **状态**：3 seeds（1/3/4）全流程完成并目视验收。parity 3/3 bit-identical；15 状态（3×5）全 resolved & settled。
> §5/§VI.7 纪律：模型侧机制分析，per-seed 描述聚合，不作临床/发作主张。

---

## 0. 一句话结论（n=3）

我们在那张约 4 万神经元、电流型 spiking 网络的**已验收 z+m 高位平台工作点**上，沿它自发爬升-停在高位的慢状态
轨迹，逐点用**完全相同的一记局部电流"踢"**去问："同一刺激在不同慢状态下，网络的有限时间空间响应长什么样、变没变
方向、还是被压住了？"三条承重结论：

1. **固定踢被慢状态压住、从不沿轴向汇招募**（P1）：baseline 三 seed 都在源头出一个局部响应（norm 1.95/0.52/0.66），
   一进 approach 就被压到近零（三 seed 一致），到 settled 平台三 seed 分化（0.22 / 0 / 1.72）；**remote_sink 恒为 0、
   无合格 arrival front**——m(+z) 是抑制器/门，不是轴化器。
2. **没有稳健可辨识的有限时间线性算子**（P2/P3）：15 个状态 0/15 过严格门（disc 0.27–0.55、split-half 0.41–0.70），
   0 对相邻双可辨识 → 无 mode trajectory。干净的 bounded negative，跨 seed/状态一致。
3. **m 是平台上的刹车，力度按"离点火边缘多近"跨 seed 分化**（P4）：从 seed3 可忽略（平台本就安静）、seed1 轻刹车
   （去 m 响应 +40%）、到 **seed4 临界门**——native m 只压住 1.14Hz，一旦 reset/uniform/shuffle 就**整场点火 >120Hz**；
   注意 uniform（保住 mean(m) 只抹平空间）也点火 → 是 m 的**空间集中**在守门，不只是均值。

---

## 1. 测了什么（第一性原理）

- **底物**：`epilepsiae_1146`、narrow montage、`template_source`/`twoend_equal` 摆位，L=20mm、≈40k E/I LIF，
  源→汇长轴 E→E（`run_m4_phaseplane.build_substrate`）。
- **慢状态**：只开两个作用在 E 细胞上的慢变量——抑制效能 `z`（越低=去抑制越强，`D=1−z̄`）和适应量 `m`（每发一个
  spike +1、慢衰减，产生一股减法适应电流 `eta_m·m`）。工作点固定：`use_z=use_m=True, I_th_EI=95.1985,
  tau_z=5000ms, tau_adp=2000ms, A_target=0.001`，`eta_m = eta_m_from_frac(0.001, I_EE_scale=272.755, peak_m=36.6036)
  = 0.007451594…`（**调用现有 helper，不 hardcode**）。这个点三 seed 都被上游 gap-dynamics 判为
  `bounded_elevated_plateau`（0/3 runaway、0/3 recovery，`D_max≈0.056/0.060/0.054`）——代表"m 把网络卡在受控高位
  平台"，不是终止发作、不是恢复循环、不是临床 seizure。
- **要回答的四个问题**（spec §IV）：随慢状态从 baseline 进入高位平台，(1) 固定踢的空间响应是变成沿轴招募还是被压制？
  (2) 是否出现可辨识的有限时间响应算子（gain/模式重组/轴向偏好/mode switching）？(3) m 主要是全局降易感、还是改
  空间图案、还是两者？(4) z-only 的 pre-onset 轴向易感性在 z+m 平台是保留/削弱/消失？

**命名纪律**：算子对象叫 **empirical finite-time SNN response operator** `K_T`，其 SVD 给 V1/U1/`σ̂1`（经验有限时间
奇异模，**不是**精确本征向量；`σ̂1` 不是 Jacobian 特征值）。禁止 exact eigenmode/eigenvalue/`Re(λ)` crossing/Hopf/
fold/Floquet/临界跨零。

---

## 2. 怎么测的（"完全随机应该长这样 vs 实测长这样"）

**慢状态注册（P0，先冻死规则再看空间结果）**：用同一 seed/底物/慢配置完整重放这条 20 秒平台轨迹（200k 步），只
用总体 `D/a/率+时间` 注册 5 个时刻（baseline=1000ms；approach_25/50/75=`D` 首次穿过 `D_base+f·(D_plateau−D_base)`
且在锁定短窗内取最低瞬时率的 resting 步；settled_plateau=尾窗 resting 中 `D` 最接近尾窗中位、且过 settled 门）。
**parity 闸**：把重放的 `D/a/率` 降采样到 5ms（4000 bin）和上游 NPZ 逐点比——同代码/底物/seed 应逐位相同，rel 超
2% 就停下报告 discrepancy。然后分段重放+resume 抓 5 个状态的**完整可恢复 checkpoint**（V/电流/环形缓冲/RNG/z/m），
存 sha 指纹。

**固定踢追踪（P1，主分析）**：5 个状态用**完全相同**的源核 Gaussian 正电流踢（RMS=0.01·I_EE_scale），在 fork 窗
内冻结 z/m（隔离快子系统），读 5/15/30/50ms 响应图、轴向 kymograph、corridor/matched-off-axis/distal 响应、整场响应
norm；只有响应过绝对地板才拟合 arrival-距离（正斜率+R²≥0.5+≥4 点，否则 fail-closed，绝不硬造行波）。

**低-k 严格算子审计（P2）**：复用直接空间模式那套修正审计——平衡低波数基（k_max=1→9 个对称 2-D 模式）、每格 RMS
对齐踢的量级、±ε 成对、1×/2× 幅度、16 个独立 continuation-noise future（+ε/−ε 共享同一 future=CRN）、两组独立 8-future
半样、饱和 fail-closed。**严格辨识门（四项同时 + 无饱和）**：full-N 差异≤0.15 且 half-A≤0.15 且 half-B≤0.15 且
cross-half 算子不稳定度≤0.15（`robust_identifiability_gate`）。不过门→该状态 V1/U1/σ̂1 留空标 unresolved；**绝不**用
full-16 均值绕过 split-half 门，**绝不**在时间曲线上插值缺失模式。

**跨态模式追踪（P3）**：只在**相邻且都过严格门**的状态之间算 sign-invariant U1 overlap、leading-subspace principal
angle、|field|²-质心位移、轴向对齐变化、Δσ̂1；`σ̂1/σ̂2<1.05` 退化时追 leading subspace 不追单向量。

**m 最小机制对照（P4）**：只在 baseline/approach_75/settled_plateau，四个条件共享同一快态/z/踢/RNG future——native
z+m / m_reset（m→0）/ m_uniform（保 mean(m) 抹平图案）/ m_shuffle（保分布打乱位置），读固定踢的响应。解释边界：
m_reset 是短时 off-manifold 反事实（只说即时 m-current 贡献）；uniform vs native 分离"均值刹车 vs 空间图案"；shuffle
是空间图案敏感性对照；**不得把这些短 fork 写成长程自然轨迹**。

---

## 3. 结果（n=3）

### 3.1 parity（P0）— PASS（bit-identical, 3/3）
三 seed 重放 vs 上游 `traj_zA_q75_tz5000_A0.001_seed{1,3,4}.npz`：D/a/率三场 `max_abs=0.0`、`rel=0.0`（n=4000）——
逐位复现上游平台轨迹。15 状态全 resolved、单调、resting、`settled=True`。`D_base→D_plateau`：seed1 0.0088→0.0455、
seed3 0.0137→0.0573、seed4 0.0132→0.0459。

### 3.2 固定踢追踪（P1）— 被压住、从不轴向招募
同一记踢的整场响应 norm（Hz）沿慢状态：

| state | seed1 | seed3 | seed4 | 判读 |
|---|---|---|---|---|
| baseline | 1.95 | 0.523 | 0.662 | 源核局部响应（src_core 1.34/0.22/0.43） |
| approach_25/50/75 | ~0.08/0.08/0.10 | 0/0/0 | 0/0/0 | **被压到近零（三 seed 一致）** |
| settled_plateau | 0.22 | 0 | 1.72 | **平台绝对幅度分化**（见 P4） |

**remote_sink 恒=0、无 distal recruitment、无合格 arrival front（全 15 状态）**。即越进入 m-active 高位平台，同一刺激
被越压越弱、始终窝在源头，**从不沿轴向汇招募**。回答 spec §IV.1=被压制（非轴向）；§IV.4=z-only pre-onset 轴向易感性
在 z+m 平台**不复现为轴向招募**。

### 3.3 低-k 严格算子审计（P2）— bounded negative（0/15）
15 个 (seed,state) **全部不过严格门**：linearity discrepancy 0.27–0.55、cross-half split-half instability 0.41–0.70，
全远超 0.15 门；`sat=0`（无 fork 饱和）。响应最强的 baseline 也不过门（disc 0.35/0.40/0.38）。即在这个 fast-subsystem
尺度、这个探针强度下，**没有稳健可辨识的线性空间响应算子**——与 z-only 前身稀疏结论同向，是 spec 明确允许的
bounded-negative 完成。

### 3.4 跨态模式追踪（P3）— 无
0 个可辨识状态 → 12 对相邻状态全 `both_identifiable=False` → **无 mode trajectory**。V1/U1/σ̂1/轴向/overlap 全 undefined，
图上诚实留空（不插值、不用 full-16 均值绕过 split-half 门）。

### 3.5 m 最小机制对照（P4）— 刹车，力度按离点火边缘跨 seed 分化
settled_plateau 固定踢响应 norm（Hz）；ignition = 该 m 扰动把场推过 runaway 率（peak>120Hz）：

| seed | native | m_reset | m_uniform | m_shuffle | 判读 |
|---|---|---|---|---|---|
| 1 | 0.22 | 0.31 | 0.24 | 0.30 | 轻刹车（去 m +40%），mean 占大头 |
| 3 | 0 | 0 | 0 | 0.10 | 平台本就安静，m 基本无关 |
| 4 | 1.72（peak 1.14Hz, 0/144 cell>10Hz） | **点火** | **点火** | **点火** | **临界门**：peak 122–128Hz、15–16/144 cell>10Hz |

baseline 处 m≈0，四条件基本相等（正确对照）；approach_75 四条件全≈0、m_reset 不恢复响应 → 该中段压制是 z-驱动、非 m。
**关键**：seed4 平台上 m_uniform（保住 mean(m)、只抹平空间图案）**也点火** → 守门的是 **m 的空间集中**（自适应堆在高发放
的源核神经元上），不只是均值。回答 spec §IV.3=**m 主要是全局刹车（gain 抑制）；其力度 + 空间集中度按 seed/离边缘程度
分化，在近临界平台（seed4）是防整场点火的关键门**。

### 3.6 cohort 判读边界（n=3）
- P1 承重口径：**固定踢随慢状态被压制、从不轴向招募（sink 恒 0、无 arrival front）**，方向三 seed 一致；平台绝对幅度
  分化（0/0.22/1.72）是"离点火边缘不同"的表现，不改"被压制/非轴向"的方向。
- P2/P3 承重口径：**0/15 稳健可辨识 → 无跨 seed 一致轴向算子、无 mode trajectory**（clean bounded negative）。
- P4 承重口径：**m=平台刹车，per-seed 力度分化（可忽略→轻→临界点火门）；空间集中比均值更关键（seed4 uniform 也点火）**。
  这是 per-seed 描述聚合，不是 per-seed 强 cohort 主张（§5/§VI.7）。

---

## 4. 允许 / 禁止的话（spec §6）

**允许**：同一 m-active 平台骨架在慢轨迹上呈现特定有限时间空间易感性；固定踢随慢状态变得/不稳定变得更轴向；m 主要
全局刹车/主要改空间/两者；z-only pre-onset 轴向易感性在 z+m 平台保留/削弱/消失；算子在各状态可辨识/不可辨识。

**禁止**：平台=临床 seizure；复现 interictal→ictal→recovery 循环；V1/U1=精确本征模；`σ̂1>1`=净放大；kymograph=行波；
Hopf/fold/Floquet/特征值跨零；无≥2 同状态稳健可辨识 seed 一致就下"跨 seed 一致轴向算子"；把短 m-fork 写成自然轨迹；
换 state/seed/ε/basis/T/eta_m 救结论。

---

## 5. 工程验收

- **TDD 24 tests 全绿**（`tests/test_topic4_mz_m_eigenmode_tracking.py`）：E1 eta_m 换算 / E2 状态注册不依赖扰动 / E3
  replay-parity / E4 checkpoint-resume parity（分段==连续，含指纹）/ E5 freeze 冻 z&m / E6-E8 m_reset/uniform/shuffle
  合同 / E9 CRN / E10 低-k 基对称正交 / E13 严格门四项 / E14 饱和 fail-closed / E15 zero-response arrival fail-closed /
  E16 模式符号不变 / E17 退化子空间追踪 / E18 resume 幂等 / E19 checkpoint 指纹 / E20 plotting fail-closed。
- parity bit-identical（§3.1）；basis 正交残差 2e-15；`sat=0`。
- **不改 engine / 不改 mz_slow_vars / 不改 direct-spatial 模块或其产物**；新模块 import-safe；固定踢/审计机器复用
  `run_topic4_mz_direct_spatial_modes`。checkpoint 去掉诊断 trace lists（不进指纹）以让每次 fork 的 deepcopy 变轻。

## 6. 产物索引

- 代码：`src/topic4_mz_m_eigenmode_tracking.py`、`scripts/run_topic4_mz_m_eigenmode_tracking.py`、
  `scripts/paper_figures/plot_figure5_mz_m_eigenmode_tracking.py`、`config/topic4_mz_m_eigenmode_tracking.yaml`、
  `tests/test_topic4_mz_m_eigenmode_tracking.py`、spec。
- 结果（gitignored 工作树，与 direct-spatial 惯例一致）：`results/topic4_sef_hfo/mz_m_eigenmode_tracking/`：
  `STATUS.md` / `state_registration.json`（含 parity + checkpoint sha）/ `fixed_kick_summary.json` /
  `operator_tracking_summary.json`（含 mode_tracking）/ `controls_summary.json` / `numerical_audit.json` /
  `checkpoint_manifest.json` / `provenance.json` / `per_seed/`。
- 图（paper-ready 候选）：`results/paper-ready-figure/fig5_mz_m_eigenmode_tracking_candidate/figures/`：
  `figure5_mz_eigenmode_A_fixed_kick_tracking.{png,pdf}`（P1 状态对齐固定踢）+
  `figure5_mz_eigenmode_B_mode_tracking.{png,pdf}`（P2 辨识 strip + P4 m 对照；0 可辨识时走 bounded-negative 紧凑版）+
  `README.md`。
