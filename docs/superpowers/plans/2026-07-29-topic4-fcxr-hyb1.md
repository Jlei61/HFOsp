# FCXR-HYB1 实施计划 —— `Z – activity-excess K – X`（– `M`）生命周期

日期：2026-07-29 · 分支 `codex/topic4-fcxr-hyb1` · 基点 `efce58a6`
Spec：`docs/superpowers/specs/2026-07-29-topic4-fcxr-hyb1-design.md`（本计划把它补到可执行）

**本文件在任何 HYB1 结果出现之前锁死。之后只允许追加"执行结果"，不允许改判据、阈值或档位。**

---

## 0. 一句话

不再修完整的构成性离子基线；改为让**每个慢变量只负责一件事**，看它们合起来能不能在固定的
E1146 40k 底座上跑出 `稀疏不规则间期 → 无 kick 自发进入有界高态 → X 终止 → 回到原统计邻域`。
**恢复对象是"能继续按原规律产生稀疏不规则尖波的状态空间邻域"，不是周期轨道。**

---

## 1. 复用清单（不重造）

| 已有 | 位置 | HYB1 怎么用 |
|---|---|---|
| RC1 底座（additive FF + recurrent conductance + `g_sat=21.6`） | `run_topic4_mz_fcxr._fc_cfg` | 原样，不动 |
| `Z`：`z_inf = H(I_th_EI − I_I)`、Euler `tau_z` | `src/snn_engine/mz_slow_vars.py` | 原样 + **新增非对称恢复** |
| `X`：持续度门控 relay，**已有非对称 `tau_x_down/up`** | 同上（LC1 已落地、契约全绿） | 原样 |
| `M`：`tau_adp_E` / `eta_m_E` 力配比适应 | 同上 | 仅 H4，固定一档 |
| `D_Z` / `D_X` = `Σp_i(1−v_i)/Σp_i` | `src.topic4_mz_fcxr_lifecycle.depletion_coordinate` | 原样 |
| 生命周期分类器（含 `PERMANENT_SILENCE` / `RAPID_RELAPSE` 反作弊） | `src.topic4_mz_fcxr_lifecycle` | 作为 Gate 3/6 的底层 |
| baseline 统计合同 | `.../lifecycle_closure/baseline_contract_seed{1,3}.json` | Gate 2/6 的比较对象 |
| flock / sentinel / resource_log / worker plan | `run_topic4_mz_fcxr` + LC1 runner | 原样 |
| 钾膜电流适配器（`g·(E_K(K) − E_K(K₀))`） | `src/snn_engine/ion_homeostasis.py` | 只借这一个函数形式 |

**新写**：`src/snn_engine/activity_excess_k.py`、`src/topic4_fcxr_hyb1.py`、
`scripts/run_topic4_fcxr_hyb1.py`、`scripts/plot_topic4_fcxr_hyb1.py`、`tests/test_topic4_fcxr_hyb1.py`。
**只改一个非 blessed 引擎文件**：`mz_slow_vars.py`（加非对称 Z）。六个 blessed 文件每阶段核 SHA。

---

## 2. activity-excess K recruitment layer（A 节，锁死）

**命名纪律**：这是 **activity-excess K recruitment layer**（活动超额钾招募层）。
**不是**完整患者离子浓度模型。`eta_pump = 0`，钠不进入 HYB1。

### 2.1 背景负荷与上包络（`b_v`）

**不允许**用 `δK = K_o − time_mean(K_o)` 然后声称"按构造不变"。改为独立的 reduced 场：

1. 跑一条 **sensor-only** 间期轨迹（RC1 已验收工作点，慢变量全 off，`g_deltaK = 0`，无 kick，T = 8 s）。
   记录每个体素 `v` 的 spike 负荷密度时间序列 `s_v(t)`（每个离子步的体素内 E+I spike 计数 / 体素细胞数 / dt）。
2. **`b_v := quantile(s_v(·), Q_BG)`，`Q_BG = 0.99`**（逐体素，registered 上包络）。
3. `b_v` 与其 sha256 写入 manifest，**此后冻结**；`Q_BG` 不得因结果调整。

### 2.2 平滑 deadband 正部

$$R_\varepsilon(u) = \begin{cases} 0 & u \le 0 \\ \dfrac{u^2}{u+\varepsilon} & u > 0\end{cases}$$

`R_ε(0)=0`、`R_ε'(0)=0`（C¹）、**背景以下严格为 0**（不是 softplus 的"接近 0"）、
大 `u` 时 → `u − ε`。**`ε = 0.1 · median_v(b_v)`**，在测量 `b_v` 后立即写死。

### 2.3 场方程

$$\tau_K \frac{\partial\,\delta K}{\partial t} = q_K\,R_\varepsilon\!\big(s_v(t) - b_v\big)\;-\;\delta K\;+\;\tau_K D_K \nabla^2 \delta K,
\qquad \delta K(\cdot,0)=0$$

- 32×32 体素、`dx = 0.625 mm`、零通量边界（复用 B2.1 的 Laplacian）；
- **`τ_K = 0.6546 s`**（B2.1 在工作点实测的 `tau_Ko_at_workpoint_s`）；
- **`D_K = 2.5e-4`**（B2.1 常数）；
- **`q_K`** 由 T7.1 的 `f' = 1.0` 锚给出，沿用 B2.1 的 `q_ion = 0.013615797289152352`；
- `δK ≡ 0` 是**结构性不动点**：源项在背景以下恒为 0，清除项与扩散项在 `δK=0` 时为 0。

### 2.4 到膜

$$I_{\delta K,i} = g_{\delta K}\,\big[E_K(K_{o,0}+\delta K_{v(i)}) - E_K(K_{o,0})\big]$$

- 复用现有钾反转电位适配器；**E 与 I 都收到**这个附加电流；
- **`g_deltaK = 1.0`**（B2.1 锚），**本 sprint 不扫增益**；
- virtual-SEEG 读出**仍是 E-only**（引擎记录器限制，与既往一致）。

### 2.5 baseline-preservation gate（**必须实测，不能只靠"按构造"**）

无 kick、`δK` 关 / 开各一次、两个已接受 baseline seed（1 和 3）：

| 子句 | 门 |
|---|---|
| δK 活跃占空比 | `mean_t,v 1[s_v > b_v] ≤ 3 × (1 − Q_BG) = 0.03`（注册背景尾部的 3 倍） |
| δK 幅度 | `q99_t,v(δK) ≤ 0.05 mM`（间期不得出现可观测钾抬升） |
| returning event rate | 落在该 seed 的 `baseline_contract` 带内 |
| IEI CV | 落在带内且 `≥ 0.5` |
| event duration / participation | 落在带内 |
| 数值安全 | `clip_frac_max = 0` 且 `tau_eff_min ≥ 2·dt` |

**任一不过 → 写 `STOP_BASELINE_DISTURBED.json` 并停止 HYB1。**
**不许**调 drive、不许改连接、不许调 `g_deltaK`、不许改 `Q_BG` 抢救。

---

## 3. Z 单轴（B 节，锁死）

### 3.1 恒等式（不是近似）

`z_i(0) = 1`，`dz_i/dt = (z_inf,i − z_i)/τ_z`，`z_inf,i = 1[I_i < I_th]`。因此在 `t=0`：

$$\left.\frac{dD_Z}{dt}\right|_{0} \;=\; \frac{\sum_i p_i\,\mathbf 1[I_i \ge I_{th}]}{\tau_z \sum_i p_i} \;=\; \frac{a_p(I_{th})}{\tau_z} \;=:\; h_Z$$

**`h_Z` = 前 onset 的 p 加权抑制失效 hazard，单位 s⁻¹。** 这就是那条一维轴。

### 3.2 观测锚（现有 artifact，无需新跑）

来自 `.../lifecycle_closure/z_only_summary_seed{1,3}_q{75,50}.json`
（`D_Z_end / T`，24 s 全程平均斜率）：

| 档 | `I_th_EI` | `τ_z` (ms) | `D_Z_end` | 观测 `h_Z` (s⁻¹) | onset latency |
|---|---:|---:|---:|---:|---:|
| q75 seed1 | 95.1985 | 5000 | 0.16544 | **6.89e-3** | 10.0 s |
| q75 seed3 | 95.1985 | 5000 | 0.18579 | **7.74e-3** | 9.0 s |
| q50 seed1 | 1.6653 | 10000 | 0.80529 | **3.36e-2** | 3.0 s |

### 3.3 单参数实现

**固定 `τ_Z_down = 5000 ms`（q75 值）、`τ_Z_up = 20000 ms`（固定，不扫），只改 `I_th_EI`。**

三档 hazard 取 q75 上界与 q50 之间的**几何等分**（`ratio = 3.36e-2 / 7.74e-3 = 4.342`，
四等分因子 `4.342^{1/4} = 1.4433`）：

| 档 | 目标 `h_Z` (s⁻¹) | 目标 `a_p = h_Z · 5 s` |
|---|---:|---:|
| **H_LO** | **1.117e-2** | 5.585e-2 |
| **H_MID** | **1.612e-2** | 8.061e-2 |
| **H_HI**  | **2.327e-2** | 1.164e-1 |

### 3.4 一次校准 probe 求 `I_th_EI`（≤6 probe 的上限用 1 个）

跑 **1 条 3 s 慢变量全 off 的 probe**，记录 E 细胞的 GABA 传感电流 `I_i` 与 `p_i` 权重，
得到 p 加权生存函数 `a_p(θ) = Σp_i 1[I_i ≥ θ] / Σp_i`（**一条曲线给出整条轴**），
再对三个目标 `a_p` 反解 `θ = I_th_EI`。

### 3.5 可辨识性检验（**先做，不过就停**）

用同一条生存曲线**预测**两个锚的 hazard，与 §3.2 的观测值比：

- `a_p(95.1985)/5 s` 应落在 `[6.89e-3, 7.74e-3]` 的 **±50%** 内；
- `a_p(1.6653)/10 s` 应落在 `3.36e-2` 的 **±50%** 内；
- 且 `a_p` 在 `[1.6653, 95.1985]` 上**严格单调**。

**任一不满足 → 写 `DESIGN_BLOCKED_Z_AXIS.json` 并停止；不得改回二维网格。**

三档 `I_th_EI` 数值在 H2 之前写回 spec + manifest，**之后不得移动**。

---

## 4. Z 非对称恢复（C 节）

在**非 blessed** 的 `mz_slow_vars.py` 增加 off-by-default 的 `tau_z_down` / `tau_z_up`，
**逐条复刻已验收的 `tau_x_down/up` 实现模式**：

```
tau_sel = where(z_inf < z, tau_z_down, tau_z_up)     # 耗竭 / 恢复
z += (dt / tau_sel) * (z_inf - z)
```

- `z_inf < z` ⟺ 传感电流仍在阈上 ⟺ **高负荷** → `tau_z_down`；
- `z_inf ≥ z` ⟺ 负荷已回落（活动停了、X 已把它掐断） → `tau_z_up`；
- **禁止瞬时 reset**：恢复仍是一阶弛豫，无跳变；
- 两者都为 `None` 时**逐比特**还原原来的单 `tau_z`；
- 只给一个必须同时给两个，否则 `ValueError`（与 X 同规则）。

> **一处明写的设计取舍**：spec 原文说"X 已终止且 persistence sensor 回落后"才切到 `tau_Z_up`。
> 本实现用 `z_inf` 与 `z` 的比较来选择——`z_inf = H(I_th − I_I)` **本身就是负荷指示**，活动被 X
> 掐断后 `I_I` 落回阈下、`z_inf` 翻到 1，自动进入恢复支。这样实现的**行为满足要求**（高负荷耗竭 /
> 负荷回落后恢复 / 无瞬时 reset），同时**不让 Z 显式依赖 X 的状态**——显式耦合会破坏本 sprint 的
> 前提（职责分离），并让"是 Z 恢复了还是 X 放手了"无法归因。这是有意偏离，记录在此。

**测试**：状态切换、上下界 `z∈[0,1]`、确定性、snapshot/restart、off-parity 逐比特。

---

## 5. 七门（D 节，锁死；**先过坏数据回归才有资格判新候选**）

| # | 门 | 数值判据 | 必须**挂**在 |
|---|---|---|---|
| 1 | 无 kick 自发进入 | `KICK_BOOST=0` 且 `t_kick=1e9`（构造可验证，写入 provenance）；且检出 onset | — |
| 2 | 间期 ≥ 8 s | `pre_interictal_ms ≥ 8000` | **q50**（3.0 s） |
| 3 | 有界持续高态 1–5 s | `1000 ≤ bout_ms ≤ 5000` 且 `bounded=True` 且 `clip_frac_max=0` | **q75**（无 ictal bout，只有 dense train） |
| 4 | 空间结构 | `recruit ≥ 12/15` **且** 源空间 onset 梯度可分（见 §5.1） | — |
| 5 | `X` 在 onset 之后 | `t(D_X 首次 ≥ 0.02) − t_onset ≥ 100 ms` | — |
| 6 | 统计恢复 | post 连续窗 `≥ 8000 ms`；post 的 event rate / IEI CV / duration / participation / silent fraction 回到 pre 与 baseline 合同带内；`IEI CV ≥ 0.5`；标签 ∈ `RECOVERED_INTERICTAL` | **q50**（`PERMANENT_SILENCE` / 恢复窗 0–1 s） |
| 7 | 零 clip / 无 runaway | `clip_frac_max = 0`、`finite`、`tau_eff_min ≥ 2dt`、末段率 < runaway 界 | **q50 无 X**（452.8 Hz） |

**已删除**：跨 seed 发作间隔 CV（单次生命周期估不出来）。

### 5.1 Gate 4 的阈值预注册 —— 以及一条**已经可以预判的**风险

`recruit ≥ 12/15` 取自 HEO3 既有判据。但 **HEO2.1 已经量过：48/48 个工作点 recruit 全部 ≥13/15、
38/48 恰为 15/15，其中包括纯同步的 tonic 16 Hz 态**。所以这条腿**很可能分不开** structured event
与 synchronous negative control。

因此按 spec 的条件规则**先做分离检验**（compute-free，用现有 HEO2/HEO3 产物）：

- 正例 = q75 structured event 窗；负例 = synchronous negative control（tonic 16 Hz / `uniform_static`）；
- 若 `recruit` **与** `onset_gradient_r2` **都**不能把两者分开 →
  **Gate 4 记 `UNRESOLVED`，不发明任何方向结论**，且生命周期候选降级为
  "lifecycle candidate with UNRESOLVED spatial leg"。
- **E1146 未确立双向传播** → 全程**禁止** source/sink 成功措辞。

---

## 6. 阶段与停机

| Stage | 内容 | 计算量 | 停机 |
|---|---|---|---|
| **H0** | 组件符号 / 时序审计（现有 artifact + 小型确定性合成流），Z 非对称只查相序 | ~0 | 符号不成立 → `TOPOLOGY_INPUT_UNRESOLVED`（**不是**机制 NO-GO） |
| **H0b** | Z 单轴校准 probe（3 s）+ §3.5 可辨识性检验 | 1 × 3 s | 不过 → `DESIGN_BLOCKED_Z_AXIS` |
| **H1** | baseline-preservation，δK off/on × seed{1,3}，8 s，无 kick | 4 × 8 s | 不过 → `STOP_BASELINE_DISTURBED` |
| **H2** | 12 格短屏：3 × `h_Z` × δK{off,on} × X{off,on}，**T = 14 s**，conn seed1 + dev noise，M off | 12 × 14 s | 全败 → 按失败模式归档 STOP，**不扩网格** |
| **H3** | ≤2 个 survivor 跑 **T = 24 s**，全七门 | ≤2 × 24 s | 无 survivor → 不跑 seed3、不跑 unseen noise、不加 M |
| **H4** | `M` 波形：candidate × {M off, M on}，`tau_adp=250 ms` + force-matched 10%，**不扫 M** | 2 × 24 s | 破坏七门 → 报 waveform trade-off，不称成功 |
| **H5** | conn seed3 + unseen noise 逐个 | 视情况 | confirmatory 失败**不回调参数** |

**H2 优先级（不加权、不 cherry-pick）**：① pre ≥ 8 s ② 有界持续高态 ③ X 在 onset 后
④ 空间招募 ⑤ 数值安全。

**运行时预算（实测锚，LC1）**：24 s 有界 run ≈ 23–36 min；**24 s 失控 run = 4.3 h**。
→ 每个 cell 加 **wall-time kill guard = 3600 s**，外加科学 early-stop（连续 2 s 平均率 > 300 Hz
判 `RUNAWAY` 并终止该 cell，记录而非重跑）。

---

## 7. 必存读出（H2 起）

流式统计 + landmark snapshot，**禁止**存 `N_cell × T` 稠密态。

1. **virtual-SEEG**：pre / early / ictal / post 四段的 raw trace、rate、TFR、1–80 Hz 分带功率；
2. **慢态**：`D_Z`、`δK` mean/max/spatial extent、`D_X`、persistence sensor `y`；
   onset / X activation / termination / recovery 四个时间标记；`Z–δK–X` 相轨迹；
3. **空间**：core_A / core_B / axis / off-axis 分区率；源空间 onset time；`onset_gradient_r2`；
   active radius；occupied voxels；15 触点 recruitment；pre/early/ictal/post 能量场快照；
4. **未来 eigenmode 用**：四段状态快照 + 固定电极 / 网格 ordering。
   **本 sprint 不自主声称 eigenmode transition。**

---

## 8. 资源（sprint baseline 于 2026-07-31 00:07 CST 记录）

`nproc=80`、`load≈60`、`MemAvailable=215 GiB`、**`swap_used baseline = 843.2 MiB`**、
sibling 40k = 2（各 5.6 GB）+ 10 个 ~1.3 GB CUDA 任务。

- `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS = 1`；
- `T < 20 s`：默认 1 worker；仅当 sibling 40k ≤ 2 **且** MemAvailable ≥ 96 GiB **且**
  ≥ 2 × 实测单 run peak RSS **且** swap 不涨时才允许 2；
- `T ≥ 20 s`：**严格 1 worker**；
- swap 相对 843.2 MiB：**> +256 MiB 停止提交新任务**；**> +512 MiB 且继续增长**→
  只终止**自己最新的**任务并写 `RESOURCE_PAUSED.json`；
- **不杀任何 sibling / user 进程**；
- 每个 run 写 `resource_log.jsonl`；flock 单实例锁 + `launcher_<stage>.pid` +
  `RUNNING_/DONE_/FAILED_<stage>.json`；断线后**不得重复提交已完成 cell**（DONE sentinel 判定）。

---

## 9. 产物

```
results/topic4_sef_hfo/mz_full_conductance_spatial_relay/hyb1_lifecycle/
  STATUS.md  run_manifest.json  component_topology.json  z_axis_calibration.json
  baseline_preservation.json  screen_map.json  candidate_verdict.json
  resource_log.jsonl  runs/  figures/{README.md, hyb1_screen_map.png,
                                      hyb1_lifecycle_diagnostic.png[, hyb1_candidate_state_readout.png]}
```

`hyb1_candidate_state_readout.png` **只有出现 candidate 才画**。每张实际生成的图必须目视检查并在
`figures/README.md` 用中文写"展示什么、关注什么"。**不画无数据占位图。**

---

## 10. 允许 / 禁止（贯穿全程）

- **禁止**：`limit cycle`、`bistability`、"真实患者离子机制"、"完整离子模型"、
  "促进传播"（未测 onset 梯度 / 延迟 / 覆盖时）、source/sink 成功措辞；
- **禁止**：为救结果改任何已锁阈值、档位、`Q_BG`、`g_deltaK`；
- **禁止**：把 engineering green 写成科学成功；
- **允许**：只有 development seed 过 → `development lifecycle candidate`；
  预注册 confirmatory 也过 → `reproducible model seizure-like lifecycle candidate`。
