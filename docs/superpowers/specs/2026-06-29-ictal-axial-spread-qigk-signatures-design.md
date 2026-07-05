# 发作内"轴向→轴外扩散"与 q_I/g_K 慢变量信号的 per-seizure 探索 — Design

- 日期: 2026-06-29
- 分支: `codex/topic4-m3a-v2-2`（承接，可另开）
- 关联: Topic 5 发作内 field 动力学 pilot（`docs/archive/topic5/ictal_field_dynamics_pilot_2026-06-28.md`）、Topic 4 M3A-v2 空间慢变量场收口（`docs/topic4_m3_stage.md` §6）
- 档位: **per-seizure 描述 + 被试内**，探索性，非队列级机制主张（详见 §7）

---

## 1. 一句话目标

逐次发作检验:发作早期活动是**"先在轴上、再依次漏到轴外"的空间扩散**（用户设想、也是 M3A-v2 慢变量机制 q_I/g_K 想要的图景），还是**"整片几乎同时点亮"**（M3A-v2 当前 SNN 衬底的实际行为）。把这个判读做成可观测、可证伪、带坏数据回归的 per-seizure metric。

## 2. 科学框架（朴素话）

**测什么**:每次发作里，每个电极的活动强度（功率相对它自己平时基线高出几个稳健标准差）随时间的轨迹。从这些轨迹读出"每个电极何时点亮、何时达峰、达峰后是否衰退"，再问这些事件在空间上的先后。

**为什么是这个问题**:
- 慢变量机制的理想链条是——轴向先放电 → 轴向疲劳累积（g_K↑）+ 轴向抑制资源耗尽（q_I↓）→ 活动脱离轴向漏到轴外 → 全局去抑制 → runaway。核心是一个**从轴向核心向外的招募波**。
- M3A-v2 模型那条线反复发现:当前神经元网络衬底点火是**"全或无、整片一起亮"**，造不出"先沿轴、再慢慢漏出去"的局部行波。所以**模型自己产生不了这个理想图景**。
- 之前 pilot 用"活动质量落在轴上还是轴外的占比"做趋势，结论跨被试不稳健；而且**那个占比量根本区分不开"真扩散"和"整片同时点亮"**——后者也会让"轴外占比升"。
- **因此真数据到底是哪一种，是一个能把两者掰开、且直接对模型有话说的关键问题。** 真数据若是依次扩散 → 是模型缺、却该有的信号；真数据若也是整片同时 → 模型行为其实忠实，"扩散"图景本身存疑。

**揭示什么的形态**:每次发作给一个判读（§5.5 的五类之一），报这些判读在被试内、全队列的分布。措辞是"看起来像扩散 / 像同时点火 / 没看清"，**不是**"证明了慢变量机制"。我们**测不到 q_I/g_K 本身**，只测机制预测的可观测信号。

（内部归档代号:Topic 4 M3A-v2 q_I/g_K 空间慢变量场、Topic 5 ictal field dynamics pilot、`positive_mass_share` 占比量。）

## 3. 数据底座（复用，不新建缓存）

复用现有长窗缓存 `results/topic5_ictal_recruitment/ictal_field_long_cache/<ds_sid>.{npz,json}`:
- `bb_zt__<idx>`:broadband（1–45 Hz）功率的稳健 z 轨迹，逐电极 × 时间，hop 0.1s，覆盖 onset 前 ~130s 到 offset+90s。**主**。
- `hfa_zt__<idx>`:HFA（60–100 Hz）同上。**次**。
- `bb_relt__<idx>` / `hfa_relt__<idx>`:每个时间 bin 相对临床 onset 的秒。
- meta `seizure[idx]`:`eeg_offset_rel`、`eeg_duration_sec`、`eeg_onset_rel` 等。

轴/分区复用现有 `scripts/run_topic5_ictal_field_dynamics.py::load_context`:
- 轴 = 两个间期模板"最早 compact core"质心连线段;四分区 MECE = `source_core` / `axial_mid` / `axis_end_noncore` / `non_axial`（`src/topic5_ictal_field_dynamics.py::axis_partition`）。
- **本设计中 "轴向" = `source_core ∪ axial_mid ∪ axis_end_noncore`（轴线走廊上的全部）;"非轴向" = `non_axial`（轴外）。**
- 每电极的"沿轴位置 t"、"离轴垂距 d"（`axis_partition` 已给）用于 §5.4 的招募-距离梯度。

队列:沿用 pilot 的 `--substrate broad`（9）+ `narrow`（7），per-substrate 输出 dir。**前置检查**（实现期）:确认 broad+narrow 并集的长缓存都存在;缺的 `[skip]` 并在 cohort summary 记。

**不新建缓存、不动现有缓存构建脚本。**

## 4. 架构与边界

三个单元，各自一个清楚职责，可独立测试:

1. **纯数学模块 `src/topic5_ictal_spread_dynamics.py`**（无 `scripts.*` 依赖，与现有 `topic5_ictal_field_dynamics.py` 同风格）:逐电极时间原语 + 两级统计 + 置换零假设 + per-seizure 判读。输入是裸 numpy 轨迹/分组/坐标，输出是数值/标签。**全部 TDD（§8）。**
2. **驱动 `scripts/run_topic5_ictal_spread_dynamics.py`**:复用 `load_context`（轴/分区）+ 读长缓存，逐被试逐发作算 per-seizure 行 + 判读，写 CSV + `per_subject/*.json` + `cohort_summary.json`。复用现有 `_slice` / `_zmean_by_name` / `_ztraces_by_name` / parity gate 思路。
3. **作图 `scripts/plot_topic5_ictal_spread_dynamics.py`**:per-subject + per-seizure 诊断图 + cohort 判读分布图;`figures/README.md`（中文，AGENTS.md 规范）。

输出根:`results/topic5_ictal_recruitment/spread_dynamics/`（broad）、`spread_dynamics_narrow/`（narrow），与现有 `field_dynamics{,_narrow}` 同级。

复用 `load_context` 用 import（现有 run 脚本已是这种 import 链）;新模块保持纯数学。

## 5. Metric 设计

### 5.1 逐电极时间原语（从每条 z 轨迹算）

对电极 c，轨迹 z_c(t)、relt（rel onset）:
- **招募时刻 `t_recruit(c)`** = z_c 第一次**持续** ≥ `THETA_RECRUIT` 且持续 ≥ `SUSTAIN_SEC` 的起点时间。从没满足 → `NaN`（= 未招募，本身是信息）。只在 ictal 段内找（relt ∈ [0, offset]）。
- **达峰 `t_peak(c)` / `z_peak(c)`** = ictal 段内 z_c 最大值的时间与值。
- **疲劳指数 `fatigue(c)`** = 达峰后、仍在发作内的衰退:在窗 `[t_peak, min(t_peak + FATIGUE_WIN_SEC, offset − TERM_GUARD_SEC)]` 上 `(z_peak − z_tail) / z_peak`，`z_tail` = 窗末段均值。要求窗长 ≥ `FATIGUE_MIN_SEC` 且 `z_peak ≥ THETA_RECRUIT`，否则 `NaN`。**`TERM_GUARD_SEC` 排除发作终止前的整体熄灭**（那是"发作结束"非"疲劳"）。

### 5.2 第一级（不依赖轴，对模型）:依次招募 vs 几乎同时

- **招募离散度 `recruit_dispersion`** = 被招募电极 `t_recruit` 的 p10–p90 跨度（秒）。也报 IQR、被招募电极数、未招募电极数。
- **判读**:`recruit_dispersion < D_SIM` → `near_simultaneous`（≈ 模型行为）;`≥ D_SIM` → `sequential`（传播过程）。`D_SIM` 是锁定且可调的验收门（§6）。
- **不用轴**，所以不怕轴定错;本身就是"真数据像不像模型"的判别器。**头条量。**
- 可选 sensitivity（文档化、非首跑）:onset-jitter 零假设（所有电极共享一个 onset + 各自 rise 锐度造的抖动）下离散度的期望，作为 `D_SIM` 的旁证。首跑用绝对秒阈 + 永远上报原始 `recruit_dispersion` 让阈可审计。

### 5.3 第二级（用轴，回答原问题）:依次招募的方向

仅当第一级 = `sequential` 且通过 §5.6 准入时计算:
- **招募先后 `dt_recruit`** = median `t_recruit`(非轴向) − median `t_recruit`(轴向)。正 = 轴先;≈0 = 同时;负 = 轴外先。
- **交接时滞 `handoff_lag`** = 轴向群体均值轨迹与轴外群体均值轨迹在 ictal 段、滞后搜索范围 ±`HANDOFF_MAX_LAG_SEC` 内的互相关峰位移（轴外滞后轴向为正）。这是"轴向在时间上领先轴外"的连续量，也承载**疲劳-交接**:轴向峰后衰退之际轴外升起。
- **招募-距离梯度 `recruit_grad`** = `t_recruit` 对（沿轴位置、离轴垂距）的最小二乘斜率。向外越远招募越晚（正离轴斜率）= 行波。
- **置换零假设**:轴/轴外标签置换 `N_PERM` 次重算 `dt_recruit`（和 `handoff_lag`），实测超出置换分布（单尾 p < `P_PERM`）才算真方向。防分区凑出假"轴先"。

### 5.4 支撑量（q_I/g_K 纹理）

- **疲劳对比** = median `fatigue`(轴向) − median `fatigue`(轴外);轴向是否更早达峰（median `t_peak` 轴向 < 轴外）。
- **晚期 runaway**:晚段（= ictal 末三分之一，排除终止 `TERM_GUARD_SEC`）参与度（活动电极占比，复用 `participation` 思路）是否逼近 1;场空间结构是否塌成均匀（梯度幅值 `grad_mag` 晚段 vs 早段下降，复用 `field_gradient`）。

### 5.5 per-seizure 判读分类（两层报告，避免轴定错污染第一级）

**两层独立报告**——第一级不用轴，即便轴坏也仍有效，不能被轴问题一起扔掉:

**`stage1_class`（不依赖轴，总是出）** ∈:
- `near_simultaneous` — 第一级离散度 `recruit_dispersion < D_SIM`。
- `sequential` — `≥ D_SIM`。
- `undetermined` — 被招募电极太少 / 发作太短（§5.6），第一级也算不了。

**`verdict`（方向层，用轴）** 仅当 `stage1_class=sequential` 且 §5.6 轴门通过时有方向含义:
- `axial_to_offaxis` — `dt_recruit > 0` 且置换 p < `P_PERM`（轴向→轴外依次扩散）。
- `offaxis_leads` — `dt_recruit < 0` 且置换显著（轴外先，反向）。
- `sequential_no_direction` — 方向不显著。
- `ill_posed` — 轴退化（单源）/ 轴冷（§5.6 门未过）/ 每组招募不足 → 方向问题没良好定义。
- 当 `stage1_class=near_simultaneous` → `verdict=near_simultaneous`（没有方向可问）;`undetermined` → `verdict=ill_posed`。

**用户原问题分两问回答**:
1. "是不是每次都是空间扩散（而非整片同时点火）" = `stage1_class` 在被试内、全队列的分布（**这一问不怕轴定错**）。
2. "依次扩散的发作里，是不是从轴向扩散" = `verdict` 在 `sequential` 子集里的分布。

### 5.6 坏数据 / 准入门（每个承重主张配回归，feedback_acceptance_gate_encode_conclusion）

- **轴退化**:`axis_degenerate=True`（单源无轴）→ `verdict=ill_posed`，第二级量全 `NaN`。负控:现有 1084 单源。
- **发作太短**:ictal 段 < `MIN_ICTAL_SEC` 或时间 bin 太少 → `ill_posed`。pilot 里 E916 中位 8s 无趋势是动机。
- **每组招募数**:轴向、轴外各 < `MIN_RECRUIT_PER_GROUP` 个被招募电极 → 第二级 `NaN`（但第一级仍可出，第一级只需全体）。
- **轴是否真在 onset 区**:`axis_onset_gate` 通过需**同时**满足两条具体判据——(a) median `t_recruit`(轴向) ≤ median `t_recruit`(全体)（轴向招募不晚于全阵列中位）;(b) 轴向在 `[0, ONSET_PROBE_SEC]` 段均 z ≥ 轴外同段均 z（轴向 onset 不比轴外冷）。任一不满足 → `verdict=ill_posed`，并单列 `axis_onset_gate=False` 标志（第一级 `stage1_class` 仍照常出）。pilot 警告:384 间期源在发作时冷、热区在另一根杆。**这既是诚实门，也是有意思的发现（间期传播轴 ≠ 发作起始区）。**

## 6. 锁定参数（spec 固定，可调;实现期写进模块常量 + 测试钉住）

| 参数 | 默认 | 含义 |
|---|---|---|
| `THETA_RECRUIT` | 2.0 | 招募阈（稳健 z）|
| `SUSTAIN_SEC` | 1.0 | 招募需持续秒数 |
| `D_SIM` | 2.0 s | 第一级"几乎同时"上限（招募 p10–p90 跨度）|
| `HANDOFF_MAX_LAG_SEC` | 30.0 | 交接互相关滞后搜索上界 |
| `FATIGUE_WIN_SEC` | 10.0 | 疲劳测量窗长上限 |
| `FATIGUE_MIN_SEC` | 3.0 | 疲劳窗最短 |
| `TERM_GUARD_SEC` | 5.0 | 排除发作终止前的整体熄灭 |
| `MIN_ICTAL_SEC` | 20.0 | 发作准入最短 ictal |
| `MIN_RECRUIT_PER_GROUP` | 3 | 第二级每组最少被招募电极 |
| `ONSET_PROBE_SEC` | 5.0 | 轴-onset 门探测窗 |
| `N_PERM` | 2000 | 标签置换次数 |
| `P_PERM` | 0.05 | 置换单尾显著阈 |
| 频段 | `bb` 主 + `hfa` 次 | 两者都算，每窗×band 一行 |

置换用固定 RNG seed（可复现）。所有阈值在 CSV/JSON 里随结果回写（可审计）。

## 7. 报告档位与诚实约束

- **per-seizure 描述 + 被试内**:发作为单位，被试内可做计数/Wilcoxon（如疲劳对比）。**非队列级机制主张。**
- **不可测 vs 可测**:q_I/g_K 是模型内部变量，数据测不到。只测机制**预测的可观测信号**（招募先后、疲劳形状、交接滞后、饱和）。
- **措辞**:"看起来像轴向扩散 / 像整片同时点火 / 没看清"。禁"证明慢变量机制 / 证明 q_I/g_K"。
- **与模型对话的边界**:可说"真数据在这些发作里看起来像/不像模型当前的整片同时点火"。不可把数据的依次扩散当成"模型机制成立"——那是两套独立证据。
- **先验**:pilot 已发现简单方向趋势不稳健、模型说衬底是整片同时点火 → 本探索**阴性偏向**。报负结果同样有价值（= 真数据也整片同时，模型行为忠实）。

## 8. 测试策略（TDD，纯数学模块每个原语一条）

合成轨迹钉死每条:
- `t_recruit`:已知越阈点;从不越阈→NaN;短暂 blip 不持续→不算招募;只在 ictal 段内找。
- `t_peak`/`fatigue`:先升后降→正疲劳;单调升→≈0 疲劳;终止熄灭被 `TERM_GUARD_SEC` 排除;窗太短→NaN。
- `recruit_dispersion`:全同时→小;摊开→大;`D_SIM` 两侧分类正确。
- `dt_recruit`:轴先→正;同时→≈0;轴外先→负。
- `handoff_lag`:构造已知滞后→恢复该滞后符号与量级。
- `recruit_grad`:沿轴线性招募→恢复斜率符号。
- 置换零:固定 seed 下 p 值确定;无真方向时 p 不显著。
- 判读分类:`stage1_class` 三类 + `verdict` 五类各可达（构造对应输入）;轴坏时 `stage1_class` 仍出。
- **坏数据回归**:轴退化→`ill_posed`+NaN;短发作→`ill_posed`;某组招募不足→第二级 NaN 但第一级出;轴冷→`axis_onset_gate` 挡。
- 全仓 `pytest` 无回归（v1 sibling `topic5_ictal_field_dynamics` 等不受影响）。

## 9. 输出

- `spread_dynamics{,_narrow}/per_seizure_metrics.csv`:每（发作×band）一行 + 逐窗时间原语聚合 + 第一/二级量 + verdict + 所有阈值回写。
- `per_subject/<ds_sid>.json`:被试内判读计数、轴元信息、坏数据标志。
- `cohort_summary.json`:五类判读全队列分布（broad / narrow 分开）= **回答用户原问题的头条表**。
- 图（`figures/` + 中文 `README.md`）:
  - per-seizure 诊断:招募时刻 vs 离轴距离散点 + 轴/轴外群体活动曲线（带交接时滞标注）。
  - per-subject:判读计数条形 + 疲劳对比。
  - cohort:五类判读堆叠分布（broad/narrow）。
  - 图自包含、paper-grade（feedback_figure_self_contained_paper_grade）。

## 10. 明确不做（YAGNI / 范围）

- 不做全时空行波速度拟合（路线 C，留作 A 出干净信号后的后续）。
- 不新建缓存、不改缓存构建脚本。
- 不做队列级机制主张、不升级假设档位。
- 不碰 M3A-v2 模型代码（这是数据侧独立探索）。
- 第一级的 onset-jitter 零假设只文档化为 sensitivity，首跑用绝对秒阈。

## 11. 与现有 pilot 的关系

不替换 pilot 的空间占比分析（保留作对照）。本设计是**补上 pilot 缺的时间维**:pilot 看"活动落在哪"，本设计看"活动先后在哪"——后者才能区分扩散与同时点火，才对得上 q_I/g_K 机制。两者输出并存于 `results/topic5_ictal_recruitment/` 下不同子目录。
