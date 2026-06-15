# Topic 5 A+B 线加固计划 — 把 network-axis 对齐做成论文主结果

- 日期：2026-06-15
- 状态：**计划（待用户 review 后执行）**
- 上游：A+B 线已执行（`docs/archive/topic5/axis_alignment_AB_result_2026-06-14.md`，定稿表
  `results/topic5_ictal_recruitment/axis_alignment/axis_alignment_FINAL.md`）
- 计划来源：用户 2026-06-14 brief（3.1–3.5 五项加固）+ 三个范围决策（见 §0）
- **审阅修订（2026-06-15，用户代码合同审阅 82/100）已并入**：P1-1 sign 选择规则 / P1-2 pre 控制窗踩
  guard / P1-3 split-half 分母与退化 / P1-4 HFA-joint PASS 定义 + 两个工程口（runner 参数化、aggregate
  带 effect-size）。逐条落点见各 §❗审阅修订 段。无 P0 阻断；不扩第六类分析。
- **P1-1 经诊断后修正（2026-06-15）**：原拟"主统计收紧到 abs-max"被只读诊断推翻（abs-max 在 36–52%
  发作发散、大差来自横向 nuisance → 高估对齐）。**用户拍板：主统计保持锁定 P-current（FINAL 不动），
  符号补充改用干净 1D 沿轴 sign(corr(rank,activation))。** 详见 §3.4。
- 关联代号：A 线 = mirror-invariant `|axis_alignment|`；4 null = channel / within_shaft /
  anchor_matched / joint；指标 = broadband[PRIMARY] / hfa[sensitivity] / ramp[sensitivity] /
  ei[B 线 exploratory]

---

## 0. 范围锁（用户已拍板，不留事后自由度）

| 决策 | 选定 | 含义 |
|---|---|---|
| 3.2 窗口扫描范围 | **主量聚焦** | 只对 broadband(主)+HFA(关键敏感) 跑 6 窗（0-5/5-10/0-10/0-20/近端[-10,0]/远端[-120,-90]负对照），B=1000；ramp/EI 维持 0–10s。一次 EDF 重抽取，其余窗切片得到。 |
| 3.5 病人级统计 | **补效应量+措辞** | 维持现有病人级重采样；补 patient-level effect size + bootstrap CI + 措辞锁。**不另建形式化 LME**（per-subject median 已折叠 seizures，混合模型无自然随机效应结构）。 |
| 交付与执行 | **写文档→批准→执行** | 本文档即待批计划；批准后按 §6 顺序执行。 |

**总纪律**：用户明令"只补 5 个，不要无限扩展"。本计划 5 个 workstream + 交付件，**不新增第六类分析**。
每个 load-bearing 结论配一个"坏数据应当失败"的回归对照（参照 memory
`feedback_acceptance_gate_encode_conclusion`：验收门必须编码结论本身）。

---

## 1. 一句话目标

把 A 线（间期传播轴 ↔ 发作早期激活共享一根粗网络骨架，符号自由共线、非逐点重放）从"已执行的探索
结论"加固成"可冻结的 topic5 论文主结果"，靠 5 项稳健性补丁封住可预见的审稿质疑。补完即冻结。

文献支撑（已核，写进 wording-lock）：Smith/Schevon/Rolston eLife 2022（人类 IED 可作 traveling
waves，与 ictal discharges 共享路径、方向关系不必同向）；Matarrese Brain 2023（interictal spike
propagation 揭示 effective connectivity、预测手术结局）；Bartolomei Brain 2008（EI = fast activity ×
相对 onset delay，HFA 对齐更强有生理依据）。三类都支持"间期传播结构有病理/临床意义"，都**不**要求
"发作早期逐触点重放"。

---

## 2. 现状核对（绑在代码上，决定每项是复用还是新建）

| 项 | 代码现状 | 这次要动的 | 量级 |
|---|---|---|---|
| 3.1 split-half | 轴自洽性（held-out ρ）已在几何侧由 `split_half_axis_validation` 验过（半数 ρ≈0.752）。**但"用 half 轴去对齐发作激活"是新的** | 见 §3.1 两条腿 | 中 |
| 3.2 window | T0 缓存**只存 0–10s**；**只 HFA 存了时间序列 `hfa_zt`，broadband/ramp 没存** → 换窗要重抽 EDF | 扩缓存（存 `bb_zt`+加长窗+≥120s pre 窗）→ 6 窗切片 → 重跑对齐 | **重（EDF 限速）** |
| 3.3 frequency | 数据全有（4 量×4 null 已跑） | 分层措辞锁 + 文献锚 + 一张 null-hierarchy 图 | 轻 |
| 3.4 sign | `corr_pair_mirror_invariant` 本就返回带符号 corr；runner 当前 `abs()` 丢掉了 | 保留符号 → 病人内符号稳定性 + 与 t_a/t_b 关系；**作补充不作判据** | 轻–中 |
| 3.5 patient-level | **已经是病人级**（per-subject median → binomial/Wilcoxon/LOSO/BH-FDR over 18 人） | 补 effect size + bootstrap CI + 措辞 | 轻 |

---

## 3. 五个 Workstream

### 3.1 Split-half template stability — 轴不是某批间期事件偶然搭出来的

**测什么。** A 线那根间期轴，会不会只是某一批间期 HFO 群体事件凑巧搭出来的？把事件分两半各搭一根轴，
看（i）两半轴自洽；（ii）两半各自跟发作激活的对齐都站得住、且病人内一致。

**怎么做（两条腿，复用点不同 —— §6.1 防"名字都叫 split-half 就混用"）。**

- **腿 (i) 轴自洽 / 反套套性** → **复用 `split_half_axis_validation`**（已存在）。它的 null 问的是
  "半A轴能否预测半B放电顺序"，正好是"轴本身稳不稳"。在 **A 线那 18 个被试的 masked 模板**上重算
  held-out ρ + bootstrap CI（事件二分用 PR-2.5 约定：**split-half OR odd-even** 各跑一次，取 OR）。
  注意 phantom 纪律：事件 rank 必须走 masked（`mask_phantom_ranks`），不得用未 masked loader。
  **❗审阅修订（P1-3）：`split_half_axis_validation` 现在只是随机 split（`rng.permutation`），没有
  确定性 first/second + odd/even 路径。需给它（或新 thin wrapper）加显式 `split={first_second,odd_even}`
  API，不能靠随机 split 充当 odd/even。**

- **腿 (ii) 对齐稳健（新的、load-bearing）** → **复用 `build_readout_record` 的事件子集路径**，
  **不要**用 `split_half_axis_validation`（它问的是 firing-order 预测，不是场-vs-发作对齐，null 对不上）。
  对每被试，把 `masked`/`lag_raw`/`bools` 按事件二分（first/second + odd/even），每半经
  `_half_along_axis`→`compute_axis_frame`→`signed_transverse_axis`→`build_readout_record` 重建一份
  **half-axis 触点平面记录**（与 `_t_a.json` 同构），再分别喂
  `scripts/run_topic5_axis_alignment.py` 跑 broadband + HFA 的 channel + joint null。
  - 实现注：`run_contact_plane_readout.py` 现在只产全集 `_t_a.json`。需给它（或新 thin runner）加一个
    `--event-split {first,second,odd,even}` 路径，输出 `*_t_a_half{1,2}.json` 到并行子目录
    `observation_readout/real_subjects_splithalf/`。

**❗审阅修订（P1-3）：分母锁死 + 退化规则（防"幸存者分析"）。**
- **分母锁死 = 原始 18 人**（不随 half 可生成性漂移）。
- 退化预定义：half 记录生成失败（半内事件太少 / 无法成帧）、`n_channels < 6`、或 joint-null
  `effective_shuffle_n < 4` → 该被试该层标 **non-evaluable**（不是 PASS、也不是 silent drop），**单独报
  attrition**（多少人因哪条退化）。non-evaluable 进分母但不计入 PASS。

**预注册数字门。**
- 轴自洽（腿 i）：held-out ρ 队列中位 **≥ 0.6**（保守，低于几何侧 0.752 的内部松弛带），且 ρ 的
  bootstrap CI 下界 > 0 的被试 **≥ 12/18**（split-half OR odd-even，分母固定 18）。
- 对齐保持（腿 ii）：
  - **broadband-channel**：half-1、half-2 各自 binomial p < 0.05（over evaluable subjects，分母 18 标 attrition）。
  - **❗审阅修订（P1-4）HFA-joint PASS 定义写硬**（不写含糊的"保持过"）= **Wilcoxon one-sided p < 0.05
    AND patient-level effect size 95% CI 下界 > 0**，over adequate（`effective_shuffle_n ≥ 4`）subjects；
    binomial 仅作伴随报告（FINAL 全数据下 HFA-joint binomial = 3/18 p=0.058 本就不显著，靠 Wilcoxon/FDR 过）。
  - **跨 18 被试配对**的 half-1 vs half-2 `|axis_alignment|` Spearman ρ **≥ 0.5**（半-1 对齐强的被试，
    半-2 也强 → 对齐强度由病人稳定结构驱动，非某半事件偶然）。
- **坏数据回归**：把事件**随机配对成两半但破坏 template 结构**（打乱事件 rank 后再二分）→ 轴自洽
  ρ 应塌向 0、对齐应不再稳过 null。这是该步的 falsifier，必须报。

---

### 3.4 Sign supplement — 方向是补充，不是成败判据

**测什么。** 主结论是符号自由 `|ρ|`。补充问：同一病人内，发作激活梯度落在间期轴的"早端更旺"还是
"晚端更旺"这个方向，稳不稳？跟正/反向模板（t_a 主 / t_b 次）有没有关系？

**❗审阅修订（P1-1）+ 诊断驱动的 pushback（2026-06-15，用户已拍板"保持锁定+1D符号"）。**
审阅 P1-1 提出现 `corr_pair_mirror_invariant()["corr"]` 取**signed corr 最大**（`max(valid, key=x[0])`，
`propagation_contact_plane_readout.py:285`），runner 再 `abs()`——直接读它的符号会偏向"更正"的 mirror
候选。原计划据此打算新增 abs-max helper 并把主统计收紧到 `max(|c_id|,|c_mir|)`。**但只读诊断（354 次真实
发作，`/tmp/absmax_diag.py`）推翻了这条收紧**：

- abs-max 与现锁定主统计 `abs(max_signed)` 在 **36–52% 发作上发散**（broadband 52.4% / hfa 36.5% /
  ramp 45.9% / ei 40.2%），最大 Δ 达 **0.84–0.95**（近乎从"完全不像"跳到"几乎一样"）。**不是小数值清理。**
- 几何上，大 Δ 只可能来自 c_id 与 c_mir **异号且量级悬殊**，即**横向(y)结构强**的发作。abs-max 会把这种
  横向 nuisance 当成"对齐"——恰恰是 mirror-invariant 本要排除的。所以 **abs-max 系统性高估对齐（永远 ≥
  P-current），高出部分主要来自横向**，不是更正确而是更脏。

**决定（用户 2026-06-15）：主统计保持锁定 P-current（`abs(max_signed)`），FINAL 不动。** A1 新增的
`corr_pair_mirror_invariant_signed`（abs-max helper）**降为诊断工具，不接主统计**。

**怎么做（符号补充 = 干净的 1D 沿轴量，与 y 镜像无关）。**
- sign supplement = `sign(corr(间期 typical_rank, 发作激活))` over matched channels（spearman 或 pearson，
  与主统计同一组 matched 通道）。rank 低=源/早端、高=汇/晚端：**符号<0=源端更旺（forward）；>0=汇端更旺
  （reverse）**。这直接答 forward/reverse，完全不碰横向手性，避开 P-current 选择被偏置成正的问题。
- 每被试多次发作 → 同号比例 / 符号熵 / signed median；比较 t_a 与 t_b（若存在）方向是否相反（呼应几何侧
  median cos=−0.977 同轴反向）。
- **主统计 `|alignment|` 计算完全不变**——只在 runner 旁挂这个 1D 沿轴 sign 字段。

**门。** sign 本身**无 pass/fail 门**——明确写"sign 稳定性不进 primary、不作成败判据"（理由：primary
用 `|·|` 方向已合并；eLife 2022 + 上一轮 echo 都提示轴本来双向）。只做 construct sanity：若方向纯随机，
同号比例应 ≈ 50%；报这个分布，不下结论。

---

### 3.5 Patient-level effect size + framing — 封住"354=独立 N"质疑

**测什么 / 现状。** 现有口径**已经是病人级**：per-subject median over seizures → binomial / Wilcoxon /
LOSO / BH-FDR over 18 人。354 次发作**已折叠成病人内精度，根本没当 354 个独立样本**。

**怎么做。** 在 `_cohort_stats` 补：
- patient-level effect size = `median_over_subjects(real_median − null_median)`，配 **bootstrap-over-
  subjects（重采样 18 病人）95% CI**；
- Wilcoxon 的伴随效应量（rank-biserial 或 Hodges–Lehmann）。
- **❗审阅修订（工程口）**：不能只改 `_cohort_stats`——`scripts/aggregate_topic5_axis_alignment.py`
  现在只汇总 p/q/n_pass，要让它把 effect size + bootstrap CI + rank-biserial 写进 **FINAL json 与
  FINAL.md 表**（新增列），否则 effect-size 算了但没落到定稿件。
- 文档措辞锁："**主统计单位 = 18 病人；354 次发作 = 病人内精度，不作独立 N。**"

**门。** broadband-channel 与 HFA-joint 的 patient-level effect size **95% CI 下界 > 0**（与现有
Wilcoxon 显著一致即通过；不一致则是 bug 信号，回查）。

---

### 3.2 Window sensitivity — 0–10s 是不是最优窗 + 负对照

**测什么。** 0–10s 是不是最优对齐窗？5–10s 会不会更强（若更强，反而支持"招募骨架"而非瞬时 replay，
不怕）？0–5s 不一定强（early first-onset 本就不稳，已知）。**guard 外的远端 pre 窗应当对不上 = 负对照**。

**❗审阅修订（P1-2）：负对照窗不能踩 peri-onset guard。** T0 合同里 `[-60s,0]` 是 peri-onset guard，
baseline 是 `[-90,-60]`（`run_topic5_t0_eligibility.py:15,59`：GUARD_SEC=60）。原计划的 `[-10,0]s` **落在
guard 内**——若它也强对齐，不能解释成"纯解剖锚"，可能是**真 pre-ictal recruitment**。所以：
- **load-bearing 负对照 = guard 外远端 pre 窗 `[-120,-90]s`**（z 化用同一 `[-90,-60]` baseline；远到尚未
  ictal recruitment、又非 baseline 归一窗本身以免退化≈0）→ 期望"对不上"。
- `[-10,0]s` 保留，但**只作 proximal pre-ictal sensitivity 描述，不作 falsifier**。

**怎么做（工程关键：不要跑 5 遍 EDF）。**
- 扩 `scripts/build_topic5_t0_feature_cache.py`（参数化 `--out-dir --post-sec 20 --store-bb-zt
  --pre-feature-window`）：加存 `bb_zt`（broadband z 时间序列，与 `hfa_zt` 同构），后窗拉到 **≥20s**，
  pre 抽取拉到 **≥120s** 以覆盖 `[-120,-90]` 远端窗 + `[-10,0]` 近端窗。一次 EDF 过完，**resumable
  per subject**，写到**并行新缓存目录** `t0_feature_cache_v2_windows/`（AGENTS.md `_masked` 并行目录
  精神 —— **不动**支撑 FINAL 表的既有 354-seizure 缓存）。
- 派生窗：0–5 / 5–10 / 0–10 / 0–20 由 post 序列切片；远端 `[-120,-90]`（负对照）+ 近端 `[-10,0]`
  （sensitivity）由 pre 序列切片。每窗算 bb_auc + hfa_auc。
- 重跑对齐：broadband + HFA × (channel + joint null) × 6 窗 × **B=1000**。ramp/EI 不进本扫描。

**预注册数字门（避免事后挑窗）。**
- **正对照复现**：0–10s 窗仍稳过 channel（复现 FINAL 的 broadband-channel + HFA-joint）。
- **全窗并报，不挑最大窗当结论**：报各窗的 n_pass + effect size 全表；"5–10s ≥ 0–10s ≥ 0–5s"是
  **预期不是门**，真实落点照实报。
- **负对照（坏数据回归，load-bearing）= 远端 `[-120,-90]` 窗**：broadband/HFA 对齐应**显著弱于** ictal
  窗（远端 channel-null n_pass 不显著 / effect size CI 含 0）。若远端也强对齐 → 对齐不是发作特异、只是
  纯解剖锚，这是必须报的 falsifier。近端 `[-10,0]` 只描述，不作判据。

---

### 3.3 Frequency specificity — 把 HFA 这条讲清楚（措辞 + 一图，无新计算）

**测什么 / 现状。** 数据全有。问题是把三档说清楚，别让 HFA 抢主结论、也别埋没它过最严 joint 的意义。

**怎么做。** 锁分层：
- **broadband** = robust activation burden（**主指标**，过 coarse channel 层）；
- **HFA 60–100 Hz** = mechanistic sensitivity（**唯一过最严 joint 层**，接 Bartolomei 2008 EI 的
  fast-activity 生理依据）；
- **ramp / EI-like** = secondary，**不抢主结论**。
一张 null-hierarchy 图（§4）同时承担本项；无新分析。

**门。** 无新计算；wording 锁 + 文献锚。

---

## 4. 交付件（图 + 文档）

- **Patient-level 图**：18 病人各自 real `|axis_alignment|` vs null 分布（点 + null 带），按是否过 joint
  着色；broadband + HFA 两版。
- **Null-hierarchy 图**：x = 4 层 null（粗→严 channel→within_shaft→anchor→joint），y = n_pass（或
  effect size），每条线一个指标 —— 一眼看出"HFA 一路稳到 joint，broadband 在 joint 掉"。**这张图同时
  承担 3.3 frequency specificity**。
- **Window-sensitivity 图**：x = 6 窗（0-5/5-10/0-10/0-20/近端[-10,0]/远端[-120,-90]负对照），y =
  effect size，broadband + HFA 两条；**远端 [-120,-90] 负对照必须肉眼可见地掉下去**。
- 三张图都进 `results/topic5_ictal_recruitment/axis_alignment/figures/`，**配 `figures/README.md` 中文
  逐图说明**（AGENTS.md 强制）。图须 paper-grade 自包含（memory
  `feedback_figure_self_contained_paper_grade` + `feedback_figure_style_guide_per_topic`：无 §X /
  cluster_id / 括号轴标；遵 T 系列 colormap 锁；render→肉眼→改→再 render 后才提交）。
- **Final wording lock**：更新 archive `axis_alignment_AB_result_2026-06-14.md`（追加 5 项加固结果 +
  允许/禁止措辞）+ 主文档 `docs/topic5_seizure_subtyping.md` §3.0 摘要（只摘要+链接，FINAL 数值表只在
  results/，遵 AGENTS.md"主文档不复制 archive 全量数值表"）。
- memory 更新 `project_topic5_network_axis_pivot.md`：追加"A+B 五项加固完成 + 冻结"。

---

## 5. TDD / §6 不变量（每个非平凡 clause 一个会因违反而失败的测试）

新增/改动集中在 `src/topic5_axis_alignment.py` + `tests/test_topic5_axis_alignment.py`（必要时
`build_topic5_t0_feature_cache.py` 的窗口切片纯函数）：
1. **事件二分** —— first/second + odd/even **确定性**划分（不依赖 rng）、覆盖全集且不重叠；空半/退化半
   返回退化标记不静默出数（接 §3.1 non-evaluable）。
2. **half-axis 记录与全集同构** —— half record 的 channel schema、x_norm/y_norm 尺度、support 门与
   `_t_a.json` 一致（否则两半与发作激活不在同一平面，对齐不公平）。
3. **sign supplement（1D 沿轴，主统计不变）** —— `sign(corr(rank, activation))`：源端更旺样例 → 负号，
   汇端更旺样例 → 正号；y 镜像翻转不改变该符号（mirror-irrelevant）。
   （A1 诊断 helper `corr_pair_mirror_invariant_signed` 的 abs-max 测试已单独绿，仅作诊断保留，不接主统计。）
4. **effect size + bootstrap CI** —— 退化样例（全相等 diff）下 CI 不崩；CI 下界单调性 sanity；落进 FINAL json/md。
5. **窗口切片** —— 0–5/5–10/0–10/0–20 切片索引边界正确（hop_sec 对齐）；远端 `[-120,-90]` 与近端
   `[-10,0]` 切片各取 onset 前正确区间（远端在 guard 外）。
6. **坏数据回归（3.1 + 3.2）** —— 打乱 template 的伪半轴自洽 ρ→0；远端 `[-120,-90]` 负对照对齐
   effect size CI 含 0。

phantom 纪律全程：任何消费 lagPat 模板的新路径走 masked（`mask_phantom_ranks` / `--masked-features`）。

---

## 6. 执行顺序（critical-path 感知）

**Phase 1（并行，文件不相交，3 个 agent 同跑，TDD 先行）：**
- **A1 sign helper**：`src/propagation_contact_plane_readout.py` 加 `corr_pair_mirror_invariant_signed`
  （abs-max 选择 + `mirror_disagree`）+ `tests/test_propagation_contact_plane_readout.py` 测试
  （`abs(signed)==abs_corr` 等）。**只加函数，不动 runner。**
- **A2 effect-size/CI**：`scripts/aggregate_topic5_axis_alignment.py` 加 patient-level median(real−null)
  + bootstrap CI + rank-biserial，写进 FINAL json/md。
- **A3 window cache**：`scripts/build_topic5_t0_feature_cache.py` + `src/topic5_t0_features.py` 加
  `--out-dir --post-sec 20 --store-bb-zt --pre-feature-window`（≥120s pre）+ 窗口切片纯函数 + 测试。

**Phase 2（串行，整合点，依赖 Phase 1；我主导 + 必要时顺序 agent）：**
1. **runner 参数化 + 1D sign 接线**：`run_topic5_axis_alignment.py` 加 `--cache-dir --axis-dir`；
   旁挂 `sign(corr(rank, activation))` 的 1D 沿轴 sign 字段（同号比例 / signed median / 符号熵）。
   **主统计 `|alignment|` 计算完全不变**（abs-max 诊断已做、已定保持 P-current，FINAL 不动）。
2. **先后台起 window 重抽取**（最慢，EDF 限速）：用 A3 的扩展跑 v2 缓存（`bb_zt`+20s+≥120s pre），
   resumable，挂后台。
3. **split-half 产出器**：`run_contact_plane_readout.py` 加 `--event-split` 事件子集路径 → 产 half-axis
   记录 → 用参数化 runner 重跑对齐（3.1）。
4. v2 缓存好后：跑 **3.2** 6-窗对齐（broadband + HFA，含远端负对照 + 近端 sensitivity）。
5. 用 A2 的 aggregate 出 FINAL（带 effect-size/CI）。
6. 出三张图 + `figures/README.md`（render→肉眼检查）。
7. **3.3** 措辞锁 → final wording lock（archive + 主文档 §3.0）+ memory 更新。

每步前 re-read 本文档对应段（§5 step-boundary re-read）；每写完一段 status/recap 做 §8 外部读者复述
自检（测了什么/怎么测/揭示了什么，代号只作括号补注）。

---

## 7. 完成判据（这阶段结束 = topic5 主结论可冻结）

- 3.1 两条腿门全过（或如实记录哪条腿弱、弱≠失败但要写明）；坏数据回归如预期失败。
- 3.2 0–10s 复现 + 远端 `[-120,-90]` 负对照如预期掉下去；6 窗全表并报。
- 3.4 sign 作为补充报告，措辞明确不作判据。
- 3.5 patient-level effect size + CI 写入，"354=病人内精度"措辞落地。
- 3.3 三档分层 + 文献锚措辞锁定；三张图 paper-grade + README 中文。
- archive + 主文档 §3.0 + memory 同步；FINAL 数值表只在 results/。

**反预期不改门**：若 5–10s 比 0–10s 更强、或某层 null 比预期弱，照实报，不回头改门（memory
`feedback_acceptance_gate_encode_conclusion`）。弱 ≠ 失败：A 线门本就允许"只到粗解剖轴"这个落点
（plan `network_axis_pivot_plan_2026-06-13.md`）。

---

## 8. 执行记录（2026-06-15）

**Phase 1（并行 agent，全绿）**：A1 `corr_pair_mirror_invariant_signed`（abs-max 诊断 helper，33 测试）；
A2 `_effect_stats`（effect-size + bootstrap CI + rank-biserial 写进 FINAL，5 测试）；A3 v2 窗口缓存构建器
（`bb_zt`+全程轨迹+relt+切片纯函数，13 测试 + §5 跨消费者修复 ei_like onset）。D runner `--cache-dir/
--axis-dir` + 1D 沿轴 sign（16 测试）；E `deterministic_event_split` + `--event-split` half-axis 产出器
（38 测试，schema 与全集一致）。

**abs-max 诊断 → 主统计保持 P-current（§3.4）**：354 发作上 mirror_disagree 36–52%，大差来自横向 nuisance
→ 不切 abs-max，FINAL 不动，符号改 1D 沿轴。

**第二轮审阅修订（用户 82/100，2026-06-15）**：
- **P1-1（已修+揭示）**：`aggregate_topic5_axis_alignment.py::_subject_diffs` 加 `adequate_only`
  （`effective_shuffle_n>=MIN_EFF=4`）；FINAL 现输出 `effect_n_all/effect_n_adequate` + adequate CI +
  `wilcoxon_p_adequate`。**HFA-joint 揭示**：剔 5 个退化-null 被试后 eff 0.022→0.050，但 **adequate CI
  下界仍≈−0.003（恰触 0）**；broadband-channel eff 0.087 CI[0.006,0.129] 下界>0 干净。→ **HFA-joint 靠
  adequate Wilcoxon(0.013)+FDR(0.029)+rb(+0.59) 显著、效应≈0.05，但效应量 CI 边界**；不得写"HFA 干净碾过
  最严 null"。`aggregate_topic5_windows.py` 同步加 adequate 过滤。
- **P1-2（已建脚本）**：`audit_topic5_v2_cache_attrition.py` —— v2 加长 pre 致部分发作越 block 边界被丢
  （已观察 epilepsiae_1077 sz5），窗口扫描前出 `v2_cache_attrition.csv`，seizure-精度 N 按实际 cached/window-
  finite 报，不再说"354 不变"；病人单位仍 18。
- **P1-3（已过门）**：split-half 产出器完成 = 144 条记录，**t_a 半轴 4 组合 ×18 = 72，0 退化、0 个
  n_channels<6**。门过 → 启动 split-half alignment。

**流水线状态（nohup）**：split-half alignment 跑批中（8 runs×B=1000）；v2 缓存构建收尾（15→19）；组合
waiter 等两者完成 → 跑 P1-2 审计 → 我审 → 启动 window sweep。剩：split-half 腿(i) 轴自洽 held-out ρ（几何
侧曾 median 0.752，需在 18 人上复算）、两套聚合、三张图、措辞锁。
