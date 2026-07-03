# Topic 5 V3a — 发作"轴向→非轴向"模态转移（data-side，seizure-paired，n_perm=1000）

date 2026-07-04 · 状态：**EXPLORATORY，脆弱阳性（fragile positive）** · 分支 `topic5-v3a-mode-transition`（隔离 worktree，off `topic5-v2-phase1`@959d156）· 前身 = V2a（`docs/archive/topic5/v2_phase2_criticality_state_layer_2026-07-01.md`）· 姊妹 = V3b（M3B 模型–数据一致性 H3d，未实现）· 设计合同 = `docs/superpowers/{specs,plans}/2026-07-02-topic5-v3a-mode-transition*.md`（rev2）

> **一句话定位**：在严格的**同一批发作内配对**的 P3→I1、narrow 主队列、被试为单位、率保/密度归一/标签/相位/块/onset-jitter 控制后——**扣率零假设后的"非轴向净流增量"达到了队列级显著（判读机械上到 tier 4 / supported=True），但这是一个很脆的阳性**：未校正的原始流大多在**下降**、以**同时共激活**为主而非定向传导、个体稳健性几乎全塌、更具体的**模态方向腿(H3c)全阴**。按预注册的 honest-coupling，这是**数据侧候选信号**，**不是**确立的"轴→非轴模态转移"，机制升级要 V3b。

---

## 0. 摘要（朴素话：测了什么 / 怎么测 / 揭示了什么）

**测了什么。** 病人两次发作之间，短暂高频异常放电总沿一条固定先后顺序在电极间传开——像一条走熟的小路（间期 HFO 传播轴 `G_HFO`）。V3a 问的**不是**"发作时这条小路更强对齐"，而是：**发作一启动，系统会不会把"沿小路的有序传播"松掉，转而在小路之外的电极上、沿小路之外的方向越来越容易被点亮/放大**。承重锚定在**发作前 30~10 秒（P3）→ 早发作 10~30 秒（I1）**，每次发作按它**自己的脑电起始**配对。

**怎么测的（三条腿 + 关键纪律）。** 非轴向触点用**纯间期 HFO 参与度**定义（从不参与间期高频事件的触点 = 天然"小路之外"，对发作全盲，防循环）。① **H3b 连锁流（承重）**：能量拔高当"点亮"，建 avalanche 转移矩阵、**排自跳**、算"轴向漏到非轴向的净流"、**按源触点均值归一** + **率保零假设校正后的净增量(surplus)**；② **H3c 模态（承重）**：全清洁触点低秩 VAR → `A^{k*=3}` **主右奇异向量**（非正规系统最大有限时放大方向）映回触点 → **密度归一**的非轴 vs 轴子空间投影差；③ **H3a 轴向减弱（仅辅助）**：线长率对间期顺序的有符号对齐强度 P3→I1 是否下降。全部跟"打乱后的随机版本"比（率保/杆内空间/标签/相位/块，各 1000 次自建），被试为单位，narrow 主 broad 复制**永不合并**，Holm 双承重端点，λ 只报 surplus。

**揭示了什么。** 见 §4。核心：**粗的"流/招募"这一侧有一个队列级显著但很脆的信号；细的"方向转移"这一侧是阴的。**

（内部代号：`G_HFO`=间期 typical_rank 顺序场；`net_offaxis_flux_surplus`=率保零假设校正的 A→N 净流；`mode_shift_density`=低秩 VAR 主奇异向量密度归一投影差；`common_drive_sensitive`=lag1≈lag0；tier 0–4；state_v3_supported=tier≥3。）

---

## 1. 与 V2a 的关系 + honest-coupling

- **V2a**（前身，`v2_phase2_criticality_state_layer_2026-07-01.md`）是"发作前变脆状态是否**沿** HFO 轴增强"的受限探索，偏阴性 + 受限实现 → 降级、reframe 成 V3a（把问题从"是否沿轴"改成"是否**从轴向搬到非轴向**"）。V2a 唯一耐用产物 = 方法学定律（λ 只报 `λ_surplus` 不报 raw），V3a 全程沿用。
- **honest-coupling（承重纪律）**：V3a 阳性 = "数据里存在 axial→non-axial 重组的候选信号"。**升级成机制主张需 V3b（模型–数据一致性）+ 表达层。无 forecasting。** V3a 单独不能下机制结论。

---

## 2. 主假设 + 承重端点

| 腿 | 朴素话 | primary 指标（P3→I1 Δ，配对） | 方向 | 角色 |
|---|---|---|---|---|
| H3a 轴向减弱 | 沿小路有序变弱 | `Δβ_axis_strength`（线长率基） | <0 | **辅助**，`module_support_flag` 恒 False |
| **H3b 非轴向流放大** | 连锁漏向非轴增多 | `Δnet_offaxis_flux_surplus`（率保校正） | >0 | **承重** |
| **H3c 模态转移** | 最易放大方向转非轴 | `Δmode_shift_density`（密度归一奇异向量） | >0 | **承重** |

support = H3b 或 H3c 队列级过 Holm；H3a 显著只加强、不能单独定支持。

---

## 3. 方法要点（详见代码 + spec）

- **触点池**：`all_clean` = 完整清洁蒙太奇（每被试 30–124 触点）；非轴向 = 间期 HFO 参与度 0 的触点（全盲）；三分类 轴/非轴严格/暧昧（`scripts/_topic5_v3_io.py::classify_subject_contacts` 单一真相源）。
- **时间窗**：按每发作 `eeg_onset_rel` 锚定（不是 cache relt=0）；短发作 I1 offset-based fallback；jitter ±10s。
- **Δ-零假设纪律**：p 算在 **Δ(I1−P3) 的置换分布**上（每置换同时扰动 P3+I1 再相减，非"各相位单独 null 再相减"）；率保为 H3b 命门，密度归一 + 标签零假设为 H3c 维度偏差控制；NaN-safe（有限抽样过滤）。
- **【关键】同一批发作内配对（P1-1 修复，2026-07-03）**：obs 与 null 都**只用 P3∩I1 的共同发作**、**逐发作先算 I1−P3、再被试中位**（不是"I1 组中位 − P3 组中位"）。这一步把结果从 tier2 翻成 tier4（§4.3）。
- **判读**：tier 只在 summary 出（`run_topic5_v3_summary.py`）；队列 Wilcoxon 单侧 + Holm(2)；narrow 主 broad 复制永不合并；`geometry_insufficient` / `compute_failed` 标记不计负。

---

## 4. 结果（paired final，n_perm=1000）

### 4.1 队列级（承重）

| 队列 | n | H3b Holm-p（中位Δ，surplus） | H3c Holm-p（中位Δ） | 过？ |
|---|---|---|---|---|
| **narrow（主）** | 7 | **0.031（+0.035）** | 0.891（−0.001） | **H3b 过**（LOO 稳，见下） |
| broad（复制） | 9 | **0.008（+0.083）** | 0.633（+0.001） | H3b 过，同向 |

→ narrow H3b 过 → tier 3；broad 同端点同向复制 → **tier 4，state_v3_supported=True**（机械判读）。H3c 两队列全阴。H3a 所有被试置换检验不显著（`h3a_strengthens` 全 False，诚实）。

**留一被试稳健性（narrow H3b Wilcoxon）**：去掉任一被试后 p ∈ [0.016, 0.031]，始终 <0.05 → **统计显著对单被试稳健**。

### 4.2 主队列逐被试（paired）

| 被试 | H3b surplus (p_rate) | H3b raw | common_drive | jitter | H3c ΔMS (Holm 内) | subject_support |
|---|---|---|---|---|---|---|
| 1096 | +0.072 (.001) T | **+0.104** | 敏感 | 否 | −0.006 | 否（jitter/common-drive 否决） |
| **1125** | +0.065 (.009) F | **−0.272** | 不敏感 | 是 | +0.007 T | **是**（经 H3c 腿） |
| 1146 | +0.108 (.001) F | +0.026 | 敏感 | 否 | +0.001 | 否 |
| 253 | +0.035 (.254) F | **−0.106** | 敏感 | 否 | −0.018 | 否 |
| 384 | +0.015 (.296) F | −0.066 | 敏感 | 否 | −0.001 | 否 |
| 442 | −0.002 (.529) F | −0.085 | 敏感 | 是 | +0.003 | 否 |
| 958 | +0.011 (.388) F | −0.125 | 敏感 | 否 | −0.005 | 否 |

**读法**：`subject_support` 主队列=1（1125，且是靠**模态腿**而非流腿）；broad `subject_support`=0/9。流腿(H3b)队列级过、但**没有任一被试单独通过流腿自己的完整稳健性门**（1096 流腿最强却被 common-drive+jitter 打掉）。

### 4.3 关键：配对修复把 tier2 → tier4

未配对（旧，"I1 组中位 − P3 组中位"）：narrow H3b Holm-p=**0.219**（不显著）→ tier 2。配对（新，"逐发作 I1−P3 再中位"）：narrow H3b Holm-p=**0.031**（显著）→ tier 4。**为什么**：未配对把 P3 与 I1 拿**不同的发作子集**做中位相减，引入跨发作错配噪声（典型 253：未配对 −0.191 → 配对 +0.035），把一个其实**符号一致**的效应打散成不显著。配对（同一批发作内前后差）是**科学上正确的 within-seizure 度量**，去噪后效应的**方向一致性**（6/7 主队列为正）浮现出来。中位 surplus 反而更小（+0.058→+0.035）但更显著——Wilcoxon 看的是**符号一致性**不是幅度。

### 4.4 为什么说这是"脆弱阳性"（四个来源）

1. **null-relative，非绝对**：主队列 **5/7 被试原始流 Δ 为负**（发作时绝对净流在降）；surplus>0 只是"降得比率零假设少"。"放大"是相对基线，不是绝对上升。
2. **同时共激活为主，非定向传导**：主队列 **6/7 `common_drive_sensitive`**（lag1≈lag0）→ 这份"流"大多是同时点亮，不是轴→非轴的方向性传导。
3. **个体稳健性几乎全塌**：onset-jitter 主队列仅 2/7 稳；**流腿 0/7 被试**通过它自己完整稳健性门；`subject_support` 1/7（还是模态腿）、broad 0/9。
4. **更具体的模态腿(H3c)全阴**：Holm 0.891/0.633 → "最易放大方向转向非轴"这个更强说法没成立。粗流有信号、细方向无。

---

## 5. 判读纪律（能说 / 不能说）

**能说**：配对、同一批发作内、率保零假设校正后，**扣基线的非轴向净流增量在主+复制队列都达到队列级显著且对留一稳健**——存在一个**队列级 off-axis 招募候选信号**；但它 null-relative、以同时共激活为主、缺稳健个体支撑、且模态方向腿全阴。**个别被试 1125** 三条腿方向最一致（模态腿个体过 + 轴向减弱大 + 流为正）。

**不能说**：不能说"V3a 确立/发作发生轴→非轴模态转移"（模态腿阴 + 脆弱）；不能说"off-axis flux 增加"（原始流在降）；不能把它当定向传导（多为共激活）；不能说"发作没临界性/没非轴向状态"；不能把 broad 单独当主结论；不能用发作结果定义非轴向；不能上机制主张（要 V3b）；`geometry_insufficient`≠阴性；`tier 4/supported=True` 是机械判读，**主文档写法必须带全部 §4.4 保留项**。

---

## 6. 工程（TDD + 审阅 + 发现并修的 bug）

- 约 3,750 行新代码（`src/topic5_v3_mode_transition.py` 纯函数 + `scripts/_topic5_v3_io.py` + 5 个 run + summary + plot）；**29 纯函数测试 + 11 集成测试**；subagent-driven-development（每任务 fresh 实现 + 独立 review，承重腿 opus 深审）；整分支 opus review 干净。
- **审阅/复查逮到并修的真 bug**（非一次到位）：① `sliding_windows` 尾部残窗 → 全窗；② H3b 方向判据 raw→surplus 并端到端传给所有闸门（CLAUDE.md §5）；③ H3a 标签零假设 NaN 稀释（p 被压小，1125 0.07→0.24）+ 端到端传到 H3b/H3c；④ `h3a_strengthens` 漏显著性门；⑤ 916 短发作 jitter 崩溃（退化窗守卫）；⑥ **触点池 recipe 我自己写错**（把每发作 drop 当通道 drop）→ 改真通道 QC + 抽共享 helper；⑦ **P1-1 同一批发作配对**——把结果从 tier2 翻成 tier4（§4.3），是最重要的一次修复。

---

## 7. 复现 + 产物 + 下一步

```bash
# worktree: .worktrees/topic5-v3a-mode-transition (branch topic5-v3a-mode-transition, off 959d156)
bash .superpowers/sdd/run_final_nperm1000.sh              # paired n_perm=1000, both cohorts -> summary -> figure
pytest tests/test_topic5_v3_mode_transition.py -q          # 29 passed
pytest -m integration tests/test_topic5_v3_integration.py  # 11 integration
```
产物：`results/topic5_ictal_recruitment/v3_mode_transition/{narrow,broad}/v3_{avalanche,dynamics,susceptibility,summary}_subject.csv + v3_cohort_tier.json`；图 `figures/v3_mode_transition_summary.png` + `figures/README.md`。

**下一步（不救 1125、按预注册）**：
- **V3p**（preictal-only 非轴向轨迹，spec/plan 已写 `topic5-v3p-preictal-trajectory`）：只看发作前 2 分钟非轴向流/可放大方向是否**逐渐爬升**、是否专门集中在非轴向触点——避开 I1 信噪比塌 + onset 对比脏。
- **敏感性**（把本 tier-4 从"脆弱"往"稳健或证伪"推）：队列 claim 对 common-drive/jitter 稳健性的敏感度；lag1-specific 的队列级检验；O 窗对照；raw-vs-surplus 的正式分解。
- **V3b**（H3d 模型–数据一致性）才是机制升级。
