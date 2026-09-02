# Group-Event State v0.2 — H2b（Agent B）阶段性验收 / 历史版本封存

日期：2026-09-02　分支：`codex/topic5-group-event-state-v02-b`（base `f0c9e075`）
机器状态：**`STAGE_CLOSED_ENGINEERING_ONLY__H2B_NOT_ESTABLISHED`**
外部审阅：`/data/hfosp_group_event_state_v0_2/shared/waiter/reviews/p0_b_cohort_report_consumes_invalid_and_unverifiable_state_registry_2026-09-02.md`

---

## 0. 一句话

H2b 这一轮**只验收工程层与分母层**。所有涉及冻结状态能否预测发作的科学陈述
**均已撤回**，原因是上游 producer 不可验收、读取对象不合合同、关键对照未执行——
三条各自独立成立。本文件封存该阶段，供后续复用与追溯。

---

## 1. 验收通过（可复用，不依赖任何 producer）

| 层 | 内容 | 关键数字 |
|---|---|---|
| 发作对齐 | 按 **recording code** crosswalk，逐 onset 核对 | 274 匹配；**零** onset 落在自己录音外、零歧义、零重复 |
| 独立性 | 成簇归并（后一次落在前一次不应期内即同一 episode） | 274 次发作 → **209 个独立事件** → 留出 **99** |
| 覆盖 | 各提前量可用锚点 | 5min 98 / 30min 89 / 2h 79 / 6h 75 |
| 边界 | 状态覆盖 vs 监测覆盖分离；断监测 → right-censored | 27,982 风险行；不应期 30/60/120 min → 留出 106/99/80 |
| 目标 | early ictal field，按**脑电起点**锚定 | 264 成功 / 10 丢弃（全为窗口越出块边界） |
| 目标 QA | 对拍已验收 `ictal_field_long_cache`（168 次 / 11 人） | 同锚点中位 **ρ=+0.9977**；通道顺序处处一致 |
| 方法学副产 | 脑电起点 vs 临床起点 | **145/168** 更早，中位 5.0 s，**最大 86.2 s** |
| 预注册基准 | 静态"病人平均场"预测留出发作头 5 s | 中位 **ρ=+0.41**，最高 +0.93 |

**这条基准是本阶段最有复用价值的产物**：它不依赖任何模型，且规定了将来任何状态
必须越过的线——不是零。

### 数据缺陷（已修或已记）

- 一位病人 8 次"发作"实为同一次被检测器切成八条（同块内 3.6 min、时长全 0）。
- 15 条发作时长恰为 0.0 s，inventory 却标"区间完整"。
- **2 条 `block_id` 指向比自己发作早约 14 h 结束的块**，已在读取层修复并各救回 1 次发作。
- 5 位 Yuquan 病人 0 条发作记录 = **未检出**，非无发作。
- 3 位 Yuquan 病人无电极坐标（13 次发作）→ 偏侧性不可算。
- ⚠️ **未结清**：`epilepsiae_922` 8 次同锚点对拍 < 0.8，已排除信号强弱 / 电极数 / 基线长度 /
  时间平移四种解释；同病人对照发作零平移处仍 +0.996。

---

## 2. 撤回的内容（不得引用）

原 2026-09-02 上午版报告中，一切基于 `P_local` / `P_slow` 的数字与结论**全部撤回**。
三条独立理由：

1. **读的不是合同定义的状态。** 合同 §4 承重对象是冻结功能读出 `S_func`，明写隐变量只作诊断；
   B1/B2 读的是 `anchor_state.npz::state`（原始 latent）。
   *精确性补注*：余弦相似度对**正交旋转**不变，故"latent 可旋转"本身不构成反驳；
   成立的是**各维异尺度重参数化**下不不变，以及合同本就指定 `S_func`。
2. **producer 不可验收**（见 §3）。
3. **关键对照未执行**：`B_multiscale`、recent/current IED、以及合同 §6 的主时间零假设
   **within-session block circular shift** 均未实现。

B1（离发作还有多久）本轮判为 **assay not estimable**——133 格仅 11 格越过
"不得差于纯截距"闸门，19 人中 15 人零可用格。**其负增量不得读作"状态有害"。**

---

## 3. 本阶段建立的验收闸门（对后续有约束力）

registry loader 加入**默认开启**的 fail-closed 校验（诊断可显式 `verify=False`，
输出带 `verified=False`）。校验两条：**producer 内部 cell 配置同质性**、
**声明的 `checkpoint_sha256` 与盘上一致**。

对 2026-09-02 04:56 版 registry 实测：

| producer | 可采纳 cell | 结论 |
|---|---:|---|
| `P_memoryless` | **22** | 配置同质 |
| `P_local` | **0** | cell 跨 **29** 种配置（单 seed 内 10 种） |
| `P_slow` | **0** | cell 跨 **27** 种（单 seed 内 9 种） |

`checkpoint_sha256` 抽查 **10/10 吻合** → **非文件损坏，是配置异质**。
承载假设的两个递归 producer 恰好全部不可采纳。

**provenance 倒挂（可验证）**：registry 声明 `source_commit=54845f4d`（提交于 **2026-09-02 04:13**），
而其描述的 artifact 写于 **2026-09-01 23:04**——**声明源码晚于自身产物 5 小时**，
不可能是产出它们的代码。（更早一版声明的 `18027162` 提交于 09-01 23:08，亦晚 4 分钟。）
> 注：审阅原文称该 commit "不含实际训练代码"；实测该 commit 存在且含 26 个 `v02` 文件。
> 上面的时间倒挂才是可验证且更强的表述。

---

## 4. 转交上游的两个阻断项

1. **`B_multiscale` 存在未来标签泄漏（已逐行确认）**
   `v02/baseline.py:251` 由 `min(since_prev, to_next)` 构造 `log_time_to_nearest_seizure`，
   而 `to_next = onsets[j] - grid.t_anchor` 是距**下一次**发作的时间。
   → 本线原「照样导出 111 维」的请求**作废**；须先删除一切 `to_next` 派生维度。
2. **`S_func` 冻结功能读出未导出**，H2b 主分析无法进行。

（两份 issue 已落在 `.../v0_2/shared/issues/`。）

---

## 5. 本阶段修正的自身错误（防复发）

| # | 错误 | 后果 | 状态 |
|---|---|---|---|
| 1 | `np.savez` 给文件名补 `.npz`，打断原子写 | 产物写不进目标路径 | 封装 `save_npz_atomic` + 回归测试 |
| 2 | 打分器要求"事件档必须被完整观测" | **会丢弃真实发作**（2.5 h 发作 + 3 h 覆盖） | 放宽为正确判据 + 边界测试 |
| 3 | registry 读取器在缺请求 seed 时静默回退 | **伪造三次重复实验** | 拒绝并列出可用 seed |
| 4 | 重跑子集截断全队列 status 表 | 271 行 → 60 行 | 改为从盘上全部 JSON 重建 |
| 5 | 把"两次留出发作相似度"写成"预测器上限" | 同一张表当场证伪（静态基线在 9/12 人反超） | 交付物内更正 |
| 6 | `eval_events` 报的是 5 min 网格行 | 916 上 1578 行仅 **40** 次发作；曾报 987 实为 **24**，**虚报 41 倍** | 拆分为行数 / 独立发作数 |
| 7 | 温度网格最大 4.0，**不含**均匀权重 | "基线是严格嵌套特例"在模型选择中不成立（距均匀 3%） | 加入 `inf`；留一现可真正选中基线 |
| 8 | 首版校验规则过严 | 把记账哈希差异当缺陷，**154 cell 全被拒**，掩盖真问题 | 改为只判 producer 内部配置异质 |

---

## 6. 工件位置

- 代码：`src/topic5_h2b_transfer/`（crosswalk / risk_grid / early_field / normalization /
  scoring / attach / field_predict / registry）、`scripts/topic5_h2b_transfer/`（10 个 CLI）
- 测试：`tests/test_topic5_h2b_*.py` — **105 项全过**（每项对应合同一条 clause）
- 支持表：`results/epi_prssm/group_event_state/v0_2/h2b/support/`（8 张）
- 图：`.../h2b/figures/`（起点对齐质检图 PNG+PDF+metadata+中文 README，已目视验收）
- 报告：`.../h2b/{plain_language_report,technical_report}.md`（**第二版，含撤回声明**）
- 逐格 JSON + 队列汇总：`.../h2b/machine/`
- 大产物：`/data/hfosp_group_event_state_v0_2/agent_b/`（22 MB，worktree 关闭后仍在）

## 7. 下一版必须先做

1. 上游导出 `S_func`（带名称、TRAIN 标准化统计、checkpoint hash），主分析只读它。
2. 上游统一 `P_local`/`P_slow` 的 cell 配置或说明 27–29 种差异来源，并修正 registry provenance。
3. 去掉 `B_multiscale` 的前瞻维度，重建因果可用基线。
4. 本线补 recent/current IED 与 **block circular shift** 两条对照；B1 补 calibration、
   seizure-level ranking、episode 内先汇总再 patient-first。
5. `epilepsiae_922` target provenance 闭合前，按已预声明的排除敏感性并行报告。

**未触碰**：formal/sealed 分区、paper-ready Fig1–Fig4、Agent A/C 的代码与队列、v0.1 输出与 tag。
