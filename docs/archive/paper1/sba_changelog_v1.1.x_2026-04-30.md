# SBA Framework Changelog v1.1.x (Archived 2026-05-22)

> **归档说明**：这是 `docs/paper1_framework_sba.md` §15 Changelog 的完整原文，记录 2026-04-30 当日三次 lock-time 修订（v1.1 → v1.1.1 → v1.1.2）。
>
> 主 doc 保留：当前版本 = **v1.1.2**，其中 BHPN-toy / BHPN-fit 已被 SEF-ITP 取代（见 [bhpn_toy_and_fit_spec_superseded_2026-04-30.md](bhpn_toy_and_fit_spec_superseded_2026-04-30.md)）。任何 paper writing 引用 framework 修订史，直接 link 到本归档即可。

---

## v1.1 — 2026-04-30（initial framework lock）

v1.0 发布后立即识别五条结构性硬伤，全部在 v1.1 根除。**任何下游 PR plan-of-record 必须继承 v1.1，不允许继承 v1.0 的判据**。

| # | v1.0 错误 | v1.1 修订 | 影响范围 |
|---|---|---|---|
| 1 | P5 用 `A_ij = cos(φ_i\*−φ_j\*)` 当 directed graph 预测 directed ictal coherence。cos 偶函数无方向，数学上不能预测 i→j vs j→i。 | P5 directional predictor 改为 `D_ij = sin(φ_j\*−φ_i\*)`（反对称） + 备选 rank gradient `R_ij`；A_ij 仅描述 storage dynamics。Source code grep 在 PR-9 必须 raise on cos-based directed predictor。 | §3 / §4.2 / §5.5 / §13 / §11 |
| 2 | P3 用 "Wilcoxon p > 0.05 → PASS"，把 fail-to-reject 当作 positive evidence for null（统计学错误）。 | P3 改用 TOST equivalence test + bootstrap CI；equivalence margin **δ_excess = 0.05** lock at framework time；PASS 必须 \|excess\| < δ + TOST p < 0.05 + CI ⊆ [−δ, +δ]。**PR-7 v1.0 PASS verdict 撤回**，待 PR-7 addendum（~2 天）重判。新增 INCONCLUSIVE 档位区分 underpowered vs negative。 | §3 / §5.3 / §8 / §11 / §13 |
| 3 | 单核心假设说 "SOZ 内存在低维结构性骨架"，预设了 "clinical SOZ = true pathological core"；但 PR-T3-1 的设计目的就是审计 clinical SOZ 可靠性，不该在框架第一行就预设它。 | 改为 "在 multi-source SOZ proxy 富集的局部脑网络（pathological core neighborhood）中"；显式声明不预设 clinical SOZ = true SOZ。 | §3 / §13 |
| 4 | Slow latent state s(t) 同时承担 (a) PR-7 短窗 mark-independent (τ_s ≪ 10s) 与 (b) 与 Topic 2 min–hour modulation 兼容——两者在时间尺度上自相矛盾。 | 拆为两个统计独立过程：`s_rate(t)` (min–hour OU, 调制 event rate, 与 Topic 2 一致) + `ε_id` (per-retrieval 随机初始相位 i.i.d., 调制 attractor identity, 与 PR-7 一致)。两过程**解耦**，无内部时间尺度矛盾。 | §4.4 / §4.5 (T3, T5, T6, T7) / §7 (B5) / §8 / §13 |
| 5 | BHPN-fit F1 承诺 "逐事件预测 fwd/rev label accuracy > 50%"，与 §4.4 i.i.d. 假设逻辑冲突——框架本身禁止该 prediction。 | F1–F4 重写为仅 aggregate-level + conditional 量：F1 template geometry / F2 marginal label distribution + i.i.d.-compatible transition stats / F3 conditional rank pattern / F4 core node anatomy。**禁止逐事件 label prediction** 写入 §6.3 lock。 | §6.3 / §13 / §11 |

## v1.1 → v1.1.1 — 2026-04-30（同日二次修订）

v1.1 发布后立即识别两条剩余问题：

| # | v1.1 问题 | v1.1.1 修订 | 影响范围 |
|---|---|---|---|
| 1 | P3 主判据第 1 条 "subject-level \|excess\| < δ_excess"，但 PR-7 cohort 中 548 是 magnitude outlier（10s = −0.201, 30s = −0.188），单一 subject 字面 PASS 几乎不可能 → v1.1 字面判据事实上不可达；事后排除 outlier 又是 rule fitting | P3 主判据**仅 cohort-level**：cohort robust median + bootstrap CI + run-length / lag-1 distribution；subject-level（含 leave-548-out）降级为 **sanity diagnostic, archive only, 不进 PASS gate**；新增 SENSITIVITY-only 档位明确禁止用 outlier 排除替代主判据 | §5.3 / §8 / §11 / §13 |
| 2 | v1.1 δ_excess 选取理由列出 "PR-7 cohort 实测 \|median\| ∈ [0.002, 0.029]"，是借数据为 δ 背书（轻度循环论证：用 PR-7 数据告诉 δ 该选哪儿，再让 PR-7 数据通过 δ）| δ 论证**仅限**与 naive coupled storage 仿真预测的对比（~0.3–0.5），**不引用 PR-7 实测数字**；toy T3 acceptance 显式声明走 large-N simulation，不依赖 PR-7 cohort | §5.3 / §4.5 (T3) / §13 |

**新增 framework 现实预期声明（lock at v1.1.1）**：

按 PR-7 现有 per-subject 表 + 548 outlier + n=6，P3 在 PR-7 addendum 后**大概率 INCONCLUSIVE 而非 PASS**——cohort median 点估计大概率落在 ±δ_excess 内，但 bootstrap CI 在 n=6 + 548 outlier 下大概率跨 ±0.05；TOST 双侧 p < 0.05 难以通过。这是 honest 等价性框架在小 cohort 下的既有代价，**SBA 不会因 P3 INCONCLUSIVE 被 falsified**。SBA 只在 P3 NULL（cohort robust |excess| > δ 且 leave-one-out 仍 > δ）时被 falsified。

## v1.1.1 → v1.1.2 — 2026-04-30（同日三次修订，PR-7 addendum 数据审计触发）

PR-7 addendum 数据审计阶段识别 v1.1.1 §5.3 criterion 3 的 assumption 漏洞：

| # | v1.1.1 问题 | v1.1.2 修订 | 影响范围 |
|---|---|---|---|
| 1 | criterion 3 写 "cohort lag-1 same-label 率 CI ⊆ [0.5 − δ, 0.5 + δ]" 与 "run-length 分布与 geometric(0.5) 兼容（KS）"。两条都预设 marginal P(fwd)=P(rev)=0.5。但 PR-7 数据审计显示 cohort marginal P(fwd) 中位 0.498、范围 [0.20, 0.61]——**548 极端 P(fwd)=0.20**，对应 lag-1 same under marginal i.i.d. = 0.68 而非 0.5。geometric(0.5) 与 lag-1 ≈ 0.5 的判据是 marginal assumption violation，会**对 marginal asymmetric subject 系统性 fail**，与 i.i.d. 是否成立无关。 | criterion 3 改为 null-relative：cohort `lag1_same_excess` vs N2 null 的 bootstrap CI ⊆ [−δ, +δ]；cohort `run_length_lift` vs N2 null 的 CI ⊆ [1−δ, 1+δ]。N2 null 保 marginal，excess/lift 都是相对 null 的偏差，不预设 marginal 50/50。 | §5.3 / §5.3 PR-7 addendum 范围 / §15 |

**v1.1.2 是 flaw fix, 非 rule fitting**：

新判据并不**比 v1.1.1 字面更松**——cohort lag1_same_excess 和 run_length_lift 仍要求落在 ±δ 内。区别仅在于：
- v1.1.1 字面判据混淆了"marginal 偏 50/50"与"i.i.d. 失败"，会把所有 marginal asymmetric subject 错判为 P3 fail，无论它们的 retrieval 实际是否 i.i.d.
- v1.1.2 把检验对准 "given marginal, 是否 i.i.d."，与 ε_id i.i.d. 假设的实际语义对齐

**Lock at v1.1.2**：marginal-aware null-relative metric 是 P3 第 3 条判据的 final 版本，不允许后续退回 marginal-50/50 形式。

## 后续修订协议

任何对 P1–P5 PASS/NULL/FAIL 判据、δ_excess、toy model 必含组件（s_rate / ε_id / core）、baseline 列表、F1–F4 spec 的修改：

1. 必须先在本文件创建 v1.x changelog 条目，写明动因 + 修订内容 + 影响范围
2. 再同步更新 §3 / §4 / §5 / §6 / §7 / §8 / §11 / §13 对应章节
3. 不允许 PR plan-of-record 单方面引入与本文件不一致的判据；与 framework 冲突的 PR 必须先改 framework
4. v1.1 lock 之后**已经验证的** P1, P2 的 PASS 判据不允许事后调整；**已经撤回的** P3 v1.0 PASS verdict 在 PR-7 addendum 完成前不允许重新写入
5. v1.1.1 lock 之后 P3 的"主判据 cohort-level / subject-level 仅 sanity / δ 不引用 PR-7 实测"三条不允许事后放宽；任何把 sanity 升级为 PASS 的尝试视为 rule fitting
