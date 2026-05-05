# PR-7 Addendum — P3 Cohort-Level Equivalence Test

> **执行日期**：2026-05-01
> **上游 framework**：`docs/paper1_framework_sba.md` v1.1.2 §5.3
> **触发动机**：framework v1.0 把 "Wilcoxon p > 0.05 → P3 PASS" 误用为 fail-to-reject = positive evidence；v1.1 → v1.1.2 改为 cohort-level robust median + bootstrap CI + TOST(δ_excess=0.05) + leave-one-out / leave-548-out sanity；本 addendum 在原 PR-7 数据基础上补做这套等价性检验
> **范围声明（lock）**：本 addendum 不重跑 cluster pipeline / pairing 算法 / null 仿真；仅消费已计算的 `excess`（N2 主 null）+ `lag1_same_excess` + `run_length_lift`，做 cohort-level 等价性判定
> **代码**：`scripts/pr7_addendum_p3_equivalence.py`
> **结果**：`results/interictal_propagation/template_pairing/pr7_addendum_p3.json`

---

## 1. 一句话结论

**P3 verdict = INCONCLUSIVE**（compatible with mark-independent within tested precision；TOST(δ=0.05) cohort CI underpowered at n=6 + structural outliers）。SBA framework **不被** falsified——v1.1.2 §5.3 已 lock INCONCLUSIVE ≠ failure；NULL 档位（cohort robust |excess| > δ + leave-one-out 仍 > δ）**未触发**。Paper 1 主文档 P3 维持 "compatible with mark-independent" 写法，**禁止**写 PASS。

---

## 2. Cohort & 输入数据

PR-2.5 forward/reverse-reproduced cohort，n=6（与 PR-7 H1 主分析一致）：

| Subject | n_used | n_T_a | n_T_b | P(fwd) | P(rev) |
|---|---:|---:|---:|---:|---:|
| epilepsiae_1073 | 193171 | 118752 | 74419 | 0.615 | 0.385 |
| epilepsiae_139 | 14438 | 8429 | 6009 | 0.584 | 0.416 |
| **epilepsiae_548** | 25282 | 5060 | 20222 | **0.200** | **0.800** |
| epilepsiae_635 | 13973 | 7252 | 6721 | 0.519 | 0.481 |
| epilepsiae_958 | 165577 | 79170 | 86407 | 0.478 | 0.522 |
| yuquan_chenziyang | 9609 | 4055 | 5554 | 0.422 | 0.578 |

**关键观察**：cohort marginal P(fwd) median = 0.498，**但** 548 极端 P(fwd)=0.20。这正是 framework v1.1.1 → v1.1.2 修订的触发原因——v1.1.1 字面 criterion 3（lag-1 ≈ 0.5、geometric(0.5) KS）与该 marginal 结构不兼容；v1.1.2 改用 null-relative metric（vs N2，N2 保 marginal）。

---

## 3. 等价性检验合同（lock）

按 framework v1.1.2 §5.3：

- **δ_excess = 0.05**（pre-registered scientific equivalence margin，**不**引用 PR-7 实测；论证仅基于 naive coupled storage 仿真预测 |excess| ~0.3–0.5，δ ≈ 1/6 至 1/10）
- **bootstrap n=10000**, RNG seed=0, alpha=0.05
- **TOST**：H1_lower (median > target − δ) AND H1_upper (median < target + δ)，bootstrap p_lower = frac(boot_medians ≤ target − δ), p_upper = frac(≥ target + δ), TOST p = max(p_lower, p_upper)
- **PASS 判据（每条 metric）**：TOST p < 0.05 AND CI ⊆ [target − δ, target + δ] AND median 落在 band 内
- **三档判定**：PASS / INCONCLUSIVE / NULL，加 SENSITIVITY-only sanity 档（archive only，**禁止**升级主结论）

主 null = N2（local-window shuffle, 30 min, preserve marginal）。

---

## 4. 主结果

### 4.1 T1 — Cohort excess(Δt) at 4 windows

| Window | Median | CI95 | TOST p | Median inside ±δ | CI inside ±δ | **Verdict** |
|---|---:|---|---:|:---:|:---:|---|
| 10s | −0.0181 | [−0.1247, +0.0219] | 0.0624 | ✓ | ✗ | INCONCLUSIVE |
| 30s | −0.0148 | [−0.1103, +0.0104] | 0.0608 | ✓ | ✗ | INCONCLUSIVE |
| 60s | −0.0101 | [−0.0947, +0.0045] | 0.0615 | ✓ | ✗ | INCONCLUSIVE |
| **1800s** | **−0.0002** | **[−0.0015, +0.0002]** | **<0.0001** | ✓ | ✓ | **PASS** |

**解读**：
- 1800s 窗 cohort-level 全部 PASS，所有 6 subject 都很接近 0（max |excess(1800s)| = 0.0025）→ 长窗 retrieval 在 cohort level 干净 i.i.d.-compatible
- 10/30/60s 窗的 INCONCLUSIVE 完全由 548 driven（参 §4.4 leave-548-out）
- TOST p ≈ 0.06 几乎踩在 alpha=0.05 边缘 → underpowered 而非 negative

### 4.2 T2 — Cohort lag1_same_excess (vs N2 null)

| Median | CI95 | TOST p | **Verdict** |
|---:|---|---:|---|
| −0.0111 | [−0.0306, +0.0082] | <0.0001 | **PASS** |

**Per-subject lag1_same_excess values**：1073 (−0.034), 139 (+0.002), 548 (+0.014), 635 (−0.028), 958 (−0.005), chenziyang (−0.017) → 全部 6/6 落在 ±δ 内。

**解读**：lag-1 same-label 的 null-relative excess（即"超出 marginal i.i.d. 的同标偏向"）cohort level **干净** PASS。这是 P3 ε_id i.i.d. 假设的最强支持点。

### 4.3 T3 — Cohort run_length_lift (vs N2 null, target=1.0)

| Median | CI95 | TOST p | **Verdict** |
|---:|---|---:|---|
| 0.9774 | [0.9396, 1.0293] | 0.1149 | INCONCLUSIVE |

**Per-subject run_length_lift**：1073 (**0.9324** ← below 0.95), 139 (1.0048), 548 (**1.0537** ← above 1.05), 635 (**0.9468** ← below 0.95), 958 (0.9894), chenziyang (0.9655)。

**解读**：T3 是结构性 INCONCLUSIVE——三个 subject (1073, 635, 548) 在 ±δ band 外；leave-one-out（§4.4）显示 drop 任何单一 subject 都不能让 CI 进入 [0.95, 1.05]。这说明 run_length lift 的 cohort 离散度（在 n=6 + 三方向 outlier 下）天然超过 δ=0.05 等价 margin 的 power。**不是单一 outlier 问题，是 cohort 量级问题**。

### 4.4 Leave-one-out / leave-548-out sanity（archive only, 不进 PASS gate）

#### T1 excess(10s) leave-one-out

| Drop subject | Cohort median | CI95 | equiv_pass |
|---|---:|---|:---:|
| 1073 | −0.0444 | [−0.2012, +0.0297] | ✗ |
| 139 | +0.0082 | [−0.2012, +0.0297] | ✗ |
| **548** | **+0.0082** | **[−0.0483, +0.0297]** | **✓** |
| 635 | −0.0444 | [−0.2012, +0.0140] | ✗ |
| 958 | +0.0082 | [−0.2012, +0.0297] | ✗ |
| chenziyang | −0.0444 | [−0.2012, +0.0297] | ✗ |

**Pattern**：仅 drop-548 让 10s 转 PASS；drop 任何其他 subject CI 仍 ⊃ 548。30s / 60s 窗模式相同。**短窗 P3 不通过完全是 548 单独 driving**。

#### T3 run_length_lift leave-one-out

| Drop subject | Cohort median | CI95 | equiv_pass |
|---|---:|---|:---:|
| 1073 | 0.9894 | [0.9468, 1.0537] | ✗ |
| 139 | 0.9655 | [0.9324, 1.0537] | ✗ |
| 548 | 0.9655 | [0.9324, 1.0048] | ✗ |
| 635 | 0.9894 | [0.9324, 1.0537] | ✗ |
| 958 | 0.9655 | [0.9324, 1.0537] | ✗ |
| chenziyang | 0.9894 | [0.9324, 1.0537] | ✗ |

**Pattern**：drop **任何**单一 subject 都不能让 T3 PASS——CI 始终跨 [0.95, 1.05]。**结构性 INCONCLUSIVE**，不是单 outlier 现象。

#### Leave-548-out 综合 verdict

| 测试 | leave-548-out 结果 |
|---|:---:|
| excess(10s) | PASS |
| excess(30s) | PASS |
| excess(60s) | PASS |
| excess(1800s) | PASS（主已 PASS） |
| lag1_same_excess | PASS（主已 PASS） |
| run_length_lift | **FAIL**（结构性，与 548 无关） |

leave-548-out 综合 = **不 PASS**（因 T3 结构性 fail）。按 v1.1.1 §5.3 SENSITIVITY-only 档位规则，本 cohort **不**满足 SENSITIVITY-only 升级条件。verdict 维持 INCONCLUSIVE。

---

## 5. v1.1.2 三档落档判定

| 档位 | 触发条件 | 本 addendum 结果 |
|---|---|:---:|
| **PASS** | T1 全 4 窗 + T2 + T3 全部 cohort-level equivalence pass | ✗ (T1 短窗 + T3 fail) |
| **NULL** | cohort robust median \|excess\| > δ AND leave-one-out 仍 > δ | ✗ (无窗 trigger) |
| **SENSITIVITY-only** | 主 INCONCLUSIVE 但 leave-one-out / leave-548-out 全部 PASS | ✗ (T3 leave-out 仍 fail) |
| **INCONCLUSIVE** | 上述都不满足 | ✓ |

→ **VERDICT = INCONCLUSIVE**

**写法纪律（v1.1.1 §5.3 lock）**：
- 主文档 P3 status：**"compatible with mark-independent within tested precision (TOST(δ=0.05) cohort CI underpowered at n=6 with structural outliers)"**
- **禁止**写"P3 PASS"
- **禁止**写"PR-7 NULL → 独立证明"
- **禁止**用 leave-548-out 作为主结论替代

---

## 6. SBA framework 受影响范围

按 framework v1.1.2 §8 失败模式表：

| 框架组件 | 受 INCONCLUSIVE 影响？ | 处置 |
|---|:---:|---|
| H₀ SBA 单核心假设 | ✗ | INCONCLUSIVE ≠ falsified；hypothesis 维持 |
| toy model BHPN-toy（§4.4 ε_id i.i.d.）| ✗ | 与 T2 PASS + 1800s PASS 一致；T1 短窗 underpowered 不构成对 ε_id i.i.d. 的反证 |
| P3 prediction | INCONCLUSIVE | 主文档 P3 写 "compatible"，**不**写 PASS |
| 其他 prediction (P1/P2/P4/P5) | ✗ | 不受影响 |
| Paper 1 ceiling | 微降 | 从 "P1 + P2 + P3 PASS" 降为 "P1 + P2 PASS, P3 compatible-pending-larger-cohort" |

**Paper 1 写作影响（lock）**：
- §1 现象学层面无变化（PR-2 / PR-2.5 / PR-6 PASS 维持）
- §3 toy model 章节维持（SBA 不被 falsified）
- §5 P3 prediction test 段必须写为 "数据 compatible with mark-independent retrieval；cohort-level TOST 在 n=6 下 underpowered；1800s 长窗 + lag1 null-relative 干净 PASS，10/30/60s 短窗与 run_length_lift 因 cohort 量级 + 结构性 outlier 维持 INCONCLUSIVE"
- Discussion 必须显式声明：**INCONCLUSIVE 不是 negative result，是小 cohort + 严格等价性框架的代价**；扩 cohort 到 n>15 是未来 follow-up

---

## 7. 后续可选 follow-up（不阻塞 Paper 1）

按 v1.1.1 §5.3 / framework §10.3 范围，下述属 Paper 3 / 未来 work，**不**进 Paper 1：

1. 扩 fwd/rev cohort（PR-2.5 H2 cohort 上界 n=6，扩展需放宽 cohort 入选门槛或纳入 endpoint_defined-only subject，留新 PR）
2. Burst-level / rate-state-conditional 重新检验（PR-7 §17 已识别为 Paper 3 候选）
3. History-dependent marked point process model（PR-7 §17 + framework §10.3 已识别）
4. T3 run_length_lift 的 INCONCLUSIVE 进一步理解：是 burst-level 真实 deviation 还是 N2 null 局部窗口效应？需要 alternative null definitions（不在本 addendum 范围）

---

## 8. 自检清单

- [x] 数据来源限于已计算 PR-7 输出，不重跑 pairing / null
- [x] δ_excess = 0.05 来自 framework v1.1.2 lock，**不**引用 PR-7 实测数据为 δ 背书
- [x] 主判据严格 cohort-level（TOST + bootstrap CI），subject-level 仅 sanity diagnostic
- [x] leave-548-out / leave-one-out 全部纳入 archive，不替代主判据
- [x] N2 主 null（保 marginal）正确选择，避开 marginal-50/50 假设漏洞
- [x] T2 / T3 用 null-relative metric（lag1_same_excess vs N2 / run_length_lift vs N2），不用 marginal-50/50 形式
- [x] verdict 三档 (PASS / INCONCLUSIVE / NULL) + SENSITIVITY-only 档位严格按 v1.1.2 §5.3 判定
- [x] 写法纪律：禁止"PR-7 PASS / 证明独立 / leave-out 替代主结论"
- [x] framework v1.1.2 §5.3 PR-7 verdict status 同步更新（见下方）
- [x] topic1 §2 一句话当前结论同步更新（P3 status 从 "PASS 撤回 / pending" 升到 "INCONCLUSIVE locked"）

---

## 9. 链接

- 上游 framework：`docs/paper1_framework_sba.md` v1.1.2 §5.3
- PR-7 主结果：`docs/archive/topic1/pr7_template_pairing/pr7_template_pairing_results_2026-04-29.md` §17
- 代码：`scripts/pr7_addendum_p3_equivalence.py`
- 数值结果：`results/interictal_propagation/template_pairing/pr7_addendum_p3.json`
- topic 主文档：`docs/topic1_within_event_dynamics.md` §2
