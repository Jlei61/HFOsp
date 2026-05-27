# H2b Direction-Axis Cohort Run — 2026-05-25

> **Contract**: `phase_h2b_direction_axis_plan_2026-05-25.md` v1.0.2
> **代码版本**：`src/sef_itp_direction_axis.py` v1.0.0 + runner v1.0.2 fixes（r² gate + SOZ double-cut + channel-order check）
> **n_perm**：1000；**seed**：0；**k_event_default**：2；`R2_DESCRIPTIVE_MIN` = 0.20
> **输入 cohort**：`results/interictal_propagation_masked/rank_displacement/per_subject/*.json` 40 个 subject
> **输出**：`results/topic4_sef_itp/direction_axis/{per_subject/*.json, figures/*.png, cohort_summary.json}`
> **状态**：v1.0.2 demoted lock；本 archive 是数字快照 + audit trail，不是 contract（contract 见 phase plan）。

---

## 0. 命名空间澄清（避免 verdict 标签 vs swap_class 混淆）

H2b 涉及两个不同的"strict / candidate / inconclusive"名字空间，本文档严格区分：

| 名字空间 | 取值 | 出处 |
|---|---|---|
| **swap_class**（label 层） | `strict` / `candidate` / `none` / `inconclusive` | 上游 `rank_displacement` 的 `swap_sweep.swap_class`，family-wise null 控下的 source/sink 角色反转判定 |
| **strict verdict**（H2b verdict label） | `axis_reversal` / `dual_source` / `same_direction` / `degenerate_geometry` / `inconclusive` / `exit_no_universe` | H2b 本 phase 输出，per archive plan §3.7 |
| **descriptive shape**（H2b shape label） | `axis_reversal_shaped` / `dual_source_shaped` / `same_direction_shaped` / `unclear` / `missing` (= exited) | H2b 本 phase 输出，per archive plan §3.6.5 |

下文 "swap=strict / swap=candidate / swap=none" 一律指 swap_class（label 层），"verdict / shape" 一律指 H2b 输出。

---

## 1. Cohort 漏斗

- 输入：rank_displacement primary pair 40 个 subject（Yuquan + Epilepsiae 合并）
- `exit_no_universe`（universe = joint_valid AND mapped_mask 后通道数 < `max(6, 2 · decision_k)`）：**17 个 subject**
- 可测（进入 verdict + shape 计数）：**23 个 subject**

按 swap_class 分层的可测数：

| swap_class | total | testable | exit_no_universe |
|---|---|---|---|
| strict     | 9  | 5  | 4 |
| candidate  | 8  | 4  | 4 |
| none       | 23 | 14 | 9 |
| **total**  | **40** | **23** | **17** |

---

## 2. Strict verdict 全 cohort 分布

| verdict | n |
|---|---|
| axis_reversal | **0** |
| dual_source | 5 |
| same_direction | 3 |
| degenerate_geometry | 6 |
| inconclusive | 9 |
| exit_no_universe | 17 |
| **total** | **40** |

按 swap_class 分层：

| | axis_reversal | dual_source | same_direction | degenerate_geometry | inconclusive | exit_no_universe |
|---|---|---|---|---|---|---|
| swap=strict (n=9)     | 0 | 0 | 0 | 2 | 3 | 4 |
| swap=candidate (n=8)  | 0 | 0 | 0 | 1 | 3 | 4 |
| swap=none (n=23)      | 0 | 5 | 3 | 3 | 3 | 9 |

**关键观察**：strict verdict 在全 40 个 subject 上 0 个 `axis_reversal`。原因来自 permutation null 自由度退化（archive plan §3.6.5）：当 `decision_k ≈ n_universe / 2` 时，per-cluster role-shuffle null 能产生的 source/sink partition 数量有限，即使观测到 `cos(v_A, −v_B) ≈ 0.95` 也常常无法 reject 该 null（p > 0.05）。

---

## 3. Descriptive shape 全 cohort 分布（v1.0.2 r² gate 修复后）

| shape | n |
|---|---|
| axis_reversal_shaped | 8 |
| dual_source_shaped | 5 |
| same_direction_shaped | 1 |
| unclear | 9 |
| missing (= exit_no_universe) | 17 |
| **total** | **40** |

按 swap_class 分层：

| | axis_reversal_shaped | dual_source_shaped | same_direction_shaped | unclear | missing |
|---|---|---|---|---|---|
| swap=strict (n=9)     | 2 | 0 | 0 | 3 | 4 |
| swap=candidate (n=8)  | 3 | 0 | 0 | 1 | 4 |
| swap=none (n=23)      | 3 | 5 | 1 | 5 | 9 |

合并 swap=strict + swap=candidate：

| | n |
|---|---|
| testable | 9 |
| axis_reversal_shaped | **5** |
| dual_source_shaped | **0** |
| same_direction_shaped | 0 |
| unclear | 4 |

**注**：4 个 unclear 中，3 个由 PCA `λ₂/λ₁ < 0.10` 单 shaft 退化触发（epi_139、yuquan_zhangjiaqi、epi_620）；1 个由 r² gate 触发（epi_1146，r²=0.185 < 0.20，cos_neg=0.948 仍很强但形状信号 r² 不达标）。

---

## 4. Falsifiability check（archive plan §8 双分母版本）

H2b 仅 falsify H_orth（"cluster A 与 cluster B 是正交无关解剖源"）假说。阈值参考：`axis_reversal_shaped ≥ 2 × dual_source_shaped`。

| 分母（archive plan §8 demoted 双报） | n testable | axis_reversal_shaped | dual_source_shaped | ratio | falsify H_orth? |
|---|---|---|---|---|---|
| swap_class ∈ {strict, candidate}（label 层；v1.0.1 用，偏 SEF-ITP-friendly） | 9 | 5 | 0 | ∞ | ✅ 通过 |
| H2 spatial-layer PASS（source ∧ sink 双 PASS，framework v1.0.5 §3.2 13/23）（plan §8 原指定分母）| 待 per-subject 列表跨文档同步 | — | — | — | pending |

**当前 v1.0.2 lock 的可写措辞**（archive plan §10.5）：

- ✅ "swap-positive 子集（label 层 strict+candidate 合并 9 个可测）shape 一致于非正交假说（5/9 axis_reversal_shaped、0/9 dual_source_shaped）"
- ✅ "shape 一致于同轴双向读取（无论是单源双向还是同轴双端各自 seed），不支持 正交 unrelated-source 解释"
- ❌ "支持 SEF-ITP 同一病理核心轴" / "证明 swap 不是双源" / "排除双端 seed 解释" —— 这些都超出 H2b scope（plan §1.5）

degenerate_geometry 占 swap-positive 可测的 3/9 = 33%，低于 50% underpowered 阈值（plan §8）。

---

## 5. 个别 subject 备注

### 5.1 swap=strict（5 个可测）

| stem | n_universe | decision_k | cos(v_A,−v_B) | slope_B_on_axisA | r²_B | strict verdict | descriptive shape |
|---|---|---|---|---|---|---|---|
| epilepsiae_1146   | 15 | 7 | 0.948 | −0.165 | 0.185 | inconclusive (perm p=0.104) | unclear (r²<0.20) |
| epilepsiae_958    | —  | — | 0.946 | −0.349 | 0.586 | inconclusive | axis_reversal_shaped |
| epilepsiae_139    | 6  | 2 | 1.000 | −0.135 | 0.810 | **degenerate_geometry**（PCA λ₂/λ₁<0.10） | unclear |
| yuquan_zhangjiaqi | 6  | 2 | 1.000 | −0.224 | 0.618 | **degenerate_geometry**（同上） | unclear |
| yuquan_zhaochenxi | —  | — | 0.891 | −0.704 | 0.484 | inconclusive | axis_reversal_shaped |

- **epi_1146**：v1.0.1 误标为 axis_reversal_shaped；v1.0.2 r² gate 修复后回正为 unclear。channel-name 5/7 + 6/7 swap-flipped role overlap 仍存在（archive plan §10.3.5），但只能 sanity centroid 数学不是伪结果，**不能**把 shape 救回 axis_reversal_shaped。
- **epi_139 / yuquan_zhangjiaqi**：cos 各为 1.000、perm p 显著（0.039 / 0.016），但 PCA 检测到单 shaft 1D 排列；Layer 4 degeneracy override 强制 verdict = degenerate_geometry、shape = unclear（archive plan §3.5 设计意图正是拦这种 case）。

### 5.2 swap=candidate（4 个可测）

| stem | cos(v_A,−v_B) | slope_B_on_axisA | r²_B | strict verdict | descriptive shape |
|---|---|---|---|---|---|
| epilepsiae_253    | 0.985 | −0.094 | 0.542 | inconclusive | axis_reversal_shaped |
| epilepsiae_384    | 0.785 | −0.104 | 0.288 | inconclusive | axis_reversal_shaped |
| epilepsiae_620    | 0.756 | −0.074 | 0.138 | degenerate_geometry | unclear |
| yuquan_liyouran   | 0.754 | −0.681 | 0.661 | inconclusive | axis_reversal_shaped |

### 5.3 swap=none（14 个可测）

shape 分布混合（archive plan §10.5 预期，非 swap 子集没有方向预测）：5 dual_source_shaped、3 axis_reversal_shaped、1 same_direction_shaped、5 unclear。详见 `cohort_summary.json::subjects`。

3 个 swap=none 但 shape=axis_reversal_shaped（epi_1096、epi_583、epi_916）：几何反向但 swap_sweep family-wise null 没认定 swap_class=strict/candidate；可能 marginal swap signal 或几何巧合；**不进** §4 falsifiability check 的分子分母。

---

## 6. v1.0.2 audit fix 后改动 vs v1.0.1 的差异

| 项 | v1.0.1 | v1.0.2（current） | 数字影响 |
|---|---|---|---|
| `assess_descriptive_geometry` r² gate | 漏掉（docstring 承诺但实现未加） | `R2_DESCRIPTIVE_MIN = 0.20` 显式 gate | strict+candidate 合并 axis_reversal_shaped：6 → **5**（epi_1146 r²=0.185 honestly 落到 unclear）|
| `compute_soz_relation` SOZ 子集 | 只 `soz ∧ universe` | 同时报 `mapped_full` + `joint_universe` 双套距离 | 不影响 verdict / shape 计数；新增 JSON 字段 |
| Runner event-layer channel order | 只检 phase0a vs rank_displacement | 加 lagpat vs rank_displacement 的 strict equality assertion | 不影响默认 cohort（runner 默认 `--with-events` 关闭）；event-layer 启用时杜绝 channel-order 漂移 |
| Cohort 措辞 layer | descriptive 当 cohort 结论层 | strict 是 PRIMARY；descriptive 是 SUPPLEMENTARY shape；falsifiability denom 双报 | 见 archive plan §10.5 demoted lock 措辞 |
| Scope 红线（§1.5 新加） | 未显式锁 | 表格化列出 H2b 能 / 不能区分的对 | "支持同核轴"等措辞 全部禁用 |

---

## 7. 关键 caveat 摘抄（archive plan §1.5 + §11）

`v_A ≈ −v_B` 在 swap_class ∈ {strict, candidate} 的 subject 上很大程度是 `swap_sweep` 角色反转定义的几何重述（source_A ≈ sink_B、sink_A ≈ source_B 由 swap_sweep 定义保证）。

H2b 能区分：

- ✅ "cluster A 与 cluster B 是正交无关解剖源"（H_orth）→ 通过 `dual_source_shaped` 捕获
- ✅ "两模板同向（非 swap）" → 通过 `same_direction_shaped` 捕获
- ✅ 单 shaft 1D 测量伪影 → Layer 4 degeneracy override 捕获

H2b 不能区分：

- ❌ "同一病理核心轴双向读取（SEF-ITP 单源双向）" vs "同轴双端各自独立 seed"——两者都预测 `cos(v_A, −v_B) ≈ +1`

区分上述对必须做 archive plan §11 Round 2：per-event seed centroid 聚类 + rank-distance 连续梯度 + source_A vs source_B 各自相对 SOZ 关系。**未实施**。

---

## 8. Artifact 索引

- 每被试 JSON：`results/topic4_sef_itp/direction_axis/per_subject/<dataset>_<subject>.json`
- 每被试图：`results/topic4_sef_itp/direction_axis/figures/<dataset>_<subject>.png`
- cohort summary JSON：`results/topic4_sef_itp/direction_axis/cohort_summary.json`
- cohort summary 图：`results/topic4_sef_itp/direction_axis/figures/cohort_summary.png`
- 图说明：`results/topic4_sef_itp/direction_axis/figures/README.md`（中文）
- 测试：`tests/test_sef_itp_direction_axis.py`，34/34 通过
- Module / runner / plotter：`src/sef_itp_direction_axis.py`、`scripts/run_sef_itp_direction_axis.py`、`scripts/plot_sef_itp_direction_axis.py`

---

## 9. 后续

- **Round 2**（archive plan §11，未实施）：per-event seed clustering + rank-distance gradient + cohort-level source-SOZ asymmetry — 必须另立 archive plan，不在本 v1.0.2 scope。
- **Spatial-layer denominator pending**：archive plan §10.3 中"H2 spatial PASS 分母"的 per-subject list 需要与 framework v1.0.5 §3.2 cohort 同步；当前数字仅在 label 层（swap_class）分母上给出。
