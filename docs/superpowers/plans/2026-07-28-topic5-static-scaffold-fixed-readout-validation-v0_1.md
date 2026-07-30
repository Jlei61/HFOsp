# Topic 5 static scaffold fixed-readout validation v0.1 execution plan

## Milestone A：上一 goal 修订和输入冻结

- 将 internal-state v0.1 拆成 static scaffold、interictal order sensitivity、
  exploratory state read-back 三层；
- Figure 状态改为 supplementary exploratory candidate；
- 写出 `POSTREVIEW_ACCEPTANCE.json`；
- metadata-only 审计 strict 16 人/106 seizures、模型场、几何、shaft 和 confound
  availability。

验收：不重新解释旧 P 值，不把 target-reused 结果写成独立确认。

## Milestone B：现有场的 fixed signed readout

- 唯一 primary field：participation；
- 唯一 primary direction：positive signed Spearman；
- 现有六模型全部重算；
- 5,000 all-contact coherent permutations；
- 5,000 within-shaft circular transforms；
- within-shaft reversal/dihedral sensitivity；
- 13 人 geometry-complete RBF smooth surrogate；
- absolute correlation 只作 sensitivity。

输出 patient-level CSV、paired model table、null audit 和 cohort JSON。

## Milestone C：正则化非递归 baseline

- 从 dataset v0.4 的 train60/validation20 构建 raw counts；
- target-free 选择 beta-binomial shrinkage；
- target-free 选择 shaft/geometry Laplacian penalty；
- Dirichlet-smoothed contact×rank histogram；
- NMF rank `1–4`；
- 每个 estimator 写参数、validation score、contact order 和 fingerprint。

验收：target 值不进入 baseline 选择；所有 field 在读取 target 前写入 freeze manifest。

## Milestone D：free rollout / teacher-forced 分解

- 复用冻结 checkpoints；
- 在相同 heldout20 prefixes 导出 event-first one-step aggregate；
- 对比 empirical、smoothed、teacher-forced、free-rollout 四个场；
- 报告场间相关、平滑度、effective degrees of freedom 和 early-ictal signed score。

验收：teacher-forced 与 free-rollout 的事件分母、contact denominator 和 seed collapse
完全记录。

## Milestone E：contact confound

### E1 快速可用层

- shaft position；
- contact spacing/local density；
- geometry PCs；
- SOZ（13 人）；
- raw participation 作为 GRU 增量 baseline，不作为 scaffold nuisance。

### E2 baseline power 长任务

- 使用已有 confound-map producer；
- 每患者单独日志和 checkpoint；
- 限制 I/O workers，避免原始数据并发过高；
- 缺失/失败患者保留原因，不阻塞 E1。

GM/WM 和 artifact rate 当前不构建、不插补。

## Milestone F：统计和图

建议六块：

| Panel | 科学含义 |
|---|---|
| A | 固定 participation readout 与 signed scoring |
| B | raw/smoothed/teacher-forced/free-rollout contact fields |
| C | full GRU 与正则化非递归 baseline 的 paired comparison |
| D | all-contact 到 shaft-preserving/geometry-smooth null ladder |
| E | contact-confound residual sensitivity |
| F | 代表患者：间期固定场与多次 seizure energy，不做 target-guided field selection |

图目录同步写中文 `README.md`；图和报告明确 target-reused internal-validation 属性。

## Milestone G：结论

分别给 Claim S1–S4 证据等级，不使用一个总 hard gate。下一步只在有证据支持的对象上继续：

- S1 阳性、S2 阴性：论文主语改为 regularized static contact scaffold；
- S2/S3 阳性：再开 ordered-history necessity / matched-state-swap；
- 强空间 null 不稳：保留 all-contact paper-compatible结果，但明确可能由 shaft/geometry
  smoothness 解释；
- confound coverage 不足：不做无混杂机制声明，转为数据补全任务。

## 资源与监控

- Milestone B/C/D 以 CPU 为主，8–12 workers，每 worker 1–2 threads；
- geometry surrogate 按患者分 cell，固定 5,000 draws；
- baseline power 原始数据任务限制 2–4 I/O workers；
- 每个 cell 原子写入状态、日志、seed、输入 hash 和 target-read flag；
- 单 cell 失败不停止其他患者，也不修改预定义 field 或 null。
