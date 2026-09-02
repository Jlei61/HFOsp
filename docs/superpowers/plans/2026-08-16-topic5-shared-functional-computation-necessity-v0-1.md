# Topic 5.2 shared functional computation necessity v0.1 — execution plan

> **SUPERSEDED / 不得作为最终执行记录引用**：v0.1 的 lesion centre/support 使用了留出事件完整后半段派生量。最终重跑与验收见 [v0.2 计划](2026-08-16-topic5-shared-functional-computation-necessity-v0-2.md)。

## Phase 0：合同与输入审计

- 冻结 spec、真实网络列表、训练/未见拆分、方向定义、三类对照、primary endpoint 和 patient-first 汇总。
- 记录 630 个 checkpoint 输入；necessity 主执行只覆盖 504 个真实顺序网络 cell，打乱网络仅提供对照方向。
- 核对现有 reference states 全部来自 heldout test，禁止直接用现有全体 patch operator 定义 primary 方向。

## Phase 1：训练侧算子与跨网络留一方向

- 新建 response-blind 训练 reference manifest：每 fit 最多 64 个 train events × 早中晚三状态。
- 对 630 cells 提取训练侧 Gaussian patch → future-contact operator。
- 对每 fit、seed、held-out real arm，用另外三个真实网络构造共同矩阵并冻结第一 SVD 成分。
- 同时冻结打乱网络成分和候选正交/PCA controls；输出方向 hash、解释方差、与 held-out 网络自身成分的只读诊断相似度。

## Phase 2：未见事件 necessity/lesion

- 对 504 个真实网络 cell 读取已有 heldout reference states。
- 在 0.25/0.50/1.00 三档删除共同成分。
- 运行不删除、等量正交、高方差、打乱网络成分分支。
- teacher-force 同一真实后续输入，保存 delayed next-contact NLL、即时 NLL、STOP、logit norm、支持检查和参数 hash。

## Phase 3：汇总与统计

- reference → seed → fit → patient 汇总。
- 计算 dose AUC、共同成分绝对损失、共同减正交、共同减打乱、共同减 PCA。
- Wilcoxon patient-level 检验；三个 primary p 值 Holm 校正；bootstrap patient median 95% CI。
- 分 held-out 网络、phase、fit scope 和患者 denominator 做敏感性分析。

## Phase 4：科学与工程审计

- 回放计数、split、target guard、方向 leave-one-topology-out、control norm matching、支持门、非有限值、参数 hash 和聚合层级逐项审计。
- 至少包含 toy tests：方向符号不变、目标成分删除投影归零、等量位移、NLL rank-set 计算、跨网络留一不含自身、seed→fit→patient 聚合。
- 生成 machine-readable claim adjudication；tests 通过不等于科学 primary 通过。

## Phase 5：图与报告

- primary 通过才更新 Figure 6 候选 I；否则主图保持不变，新结果进补充材料。
- 图必须有 patient-level points/curves、95% CI、实际 n；不使用单患者例子代替 cohort inference。
- PNG/PDF/SVG 同状态重画并逐面板目视 QA；同步更新 figures/README.md。
- 最终中文报告用白话解释：共同成分是什么、如何从三种网络定义、删除了什么、为何对照足够、结果支持到哪一层。
