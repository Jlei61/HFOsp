# Topic 4 Archive Index — Model layer (BHPN-toy / BHPN-fit)

> **主架构入口**：`docs/paper1_framework_sba.md`（v1.1.2 lock 2026-04-30 / PR-7 addendum 2026-05-01）
> **范围**：Topic 1 / Topic 3 实验数据之上的模型层——**Stereotyped Bidirectional Attractor (SBA)** 框架，包含 BHPN-toy（large-N simulation 验证 P3）+ BHPN-fit（aggregate + conditional predictions）+ 5 dumb baselines。
> **不属于**：Topic 2（事件间 ~2 Hz / refractory + slow modulation；属独立 Paper 2）。

模型层主文档（formal entry）尚未单独立 `docs/topic4_model.md`；当前所有 model-level 内容都在
`docs/paper1_framework_sba.md` 中：§5 sharp predictions、§6 toy model 数学合同、§7 fitted model spec、§8 dumb baselines、§9 失败模式判据。

本目录归档：模型层的 plan、被取代的早期框架、未来 PR-T-4-* / PR-9 等 model-development 工作。

## 文档

### `layered_model_framework.md`
2026 年 4 月被 `paper1_framework_sba.md` 取代的早期分层模型框架（intra-event Kuramoto 保留 + 不再追求 inter-event 大一统）。SBA 框架的概念前身——保留作为思路演变的历史记录。

## 子目录

### `pr_t4_1_bhpn_toy/` — BHPN-toy plan
- `pr_t4_1_bhpn_toy_plan_2026-05-01.md` — plan-of-record：toy model 最小数学合同 + large-N simulation 验证 P3

## 待写

- `docs/topic4_model.md`（formal entry）— 当 BHPN-toy/BHPN-fit 有独立结果且需要从 paper1_framework_sba.md 抽出 model-only 描述时建立
