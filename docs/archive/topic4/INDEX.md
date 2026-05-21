# Topic 4 Archive Index — Model layer

> **Topic 4 formal entry（2026-05-20 起）**：`docs/topic4_sef_itp_framework.md` (**SEF-ITP framework v1**)
> **上游 SBA framework**：`docs/paper1_framework_sba.md`（v1.1.2 lock 2026-04-30 / PR-7 addendum 2026-05-01）—— SEF-ITP 取代其 BHPN-toy / BHPN-fit toy-mechanism 部分；保留其 P1/P2/P3/P5 红线
> **硬前置**：`docs/topic0_methodology_audits.md` §3.1 phantom-rank 修复 + §5 broad re-derivation roadmap
> **范围**：Topic 1 / Topic 3 实验数据之上的模型层；当前主路径 = SEF-ITP（间期模板传播的空间易激场模型）
> **不属于**：Topic 2（事件间 ~2 Hz / refractory + slow modulation；属独立 Paper 2）。

## 当前主路径（2026-05-20 起）

**SEF-ITP**（Spatial Excitable Field model for Interictal Template Propagation）：

- 主文档：`docs/topic4_sef_itp_framework.md`
- 核心断言：间期群体 HFO = 空间组织化病理易激场 θ(x) 被扩散物理反复采样的痕迹
- 6 条 pre-registered 预测（H1–H6）：endpoint compactness / source-sink reversal / mark independence + stable geometry / rate-geometry decoupling / pre-post-ictal endpoint shift / participation-field segregation
- 5 phase 路线：Phase 0 phantom-rank 硬前置 → Phase 1 空间几何免费检验 → Phase 2 temporal × geometry 联合 → Phase 3 ictal-adjacent → Phase 4 最小 FHN neural-field toy
- 转向触发点（2026-05-20）：识别 BHPN-toy 是循环论证（A_ij 矩阵预编码现象 → Kuramoto 必然演化出双稳态 → "复现"是 tautology）

## 历史归档

### `layered_model_framework.md`
2026 年 4 月被 `paper1_framework_sba.md` 取代的早期分层模型框架（intra-event Kuramoto 保留 + 不再追求 inter-event 大一统）。SBA 框架的概念前身——保留作为思路演变的历史记录。

### `pr_t4_1_bhpn_toy/` — BHPN-toy plan（**SUPERSEDED 2026-05-20**）
- `pr_t4_1_bhpn_toy_plan_2026-05-01.md` — plan-of-record v2，**已被 SEF-ITP 取代**
  - 顶部有 SUPERSEDED banner 指向 `docs/topic4_sef_itp_framework.md`
  - 保留为历史归档：v2 修订过程（rotating-frame rank / direction-agnostic mark dependence / TDD unit vs integration 拆分）作为方法学训练范例；其中 "k=2 是对称 Hebbian 数学必然，不是 mechanism discovery" 是触发 SEF-ITP 转向的关键认知

## 跨文档链接

- `docs/topic4_sef_itp_framework.md` §9.1 — SBA framework 取代 / 保留范围
- `docs/topic4_sef_itp_framework.md` §13 — 历史文档索引（含本 INDEX 反向链接）
- `docs/topic0_methodology_audits.md` §5h — Topic 4 attractor on masked re-run 是 SEF-ITP Phase 0 子步
