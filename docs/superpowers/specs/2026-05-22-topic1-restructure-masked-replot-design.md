# Topic 1 主文档重排 + masked 数据补图（D-A + F-A）

> 状态：spec / plan-of-record
> 起源：用户 2026-05-22 反馈 "现在的主文档挡住了之前的科学结论，很多东西都应该降级；PR-6 channel-level swap-k 比 PR-2/2.5 更能说明 sink-source 切换；模板本身的稳定性应该再次强调；topic0 重排了非参与节点的 rank 之后这些数应该有 masked 版的图"
> Advisor 3 条 guardrail 已解决（见 §0）
> 配套：完成后由 writing-plans skill 转为 implementation plan

---

## §0 三条 guardrail（advisor 2026-05-22 强制）

写新 §2 / §3 时必须满足以下三条（不满足 = spec 不合格）：

1. **Pre-registered tier 规则（CLAUDE.md §5）不可破坏**：PR-6 H2 forward/reverse swap 在 archive plan `pr6_template_endpoint_anchoring_plan_2026-04-25.md` §3.3 + AGENTS.md cross-PR 段都注册为 **"directional mechanism sanity, not cohort claim"**。**禁止**在新 §2 把 swap 写成 "primary cohort PASS"。允许写："最强**机制层**证据 = forward/reverse template swap geometry (PR-6 H2 mechanism sanity, fwd/rev 子集 8/8 cleanly positive on masked, **不是** H1 cohort PASS — H1 SOZ anchoring 仍 NULL)"。
2. **K=2 framing 必须精确（CLAUDE.md §8）**：禁止写 "universal k=2"；正确写法 "dominant compression at k=2 (Tier 0: 27/30 + 2×k=4 + 1×k=6; Tier 1: 30/33 + 2×k=4 + 1×k=6)"——多模态少数派必须保留。
3. **§2 / §3 引用 swap classifier 必须用 masked 数**：Step 5f 已确认 swap_class_strict 10→9, candidate 8→6, Step 6 concordance 0.69 → **0.82**（实质提升）, fwd/rev subset 8/8 (masked, was 9/9), H1 SOZ anchoring p 仍 NULL, node anatomy h1_eligible Wilcoxon p=0.014→**0.059** (secondary metric 进一步弱化)。Source: `docs/archive/topic0/lagpat_phantom_rank/step5f_pr6_results_2026-05-21.md`。

---

## §1 三个 deliverable

| Deliverable | 目标 | 输入 | 输出 |
|---|---|---|---|
| **D1: 目录迁移** | `interictal_propagation_masked/` 提升到 results/ 顶层，符合 AGENTS.md parallel-dir 约定 | `results/interictal_propagation_vs_masked/interictal_propagation_masked/` | `results/interictal_propagation_masked/`（git mv）+ 修复 AGENTS.md + topic0 archive + scripts 里所有"嵌套"路径引用 |
| **D2: masked figure 重画（F-A 分两批）** | 在 masked 数据上重画支撑两条主结论的 figures | masked per_subject JSON + cohort_summary | Batch 1 高优先（PR-2/3 cohort + PR-6 template_anchoring main + PR-6 swap/displacement）→ 用户 eyeball → Batch 2 次要（PR-1 heatmap + per_subject_mi + ppt + pr4a daynight + PR-6 supp）|
| **D3: topic1 主文档重排（D-A）** | `docs/topic1_within_event_dynamics.md` 完整 TOC 重排，把 K=2 stability + swap geometry 升到第一序结论 | 现有 526 行主文档 + advisor §0 guardrail | 新版 ~300 行；§-anchor 迁移表挂在文档顶部 |

D2 Batch 1 必须先 commit 让用户 eyeball ⇒ D3 才能开始（D3 要引用 Batch 1 的 figure path）。

---

## §2 新 topic1 文档结构（D-A 详细 TOC）

按以下骨架重写 `docs/topic1_within_event_dynamics.md`：

### TOC

```
0. §-anchor 迁移表（旧 §X → 新 §Y，给协作者）
1. 这个 topic 只回答什么问题（保留原 §1）
2. 当前最强结论（=== 新 §2，重写 ===）
   2.1 现象层：K=2 dominant stereotyped sequence（PR-2/2.5）
   2.2 机制层：forward/reverse template swap geometry（PR-6 H2 + Step 6）
   2.3 cohort-level NULL 列表（H1 SOZ anchoring / PR-4 慢调制 / PR-7 mark dependence）
   2.4 唯一可追的同步性 exploratory 线（phase_e pre>post）
3. 证据骨架（=== 新 §3，重写顺序 ===）
   3.1 K=2 stereotyped sequence 的实证（合并旧 §3.1 + §3.1b）
   3.2 forward/reverse swap geometry 的实证（=== 新增主章，从旧 §7.10 上提 ===）
   3.3 cluster geometry 描述层（保留旧 §3.1d，缩短）
   3.4 occupancy / synchrony / SOZ 描述层（合并旧 §3.1c + §3.2–§3.4，缩短）
4. PR-by-PR status 表（=== 新增 ===，每条 PR 1 段 + archive 链接）
5. 已知风险与未解问题（保留旧 §5，加入 swap 不能升为 cohort claim 的 caveat）
6. 推荐的下一步（保留旧 §7 / §7.10 末尾的 "下一步验证" 5 条）
7. 代码与结果入口（保留旧 §8，更新到 masked path）
8. 与其他 topic 的边界（保留旧 §9）
9. 历史文档索引（瘦身，仅保留正式 archive 入口 + Topic 0 phantom rerun 索引）
```

### 新 §2 关键段落 wording 草案（advisor §0 guardrail-enforced）

> ### 2. 当前最强结论
>
> Topic 0 phantom-rank 修过后（2026-05-21 phase 0 broad re-derivation 完成，见 [docs/topic0_methodology_audits.md §3.1](topic0_methodology_audits.md) + step5a–h），本 topic 在 masked 数据上有两条互锁的结论：
>
> #### 2.1 现象层：K=2 dominant stereotyped sequence（PR-2 / PR-2.5）
>
> 间期群体事件**普遍**存在 within-event 时序刻板性，最常见的压缩是 **k=2**（Tier 0 n=30：27/30 + 2×k=4 + 1×k=6；Tier 1 n=33：30/33 + 2×k=4 + 1×k=6；Tier 2 n=40：35/40 + 2×k=4 + 2×k=5 + 1×k=6，含两个 4-ch path-D 极端 outlier）。模板在时间分块复现上 **23/30 strong + 7/30 moderate + 0/30 weak**；split-half 中位模板相关 0.899，odd/even block 0.985。masked rerun 不改变这条结论（簇内 raw τ +0.054，39/40 同向 p=1.27e-10；bias_fraction 87.9 → **92.2%** ；详见 [step5c PR-3 results](archive/topic0/lagpat_phantom_rank/step5c_pr3_results_2026-05-20.md)）。
>
> #### 2.2 机制层：forward/reverse template swap geometry（PR-6 H2 + Step 6）
>
> 在 k=2 subject 中有一个稳定的 **机制层**（不是 cohort-level）现象：少数 subject 的两个 template 之间，T_a 的 start-rank 通道在 T_b 里变成 end-rank 通道（"swap"）。三个独立 masked 测量互相印证：
> - **PR-6 H2 forward/reverse 子集 (n=8/8 on masked)** sign-test cleanly positive（fwd/rev 几何 swap）
> - **PR-6 Step 6 held-out swap_class concordance 0.69 → 0.82**（masked rerun 实质提升，n=28 like-for-like）——同一通道 swap 类型在前半 / 后半时间分块上稳定
> - **PR-6 supplementary rank displacement swap_class** strict 10→9 + candidate 8→6 (masked) 维持相似分布；Kendall τ 与 F_norm 的强负相关 ρ = −0.92 不变
>
> **重要：这是机制层证据，不是 cohort-level claim。** PR-6 plan §3.3 把 H2 swap pre-register 为 "directional mechanism sanity, not cohort claim"。PR-6 H1 (SOZ anchoring) cohort Wilcoxon **仍 NULL**（masked p=0.388 → 与 orig 一致）；node anatomy h1_eligible secondary Wilcoxon p=0.014 → 0.059 (**masked 进一步弱化**，方向保持)。所以 swap 是稳健的几何现象，但**不能**写成"间期 HFO 模板锚定 SOZ"。
>
> #### 2.3 cohort-level NULL 列表（masked 重跑后仍 null）
>
> - **PR-4** 慢调制三层全 cohort null（L1 模板混合 / L2 模板内顺序一致性 / L3 模板内相对时延结构），L3 高置信子集 (n=8) Pearson r p=0.016 → **0.547**（masked，exploratory tier 失效；不入主结论）
> - **PR-4C** 模板内部几何无稳健发作邻近调制（封板阴性）
> - **PR-5-B share** composition diagnostic 不复制 panel d（masked share post-base p=0.86，进一步弱化）
> - **PR-6 H1** SOZ anchoring cohort 仍 NULL；H3 focus_rel i/l/e 仍 NULL
> - **PR-7** mark dependence 三类 metric 全 NULL（compatible with mark-independent within tested precision，**不**等于证明独立）
>
> 唯一未在 cohort 维度 null 的：**PR-5-B 候选 A dominant template 绝对事件率 post_minus_baseline** main p=0.0004 (masked), median +65.66 ev/h Bonferroni-pass。
>
> #### 2.4 同步性线唯一 exploratory 信号
>
> Epilepsiae 区域分层 `phase_e` pre>post p=0.012, r=0.31。其余全 null。

### 新 §3.2 章草案（从旧 §7.10 + 旧 archive §8 上提合并）

> ### 3.2 forward/reverse template swap geometry 的实证（PR-6 主线）
>
> PR-6 在 2026-04-25 从 PR-6-A multi-anchor consensus pivot 到 stable template endpoint anatomical anchoring（pivot 来源 + 文献背景见 [archive plan](archive/topic1/pr6_template_anchoring/pr6_template_endpoint_anchoring_plan_2026-04-25.md)）。Pivot 后三条互锁证据：
>
> #### 3.2.1 PR-6 Step 4b node-level swap–same 几何（masked 重跑保持）
>
> ... [合并 旧 §7.10 当前状态段 + Step 4b 数字 + masked rerun n=21 → n=21 H1 NULL stays NULL] ...
>
> #### 3.2.2 PR-6 supplementary rank displacement + swap classifier dual-tier
>
> ... [合并 旧 §7.10 末尾 Step 6+supp 表的 rank_displacement 行 + variable-k swap §8 行 → masked 数字: strict 10→9, candidate 8→6, F_norm 0.800→0.789, τ −0.20→−0.24, ρ=−0.921] ...
>
> #### 3.2.3 PR-6 Step 6 held-out time validation
>
> ... [合并 旧 §7.10 末尾 Step 6 行 → masked: swap_class concordance 0.82, tier strong/moderate/weak 17/10/1/0 of 28] ...
>
> #### 3.2.4 swap × clinical SOZ set-relationship（旧 §9 supp）
>
> ... [缩短：strict ∩ informative n=5 sign p=0.5 NULL, candidate ∩ informative n=5 4/5 positive; typology 多为 S⊊E; channel-selection circular caveat 写明] ...

### §-anchor 迁移表（必须挂在新主 doc 顶部）

| 旧 § | 新 § | 备注 |
|---|---|---|
| §2 一句话结论 | §2.1–§2.4 | 重写 + 拆分 |
| §3.1 内部传播刻板性 | §3.1（合并） | 与 §3.1b 合并 |
| §3.1b 数据合同/聚类稳定性 | §3.1（合并） | 数字保留 |
| §3.1b.1 Tier 1/2 表 | §3.1（最末） | 表不动 |
| §3.1c PR-3/PR-4A occupancy | §3.4 | 降级到描述层 |
| §3.1d Cluster geometry viz | §3.3（缩短） | 跨段缩短 |
| §3.2 Identity bias 簇内 | §3.1（最末段） | 合并 |
| §3.3–§3.4 synchrony | §3.4 | 合并 |
| §4–§5 当前可信结果 + 风险 | §5 | 合并保留 |
| §6 三类读数框架 | §4 / archive | 移到 PR-by-PR 表的 PR-4 行链接 |
| §7.1–§7.4 PR-4 验收 | §4 / archive | 一段总结，详 archive |
| §7.5 优先级 | §6 推荐下一步 | 缩短 |
| §7.6–§7.9 模型层 / 子集 | archive | 主 doc 删 |
| §7.10 PR-6 endpoint anchoring | §3.2 + §4 PR-6 行 | 主要内容上提到 §3.2 |
| §7.11 PR-7 pairing | §4 PR-7 行 | 一段总结 |
| §8 代码入口 | §7 | 更新 masked path |
| §9 跨 topic 边界 | §8 | 不动 |
| §10 历史文档索引 | §9 | 瘦身，删 SUPERSEDED 链接 |
| §11 文档整理里程碑 | §9 末尾 | 加 2026-05-22 重排 entry |

---

## §3 D1：目录迁移细节

### Step 1: git mv

```bash
git mv results/interictal_propagation_vs_masked/interictal_propagation_masked \
       results/interictal_propagation_masked
```

### Step 2: 修复路径引用

grep 全 repo 找 `interictal_propagation_vs_masked/interictal_propagation_masked` 引用，替换为 `interictal_propagation_masked`：

- `AGENTS.md` Cross-PR §lagPatRank 段（可能有路径示例）
- `docs/topic0_methodology_audits.md` §3.1 + §4 + §5
- `docs/archive/topic0/lagpat_phantom_rank/*.md`（14 个 step5*.md）
- `scripts/` 下任何 hardcoded path（特别是 `_apply_masked_paths()` 函数所在的 8+ 脚本）
- `tests/` 下 masked path routing tests（已修改的 4 个 test files 在 git status）

**Acceptance**: `grep -r "interictal_propagation_vs_masked/interictal_propagation_masked" .` 返回 0 行。

### Step 3: vs_masked/ 残留处理

`results/interictal_propagation_vs_masked/` 留下：
- `figures/` (2 张诊断对比图保留)
- `figures/README.md`
- `pr2_comparison.csv`
- `pr2_comparison_summary.md`

加一个 `results/interictal_propagation_vs_masked/README.md` 说明 "此目录是 phantom vs masked 对比诊断专用；masked 数据主体已 promote 到 ../interictal_propagation_masked/"。

---

## §4 D2：figure 重画清单（F-A 分两批）

### Batch 1（优先，blocks D3）

| Figure | 旧 path | 新 path（masked） | 生成脚本 | masked 等价 invocation |
|---|---|---|---|---|
| PR-2/3 cohort 6-panel default | `results/interictal_propagation/figures/cohort_propagation_summary.png` | `results/interictal_propagation_masked/figures/cohort_propagation_summary.png` | `scripts/plot_interictal_propagation.py` | 加 `--masked-features` flag（已 implement）|
| PR-2/3 per_subject 40 张 | `results/interictal_propagation/figures/per_subject/*_propagation.png` | `results/interictal_propagation_masked/figures/per_subject/*_propagation.png` | `scripts/plot_interictal_propagation.py` | 同上 |
| PR-6 template_anchoring main 6-panel | `results/interictal_propagation/template_anchoring/figures/pr6_template_pair_geometry_main.png` | `results/interictal_propagation_masked/template_anchoring/figures/pr6_template_pair_geometry_main.png` | `scripts/plot_pr6_template_anchoring.py` | 加 `--masked-features` |
| PR-6 swap supp multiples | `.../template_anchoring/figures/pr6_supp_swap_cluster_rank_multiples.png` + `_nonstrong.png` | `.../template_anchoring/figures/...` | `scripts/plot_pr6_swap_cluster_rank_multiples.py` | 同上 |
| PR-6 rank_displacement cohort heatmap + strict markers | `results/interictal_propagation/rank_displacement/figures/*.{png,pdf}` | `results/interictal_propagation_masked/rank_displacement/figures/*` | `scripts/plot_rank_displacement.py` | 加 `--masked-features` |

**Batch 1 acceptance**:
- 所有 5 组 figure 在新 masked path 下生成
- 每个 figures 目录配套 `figures/README.md`（CLAUDE.md AGENTS.md figures 规范）
- 用户 eyeball 通过（人工 checkpoint）
- commit 后 D3 才能 start

### Batch 2（次要，与 D3 并行）

| Figure | 生成脚本 | 备注 |
|---|---|---|
| PR-1 propagation heatmap examples | `scripts/plot_interictal_propagation.py` | 默认非 masked path 可能不变 |
| per_subject_mi 40 张 MI distributions | `scripts/plot_interictal_propagation.py` | masked rerun MI distribution |
| ppt panels（旧 `figures/ppt/`） | `scripts/plot_topic1_pr4_ppt.py` | PR-4 PPT 已降级到描述层 |
| pr4a daynight | `scripts/plot_interictal_propagation.py` | 描述层 |
| PR-6 supp coreness + jaccard 小图 | `scripts/plot_pr6_template_anchoring.py` | 已含在 main 脚本 |
| PR-6 step6 figures（如已存在） | `scripts/plot_pr6_step6.py` | masked Step 6 has 28 subjects |

Batch 2 不 block D3 落地；可在 D3 写完后补。

---

## §5 D3：topic1 主文档重排执行细节

执行顺序：

1. 在 `docs/topic1_within_event_dynamics.md` 顶部插入 §0 §-anchor 迁移表
2. 重写 §2（用 §2 中草案 wording，advisor §0 guardrail 必须满足）
3. 重写 §3（按 §2 TOC 顺序）
4. 新建 §4 PR-by-PR status 表（每条 PR 1 段：当前判定 + 主数 + archive 链接，不重述阈值）
5. 缩短 §5 / §6 / §9（删 SUPERSEDED 链接，删旧 §7.6–§7.9 重复内容）
6. 更新 §7（旧 §8）的代码与结果入口到 masked path
7. 在 §9 末尾加 "2026-05-22 主文档重排" 里程碑

**禁止做**：

- 不动 archive doc（只更新主 doc）
- 不动 paper_overview / topic0 / topic4 / paper1_framework_sba（用户选 D-A only）
- 不删 archive 文件
- 不动 `.cursor/rules/` 里的 rule 文件

---

## §6 失败合同

| 失败模式 | 后果 | 缓解 |
|---|---|---|
| 把 PR-6 swap 写成 "cohort-level PASS" | 违反 CLAUDE.md §5 pre-registered tier 规则；advisor 在 spec review 阶段会 reject | §2.2 wording 必须照 spec §2 草案；不允许 paraphrase |
| 用 orig（phantom）swap_class n=10/8 而非 masked 9/6 | §2 主结论数字未审计 | §2.2 第 3 个 bullet 必须引 masked 数字 + step5f link |
| 漏掉 K=2 "27/30 + 3 multi-mode" 的少数派 | CLAUDE.md §8 plain-language flattening | §2.1 必须 enumerate 所有 stable_k values |
| §-anchor 迁移表缺漏，archive 引用断链 | 协作者读 archive 找不到对应主 doc 段 | §2 末尾的迁移表必须穷举所有旧 §X |
| Batch 1 figure 数据用了非 masked path | 论文级图含 phantom 污染 | 所有 plotting scripts 必须显式传 `--masked-features` 或 `--mask-phantom` flag |
| git mv 后老 archive 路径引用 dangling | trace-by-artifact 断链 | git mv 后立即 grep + sed 全 repo 替换，附 grep 0-line acceptance test |
| 一次性写完所有 figures 跳过 user eyeball | 违反 figure discipline | Batch 1 commit 后必须 stop, 等 user 验收 → Batch 2 + D3 |

---

## §7 工作量估算（informational）

- D1 git mv + grep/sed 路径替换：~30 分钟
- D2 Batch 1 figure 重画：~45 分钟（5 脚本 sequential，每脚本 ~10 分钟）
- D3 topic1 主文档重排：~90 分钟（重写 + cross-link 检查 + 迁移表）
- D2 Batch 2 figure 重画：~30 分钟（可与 D3 并行）

Total: ~3 小时（实际可能更长，取决于 figure 渲染时间 + masked 脚本是否需要补 flag）

---

## §8 下一步

1. **User review this spec** — 重点看 §2.1–§2.4 wording 草案 + §-anchor 迁移表
2. User approval 后由 `superpowers:writing-plans` skill 转为 implementation plan
3. Implementation plan 按 D1 → D2 Batch 1 → user eyeball → D3 + D2 Batch 2 串行

---

## §9 引用

- `docs/topic1_within_event_dynamics.md`（当前主文档）
- `docs/topic0_methodology_audits.md` §3.1 + §5
- `docs/archive/topic0/lagpat_phantom_rank/step5f_pr6_results_2026-05-21.md`（PR-6 masked rerun 主数）
- `docs/archive/topic1/pr6_template_anchoring/pr6_template_endpoint_anchoring_plan_2026-04-25.md` §3.3（H2 pre-registered tier）
- `docs/archive/topic1/propagation/interictal_group_event_internal_propagation.md`（PR-2/2.5 详细数）
- AGENTS.md Cross-PR §lagPatRank（masked path routing + 8+ scripts list）
- CLAUDE.md §5（re-consult contract）+ §6（function contract）+ §7（figure discipline）+ §8（first-principles plain language）
