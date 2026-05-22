# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## Project Context

For project-specific context (research topics, PR plans, subject configs, directory conventions, and ongoing initiatives), consult:

- @AGENTS.md — primary project handbook
- @.cursor/rules/topic1-within-event-dynamics.mdc
- @.cursor/rules/topic2-between-event-dynamics.mdc
- @.cursor/rules/topic3-spatial-soz-modulation.mdc
- @.cursor/rules/event-periodicity-pr-plan.mdc
- @.cursor/rules/interictal-propagation-pr-plan.mdc

The behavioral rules below apply on top of that context — when project conventions and these rules conflict, the project handbook wins for domain decisions; these rules win for coding style and change discipline.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Re-consult the Contract at Every Step Boundary

When advancing through a multi-step plan, the plan-of-record is the source of truth. Mental summaries decay between steps and silently flatten distinctions. At each step boundary, re-read the relevant plan sections instead of acting on memory.

**Apply safety fixes end-to-end, not just at the site that surfaced the bug.** When a robustness fix is applied to one consumer of upstream data (e.g. split-half), audit every other consumer of the same upstream artifact (e.g. full-data main analysis). The same upstream pollution feeds multiple paths. When a helper grows a new safety parameter (`valid_mask=`, `min_n_channels=`), every existing caller must pass it — `default=None` silently restores the buggy path. The test: search the codebase for every call site of the helper and verify the new parameter is set.

> **Worked example**: `lagPatRank` phantom (full chain: bug discovery → 15-consumer-script audit → parallel `_masked/` results tree → cohort reconcile → delayed default flip). See `docs/topic0_methodology_audits.md` §3.1 + §5 + AGENTS.md Cross-PR §lagPatRank. **Mechanical lesson**: when `default=False` exists for backward-compatible safety, the audit isn't done until you (1) re-run every downstream consumer with `True` to a parallel results tree, (2) sanity-gate against orig with like-for-like cohort comparison, (3) flip the default to `True` only after all flips are reconciled.

**Re-read upstream definitions before using a derived flag.** A JSON field name (e.g. `forward_reverse_reproduced`) encodes only a subset of the contract. The accepted definition lives in the upstream PR's archive doc — not in the field name. Look up the definition before using the field. One JSON key may capture only one of multiple alternatives the upstream rule allows (e.g. "split-half OR odd-even"). Acting on the field name alone undercounts.

**Sensitivity gates are pre-conditions for main-doc conclusions, not afterthoughts.** A primary statistic from one run is preliminary. Don't write paper-level framing into main docs until alternate metric definitions, robustness checks, and held-out validations have all run. Preliminary numbers go to archive only, tagged "preliminary, pending sensitivity". The bar to enter the topic main doc is "all sensitivity gates passed", not "the first run produced a number".

**Pre-registered hypothesis tier is fixed at planning time.** When a plan defines H2 as "directional mechanism sanity, not cohort claim", report it that way regardless of how strong the directional signal looks. Don't upgrade a hypothesis tier in results — the tier is part of the design, not the result. Phrases like "independent, publishable finding" require the hypothesis to have been pre-registered as primary; if it was registered as sanity check, it stays sanity check.

The unifying principle: every step boundary is a re-read checkpoint. Memory of "what the plan said" is not enough — open the archive doc and verify.

## 6. Implement to the Plan's Full Contract, Not the Function Signature

Origin: PR-7 review (2026-04-28) caught 5 science-contract violations that function signatures alone would not surface. Read the plan section RIGHT BEFORE writing the function body — the contract lives in prose, not signature. Five recurring patterns (each violation = silent science contamination):

- **Boundary parameters propagate everywhere.** Main estimator takes `block_time_ranges` → every secondary helper walking the same time series must too. `compute_transition_odds()` without `block_time_ranges` silently treats recording-gap hours as neural transition.
- **Paired-cohort tests enforce subject-key match.** `np.array(list(d.values()))` aligns by dict order, not subject. Required: `sorted(set(d_a.keys()))` after `assert set(d_a.keys()) == set(d_b.keys())`; raise on mismatch.
- **Stubs raise `NotImplementedError`.** A stub that silently returns plausible values gets called in production. "Conditional follow-up, not in main TDD" → MUST raise; loud failure beats silent contamination.
- **"Reported alongside" = first-pass requirement.** H1b direction asymmetry "alongside H1 main symmetric metric" goes into the FIRST implementation of the core estimator. Deferred secondary fields get forgotten and patched ad-hoc → inconsistent schemas across per-subject JSONs.
- **Surrogate construction details ARE the contract.** "30 min window, 50% overlap, first-covering rule, per-block independent" — every clause is science contract. Each non-trivial clause needs its own TDD test that would fail if violated.

The unifying principle: plan prose lists invariants the implementation must hold simultaneously. Surface-behavior tests (returns a number? permutes?) miss invariants.

**Skill companion (operationalizes §6)**: `.claude/skills/hfosp-deep-contract-verify/SKILL.md` — invoke before writing the body of any function whose plan specifies multi-clause invariants (boundary params / paired-cohort keys / surrogate construction / "reported alongside" secondary metrics / "re-use don't re-invent" helpers). §6 is the *why*; the skill is the *how*.

### 6.1 Helper reuse requires question-match, not signature-match

Origin: SEF-ITP Phase 1 H2 v1.0.6 → v1.0.9 (4 cascading errors, 2026-05-22). Same lesson as §6 "re-use don't re-invent" but the inverse failure mode: **re-using an existing helper because the signature fits, without checking the helper's null contract matches the new hypothesis's question.**

> **Worked example**: H1 strict layer uses `compute_h1_compactness` whose matched-null preserves **shaft distribution** (correct: H1 asks "is endpoint more compact than other 3-channel subsets on the same shaft(s)?"). When H2 needed a "swap-k node spatial compactness" check, an early implementation reused `compute_h1_compactness` because the signature fit (`members`, `candidate_pool`, `D`, `n_null`). But **H2's question is "are swap-k nodes more compact than other valid SEEG nodes irrespective of shaft" — same-shaft matching strangles H2's null (most pools end up < n_members → `INSUFFICIENT_NULL`) AND silently subtracts the shaft signal that's part of the H2 mechanism**.

The discipline:
- Before calling an existing helper from a new hypothesis, **state in one sentence what the helper's null asks**, then **state what your hypothesis asks**. If those two sentences differ on what gets controlled or what gets compared, the helper is the wrong shape — write a new one or extend with an opt-in mode.
- "It produces a verdict with the right key" is not a fit — verdict labels can be the same while the underlying question is different.
- Look at the helper's **null construction**, not its return shape. The null is the contract.

### 6.2 Source-of-truth has resolution levels — name the level before you act

Origin: SEF-ITP Phase 1 H2 input order (cohort_run §9, 2026-05-22). The same nominal concept ("swap" / "forward-reverse" / "endpoint") lives at three resolution levels in this repo, each producing a different per-subject answer:

- **Template level**: PR-2 `candidate_forward_reverse_pairs` (whole-template Spearman r) — answers "this subject has two templates that look like negatives of each other". Diluted by shared channels; not channel-level.
- **Endpoint summary level**: PR-6 anchoring fixed top-k source/sink + `h2_swap_check` Jaccard — answers "the top-k channels on each side swap roles between templates". Fixed k=3, audit-derived cohort gate.
- **Channel level**: rank-displacement `swap_sweep.decision_k` (variable k, family-wise null per subject) — answers "which channels actually contribute to the swap, and how many?".

If a downstream task needs channel-level inputs (e.g., a spatial compactness test on the swap nodes), reading PR-2's `candidate_forward_reverse_pairs` produces a syntactically valid input that is **scientifically the wrong layer**. The JSON field name doesn't carry the resolution level — the upstream archive does.

The discipline:
- Before pulling a field from an upstream JSON, write one sentence: "this field answers \<question at level X\>". Compare to "my task needs \<question at level Y\>". If X ≠ Y, you have a layer mismatch even when the names match.
- For repeated concepts that span multiple resolution levels (template / endpoint summary / channel), keep an entry in AGENTS.md "Cross-PR Contract Lookups" pinning the layer of each field. Without the lookup, "swap" silently means whichever layer the most recent commit was working on.

### 6.3 Pronoun discipline in cross-layer narration

Origin: same H2 cascade. The agent wrote things like "swap-k node spatial compactness shows H2 PASS" — collapsing two distinct findings ("which channels are labeled as swap" + "are those channels spatially clustered") into a single "H2 PASS" pronoun. Readers (and future agents) silently re-expand the pronoun to whichever layer they care about, producing over-broad claims.

The discipline:
- When a result combines two layers (label + geometric, or label + statistical), **state both layers in the sentence**, e.g. "rank-displacement swap-k label was `strict` AND swap-k source-side spatial compactness was PASS". Do not write "H2 PASS" without specifying which layer.
- The pre-registered tier (§5 last bullet) determines what kind of statement you're allowed: mechanism-sanity tier → per-subject descriptive sentences only; cohort-claim tier → cohort-level summary statistics allowed. **"H2 PASS" as a cohort statement requires both layers AND a cohort-claim tier — neither alone is enough.**
- When tightening a section's wording, ask: if the reader only sees this sentence (no surrounding context), would they re-expand "H2" to "cohort-level reversal PASS" or to "swap-k spatial sanity supported per subject"? The two readings have very different implications; the wording must disambiguate.

## 7. Multi-Panel Figure Discipline

**Each panel must answer one independent scientific question. Two panels
showing the same construct from different angles is redundancy, not
coverage.**

Before drawing a multi-panel figure (paper figure, cohort summary,
per-subject diagnostic):

- List the questions the figure must answer; assign one panel per question.
- Reject any candidate panel that collapses to the same underlying
  construct as another panel — pick the most direct representation and
  drop the duplicate.
- A joint scatter X-vs-Y is justified only when the marginals X and Y
  cannot reveal the coupling on their own; otherwise it is the same
  information drawn twice.
- Two embeddings of the same data under metrics that are rotation /
  permutation of each other are one panel, not two.
- More panels are not better. Three panels where each is independent
  beat six panels where half are reformulations.

If two panels answer the same question, the figure has redundancy and
one panel must be replaced.

## 8. 第一性原理表达：避免代号雪球

**项目越长，代号越多。代号越多，错位越深。**

本仓库已经堆了一大堆代号 / 锁定字段 / 状态档位：`PR-T4-1`, `Λ_gap`,
`stable_k=2`, `δ_excess=0.05`, `INCONCLUSIVE-locked`, `lambda_fragile`,
`forward_reverse_reproduced`, `producer_health`, `clinical_concordance` …
这些代号在写 plan / archive 时是合法且必要的（精度 + 可索引），
但**在跟用户解释 "我们测了什么 / 怎么测的 / 揭示了什么" 时是污染源**：
两个人都以为对方理解同一个代号，错位悄悄堆积。

**强制规则**：所有面向用户的解释 / 回顾 / 现状汇报，必须先用**第一性原理朴素话**
把内容讲一遍；archive 代号 / 锁定字段 / PR 编号只作为括号补注或链接，不
代替朴素描述。三段式骨架：

1. **测了什么** — 用日常物理 / 日常对象 / 日常因果讲清楚被测的现象
2. **怎么测的** — 把核心算法步骤用"如果完全随机的话，应该长这样；实测长这样"的对比说出来
3. **揭示了什么** — 不是"PASS / NULL / INCONCLUSIVE"，而是"在这个尺度上看起来像 / 不像 / 没看清"

**反例（禁止形态）**：

> "PR-7 addendum 1800s window + lag1_same_excess null-relative 干净 PASS，
> 10/30/60s + run_length_lift CI underpowered at n=6 with structural outliers"

**正例（要求形态）**：

> "我们看：相邻两次事件挑的是不是同一个模板。如果完全像抛硬币一样独立，
> 平均下来同模板比例应该等于 P(模板A)² + P(模板B)²，没有偏离这个数学预期。
> 实测在 30 分钟尺度的窗口下确实就是这个数，差距小到 0.0002 — 看起来就是
> 独立抛硬币，没有'刚才挑了 A 这次更倾向继续挑 A'的记忆效应。
> 但在 10 秒 / 30 秒 / 60 秒短窗里我们只有 6 个被试，置信区间宽到不能下结论 —
> 我们没说短窗里也是独立的，只能说 '在我们的精度内看起来兼容独立'。
> （内部归档代号：P3, PR-7 addendum, lag1_same_excess, run_length_lift, δ=0.05）"

**适用范围**：

- 用户问"现在做到哪一步了 / 你为什么这么设计 / 我们到底揭示了什么"时
- 写跨 topic 的对外 / 对协作者邮件、PPT、口头汇报底稿时
- 在 main doc 引言段、§ 章节首句、archive doc 的 abstract 段时

**不适用范围**：

- archive doc 内部正文 / TDD 列表 / per-subject JSON schema — 这些需要精度
  代号，不能朴素化
- 代码注释 — 代码引用代号是工程必要

**触发动作**：每写完一段 status / recap / explanation，自检 — 如果一个不熟悉
此项目的人（或半年后的自己）只读这一段，他能不能复述"测了什么、怎么测的、
揭示了什么"？不能 → 重写。

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, clarifying questions come before implementation rather than after mistakes, and helpers handle every plan-specified invariant on the first pass.
