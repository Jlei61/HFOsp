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

**Re-read upstream definitions before using a derived flag.** A JSON field name (e.g. `forward_reverse_reproduced`) encodes only a subset of the contract. The accepted definition lives in the upstream PR's archive doc — not in the field name. Look up the definition before using the field. One JSON key may capture only one of multiple alternatives the upstream rule allows (e.g. "split-half OR odd-even"). Acting on the field name alone undercounts.

**Sensitivity gates are pre-conditions for main-doc conclusions, not afterthoughts.** A primary statistic from one run is preliminary. Don't write paper-level framing into main docs until alternate metric definitions, robustness checks, and held-out validations have all run. Preliminary numbers go to archive only, tagged "preliminary, pending sensitivity". The bar to enter the topic main doc is "all sensitivity gates passed", not "the first run produced a number".

**Pre-registered hypothesis tier is fixed at planning time.** When a plan defines H2 as "directional mechanism sanity, not cohort claim", report it that way regardless of how strong the directional signal looks. Don't upgrade a hypothesis tier in results — the tier is part of the design, not the result. Phrases like "independent, publishable finding" require the hypothesis to have been pre-registered as primary; if it was registered as sanity check, it stays sanity check.

The unifying principle: every step boundary is a re-read checkpoint. Memory of "what the plan said" is not enough — open the archive doc and verify.

## 6. Implement to the Plan's Full Contract, Not the Function Signature

This rule comes from a PR-7 review (2026-04-28) that caught five
science-contract violations the function signatures alone would not surface.
"It compiles and the basic test passes" is not the bar. Read the plan
section RIGHT BEFORE writing the function body — the contract lives in
prose, not signature.

**Boundary parameters propagate through every related helper.** If the main
estimator takes `block_time_ranges` (or any "valid range" / "gap" / "boundary"
parameter), every secondary helper that walks the same time series must
take it too. A `compute_transition_odds()` that lacks `block_time_ranges`
silently treats a recording-gap of hours as a neural transition. The fix is
mechanical: scan every function in the module that consumes `event_abs_times`
or per-event labels, confirm each takes the boundary parameter and
respects it. No defaults that revert to "everything is one block".

**Paired-cohort tests must enforce subject-key match.** Code like
`np.array(list(d_10s.values()))` followed by Wilcoxon against
`np.array(list(d_30s.values()))` is broken — the two arrays may not align
by subject, so subject-A's 10s metric pairs against subject-B's 30s
metric. The fix: sort by `sorted(set(d_a.keys()))` after asserting
`set(d_a.keys()) == set(d_b.keys())`; raise on mismatch, never silently
align by index.

**Stubs must `raise NotImplementedError`.** A stub of a planned helper that
silently returns plausible values gets called in production and pollutes
results. If the plan explicitly marks a helper as "conditional follow-up,
not in main TDD", the implementation MUST raise — not return a "best-effort"
shuffle of ISIs that happens to look right on a unit test. Loud failure
beats silent contamination.

**"Reported but not main metric" requirements are first-pass requirements.**
When a plan says "report H1b direction asymmetry alongside H1 main symmetric
metric", the directional fields go into the FIRST implementation of the
core estimator — not deferred to "Step N will add it". Deferred secondary
fields get forgotten and later patched ad-hoc, producing inconsistent
schemas across per-subject JSONs.

**Surrogate construction details ARE the contract.** When the plan
specifies a null with "30 min window, 50% overlap, first-covering rule,
per-block independent", every clause is part of the science contract.
Implementing only "30 min non-overlapping windows from t_min" because
"the test passes" silently changes the null. Re-read the surrogate clause
before writing the shuffle function, and write a TDD test for each
non-trivial clause (overlap behavior, first-covering rule, block isolation).

The unifying principle: a plan's prose lists invariants the implementation
must hold simultaneously. Tests that exercise only the function's surface
behavior (does it return a number? does it permute?) miss invariants. Each
science-contract clause needs its own test that would fail if that specific
clause were violated.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, clarifying questions come before implementation rather than after mistakes, and helpers handle every plan-specified invariant on the first pass.
