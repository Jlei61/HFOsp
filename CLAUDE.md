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

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
