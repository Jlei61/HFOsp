---
name: hfosp-deep-contract-verify
description: Use when implementing a function whose plan or PR doc states multi-clause invariants in prose (boundary handling like block_time_ranges, paired-subject key alignment, surrogate construction details, stub semantics, "reported alongside" secondary metrics, "re-use don't re-invent" helpers). Forces explicit contract enumeration via TodoWrite before code is written. Catches signature-vs-prose mismatches and missing-helper situations that agents otherwise silently work around.
---

# HFOsp Deep Contract Verify

## Overview

CLAUDE.md §6 lists 5 contract-clause violation patterns this codebase has paid real cost for. Reading §6 once is necessary but not sufficient — under implementation pressure, agents still silently patch contracts instead of surfacing conflicts.

**This skill is the ritual that makes §6 mechanical**: enumerate every contract clause as a TodoWrite item before writing the function body, then verify each by name in the implementation.

**Core insight**: signatures only hint at the contract. The contract lives in prose — boundary parameters, paired-test rules, surrogate construction details, stub semantics. Each clause is a separate invariant the implementation must hold.

## When to use

- Implementing any function whose plan / PR doc has prose specifying invariants beyond the signature
- New estimator, surrogate constructor, paired-cohort test, or boundary-aware helper
- Any function that touches `event_abs_times` / per-event labels / per-subject dicts / per-block analysis
- Any function the plan flags as stub, "conditional follow-up", "not in main TDD"

Don't use for: pure refactors, trivial wrappers, plotting glue. §6-style cases only.

## Mandatory ritual (4 steps via TodoWrite)

**Before writing the function body**, create one TodoWrite item per contract clause. Use this checklist:

1. **Re-read the plan section twice.** Once for the signature, once for the prose. Open the archive doc (`docs/archive/<topic>/<plan>_<date>.md`) — do NOT work from memory of "what the plan said." Memory flattens distinctions.

2. **Enumerate contract clauses.** For each item below, either confirm it does not apply, or add it as a TodoWrite item:

   - **Boundary clause** — does this function or any helper it calls walk `event_abs_times` / per-event labels? If yes: `block_time_ranges` must be a required parameter (no `=None` default that reverts to "everything is one block"). Walk every helper called inside.
   - **Paired-cohort clause** — does any step compare two `dict[subject_id → value]`? Enforce `set(d_a.keys()) == set(d_b.keys())`, raise on mismatch, align by `sorted(keys)` — never by `list(d.values())`.
   - **Stub clause** — any helper the plan marks "conditional follow-up", "placeholder", "not in main TDD"? Must `raise NotImplementedError` with a one-line plan reference. Never return a plausible-looking value (no "best-effort" shuffle, no "approximate" anything).
   - **Signature vs prose conflict** — if the plan's signature is missing a parameter the prose requires, STOP. Surface the conflict in your message to the user. Do NOT silently correct it — the plan needs reconciliation, not a patch.
   - **"Re-use, don't re-invent" clause** — does the plan name an existing helper? `grep` for it. If it doesn't exist where the plan says it does, STOP and ask. Do NOT inline a replacement. A missing named helper is a planning bug, not your problem to silently fix.
   - **"Reported but not main" clause** — any field listed as "reported alongside" / "directional alongside symmetric" / "secondary"? It is a first-pass requirement. Implement it now. Do not defer.
   - **Surrogate / null construction clause** — any clause specifying window size, overlap, "first-covering rule", "per-block independent", "preserve first event time", "labels travel with events"? Each sub-clause is part of the contract; write a TDD test per clause, or at minimum a comment block enumerating them in the implementation.

3. **Implement.** Address each TodoWrite item by name in the code (comment referencing the clause). When violating a clause is unavoidable (e.g., the named helper genuinely doesn't exist), STOP, surface, ask — do not work around silently.

4. **Verify.** Walk the TodoWrite list and for each item point to the line of code that honors it. Then mark it complete.

## Red flags — STOP and surface, do NOT silently fix

| Thought | Reality |
|---|---|
| "The signature doesn't have it but I'll just add it" | The plan needs reconciliation. Surface the mismatch loudly in your message — don't bury it in a "decisions" footer. |
| "The named helper doesn't exist, I'll just inline a version" | The plan said don't re-invent. Stop. Ask whether to land the helper first or update the plan. |
| "This stub can return a sensible default while I work on the rest" | No. `raise NotImplementedError`. Silent contamination of downstream results is worse than a loud failure. |
| "I'll defer the directional metric to step N" | If the plan calls it "reported alongside", it's first-pass. Implement it now. |
| "Subject keys probably match — I'll align by index" | Probably is not certainty. Assert key-set equality and raise on mismatch. Index alignment silently swaps subject A's value against subject B's. |
| "`block_time_ranges=None` is fine as default for now" | No. That silently restores the bug §6 was written to prevent. Required parameter; `ValueError` on `None`. |
| "I'll write the function first and check the contract after" | Inverted ritual. The TodoWrite items must exist before the function body. After-the-fact verification misses what you forgot to ask. |

## Rationalizations the baseline agent used (and why they fail)

Captured from a baseline RED test on a synthetic PR-T4-2 transition-odds estimator:

- *"The plan said re-use `_iter_consecutive_pairs` but grep didn't find it. I'll inline a faithful version."* → Inlining is reinventing. Stop and surface that the named helper is missing; let the user decide whether to land it first.
- *"The signature is missing `block_time_ranges`. I'll add it and note this as assumption #1 at the bottom."* → Burying the conflict in a footnote is how plans drift. Surface the mismatch at the top of the response, before the code.
- *"I followed the rules and listed my assumptions clearly."* → Following the rules ≠ stopping when the rules cannot be followed. The discipline is the stop, not the workaround.

## Reference

- `CLAUDE.md` §6 — source-of-truth narrative with the 5 original failure patterns.
- This skill is the ritual; CLAUDE.md §6 is the why.
- Related: `superpowers:verification-before-completion` (final-step verification of the same invariants).
