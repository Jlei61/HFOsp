# Axis A — boundary-coupling relay: can an engineered condition turn local E/I nucleation into an outward-relaying template? (cm-SNN)

**Date:** 2026-06-15
**Topic:** Topic 4 SEF-HFO observation layer → nucleation-vs-propagation-relay
**Status:** DRAFT — pending user GO (user review 2026-06-15 P0/P1 incorporated)
**Origin:** axis-A both arms came back screen-NULL (`docs/archive/topic4/sef_hfo/axisA_fingerprint_execution_2026-06-15.md`); the reframe = **local ignition ≠ outward propagation-relay**. This spec tests whether an engineered **relay condition** rescues the local E/I lesion, with the primary mechanism = **boundary coupling**, under a control design that isolates the E/I contribution from "the condition is just V_th↓ in disguise / a general relay enabler".

---

## §0 Question, framing, discipline

**测了什么(一句话)** — 上一轮发现:把核内抑制刹车调弱 / 自激加粗(局部 E/I 病灶),核**能自发点火但事件困在核内、传不出去**;而"降阈值"(`V_th↓`)能让核形成一个向外接力的 pulse。这一轮问:**给核加一个"帮波跨出核边界"的工程条件(主:boundary coupling = 加粗"核内源→核外边界靶"的 E→E 连接),能不能把局部 E/I 的成核救成一个安静兼容、向外接力的可读方向模板;并且 E/I 是真的参与了 relay,还是这个条件对任意成核源都通用(那 E/I 只是成核源)。**

**The scientific question this design must answer:** does E/I **only nucleate**, or does it **also participate in forming a propagable pulse**? A positive `E/I+boundary` relays only answers "boundary coupling + *some* source can relay" — to claim the E/I contributes, the **matched no-E/I source** control (§2/§3) must FAIL to relay under the same boundary condition.

**Tier discipline (this is a CONDITION-FINDING SCREEN, not a verdict):**

| Layer | Tier |
|---|---|
| Does any engineered relay condition let local E/I nucleation produce a quiet-compatible directional template? | **primary screen** (gate-only, pilot-first) |
| Is the relay E/I-specific (vs a general relay enabler) — decided by the matched no-E/I source control | **the load-bearing control**; the positive claim is conditional on it |
| Mechanistic truth about patient HFO generators | **out of scope** |

**Positive-result wording cap (LOCKED):** the strongest allowed positive claim is **"a boundary relay condition can rescue local E/I nucleation into a quiet-compatible directional template under tested settings"** — and only **"the rescue is E/I-specific"** if the matched no-E/I source does NOT relay. NEVER "E/I genuinely contributes / E/I mechanism solved / E/I relays" without that control being negative. Screen-level (pilot-first, single-seed gate → 3-seed confirm on survivors only); same read-out / tested-settings framing as the rest of axis A.

---

## §1 Engine extension (guarded) + Step 0 boundary-shell lock

The existing `build_connectivity_rot` (src/snn_engine, post-migration) already carries `local_scale_EI` (w_EI↓ target-in-core) and `w_EE_gain_core`+`core_mask_E` (w_EE↑ **both**-in-core). The relay screen showed both-in-core w_EE↑ and larger core do NOT rescue E/I. The genuinely new lever is **boundary coupling**: strengthen the E→E edges that carry core activity OUT across the core boundary.

**New optional field:** `w_EE_gain_boundary` (default 1.0) + a `shell_mask_E` (length-NE E-local bool).

**Step 0 indexing contract (LOCKED — engine loop has `i = TARGET`, `cols = SOURCE`):**
- **Boundary shell (LOCKED def):** `target E in shell ⇔ core_r < dist(focus, target) ≤ core_r + boundary_width` (pilot default `boundary_width = core_r` ≈ 1.5; tunable via `--boundary-width`).
- **Scale ONLY** the E→E AMPA edge where **source ∈ core** (`core_mask_E[cols]`) **AND target ∈ shell** (`shell_mask_E[i]`): multiply that edge's w_EE by `w_EE_gain_boundary`.
- **Untouched (must verify by TDD):** core→core (both-in-core, the existing field's domain), core→far-outside (target out-core but beyond the shell), shell-source→shell-target, outside→shell, everything else. This is **EDGE-indexed, source-in-core ∧ target-in-shell** — NOT source-only (source-only would heat all outside tissue and break bare-sheet-quiet, the exact failure the relay screen already saw).
- Applied on the per-edge weight array AFTER partner sampling (rng draw order unchanged) → `w_EE_gain_boundary=1.0` (or `shell_mask_E=None`) is **BIT-IDENTICAL** to the current engine.

**TDD (≥6, mirrors the existing 5; the shell discriminator is load-bearing):**
1. default (`gain=1.0`) bit-identical to pre-edit (vs a snapshot, as before).
2. core→shell E→E edges scaled by `gain`.
3. core→core E→E edges UNTOUCHED (not double-scaled with the existing both-in-core field; the two fields are independent).
4. core→far-outside (target out-core, beyond shell) UNTOUCHED.
5. shell-source→shell-target AND outside→shell UNTOUCHED (source must be in core; this is the source-only-not-generalized discriminator).
6. no-E/I boundary-only stays quiet (pilot artifact-gate: boundary coupling alone, no nucleation lesion → bare sheet quiet, ~0 events).

**Re-bless** `engine_versions.json` (src/snn_engine paths) with a logged old→new sha diff, only `connectivity_rot.py` changing.

---

## §2 The screen (gated, pilot-first, 1 seed, T=2000, gate-only)

### Phase A0 — calibrate the matched no-E/I source (prerequisite for the controls)

The control that isolates the E/I contribution needs a **non-E/I nucleation source matched to the E/I lesion**: same nucleation rate (E/I ei=0.5 ≈ 8 events at T=2000), and LOCAL (does NOT relay on its own). Construct it as a **rate-matched weak `V_th` core**: scan a few weak `V_th` cores (e.g. core_mean ∈ {17.8, 17.6, 17.4}, narrow std) for one whose nucleation rate is **within 0.8–1.25× of E/I's** (the same band as A3) AND that stays local (≈0 clean directional templates), like the E/I lesion.

**Honest fork (record whichever happens):** if NO weak `V_th` core is "local non-relaying" — i.e. any `V_th` core that nucleates also relays — then `V_th` and E/I nucleation are **qualitatively different** (V_th relays whenever it fires; E/I never does). In that case (a) report that dichotomy (itself a clean finding), and (b) fall back to the user's option 3 — an **artificial source matched to E/I's nucleation count** (periodic in-core seed events at the E/I rate, no threshold/EI change) — as the matched no-E/I source. The matched source's REQUIREMENT is fixed (non-E/I, rate-matched, local-non-relaying); its construction is whichever of the two satisfies it.

### Phase A — boundary coupling (primary), the 3-way control set

| condition | what it is | role |
|---|---|---|
| `E/I` (ei=0.5) | local E/I lesion, no boundary | ref (doesn't relay) |
| `matched-source` | the Phase-A0 no-E/I local source, no boundary | ref (doesn't relay; confirms it's matched/local) |
| **`E/I + boundary`** (gain ∈ {1.5, 2.0, 3.0}) | the TEST | does boundary rescue E/I? |
| **`matched-source + boundary`** (same gains) | the KEY control | does boundary rescue a *non-E/I* matched source? |
| `boundary-only` (no lesion) | artifact control | must stay quiet, ~0 events (boundary mustn't self-ignite) |
| `V_th↓ + boundary` | sanity | boundary must NOT break V_th↓'s relay; **fingerprint MAY change — that's a result, not a failure** |

### Phase B — stronger V_th seed (a SECOND matched-source axis, not a curiosity)

`E/I + seed{17.0, 16.5}` each PAIRED with its **`seed-alone` (no E/I)** control. Here the seed-alone IS the matched no-E/I source for that condition: if seed-alone relays → the seed is V_th↓ in disguise and E/I is irrelevant; if only E/I+seed relays → E/I-specific. Explicitly a matched-source control, run as a pair.

### Phase C — larger core + boundary (secondary)

Larger core ALONE already failed (relay screen). Test only `E/I + larger-core + boundary` (does enlarging the core help the boundary lever), with the matched-source + larger-core + boundary control.

**Survivors (GO per §3) → 3-seed confirm.** No survivor → boundary coupling does not rescue E/I under tested settings (firms nucleation≠relay further); STOP (no tau/sigma sea).

---

## §3 Acceptance gate (QUANTIFIED) + interpretation matrix

**GO gate per condition (all numeric — no natural-language "covers axis"):**
- clean directional events **≥ 6**;
- per-clean-event `axis_err` **< 25°**;
- per-clean-event `n_part` **≥ 7**;
- **axial-span diagnostic**: median over clean events of the participating-contact span projected onto the LOCKED geometry axis **≥ 8 mm** (≈ ≥3 contacts at 4 mm pitch; equivalently covers non-adjacent A-shaft contacts) — this is what "covers the axis, not just barely 7" means, made numeric;
- **bare-sheet quiet PRESERVED** (`true_inter_event_floor < 0.001`).

A condition that "relays" while **breaking quiet** is a FAIL (background contamination), not a survivor.

**Interpretation matrix (X = a relay condition, e.g. boundary or seed):**

| `E/I+X` | `X-only` quiet control | `X + matched no-E/I source` | 结论 |
|---|---|---|---|
| yes | quiet | **no** | **E/I + X is a SPECIFIC rescue** — X rescues E/I and not a matched non-E/I source → E/I genuinely participates in relay (the interesting result) |
| yes | quiet | yes | X is a **general relay condition**; E/I only supplies nucleation (X rescues any matched source) |
| no | any | any | X does **not** rescue E/I |
| yes | **noisy / self-ignites** | any | **FAIL** — X contaminates the background (broke bare-sheet-quiet); not a valid rescue |

**Wording rule from the matrix (LOCKED, replaces the earlier loose phrasing):** "X rescues E/I" may be written **only if** the matched no-E/I source does NOT relay under X; otherwise write "X is a general relay condition and E/I only supplies nucleation". `V_th↓ + boundary` sanity passes if it stays quiet + directional-readable; a changed `pathway_width` / event size there is a reported result, not a failure.

---

## §4 Reuse (no re-invention)

- Engine: extend `build_connectivity_rot` (src/snn_engine) — the boundary field sits beside the existing `local_scale_EI` / `w_EE_gain_core` fields; same per-edge-after-sampling pattern; same guard/re-bless flow as the A2 edit.
- Runner `scripts/run_sef_hfo_snn_cm_spontaneous_readout.py`: add a boundary-shell lesion option (`--boundary-gain`, `--boundary-width`) + the control modes (matched-source, boundary-only-no-lesion); thread `w_EE_gain_boundary` + `shell_mask_E` into the connectivity build (NOT simulate_kick).
- Read-out + frozen fingerprint instrument `src/sef_hfo_fingerprint.py` unchanged; gate logic reuses the relay-screen orchestrator pattern (`run_sef_hfo_axisA_relay_screen.py` / `ei_param_scan.py`).
- Figures per `docs/figure_style_guide.md`; results-dir + `figures/README.md` standard.

---

## §5 Code units

- **Engine (guarded):** `w_EE_gain_boundary` + `shell_mask_E` in `build_connectivity_rot`; Step-0 source-in-core ∧ target-in-shell indexing. Re-bless logged diff.
- **Lesion/geometry builder:** `shell_mask_E` from `core_r < dist(focus) ≤ core_r + boundary_width`; the matched no-E/I source (rate-matched weak V_th core, with the artificial-source fallback).
- **Runner:** boundary + control lesion modes + CLI.
- **Tests:** `tests/test_sef_hfo_axisA_boundary_engine.py` (the ≥6 TDD); plus a gate-logic unit test on synthetic readouts (axial-span diagnostic).
- **Orchestrator:** gated Phase A0→A→B→C, resume-able, gate-only → survivors → 3-seed confirm; writes `boundary_relay/relay_matrix.json` + figure.

---

## §6 Execution route (gated)

1. **Spec** (this doc) — pending user GO.
2. **Engine extension + TDD + re-bless** (guarded, logged) — after GO.
3. **Phase A0** calibrate the matched no-E/I source (pilot); record the local-non-relaying V_th core OR the V_th-always-relays dichotomy + artificial-source fallback.
4. **Phase A** (boundary 3-way controls, 1 seed gate) → read the interpretation matrix.
5. **Phase B/C** only if Phase A is interesting or inconclusive.
6. **3-seed confirm** survivors only; archive + figure; topic4 framework pointer (screen altitude).

---

（内部归档代号：axis A boundary-relay；新引擎场 `w_EE_gain_boundary`+`shell_mask_E`(source-in-core ∧ target-in-shell,shell = core_r<dist≤core_r+boundary_width);matched no-E/I source = rate-matched 弱 V_th core(local-non-relaying)或 artificial matched source;3-way 矩阵 = E/I+X vs X-only quiet vs X+matched no-E/I source;gate 量化 = clean≥6 ∧ axis_err<25 ∧ n_part≥7 ∧ axial span≥8mm ∧ quiet;positive wording cap = "boundary relay condition can rescue local E/I nucleation under tested settings",E/I-specific 仅当 matched source 不 relay。screen / pilot-first,非 verdict。)
