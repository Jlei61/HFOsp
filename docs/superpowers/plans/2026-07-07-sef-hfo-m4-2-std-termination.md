# M4-2 —— STD 终止 M4 有界态 · Implementation Plan (Plan-1: instrumentation → runner → classifier → P1)

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> Re-read the spec section named in each Task **right before** writing that task's body (CLAUDE.md §5/§6).

**Goal:** Make the engine able to (a) turn on the existing presynaptic E→E depression `ee_std` in the M4 dynamic
runner, (b) report the `x_dep` depression trace, (c) fire a post-offset re-trigger kick in a state-continuous way,
and (d) classify each long run into `termination_class` + `retrigger_probe` — then run the **P1 phase plane**
(`ee_std_u × ee_std_tau_ms`, Arm 0 vs Arm 1, spontaneous, gK off) up to (but the FULL multi-seed sweep NOT past)
the §7.2 go/no-go. This answers: **does STD terminate the pass-1 bounded attractor into a re-triggerable event?**

**Spec of record:** `docs/superpowers/specs/2026-07-07-sef-hfo-m4-2-std-termination-design.md` (rev2). Re-read
§5 (protocol + planes), §7.1 (two-field schema + calibration), §8 (A/B/C/D engineering split) at each Task boundary.

**Architecture:** No new dynamics. `ee_std` (presynaptic STD) + divisive pool `S_G` already compose in the engine
(spec §4). Plan-1 adds only **non-behavioral instrumentation** to `simulate_kick` (an `x_dep` trace + a second
kick window), **runner pass-through** in `run_m4_dynamic_qi.py`, and a **stateless classifier** module. Primary
protocol is **spontaneous / no-kick** (matches the pass-1 bounded state we are terminating, spec §5 LOCK); the
only kick is the post-offset `retrigger_probe` (spec §7.1).

**Tech Stack:** Python, NumPy, pytest. SNN engine at `src/snn_engine/` (tests import via
`sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))`). Runner `scripts/run_m4_dynamic_qi.py`.

## Global Constraints

- **OFF-by-default byte-parity (承重).** New `simulate_kick` params (`dump_ee_std_trace=False`,
  `t_kick2=None`, `KICK_BOOST2=0.0`, `ee_std_trace_maskE=None`) MUST leave the engine byte-identical when unset:
  no alloc, no RNG draw, no float touch. All new code sits inside `if <param>:` branches. Verified by the existing
  `test_a1c_feedback.py` T9-style spike-output fixtures + the M4 pass-1 parity test.
- **`ee_std_u=0` parity.** With `ee_std_u=0` (STD off) AND the new hooks off, output is byte-identical to today
  (the current spontaneous M4 runs). This is the primary parity gate for the runner wiring.
- **Re-bless after editing `kick_probe.py`.** `test_a1c_feedback.py::test_T8_engine_blessed` sha256-checks
  `src/snn_engine/kick_probe.py` against `results/topic4_sef_hfo/snn_heterogeneity/engine_versions.json`. Re-bless
  ONLY after the T9 output-parity fixtures still pass.
- **Spontaneous protocol is the denominator (spec §5 LOCK).** Primary runs stay `KICK_BOOST=0.0, t_kick=1e9`
  (self-ignition via q_I depletion). The re-trigger kick is the ONLY kick, via the second window. Do NOT switch
  the primary run to a triggered kick — that would terminate a different state than pass-1's.
- **gK NOT wired in Plan-1.** P1 is gK-off (`eta_K=0`, `use_gK=False`, `k_K=0`). gK (Arm 3 / P3) is a later plan,
  gated on P1 showing `rebound` (spec §5 P3 LOCK, §9 step 5).
- **Two-field schema (spec §7.1).** Every run reports `termination_class` AND `retrigger_probe` separately; never
  merge. `go(cell) = terminate_clean AND retrigger pass`.
- **Classifier: synthetic fixtures BEFORE real data (spec §7.1, avoid threshold circularity).** `classify_termination`
  thresholds/logic are unit-tested on hand-built synthetic traces first; pass-1 real instances are sanity only.
- **Naming guard (spec §2):** the STD knob is `ee_std_u` / `ee_std_tau_ms` (the `ee_std` primitive). Do NOT
  introduce a `d_EE` field. Do NOT touch the static `D_EE` structural lever.
- **STOP LINE:** Execute **Tasks 1–4 only** (instrumentation + runner + classifier + P1 orchestration + ONE timing
  cell). **Task 5 (the full P1 multi-seed sweep) MUST NOT run** until the user reviews the implementation + the
  Task-4 wall-clock budget.

---

## File Structure

- **Modify** `src/snn_engine/kick_probe.py` — add gated `x_dep` trace + second kick window (`t_kick2`,
  `KICK_BOOST2`). (Task 1)
- **Modify** `results/topic4_sef_hfo/snn_heterogeneity/engine_versions.json` — re-bless `kick_probe.py` sha. (Task 1)
- **Modify** `scripts/run_m4_dynamic_qi.py` — `run_arm` pass-through of `ee_std_u`/`ee_std_tau_ms`; wire trace +
  retrigger schedule; P1 sweep orchestration (`--p1`). (Tasks 2, 4)
- **Create** `src/sef_hfo_m4_termination.py` — stateless `classify_termination()` + `retrigger_verdict()`. (Task 3)
- **Create** `tests/test_m4_2_termination.py` — engine parity + synthetic fixtures + real-instance sanity. (Tasks 1–3)

---

## Task 1: Engine instrumentation — `x_dep` trace + second kick window (gated, byte-parity) + re-bless

**Spec:** §8B (instrumentation), §5 (diagnostic readout), §7.1 (retrigger). **Files:** `kick_probe.py`
(signature `:91`; state init `:186`; kick block `:258-264`; recovery `:259`; low-pass `:267`), `engine_versions.json`,
`tests/test_m4_2_termination.py`.

- [ ] **Step 1: Add params to `simulate_kick` signature (all default = off).**

```python
def simulate_kick(p, net, KICK_BOOST, slow=None, ..., ee_std_u=0.0, ee_std_tau_ms=0.0,
                  dump_ee_std_trace=False, ee_std_trace_maskE=None,   # NEW (Task 1)
                  t_kick2=None, KICK_BOOST2=0.0,                        # NEW (Task 1)
                  ...):
```

- [ ] **Step 2: Second kick window (spontaneous-safe).** In the kick block (`~:260`), after the existing
  `if tk <= tm < tk + DUR_KICK:` window, add a second window reusing the SAME `kick_mask` (same source core):

```python
        if t_kick2 is not None and t_kick2 <= tm < t_kick2 + DUR_KICK:
            nu_vec[kick_mask] += KICK_BOOST2
```
`t_kick2=None` (default) → branch skipped → no RNG/float change → byte-parity. The `nu_vec` poisson draw already
happens every step; adding to `nu_vec` draws no extra RNG.

- [ ] **Step 3: `x_dep` trace recorder (gated).** Near the recorders (`~:215`), allocate only when asked:

```python
        rec_ee = dump_ee_std_trace and ee_std_on
        if rec_ee:
            xdep_mean = np.zeros(nsteps); xdep_min = np.zeros(nsteps)
            xdep_mask_mean = np.zeros(nsteps) if ee_std_trace_maskE is not None else None
```
Inside the loop, AFTER the `x_dep` recovery/deplete are applied for the step (so the trace reflects post-update
availability), record:

```python
        if rec_ee:
            xdep_mean[t] = x_dep.mean(); xdep_min[t] = x_dep.min()
            if xdep_mask_mean is not None:
                xdep_mask_mean[t] = x_dep[ee_std_trace_maskE].mean()
```
Add `xdep_mean`/`xdep_min`/`xdep_mask_mean` to the returned dict (only when `rec_ee`). When `dump_ee_std_trace=False`
or `ee_std_u=0` → nothing allocated/written → parity.

- [ ] **Step 4: Parity test — expect default path byte-identical.** Add to `tests/test_m4_2_termination.py` a test
  that runs `simulate_kick` twice (once as today, once with the new params at their defaults) and asserts
  `E_spk_bool`, `rate_E`, and final RNG state are byte-identical. Also assert `ee_std_u>0, dump_ee_std_trace=False`
  path is unaffected by the trace param.

Run: `python -m pytest tests/test_m4_2_termination.py -q -k parity` and `python -m pytest tests/test_a1c_feedback.py -q`
Expected: parity PASS; `test_T9_*` PASS; **`test_T8_engine_blessed` FAILS** (source hash changed — expected).

- [ ] **Step 5: Re-bless `kick_probe.py`** (only after Step 4 parity PASS).

```bash
cd /home/honglab/leijiaxin/HFOsp/.worktrees/topic4-m4-divisive-sg
python - <<'PY'
import hashlib, json
kp="src/snn_engine/kick_probe.py"; ev="results/topic4_sef_hfo/snn_heterogeneity/engine_versions.json"
rec=json.load(open(ev)); rec[kp]=hashlib.sha256(open(kp,"rb").read()).hexdigest()
json.dump(rec, open(ev,"w"), indent=2); print("re-blessed:", rec[kp])
PY
```

- [ ] **Step 6: Full a1c suite — expect all PASS.** `python -m pytest tests/test_a1c_feedback.py -q`

- [ ] **Step 7: Commit.**
```bash
git add src/snn_engine/kick_probe.py results/topic4_sef_hfo/snn_heterogeneity/engine_versions.json tests/test_m4_2_termination.py
git commit -m "feat(m4-2): gated x_dep trace + second-kick window in simulate_kick (byte-parity) + re-bless"
```

---

## Task 2: Runner wiring — `ee_std` pass-through + trace + retrigger schedule in `run_m4_dynamic_qi.py`

**Spec:** §8C. **Files:** `scripts/run_m4_dynamic_qi.py` (`run_arm` `:177`, `simulate_kick` call `:189`).

- [ ] **Step 1: `run_arm` gains STD + trace + retrigger args.** Extend the signature with
  `ee_std_u=0.0, ee_std_tau_ms=0.0, retrigger=None` (retrigger = `dict(t_kick2, KICK_BOOST2)` or None). Pass them
  through to `simulate_kick` alongside the existing spontaneous `KICK_BOOST=0.0, t_kick=1e9`. Build the axis/active
  E mask (reuse the E1146 source→axis geometry already in `S`) and pass as `ee_std_trace_maskE`, with
  `dump_ee_std_trace=True`. **Do NOT touch `use_gK`/`k_K`** (stay off for P1).

- [ ] **Step 2: Record the trace + retrigger outputs.** Add `xdep_mean`/`xdep_min`/`xdep_mask_mean` and (if a
  retrigger was scheduled) the post-`t_kick2` activity window to the per-run result dict, so Task 3 can classify.

- [ ] **Step 3: Parity — `ee_std_u=0` reproduces current runs.** Add a runner-level test (or a manual check
  logged in the PR) that `run_arm(..., ee_std_u=0.0, retrigger=None)` gives the same `verdict`/`rate` as the
  current spontaneous run for one confirmed-bounded cell. This guards the pass-through.

- [ ] **Step 4: Commit.**
```bash
git add scripts/run_m4_dynamic_qi.py
git commit -m "feat(m4-2): run_arm ee_std pass-through + x_dep trace + retrigger schedule (P1 wiring; gK still off)"
```

---

## Task 3: `classify_termination` two-field classifier + synthetic fixtures + real sanity

**Spec:** §7.1 (承重: two fields, synthetic fixtures BEFORE real data). **Files:** create
`src/sef_hfo_m4_termination.py`, tests in `tests/test_m4_2_termination.py`.

- [ ] **Step 1 (TDD): write synthetic-fixture tests FIRST.** In `tests/test_m4_2_termination.py`, hand-build
  activity traces (no simulation) and assert the classifier labels them:
  - synthetic **plateau → sharp offset → quiet tail** → `terminate_clean`
  - synthetic **monotone decay** → `fade` (NOT terminate_clean)
  - synthetic **intermittent stutter bursts** → `fragment`
  - synthetic **monotone-saturating high plateau, no offset** → `persist`
  - synthetic **near-zero throughout after onset** → `suppress`
  - synthetic **offset then spontaneous re-ignition** → `rebound`
  - `retrigger_verdict`: quiet tail + post-kick re-ignition → `pass`; quiet tail + post-kick fizzle → `fail`;
    `termination_class != terminate_clean` → `not_run`.

- [ ] **Step 2: implement `classify_termination(af_or_rate, bin_ms, ...) -> termination_class` and
  `retrigger_verdict(post_kick_window, ...) -> {pass,fail,not_run}`** to make Step 1 pass. Judge on: onset,
  peak/plateau, offset slope (sharp vs monotone), post-offset tail level, stutter/intermittency, saturation.
  Keep it stateless / trace-in → label-out. NO thresholds tuned against real sim traces here.

- [ ] **Step 3: real-instance sanity (NOT a threshold source).** Load pass-1 real runs from
  `results/topic4_m4_dynamic_multiseed/` (or re-run 1 cell): assert Arm 0 (pool only) → `persist`, and a known
  pass-1 runaway → NOT `terminate_clean`. If a real instance is mislabeled, fix the *synthetic fixture* to cover
  that shape and re-tune on the fixture — never tune directly on the real trace.

- [ ] **Step 4: lock thresholds into the spec §7.1 calibration table** (write the final numeric thresholds back
  into the spec).

- [ ] **Step 5: Commit.**
```bash
git add src/sef_hfo_m4_termination.py tests/test_m4_2_termination.py docs/superpowers/specs/2026-07-07-sef-hfo-m4-2-std-termination-design.md
git commit -m "feat(m4-2): classify_termination two-field classifier + synthetic fixtures; lock §7.1 thresholds"
```

---

## Task 4: P1 orchestration + ONE timing cell (import-safe, `--confirm-run` gated) — STOP before full sweep

**Spec:** §5 (P1), §7.2 (go/no-go), §9 (ordering + compute). **Files:** `scripts/run_m4_dynamic_qi.py` (add `--p1`).

- [ ] **Step 1: Pin the P1 anchor + grids (re-read pass-1 verdict first).** Anchor at the pass-1 confirmed-bounded
  operating point: `k_q=0.10`, `alpha_G` = the confirmed-bounded value(s) (read from
  `results/topic4_m4_dynamic_multiseed/` / pass-1 phase-diagram verdict — the aG16 bounded strip). Starter grids
  (refine after timing): `ee_std_u ∈ {0.05, 0.1, 0.2, 0.35, 0.5}`, `ee_std_tau_ms ∈ {200, 500, 1000, 2000, 4000}`.
  Write the pinned values into the spec §5.

- [ ] **Step 2: `--p1` orchestration (import-safe, `main()` gated on `--confirm-run`).** For each cell: Arm 0
  (`ee_std_u=0`) and Arm 1 (`ee_std_u>0`), spontaneous, gK off, `T=15000`, with a retrigger scheduled after a
  recovery window `t_kick2 = offset + few×max(ee_std_tau_ms, tau_q)` (offset detected from the first pass, or a
  fixed conservative `t_kick2` per spec §8B option (a)). Emit per-cell `termination_class` + `retrigger_probe` +
  `(⟨x_dep⟩, ⟨q_I⟩)` diagnostic. OMP=1, fork-COW workers like the existing `--sweep`.

- [ ] **Step 3: Run ONE timing cell.** One Arm-1 cell at `T=15000` + retrigger on the anchor point; measure
  wall-clock. **Write the measured per-cell and projected P1-total wall-clock back into spec §9.** This is the
  compute budget the user approves before the full sweep.

- [ ] **Step 4: STOP.** Report to the user: (a) timing cell result (does that one cell terminate? classes?),
  (b) P1 wall-clock budget, (c) confirm grids. **DO NOT run the full P1 multi-seed sweep** until the user approves.

- [ ] **Step 5: Commit the orchestration (not results).**
```bash
git add scripts/run_m4_dynamic_qi.py docs/superpowers/specs/2026-07-07-sef-hfo-m4-2-std-termination-design.md
git commit -m "feat(m4-2): P1 orchestration (--p1, import-safe, --confirm-run gated) + timing cell; STOP before sweep"
```

---

## Task 5 (SCIENCE, DO NOT RUN until user reviews Tasks 1–4 + budget): full P1 phase plane

**Gated on user go-ahead.** Run the full `ee_std_u × ee_std_tau_ms` P1 grid, multi-seed (≥4), Arm 0 vs Arm 1,
spontaneous + retrigger, `T=15000`. Compute §7.2 go/no-go (`go(cell)=terminate_clean AND retrigger pass`;
`go(plane)`= connected go-area in Arm 1 not Arm 0). Emit `(⟨x_dep⟩,⟨q_I⟩)` diagnostic per cell + a P1 phase-diagram
figure + `figures/README.md` (中文, per AGENTS.md results standard). Report the verdict (go / clean no-go) with the
spec §10 framing lock ("actual M4-2 SIMULATION"; clean no-go → points to `D_EE`/substrate).

---

## Verification Summary (per CLAUDE.md §4)

| Task | Success check |
| --- | --- |
| 1 | default-path + `ee_std_u=0` byte-parity PASS; a1c suite PASS after re-bless |
| 2 | `run_arm(ee_std_u=0)` reproduces current run; trace + retrigger outputs present |
| 3 | all synthetic-fixture labels correct; pass-1 Arm 0 → `persist`, runaway → not `terminate_clean` |
| 4 | one timing cell runs; wall-clock written to spec §9; STOP (no full sweep) |
| 5 (gated) | P1 §7.2 go/no-go verdict + figure + README |
