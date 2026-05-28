# SEF-ITP Phase 4 Stage 1b (Burst-Envelope Observation-Unit Calibration) Plan — **v1**

> **Why Stage 1b exists**: Stage 1 (`stage1_results_2026-05-28.md`) shipped a
> *spike-level* event detector (`detect_bursts`, `BurstConfig` recalibrated to
> HR's fast-spike timescale) and explicitly flagged — as a **hard pre-Stage-2
> prerequisite, not a footnote** — that the observation unit (single spike vs
> multi-spike burst envelope) is undecided and load-bearing for propagation
> order / node participation in Stage 2-3.
>
> **User decision (2026-05-28, ratified)**: the **primary event = a node's burst
> envelope (onset of a cluster of spikes), NOT a single spike excursion**. This
> matches the real data object (we detect HFO / group-HFO, not single-neuron
> spikes; HR nodes were chosen because they *burst*). Spike-level detail becomes
> a *within-burst secondary* descriptor only.
>
> **Stage 1b is the minimal single-node calibration that ratifies that decision
> with data before Stage 2 builds the 2D grid on top of it.** Group-level event
> definition (multiple nodes' burst onsets co-occurring in a short window) is
> Stage 2/3 — NOT implemented here (depends on `grid2d.py` which does not exist).

> **For agentic workers:** drive with `superpowers:executing-plans`. Before
> writing `detect_burst_envelopes` body, the multi-clause prose contract is
> enumerated as TodoWrite items per `hfosp-deep-contract-verify` (CLAUDE.md §6).

**Goal:** add a single-node burst-envelope detector on top of the existing
spike-level `detect_bursts`, then run a calibration at the Stage 1 baseline
(I\*=1.0, r\*=0.006) that compares spike-unit vs envelope-unit event statistics
across σ ∈ {0, 0.2, 0.4, 0.6}, and archive the comparison. **Only if the
calibration passes**, lock "Stage 2 primary event = node burst envelope" into
the design spec + framework.

**Spec source:** `docs/superpowers/specs/2026-05-27-sef-itp-phase4-v1-design.md`
v0.2 §5.6 (event extraction) + Stage 1 archive ⚠️ section.

**Plan boundary:** Stage 1b ONLY (single node). No 2D grid, no group event, no
spec/framework edit until calibration passes.

---

## Empirical grounding (probe run 2026-05-28, before any code)

At baseline (I=1.0, r=0.006), T=1000, burn_in=100, spike-level `detect_bursts`:

| σ | n spike events | inter-spike gap percentiles [10,25,50,75,90] | structure |
|---|---|---|---|
| 0.0 | **0** | — | silent rest (x ∈ [−1.55, −1.34]) |
| 0.2 | 8 | [14.8, 17.4, 103, 172, 211] | intra-burst ~15, inter-burst ~100+ |
| 0.4 | 16 | [5.9, 8.1, 14, **59.7**, 166] | intra <20, top-quartile jumps >50 |
| 0.6 | 20 | [5.3, 6.7, 12, **48.8**, 126] | intra <20, valley tighter (~48) |

**Finding**: spikes cluster into bursts at baseline (bimodal gap distribution;
clear valley ~20–50 between intra-burst and inter-burst gaps). The spike-vs-
envelope distinction is therefore real at baseline, not only at the
suprathreshold I=2.0 case the Stage 1 archive cited. **`envelope_gap = 30.0`**
sits in the valley for all σ; it is *measured*, not tuned-to-taste (do NOT
sweep envelope_gap to make a regime look prettier — that is the result-tuning
trap the user flagged).

---

## Files

**Modified:**
- `src/topic4_modeling/hr_config.py` — add `envelope_gap: float = 30.0` to `BurstConfig`
- `src/topic4_modeling/hr_dynamics.py` — add `BurstEnvelope` dataclass + `detect_burst_envelopes`
- `tests/test_topic4_modeling_hr_dynamics.py` — add envelope-detector tests

**Created:**
- `scripts/run_topic4_phase4_stage1b_calibration.py` — calibration CLI
- `tests/test_topic4_modeling_stage1b_cli.py` — CLI smoke + exit-contract test
- `docs/archive/topic4/sef_itp_phase4_v1/stage1b_results_2026-05-28.md` — results archive
- `results/topic4_sef_itp/phase4_modeling/stage1b_envelope_calibration/` — outputs
  (`comparison.json`, `figures/spike_vs_envelope.png`, `figures/README.md`)

**Untouched (hard rule):** the dirty `sef_itp_phase2` files in git status are a
separate line of work — never `git add -A`; stage Stage-1b files by exact path.

---

## Contract clauses (enumerate as TodoWrite before writing the function body)

`detect_burst_envelopes(x, t, cfg)` prose contract:

1. **Re-use clause (§6.1)** — spike atoms come from existing `detect_bursts`
   (`src/topic4_modeling/hr_dynamics.py:77`); do NOT reinvent spike detection.
2. **Merge-by-gap clause** — adjacent spikes with gap `(next.onset −
   prev.offset) < cfg.envelope_gap` collapse into one envelope; gap ≥
   envelope_gap starts a new envelope.
3. **Onset clause (load-bearing)** — envelope `onset` = the FIRST spike's start
   within the cluster (NOT peak, NOT centroid) — this is the propagation
   timestamp Stage 2-3 will diff between nodes.
4. **Secondary-fields clause (first-pass, not deferred)** — each envelope
   carries `offset`, `duration` (= offset − onset), `n_spikes` (count of merged
   atoms), `peak_x` (max x over [onset, offset]).
5. **Config clause** — `envelope_gap` is a documented `BurstConfig` field
   (default 30.0, calibrated to probe valley), not a magic literal.
6. **Don't-reuse clause (§6.1)** — do NOT route envelopes through
   `classify_regime`: `RegimeConfig` (excitable_max_burst=50, excitable_min_ibi=30)
   is spike-unit tuned; envelope spans routinely exceed 50 and would mislabel.
   Stage 1b reports **raw envelope statistics only** (count / mean duration /
   mean n_spikes / mean IBI). "excitable-like" is judged from raw stats.

---

## Task 1: envelope_gap config field

- [ ] Add `envelope_gap: float = 30.0` to `BurstConfig` with a docstring noting
      the probe-measured valley (intra-burst <20, inter-burst >50, gap=30 in
      between) and the "do not tune" rule.
- [ ] Existing `test_burst_thresholds_defaults` still passes (asserts the 3
      spike fields; adding a 4th field does not break it). Optionally add
      `assert c.envelope_gap == 30.0`.

## Task 2: TDD detect_burst_envelopes

- [ ] **Write failing tests** in `tests/test_topic4_modeling_hr_dynamics.py`:
  - `test_burst_envelopes_empty_on_silent` — silent trace → `[]`
  - `test_burst_envelopes_single_spike_is_one_envelope_n1` — one isolated
    spike → 1 envelope, `n_spikes == 1`, onset = spike start
  - `test_burst_envelopes_merges_close_spikes` — two spikes with gap <
    envelope_gap → 1 envelope, `n_spikes == 2`, onset = first spike start,
    offset = second spike end
  - `test_burst_envelopes_separates_far_spikes` — two spikes with gap >
    envelope_gap → 2 envelopes
  - `test_burst_envelopes_onset_is_first_spike_start` — onset equals first
    spike's start, not peak time (construct a trace whose peak is later)
  - `test_burst_envelopes_peak_x_and_duration` — peak_x = max x over the
    envelope window; duration = offset − onset
- [ ] Run, verify FAIL (function/dataclass don't exist).
- [ ] **Implement** `BurstEnvelope` (frozen dataclass: onset, offset, n_spikes,
      peak_x; `duration` as a property) + `detect_burst_envelopes` reusing
      `detect_bursts` then merging by `cfg.envelope_gap`.
- [ ] Run, verify all envelope tests + the full existing suite GREEN.

## Task 3: Stage 1b calibration CLI

- [ ] `scripts/run_topic4_phase4_stage1b_calibration.py`:
  - baseline (I=1.0, r=0.006), σ ∈ {0, 0.2, 0.4, 0.6}, T=1000, burn_in=100,
    multiple seeds (e.g. 0–4) averaged.
  - per σ, per unit (spike / envelope): event count, mean duration, mean IBI;
    envelope also mean n_spikes_per_burst.
  - write `comparison.json` (per-σ spike-vs-envelope table + envelope_gap used +
    pass-flags) and `figures/spike_vs_envelope.png` (count + n_spikes/burst vs σ).
  - **exit code 1** if acceptance fails (σ=0 envelope count > tolerance, OR no
    envelope events at any σ>0), exit 0 otherwise.
- [ ] `tests/test_topic4_modeling_stage1b_cli.py` — `--help` smoke + a synthetic
      exit-contract test (mirror Stage 1 CLI test pattern).
- [ ] **Run** `python scripts/run_topic4_phase4_stage1b_calibration.py`.

## Task 4: Acceptance check (no tuning)

- [ ] σ=0 → 0 (or ≪1) envelope events. **PASS gate.**
- [ ] σ ∈ {0.2, 0.4, 0.6} → envelope events present, sparse, brief relative to T,
      large IBI = excitable-like (judged from raw stats, no regime label).
- [ ] **σ=0.6 boundary**: at envelope_gap=30, is the envelope count small
      (excitable-like) or does it look repetitive? Report honestly; if borderline,
      surface in archive — do NOT tune envelope_gap to fix it.
- [ ] Eyeball `figures/spike_vs_envelope.png` before archiving.

## Task 5: Archive (Chinese 三段式)

- [ ] `docs/archive/topic4/sef_itp_phase4_v1/stage1b_results_2026-05-28.md` —
      invoke `hfosp-plain-language-recap`; 三段式 (测了什么/怎么测的/揭示了什么) +
      spike-vs-envelope comparison table + envelope_gap calibration rationale +
      σ=0.6 boundary note + PASS/FAIL.
- [ ] `results/.../stage1b_envelope_calibration/figures/README.md` (Chinese per
      AGENTS.md figures README spec).

## Task 6: Lock decision — ONLY if Stage 1b PASSES

- [ ] Update spec §5.6 (event extraction) + framework banner: "Stage 2 primary
      event = node burst envelope (onset of merged spike cluster); spike-level =
      within-burst secondary descriptor."
- [ ] Cross-link Stage 1b archive.

## Task 7: Verify + commit

- [ ] Full phase4 suite GREEN (`verification-before-completion`).
- [ ] Commit Stage-1b files by **exact path** (never `git add -A`; phase2 dirty
      files stay untouched).
- [ ] Present results; state Stage 2 (2D grid) needs its own plan + advisor gate
      + user figure eyeballing per the spec's staged discipline.

---

## Exit contract

- [ ] `detect_burst_envelopes` lands with all 6 contract clauses honored; tests GREEN.
- [ ] Calibration shows σ=0 → 0 envelope events, σ>0 → sparse excitable-like envelopes.
- [ ] Spike-vs-envelope comparison archived with envelope_gap calibration + 三段式.
- [ ] Decision locked into spec/framework (if PASS).
- [ ] Stage 2 handed off: per-node baseline (I\*=1.0, r\*=0.006, σ\*=0.4) + unit =
      burst envelope (onset = first spike start).

## Failure modes (pre-registered)

| failure | action |
|---|---|
| σ=0 produces envelope events | spike detector false-positives at rest — revisit `BurstConfig` spike fields, NOT envelope_gap; stop-and-review |
| σ>0 produces 0 envelopes | envelope_gap too large merged everything OR spike detector misses — report; do not silently tune |
| σ=0.6 looks repetitive at gap=30 | surface in archive as a boundary caveat for Stage 2; do NOT tune gap to hide it |
