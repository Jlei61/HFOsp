# Group-Event State v0.3.3 — Workstream A (Evaluator / Assay / Data Contract) Implementation Plan

> **For agentic workers:** executed inline in this session (superpowers:executing-plans style, TDD per task). Steps use checkbox syntax. Re-read the clause list of a task **before** writing that task's function bodies (CLAUDE.md §5/§6).

**Goal:** Deliver Workstream A of v0.3.3: one canonical per-anchor proper-score evaluator that the training branch, the independent evaluator and the figure payload all reuse; a row-level explanation of the v0.3.2 E1146 sign flip (+0.1277 vs −0.3291); Oracle Level 0–2 recovery on synthetic marks; D0–D4 (+ small D5) power curves on real time axes; per endpoint/horizon estimability from real coverage segments; a frozen data-boundary contract.

**Architecture:** New package `src/topic5_group_event_state/v033_evaluator/` (pure numpy/scipy core, torch only where the Level-2 encoder needs it). The canonical score is a *pure function of (target, prediction, dispersion, mask, weight)* — nothing is fitted inside it. Legacy v0.3.2 code is read-only (import only; never modified). Real-data access in this workstream is read-only (timelines, registry, v0.3.2 artefacts). No human training run is started: `V0_3_3_EXECUTION_RELEASE.json` does not exist.

**Tech Stack:** Python 3.11 (`/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python`), numpy 1.26, scipy 1.16, torch 2.5.1 (CPU), pytest 9. All CPU workers `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1`.

**Spec:** `docs/archive/topic5/group_event_state_v0_3_3_dual_view_state_spec_2026-09-02.md` (sha256 7f75dc20…) and `..._dual_view_state_plan_2026-09-02.md` (sha256 58aaf467…), Agent A handoff `..._agent_a_evaluator_assay_handoff_2026-09-02.md` (sha256 14ebf162…) — all currently untracked in `.worktrees/topic5-ges-v032-closeout/docs/archive/topic5/`. Status `V0_3_3_REVISED_DRAFT_FOR_REVIEW_DO_NOT_EXECUTE`.

## Global Constraints (verbatim from spec/handoff)

- Base commit for this worktree: `233f3ad1` (last commit of `codex/topic5-group-event-state-v032-closeout`, the v0.3.2 result-producing commit). **Assumption**: no supervisor release commit exists yet; recorded in status for supervisor confirmation.
- Without `V0_3_3_EXECUTION_RELEASE.json`: read-only audit, implementation, unit tests, synthetic smoke, resource sentinel only. **No load-bearing human runs.**
- Old v0.3.2 and Topic 4 jobs are read-only; never stop / overwrite / reuse their output keys. No `pkill -f`.
- Write scope: `/data/hfosp_group_event_state_v0_3_3/agent_a/`, `results/group_event_state/v0_3_3/evaluator_assay/`, `/data/hfosp_group_event_state_v0_3_3/shared/{evaluator_contract,eligibility}/`. Never edit Agent B training config or Agent C endpoints.
- Global hard stops: sealed partition read; time/patient/seizure/normalisation/target leakage; canonical evaluator giving two scores for one checkpoint/anchor.
- Sealed partition stays closed; only the v0.3.2 development partition (base_fit 0–60 / inner_val 60–70 / dev_val 70–80 / dev_test 80–100 on recorded time) is touched.
- Synthetic cadence: 3 replicates per code change (smoke); nightly 10; milestone 20–30. D5 only a few expected-failure runs.
- Effect size = oracle held-out deviance gain / block SNR — never a raw β, never a pass count.
- Eligibility must call the real window builder's coverage segments (`build_carry_segments` / `build_anchor_grid` / `phase_block_counts`), never session counts or sliding-window totals.
- Conclusions are reported in three separate layers: engineering consistency / assay power / human estimability. Green tests ≠ a state exists.

---

## Task 0: Status scaffolding

**Files:** Create `results/group_event_state/v0_3_3/evaluator_assay/CURRENT_HANDOFF.md`, `/data/hfosp_group_event_state_v0_3_3/agent_a/agent_a.status.json` (atomic JSON, refreshed at every milestone: commit, done, running, pending, failures, resources, next).

## Task 1: Canonical per-anchor evaluator (A1)

**Files:** Create `src/topic5_group_event_state/v033_evaluator/__init__.py`, `.../canonical.py`; Test `tests/test_group_event_state_v033_canonical.py`.

**Interfaces (Produces):**
```python
SCHEMA_VERSION = "group_event_state_v0_3_3_canonical_per_anchor_1"
TOLERANCE_NATS = 1e-6
SCHEMA_COLUMNS = ("subject","seed","checkpoint_hash","anchor_time","split","target",
                  "prediction_H","prediction_H_plus_state","dispersion","mask","weight",
                  "per_anchor_NLL_H","per_anchor_NLL_H_plus_state","eligibility","evidence_label")
def nb_nll(target, log_mu, log_r) -> np.ndarray            # float64 per row; log_r scalar or per-row
def nb_nll_torch(target, log_mu, log_r) -> torch.Tensor     # identical formula, float64
def alpha_to_log_r(alpha) -> float                          # NB2 Var = mu + alpha mu^2  <->  r = 1/alpha
def build_per_anchor_table(*, subject, seed, checkpoint_hash, split, anchor_time, target,
        prediction_H, prediction_H_plus_state, dispersion, mask, weight, eligibility, evidence_label,
        dispersion_rule="shared") -> dict[str, np.ndarray]   # dispersion: scalar (shared) or {"H":..,"H_plus_state":..} (per_arm)
def paired_gain(table, *, control="H", treated="H_plus_state", reduction="mean",
                block=None) -> dict                          # gain = NLL_control - NLL_treated; positive favours treated
def assert_tables_agree(a, b, *, tolerance=TOLERANCE_NATS) -> None   # hard-stop detector
```

**Contract clauses (deep-contract-verify ritual):**
- [C1] Score is a pure function of inputs; nothing (intercept, dispersion, ridge) is estimated inside. An intercept re-calibration is a *declared arm* (`H_plus_intercept`), never hidden.
- [C2] One NB formula for numpy and torch branches, float64 internal; agreement ≤ 1e-9 nats.
- [C3] Legacy mapping is explicit: v0.3.2 model used `log r`, v0.3.2 eval used `alpha`; `alpha_to_log_r` and tests pin both legacy functions to the canonical value.
- [C4] Anchor permutation permutes rows and leaves every reduction invariant.
- [C5] Masked rows keep NaN NLL in the table and are excluded from every reduction; the count of masked rows is reported.
- [C6] `dispersion_rule="shared"` requires a single `log_r` for all arms (raise otherwise); `"per_arm"` requires an explicit per-arm value (raise on missing arm). No default that silently picks one.
- [C7] Weights enter the reduction as a weighted mean (Σ w·gain / Σ w); default weight 1.
- [C8] Sign: gain = control − treated; positive favours treated. Reported as `direction`.
- [C9] Reductions: `mean`, `sum`, `block_mean` (mean of per-block means over `block` ids); the same rows feed every arm.
- [C10] Schema columns always present, equal length, `split`/`eligibility`/`evidence_label` are per-row strings.
- [C11] `assert_tables_agree` raises `EvaluatorDisagreement` listing the first offending row when any per-anchor NLL differs by more than `tolerance` — this is the machine form of global hard stop #3.

Steps: write failing tests for C1–C11 → run (fail: module missing) → implement → pass → commit `feat(topic5): v033 canonical per-anchor evaluator`.

## Task 2: E1146 row-level discrepancy audit (A2)

**Files:** Create `.../v033_evaluator/e1146_audit.py`, `scripts/audit_group_event_state_v033_e1146.py`; Test `tests/test_group_event_state_v033_e1146_audit.py` (on synthetic mini-artefacts); Output `results/group_event_state/v0_3_3/evaluator_assay/e1146_discrepancy_audit.json` (mirror under `/data/.../agent_a/`).

**Read-only inputs:** `/data/hfosp_group_event_state_v0_3_2/{model/runs/leaky_bank/epilepsiae_1146/seed_*/{evaluation,result}.json, model/frozen_states/..., evaluation/h1/epilepsiae_1146/h1_{result,arrays}_seed_*.{json,npz}, shared/history_baseline_registry.json + its npz}`.

**Ordered steps (each with a numeric comparison and a `diverges` flag):** checkpoint → anchor set → target → prediction_H → prediction_H_plus_state → dispersion/intercept → weight → seed aggregation → score sign/reduction. Output `first_divergence` (the first step with `diverges=True`) and, separately, `sign_flip_origin` (the step whose contribution changes the sign), plus a canonical re-score of the checkpoint's own predictions under shared and per-arm dispersion. The audit reproduces both legacy numbers (+0.12772, −0.32912) from the per-anchor rows before comparing.

Clauses: [E1] never write into v0.3.2 directories; [E2] reproduce both published numbers to 1e-6 before diagnosing; [E3] every step compares rows, not summaries; [E4] the report names the commit/feature-count difference (registry 89e55a58 / 125 features vs H1 eval 81d36b74 / 126 features) only if the row comparison shows it.

## Task 3: Data-boundary contract (A6)

**Files:** Create `.../v033_evaluator/boundaries.py`, `scripts/audit_group_event_state_v033_boundaries.py`; Test `tests/test_group_event_state_v033_boundaries.py`; Output `data_boundary_audit.json`.

**Interfaces:**
```python
def state_carry_units(sessions) -> list[CarryUnit]                       # hard reset only at recorded gap / session edge
def event_update_mask(event_times, seizures, *, postictal_seconds) -> np.ndarray   # False inside [onset, offset+postictal)
def target_window_valid(t, horizon, segments, partition) -> np.ndarray   # whole [t,t+h) inside one target segment and one phase
def anchor_carry_index(t_anchor, carry_units) -> np.ndarray
def boundary_variants() -> dict   # {"mainline": autonomous flow across seizure, "sensitivity": hard reset at seizure}
```
Clauses: [B1] target never crosses gap/split/seizure (uses `build_carry_segments` + `EvalPartition.window_within_phase`); [B2] seizure and immediate-postictal events never update state (mask False); [B3] autonomous flow continues across the excluded interval (state decays, not reset); [B4] real gap/session boundary is a hard reset (state 0 at unit start); [B5] hard reset at seizure is a named sensitivity variant, never the default; [B6] audit on the real E1146 timeline reports counts for each clause.

## Task 4: Synthetic DGPs D0–D5 on real scaffolds (A4 data side)

**Files:** Create `.../v033_evaluator/scaffold.py` (real timeline + registry H → `Scaffold`), `.../dgp.py`; Test `tests/test_group_event_state_v033_dgp.py`.

**Interfaces:**
```python
@dataclass class Scaffold: subject, t_anchor, anchor_segment, anchor_session, anchor_phase, eligible, event_times, event_segment, event_session, participation_vocab (N,C) bool, event_size K (N,), log_mu_h {300:..,1800:..}, partition, segments, carry_index...
def load_real_scaffold(subject, cfg) -> Scaffold     # read-only; uses load_eval_timeline + history_baseline_registry
def hidden_leaky_state(marks, event_times, carry_index, t_anchor, anchor_carry, last_event_pos, tau) -> np.ndarray
def generate(scaffold, kind: "D0".."D5", *, beta_count, beta_grammar, seed) -> SyntheticData  # counts per horizon, participation subsets, truth z_N, z_G, marks (visible for Level 2 except D5)
def conditional_bernoulli_logpmf(logits, subset, K) -> float; def sample_conditional_bernoulli(rng, logits, K)
```
Clauses: [D-1] real anchors/coverage/split/event times are never altered; only targets (and the visible mark channel) are synthetic; [D-2] hidden state drives only its declared view (D1 count, D2 grammar, D3 both with one z, D4 both with independent z); [D-3] D5 = state exists but the mark channel is invisible (background-only); [D-4] grammar subsets are sampled *conditional on the real size K*; [D-5] synthetic count NB draws use the registry `log_mu_H` as base rate (so H stays correct under D0); [D-6] generator and noise seeds separate and recorded.

## Task 5: Oracle Level 0–2 estimators (A3)

**Files:** Create `.../v033_evaluator/oracle.py`; Test `tests/test_group_event_state_v033_oracle.py`.

Level 0 (true state, fit head only) / Level 1 (true innovation, fixed leaky scan {300,1800,7200} s, fit readout) / Level 2 (visible mark columns, train encoder+readout: count view via the v0.3.2 leaky-bank trainer on CPU; grammar view via a minimal torch encoder→bank→conditional-Bernoulli head). Each level returns truth, prediction, held-out continuous gain (canonical evaluator), false-positive readout under D0 and a `failure_location` string (`head` / `scan_alignment` / `encoder_optimizer` / `none`).

Clauses: [O1] heads are fitted on TRAIN (base_fit) rows only and scored on development rows only; [O2] the H arm gets the same TRAIN-only recalibration as H+S (intercept + dispersion) so gain is not an intercept artefact; [O3] scoring goes through `build_per_anchor_table` + `paired_gain` — no private NLL; [O4] Level 1 must reproduce Level 0 within tolerance when τ matches the DGP (alignment check); [O5] failure location is derived from which level first loses the Level-0 gain.

## Task 6: Power curves and smoke runner (A4)

**Files:** Create `.../v033_evaluator/power.py`, `scripts/run_group_event_state_v033_assay.py`; Output `oracle_level_0_2.json`, `d0_d4_power_curve.json` (+ per-replicate JSON under `/data/.../agent_a/assay/`).

Clauses: [P1] effect axis = Level-0 oracle held-out deviance gain (nats/anchor) and block SNR (gain × √n_blocks / block sd); [P2] power = fraction of replicates whose estimator gain CI (block bootstrap, blocks = non-overlapping horizon bins inside a segment) excludes 0 below; false positive = same rule on D0; [P3] cadence 3 replicates now, fields for 10/20–30 later; [P4] D5 only a handful of runs, labelled expected-failure; [P5] every replicate records seeds, scaffold, commit, wall time, peak RSS (sentinel).

## Task 7: Eligibility by endpoint × horizon (A5)

**Files:** Create `.../v033_evaluator/eligibility.py`, `scripts/build_group_event_state_v033_eligibility.py`; Output `eligibility_by_endpoint_horizon.json` (+ `/data/.../shared/eligibility/`).

Clauses: [G1] independent-block counts per phase come from `phase_block_counts`-style coverage on real `build_carry_segments` output, never session counts; [G2] required blocks per endpoint/horizon are read from the medium-oracle power curve (Task 6); missing curve → `power_curve_pending`, never a guessed threshold; [G3] endpoints: count 5/30 min profile, conditional grammar (N_future>0 anchors), H2a positive-K events, H2b seizures in development phases; [G4] a patient can leave a denominator for support only, never for a result.

## Task 8: Reports, contract publication, memory

**Files:** `canonical_evaluator.json`, `plain_report.md`, `technical_report.md` in `results/group_event_state/v0_3_3/evaluator_assay/`; `/data/.../shared/evaluator_contract/canonical_evaluator.json` (atomic rename). Three separated conclusion layers. Memory note for the agent (`~/.claude/.../memory/`).
