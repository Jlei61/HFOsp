# FCXR-HEO1 — High-Energy Oscillatory Branch Acquisition (design lock)

> **For agentic workers:** implement task-by-task with TDD; every numeric gate in
> §3 is **pre-registered** (locked before any scientific run) so GO/NO-GO cannot be
> tuned to the outcome. Branch `codex/topic4-mz-fcxr-heo1`, base `01225ff7` (LC1 tip).

**Goal:** On the locked E1146 / L=20 / N=40000 / dt=0.05 FCXR-RC1 substrate, test whether an
OFF-by-default *cooperative recurrent conductance* gate — low activity fully closed, mid activity
super-linear recurrent gain, high activity still limited by the existing tanh conductance
saturation — can produce a **repeatable, sustained, oscillatory, high-frequency broadband,
full-15-contact-platform, numerically-bounded high-energy branch** on the frozen slow state.

**Architecture:** One non-blessed engine-plugin change (`mz_slow_vars.py`): insert a monotone
cooperative transform on the *raw* recurrent conductance `gErec_raw` **before** the existing
`g_sat·tanh(·/g_sat)` saturation, plus a pure-side-effect streaming histogram of `gErec_raw` for
baseline calibration. One new pure spectral/HEO classifier module + synthetic TDD. One new runner
with resource/nohup contracts driving Stages F0–F5. Three diagnostic figures. All new config
OFF by default → byte-parity with RC1 preserved.

**Tech stack:** numpy, scipy.signal (decimate/welch), matplotlib; existing `src/snn_engine`
(`kick_probe.simulate_kick`, `lfp.LFPRecorder`, `mz_slow_vars.MZSlowVars`), `scripts/run_m4_phaseplane.build_substrate`, `src/topic4_mz_fcxr_dynamics` (frozen field + workpoint classifier).

---

## Global Constraints (verbatim locks)

- **Substrate anchors (NEVER change):** subject `epilepsiae_1146`, montage `narrow`, `L=20.0`,
  `DENSITY=100.0` (→ N=40000, NE=32000), `dt=0.05`, connectivity/core placement/external drive as
  built by `PP.build_substrate(seed)`. No montage/geometry/drive edits to rescue a result.
- **Fast-system anchors (arm-C, from `_fc_cfg`):** `membrane_mode="full_conductance"`, `E_E=58.0`,
  `c_E=1.0`, `v_match=18.0` (→ `denomE=40.0`), `e_gaba=0.0`, `e_k=0.0`, `ff_conductance=False`
  (external/feedforward AMPA stays **additive**), `rec_conductance=True`, **`rec_sat_g = g_sat =
  21.6` (main anchor, kept)**, `gaba_gain=1.125`, `z_scope="local_only"`,
  `global_gaba_mode="additive"`, `global_gaba_fraction=0.0`, `max_total_conductance=99.0`.
  `I_th_EI`/`tau_z`/`use_z` are **unused** here (frozen field only; no dynamic Z).
- **Frozen slow state:** `z_frozen_E = frozen_z_field(p_i, D) = clip(1 − D·p_i, 0, 1)` with
  `p_i = load_onset_depletion_pi(snapshots/zA_q75_tz5000/seed_{seed}.npz)["p_i"]`
  (`dep/mean(dep)`, `dep=clip(1−z_E[onset],0,∞)`). `use_z=False`. Field must pass
  `assert_field_substrate_aligned(pi_pack, S)` before every run.
- **6 blessed engine files NEVER touched:** `kick_probe.py`, `params.py`, `model.py`,
  `connectivity.py`, `connectivity_rot.py`, `lfp.py` (hashes in `results/.../engine_versions.json`).
- **New cooperative mechanism:** `n=4` fixed (no n-grid); `g_sat=21.6` primary; no hard clip as a
  science mechanism; `max_total_conductance=99.0` safety gate + clip audit stay in effect.
- **Priority edit set only:** `src/snn_engine/mz_slow_vars.py`, `src/topic4_mz_fcxr_heo1.py`,
  `scripts/run_topic4_mz_fcxr_heo1.py`, `scripts/plot_topic4_mz_fcxr_heo1.py`,
  `tests/test_topic4_mz_fcxr_heo1.py`. Widening scope requires a STOP/STATUS note first.
- **LC1 conclusions accepted (not re-litigated):** (1) RC1 slow-off workpoint accepted; (2) Stage D
  = no robust independent finite-high branch, only continuous densification of the same core→axis
  self-terminating events; (3) q75 dynamic-Z = bounded dense-event train, not ictal; (4) q50+X =
  persistence-gated X has termination authority but no full pre/ictal/post recovery. **Do NOT** re-scan
  q50/q75 as a main task, add divisive/global inhibition, or run dynamic Z in this sprint.
- **Not-success:** "more events / shorter inter-event gaps" is NOT a win. High rate ≠ broadband
  energy. Local core activity ≠ full platform. Kick-sustained ≠ autonomous lifecycle.
- **Total budget ~8 h.** No new simulation after **7 h 30 m**; last 30 min = assemble/figures/
  README/archive/STATUS/commits. A clean NO-GO is a valid deliverable.

---

## New mechanism (locked equations)

Insert in `MZSlowVars.membrane_terms`, full-conductance branch, on `gErec_raw = c_E·I_recE/denomE`
**before** the tanh (currently `mz_slow_vars.py:284→287`):

```
u        = gErec_raw                             # raw recurrent conductance (kept raw for histogram/clip audit)
H        = relu(u − u_c)^n / (K_c^n + relu(u − u_c)^n)     # Hill cooperative gate, n=4
u_tilde  = u · (1 + A_c · H)                      # super-linear mid-activity gain
gErec    = g_sat · tanh(u_tilde / g_sat)          # SAME existing saturation (g_sat = rec_sat_g = 21.6)
```

New `MZSlowVarsConfig` fields (all OFF by default):
`coop_A: float = 0.0` (A_c), `coop_uc: float = 0.0` (u_c), `coop_Kc: float = 0.0` (K_c),
`coop_n: int = 4`.

Invariants (each is a TDD clause, §T1):
1. `coop_A == 0.0` → cooperative block skipped entirely → **byte-identical** to RC1 (full
   `simulate_kick` parity, extends `test_engine_byte_parity_both_off_equals_slow_none`).
2. `u ≤ u_c` cells → `relu=0 → H=0 → u_tilde = u·1.0 = u` exactly (bit-exact even when `A_c>0`).
3. Transform is monotone non-decreasing, non-negative, finite in `u`.
4. Mid-region effective slope `d gErec/d u` can exceed the RC1 slope (super-linear bump).
5. Large `u → tanh` still saturates `gErec → g_sat` (bounded).
6. Acts on recurrent E→E conductance only (gEff, gI, gM, I-cells, `gErec_raw` audit value unchanged).
7. Default config → all existing tests green.
8. Streaming `gErec_raw` histogram does not change the trajectory (pure side-effect).
9. LFP recorder on/off does not change spikes (engine-level; already true, add regression).
10. Deterministic: same seed+config → identical state.

**Validation** (in `_validate_config`): `coop_A ≥ 0`; if `coop_A > 0` require
`membrane_mode=="full_conductance" and rec_conductance and rec_sat_g > 0` and `coop_uc > 0` and
`coop_Kc > 0` and `int(coop_n) ≥ 1` (cooperative gain with no saturation is forbidden).

**Streaming baseline calibration** (pure side-effect, new fields
`record_gerec_hist: bool=False`, `gerec_hist_edges: np.ndarray|None=None`): fixed-edge cumulative
int64 histograms of `gErec_raw` for **overall / core / surround** (via `core_e_idx`/`surr_e_idx`),
summed each step in `membrane_terms`. NO n_cell×n_time matrix. Calibration is read only from the
A_c=0 slow-off baseline (never from a cooperative run). Engagement traces (appended only when
`coop_A>0`): `trace_coop_engaged_frac` = mean(`u>u_c`), `trace_coop_H_mean` = mean(H).

---

## §3 Pre-registered HEO classifier + gate thresholds (LOCKED)

Pure module `src/topic4_mz_fcxr_heo1.py`. Consumes `lfp_trace (nsteps,15)`, `rate_E (nsteps,)`,
`dt`, `contact_names`, `scl_mask (15,)`, and a baseline reference computed from the F0 slow-off run.

**Spectral setup:** LFP + population rate sampled at 20 kHz (every step). Decimate to
`FS_WORK = 1000 Hz` (`scipy.signal.decimate`, FIR anti-alias, factor 20). Spectrogram window
`WIN_MS = 1000`, hop `HOP_MS = 100`; per window Welch PSD (Hann, `nperseg=500`, 50% overlap),
integrate over six bands **`BANDS = [(1,4),(4,8),(8,13),(13,30),(30,80),(80,150)] Hz`**.

**Baseline reference (F0 slow-off, A_c=0):** per (contact, band) over all F0 windows:
`base_med = median(log10 power)`, `base_mad = MAD(log10 power)`, `base_q99 = q99(power)`.
`base_mad == 0` → that (contact,band) **fails closed** (robust-z undefined → treated as
not-passing + flagged; never divide-by-zero into a huge z).

**Per test window, per (contact, band):**
`robust_z = (log10 power − base_med) / (1.4826·base_mad)`.
`band_pass = (robust_z ≥ Z_GATE) AND (power ≥ base_q99)`, with **`Z_GATE = 3.0`**.

| Gate | Locked criterion |
|---|---|
| **B broadband** (per contact, per window) | `n_bands_pass ≥ 5/6` **AND** band_pass for (30,80) **AND** (80,150) **AND** median over 6 bands of `10·log10(power/base_med_power) ≥ 6.0 dB` → `contact_broadband_high` |
| **C platform** (per window) | `n_contacts_broadband_high ≥ 11/15` **AND** `scl_broadband_high ≥ 3/4` → `platform_high` |
| **A plateau** | longest run of `platform_high` merging gaps `<100 ms`; require duration `≥ 1000 ms`, occupancy (fraction platform_high) `≥ 0.80`, no return-to-baseline gap `≥ 100 ms` |
| **D oscillation** (on plateau window) | population-rate PSD **and** `≥ half` of passing contacts each have a non-DC peak in 30–150 Hz with prominence `≥ 6 dB` over local (±20 Hz) background; `≥ half` of those contact peaks within `±15 Hz` of the median center. Report rate PSD, contact PSD, autocorrelation. Tonic/flat → fail |
| **E numerical** | all finite; `clip_frac_max == 0`; `tau_eff_min_ms ≥ 2·dt (=0.1)`; NOT runaway (`runaway_early_stop_ms is None`); plateau mean E rate `< CEIL_HZ = 400 Hz` (not pinned at 500 Hz refractory ceiling) |
| **F workpoint** (arm-level, D=0/no-kick) | `classify_run_workpoint → INTERICTAL_WORKPOINT` **AND** returning sparse irregular IED present **AND** numerically safe |

`HEO_BRANCH = A AND B AND C AND D AND E`. **F** is the arm precondition checked before D>0 screening.
Single-seed or single-IC only → mark **provisional**, never "confirmed branch". "platform readout"
(not "full tissue recruitment") whenever only virtual-LFP passes without SCL-neighborhood tissue
recruitment corroboration.

**Runaway operational stop:** `simulate_kick(..., early_stop_runaway=True, es_thresh_hz=250.0,
es_dur_ms=100.0)` → rolling >250 Hz for 100 ms ends the run as `OPERATIONAL_RUNAWAY` (not HEO).

**Synthetic classifier TDD (7 cases, §T3):** build synthetic baseline + test (contact×time) LFP:
sparse-irregular-IED → not HEO (A fails); dense-event-train-zero-gaps → not HEO (A occupancy<0.80);
tonic-ceiling → not HEO (D no peak / E ceiling); narrow-band-local-only → not HEO (B or C fails);
broadband-nonoscillatory → not HEO (D fails); oscillatory-broadband-platform → HEO; silent-post-tail
→ silence not "recovered".

---

## Resolved ambiguities (locked)

- **IC labels:** `no-kick` = IC low (`t_kick=1e9`, `kick_boost=0.0`); `kick3` = IC high1
  (`t_kick=120.0`, `kick_boost=3.0`); `kick12` = IC high2 (`kick_boost=12.0`). One kick disk at
  `S["src_xy"]`, `r_kick=PP.R_KICK=0.3`, window `[t_kick, t_kick+18)`.
- **u_c calibration:** `u_c = quantile(gErec_raw_overall_baseline, gate_quantile)` from the F0
  streaming histogram; `gate_quantile ∈ {0.999, 0.9999}` (= Q99.9 / Q99.99); `K_c = 0.25·u_c`.
- **`fail_on_clip=False`** during all HEO runs (record `clip_frac`, disqualify HEO if `>0`), matching
  Stage-D discipline; `max_total_conductance=99.0` gate stays active. Hard clip is never the mechanism.
- **Histogram edges:** finalized in F1 smoke from a short slow-off `max_raw_gErec` probe; linear grid
  `[0, G_HIST_MAX]` (`G_HIST_MAX` ≥ 1.2× observed baseline max, default 21.6) with fine bins +
  an `inf` overflow bin; F0 must show Q99.9/Q99.99 not in the overflow bin (else widen + rerun F0).
- **Montage access** `S["reg"]["montage_sheet"].contacts/.names`, SCL mask, and snapshot/substrate
  readability from the worktree are **verify-first** in F1 smoke; any failure → STOP + report (§0/§6).

---

## File structure

- **Modify** `src/snn_engine/mz_slow_vars.py` — cooperative transform (T1) + `gErec_raw` histogram &
  engagement traces (T2). Non-blessed; existing tests stay green.
- **Create** `src/topic4_mz_fcxr_heo1.py` — spectral/HEO classifier + baseline reference builder (T3).
- **Create** `tests/test_topic4_mz_fcxr_heo1.py` — cooperative-transform parity/monotone/saturation
  clauses (T1/T2) + HEO synthetic classifier clauses (T3).
- **Create** `scripts/run_topic4_mz_fcxr_heo1.py` — modes `baseline` (F0) / `smoke` (F1) /
  `screen` (F2) / `confirm` (F3); resource + nohup + flock + manifest + sentinels (T4).
- **Create** `scripts/plot_topic4_mz_fcxr_heo1.py` — branch map / virtual-SEEG spectral / spatial
  modes diagnostics (T6).
- **Results root:** `results/topic4_sef_hfo/mz_full_conductance_spatial_relay/high_energy_oscillatory_branch/`.

---

## Config dicts (exact)

`heo_cfg(A_c=0.0, u_c=0.0, D=None, seed=..., record_hist=False, edges=None)` returns
`MZSlowVarsConfig(**_fc_cfg(1.0, ff_conductance=False, rec_conductance=True, fail_on_clip=False,
rec_sat_g=21.6))` overridden with: `coop_A=A_c, coop_uc=u_c, coop_Kc=0.25*u_c, coop_n=4,
record_clip_identity=True`; and `record_gerec_hist=record_hist, gerec_hist_edges=edges`; and
`z_frozen_E = None if D in (None,0) else frozen_z_field(p_i, D)`.

- **F0 baseline:** `A_c=0`, `D=None`, `record_hist=True`, seed1, T=8000 ms, LFP on 15 contacts.
- **F2 workpoint gate (per arm):** `A_c>0`, `u_c` per gate_quantile, `D=None`, no-kick, T=4000 ms.
- **F2 screen cell:** `A_c>0`, `u_c`, `D∈{0.13,0.15}`, IC∈{no-kick,kick3,kick12}, T=4000 ms.
- **F3 confirm:** locked minimal candidate, seed1 {no-kick,kick3,kick12} + seed3 {no-kick + min kick}
  + matched cooperative-OFF control, T=8000 ms.

---

## Tasks (TDD, commit per task)

- **T1 Cooperative transform + validation** — edit `mz_slow_vars.py`; add `test_topic4_mz_fcxr_heo1.py`
  clauses 1–7,10 (byte-parity both A_c=0 unit + full engine; `u≤u_c` exact; monotone/non-neg/finite;
  mid-slope>RC1; saturation; recurrent-only; validation raises). Run existing `test_mz_slow_vars.py`
  + `test_mz_full_conductance_spatial_relay.py` → all green. Commit.
- **T2 gErec_raw streaming histogram + engagement traces** — clause 8 (histogram no-perturb: spikes
  identical with/without) + engagement-trace lengths; slow-off histogram quantile helper
  `gerec_baseline_quantiles(mz, qs)`. Commit.
- **T3 HEO spectral classifier** — `src/topic4_mz_fcxr_heo1.py`: `build_baseline_reference(lfp,rate,dt)`,
  `spectrogram_bandpower(...)`, `classify_heo(lfp,rate,dt,scl_mask,baseline_ref)` → verdict dict
  (gates A–E + plateau window + oscillation metrics). 7 synthetic clauses. Commit.
- **T4 Runner + resource/nohup contract** — `scripts/run_topic4_mz_fcxr_heo1.py`; reuse
  `build_substrate`, `build_core_masks`, `LFPRecorder`, `simulate_kick`, workpoint classifier;
  `_apply` OMP=1 env, flock launcher, `_plan_workers`, per-cell wall guard + runaway early-stop,
  `run_manifest.json`, `launcher.pid`, `RUNNING.json`, `resource_log.jsonl`, `DONE/FAILED/RESOURCE_ABORTED.json`.
  Dry-run RSS/raster estimate. Commit.
- **T5 Stage execution F0→F3** (see stage plan). Commit artifacts per stage.
- **T6 Figures + README + archive + STATUS** — 3 diagnostics; `figures/README.md` (中文);
  `docs/archive/topic4/sef_hfo/mz_fcxr_heo1_2026-07-24.md`; `STATUS.md`. Commit.

---

## Stage plan (GO/NO-GO)

- **F0 baseline** (seed1, D=0, slow-off, A_c=0, T=8000): write `baseline_spectral_contract_seed1.json`,
  `baseline_rec_hist_seed1.npz`, `baseline_lfp_seed1.npz`. Confirm `INTERICTAL_WORKPOINT` + numerical
  safe + Q99.9/Q99.99 in range. **Baseline not reproduced → STOP**, do not enter new mechanism.
  seed3 baseline only when a candidate is confirmed.
- **F1 smoke** (L=20, 500–1000 ms): A_c=0 parity vs RC1; A_c>0 finite; histogram/LFP plumbing;
  verify montage/SCL/snapshot/substrate wiring; finalize histogram edges. Commit on pass.
- **F2 screen:** `gate_quantile∈{0.999,0.9999}` × `A_c∈{1,2,4,8}` (n=4, K_c=0.25·u_c, g_sat=21.6).
  For each arm: D=0/no-kick/T=4000 **workpoint gate** — arms that break baseline are dropped.
  Surviving arms: `D∈{0.13,0.15}` × IC∈{no-kick,kick3,kick12}, T=4000. Per cell save numerical
  safety, rate/active-fraction, plateau duration/occupancy, six-band contact metrics, oscillation
  metrics, contact coverage, core/axis/off-axis recruitment, gate engagement fraction, gErec
  raw/effective summaries, IC/kick provenance. If all A_c up to 8 give no HEO **but** cooperative arms
  are already broadly saturated, one pre-registered sensitivity on the ≤2 closest arms at
  `g_sat=25.9` (+20%). No further g_sat grid, no drive change.
- **F3 confirm:** lexicographic minimal candidate (numerical safe → D=0 workpoint preserved → HEO full
  gate → min A_c → prefer g_sat=21.6 → min kick/IC). seed1 T=8000 {no-kick,kick3,kick12}; seed3
  T=8000 {no-kick + min kick}; matched cooperative-OFF control. Do not re-select params on seed3.
  Single-seed/IC → provisional. **F3 spatial/eigenmode readout** (candidate only): baseline-IED /
  early-high (pre-onset 200 ms) / plateau energy fields, leading mode, IPR, coverage; report
  baseline↔early-high mode cosine + plateau breadth/IPR; sparse/LinearOperator only (no 32000² dense).
- **F4 phi rescue (optional):** only if a sustained broadband full platform exists but fails the
  oscillation gate — `tau_phi∈{20,50,100}`, `delta_phi∈{0.25,0.5,1.0}` on the top candidate, seed1
  T=4000; min phi that passes oscillation → seed3 confirm. No plateau → phi forbidden.
- **F5 X-authority preview (optional):** only after seed1+seed3 confirmed HEO and ≥90 min left;
  matched X-off/X-on on a data-like HEO, frozen Z, LC1-minimal X params; asks only whether X still
  terminates. No dynamic Z, no lifecycle/recovery claim.

---

## Resource / nohup / OOM contract (§7)

- Before every long sim: log `MemAvailable/SwapTotal/SwapFree`, swap baseline; check sibling 40k jobs.
- Single flock launcher; `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=1`.
- Workers: `T≤8000` → ≤2; `T>8000` → 1; many sibling jobs → 1. No large 40k worker pools.
- No dense 32000² matrix; no extra cell×time conductance arrays; LFP = 15 contacts only. Raster memory
  estimated in dry-run + written to manifest.
- Resource gates — **soft:** `MemAvailable<64 GiB` or swap delta `≥256 MiB` → stop submitting, wait/
  downshift. **hard:** `MemAvailable<32 GiB` or swap delta `≥512 MiB` → kill *this sprint's* latest own
  worker, write `RESOURCE_ABORTED.json`. Only kill PIDs provenance-tagged to this sprint.
- Long commands: `setsid nohup <cmd> > <run_dir>/nohup.log 2>&1 < /dev/null &`; immediately write
  `launcher.pid`, `launch_baseline.json`, `RUNNING.json`, `resource_log.jsonl`; confirm detached via
  `kill -0` + `ps`. Per-cell wall guard; unsafe / >250 Hz / timeout → safe self-stop.

## Stop rules (§6, any → archive clean NO-GO)

baseline workpoint not preserved; A_c=0 not byte-parity; all safe arms only dense event trains; high
activity only via clip/unsafe/450 Hz ceiling; only core contacts (no platform); only broadband noise
(no oscillation); only narrow-band oscillation (no broadband platform); needs drive/connectivity/
montage change; needs divisive/global mechanism; 8 h boundary reached.

## Forbidden claims

seizure lifecycle achieved; clinical seizure reproduced; Hopf proven (unless full Jacobian
continuation done); bistability proven (unless multi-IC + hysteresis/branch evidence); kick-induced
branch as spontaneous transition; virtual-LFP platform as full tissue recruitment; high rate as
broadband energy; dense IED train as ictal plateau.
