# Z/M ictal-carrier gate — pre-registered design (2026-07-24)

**Status: LOCKED before any carrier simulation runs.** Thresholds here are frozen; they must NOT be
edited after seeing results. Any post-hoc change requires a new dated spec revision with an explicit
rationale, and the old thresholds stay in the git history as the pre-registration of record.

Branch `codex/topic4-m4-snn-native-exit`. Companion archive:
`docs/archive/topic4/sef_hfo/zm_ictal_carrier_gate_2026-07-24.md`.

---

## 0. The one scientific question this gate answers

On the original 2-D anisotropic E/I spiking substrate (E1146 `twoend_equal`, L=20, N=40000) with the
locked per-neuron Z/M slow variables (`use_z=True, use_m=True, use_qI=False, use_gK=False`, lockpoint
`zA_q75_tz5000__mA0p001_tau500`), plus the divisive shared-inhibition pool S_G:

> Has the network produced a **sustained ictal carrier** — a spatially-localized-but-recruiting
> macroepisode whose high-frequency energy on the virtual SEEG stays **continuously elevated** —
> or only a **train of separated HFO-like bursts** whose energy returns to baseline between events?

A very high *core firing rate* does NOT settle this. A burst train can have core peaks of 100–400 Hz and
still be a train of separated spikes on the electrode. The gate is deliberately built so that "core rate
is high" cannot pass it; only **sustained, occupancy-gated, gap-bounded** elevation can.

---

## 1. The signals (precise definitions, units, sampling)

### 1.1 Source-space energy (Gate A input)
Computed from `E_spk_bool` (shape `(nsteps, NE)`, the per-step E spike raster returned by
`simulate_kick`). NOT from the LFP proxy — Gate A is the neural field itself.

- **core E rate `r_core(t)`** (Hz): population firing rate of the E cells inside the two low-V_th cores
  (`M4._e_disk_mask` around source/sink centroids, `PP.CORE_R`), binned at **BIN_FINE_MS = 5 ms**.
- **surround / all-E rate**: same, over non-core E / all E.
- **active-area fraction `A(t)`**: fraction of E cells that fired ≥1 spike in the 5 ms bin
  (a spatial-extent proxy).
- **axis / transverse kymographs**: E spikes projected onto the anisotropy axis (source→sink unit
  vector `axis_unit`) and its transverse; binned in **mm** × time. Used for onset-gradient /
  whole-field-flash detection and figures.
- **Gate-A energy trace `e_A(t)` = `r_core(t)`** (the source is the core). Smoothed with a 10 ms
  moving average before macroepisode analysis (SMOOTH_MS = 10).

### 1.2 Observed virtual-SEEG (Gate B input)
- The canonical LFP proxy (`src/snn_engine/lfp.py::LFPRecorder`), driven by the engine's built-in
  `lfp_recorder=` hook (`kick_probe.py:291`). Per contact `i`, over E neurons `j` within `p.Rr=0.278 mm`:
  `LFP(i,t) = Σ_j w_ij (|I_E(j,t)| + |I_I(j,t)|)`, `w_ij = f(d_ij)/Σf`, `f` per Methods Eq 9.
- **Units: mV** (voltage-equivalent synaptic drive). **Non-negative, rectified** — it is a
  `|current|` amplitude, NOT a signed field potential. Consequence: its band power is *envelope*-band
  power, folded by rectification. This is acceptable for the sustained-vs-intermittent discrimination
  (the question is whether band-limited power stays elevated or returns to baseline), but the gate
  language must say "envelope band power", never "raw LFP oscillation at f Hz".
- **Montage**: the E1146 15-contact montage `S["reg"]["montage_sheet"]` (SCL6–9, ICL1–11; source core
  SCL9/ICL9/ICL11, sink core ICL1/ICL2/ICL3), registered on the L=20 sheet. Reused as-is, not rebuilt.
- **Sampling: the engine samples every dt = 0.1 ms → 10 kHz.** We STORE the LFP downsampled to
  **LFP_STORE_HZ = 2000 Hz** (decimate by 5 with an anti-alias FIR at 800 Hz) to bound disk/memory.
  **Nyquist gate (hard, §11 stop condition #1): stored rate 2000 Hz → Nyquist 1000 Hz > 150 Hz. PASS.**
  If for any reason the stored rate would fall to ≤ 300 Hz (Nyquist ≤ 150 Hz), STOP — do not analyse.

### 1.3 Spectral parameters (Gate B), locked
- **STFT**: window **250 ms** Hann, hop **25 ms**, on the 2000 Hz LFP after **linear detrend + mean
  removal** per window (the raw proxy has a large DC offset; a PSD without detrend is DC-dominated).
- **Bands**: `low-gamma 30–80 Hz`, `high-freq 80–150 Hz`, `broadband 1–150 Hz`. Per-window band power =
  integral of the periodogram over the band.
- **Baseline**: the **pre-onset window** (t < onset, see §2) per contact. Report band power as
  **dB relative to the pre-onset median**: `10·log10(P / median_pre)`. Also compute a robust-z
  (`(P − median_pre)/(1.4826·MAD_pre)`) but **gate on dB, not robust-z** — MAD can be ~0 in a quiet
  pre-onset window and inflate robust-z spuriously (report both; the dB metric is load-bearing).
- **"Enhanced" threshold: ENH_DB = 6 dB** (≈ 4× power) above the pre-onset median, sustained.

---

## 2. Macroepisode / occupancy / gap (shared machinery, locked)

For any 1-D energy trace `e(t)` (source `e_A`, or a contact's band-power envelope):

1. baseline `b` = median of the **pre-onset window**; `amp = peak − b` where `peak = max(e)`.
   Pre-onset window = `[0, onset)`; `onset` = first time `e` crosses **ON = b + ON_FRAC·amp** and stays
   ≥ ON for ≥ **MIN_ONSET_MS**. If no such crossing → `no_onset` (Gate A fails trivially).
2. **FLOOR = b + FLOOR_FRAC·amp** — the "has not returned to baseline" level. Troughs above FLOOR mean
   the state stays elevated between microbursts.
3. **macroepisode** = the longest contiguous span `[t0, t1]` in which `e ≥ FLOOR` except for
   sub-FLOOR gaps each **≤ MAX_GAP_MS**. `duration = t1 − t0`.
4. **occupancy** = fraction of `[t0, t1]` with `e ≥ FLOOR`.
5. **max_gap** = longest contiguous sub-FLOOR run inside `[t0, t1]`.

**Locked constants**
| name | value | meaning |
|---|---|---|
| `ON_FRAC` | 0.30 | event-on level (matches the termination classifier's `on_frac`) |
| `FLOOR_FRAC` | 0.20 | sustained floor; troughs above ⇒ "not returned to baseline" |
| `MIN_ONSET_MS` | 100 | ON must persist this long to count as onset |
| `MAX_GAP_MS` | 250 | max sub-FLOOR gap inside a carrier macroepisode |
| `MIN_MACRO_MS` | 2000 | min carrier macroepisode duration |
| `OCCUPANCY_MIN` | 0.80 | min fraction of the macroepisode above FLOOR |
| `BIN_FINE_MS` | 5 | source-rate bin |
| `SMOOTH_MS` | 10 | source-rate moving-average before macroepisode analysis |

A trace is **"sustained"** iff `duration ≥ MIN_MACRO_MS AND occupancy ≥ OCCUPANCY_MIN AND max_gap ≤ MAX_GAP_MS`.
A trace is a **"burst train"** iff it has an onset but is not sustained (occupancy < min OR max_gap > 250 ms:
its energy returns below FLOOR — to near baseline — for long stretches).

---

## 3. Gate A — source-space carrier (all 8 clauses must hold)

| # | clause | operationalization |
|---|---|---|
| A1 | continuous macroepisode ≥ 2 s | `e_A` macroepisode `duration ≥ MIN_MACRO_MS` |
| A2 | not runaway/saturation | engine `runaway_early_stop_ms is None` AND tail rate not monotonically escalating |
| A3 | between microbursts source energy does not fully return to baseline | inter-burst troughs stay ≥ FLOOR (⇔ occupancy high + gaps short; A3 ≡ A4∧A5 by construction) |
| A4 | energy occupancy ≥ 80% | `occupancy ≥ OCCUPANCY_MIN` |
| A5 | max full-return gap ≤ 250 ms | `max_gap ≤ MAX_GAP_MS` |
| A6 | identifiable recruitment or sustained active region after local onset | active-area `A(t)` rises after onset AND stays elevated (macroepisode on `A(t)` too), OR axis kymograph shows spread from the onset core |
| A7 | form significantly different from early interictal discrete events | macroepisode {peak rate, duration, active-area} exceeds the median pre-onset IED event by ≥ `SEP_FACTOR` on ≥ 2 of 3 |
| A8 | not whole-field simultaneous flash | at onset, < `FLASH_FRAC` of the eventual active area ignites within `FLASH_WINDOW_MS`; there is a spatial onset gradient along/around the axis |

`SEP_FACTOR = 2.0`, `FLASH_FRAC = 0.80`, `FLASH_WINDOW_MS = 50`.

Gate A **passes** iff A1∧A2∧A3∧A4∧A5∧A6∧A7∧A8.

---

## 4. Gate B — observed virtual-SEEG carrier (all 6 clauses must hold)

| # | clause | operationalization |
|---|---|---|
| B1 | ≥ 2 valid contacts with sustained 30–80 Hz enhancement | ≥ `N_CONTACTS_MIN` contacts whose 30–80 Hz dB-envelope is "sustained" (§2) AND peaks ≥ `ENH_DB` |
| B2 | ≥ 1 high-freq index simultaneously enhanced | in the same macroepisode window, 80–150 Hz OR 1–150 Hz dB ≥ `ENH_DB` on ≥1 contact, overlapping the B1 window |
| B3 | occupancy ≥ 80% in a ≥ 2 s macroepisode | the B1 contacts' envelopes satisfy `duration ≥ MIN_MACRO_MS AND occupancy ≥ OCCUPANCY_MIN` |
| B4 | max sub-threshold gap ≤ 250 ms | B1 contacts' `max_gap ≤ MAX_GAP_MS` |
| B5 | not a train of separated narrow spikes returning to baseline | equivalent to B3∧B4 holding on the stacked band-power; if any B1-candidate contact has occupancy < min it is a train, not a carrier |
| B6 | ≥ 3-of-4 dims separated vs pre-onset returning events | macroepisode {duration, duty-cycle, total band-energy, spatial extent = #enhanced contacts} each ≥ `SEP_FACTOR`× the median pre-onset IED event value, on ≥ `DIMS_REQUIRED` of 4 |

`N_CONTACTS_MIN = 2`, `ENH_DB = 6`, `SEP_FACTOR = 2.0`, `DIMS_REQUIRED = 3`.

Gate B **passes** iff B1∧B2∧B3∧B4∧B5∧B6.

---

## 5. `ictal_carrier_verdict` (priority-ordered; the ONLY thing allowed to say "carrier")

```
if runaway (engine flag OR escalating tail):        fail_runaway
elif whole_field_flash OR saturated_plateau OR not has_recruitment:  fail_plateau
elif Gate A passes:
        candidate_observed_carrier   if Gate B passes
        candidate_source_only        otherwise
else:                                                fail_hfo_like_train
```

- **`candidate_observed_carrier`** — A+B pass → allowed to write *"candidate sustained ictal carrier on
  the original spatial SNN substrate"*.
- **`candidate_source_only`** — A passes, B fails → *"source-space candidate without an observed
  virtual-SEEG ictal carrier"*. NOT a carrier claim.
- **`fail_hfo_like_train` / `fail_plateau` / `fail_runaway`** — no carrier. These are the honest negative
  outcomes; each routes Phase 2 differently (§7 of the task).

The old M4-2 termination classifier's `fragment` label is **not** allowed to stand in for any of these.
`fragment` describes the *activity-trace shape* for termination; it does not adjudicate whether an ictal
carrier exists. The two vocabularies are kept strictly separate.

---

## 6. `lifecycle_verdict` (gated behind carrier pass)

Emitted ONLY when `ictal_carrier_verdict ∈ {candidate_source_only, candidate_observed_carrier}`; else
returns the sentinel `carrier_not_established` (never one of the 6 lifecycle labels).

```
if carrier not established:              carrier_not_established
elif prevented (matched control had onset, this arm did not):  prevention
elif not onset_detected:                 no_onset
elif not terminated:                     persistent
elif reignited:                          terminate_then_reignite
elif interictal_recovered:               terminate_and_recover
else:                                    terminate_to_silence
```

Phase 1 arms are H-OFF (no termination actuator): a carrier-passing arm can only be `persistent`; a
carrier-failing arm gets `carrier_not_established`. The `terminate_*` / `prevention` labels are Phase-2
Path-A outcomes and require an H-on arm + a matched control.

---

## 7. Arms (seed 1 first; seeds 3/4 only if seed 1 passes Gate A)

| arm | config | purpose | H |
|---|---|---|---|
| `interictal_ctrl` | Z/M frozen (slow=None-equivalent: z=1,m=0) OR the pre-onset window itself | baseline LFP band-power + returning-event reference | off |
| `bare` | Z/M only | natural interictal→onset→runaway reference; runaway-truncated ~2.9 s | off |
| `sg` | Z/M + S_G (α_G=16) | THE candidate carrier; 15 s | off |

H is frozen off in Phase 1. Do not tune α_H / τ_H / burst-count until the carrier question is settled
(task §6, §7). Pre-onset returning-events reference is taken from the `sg` run's own `[0, onset)` window
(same substrate, same seed); `interictal_ctrl` is a cross-check.

---

## 8. Forbidden claims (until the gate says so)

- No "ictal attractor / limit cycle / lifecycle complete" without §8-of-task slow-fast evidence.
- No "sustained ictal carrier" unless `candidate_observed_carrier` (A+B).
- No "high core rate ⇒ high-freq SEEG energy".
- No "H can't build" / "sensor is the problem" — H is frozen off here; that is a Phase-2 question.
- No claim the old q_I substrate proved anything about a Z/M interictal attractor.
- "candidate" everywhere: seed-1 pilot until seeds 3/4 replicate.

---

## 9. Outputs
- `results/topic4_sef_hfo/zm_ictal_carrier_gate/{arm}_seed{n}.npz` (LFP@2kHz, rates, kymographs, field snaps, slow traces)
- `.../{arm}_seed{n}.json` (per-arm provenance + gate metrics + `ictal_carrier_verdict` + `lifecycle_verdict`)
- `.../carrier_gate_seed{n}.json` (accumulated manifest across arms; crash-safe, per-arm atomic)
- `.../figures/carrier_diagnostic_seed{n}.png` (+ README.md, Chinese, per repo standard)
