# SNN-native M4 containment-to-exit lifecycle — locked design + lineage audit (2026-07-21)

> **Status**: Stage-0 lock. Mechanism SCREEN, not a seizure claim. This spec is the plan-of-record;
> re-read the relevant section at every stage boundary (CLAUDE.md §5).
> **Branch**: `codex/topic4-m4-snn-native-exit`  **Base SHA**: `4d40b03` (= `main` HEAD at lock time).
> **Worktree**: `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-m4-snn-native-exit`.

---

## 0. 一句话（朴素表述，CLAUDE.md §8）

**测什么** —— 我们有一个二维兴奋/抑制脉冲网络（真 spiking，不是 rate toy），它已经会做两件事：
(1) 背景噪声下自发点火、沿一条连接长轴来回放电，然后自己停下来（间期事件）；
(2) 当一个"抑制油箱" `q_I` 慢慢漏空、再叠一个"除法式全局抑制池" `S_G` 兜住时，网络会掉进一个**不失控但也不结束**的持续放电态（M4 有界第三态）。缺的就是**结束**：一个能让这个持续态自己停下来、并且之后还能恢复回间期的机制。

**怎么测** —— 先不建全套慢机制。先在真网络里把这个有界持续态复现出来，然后**冻结**它、往里加一个**可控的、有大小有时长的"恢复位移"**（先用已有的抬阈值 `inhibitory_pulse` 钩子，零改代码），看它到底能不能被推出这个态、推出后会不会又弹回失控。只有当"确实存在一个能干净退出又不弹回"的窗口时，才去建那个**新机制**：一个**只在活动持续够久之后才积累**的局部恢复变量 `p(x,t)`——短间期事件里它几乎不涨（所以不误杀间期事件），只有广泛持续招募建立后才涨起来，把持续态推过退出边界、留下一个局部"不应期尾迹"，然后慢慢衰减、让易感性恢复。

**为什么这次不一样** —— 过去所有加"结束变量"的尝试都在两个点上翻车之一：**执行器类型错**（削递归增益的 STD、电导 shunt——只会碎裂/压死/反而把有界态推成失控），或**传感器时机错**（按 spike 计数或瞬时活动的 `m`/`g_K`/均匀全局刹车——从安静基线就开始积累，把普通间期事件也刹掉）。从没有人把**"持续时间门控的传感器"**（`h_G` 已经证明这种门控可造，只是接错了执行器）配上**"局部外向恢复电流"**（`g_K` 的执行器方向本来就对，只是它的传感器是瞬时活动、在 ~55ms 的短事件里根本来不及积累）**并放在 M4 有界态上测**——而 M4 有界态恰好是一个**持续**态，第一次给了这个传感器一个能积累的持续输入、也第一次给了执行器一个真正需要被终止的态。（内部代号：M3A-A1c uniform additive、M3A q_I/g_K field、M4 S_G divisive pool、M4-2 STD、M4-3A shunt、MZ z/m、M3A-v2.2 h_G、R4 latch。）

---

## 1. Scientific goal (unchanged from task brief)

In the original 2D E/I spiking network — anisotropic E→E scaffold, two low-threshold cores at the axis
ends, background-noise spontaneous events, virtual-SEEG readout — produce, along **one continuous SNN
trajectory** with no manual reset:

interictal baseline → several returning spontaneous interictal events → natural entry into a **bounded**
ictal-like spatiotemporal state (autonomous bursts / sustained oscillation, **not** a flat rate ceiling or
numerical ceiling) → self-termination → recovery to the **same** interictal basin → returning interictal
events again — readable in both source space and virtual SEEG.

This is the Figure-5 candidate. Any low-dimensional analysis serves and returns to the spatial SNN; no rate
toy substitutes for the SNN goal.

---

## 2. Base SHA, no-touch boundaries, independence

- **Base SHA**: `4d40b03` (`main` HEAD; verified `4d40b03` == `main`, and R4 `3fd7b9b`, FCXR `c0ecb1a`,
  old-MZ `6dc71c0` are all **NOT** ancestors of `main` — base is clean of those lines).
- **Worktree** (develop here only): `.worktrees/topic4-m4-snn-native-exit`, branch `codex/topic4-m4-snn-native-exit`.
- **READ-ONLY, never modify** (peer worktrees): `topic4-mz-conductance`, `topic4-mz-fcxr-stage-d`,
  `topic4-mz-divisive-lifecycle`, `topic4-mz-slowvars`, `topic4-mz-direct-spatial-modes`,
  `topic4-mz-onset-dynamics`, `topic4-mz-early-bridge-v2`, `topic4-mz-slow-fast-transition`, `topic4-early-readout`.
- **Actively running, do not disturb**: an FCXR Stage-D grid runs in `topic4-mz-fcxr-stage-d`
  (`run_topic4_mz_fcxr_stage_d.py`, ~2 workers + parent, ~20 GB RSS, ~2 cores). My resource budget must
  leave it untouched.
- **Independence from FCXR/conductance line**: do NOT import its fast-structure equations or constants —
  full-conductance membrane `τ_m V̇ = −V + c_E·I_E^ff + g_rec_eff·(E_E−V) + g_I(E_I−V) + g_M(E_K−V)` with
  `E_E=58`; force-match `c_E∈{0.85,1,1.15}` at `V_match=18`; recurrent saturation
  `g_rec_eff = g_sat·tanh(g_rec_raw/g_sat)`, `g_sat=21.6`; persistence-gated E→E relay `x_j` (Hill n=4).
  My line keeps E→E connectivity/kernel/delay and the **additive-current** membrane exactly as the M4
  baseline, and adds only its own slow exit/recovery coordinate.

**Do not change**: E→E topology/kernel/AR/direction/delay; neuron positions; core geometry/thresholds;
E/I counts and connection cardinality; virtual-SEEG montage; baseline noise/seed contract.

---

## 3. Mechanism ledger (lineage audit — what exists, what is exhausted, what is open)

Two orthogonal failure axes recur across the whole lineage: **actuator-type** (what lever) and
**sensor-timing** (when it accumulates). Every prior exit attempt was wrong on one axis.

| # | Mechanism (code) | Sensor | Actuator (action point) | Spatial | τ | Tested grid (EXACT — DO NOT re-run) | Result / failure | Reusable for exit? |
|---|---|---|---|---|---|---|---|---|
| E1 | q_I depletion field (M3A/M4) | local E+I activity | scales E inhibition (disinhibition) | per-neuron field | τ_q=5000ms | Step2: 3 substrate × 4σ_q × 3q_min × 5Δq × 4seed = 576 (g_K=0) | **ENTRY** ✓ (drives up); alone → 532 no-effect/44 runaway/0 expanded | **ENTRY only** — reuse at k_q=0.10 |
| E2 | z-only per-neuron disinhibition (MZ) | I_I current | scales E inhibition | per-neuron | τ_z 2500–10000 | I_th∈{q50,q75,q90}×τ_z{2500,5000,10000}, seed1; +seeds1/3/4 4 cells | **ENTRY** ✓ (bounded→runaway, 3-seed); not exit | ENTRY (independent path; not used here) |
| C1 | **S_G divisive shared pool (M4 Pass-1)** | fast E rate → pool | **÷ recurrent-E gain** `I_rec_E/(1+α_G·S_G)` | global pool | τ_mu=30/τ_S=80 | anchor `k_q=0.10, α_G=16`; 40s seeds1/3/4 (seed2 delayed) | **CONTAINMENT** ✓ (bounded 3rd state, 3/4 seed); broad/marginal/**non-terminating** | **CONTAINMENT** — reuse at α_G=16 |
| X1 | uniform additive global feedback (A1c) | global mean E rate (EMA) | subtractive current, **all E uniformly** | uniform scalar | τ 150/2000 | gain{0,8,16,32}×τ{150,2000}×3seed on l1/l2 | no-go: **actuator SPATIAL-SUPPORT wrong** — can't hit core without over-pressing surround (`joint_window=false`) | NO (wrong spatial support) |
| X2 | g_K fatigue field (M3A Step3) | **instantaneous** local E activity | **local outward K current** `−η_K·g_K` | per-neuron field | τ_K=5000 | σ_q/σ_K{1⁺,1.5,2}×Γ_K{0,.5,1,1.5,2}×4seed = 432 | no-go: only brakes/still-axial; **achieved Γ_K 0.082 vs target 2.0 — event too short (~55ms) to accumulate** → **SENSOR-TIMING** | **ACTUATOR reusable if driven by a persistence sensor** ← key |
| X3 | q_I+g_K + low-q preload (Step4) | local activity | q_I disinhib + g_K | fields | — | 2sub×seed{1,2}×kq{.02,.05,.10}(24)+finer(12)=36 | 0/36: no stable intermediate q on sampled grid; full-state co-preload **untested** | partial (co-preload open) |
| X4 | spike-count adaptation m (MZ Arm B) | **E spike count** (leaky) | subtractive current `−η_m·m` | per-neuron | τ_adp 500–5000 | τ_adp{500,2000,5000}×frac{.05,.10,.20} seed1 | no-go: **always-on prevention** (6/9 suppress) — charges from quiet baseline → **SENSOR-TIMING** | **actuator ok; sensor wrong (persistence gate never tried)** |
| X5 | z+m combined (MZ Arm C) | I_I + spike count | disinhib + adaptation | per-neuron | — | collapsed to strong-z×strongest-m corner, seed1 | **UNTESTED-MIDDLE** (m cancels z to baseline in the one corner) | middle open (not our path) |
| X6 | STD E→E depression (M4-2) | E presyn spikes | **÷ recurrent-E gain** (Tsodyks x) | per-synapse | τ_rec 1000–5000 | u{.15,.30,.50}×τ{1k,2.5k,5k}+Arm0, seeds1/3/4; +seed1 low-u{.05,.08,.11} | **clean no-go 3-seed**: fragment(fast)/suppress(slow), no hold-window → **ACTUATOR-TYPE** | NO (recurrent-gain lever) |
| X7 | continuous conductance shunt (M4-3A) | activity load a | **conductance shunt** `1+g_A` | iso field | τ_n 5k–40k | α_A{2,4,8}×τ_n{5k,20k,40k}+Arm0 seeds1/3/4; +α_A{5,6,7}@τ_n20k = 30 | **clean no-go 3-seed, WORSE**: weak→**runaway** (starves S_G), strong→fragment → **ACTUATOR-TYPE** | NO (starves S_G) |
| X8 | h_G global recovery (M3A-v2.2) | **globality M/B/Π Hill-AND** | subtractive current, **all E** | uniform scalar | τ_G 600 | 3184 sims sustained ramp+HOLD; η_G ladder{0..80}; L16 repl | no-go on **pre-M4 all-or-none** substrate: **sensor WORKS** (stays ~0 in local events, rises only in runaway), subtractive-global **actuator can't pull back saturated avalanche** | **SENSOR reusable**; actuator no |
| R4 | regional q–p–M hybrid (rate-patch) | prescribed schedule | additive-M + **discrete latch** | 3 fixed patches+bath | .225s/12s | single center point | closes full loop but with FORBIDDEN ingredients (see §7) | **GOALPOST only**, not code |

### 3.1 The convergent gap and the never-built combination

- **ENTRY** exists (E1 q_I; also E2 z). **CONTAINMENT** exists (C1 S_G). **EXIT is the universal gap.**
- Every failed exit is wrong-actuator (X1 uniform, X6 STD, X7 shunt, X8-actuator) **or** wrong-sensor
  (X2/X4/X8-actuator-context: instantaneous/spike-count/uniform → prevent ordinary interictal events).
- The **never-built combination**, which every doc independently points to:
  a **persistence/recruitment-gated sensor** (X8 proves the globality gate is buildable; X2 diagnosis says
  the missing ingredient is exactly "accumulate only after established recruitment, not during short
  events") driving a **spatially-local outward recovery current** (X2's actuator is the right *type* and
  *direction*; only its sensor was wrong; X1/X8 rule out uniform/global spatial support; X6/X7 rule out
  recurrent-gain / conductance-shunt levers) — tested **on the M4 bounded state** (C1), which is the first
  genuinely *sustained* state (feeds the sensor) and the first real state that *needs* terminating.

---

## 4. New mechanism vs old (explicit differences)

**New mechanism = persistence-gated local outward recovery current on the M4 bounded state.**

| axis | old failures | this line |
|---|---|---|
| actuator lever | recurrent-gain (STD X6), conductance shunt (X7 — starves S_G), uniform/global subtractive (X1/X8) | **local outward current on E membrane** (X2 type, un-refuted), never recurrent-gain / shunt / uniform |
| spatial support | uniform scalar (X1/X8) | **per-neuron spatial field** `p(x,t)` |
| sensor | instantaneous activity (X2/g_K), spike count from baseline (X4/m), instantaneous globality (X8) | **persistence-gated**: slow leaky integral of supra-threshold local activity — stays ~0 over short IEDs, saturates only under sustained recruitment |
| substrate under test | short ~55ms events (X2 never accumulated) / pre-M4 all-or-none avalanche (X8 nothing bounded to terminate) | **M4 bounded sustained state** (C1) — first sustained input to the sensor, first real bounded state to exit |
| closure | discrete latch + fixed bath + prescribed schedule + manual q refill (R4) | **continuous dynamics, zero-input spontaneous, no latch/bath/schedule/refill** |

This is not renaming m/g_K/shunt: it changes the actuator's **spatial support + driving sensor** to the
one combination the lineage left un-run, and tests it where a bounded state actually exists.

---

## 5. Equations, units, timescales (the added coordinate)

Baseline SNN (unchanged, §A of `snn_core_model_equations.md`): E/I LIF, `V_θ/V_r = 18/11 mV`,
E→E rotated anisotropic kernel (`θ_EE=45°`, AR=2, `ℓ_EE=0.380mm`), twoend_equal cores
(`μ_core=17.5, σ_core=1.0, R=1.5`), `L=20`, `N≈40000` (`NE≈32000`), `G=3.6`, `drive=0.6`, `dt=0.1ms`,
KICK_BOOST=0 spontaneous.

M4 layer (unchanged, reused as-is): `I_net,i = I_E,i·[recurrent ÷ (1+α_G S_G)] − q_I(x_i,t)·I_I,i` on E,
with `k_q=0.10`, `α_G=16`, plus the fixed pool/field params from `run_m4_dynamic_qi.py`
(`τ_q=5000, τ_a=20, σ_q=1.5, q_min=0.05; r50_psi=0.4, n_psi=2, p_pool=3, τ_mu=30, τ_S=80, S_max=1`).

**New persistence field `p(x,t)` (added; off-by-default):**

Spatially-smoothed E rate field (reuse existing `r_E(x,t)` from SpatialSlowField):
`a_p(x,t) = K_p * r_E(x,t)` (isotropic Gaussian width `σ_p`).

Persistence sensor (slow leaky integrator of supra-threshold activity):
```
τ_p · ∂_t p(x,t) = Ψ( a_p(x,t) − θ_p ) − p(x,t) ,      0 ≤ p ≤ 1
Ψ(u) = [u]_+ / ([u]_+ + a50_p)      (saturating rectifier; a single spike cannot fill p)
```
Duration selectivity: for a brief IED (supra-θ_p for ~`t_IED`≈55ms ≪ `τ_p`), `p` charges to only
`~(t_IED/τ_p)·Ψ_max` (small); for the sustained bounded state (supra-θ_p for seconds), `p → Ψ_max`.
The separation is set by `τ_p` relative to (IED-duration vs bounded-state-duration) and by `θ_p`; both
are calibrated from Stage-1 measured distributions, **not** tuned to the phenotype.

Local outward recovery current (the actuator; E cells only), gated by `p`:
```
I_net,i  −=  η_r · Φ(p_i) ,      Φ(p) = p   (primary)  |  p^{n_r}/(p^{n_r}+p50^{n_r})  (Hill, arm)
```
Same *type* as g_K (`−η_K·g_K`), but driven by persistence, spatially local, decaying with `τ_p` (its
decay after termination IS the refractory-wake / recovery timescale: `p` stays high early → low
early-retrigger; decays over `τ_p` → late susceptibility returns).

Optional extent gate (pre-registered arm, default OFF): multiply the actuator by a global recruitment
scalar `A(t)` built from the existing h_G globality sensors (`global_M/B/participation → χ_G`), so the
current fires only under **broad** sustained recruitment: `I_net,i −= η_r·Φ(p_i)·A(t)`.

**Off-by-default byte-parity**: `use_persist=False` (or `η_r=0` and no `p` allocation) ⇒ `apply_currents`
and `step` are byte-identical to the current SpatialSlowField ⇒ `BASELINE_SHA` (`da5fc18c27d5340a`) and the
existing `slow=None` parity gate are preserved. All new state lives in `slow_field.py` (NOT a guarded
engine file — `engine_versions.json` pins only kick_probe/params/model/connectivity/connectivity_rot/lfp),
so **no engine re-bless is required**; `kick_probe.py` is not edited.

**Units**: `p` dimensionless [0,1]; `θ_p, a_p` in rate-field units (same as `r_E`); `η_r` in mV (membrane
current units, comparable to the `18−11=7mV` reset→threshold span and to `η_K`); `τ_p` in ms.

---

## 6. State / readout schema

**Per-run summary JSON** (strict JSON, `allow_nan=False`; non-finite → null; NO N×T float dumps):
`arm, seed, k_q, alpha_G, use_persist, tau_p, theta_p, a50_p, eta_r, sigma_p, extent_gate,
T_ms, base_sha, engine_versions, argv, verdict, runaway_ms, max_rate_hz,
q_min_final, q_mean_final, S_G_max, p_max, p_mean_final,
n_interictal_pre, n_ictal_bursts, ictal_duration_ms, terminate_class, retrigger_verdict,
active_area_peak, active_area_tail, tail_frac_gt_0p5, axis_score_on/off, recovered_returning_events`.

**Low-dim state coordinate `X(t)`** (online summaries, streamed — no full state matrix):
`[R_core, R_axis, R_offaxis, R_I, active_area, source_onset_gradient, axis_alignment,
perpendicular_spread, low_k_globality, S_G, q_core/q_surround, p_mean, p_max]`.

**Checkpoints**: only a few registered per-neuron snapshots (V, ref, fields at branch times); continuous
readouts via online binning. Raw traces (`*_traces.npz`) allowed as gitignored artifact; summaries +
figures + metadata + README committed.

**Termination / retrigger classification**: reuse `src.sef_hfo_m4_termination.classify_termination`
(6-class) + `run_cell_with_retrigger` (two-pass same-seed, pre-window identity) so results are directly
comparable to M4-2 / M4-3A. (§6.1 helper-reuse discipline: the classifier's question — "did the bounded
state terminate cleanly and stay down, or fragment/suppress/persist/runaway" — matches ours exactly.)

**Source-space + virtual SEEG**: `per_neuron_onset` (onset gradient), `event_field_geometry`
(ext/r95/reach_axis/edge_margin), `active_fraction`, `LFPRecorder` montage (shaft A ∥ axis, shaft B ⊥,
PITCH=4mm). Direction via `endpoint_centroid_axis` (k_dir=3, part_min=7).

---

## 7. Forbidden ingredients (from R4 — the goalpost, not the recipe)

The SNN line must produce entry → bounded burst → finite exit → reset → same-basin recovery with **none**
of R4's shortcuts:
1. **No discrete hysteretic latch** (`L:1→0` operational bit). Exit must be continuous dynamics.
2. **No fixed core/annulus/bath patches** — spatial order must self-organize on the free SNN, not be imposed.
3. **No response-scheduled / prescribed state switch** — onset must be zero-input spontaneous.
4. **No active-low M freeze** (`dm/dt=0` while latched).
5. **No manual / analytic q refill or reset.**

First version also forbids (task brief §6): discrete latch, fixed-bath, active-low freeze, manual refill,
copying R4 params into the SNN, changing E→E connectivity, large blind grids.

---

## 8. Pre-registered arms (Stage 2)

Params sourced from Stage-1 measured distributions (slow-off IED rate/duration/extent; M4 bounded-state
rate/duration/extent; frozen-state exit displacement), NOT tuned to the final phenotype.

- **A — slow-off baseline**: `use_persist=F, use_qI=F, use_SG=F`. Must reproduce the returning-IED baseline
  (byte-parity to accepted baseline).
- **B — M4 containment anchor**: `use_qI=T (k_q=0.10), use_SG=T (α_G=16), use_persist=F`. Must reproduce the
  bounded sustained state.
- **C — sensor on, actuator clamped off**: B + `p` evolving but `η_r=0`. Verifies `p` stays ~0 during IEDs
  and rises on the bounded state, WITHOUT affecting dynamics (parity to B: `η_r=0` ⇒ no coupling).
- **D — full mechanism**: B + persistence-gated recovery actuator `η_r>0` (Stage-1 calibrated). The
  lifecycle test.
- **E — ablations**:
  - E1 `q_I` off → entry disappears? (accumulation gone)
  - E2 `S_G` off → runaway returns? (containment gone)
  - E3 sensor/actuator off → back to bounded-persist? (no termination)
  - E4 clamp `p` / `η_r` constant (open-loop) → dynamic exit vs constant prevention
  - E5 **matched instantaneous actuator** (replace `p` with instantaneous activity of equal integrated
    strength) → does duration-selectivity matter? (should fragment / prevent IEDs like X2/X4)

**≥2/3 primary seeds (1/3/4) same label** required for any cross-seed statement.

---

## 9. Acceptance / no-go (report per level, not one bit)

**Completion levels reported separately**: engineering-green | fast-state existence | dynamic accessibility
| termination | recovery | spatial pattern | virtual SEEG | cross-seed.

**PASS (full lifecycle candidate) requires ALL**: slow-off reproduces baseline; ≥ several returning IEDs
before onset; onset not externally kicked; bounded ictal, no numerical ceiling / no early-stop runaway;
multiple autonomous bursts or sustained oscillation (not flat plateau); spatially local entry + progressive
recruitment; interpretable containment + termination; no-reset return to the same basin; low early-retrigger
+ recovered late susceptibility; returning IEDs after recovery; ≥2/3 seeds; ablations consistent with
entry/containment/exit division of labor; dt/seed/param small-sensitivity label-stable; same trajectory
→ source-space AND virtual-SEEG readout.

**Not success**: truncated runaway; high plateau; external-pulse-off natural decay; prevention; fragment;
rate-model-only; seed1-only; fixed-bath containment; latch closure; manual q/M reset; pretty electrodes but
source space fails.

**Clean no-go (bounded-negative) if**: Stage-1 exit-boundary probe shows NO clean-exit-and-stay window for
any (actuator type, magnitude, hold-duration) reachable displacement → stop, do not build/scan the dynamic
field. Name which dynamical object is missing.

**Highest-risk criterion (flagged now)**: the M4 bounded state is a **broad ~60% stripe, symmetry-broken,
NOT a localized ictal core** (M4 Pass-1). So Stage-4 "localized onset + progressive recruitment" is a
substrate property an exit coordinate does not fix. Treat SPATIAL organization as a separate gate that may
return bounded-negative even if the TEMPORAL lifecycle (entry→bounded→exit→recovery) succeeds. Do not
inflate a temporal-only success into a spatial claim.

---

## 10. CPU / RAM / OOM budget

Machine at lock: 80 logical / 40 physical cores, 251 GiB RAM (~232–244 GiB available), swap 2 GiB
(691 MiB used), load ~3.3. FCXR Stage-D grid running (~20 GB RSS, ~2 cores) — protected.

- Every worker: `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMEXPR_NUM_THREADS=1`.
- Canary: one real L=20 M4 SNN at (k_q=0.10, α_G=16), `/usr/bin/time -v` → peak RSS, wall, CPU, output size.
- `workers = min( floor(0.65·MemAvail / (1.25·canary_RSS)), physical_cores − ≥4 reserved, 8 )`; **first
  batch ≤2 workers**, ramp only after RSS/COW/swap confirmed sane.
- `resource_monitor.py` → `resource_log.jsonl` every 30–60s (ts, loadavg, MemAvailable, swap, per-PID
  RSS/CPU for THIS line's PID manifest). Stop launching new work if MemAvailable < 30% total; gracefully
  kill THIS line's newest worker if < 20% or swap grows. Only kill PIDs in this line's manifest — never
  global `pkill python`. Reserve ≥25–30% RAM and ≥4 cores for others.
- Per-cell checkpoint + failure JSON (no silent dropped cells). Stop launching new long tasks by ~9.25h.

---

## 11. Figure contract

- **If PASS** → `results/paper-ready-figure/fig5_snn_interictal_ictal_recovery/figures/` (PNG+PDF+metadata
  JSON + Chinese `figures/README.md`; script in `scripts/paper_figures/`). Main lifecycle figure (one
  continuous trajectory): A mechanism/substrate (E→E axis, cores, virtual electrodes, where q_I/S_G/p act —
  no fixed bath drawn); B continuous virtual SEEG (returning IEDs → onset → ictal bursts → termination →
  recovered IEDs marked; fixed montage/order; no spliced segments); C source-space axis×time + off-axis/area;
  D slow variables (q_I, S_G, p, actuator) with entry/containment/exit/recovery times; E representative
  spatial frames (baseline IED, pre-onset, onset, established burst, termination, recovered IED). Plus
  optional classic 4-col (mechanism | tempA source | tempB source | electrode readout) to show recovered
  template preservation. Style: plasma mechanism, viridis events (early purple→late yellow), orange
  (axis shaft A) / cyan (transverse shaft B) electrodes; shading stops at runaway onset if any.
- **If NO-GO** → `results/topic4_sef_hfo/m4_snn_native_exit/figures/`, marked diagnostic; draw the ACTUAL
  failure type; no cherry-picked pretty segment.

---

## 12. Engineering / reproducibility contract

- Slow mechanism OFF ⇒ byte-identical to accepted baseline (BASELINE_SHA gate + own parity test).
- No acceptance-threshold tuning to rescue results.
- Record seed, config, base SHA, engine sha256 (`record_versions`), argv, artifact inputs in every JSON.
- All JSON strict (`allow_nan=False`). No full N×T float state saved. Continuous readouts via online binning.
- Engine changes confined to `slow_field.py` (unguarded) → no re-bless; add own TDD (`hfosp-deep-contract-verify`
  before writing the `p`-field body — multi-clause invariants: off-by-default parity, σ_p footprint,
  duration selectivity, extent-gate, S_G-starvation guard). New runner should call `_engine_guard()`.
- Commits (logical): (1) docs spec+ledger; (2) feat persistence field + fork/probe machinery + tests;
  (3) feat exit atlas + dynamic arms + diagnostics; (4) docs verdict + archive report.
- Do not commit other worktrees' dirty state.

---

## 13. Stage plan (cheap-first)

- **Stage 1a (zero new code)**: reproduce bounded M4 state at (0.10,16); reuse `perturb=inhibitory_pulse`
  (clamped V_th displacement = threshold actuator) to sweep (ΔV_th magnitude × hold-duration × spatial
  mask) → classify exit/hold/reignite/runaway/suppress via `classify_termination`. Also measure IED vs
  bounded-state duration/rate/extent distributions. **Gate**: does any clean-exit-and-stay window exist?
- **Stage 1b**: implement `p` field (slow_field.py, off-by-default, TDD) + a clamp mode for the
  current-actuator open-loop probe; repeat the exit-boundary atlas for the current actuator (and qI-refill
  inhibition actuator as control). Confirm which actuator type + minimal (η_r, τ_p-hold) exits without
  S_G-starvation runaway.
- **Stage 2**: arms A–E (dynamic), params from Stage-1. seed1 first.
- **Stage 3**: spontaneous full trajectory, KICK_BOOST=0; seed1 canary → seeds 1/3/4.
- **Stage 4**: source-space + virtual-SEEG spatial gate + Figure 5 (honest, pass or no-go).

Re-read this spec at each stage boundary. Preliminary numbers → archive tagged "pending sensitivity";
main-doc framing only after all gates pass.
