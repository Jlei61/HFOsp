# Topic 4 — State-conditioned spatial susceptibility along one MZ z-depletion trajectory (2026-07-19)

> Exploratory **model-side** mechanism/readout diagnostic. NOT a seizure, termination, or patient claim.
> Base: `codex/topic4-mz-slowvars` @ `66a4d93`. Design/plan:
> `docs/superpowers/specs|plans/2026-07-19-topic4-state-conditioned-spatial-susceptibility-*.md`.
> Candidate `zA_q50_tz10000` (z-only, m≡0), seeds 1/3/4. Autonomous overnight run.

## Abstract (plain-language)

**测了什么**：一个 3.2 万兴奋细胞、固定带方向骨架的仿真皮层片；戳一下沿一条固定横轴短暂传一小段再停。让每个细胞的抑制刹车效力沿一条固定轨迹慢慢失效直到失控，问：走向失控前，这张固定骨架对空间扰动的短时放大能力怎么变。**怎么测的**：重放轨迹，在五个时刻抓每个细胞的刹车效力铺成粗地图，喂进已有率场算子求工作点、线性化，用不同波长/朝向探针量 30 毫秒内被放大多少；再和"抹平/旋转/打乱/不失效/去骨架方向"五种对照比。**揭示了什么**：随失效逼近失控，短时放大从约 0.2 涨到约 0.75、偏好尺度掉到沿轴大尺度、主本征模式从全局转沿轴，三种子一致；放大强度由失效的空间图案决定（真实图案≈抹平的两倍），而"沿轴略强于垂直"这点方向偏好主要由各向异性骨架给（旋转/打乱不变、去各向异性减半）。**刹车失效是在放大一条本已存在的骨架轴，不是创造或转动它。**

（内部归档代号：M4-MZ per-neuron `z_i` 轨迹；`zA_q50_tz10000` seeds 1/3/4；M3B finite-Jacobian 率场算子 + `rate_eigenpairs` + 非正规有限时响应 `C exp(J T) B`；Gabor/Fourier probe dictionary；fixed backdrop `mu_core=0.6, w_ee_mult=1.05, ratio=1.0, ell_perp=0.6, ar=2, theta=0`。）

## 1. Question and locked inputs

As inhibitory efficacy `z_i` (phenomenological, E-cells only) evolves along ONE fixed pre-computed MZ
trajectory toward runoff, does the finite-time spatial susceptibility of the fixed E1146 scaffold change
in a way that preserves/strengthens the interictal propagation axis before runoff? No result direction is
a success gate (design §0, §11).

Locked from committed MZ artifacts (`results/topic4_sef_hfo/mz_slowvars/per_seed/multiseed_summary.json`):
candidate cfg `{use_z:true, use_m:false, I_th_EI:1.6652801609959704, tau_z:10000.0}`; runaway onsets
seed1=4937.0, seed3=4706.8, seed4=4861.5 ms; dt=0.1; subject epilepsiae_1146, montage narrow; NE=32000.
The 6 guarded engine SHAs match the multiseed provenance exactly → replay reproduces onsets to the ms.

## 2. Method (reuse, not reinvent)

1. **Snapshot observer** (`src/snn_engine/mz_slow_vars.py`, off-by-default, 6 guarded engine files
   untouched): copies `z_E`/`m_E` at 5 pre-declared integer steps `round(t/dt)` AFTER the slow update;
   `snapshot.z_E.mean() == trace_z_mean[step]` pins the index/time. States: baseline_1000ms,
   mid_fraction=0.5·onset, pre_onset_500ms, pre_onset_100ms, onset.
2. **Replay** (`scripts/run_topic4_state_conditioned_susceptibility.py capture-snapshots`): mirrors the
   locked `run_mz_cell` (same substrate, noise seed, spontaneous no-kick), adding the observer; gate =
   onset within 5 ms + all-5-states + z∈[0,1] + m≡0 + I-cells pinned. All 3 seeds passed; onsets exact.
3. **Coarse mapping** (`src/topic4_state_conditioned_susceptibility.py`, pure): affine E1146
   [0,20]²→normalized L=5.0 square (axis horizontal → theta=0); bin z_E onto n=12 grid (occupancy 100%);
   `q(x)=clip(z_bar, 0.05, 1)` = the M3B inhibition-efficacy field (z_min_realized≈0.76 so q_floor never
   binds). The MZ z-depletion thus traverses M3B's disinhibition (q) axis, spatially resolved.
4. **Operator** (reuse `topic4_m3b_spectral_phase`): fixed anisotropic kernels (ell_perp=0.6, ar=2,
   theta=0) + fixed two-core excitability (`mu_core`) at the transformed src/snk; per-state
   `solve_operating_point` → `build_jacobian_dense` (6N=864-dim); `rate_eigenpairs` for the true
   eigenmode; phase-paired Gabor probe atlas via `expm_multiply` for `R_s(T)=C exp(J_s T)B`, C=E-rate.
5. **Controls** (design §7): real / uniform_mean / rotated_90 / spatial_shuffle / z_blocked at every
   state; AR1 (isotropic) and n8/n12 resolution at baseline + pre_onset_100ms.

## 3. Fixed-backdrop margin calibration (transparency)

The M3B single-core phase-map anchor (`mu_core=0.8, w_ee_mult=1.3`) puts the TWO-core surrogate deep in
the runaway corner once q reaches ~0.9, so it **saturates at `mid` (~2468 ms)** — inconsistent with the
SNN, which is interictal until onset (~4937 ms). A resolved-fraction sweep (op-STATUS only, before any
susceptibility was read) shows `w_ee_mult` is the margin knob (`mu_core` barely matters):

| w_ee_mult | resolved states (of 5), seeds 1/3 |
|---|---|
| 1.30 (M3B anchor) | 1/5 (baseline only) |
| 1.15–1.20 | 2/5 |
| 1.10 | 3/5 |
| **1.05 (chosen)** | **4/5 (baseline→pre_onset_100ms; onset = boundary)** |

Chosen `w_ee_mult=1.05, mu_core=0.6`: the surrogate stays resolved across the whole pre-onset trajectory
and hits the boundary right AT onset — matching the SNN's interictal-until-onset margin. This is a
numerical-margin calibration on op status only (the absolute operating point of a coarse rate-field
surrogate is a free normalization), **not** tuning against a susceptibility score (design §2). Anchor
values retained in `config/topic4_state_conditioned_susceptibility.yaml` comments.

## 4. Results

### 4.1 Susceptibility trajectory (T=30 ms, per seed; onset = unresolved boundary, fail-closed)

| state | axial | perp | global | peak_k | orient | eig growth | eig glob | eig axis |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.19 | 0.20 | 0.17 | 7.11 | 45/135° | −0.049 | 0.99 | 0.05 |
| mid | 0.16 | 0.19 | 0.19 | 7.11 | 135° | −0.046 | 0.87 | 0.19 |
| pre_onset_500ms | 0.47/0.54 | 0.39/0.44 | 0.44/0.50 | 1.26 | 0° | −0.021 | 0.15 | 0.90 |
| pre_onset_100ms | 0.75/0.80 | 0.66/0.69 | 0.75/0.79 | 1.26 | 0° | −0.011 | 0.16 | 0.88 |
| onset | — unresolved/saturated (fail-closed) — |

Within-seed Δ (baseline→pre_onset_100ms, 3-seed median): axial +0.569, perp +0.468, **global +0.585**,
axis−perp **+0.106** (0.106/0.119/0.102), peak_k **−5.85** (7.11→1.26, all seeds). Highly consistent.

### 4.2 Controls (pre_onset_100ms, 3-seed median)

real axial 0.756 / ax−perp +0.088; **uniform_mean 0.363 / +0.056** (real ≈2× → spatial pattern sets
magnitude); rotated_90 0.436 / **+0.089**, spatial_shuffle 0.405 / **+0.083** (magnitude drops, axial
margin unchanged → direction not from pattern orientation); z_blocked 0.184 / −0.007 (no depletion → no
amplification/direction); AR1 isotropic ax−perp **+0.037** median (halved vs +0.088 → axial direction
mostly from scaffold anisotropy; small residual from two-core-along-axis geometry). Resolution: n8 vs n12
at pre_onset_100ms axial 0.715 vs 0.747, ax−perp +0.089 vs +0.092 (resolution-robust). Nonlinear
two-amplitude linear-regime check PASS at baseline (J is linear; confirms embedding/readout linearity).

## 5. Answers to the ten Task-8 questions

1. **Replay w/o observer drift?** Yes — byte-parity proven on the real substrate; onsets exact to the ms.
2. **z depletion patterned or uniform?** Patterned: a horizontal band concentrated in the core corridor
   (z_mean 0.98→0.89, z_min→0.76); its on-axis concentration is what drives the magnitude (real≫uniform).
3. **Axial/perp/global gains within seed?** All rise (~0.2→0.75); global≥axial>perp; consistent 3 seeds.
4. **Orientation stable/rotate/unresolved?** Preferred orientation is axial (0°) once gain develops
   (pre-onset), stable across pre-onset states and seeds; baseline "peak" is high-k but gains are
   near-flat there (peak ill-defined).
5. **Scale toward lower k before onset?** Yes — peak_k 7.11→1.26 (all seeds).
6. **real vs uniform/rotate/shuffle/z-blocked?** Yes in magnitude (real 2× uniform); axial margin similar
   for rotate/shuffle (scaffold-set); z-blocked flat.
7. **Consistent across seeds?** Yes — deltas within a few %.
8. **True eigen-subspace vs non-normal optimal response — same story?** Consistent near the boundary
   (both axial) but DISTINCT objects: at baseline the leading eigenmode is global (0.99) while the
   transient is near-isotropic low-gain; as z depletes the eigenmode ITSELF rotates to axial (globality
   0.99→0.15, axis 0.05→0.9, growth→0) and the transient amplifies along the axis. NOT "the global
   eigenmode is axial" — measured directly on the eigenvector.
9. **Optional stages completed/failed/not_run?** Completed: capture, atlas, 5 controls, AR1, resolution,
   nonlinear, both figures. not_run: P4 read-only early-readout comparison. failed(environmental): 7 M3B
   output-artifact tests (see §7).
10. **Safest claim / largest gap / next step?** See §6 + STATUS.

## 6. Interpretation contract (design §11)

**Safest claim**: In this coarse rate-field surrogate of the E1146 scaffold, as the phenomenological
inhibitory-efficacy field depletes along one fixed MZ trajectory toward runoff, the finite-time spatial
susceptibility rises sharply and its preferred scale collapses to the largest along-axis mode,
consistently across all three locked seeds; the amplification is predominantly **global** with a small,
reproducible **axial-over-perpendicular** margin (~+0.09) set by the fixed anisotropic scaffold (invariant
to rotating/shuffling the depletion field; halved for an isotropic scaffold), while the amplification
**magnitude** depends on the spatial depletion pattern (real ≈2× uniform-mean). The interictal axis is
**preserved and its gain strengthened before runoff**, but the axis is a fixed scaffold property amplified
by depletion, not created or rotated by it. Result-neutral descriptions that co-apply (design §11):
**same-axis gain increase** + **global/uniform amplification** + **spatial pattern adds effect beyond mean z**.

**Forbidden wording avoided**: not a seizure/ictal event; `z_i` is a phenomenological inhibitory-efficacy
variable (not proven chloride/GABA failure); Gabor/Fourier objects are probes (not eigenmodes); the leading
global eigenmode is NOT called axial (the eigenmode itself is measured to rotate); no single-seed bridge; no
termination/full-cycle/patient claim; unresolved/saturated states never labelled stable/axial.

## 7. Tests, failures, reproducibility

Targeted: `test_mz_slow_vars` 24 ✓, `test_topic4_mz_slowvars` 18 ✓,
`test_topic4_state_conditioned_susceptibility` 12 ✓ (Gate C/D), `test_topic4_m3b_spectral_phase`
**81 ✓ / 7 ✗**. The 7 failures are ALL `FileNotFoundError` on the git-ignored build-artifact dir
`results/topic4_sef_hfo/m3b_spectral_phase_map/` (STATUS/verdict/figure existence contracts); they passed
at Gate A and the dir was removed since by a **co-active session sharing this worktree**
(`topic4_mz_early_field_bridge` / `mz-onset-dynamics` files appeared mid-run). Unrelated to this change —
the 81 M3B logic tests validating the reused operator all pass. Regenerate with
`python scripts/build_m3b_spectral_outputs.py`.

Reproducibility: engine SHAs match locked MZ provenance; every JSON carries schema/upstream-paths/git+engine
SHA/config/candidate-seed-state lists. Artifacts under
`results/topic4_sef_hfo/state_conditioned_susceptibility/` (snapshots/, susceptibility_atlas.json,
susceptibility_arrays.npz, control_summary.json, numerical_audit.json, nonlinear_spotcheck_summary.json,
snapshot_contract.json, STATUS.md, figures/{diagnostic,controls}.{png,pdf} + README.md).

## 8. Largest gap and next single experiment

Gap: coarse rate-field surrogate; the fixed backdrop margin was recalibrated (`w_ee_mult` 1.3→1.05) so the
surrogate stays resolved through pre-onset and hits the boundary at onset (matching the SNN margin) — the
absolute operating point is a free normalization; the axial-strengthening prediction is **not yet confirmed
in the spiking network**. Next: drive the actual SNN with a source-core kick at the captured pre-onset slow
state (vs baseline) and read the finite-time E-field response through the virtual-SEEG plane, testing
whether along-axis gain rises as the surrogate predicts — bridging surrogate → spiking network.
