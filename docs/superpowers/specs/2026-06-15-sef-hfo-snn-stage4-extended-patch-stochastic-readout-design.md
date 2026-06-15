# Stage 4 — Extended single-patch stochastic readout (design / mini-spec)

- **Date**: 2026-06-15
- **Topic**: Topic 4 SEF-HFO SNN
- **Status**: design (pre-implementation). Forks locked with user 2026-06-15; awaiting spec review before writing-plans.
- **Branch note**: Stage 3 detail-mechanism work is ongoing in parallel on `topic4-sef-hfo-snn-stage3`. Stage 4 implementation should run on its own branch / worktree to stay isolated from Stage 3 (see §10).

---

## 0. Abstract（朴素话，§8）

**测了什么。** 我们在模型里放**一个大的、容易点着的病理区**，这个区大到本身就横跨 4–5 个虚拟电极。区**内部**阈值低（容易点火），外面阈值高（点不着）。背景噪声让这个大区**每次在内部不同的小角落先点着**。我们问：透过**和真实病人一样的那套 SEEG 读出流程**去看，这些"内部哪里先点着是随机的"事件，读出来的**传播方向**会不会是"随机但有规律"的——也就是既不是每次都一个方向、也不是乱成一团，而是**哪端先点着就读成哪个方向**。

**怎么测的。** 我们对比两件事。第一，**仿真里我们确切知道每次事件在大区内部哪里先点着**（地面真值），再看电极读出的方向是不是能被这个"先点着的位置"预测出来——如果读出忠实，二者应当强相关。第二，把一整段记录里每次事件的方向收集成一个分布，看它是不是"散开且有结构"（同时出现正向和反向、但都贴着那条固定的传播轴），而不是退化成一个点、也不是均匀乱撒。**关键防混淆对照有两个**：(1) 把灶缩成一个点（成核位置基本固定，**而且要把它调到能读出**——抬高 drive / 用诱发 / 只挑全局事件），读出方向就该一致、不散；一个**能读**的点灶若也方向乱飞，那就是电极采样把信号糊乱了，不是灶内机制。(2) 把每次事件的"真实成核位置"和"读出方向"随机打乱配对，真实的对应关系应当消失——消失才说明读出是忠实跟着成核走的。至于"内部阈值不均"本身贡献多少，用均质 vs 不均质（发放率配平）单独看——这是**次要**的机制旁证，不是防混淆对照，也不是主结论。

**预期揭示什么（口径保守）。** 如果成立，我们只敢说：**在固定的各向异性传播几何下，一个横跨多电极的易激大区，能产生"随机但忠实反映内部成核位置"的电极级方向读出**——这恰好能解释真实数据里同一个被试为什么会出现一对正/反模板（不需要两个独立小灶轮流点火）。**不会**写"大区解释了真实病人的发病机制"，也**不会**写"异质性本身导致了方向模板"。

（内部归档代号：Stage 4 = extended single-patch stochastic readout；复用 Stage 2 instrument-compatibility / Stage 3 sidecar+collision 资产；readout = `endpoint_centroid_axis` + masked lagPat pipeline；连接轴 = `theta_EE`/`AR`；异质性 = `core_mean`/`core_std`/`core_r`；防混淆控制 = `C-point-readable` anti-blur + within-event shuffle；`C-homog` = rate-matched (`mean_match_vth`) mechanism-modifier；几何 = L≈32 near-paper-N target + tail-bounded spatial-bin/KDTree feasibility gate。）

---

## 1. Problem statement (frozen)

> 一个覆盖多个电极的易激大灶，在灶内随机成核时，是否会通过**同一真实 SEEG readout** 产生**随机但有结构**的方向 / rank 模板？（≠ 两个独立小灶稳定轮流点火）

This is **not** a continuation of the dual-focus ("双灶") line. It is a single extended excitable patch.

### 1.1 Why this design (motivation from the other lines, 2026-06-15 survey)

- **Stage 2 closed the *structure / instrument* layer only.** Single-end lesions, run separately then pooled, are read by the real masked pipeline into `stable_k=2` forward/reverse templates with endpoint swap. That proves *pipeline compatibility*, not mechanism.
- **Stage 3 is not a pass.** Two equal foci in one network: nucleation at both ends happens, but **most events stay local (contained / relay-failure); only ~9% become readable global** events. The primary label-timing-independence question was never truly testable. Root cause = a geometric dead-lock at L=20: "readable by electrodes (≥7 contacts) ≈ sheet-wide ≈ coupled to the other focus → collision."
- **The extended patch relaxes that dead-lock.** Because the electrode array sits *on top of* the large focus, intra-patch nucleation has a **much higher probability of being sampled by ≥7 contacts** than Stage 3's two point foci — relaxing, **not eliminating**, the read-out bottleneck. Stage 3's lesson stands: even strong nucleation can stay contained/unreadable, so "readable" is **never assumed** — it is a Phase 1 gate (§8). And there is no second focus to collide with, so the one-core-dominance + collision traps are gone by construction.
- **SNN heterogeneity line result constrains the claim.** Mean excitability gates ignition; **heterogeneity *width* alone is not a strong mechanism conclusion**, and prior core-size scans used *small* cores only. So Stage 4 treats `core_std` as a secondary modulator, never the headline.
- **Topic 5 real-data direction supports the framing.** The replay line pivoted to a *network-axis* reading: early-ictal and interictal propagation share a coarse common axis, not point-by-point replay. Stage 4's "fixed anisotropic axis + random intra-patch nucleation → variable readout" is the model-side analogue.

### 1.2 Locked forks (user 2026-06-15)

- **Geometry / scale**: grow the sheet to **L ≈ 32** for a realistic surround (real-ish ~4 mm pitch × 4–5 contacts ≈ 16–20 mm focus + surround). Phase 0 must make this practical (§5).
- **Primary endpoint**: **two co-primary** — (A) nucleation-position → readout-direction correspondence, AND (B) readout-direction distribution is structured. Real masked-pipeline `stable_k` is **secondary**.

---

## 2. Substrate & geometry

- Reuse the validated 2D anisotropic LIF E–I engine (now git-tracked at `src/snn_engine/`, moved out of the gitignored `results/` tree 2026-06-15 — see `refactor(snn-engine)` commit) and the spontaneous runner `scripts/run_sef_hfo_snn_cm_spontaneous_readout.py`.
- **Sheet**: `--L ≈ 32`, `--density 100` → **102,400 neurons** (≈1.0e5; near the engine's *paper-N*, but not paper biological scale). That is **2.56× the L=20/d100 ≈ 40k** smoke regime — build-time sensitivity is exactly why Phase 0 is gated (§8). Connectivity axis `--theta`, anisotropy `--AR 2.0` (the *fixed* propagation geometry; never randomized in the main runs).
- **Extended patch** = a single low-`V_th` disk via a new `extended_patch` branch in `build_lesion_vth`, parameterized by `--core-mean` (mean threshold, the ignitability dial), `--core-std` (intra-patch dispersion), `--core-r` (patch radius). **Geometry math** (4 mm pitch): to span *5* in-patch contacts need `core_r ≈ 8 mm` (patch Ø16 mm); for *4* contacts `core_r ≈ 6 mm` (patch Ø12 mm). Outside the patch: base threshold (18.0, no heterogeneity).
- **Virtual montage**: ≥2 non-parallel shafts (`spans_2d()` must hold). Prefer **odd `--nc`** so a center contact lands on the patch centre — `nc=7` at L=32/pitch 4 spans 24 mm (5 in-patch + 2 surround witnesses), leaving ≥~6–8 mm surround each side; `nc=9` only if pitch<4 or L>32 (else it hits the L=32 boundary). **Target = L≈32 for the 5-contact patch; L≈28 is the 4-contact fallback** if Phase 0 build-time forces it down. ≥7 participating contacts is the readability *floor* — a Phase 1 gate, not assumed.
- **Patch shape** is a pilot knob: default disk; an axis-elongated patch maximizes forward/reverse sign variability, a disk allows more angular spread. Swept in Phase 1, not a blocking decision.

---

## 3. The randomness model (two levels — state both, never collapse)

- **Quenched layer (the "hot-spot menu")**: the intra-patch threshold field (`core_std`) carves where the low-threshold sub-regions sit. Fixed once per network (per seed).
- **Annealed layer (event-to-event)**: background OU drive / synaptic noise makes **each spontaneous event nucleate at a different sub-region** within one recording. This is the source of the *within-recording* direction variability that maps to within-subject template variability in real data.
- **`core_std` is a secondary modulator, not the headline.** Per the locked口径, the claim is "extended patch + internal randomness → structured stochastic readout"; whether heterogeneity *width* is *necessary* is answered descriptively by the homogeneous-vs-heterogeneous control (§5 controls), not asserted as cause.

### 3.1 Predicted shape of the variability (interpretation anchor)

Because connectivity is anisotropic along a *fixed* axis (`theta_EE`), a wave propagates along that axis regardless of where it seeds. So intra-patch nucleation variability manifests primarily as **forward/reverse sign flips** (which end of the patch seeded first) **plus angular jitter** (transverse offset of the seed). This is exactly the real-data forward/reverse template pair — **reproduced from one focus**, no two foci required. Central / transverse nucleation may read as radial / unreadable (a legitimate third outcome — §4.3).

---

## 4. Observables, metrics & nulls

### 4.1 Per-event record (new + reused)

For each detected spontaneous event:

- **Ground-truth nucleation centroid** `(x_nuc, y_nuc)`, axis projection `s_nuc`, transverse offset `r_off`, and `n_early_cells` — **NEW** `nucleation_centroid()`, **robust** definition (a single early cell must not bias it): (1) find the bin where the **patch active-fraction first crosses** the onset threshold; (2) take the **robust centroid** (median / trimmed) of the earliest `K` fired patch-E cells (or all fired in the next 1–2 ms), *not* the first single spike; (3) project onto `axis_unit(theta_EE)` → `s_nuc`, and record the transverse component `r_off`. Record `n_early_cells`; **if too few, exclude the event from the correspondence test** (unstable centroid). Extends `build_sidecar` (reuse its event-windowing).
- **Readout** through the real virtual-SEEG path: recovered axis `θ_read`, **sign** `s ∈ {+1 forward, −1 reverse, None unreadable}` (`endpoint_centroid_axis`), `direction_readability`, `axis_err_deg`, `n_participants`.
- **Reach**: local vs global (`n_participants ≥ 7` readable), spread (#contacts), duration.
- **First-active contact** index.

### 4.2 Co-primary A — nucleation → readout correspondence (instrument fidelity)

- **Claim**: through the noisy real instrument, recovered direction/sign **tracks** ground-truth `s_nuc`. (The source-level mapping nucleation-end → wave-direction is near-tautological; the non-trivial test is whether *the readout preserves it* rather than blurring it away.)
- **Statistic (two-stage — do NOT condition the unreadable events away)**: **Stage 1** `P(readable | s_nuc, r_off)` — does nucleation position predict whether the event becomes readable at all (expectation: end-like `s_nuc` → readable; central / large-`r_off` → radial/unreadable). **Stage 2** among readable events, logistic AUC of `s_nuc → sign(s)` (+ circular-linear corr of `s_nuc → θ_read`). Reporting only the Stage-2 AUC on readable events would silently drop the central/radial events that are themselves part of the signal.
- **Null**: within-event **shuffle** — permute which `s_nuc` pairs with which readout; correspondence must collapse to chance. (Reuse Stage 3's block-aware shuffle discipline.)
- **Anti-blur backbone**: the within-event shuffle null (above) **+ the `C-point-readable` control**: a spatially-confined source, *made readable* (raise `--drive` / evoked kick / select global events), should give *low* readout dispersion and a consistent direction; if a *readable* point patch already disperses, that dispersion is instrument blur, not source. The **homogeneous large patch is NOT an anti-blur control** — it shares the identical nucleation→direction physics (the wave still runs along the fixed axis from wherever it seeds), so it would also show correspondence; it tests heterogeneity's own contribution instead (`C-homog`, mechanism-modifier).

### 4.3 Co-primary B — structured stochastic readout

- **Claim**: over a recording the readout-direction distribution is **dispersed** (not a single point) **and non-uniform / modal** (forward+reverse mix near the axis).
- **Statistics**: circular variance / entropy of `θ_read`; forward/reverse balance of `s`; first-contact entropy; readable (global) fraction.
- **Nulls**: (i) **single-point / degenerate** — dispersion must exceed the **`C-point-readable`** baseline (a spatially-confined *readable* source reads ≈one direction). **Not** the homogeneous patch — that is a mechanism-modifier, not a degeneracy null. (ii) **uniform-random** — distribution must be non-uniform (Rayleigh/Kuiper) and concentrated near `theta_EE`, not isotropic.
- **Unreadable events are a class, not a forced sign** — report the unreadable fraction; do not coerce radial events into ±.

### 4.4 Secondary — instrument-compatibility continuity (Stage 2)

Run model events through the **real masked lagPat pipeline** (`mask_phantom=True` — phantom-rank hard contract) and check they still cluster into `stable_k=2` forward/reverse templates. This is continuity with Stage 2, **secondary**, never the headline.

---

## 5. Controls (the falsifiability backbone)

| # | Control | What it rules out | Expected if Stage 4 holds |
|---|---|---|---|
| C-point-readable | small `core_r` point patch, made **readable** (raise `--drive` / evoked kick / select global events) | **sampling blur** (consistent source ⇒ consistent readout) — the anti-blur calibration | reads ≈one direction, low dispersion. If a *readable* point patch already disperses → blur flagged. **This is the single-point null for co-primary B.** |
| C-point-bottleneck | small `core_r` point patch, spontaneous | — (boundary phenomenon only) | read-out bottleneck + low readable fraction; **descriptive record, NOT an anti-blur gate** (an unreadable source can't calibrate blur). |
| C-extended | large `core_r` vs the point patches | that the phenomenon is generic | extended → readable + dispersed + high first-contact entropy |
| C-homog | homogeneous (`core_std≈0`) vs heterogeneous, **rate-matched** (`mean_match_vth`) | that intra-patch threshold structure (not just dynamic noise) shapes the nucleation distribution | heterogeneous → nucleation concentrates at low-threshold hot-spots (sharper modal structure); homogeneous → noise-driven diffuse nucleation. Both may disperse; this measures the *origin/shape*, not blur. **Mechanism-modifier (descriptive).** |
| C-rot | montage rotation invariance (C3 harness) | montage-geometry artifact | readout distribution invariant to electrode-array rotation |
| C-iso | isotropic connectivity `AR=1` (C4 harness) | readout fabricating direction from noise | must NOT read direction (`readability < 0.3`) |
| C-rate | event-rate matched across conditions | "more events → more apparent dispersion" | dispersion structure survives rate matching |
| C-seed | seed control (+ optional `--swap-vth`/`--mirror-vth`) | a single quenched-field accident | pattern holds across seeds / under field swap |

**Anti-blur backbone = `C-point-readable` + the within-event shuffle null** (§4.2): a spatially-confined *readable* source must read a consistent direction, and a faithful readout must track *this* event's true nucleation. These are numerically gated, not narrative. **`C-homog` is a mechanism-modifier, not a degeneracy null.** Co-primary B is tested against `C-point-readable` and uniform-random nulls; homogeneous large-patch runs test whether quenched threshold structure sharpens / discretizes the nucleation menu beyond annealed noise alone (and whether `core_std` is merely shape modulation, not a necessary cause).

---

## 6. Pre-registered tiers (fixed at planning time — do not upgrade from data strength)

- **Precondition (gate, not a claim)**: extended patch self-ignites reliably, no runaway, most events readable, enough events. [Phase 0/1 gates]
- **Co-primary A (two-stage, §4.2)**: among readable events, nucleation→sign correspondence beats the within-event shuffle null; and `C-point-readable` reads a consistent low-dispersion direction (anti-blur backbone). Stage-1 readability-vs-position reported alongside, not conditioned away.
- **Co-primary B**: readout-direction distribution is dispersed-and-non-uniform vs the `C-point-readable` single-point null + the uniform-random null.
- **Secondary**: model events still read as `stable_k=2` forward/reverse templates through the real masked pipeline.
- **Mechanism-modifier (descriptive, NOT headline)**: `core_std` modulates dispersion (C-homog contrast). Reported descriptively; **no "heterogeneity causes the templates" claim**.

---

## 7. Success criteria (conservative — locked with user)

- **May write**: "An extended excitable patch, under fixed anisotropic propagation geometry, generates **stochastic but structured** electrode-level direction readouts that **faithfully track intra-patch nucleation location** (instrument-fidelity), reproducing a forward/reverse template pair from a single focus."
- **May NOT write**: "the large patch explains real-patient mechanism"; "heterogeneity itself causes the direction templates"; or any cohort/clinical claim. This is a **model existence/screen** result, not a data-match.

---

## 8. Phases & hard-stop gates

- **Phase 0 — engine feasibility @ L≈32 + montage geometry (HARD STOP). No ensemble / no science runs here.** Memory is fine (243 GB free); the binding constraint is **build + 1 s sim wall-time** at 102,400 neurons (2.56× the L=20 smoke regime). Add a **tail-bounded** spatial-bin/KDTree candidate restriction to `build_connectivity_rot` — a finite prune radius truncates the exponential kernel's long tail, so it is an *approximation*, **not** an exact contract (do not claim "kernel unchanged"). **Equivalence gate** (all required before calling it *practically* contract-preserving): (1) **in-degree exact** = C_EE 800 ⇒ prune radius must hold ≥800 weighted E-candidates at d100, else widen it; (2) **partner-distance KS** naive-vs-pruned on a small net; (3) **delay quantile** match; (4) **realized axis covariance / AR** match; (5) **edge tail-mass bound** (analytic kernel mass beyond the prune radius); (6) **dynamics smoke** — a `oneend` known-source run reproduces the naive-sampler readout. Then montage geometry: place `extended_patch` + odd-`nc` montage; confirm ≥2 shafts + adequate surround. **Gate**: equivalence (1)–(6) pass AND build+1 s sim within a practical budget. (Readability of intra-patch nucleation is asserted nowhere here — that is a Phase 1 gate.)
- **Phase 1 — short pilot (HARD STOP).** 2–3 patch sizes (`core_r`) × small (`core_mean`, `core_std`) grid around the known-ignitable workpoint × 2–3 seeds × short `T`. **Hard-stops**: no runaway; ≥ N events/run; first-contact entropy non-degenerate (not all same contact first); readable fraction not ≈0. Decide go/no-go + pick workpoint(s) + patch shape.
- **Phase 2 — ensemble + controls.** At chosen workpoint(s), long-`T` (and/or many-seed) ensemble to build the readout-direction distribution; run the full control matrix (§5); run the secondary masked-pipeline readout.
- **Phase 3 — analysis + archive.** Co-primary A/B tests + controls + secondary; archive doc + figures (multi-panel discipline §7 — one question per panel); update the Topic 4 framework main-doc with a **pointer only** if a conclusion lands.

---

## 9. Reuse vs new code

**Reuse as-is**: runner skeleton + `--L/--density/--theta/--AR/--core-mean/--core-std/--core-r/--drive/--nc/--seed`; `sample_core_field`; `mean_match_vth` (C-homog rate match); `build_connectivity_rot`; `detect_events`/`calibrate_detector`; `endpoint_centroid_axis` + `direction_readability`; C1–C4 contrast harness (C-rot/C-iso); `build_sidecar` event-windowing; `--dump-fullfield`; `--swap-vth`/`--mirror-vth`; real masked lagPat / contact-plane readout (`mask_phantom=True`).

**New code**:
- `extended_patch` branch in `build_lesion_vth` + `--lesion` choice.
- `nucleation_centroid()` spatial extractor + sidecar extension (per-event `nucleation_xy`, `s_nuc`).
- Direction-distribution + first-contact-entropy quantifiers over a recording.
- Ensemble runner / aggregation (collect per-event readouts across long `T` / seeds; drive the control matrix).
- Numeric anti-blur discriminator (two-stage correspondence vs within-event shuffle; `C-point-readable` consistency). Separately: `C-homog` dispersion contrast (mechanism-modifier, not anti-blur).

**Conditional new (Phase 0)**: a **tail-bounded** spatial-bin/KDTree candidate restriction in the connectivity sampler — the code's own TODO. In-degree stays exact and the kernel *shape* is unchanged, but the long tail is truncated, so it is an error-bounded **approximation** that must pass the §8 equivalence gate (KS / delay / AR / tail-mass / dynamics smoke) before use. Required to make the L≈32 ensemble + controls practical.

---

## 10. Open risks

- **Central/transverse nucleation → radial/unreadable**: the correspondence test must handle the unreadable class as a third outcome, not a forced sign.
- **Runaway burst**: a wide-everywhere excitable sheet can self-ignite into a high-rate burst (the 328 Hz risk from the heterogeneity line). Phase 1 runaway hard-stop + `core_mean`/`drive` tuning.
- **Patch shape confound**: disk vs axis-elongated changes the sign-vs-angle balance — swept in Phase 1, default disk.
- **Density floor**: the spatial-bin prune radius must contain ≥800 E-candidates (OK at d100, ~3–4 mm radius); if a lower density is ever used, re-check.
- **Isolation**: run on a dedicated branch / worktree so this does not entangle the parallel Stage 3 detail work.
