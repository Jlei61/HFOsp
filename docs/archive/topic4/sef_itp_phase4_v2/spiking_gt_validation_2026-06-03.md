# Step-0 spiking-ground-truth validation + acceptance (2026-06-03)

> **Status: Step 0 ACCEPTED.** The Step-0 rate-field mechanism is validated against a
> spiking ground-truth network (coworker Zou's Bachschmid-Romano/Hatsopoulos/Brunel 2026
> spatial E-I LIF engine), with ONE correction to the ratified conclusion: the
> interictal-like operating point is **lower drive (ν_ext/ν_θ ≈ 0.6), not 1.0**;
> drive ≈ 1.0 is the sustained-oscillation / seizure-ward regime.
>
> Engine + probes + figures: `results/topic4_sef_hfo/lif_snn/{engine,figures,data}`.
> This archive is the detailed record; the main framework Step-0 banner should point here.

## Take-home (first-principles)

我们把"安静可激斑块 + 踢一下 → 定向自限传播事件"这个 Step-0 机制，搬到一个**真·一个个神经元放电的网络**里检验（同一套 Brunel 常数、带 18ms 抑制滤波，所以"噪声有记忆"这条真实存在）。结论：

1. **机制在放电真值里站住了。** 在一个**较低、安静的外部驱动点**（驱动比 ≈0.6，网络平时近静默），给一个局部瞬时扰动，会触发一个**逐步外扩、慢传播、然后自己熄灭**的群体事件——不是"点一下就灭"，也不是"整片同时闪"。前沿速度量级 ~0.07–0.08 mm/ms，穿过 ~1cm 约 ~125ms，落在实测通道延迟 50–178ms 的区间（比 rate-field regime-b 的 ~0.9mm/ms 合理得多）。

2. **承重判据（方向由连接定、不由几何定）也站住了。** 用 onset-front（踢后最早 8ms 点亮的前沿，饱和之前）量方向，把 E→E 连接轴转 0/45/90°，**前沿主轴跟着转，误差 ~1°，拉长比 2.4–3.8**；把连接核改成各向同性，**前沿没有优势轴（比 1.13）**、而且几乎点不着。所以连接各向异性是"既让它传得动、又给它定方向"的必要条件。

3. **唯一的更正：间期工作点不是 1.0，是 ~0.6。** 之前 Step-0 把工作点放在驱动 1.0，并下了"稳定可激、非 Hopf"的结论。放电真值显示 **1.0 处是真·自持振荡**（N-scaling 判定：神经元数翻 4 倍振荡强度不降、频率稳在 26.7Hz = 确定性 Hopf，不是有限尺寸噪声）。安静可激静息在更低驱动 ~0.6。**这正好支持"促临界↔稳态回拉"那根轴：间期 = 低驱动安静可激，趋发作 = 高驱动自持振荡。**

> 验收口径仍是 **mechanism-scale**（量级一致即可，非逐病人定量拟合）；现在比 Step-0 原验收又多了一层：**rate-field 与 spiking 双重一致**，且承重判据在两边都过。

## Cohort numbers (spiking ground truth, g=3.6, Brunel filtered synapses)

### Operating-point isomorphism (rate↔spiking, ratios 0.6–1.3, L=1mm density 4000)
- **I-rate matches our white-noise mean-field ν_I tightly (~×0.9 across the band).**
- **E-rate ~3–4× our ν_E** (same order, both sub-1Hz) — MOSTLY a comparison artifact (stationary FP vs oscillating net, Jensen on the convex low-E tail), NOT a transfer readout. The robust I-match is the mean-driven (easy) population; the E population is fluctuation/gain-sensitive.

### Drive-axis phase structure (fine sweep)
- 0.5 dead (E=0.00) → 0.6–0.65 quiet (E=0.10–0.26) → 0.70 strong low-rate oscillation (prom 23) → 0.75+ high-rate. Sharp, sensitive transition (0.75 non-monotonic vs 0.8 = probable multistability; single-run transition points unreliable).

### N-scaling at ratio 1.0 (finite-size SI vs genuine Hopf) — SETTLED
- density 4000/8000/16000 → prominence **8.8/10.6/8.7 (flat)**, frequency stable 26.7Hz, rates N-independent → **genuine deterministic Hopf** (finite-size would drop ~1/√N to ~4.4). → ratio 1.0 genuinely oscillates; the ratified "stable at the operating point" was wrong AT 1.0.

### Kick-from-quiet-rest (excitability) — ratio 0.6
- L=1mm: kick → large all-or-none event (peak ~510Hz, outside-disk recruitment ~190× the no-kick control), returns to baseline → EXCITABLE. But fills the sheet in ~2ms (= conduction time) → can't separate traveling-wave from synchronous-flash, nor dynamical self-limit from finite-size exhaustion.
- L=2mm: active fraction grows GRADUALLY over ~20ms, radius 0.14→1.16mm → **traveling, not synchronous**; peak active frac 0.76 (returns) but still reaches all corners (near-global at this size).

### Anisotropy onset-front cohort (the load-bearing discriminator) — PASS
- Fix vs the inconclusive L=2 peak-covariance probe: L=3mm (event sub-global, max active frac 0.26–0.43) + **onset-front metric** (principal axis of E neurons firing in the first 8ms, pre-saturation) + rotated E→E kernel (`build_connectivity_rot`, AR=2 matches original ρ_EE=0.6 to 45.1°/ratio1.85) + 3 seeds × {θ=0,45,90} + isotropic control.
- **Result:** front axis tracks θ_EE to **mean error 0.5–1.1°** each, elongation ratio **2.4–3.8** (>1.3); **isotropic control ratio 1.13** (<1.3), no axis. Excess≈raw (no background contamination at L=3); pre_kick identical on all 12 runs; ratio sharpest at thinnest front (6>8>10ms = expanding anisotropic front). Figure `figures/anisotropy_front_test.png` is visually clean.
- **Honest caveat:** isotropic control is a near-FIZZLE (max active frac 0.005–0.008), not a big directionless wave → precise statement = "anisotropic E→E reach (extended long-axis ℓ_par) is NECESSARY for both strong propagation AND a directional front; circular kernel → weak directionless recruitment."

### Figure provenance (2026-06-05 redesign — user catch on the wide-window panel)
- The direction discriminator is now its **own** paper-grade supplementary `figures/anisotropy_front_test.png` (`scripts/plot_sef_hfo_anisotropy_front.py`): 4 spatial panels (representative seed 1, **8 ms onset window**, front colored by recruitment time, imposed-axis dashed, kick disk) + 2 quantitative panels (3-seed mean±sd elongation ratio with the 1.3 line; 3-seed direction error with the 25° tolerance). It was **removed from the main figure** `sef_hfo_spiking_validation.png` (now a/b/c only).
- **Why:** the earlier main-figure "panel d" drew the front over a **12 ms** window from a **single seed** and uniform bright-red (no onset-time coloring). At 12 ms the L=3 event has partly filled, so the panel showed realization-dependent central holes (seed-1: θ=45° emptiest, θ=90° fullest) that read as a per-angle effect. It is **not** a mechanism: at W=8 ms the per-seed front size swings 2.5× (θ=0: 2106/5184/5010 over seeds 1/2/3) — larger than any between-angle gap — while the **verdict-bearing 3-seed excess metric is angle-symmetric and PASSES** (err 0.5–1.1°, ratio 2.4–3.8, iso 1.13). The redesign fixes this by (i) the 8 ms window, (ii) onset-time coloring so the early along-axis edge is legible, and (iii) carrying the numbers in the 3-seed quant panels rather than one seed. The central black circle = the kicked disk (R_KICK=0.15 mm); its hollow center is the kicked patch firing during the kick (onset < window start) then recovering, i.e. the wave moving outward — by construction, not an artifact.

## Accepted status (supersedes prior "spiking 0d OPEN")
- **Step-0 mechanism core: SUPPORTED in spiking GT** (quiet excitable rest + finite kick → slow self-limited directional propagation; reasonable speed scale).
- **Step-0 anisotropy discriminator: SUPPORTED in spiking GT** (onset-front tracks connectivity axis through rotation; isotropic → no axis). rate-field 0d PASSED + spiking 0d PASSED.
- **Operating-point correction: interictal-like ≈ drive 0.6, NOT 1.0; drive 1.0 = sustained oscillation / seizure-ward.**

## Not-blocking remaining
- More seeds/conditions for a publication-grade cohort (multi-amplitude low value — all-or-none verified).
- Dispersion-with-delays at the measured rates to explain WHY 1.0 is Hopf (connects to the pro-critical↔homeostatic axis).
- Fix the front-speed estimator (returned nan; magnitude ~0.07–0.08mm/ms read off the radius-vs-time table).
- Whether a cleaner "stays-local self-limit" holds at even larger L.
- Parallel rate-field Step-1 work (events/discreteness/direction, §9) was developed before this relocation — its direction discriminator is independently OPEN/grid-contaminated; revisit at the relocated operating point.

## Cross-model closures (2026-06-04) — rate-field ↔ spiking consistency settled

朴素话:之前我差点写下"LIF rate-field 没有 Hopf,1.0 的振荡只有放电网络才有"——这是错的,而且和我们自己的证据矛盾。两个解析/仿真收口把它纠正过来:**rate-field 和放电网络其实一致**——间期(低驱动)两个模型都是安静可激、踢一下给慢传播自终止事件;往发作方向(高驱动)两个模型都有一个 finite-k 振荡(Hopf)模式。之前"没 Hopf"只是因为白噪声平均场把兴奋率放太低、低估了增益。

### Closure #1 — dispersion at self-consistent vs corrected gain (`scripts/sef_hfo_lif_dispersion_closure.py` → `results/topic4_sef_hfo/lif_snn/data/dispersion_closure.json`)
Same rate dispersion (step0a `char_det`: anisotropic E→E, AMPA/GABA single-pole + 1 ms conduction delay, slow GABA 18 ms), across drive, TWO gains:
- **A = white-noise fsolve mean-field gain** (loop 0–0.58): **0/9 ratios Hopf, stable** — the "no Hopf" we'd been reporting.
- **B = gain at the SPIKING-measured rate** (find μ s.t. `lif_rate`=spiking νE at the mean-field σ; loop 1.5–1.85): **7/9 ratios finite-k Hopf, freq 11.6→23 Hz rising with drive (20 Hz @ ratio 1.0)** — ballpark of the spiking 26.7 Hz / coworker 29 Hz. Hopf onset ~0.65–0.7; NO Hopf at 0.5–0.6.
- **Verdict:** the rate-field DOES have the finite-k Hopf; **"no Hopf" was a white-noise gain-UNDERESTIMATE artifact** (white-noise νE=0.2 vs real 0.8; steep low-rate f-I → true gain ~3×). Honest nuance: even at gain B, Re λ ≈ −0.03 (near-critical, marginally stable) — the rate-field has the Hopf MODE; the spiking net sits just past onset (last bit = colored-noise/finite-size the white-noise rate-field underestimates). Do NOT write "rate-field has no Hopf / oscillation is spiking-only."

### Closure #2 — rate-field excitable window across drive (`scripts/sef_hfo_lif_kick_sweep.py` → `data/kick_sweep.json`)
`integrate_lif_field` finite kick (A=8) at each drive (white-noise gain): **self_limited_propagation at EVERY ratio 0.5–1.3** (front 6.9 mm, returns). The advisor's "kick fizzles at 0.6" did NOT happen — 0.6 gives a clean self-limited event, **~98 ms** (in the data envelope ~100–300 ms low end); duration shortens with drive (102→64 ms). Caveat: white-noise gain → never reaches the Hopf → excitable everywhere (expected; Hopf side is closure #1's analytic result; corrected-gain field re-run = refinement).

### Settled cross-model story
The drive axis = pro-critical ↔ homeostatic axis = finite-k-Hopf-onset axis, **consistent in BOTH models**: interictal ~0.6 = below Hopf (quiet excitable, kick→self-limited, ~98 ms) ; oscillatory ~0.7+ = finite-k Hopf ON (spiking oscillates 26.7 Hz; rate-field at corrected gain has the ~20 Hz Hopf mode). The single white-noise gain underestimate caused BOTH the earlier "no Hopf" and "excitable everywhere". (Seizure connection of the oscillatory regime is a Step 4-5 / SNN question per the 2026-06-04 division of labor, NOT a Step-0 claim — red line: Topic-4 main model does not explain clinical seizure onset.) **Heterogeneity (the OTHER gain knob: low heterogeneity ≈ sharper transfer ≈ higher gain → toward the Hopf, co-equal with drive) is NOT yet swept — deferred to Step 3.**
