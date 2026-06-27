# M3B-R2 finite Jacobian — LOCKED design contract (TDD-1..6 source of truth)

> Status: LOCKED, 2026-06-27. Reconciles 3 independent derivations + the parent candidate against
> `src/sef_hfo_lif.py::{_char_det, integrate_lif_field}`. Arbiter = synaptic Schur-complement
> elimination, cross-checked numerically (`scratchpad/verify_jacobian.py`).
> Verdict: **candidate A(k) is CORRECT as-is, all 36 entries, no sign change.**
> Code already partly built: `src/topic4_m3b_spectral_phase.py` (TDD-1, TDD-2). This doc locks
> TDD-3 (op point), TDD-5 (Jacobian), and the core/q parameterization.

---

## 0. Verdict in one line

The parent candidate's per-cell 6x6 block `A(k)` (ordering `[rE, rI, sEE, sEI, sIE, sII]`) reproduces
the linearization of `integrate_lif_field` term-by-term and, with the four synaptic states eliminated,
collapses **exactly** to `src.sef_hfo_lif._char_det` with delay set to 0. All three derivers
(verdicts CONFIRM/CONFIRM/CORRECT) agree on every matrix entry. There are **no matrix sign disputes**;
the only disagreements are interpretive caveats about which eigenvalue is "leading," resolved in §5.

---

## 1. The locked per-wavenumber block A(k) (6x6, homogeneous, NO delay)

State `z = [rE, rI, sEE, sEI, sIE, sII]` per cell. With
`gp = ghat(k, ELL_PAR) = exp(-0.5*ELL_PAR^2*k^2)`,
`gi = ghat(k, L_INH)  = exp(-0.5*L_INH^2*k^2)`,
`wee = w_ee_mult*W_EE`:

| row \ col | rE | rI | sEE | sEI | sIE | sII |
|---|---|---|---|---|---|---|
| **rE**  | `-1/TAU_ME` | `0` | `+gE*C_EE*wee/TAU_ME` | `-gE*C_EI*W_EI/TAU_ME` | `0` | `0` |
| **rI**  | `0` | `-1/TAU_MI` | `0` | `0` | `+gI*C_IE*W_IE/TAU_MI` | `-gI*C_II*W_II/TAU_MI` |
| **sEE** | `+gp/TAU_AMPA` | `0` | `-1/TAU_AMPA` | `0` | `0` | `0` |
| **sEI** | `0` | `+gi/TAU_GABA` | `0` | `-1/TAU_GABA` | `0` | `0` |
| **sIE** | `+gi/TAU_AMPA` | `0` | `0` | `0` | `-1/TAU_AMPA` | `0` |
| **sII** | `0` | `+gi/TAU_GABA` | `0` | `0` | `0` | `-1/TAU_GABA` |

`gE, gI` are the dimensionless local gains `(dPhi/dmu)*tau_m` from `lif_gains(op)` (homogeneous case)
or `local_gain(...)` per cell (heterogeneous case, §3).

**Subtle entry confirmed (Deriver 3):** `A[sIE, rE] = +gi/TAU_AMPA` uses the **AMPA time constant**
(it is a glutamatergic E->I synapse) but the **isotropic inhibitory spatial kernel `gi`** — because
`integrate_lif_field` does `sIE += dt/TAU_AMPA*(convolve(rE, K_I) - sIE)`, i.e. `K_I` (l_inh) for the
spatial factor, `TAU_AMPA` for the time constant. This is not a typo; do not "fix" it to `gp`.

---

## 2. Homogeneous-reduction proof sketch (the arbiter)

Continuous-time ODEs that `integrate_lif_field` forward-Euler-integrates (delay dropped, recovery
frozen, per cell; convolutions written in Fourier space at wavenumber k so `convolve(.,K)->ghat*.`):

```
ds_EE/dt = (1/TAU_AMPA)(gp*rE - sEE)
ds_EI/dt = (1/TAU_GABA)(gi*rI - sEI)
ds_IE/dt = (1/TAU_AMPA)(gi*rE - sIE)
ds_II/dt = (1/TAU_GABA)(gi*rI - sII)
muE = TAU_ME*(C_EE*wee*sEE - C_EI*W_EI*sEI) + muxE
muI = TAU_MI*(C_IE*W_IE*sIE - C_II*W_II*sII) + muxI
drE/dt = (1/TAU_ME)(-rE + Phi_E(muE)),   dPhi_E/dmuE = gE/TAU_ME
drI/dt = (1/TAU_MI)(-rI + Phi_I(muI)),   dPhi_I/dmuI = gI/TAU_MI
```

Linearize; the rate rows give exactly the candidate's rate rows (the `TAU_ME` in `muE` cancels the
`gE/TAU_ME` in `dPhi_E/dmuE`, leaving `gE*C_EE*wee/TAU_ME` etc.). Synaptic rows give the candidate's
synaptic rows directly.

**Eliminate the 4 synaptic variables at eigenvalue `lam`.** Each synaptic row of `(lam*I - A)z=0` is
diagonal: `(lam + 1/tau_s) ds = (amp/tau_s) dr`, so with `H_A = 1/(1+lam*TAU_AMPA)`,
`H_G = 1/(1+lam*TAU_GABA)` (= the delay=0 `H(ts)`):

```
dsEE = gp*H_A*drE,   dsEI = gi*H_G*drI,   dsIE = gi*H_A*drE,   dsII = gi*H_G*drI
```

Substitute into the rate rows, multiply the rE row by TAU_ME and the rI row by TAU_MI:

```
[(1+TAU_ME*lam) - gE*WEE*H_A] drE + [gE*WEI*H_G] drI = 0   ->  a*drE + b*drI = 0
[-gI*WIE*H_A] drE + [(1+TAU_MI*lam) + gI*WII*H_G] drI = 0  ->  c*drE + d*drI = 0
WEE = C_EE*W_EE*w_ee_mult*gp,  WEI = C_EI*W_EI*gi,  WIE = C_IE*W_IE*gi,  WII = C_II*W_II*gi
```

These `a,b,c,d` and the `W*` definitions are **identical term-by-term** to `_char_det` with
`H(ts)=1/(1+lam*ts)` (delay=0). Nontrivial solution `=> a*d - b*c = 0 = _char_det(delay=0)`. QED.

**Residual / spurious-root factor (exact, Deriver 3's constant is the right one).** Schur complement
on blocks `[rE,rI | s]`:

```
det(lam*I - A) = (1/(TAU_ME*TAU_MI*TAU_AMPA^2*TAU_GABA^2)) * (1+lam*TAU_AMPA)*(1+lam*TAU_GABA) * N(lam,k)
N(lam,k) = [ (1+TAU_ME*lam)(1+lam*TAU_AMPA) - gE*WEE ] * [ (1+TAU_MI*lam)(1+lam*TAU_GABA) + gI*WII ]
           + gE*gI*WEI*WIE      (degree-4 numerator of char_det0)
```

So the **6 eigenvalues of A(k) = {4 roots of N(lam,k)} ∪ {-1/TAU_AMPA, -1/TAU_GABA}**. The two extra
roots are the synaptic-decay poles of `_char_det` (its zeros are the physical/rate roots; its poles
are not). Deriver 2's residual factor is correct up to the irrelevant constant `TAU_AMPA^2*TAU_GABA^2`
(does not move roots); Deriver 1's `(lam+1/TA)^2(lam+1/TG)^2*(a*d-b*c)/(TAU_ME*TAU_MI)` is the same
identity before clearing the H denominators.

**Numerical confirmation** (`scratchpad/verify_jacobian.py`):
- PART 1: 3 ops x 4 k. `max |N(rate eig)| ~ 1e-13`; rate-branch rightmost eig == char_det0 rightmost
  root to machine precision; exactly 2 pole eigenvalues at -1/TA, -1/TG every time.
- PART 2: `gE=gI=0` gives exactly `{-1/TAU_ME, -1/TAU_MI, -1/TAU_AMPA(x2), -1/TAU_GABA(x2)}`.
- PART 4: full finite 6N dense J spectrum == union over allowed 2-D modes of A(k) spectrum to **3.2e-15**
  when A(k) uses the *discrete* kernel FFT amplitudes (analytic `_ghat` differs only at ~4.6e-5 from
  the discretized L1-normalized Gaussian — a kernel-sampling effect, not a Jacobian error).

---

## 3. Full finite 6N x 6N Jacobian J (the main M3B object)

Periodic `n x n` grid, `N = n*n`, state vector `6N` packed `[rE | rI | sEE | sEI | sIE | sII]`
(field-major; within each field, C-order ravel of the (n,n) grid). Block layout (each block is N x N):

```
            rE              rI              sEE        sEI        sIE        sII
rE   diag(-1/TAU_ME)        0          diag(GE_EE)  diag(GE_EI)    0          0
rI        0          diag(-1/TAU_MI)       0          0       diag(GI_IE) diag(GI_II)
sEE  (1/TAU_AMPA)KEE        0          -I/TAU_AMPA    0          0          0
sEI       0          (1/TAU_GABA)KI        0      -I/TAU_GABA    0          0
sIE  (1/TAU_AMPA)KI         0             0          0      -I/TAU_AMPA     0
sII       0          (1/TAU_GABA)KI        0          0          0      -I/TAU_GABA
```

where the rate-row coefficient vectors (length N, ALL the heterogeneity lives here) are

```
GE_EE[i] = +gE_i * C_EE * wee        / TAU_ME
GE_EI[i] = -gE_i * C_EI * (q_i*W_EI) / TAU_ME     # q_i = q_global * (q_core if i in core else 1)
GI_IE[i] = +gI_i * C_IE * W_IE       / TAU_MI
GI_II[i] = -gI_i * C_II * (q_i*W_II) / TAU_MI     # only if InhibitionField.scale_II (default OFF)
```

- `KEE`, `KI` are the **block-circulant** periodic-convolution operators (kernels `K_EE`=anisotropic,
  `K_I`=isotropic, both L1-normalized). In the homogeneous case J FFT-diagonalizes into A(k) per
  allowed 2-D mode (PART 4). Core heterogeneity makes only `gE_i, gI_i, q_i` spatially varying, which
  breaks circulance **on the rate rows only** — the convolution blocks stay homogeneous.
- **Synaptic-pole eigenvalues are structural:** `det(lam*I - J)` carries the k-independent factor
  `(1+lam*TAU_AMPA)(1+lam*TAU_GABA)`, so `-1/TAU_AMPA = -0.2857` and `-1/TAU_GABA = -0.0556` are
  eigenvalues for every mode. They are always negative, never cross 0, and are irrelevant to the
  `alpha_1 = 0` instability contour.

### 3.1 Two build modes (TDD-5 contract)

- **Dense (tiny grids, debug):** materialize the four N x N circulant matrices by feeding unit impulses
  through `convolve_periodic`, assemble the 6N x 6N dense array, `numpy.linalg.eig`. Used by
  `test_dense_jacobian_shape`, `test_jvp_matches_finite_difference_tiny_grid`,
  `test_no_core_jacobian_eigs_match_homogeneous_dispersion_samples`.
- **Matrix-free (real grids):** a `scipy.sparse.linalg.LinearOperator` of shape `(6N, 6N)` whose
  matvec never forms an N x N matrix:
  1. unpack `v -> {drE, drI, dsEE, dsEI, dsIE, dsII}` (each (n,n));
  2. rate rows: pure elementwise — `out_rE = -drE/TAU_ME + GE_EE*dsEE + GE_EI*dsEI`,
     `out_rI = -drI/TAU_MI + GI_IE*dsIE + GI_II*dsII` (GE_*/GI_* are (n,n) gain fields);
  3. synaptic rows: one `convolve_periodic` each —
     `out_sEE = (convolve(drE,K_EE) - dsEE)/TAU_AMPA`, `out_sEI = (convolve(drI,K_I)-dsEI)/TAU_GABA`,
     `out_sIE = (convolve(drE,K_I)-dsIE)/TAU_AMPA`, `out_sII = (convolve(drI,K_I)-dsII)/TAU_GABA`;
  4. pack back to `6N`.
  Feed to `scipy.sparse.linalg.eigs(which='LR', k=8, sigma=...)` for the leading rate-branch modes.
  Left modes from the same operator's adjoint (`J^T` matvec = transpose each block: rate->syn blocks
  become `convolve(., K)` with the SAME zero-phase kernel since K is symmetric; gains move to the
  syn->rate columns). `test_linear_operator_matvec_matches_dense` pins the two builds equal on a tiny
  grid.

### 3.2 Leading-eigenvalue extraction (resolve the candidate's claim, §5)

`alpha_1 := max Re(lam)` over the **rate branch** (the N(lam,k) roots), i.e. excluding the two known
synaptic-pole modes at -1/TA, -1/TG. In the instability-relevant regime (`alpha_1 >= -1/TG`) the
literal rightmost eigenvalue of J already IS a rate-branch root, so `which='LR'` returns it directly.
Only in the deeply-stable regime (every rate root below -1/TG = -0.0556) does the literal rightmost
eigenvalue floor at the -1/TG synaptic-decay mode; there `alpha_1` should be read as "<= -0.0556,
deeply stable" — harmless for the `alpha_1=0` contour. Record `synaptic_pole_floor_active: bool`.

---

## 4. Operating point (TDD-3) — what is being linearized

The heterogeneous core makes `z_star` **spatially non-uniform**; it must be solved before any gain.

- **Per-cell op:** for each cell i with its own `v_th_i` (ExcitabilityField) and `q_i`
  (InhibitionField), the operating point is `(nuE_i, nuI_i, muE_i, muI_i, sigmaE_i, sigmaI_i)`. Source =
  deterministic rate-field integration-to-steady of `integrate_lif_field` at frozen parameters
  (primary), or `mean_field`-style local fixed point (homogeneous control reduces to `mean_field`).
- **Per-cell gain:** `gE_i = local_gain(muE_i, sigmaE_i, pop='E', v_th=v_th_i)`,
  `gI_i = local_gain(muI_i, sigmaI_i, pop='I')` — the SAME finite-difference convention as `lif_gains`
  (`h=1e-3`, already in `topic4_m3b_spectral_phase.local_gain`). A low-threshold / high-drive core
  cell sits on a steeper part of the Siegert transfer => larger `gE_i`.
- `operating_point_source in {ratefield_steady, snn_baseline, frozen_slow_sample}` is RECORDED on every
  spectrum (design §3). Non-converged / saturated points are `unresolved` / `saturated`, **never
  silently `stable` or `axial`** (TDD-3 tests).
- **CRITICAL consistency (Deriver 3):** the op-point solver MUST use the SAME `q` and `v_th` fields the
  Jacobian uses (analogous to how `integrate_lif_field` reuses `op["w_ee_mult"]`). A Jacobian built with
  `q` while the op was solved at `q=1` is a silent contract violation. See Open Question O1 — `_ms` /
  `mean_field` currently have no `q` parameter.

---

## 5. Sign / interpretation disputes — each resolved by algebra against char_det0

1. **All 36 matrix entries: NO dispute.** Candidate, Deriver 1, Deriver 2, Deriver 3 are identical.
   Confirmed by §2 elimination (term-by-term match to `_char_det`) and PART 1/PART 4 numerics. Locked
   as-is.
2. **Candidate claim "leading eigenvalue of A(k) == rightmost root of _char_det" — REFINED, not
   overturned (Deriver 3 vs Deriver 1/2).** True for the **rate branch** at every k. The literal
   rightmost eigenvalue of the bare 6x6 equals the rightmost rate root ONLY while that root is
   `>= -1/TG`. PART 1 shows at k=1.5, 3.0 the rate rightmost (-0.0605, -0.067) sinks below the
   synaptic pole, so the bare rightmost floors at -0.055556. Resolution: report `alpha_1` on the rate
   branch (§3.2); the floor is benign for the `alpha_1=0` contour. Deriver 1/2 phrased this as a
   "floor caveat"; Deriver 3 made it precise; both are consistent — adopt Deriver 3's wording.
3. **Eigenvalue anatomy "2 fast + 4 slow" (Deriver 3) vs "4 fast near the two poles" (Deriver 1/2) —
   Deriver 3 correct.** Only ~2 eigenvalues are genuinely fast (near -1/TA = -0.286). `-1/TG = -0.0556`
   sits in the SLOW cluster because `TAU_GABA = 18 ms` is comparable to the membrane taus
   (`-1/TAU_ME = -0.05`, `-1/TAU_MI = -0.1`). PART 1's rate rightmost ~ -0.052 and the pole -0.0556 are
   neighbors. Cosmetic for stability, but the spotcheck/figure narration must not call -1/TG a "fast
   floor."
4. **Residual-factor constant (Deriver 2 vs Deriver 3) — Deriver 3 exact.** The precise prefactor is
   `1/(TAU_ME*TAU_MI*TAU_AMPA^2*TAU_GABA^2)`; Deriver 2 dropped `TAU_AMPA^2*TAU_GABA^2`. Irrelevant for
   roots, but §2 records the exact form.

No entry of A(k) changes. **Final verdict: CONFIRM the candidate A(k) as-is.**

---

## 6. Core-excitability and q_global / q_core — LOCKED parameterization

**Two orthogonal knobs, both touching ONLY rate-row diagonal coefficients, never the convolution blocks.**

### 6.1 Core excitability (phase-map x-axis, "core excitability")
Canonical knob = `dVth_core` (lower core threshold) and/or `mu_core` (extra core drive). Enters in two
places, both feeding the per-cell gain, NEITHER touching coupling:
1. **Operating point:** `v_th_i = V_TH - dVth_core` and/or `muxE_i += mu_core` on core cells shifts that
   cell's fixed point up (the `muxE`/`stim` term of `integrate_lif_field`).
2. **Per-cell gain:** `gE_i` is evaluated at the shifted op, so core cells get a larger diagonal gain.
   In J this makes `GE_EE[i], GE_EI[i], GI_IE[i], GI_II[i]` (the rate-row coefficients of cell i)
   spatially varying; the block-circulant `K_EE, K_I` and the structural constants `C_*, W_*` are
   UNCHANGED. `gE_i` multiplies BOTH cell i's E (`sEE`) and I (`sEI`) rate-row entries.
- **Exception explicitly excluded:** implementing "core drive" as a locally elevated `w_ee_mult` WOULD
  rescale excitatory coupling locally. The canonical knob is `dVth_core`/`mu_core`, not local
  `w_ee_mult`. (Deriver 1's note, adopted as a forbidden alternative for this round.)

### 6.2 q = inhibition efficacy (phase-map y-axis disinhibition + core nucleation)
`q in (0, 1]`, dimensionless GABA-efficacy multiplier on the **I->E post-synaptic weight `W_EI`** (the
load-bearing disinhibition lever). `q=1` = baseline; lower q = weaker inhibition = more disinhibition.
- In A(k): `A[rE, sEI] = -gE*C_EI*(q*W_EI)/TAU_ME`, and `_char_det`'s `b = gE*(C_EI*(q*W_EI)*gi)*H_G`.
- **Sign check (locks the contract direction):** lowering q shrinks the magnitude of this negative brake
  entry => weaker inhibitory pull-back on rE => leading rate root moves RIGHT => phenotype coordinate
  rises as q falls. The contract transform is `reciprocal_affine`, `out = clip(a/q + b)`,
  `expected_direction = decreasing_in_input` over `q in [0.2, 1.0]` — satisfied:
  `phase_y_global` decreasing in `q_global`, `phase_x_core` decreasing in `q_core`
  (`src/sef_hfo_m3_interface.check_sign_direction`).
- **`q_global`** = field-wide scalar on every cell's `W_EI` (homogeneous y-axis).
- **`q_core`** = SAME scalar but SPATIALLY MASKED to core cells: `q_i = q_global * (q_core if i in core
  else 1)`. Because `W_EI` is a post-synaptic weight applied at the RECEIVING E cell
  (`muE_i = TAU_ME(C_EE*wee*sEE_i - C_EI*W_EI_i*sEI_i) + muxE`), `q_core` makes the rE-row `sEI` entry
  per-cell — a localized disinhibition / core nucleation site. The `K_I` convolution producing `sEI`
  is untouched; only the receiving-cell weight is masked. This is **distinct** from core excitability
  (which touches only `gE_i`): `q_core` additionally scales the `sEI` column of the core cell's rate row.
- **`W_II` scaling: OPTIONAL, default OFF** (`InhibitionField.scale_II=False`). Scaling `W_II` (I->I)
  disinhibits the I population too and partially cancels the E release, muddying the clean disinhibition
  axis (Derivers 2, 3). Default = scale `W_EI` only. If `scale_II=True`, the SAME `q_i` multiplies
  `GI_II[i]` and MUST be threaded into the op-point too.
- **e_GABA path:** in this current-based reduction the contract's `e_GABA` disinhibition maps onto the
  `W_EI` scaling; `shunt_path_active` gating stays a recorded flag (interface §4). e_GABA is assigned to
  exactly one coordinate.
- **Two-core (validation geometry):** `q_core_L`, `q_core_R` per the contract's `two_core_reduction`
  (`source_core | min_q | mean_q`) collapse to `phase_x_core`. The reduction value is a recorded science
  choice (Open Question O3 / interface §9).

---

## 7. Dataclass API for `src/topic4_m3b_spectral_phase.py`

`Grid`, `CoreMask`, `pack_state`/`unpack_state`, `STATE_FIELDS` already exist (TDD-1) — keep as-is.
Add `Kernels`, `ExcitabilityField`, `InhibitionField`, `OperatingPoint` for TDD-3/TDD-5.

```python
@dataclass(frozen=True)
class Grid:                       # EXISTS — keep
    n: int = 48
    L: float = 12.0
    # .size -> n*n ; .spacing -> L/n ; .coords() -> (X,Y) meshgrid indexing='ij'

@dataclass(frozen=True)
class Kernels:                    # NEW (TDD-5 — the two coupling operators + FFT amps for matvec)
    K_EE: np.ndarray              # (n,n) anisotropic, L1-normalized (E->E scaffold)
    K_I:  np.ndarray              # (n,n) isotropic, L1-normalized (used by sEI<-rI, sIE<-rE, sII<-rI)
    ghat_EE: np.ndarray           # (n,n) = real(fft2(ifftshift(K_EE)))  precomputed DFT amplitudes
    ghat_I:  np.ndarray           # (n,n) = real(fft2(ifftshift(K_I)))
    ell_par: float                # mm
    ell_perp: float               # mm
    l_inh: float                  # mm
    theta: float                  # rad, E->E major axis (default THETA_EE = pi/4)

@dataclass(frozen=True)
class CoreMask:                   # EXISTS — keep
    kind: str                     # {"none","single","two","off_axis"}
    mask: np.ndarray              # (n,n) bool
    centers: tuple[tuple[float,float], ...]
    radius: float
    theta: float
    # .area_fraction -> mask.mean()

@dataclass(frozen=True)
class ExcitabilityField:          # NEW (TDD-3 x-axis knob — operating-point shift, NOT coupling)
    v_th: np.ndarray              # (n,n) per-cell firing threshold (mV); V_TH off-core, V_TH-dVth_core on core
    mu_core: np.ndarray           # (n,n) per-cell additive drive into muE (mV); 0 off-core
    core: CoreMask
    dVth_core: float              # scalar that built v_th on the core (mV, >=0)
    mu_core_value: float          # scalar additive core drive (mV, >=0)

@dataclass(frozen=True)
class InhibitionField:            # NEW (TDD-2/TDD-3 y-axis + core-nucleation knob; q efficacy)
    q: np.ndarray                 # (n,n) effective I->E efficacy multiplier in (0,1]; q_global*(q_core on core)
    q_global: float               # field-wide GABA efficacy (1.0 baseline)
    q_core: float                 # extra core-local efficacy multiplier (1.0 = none)
    scale_II: bool                # also scale W_II by q? default False (clean disinhibition axis)
    core: CoreMask

@dataclass(frozen=True)
class OperatingPoint:             # NEW (TDD-3 — the frozen linearization state + per-cell gains)
    rE: np.ndarray; rI: np.ndarray            # (n,n) kHz
    muE: np.ndarray; muI: np.ndarray          # (n,n) mV
    sigmaE: np.ndarray; sigmaI: np.ndarray    # (n,n) mV
    gE: np.ndarray; gI: np.ndarray            # (n,n) dimensionless local gains (diagonal Jacobian scales)
    source: str                   # {"ratefield_steady","snn_baseline","frozen_slow_sample"}
    converged: bool
    residual: float
    saturated: bool               # high-rate runaway flag (=> mode_class 'runaway', never 'axial')
    excitability: ExcitabilityField
    inhibition: InhibitionField
    wee_mult: float               # recurrent E->E gain used (matches the op, like integrate_lif_field)
    nuext: float                  # external drive (kHz)
```

**Pack/unpack state-vector convention (EXISTS, locked):** `z` is `6*N` floats, field-major in
`STATE_FIELDS = ("rE","rI","sEE","sEI","sIE","sII")` order; within each field, C-order `.ravel()` of
the `(n,n)` grid. `pack_state(dict, grid)` concatenates `[state[f].ravel() for f in STATE_FIELDS]`;
`unpack_state(z, grid)` slices `z[i*N:(i+1)*N].reshape(n,n)`. The Jacobian's 6 row-blocks and 6
col-blocks index in this exact order, so block `(r,c)` occupies `J[r*N:(r+1)*N, c*N:(c+1)*N]`.

---

## 8. Open questions (need science / user decision before TDD-3/5 wiring)

- **O1 — q has no operating-point path yet.** `src.sef_hfo_lif._ms` / `mean_field` take `w_ee_mult` but
  NOT a `q` multiplier on `W_EI`. To keep the op point and the Jacobian on the SAME q (§4), either
  (a) add a `q_ei_mult` (and optional `q_ii_mult`) parameter to `_ms`/`mean_field` mirroring
  `w_ee_mult`, or (b) build the M3B heterogeneous op-point solver to apply q per cell directly. The
  heterogeneous core needs a spatial fixed point regardless, so (b) is likely; but the homogeneous
  control should still reduce to a q-aware `mean_field`. Decide before TDD-3.
- **O2 — primary operating-point source.** Design §3 lists `ratefield_steady` (primary),
  `snn_baseline`, `frozen_slow_sample`. The heterogeneous steady state of `integrate_lif_field` at
  frozen params is the natural primary, but its convergence/uniqueness in the high-disinhibition corner
  is unverified. Need a TDD-3 convergence gate + the `unresolved` fallback wired before scanning.
- **O3 — two_core_reduction value** (`source_core` vs `min_q` vs `mean_q`) for collapsing
  `q_core_L,q_core_R -> phase_x_core` — recorded as data, science choice (interface §9). Single-core is
  the primary atlas, so this only blocks the two-core validation leg.
- **O4 — e_GABA axis assignment** (core vs global) and whether the current-based `W_EI`-scaling
  realization is an acceptable stand-in for the conductance/shunt e_GABA path (`shunt_path_active`),
  decided at M3A-A1 calibration; M3B only enforces it is recorded and shunt-gated.
- **O5 — deeply-stable alpha_1 reporting.** When all rate roots fall below `-1/TAU_GABA`, `which='LR'`
  returns the synaptic-pole mode. Confirm the agreed convention: report `alpha_1` on the rate branch
  with a `synaptic_pole_floor_active` flag, rather than the literal rightmost eigenvalue (§3.2). This is
  a reporting choice, benign for the `alpha_1=0` contour but it changes the printed number deep in the
  stable region.
- **O6 — `sigma` (input s.d.) heterogeneity in the gain.** `local_gain` is evaluated at each cell's
  `(mu_i, sigma_i)`. Core cells have a different `sigma_i` (the `sE`/`sI` from `_ms`), which also moves
  the Siegert slope. Confirm the op-point solver returns per-cell `sigma_i` (it does, via `_ms`) and
  that `local_gain` is called with it, not a global sigma.
```
